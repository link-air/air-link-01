"""阶段 2（画像与印证）的测试。

对应阶段 2 的交付：`distill` step2/step3 + 印证收敛 + 可否决 + 双时间戳。

这一层的测试重点是**「不武断」这条边界**——它能被测试，是因为它被写成了
可计算的判据（印证数、同情境、跨来源、可追溯），而不是一句态度声明：
  - 证据不够 → 宁可 pending，不许 established
  - 情境不同 → 同类反应也不算同向收敛
  - 用户否决 → 必须生效（这是一条能被证伪的承诺）
"""
# 用例分组：
#   脚手架  Base（临时库 + 假模型 + 造场景）
#   提炼    TestStep2 聚合 · TestStep3 抽象 · TestTopic 主题 · TestCycle 周期
#   收敛    TestConverge 四道关 · TestAgeOut 老化 · TestReject 否决 ·
#           ProfileCapTest 总量上限
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core.distill import (age_out_profiles, cap_profiles, converge, converge_detail,
                          distill_step2, distill_step3, resolve_topic, revise_profile,
                          run_distill_cycle)
from core.llm import LLM
from core.model import Profile, Scene, Summary
from core.store import Store, now_str
from core.weave import user_reject_profile

TOPIC = "用户·面对压力的反应"


# ---------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------

def distill_llm(summaries=None, profiles=None, revisions=None, topics=None) -> LLM:
    """按 schema 名分发的剧本 LLM（每个队列最后一项会被重复使用）。

    分发的意义：提炼链路里一次周期会依次调用四种 prompt，
    用单一返回值的话，测不出「step2 和 step3 各拿到自己的那份」。
    """
    q_s = list(summaries) if summaries else [{"text": "他在几次压力下的反应叙述。"}]
    q_p = list(profiles) if profiles else [{"statement": "遇到被评价的情境会先退出"}]
    q_r = list(revisions) if revisions else [{"verdict": "holds", "statement": ""}]
    q_t = list(topics) if topics else [{"topic": TOPIC}]

    def pick(q):
        return q.pop(0) if len(q) > 1 else q[0]

    def fn(prompt, schema):
        name = schema.get("name")
        if name == "topic_summary":
            return pick(q_s)
        if name == "profile_abstraction":
            return pick(q_p)
        if name == "profile_revision":
            return pick(q_r)
        if name == "topic_choice":
            return pick(q_t)
        return {}

    llm = LLM()
    llm.set_mock(fn)
    return llm


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = Store(self.root / "t.db")
        # 三个目录都要重定向：留痕、原文文档、备份——少一个就会被测试污染真实目录
        self._old_paths = {k: cfgmod.PATHS[k]
                           # shortterm 也要：有些测试直接建 ChatSession（没传
                           # state_path），默认会落到真实的窗口文件——漏了它，
                           # 跑一次测试就往真窗口里塞几条消息（实锤过）
                           for k in ("trace_dir", "raws_dir", "backup_dir",
                                     "shortterm")}
        cfgmod.PATHS["trace_dir"] = str(self.root / "trace")
        cfgmod.PATHS["raws_dir"] = str(self.root / "raws")
        cfgmod.PATHS["backup_dir"] = str(self.root / "backups")
        # 窗口也要重定向：直接建 `ChatSession`（没传 state_path）的测试
        # 默认会落到真实窗口文件——漏了它，跑一次测试就往真窗口里塞几条消息
        cfgmod.PATHS["shortterm"] = str(self.root / "shortterm.json")

    def tearDown(self):
        for k, v in self._old_paths.items():
            cfgmod.PATHS[k] = v
        self.store.close()
        self._tmp.cleanup()

    def scene(self, topic: str = TOPIC, trigger_class: str = "被评价",
              subject: str = "user", title: str = "被批评",
              trigger: str = "被组长批评", **over) -> Scene:
        s = Scene(topic=topic, trigger_class=trigger_class, subject=subject,
                  title=title, text=f"{title}的摘要", trigger=trigger,
                  time_event=f"2026-08-{10 + self.store.count('scenes'):02d} 09:00:00", **over)
        self.store.add_scene(s)
        return s

    def pending_profile(self, n_scenes: int = 3, profile_subject: str | None = None,
                        **scene_over) -> tuple[Profile, list[Scene]]:
        """造一条待验证画像 + 它的证据场景。

        画像的 subject 默认跟随场景的 subject——跨来源规则判的就是这两者是否一致，
        写死了就测不出「world 类可以跨来源」。
        """
        scenes = [self.scene(**scene_over) for _ in range(n_scenes)]
        p = Profile(topic=TOPIC,
                    subject=profile_subject or scene_over.get("subject", "user"),
                    statement="遇到被评价会先退出",
                    status="pending", evidence=n_scenes,
                    sources=[s.id for s in scenes])
        self.store.add_profile(p)
        return p, scenes


# ---------------------------------------------------------------------
# 一、提炼 2：S1 → S2 聚合
# ---------------------------------------------------------------------

class TestStep2(Base):
    def test_needs_enough_scenes(self):
        """攒不够 n 条不聚合——聚合的原料是「同主题的多次出现」。"""
        for _ in range(2):
            self.scene()
        self.assertIsNone(distill_step2(self.store, TOPIC, distill_llm()))

    def test_aggregates_and_is_idempotent(self):
        scenes = [self.scene() for _ in range(3)]
        llm = distill_llm()
        s2 = distill_step2(self.store, TOPIC, llm)

        self.assertIsNotNone(s2)
        self.assertEqual(len(s2.sources), 3)
        self.assertEqual(self.store.count("summaries"), 1)
        # 追溯靠 `Summary.sources`（2026-09-24：compose 边已退役——它是这份
        # 列表的冗余存储，"能派生的不存"）。反向查（谁收了我）也走它（`_fresh_scenes`）。
        for s in scenes:
            self.assertIn(s.id, s2.sources)

        # 幂等：同一批场景重跑不该再产出一条摘要（后台任务会被反复触发）
        self.assertIsNone(distill_step2(self.store, TOPIC, distill_llm()))
        self.assertEqual(self.store.count("summaries"), 1)

    def test_incremental_aggregation_for_new_scenes(self):
        """新场景攒够了要能聚成**新的** S2，而不是覆盖旧的（聚合可逆）。"""
        for _ in range(3):
            self.scene()
        distill_step2(self.store, TOPIC, distill_llm())
        for _ in range(3):
            self.scene()
        distill_step2(self.store, TOPIC, distill_llm())
        self.assertEqual(self.store.count("summaries"), 2)

    def test_empty_summary_is_dropped(self):
        for _ in range(3):
            self.scene()
        self.assertIsNone(distill_step2(self.store, TOPIC, distill_llm(summaries=[{"text": ""}])))

    def test_extra_topics_only_from_existing(self):
        """附加主题（2026-09-24 晚）：**只认已有主题原样**——编的丢掉；主主题排第一。"""
        self.store.add_summary(Summary(topic="用户·工作", text="旧的", sources=[]))
        self.store.add_profile(Profile(topic="用户·边界", statement="旧的",
                                       status="pending", evidence=1))
        for _ in range(3):
            self.scene()
        llm = distill_llm(summaries=[{"text": "叙述",
                                      "extra_topics": ["用户·工作", "编的主题"]}])
        s2 = distill_step2(self.store, TOPIC, llm)

        self.assertEqual(s2.topics, [TOPIC, "用户·工作"], "主主题在前；编的被过滤")
        got = self.store.get_summary(s2.id)
        self.assertEqual(got.topics, [TOPIC, "用户·工作"], "落库一致")
        self.assertEqual(got.topic, TOPIC, "`topic` 列 = topics[0]（聚合键不动）")


# ---------------------------------------------------------------------
# 二、印证收敛（提炼 3 的四道关）
# ---------------------------------------------------------------------

class TestConverge(Base):
    def test_three_same_situation_converges(self):
        p, _ = self.pending_profile(3)
        self.assertTrue(converge(self.store, p))
        self.assertEqual(self.store.get_profile(p.id).status, "established")

    def test_two_scenes_stay_pending(self):
        """证据不够 → 宁可 pending。「不武断」在这里是一条可测的判据。"""
        p, _ = self.pending_profile(2)
        self.assertFalse(converge(self.store, p))
        self.assertEqual(self.store.get_profile(p.id).status, "pending")

    def test_different_situations_do_not_converge(self):
        """不同情境下的同类反应**不算**同向收敛。

        「被批评后退出」和「被表白后退出」是两件事——
        拼在一起就得到一个假的「他一遇事就退」。
        """
        self.scene(trigger_class="被评价")
        self.scene(trigger_class="关系冲突")
        self.scene(trigger_class="失去")
        scenes = self.store.query_scenes(topic=TOPIC)
        p = Profile(topic=TOPIC, subject="user", statement="一遇事就退",
                    status="pending", sources=[s.id for s in scenes])
        self.store.add_profile(p)
        self.assertFalse(converge(self.store, p))
        self.assertEqual(self.store.get_profile(p.id).status, "pending")

    def test_single_misclassified_scene_does_not_block(self):
        """**单条分类噪声不该卡死整条画像**（2026-09-12 加）。

        `trigger_class` 是 LLM 判的，会有波动——实测里「被组长当众批评」和
        「开会时被打断」被分到了两类，于是画像永远升不了级。
        所以判定改成「多数同类（≥60% 且至少 2 条）」而不是「完全相同」。
        """
        self.scene(trigger_class="被评价")
        self.scene(trigger_class="被评价")
        self.scene(trigger_class="关系冲突")          # 单条误判
        scenes = self.store.query_scenes(topic=TOPIC)
        p = Profile(topic=TOPIC, subject="user", statement="s", status="pending",
                    sources=[s.id for s in scenes])
        self.store.add_profile(p)
        self.assertTrue(converge(self.store, p), "三条里两条同类 → 该通过")

    def test_all_different_still_blocked(self):
        """真·不同情境凑不出多数，照样不通过（设计意图没被放宽掉）。"""
        for cls in ("被评价", "关系冲突", "失去"):
            self.scene(trigger_class=cls)
        scenes = self.store.query_scenes(topic=TOPIC)
        p = Profile(topic=TOPIC, subject="user", statement="s", status="pending",
                    sources=[s.id for s in scenes])
        self.store.add_profile(p)
        self.assertFalse(converge(self.store, p))

    def test_no_trigger_class_means_no_convergence(self):
        """没有情境标签就判不通过——降级时宁可保守（少判一次升级，不会错判）。"""
        p, _ = self.pending_profile(3, trigger_class="")
        self.assertFalse(converge(self.store, p))

    def test_user_profile_rejects_cross_source(self):
        """「关于用户本人」的画像只认用户本人的直接表达（别人转述的不算）。"""
        p, scenes = self.pending_profile(3)
        self.store.conn.execute("UPDATE scenes SET subject = 'world' WHERE id = ?",
                                (scenes[0].id,))
        self.store.conn.commit()
        self.assertFalse(converge(self.store, p))

    def test_world_profile_allows_cross_source(self):
        """「关于世界」的画像可以跨来源计数（不同角度印证同一件事）。"""
        p, scenes = self.pending_profile(3, subject="world")
        self.store.conn.execute("UPDATE scenes SET subject = 'air' WHERE id = ?",
                                (scenes[0].id,))
        self.store.conn.commit()
        self.assertTrue(converge(self.store, p))

    def test_same_situation_compares_trigger_not_scene(self):
        """同情境比的是 `trigger`，**不是整个场景的向量**。

        这个偏差是跑实验才发现的：三条场景 class 同、topic 同、印证 3 次，
        却因为「整体向量掺着谁/什么反应/什么结果」而永远升不了级。
        规格写的是「用前因后果的 trigger 比对 + 语义相近」，实现时偷懒用了 `scene.emb`。

        这里场景一律**没有向量**（整体差异无从比较），只有 trigger 向量相近——
        如果实现还去看 scene.emb，这个用例就会失败。
        """
        p, _ = self.pending_profile(3, trigger="被组长当众批评")

        class FakeEmb:
            available = True

            def __init__(self, vecs):
                self._vecs = vecs

            def embed(self, texts):
                return self._vecs[:len(texts)]

        near = FakeEmb([[1.0, 0.0], [1.0, 0.05], [1.0, 0.1]])
        self.assertTrue(converge(self.store, p, emb=near),
                        "trigger 语义相近 → 该判同情境")

        self.store.downgrade_profile(p.id)          # 复位成 pending 再试另一组
        far = FakeEmb([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.assertFalse(converge(self.store, p, emb=far),
                         "trigger 语义差得远 → 不算同向收敛")

    def test_no_trigger_text_falls_back_to_class(self):
        """没有 trigger 文本可比时只靠 class 那一层（降级，不是设计）。"""
        p, _ = self.pending_profile(3, trigger="")
        self.assertTrue(converge(self.store, p))

    def test_evidence_record_and_cited_counter(self):
        """印证要留下痕迹：「引用起点时间」+ `cited_by_profile` +1（重要性，不设上限）。

        2026-09-24 起不再写 evidence 边——那条边是对 `sources` 的冗余存储，
        唯一的真数据（被引用那一刻，保护期起算）已搬进画像的 `evidence_at`。
        """
        from core.distill import _mark_evidence
        p, scenes = self.pending_profile(3)
        for s in scenes:
            _mark_evidence(self.store, s.id, p.id)

        got = self.store.get_profile(p.id)
        self.assertIn(scenes[0].id, got.evidence_at, "引用起点时间要记下（保护期起算）")
        self.assertEqual(self.store.get_scene(scenes[0].id).cited_by_profile, 1)

        _mark_evidence(self.store, scenes[0].id, p.id)      # 重复记
        self.assertEqual(self.store.get_scene(scenes[0].id).cited_by_profile, 1,
                         "同一条证据记两次不能让印证数虚高（虚高等于放水）")

    def test_already_established_is_not_reconverged(self):
        p, _ = self.pending_profile(3)
        self.assertTrue(converge(self.store, p))
        self.assertFalse(converge(self.store, p), "已是 established 不该重复判定")

    # ---- 诊断输出（2026-09-12）：失败必须说得出**卡在哪一关** ----
    #
    # 四道关原来全是 `return False`，而失败的表现只有一个：「画像没立」。
    # 可「印证不够」「类别漂了」「阈值太高」的对策完全不同——
    # 分不清就只能翻库逐条猜，那是这类系统最贵的调试成本。

    def test_detail_says_insufficient_evidence(self):
        """印证不足要给出「现有几条 / 需要几条」。"""
        p, _ = self.pending_profile(2)
        ok, why = converge_detail(self.store, p)
        self.assertFalse(ok)
        self.assertIn("印证不足", why)
        self.assertIn("2/3", why)

    def test_detail_names_class_noise(self):
        """情境不一致要列清「哪几类各几条」——那才是能拿去改 prompt 的信息。"""
        for cls in ("被评价", "关系冲突", "失去"):
            self.scene(trigger_class=cls)
        scenes = self.store.query_scenes(topic=TOPIC)
        p = Profile(topic=TOPIC, subject="user", statement="s", status="pending",
                    sources=[s.id for s in scenes])
        self.store.add_profile(p)
        ok, why = converge_detail(self.store, p)
        self.assertFalse(ok)
        self.assertIn("情境不一致", why)
        for cls in ("被评价", "关系冲突", "失去"):
            self.assertIn(cls, why, "得说清是哪几类在打架")

    def test_detail_separates_missing_labels_from_mismatch(self):
        """class 全空是**另一种**病（模型没在分类），不能混进「不一致」。

        两者的对策不同：一个是抽取 prompt 的问题，一个是类别边界的问题。
        混成一个原因，就白做这个诊断了。
        """
        p, _ = self.pending_profile(3, trigger_class="")
        ok, why = converge_detail(self.store, p)
        self.assertFalse(ok)
        self.assertIn("情境标签不足", why)
        self.assertNotIn("不一致", why)

    def test_detail_explains_success_too(self):
        """通过时也给理由（含语义相似度）——那是标定阈值时要看的数。"""
        p, _ = self.pending_profile(3)
        ok, why = converge_detail(self.store, p)
        self.assertTrue(ok)
        self.assertIn("升级", why)


# ---------------------------------------------------------------------
# 三、提炼 3：S2 → S3 抽象 + 修正
# ---------------------------------------------------------------------

class TestStep3(Base):
    def _with_summary(self, n_scenes: int = 3, **over) -> list[Scene]:
        scenes = [self.scene(**over) for _ in range(n_scenes)]
        summary = Summary(topic=TOPIC, text="叙述", sources=[s.id for s in scenes])
        self.store.add_summary(summary)
        return scenes

    def test_no_summary_no_abstraction(self):
        """没有聚合就没有抽象（S2→S3 的次序不能倒）。"""
        self.scene()
        self.assertIsNone(distill_step3(self.store, TOPIC, distill_llm()))

    def test_thin_evidence_stays_pending(self):
        """证据不足 → 画像不写。prompt 要求证据不足给空串，这里验证它被尊重。"""
        self._with_summary(1)
        p = distill_step3(self.store, TOPIC, distill_llm(profiles=[{"statement": ""}]))
        self.assertIsNone(p)
        self.assertEqual(self.store.count("profiles"), 0)

    def test_writes_pending_then_converges(self):
        self._with_summary(3)
        p = distill_step3(self.store, TOPIC, distill_llm())
        self.assertIsNotNone(p)
        self.assertEqual(self.store.get_profile(p.id).status, "established",
                         "三条同情境证据 → 直接收敛为已立")

    def test_sources_link_summaries_to_profile(self):
        """画像的追溯靠 `Profile.sources`（含 S2）——compose 边已退役（2026-09-24）。"""
        self._with_summary(3)
        p = distill_step3(self.store, TOPIC, distill_llm())
        s2 = self.store.summaries_by_topic(TOPIC)[0]
        self.assertIn(s2.id, p.sources)

    def test_profile_topics_include_extra_from_existing(self):
        """画像的附加主题同样只认已有（2026-09-24 晚）——版本序列仍认主主题。"""
        self._with_summary(3)
        self.store.add_summary(Summary(topic="用户·工作", text="旧的", sources=[]))
        llm = distill_llm(profiles=[{"statement": "被评价的场合，他把不被卷入放在前面",
                                     "extra_topics": ["用户·工作", "野主题"]}])
        p = distill_step3(self.store, TOPIC, llm)
        self.assertEqual(self.store.get_profile(p.id).topics, [TOPIC, "用户·工作"])

        # 修正（改陈述）**不该丢附加主题**：主主题是版本键、附加主题是标签，
        # 人改"说法"时它们都没变（`revise_profile` 随版本拷贝 topics）。
        self.scene()
        llm2 = distill_llm(revisions=[{"verdict": "revise", "statement": "换个说法"}])
        new_p = distill_step3(self.store, TOPIC, llm2)
        self.assertEqual(self.store.get_profile(new_p.id).topics, [TOPIC, "用户·工作"])

    def test_holds_appends_evidence(self):
        """新证据支持旧判断 → 追加来源、印证数上升，**不产生新版本**。"""
        self._with_summary(3)
        p = distill_step3(self.store, TOPIC, distill_llm())
        self.assertTrue(p.sources)

        self.scene()                                    # 新证据
        llm = distill_llm(revisions=[{"verdict": "holds", "statement": ""}])
        updated = distill_step3(self.store, TOPIC, llm)
        self.assertEqual(updated.id, p.id, "holds 不该换版本")
        self.assertGreater(len(updated.sources), len(p.sources))
        self.assertEqual(self.store.count("profiles"), 1)

    def test_revise_creates_new_version_and_keeps_old(self):
        """修正 = 旧记录填 invalidated_at + 新记录 valid_at，同 topic 串联。"""
        self._with_summary(3)
        p = distill_step3(self.store, TOPIC, distill_llm())
        self.scene()

        llm = distill_llm(revisions=[{"verdict": "revise", "statement": "遇到被评价会先扛一下"}])
        new_p = distill_step3(self.store, TOPIC, llm)

        self.assertNotEqual(new_p.id, p.id)
        self.assertEqual(self.store.get_profile(p.id).invalidated_at != "", True,
                         "旧版本要被填 invalidated_at（进历史，不删）")
        hist = self.store.profile_history(TOPIC)
        self.assertEqual(len(hist), 2, "同 topic 的两条 = 一条变化轨迹")
        # 新版本是 pending（还没重新攒够印证），所以查「当前有效」要把 status 放开
        self.assertEqual([x.statement for x in self.store.current_profiles(status=None)],
                         ["遇到被评价会先扛一下"])

    def test_overturn_also_revises(self):
        """被推翻 = 修正的特例（新陈述与旧的相反），处理路径相同。"""
        self._with_summary(3)
        p = distill_step3(self.store, TOPIC, distill_llm())
        self.scene()
        llm = distill_llm(revisions=[{"verdict": "overturn", "statement": "其实他会正面沟通"}])
        new_p = distill_step3(self.store, TOPIC, llm)
        self.assertNotEqual(new_p.id, p.id)
        self.assertEqual(self.store.get_profile(p.id).invalidated_at != "", True)

    def test_revision_without_statement_is_ignored(self):
        """说「要改」却没给新陈述 → 当作 holds（不能把旧版本作废了却没新的）。"""
        self._with_summary(3)
        p = distill_step3(self.store, TOPIC, distill_llm())
        self.scene()
        llm = distill_llm(revisions=[{"verdict": "revise", "statement": ""}])
        same = distill_step3(self.store, TOPIC, llm)
        self.assertEqual(same.id, p.id)
        self.assertEqual(self.store.get_profile(p.id).invalidated_at, "")


# ---------------------------------------------------------------------
# 四、topic：新建 vs 复用
# ---------------------------------------------------------------------

class TestTopic(Base):
    def test_resolve_topic_returns_llm_choice(self):
        self.assertEqual(resolve_topic(self.store, "他遇到压力会退出", "user", distill_llm()),
                         TOPIC)

    def test_unknown_topic_is_skipped_not_guessed(self):
        """定不了 topic 就放弃这次抽象——不编一个（编错会污染版本序列）。"""
        self.assertEqual(resolve_topic(self.store, "陈述", "user",
                                       distill_llm(topics=[{"topic": ""}])), "")

    def test_long_topic_is_truncated(self):
        long_topic = "他" * 100
        got = resolve_topic(self.store, "陈述", "user", distill_llm(topics=[{"topic": long_topic}]))
        self.assertEqual(len(got), 40, "topic 是索引键，不能是一整句话")

    def test_merge_topics_reconnects_sequences(self):
        """「拿不准就新建」的补救口：裂开的两个 topic 能合回来。"""
        p1 = Profile(topic="用户·面对压力的反应", statement="a", status="pending")
        p2 = Profile(topic="用户·应对压力的方式", statement="b", status="pending")
        self.store.add_profile(p1)
        self.store.add_profile(p2)
        self.scene(topic="用户·应对压力的方式")

        n = self.store.merge_topics("用户·应对压力的方式", "用户·面对压力的反应")
        self.assertGreater(n, 0)
        self.assertEqual(self.store.get_profile(p2.id).topic, "用户·面对压力的反应")
        self.assertEqual(sorted(self.store.list_topics()), ["用户·面对压力的反应"])


# ---------------------------------------------------------------------
# 五、老化（不滞留的被动版本）
# ---------------------------------------------------------------------

class TestAgeOut(Base):
    def test_downgrades_without_invalidating(self):
        p, _ = self.pending_profile(3)
        self.assertTrue(converge(self.store, p))
        self.store.conn.execute(
            "UPDATE profiles SET last_support_at = '2020-01-01 00:00:00' WHERE id = ?", (p.id,))
        self.store.conn.commit()

        aged = age_out_profiles(self.store)
        self.assertIn(p.id, aged)
        got = self.store.get_profile(p.id)
        self.assertEqual(got.status, "pending")
        self.assertEqual(got.invalidated_at, "", "老化不写 invalidated_at（那是修正/否决专用）")
        self.assertEqual(len(self.store.current_profiles(status=None)), 1,
                         "降级后仍能取到——记录不丢")

    def test_fresh_profile_is_not_aged(self):
        p, _ = self.pending_profile(3)
        converge(self.store, p)
        self.assertEqual(age_out_profiles(self.store), [])


# ---------------------------------------------------------------------
# 六、可否决
# ---------------------------------------------------------------------

class TestReject(Base):
    def test_reject_established_deletes(self):
        """否决 = **真删**（2026-09-24）：删的是"一条不成立的推断"——
        素材都在，判断真成立会被重新立出来。"""
        p, _ = self.pending_profile(3)
        converge(self.store, p)
        outcome = user_reject_profile(self.store, p.id)
        self.assertEqual(outcome, "deleted")
        self.assertIsNone(self.store.get_profile(p.id), "否决 = 真删，不是作废留着")
        self.assertNotIn(p.id, [x.id for x in self.store.current_profiles(status=None)])

    def test_reject_pending_also_deletes(self):
        """pending 被否 → **也删**——不分 pending / established
        （2026-09-11 起如此，2026-09-24 起动作从"作废"变"真删"）。

        关键在**不许它留下来被扶正**：保持 pending 会让它将来被 `converge`
        自动升为事实——用户说了「不对」，系统过一阵自己把它扶正，等于系统
        覆盖了人的否决。真删之后这个可能彻底没有了：行不在了，收敛循环
        （`current_profiles`）根本取不到它。
        """
        p, _ = self.pending_profile(2)
        outcome = user_reject_profile(self.store, p.id)
        self.assertEqual(outcome, "deleted")
        self.assertIsNone(self.store.get_profile(p.id))
        self.assertEqual(self.store.current_profiles(status=None), [],
                         "删掉之后不该再有任何「当前有效」状态")

    def test_revision_and_rejection_are_different_actions(self):
        """「air 修正」和「用户否决」现在是两种动作，必须分得开：
        修正 = 旧版本填 `invalidated_at` 留进历史（系统的动作，不删数据）；
        否决 = 真删（人的动作）——混在一起就永远算不出「猜错多少次」。"""
        p1, scenes = self.pending_profile(3)
        converge(self.store, p1)
        revise_profile(self.store, p1.id, "新陈述",
                       [s.id for s in scenes])
        self.assertEqual(self.store.get_profile(p1.id).invalidated_by, "revision")

        self.store.add_profile(Profile(topic="用户·另一件事", statement="s",
                                       status="established"))
        p2 = self.store.current_profile_by_topic("用户·另一件事")
        user_reject_profile(self.store, p2.id)
        self.assertIsNone(self.store.get_profile(p2.id),
                          "否决是真删——不留 `invalidated_by='user'` 的历史行")

    def test_reject_removes_the_row_and_proposals(self):
        """删画像**连带清它的问卷**：行 + 体检提议——留着就是悬空编号。

        边表已退役（2026-09-24）：`sources` / `evidence_at` 是画像自己的字段，
        跟行一起走，没有第二张表要清。场景（素材）不动。
        """
        p, scenes = self.pending_profile(3)
        self.store.add_profile_review(p.id, "wrong", "不对")
        self.assertEqual(user_reject_profile(self.store, p.id), "deleted")

        self.assertIsNone(self.store.get_profile(p.id), "行不在了 = 它的字段也没了")
        self.assertEqual(self.store.unhandled_reviews(), [], "体检提议也要清")
        self.assertTrue(all(self.store.get_scene(s.id) is not None for s in scenes),
                        "场景是素材——一条都不动（与删场景的差别：那边删的是素材）")

    def test_reject_is_idempotent(self):
        p, _ = self.pending_profile(2)
        self.assertEqual(user_reject_profile(self.store, p.id), "deleted")
        self.assertEqual(user_reject_profile(self.store, p.id),
                         "not_found", "删过就没有了——第二次是找不到，不是又删一次")

    def test_reject_leaves_no_trace(self):
        """否决**不留痕**（2026-09-24）：`否决-*.jsonl` 从此不再写——
        同「删除」的规则：纯动作不留痕（那个"为什么不对"没有人读）。"""
        p, _ = self.pending_profile(2)
        user_reject_profile(self.store, p.id)
        self.assertEqual(list((self.root / "trace").glob("否决-*.jsonl")), [],
                         "删干净")

    def test_reject_unknown_profile(self):
        self.assertEqual(user_reject_profile(self.store, "S3-9999"), "not_found")


# ---------------------------------------------------------------------
# 七、后台周期（幂等）
# ---------------------------------------------------------------------

class TestCycle(Base):
    def test_cycle_end_to_end_then_idempotent(self):
        for _ in range(3):
            self.scene()
        store, llm = self.store, distill_llm()

        first = run_distill_cycle(store, llm)
        self.assertEqual(len(first["s2_new"]), 1, "三条同主题 → 聚合成一条 S2")
        self.assertTrue(first["s3_new"], "有 S2 就该抽出画像")
        self.assertEqual(store.count("summaries"), 1)
        self.assertEqual(store.count("profiles"), 1)

        second = run_distill_cycle(store, distill_llm())
        self.assertEqual(second["s2_new"], [], "没有新场景就不该再聚合")
        self.assertEqual(second["s3_new"] + second["revised"], [], "没有新证据就不该再抽象")
        self.assertEqual(store.count("summaries"), 1)
        self.assertEqual(store.count("profiles"), 1)

    def test_cycle_establishes_three_same_situation(self):
        for _ in range(3):
            self.scene()
        run_distill_cycle(self.store, distill_llm())
        profiles = self.store.current_profiles(status="established")
        self.assertEqual(len(profiles), 1, "三条同情境证据 → 收敛为已立")
        self.assertEqual(profiles[0].evidence, 3)

    def test_cycle_reports_step3_upgrade(self):
        """step3 内部升级的也要计入「本轮升级」——统计不许说谎。

        只靠收敛循环会漏掉它们：画像在 step3 里就立了，轮到循环时已经不是
        pending，于是报告里「收敛为已立 []」看着像收敛逻辑没工作（真踩过）。
        """
        for _ in range(3):
            self.scene()
        stats = run_distill_cycle(self.store, distill_llm())
        est = [p.id for p in self.store.current_profiles(status="established")]
        self.assertEqual(est, stats["established"], "库里立了几条，报告就该记几条")
        self.assertEqual(stats["blocked"], [], "升了的画像不该还在未收敛名单里")

    def test_archive_protects_cited_scenes(self):
        """被画像引用的场景不归档——这是「用户当场否决时能立刻拿出依据」的前提。

        反过来说：同 topic 的场景一旦被画像收进 sources，它们就受保护，
        `archive_s1` 只能在**剩下的**里面挑。所以这个用例用互不相干的 topic，
        才是干净的容量测试。
        """
        for i in range(6):
            self.scene(title=f"场景{i}", topic=f"独立话题{i}")
        n = self.store.archive_s1(cap=2)
        self.assertEqual(n, 4)
        self.assertEqual(self.store.count("scenes"), 6, "归档只标记不删行")
        self.assertEqual(len(self.store.query_scenes()), 2)

    def test_archive_skips_everything_when_all_protected(self):
        """所有场景都在保护期内 → 一条都不归档（宁可超容量，也不让追溯失效）。"""
        scenes = [self.scene(title=f"场景{i}") for i in range(4)]
        p = Profile(topic=TOPIC, statement="s", status="pending",
                    sources=[s.id for s in scenes])
        self.store.add_profile(p)
        self.store.set_profile_evidence_at(p.id, {s.id: now_str() for s in scenes})
        self.assertEqual(self.store.archive_s1(cap=1), 0)
        self.assertEqual(len(self.store.query_scenes()), 4)

    def test_protection_expires(self):
        """**限时保护**：过了保护期就能归档——否则印证的场景会一直新增、
        保护集只涨不消，`s1_cap` 永远不起作用（这正是 2026-09-11 要解的问题）。"""
        from core.store import _add_days
        scenes = [self.scene(title=f"场景{i}") for i in range(4)]
        p = Profile(topic=TOPIC, statement="s", status="pending",
                    sources=[s.id for s in scenes])
        self.store.add_profile(p)
        self.store.set_profile_evidence_at(p.id, {s.id: now_str() for s in scenes})

        grace = cfgmod.cfg("capacity", "citation_grace_days")
        future = _add_days(now_str(), grace + 1)
        self.assertEqual(self.store.protected_scene_ids(future), set(),
                         "保护期内受保护，期外就该放开")
        self.assertGreater(self.store.archive_s1(cap=1, now=future), 0)

    def test_forming_protects_longer_than_supporting(self):
        """出处的保护期比印证长：它是「凭什么」的答案，印证只是支撑量。"""
        forming_scene = self.scene(title="出处")
        supporting_scene = self.scene(title="印证")
        # role 由 `evidence_pack` 派生：出处进包 = forming（长保护期）
        p = Profile(topic=TOPIC, statement="s", status="pending", sources=[],
                    evidence_pack=[{"id": forming_scene.id, "title": "出处"}])
        self.store.add_profile(p)
        self.store.set_profile_evidence_at(p.id, {forming_scene.id: now_str(),
                                                  supporting_scene.id: now_str()})

        cite = int(cfgmod.cfg("capacity", "citation_grace_days"))
        form = int(cfgmod.cfg("capacity", "forming_grace_days"))
        self.assertGreater(form, cite)

        mid = now_str()
        from core.store import _add_days
        at = _add_days(mid, cite + 1)          # 印证已过期、出处还没
        protected = self.store.protected_scene_ids(at)
        self.assertIn(forming_scene.id, protected)
        self.assertNotIn(supporting_scene.id, protected)

    def test_invalidated_profile_stops_protecting(self):
        """画像失效后它旧版本的依据不必继续占保护位（新版本有自己的）。"""
        s = self.scene()
        p = Profile(topic=TOPIC, statement="s", status="established", sources=[s.id])
        self.store.add_profile(p)
        self.store.set_profile_evidence_at(p.id, {s.id: now_str()})
        self.assertIn(s.id, self.store.protected_scene_ids())

        self.store.invalidate_profile(p.id, by="user", reason="不对")
        self.assertNotIn(s.id, self.store.protected_scene_ids())


class ProfileCapTest(Base):
    """画像总量上限：超了**降档，不删**。"""

    def _p(self, topic, ev, status="established"):
        p = Profile(topic=topic, subject="user", statement=topic, status=status,
                    evidence=ev, sources=[])
        self.store.add_profile(p)
        return p

    def test_over_cap_downgrades_the_weakest(self):
        strong = self._p("强的", 9)
        for i in range(5):
            self._p(f"话题{i}", 1)

        out = cap_profiles(self.store, cap=3)

        self.assertEqual(len(out), 3, "超三条就降三条")
        self.assertEqual(self.store.get_profile(strong.id).status, "established",
                         "印证最多的不能动")
        self.assertNotIn(strong.id, out)

    def test_downgraded_profiles_are_still_there(self):
        """降档不是删除：记录还在、还可召回，证据够了还能升回来。"""
        ids = [self._p(f"话题{i}", 1).id for i in range(5)]

        out = cap_profiles(self.store, cap=2)

        for pid in out:
            p = self.store.get_profile(pid)
            self.assertIsNotNone(p, "系统的动作不该删画像（封顶只降档）")
            self.assertEqual(p.status, "pending")
            self.assertFalse(p.invalidated_at, "失效只留给修正")


if __name__ == "__main__":
    unittest.main(verbosity=2)
