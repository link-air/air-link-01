"""阶段 1 最小闭环的测试。

跑法：
    python -m unittest discover -s tests -v
    （或 python tests/test_phase1.py）

两条测试策略：
  - **mock LLM**：`LLM.structured()` 被替换成固定 JSON，把流程与模型解耦。
    不用 mock 就得联网跑测试，那测试就会时好时坏，最后没人跑。
  - **mock 向量**：`should_cut` 是纯函数，直接喂构造好的向量，
    不需要 embedding 服务——测试不该依赖外部服务。

每个用例都对着一条**验收标准**写，命名尽量直白，
让失败信息本身就能说清是哪条约束破了。
"""
# 用例分组：
#   脚手架  Base（临时库 + 假模型）
#   写入侧  TestCut 切分 · TestIntensity 强度 · TestExtract 抽取 ·
#           TestTrivial 寒暄 · TestWorthSaving 值不值得存
#   降级    TestLLMDegrade · TestEmbeddingFallback · TestS0NotSearchable
#   存储    TestSceneVector · TestIdSequence · TestArchive · TestEntities ·
#           TestShortTerm · WindowStateTest
#   读取侧  TestRecall · TestProfiles · ProfileInjectionTest · TestWriteLink
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core.embedding import EmbeddingService
from core.entity import link_entities, match_known_entities, recall_by_entities
from core.llm import LLM
from core.model import Profile, Scene
from core.prompts import SCENE_SCHEMA
from core.recall import (bump_counters, char_overlap, compute_cues, core_score,
                         cue_hits, cue_votes, multi_hit, profile_decay, r0_should_recall,
                         recall, recall_for_message)
from core.distill import distill_step1
from core.scene import behavior_intensity, extract_scene, is_trivial, should_cut
from core.shortterm import ShortTerm, estimate_tokens
from core.store import Store


# ---------------------------------------------------------------------
# 测试替身 / 工具
# ---------------------------------------------------------------------

def unit(angle: float) -> list[float]:
    """2D 单位向量：两个向量的余弦距离 = 1 - cos(夹角差)，便于精确构造用例。"""
    return [math.cos(angle), math.sin(angle)]


def default_card(**over) -> dict:
    """一张完整的场景卡（默认值 = mock LLM 的返回值）。"""
    card = {
        "title": "被组长批评", "keywords": ["组长", "批评"], "summary": "被批评后想走",
        "window_digest": "用户说被组长当众批评，air 表示理解；用户说不想争了。",
        "valence": -1, "arousal": 1, "trigger": "被组长当众批评", "trigger_class": "被评价",
        "reaction": "不想争", "outcome": "", "subject": "user",
        "topic": "用户·面对工作压力的反应", "self_ref": False, "air_stance": "",
        "sensitive": False, "entities": [{"name": "组长", "kind": "person",
                                          "relation": "组长"}],
        "open_loops": [],
    }
    card.update(over)
    return card


def scripted_llm(cards: list[dict] | None = None, cues: dict | None = None) -> LLM:
    """按剧本返回的假 LLM：场景卡按顺序取，最后一张会被重复使用。

    「最后一张重复使用」是刻意的：用例里灌 25 条消息时不想准备 25 张卡，
    但也不想因为 pop 空而抛异常——那会让测试的失败原因变得含混。
    """
    seq = list(cards) if cards else [default_card()]
    fixed_cues = dict(cues) if cues else {}

    def fn(prompt, schema):
        name = schema.get("name")
        if name == "cue_judgement":
            return fixed_cues
        if name == "scene_card":
            return seq.pop(0) if len(seq) > 1 else seq[0]
        return {}

    llm = LLM()
    llm.set_mock(fn)
    return llm


class Base(unittest.TestCase):
    """每个用例一个临时库 + 临时 trace 目录（互不污染，也不写进项目 data/）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = Store(self.root / "test.db")
        # 三个目录都要重定向：留痕、原文文档、备份。
        # 少一个就会被测试污染真实目录（原文文档和备份是后加的两个，
        # 没重定向时真的往 data/raws/ 写过文件——文件里躺着「字字字字」）。
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

    def add(self, **over) -> Scene:
        s = Scene(**over)
        self.store.add_scene(s)
        return s


# ---------------------------------------------------------------------
# 一、切场景（保守：宁可切粗）
# ---------------------------------------------------------------------

class TestCut(Base):
    def test_detects_topic_shift(self):
        """话题转向 → 切；顺着说 → 不切。"""
        window = [unit(0.0), unit(0.02), unit(0.04)]
        self.assertTrue(should_cut(unit(1.0), window, gamma=1.0),
                        "明显转向的新消息应判为边界")
        self.assertFalse(should_cut(unit(0.06), window, gamma=1.0),
                         "顺延同一话题不该切")

    def test_short_window_never_cuts(self):
        """样本不足时不切——短窗口的微小波动不该判成边界（宁可切粗）。"""
        self.assertFalse(should_cut(unit(2.0), [unit(0.0), unit(0.01)]))

    def test_no_embedding_never_cuts(self):
        """向量缺失（embedding 降级）时不切：宁可不切，也不凭猜切。"""
        self.assertFalse(should_cut(None, [unit(0.0), unit(0.1), unit(0.2)]))


# ---------------------------------------------------------------------
# 二、强度是行为信号（不依赖 LLM）
# ---------------------------------------------------------------------

class TestIntensity(Base):
    def test_intensity_prefers_longer_and_repeated(self):
        short = [{"speaker": "user", "text": "嗯"}]
        long_ = [{"speaker": "user", "text": "我" * 300}]
        self.assertGreater(behavior_intensity(long_), behavior_intensity(short))
        self.assertGreater(behavior_intensity(long_, prev_mentions=3),
                           behavior_intensity(long_, prev_mentions=0))

    def test_intensity_bounded(self):
        msgs = [{"speaker": "user", "text": "我" * 1000 + "?"}]
        val = behavior_intensity(msgs, prev_mentions=99)
        self.assertLessEqual(val, 1.0)
        self.assertGreaterEqual(val, 0.0)


# ---------------------------------------------------------------------
# 三、抽取：两个摘要不混 + 保守默认
# ---------------------------------------------------------------------

class TestExtract(Base):
    def test_two_summaries_are_separate(self):
        """`summary`（一句话）与 `window_digest`（压缩对话）必须分字段。"""
        card = default_card(summary="一句话", window_digest="一段长很多的来龙去脉" * 5)
        scene, digest, entities, _ = extract_scene(
            [{"speaker": "user", "text": "x"}], scripted_llm([card]))
        self.assertEqual(scene.text, "一句话")
        self.assertIn("来龙去脉", digest)
        self.assertNotEqual(scene.text, digest)

    def test_unsure_emotion_stays_none(self):
        """valence/arousal 拿不准 → None（**不是 0**：0 表示确定的中性）。"""
        scene, _, _, _ = extract_scene(
            [{"speaker": "user", "text": "x"}],
            scripted_llm([default_card(valence=None, arousal=None)]))
        self.assertIsNone(scene.valence)
        self.assertIsNone(scene.arousal)

    def test_illegal_trigger_class_dropped(self):
        """trigger_class 不在 7 类枚举内 → 空串（不猜；它是统计索引）。"""
        scene, _, _, _ = extract_scene(
            [{"speaker": "user", "text": "x"}],
            scripted_llm([default_card(trigger_class="随便编的")]))
        self.assertEqual(scene.trigger_class, "")

    def test_sensitive_defaults_to_true_on_failure(self):
        """LLM 失败 → 保守取敏感（它只影响说出口力度，不影响召回）。"""
        llm = LLM()
        llm.set_mock(lambda p, s: (_ for _ in ()).throw(RuntimeError("boom")))
        scene, _, _, _ = extract_scene([{"speaker": "user", "text": "x"}], llm)
        self.assertEqual(scene.sensitive, 1)
        self.assertIsNone(scene.valence)

    def test_entity_without_relation_is_dropped(self):
        """没关系的名字不进实体（2026-09-25）：公共人物 / 话题词连场景卡都不进。

        判据放在抽取层（`scene._norm_entities`）——「他提一百次也不记」，
        除非出现关系 / 态度表述（那时才给 relation、才留下）。
        """
        card = default_card(entities=[
            {"name": "特朗普", "kind": "person"},                       # 无关系 → 丢
            {"name": "我妈", "kind": "person", "relation": "妈妈"},      # 有关系 → 留
        ])
        _, _, entities, _ = extract_scene(
            [{"speaker": "user", "text": "x"}], scripted_llm([card]))
        self.assertEqual([e["name"] for e in entities], ["我妈"])
        self.assertEqual(entities[0]["relation"], "妈妈")


# ---------------------------------------------------------------------
# 四、LLM 降级不崩
# ---------------------------------------------------------------------

class TestLLMDegrade(Base):
    def test_structured_never_raises(self):
        broken = LLM()
        broken.set_mock(lambda p, s: (_ for _ in ()).throw(RuntimeError("boom")))
        data = broken.structured("p", SCENE_SCHEMA)
        self.assertEqual(data["sensitive"], True)
        self.assertEqual(data["subject"], "user")
        self.assertIsNone(data["valence"])

    def test_unavailable_llm_returns_defaults(self):
        """没配服务 → 直接给默认值，不抛、不阻塞链路。"""
        data = LLM().structured("p", SCENE_SCHEMA)
        self.assertEqual(set(data.keys()), set(SCENE_SCHEMA["defaults"].keys()))

    def test_fill_drops_unknown_keys_and_fixes_types(self):
        llm = LLM()
        llm.set_mock(lambda p, s: {"keywords": "不是数组", "title": "标题", "额外字段": 1})
        data = llm.structured("p", SCENE_SCHEMA)
        self.assertNotIn("额外字段", data)
        self.assertEqual(data["keywords"], [])
        self.assertEqual(data["title"], "标题")


# ---------------------------------------------------------------------
# 五、embedding 降级 + 字符重叠兜底
# ---------------------------------------------------------------------

class TestEmbeddingFallback(Base):
    def test_unconfigured_returns_none(self):
        self.assertIsNone(EmbeddingService().embed_one("你好"))

    def test_char_overlap_is_meaningful(self):
        """降级兜底得有方向感：同主题文本的重叠明显高于无关文本。"""
        near = char_overlap("今天又被组长当众批评了", "被组长当众批评后想离开")
        far = char_overlap("今天又被组长当众批评了", "dataclass 和 pydantic 的区别")
        self.assertGreater(near, far)
        self.assertEqual(char_overlap("", "x"), 0.0)


# ---------------------------------------------------------------------
# 六、S0 不可直接检索（物理约束，不是调用方自觉）
# ---------------------------------------------------------------------

class TestSceneVector(Base):
    """向量的存读往返——C1 语义检索全压在这上面。

    这个用例是补的，因为发现得晚：此前所有测试都在「没配向量服务」的降级状态跑，
    向量全是 None，于是「存不进去」和「本来就是空的」看起来一模一样。
    **降级路径跑通 ≠ 正常路径跑通**——这个坑值得留一个用例钉住。
    """

    def test_vector_roundtrip(self):
        s = Scene(text="x", emb=[0.1, 0.2, 0.3])
        self.store.add_scene(s)

        got = self.store.get_scene(s.id)
        self.assertIsNotNone(got.emb, "向量该被存下来（BLOB 通道）")
        self.assertEqual(len(got.emb), 3)
        self.assertAlmostEqual(got.emb[0], 0.1, places=5)

    def test_all_embeddings_sees_it(self):
        self.store.add_scene(Scene(text="a", emb=[1.0, 0.0]))
        self.store.add_scene(Scene(text="b"))          # 没有向量
        pairs = self.store.all_embeddings()
        self.assertEqual(len(pairs), 1, "全量加载只该返回有向量的那些")

    def test_all_embeddings_cache_stays_consistent(self):
        """向量表缓存（2026-09-24）：读得快，但**写后必须立刻看得见**。

        失效口径是"宁可多失效"——漏失效的后果是"检索到已删 / 已归档的，
        或漏掉刚写入的"，静默且难查（`store._bump_emb` 的五个调用点）。
        """
        a = self.store.add_scene(Scene(text="a", emb=[1.0, 0.0]))
        self.assertEqual([i for i, _ in self.store.all_embeddings()], [a])

        b = self.store.add_scene(Scene(text="b", emb=[0.0, 1.0]))
        self.assertIn(b, [i for i, _ in self.store.all_embeddings()], "写入后立刻看得见")

        self.store.archive_scene(a)
        self.assertNotIn(a, [i for i, _ in self.store.all_embeddings()],
                         "归档后不该再出现在检索输入里")

        self.store.unarchive_scene(a)
        self.assertIn(a, [i for i, _ in self.store.all_embeddings()], "回热层后要回来")

        self.store.delete_scene(b)
        self.assertNotIn(b, [i for i, _ in self.store.all_embeddings()], "删了就不该在")


class TestIdSequence(Base):
    """id 必须按**数值**递增——第 10000 条不能撞回 `S1-9999`。

    `ORDER BY id DESC` 是字典序：`"S1-10000" < "S1-9999"`（比到第 4 位，`1` < `9`），
    到五位数时它会取到错的"最大值"，算出重复 id → INSERT 撞主键。
    归档只标记不删行，总行数会一直涨：这条线迟早被跨过去。
    """

    def test_next_id_is_numeric_not_lexicographic(self):
        with self.store.conn:
            self.store.conn.execute(
                "INSERT INTO scenes (id, title) VALUES ('S1-9999', '九')")
            self.store.conn.execute(
                "INSERT INTO scenes (id, title) VALUES ('S1-10000', '万')")
        self.assertEqual(self.store._next_id("scenes"), "S1-10001")


class TestS0NotSearchable(Base):
    def test_s0_lives_in_documents_not_tables(self):
        """S0 不可检索是**物理实现**的，不靠调用方自觉。

        原来是检查 `raws` 表只有最小列；现在更进一步——**它根本不建表**
        （连"多一列"的机会都没有）。原文按天存文档，只能经 `scene_id` 下钻。
        """
        tables = {r["name"] for r in self.store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("raws", tables, "原文不再入库；库里只留结构化记忆")

    def test_raw_lookup_needs_scene_id(self):
        """**下钻**的入口只有 `get_raws_by_scene`——没有"按内容/时间查原文"的口子。

        （打捞 `salvage.search_raws` 能按日期 + 语义翻原文，但那是**人主动发起**的
        另一条路，不进自动唤醒，也不经过这个入口——两者不冲突。）
        """
        import inspect
        params = set(inspect.signature(self.store.get_raws_by_scene).parameters)
        self.assertEqual(params, {"scene_id", "on_date"},
                         "唯一入口只该收 scene_id（和可选的日期定位），"
                         "不能出现关键词、时间这类检索参数")
        self.assertFalse(self.store.get_raws_by_scene(""))

    def test_scenes_query_has_no_sensitive_filter(self):
        """`query_scenes` 不得提供 sensitive 过滤——敏感场景仍要参与召回。"""
        import inspect
        params = inspect.signature(self.store.query_scenes).parameters
        self.assertNotIn("sensitive", params)


# ---------------------------------------------------------------------
# 七、画像：双时间戳 / 老化不丢 / pending 不常驻
# ---------------------------------------------------------------------

class TestProfiles(Base):
    def test_double_timestamp_revision(self):
        old = Profile(topic="用户·面对压力的反应", statement="他会退出", status="established")
        self.store.add_profile(old)
        self.store.invalidate_profile(old.id, at="2026-03-01 00:00:00")
        new = Profile(topic="用户·面对压力的反应", statement="他会先扛一下", status="established")
        self.store.add_profile(new)

        cur = self.store.current_profiles()
        self.assertEqual([p.statement for p in cur], ["他会先扛一下"])
        self.assertEqual(len(self.store.profile_history("用户·面对压力的反应")), 2)

    def test_age_out_keeps_record(self):
        """老化降级 = 回到待验证，**不填 invalidated_at、记录不丢**。"""
        p = Profile(topic="用户·X", statement="S", status="established")
        self.store.add_profile(p)
        self.store.downgrade_profile(p.id)

        got = self.store.get_profile(p.id)
        self.assertEqual(got.status, "pending")
        self.assertEqual(got.invalidated_at, "", "老化不该写 invalidated_at（那是修正/否决专用）")
        self.assertIn(p.id, [x.id for x in self.store.current_profiles(status=None)],
                      "降级后的画像仍应能取到（可召回，只是不进常驻）")

    def test_set_profile_status_rejects_unknown(self):
        """写入口校验：不认识的画像状态**当场报错**（不静默写脏值）。

        `PROFILE_STATUS` 域元组的唯一读点就是这里——状态是统计与准入的
        索引，拼错一个值写进去，这条画像两头都不算。
        """
        p = Profile(topic="用户·状态校验", statement="S", status="pending")
        self.store.add_profile(p)
        self.store.set_profile_status(p.id, "established")
        self.assertEqual(self.store.get_profile(p.id).status, "established")

        with self.assertRaises(ValueError, msg="拼错的状态必须当场报错"):
            self.store.set_profile_status(p.id, "established ")
        with self.assertRaises(ValueError):
            self.store.set_profile_status(p.id, "published")
        self.assertEqual(self.store.get_profile(p.id).status, "established",
                         "被拒的写入不该改动库里的值")

    def test_pending_not_resident(self):
        self.store.add_profile(Profile(topic="用户·Y", statement="猜测", status="pending"))
        self.assertEqual(self.store.current_profiles(), [],
                         "pending 不进常驻（R4 只取 established）")
        self.assertEqual(len(self.store.current_profiles(status=None)), 1)

    def test_invalidate_is_not_age_out(self):
        """修正 / 否决走 invalidated_at；老化走 status——两条路不能混。"""
        p = Profile(topic="用户·Z", statement="S", status="established")
        self.store.add_profile(p)
        self.store.invalidate_profile(p.id, at="2026-01-01 00:00:00")
        got = self.store.get_profile(p.id)
        self.assertEqual(got.invalidated_at, "2026-01-01 00:00:00")
        self.assertEqual(got.status, "established", "否决不改 status（状态与失效是两回事）")


# ---------------------------------------------------------------------
# 八、归档：容量有界 + 被引用保护（只标记不删行）
# ---------------------------------------------------------------------

class TestArchive(Base):
    def test_cited_scenes_are_protected(self):
        """被**当前有效画像**引用、且在保护期内的场景不归档（追溯要即时）。

        注意保护的依据是**画像侧的"引用起点时间"**（`profile.evidence_at`；
        2026-09-24 前它住在 evidence 边里），**不是** `cited_by_profile`（2026-09-11 改）：
        计数器只说明「被引用过多少次」，说明不了「是不是当前有效画像的出处」，
        更没法限时——而限时是必需的，否则印证的场景会一直新增、保护集只涨不消。
        """
        from core.model import Profile
        from core.store import now_str
        kept = self.add(text="被画像引用的那条")
        p = Profile(topic="用户·X", statement="s", status="pending", sources=[kept.id])
        self.store.add_profile(p)
        self.store.set_profile_evidence_at(p.id, {kept.id: now_str()})
        others = [self.add(text=f"普通{i}") for i in range(4)]

        self.store.archive_s1(cap=2)

        self.assertEqual(self.store.get_scene(kept.id).archived, 0, "被引用的场景不该进冷层")
        archived = [s for s in others if self.store.get_scene(s.id).archived == 1]
        self.assertEqual(len(archived), 3)

    def test_archive_marks_but_never_deletes(self):
        for i in range(3):
            self.add(text=f"x{i}")
        self.store.archive_s1(cap=1)
        self.assertEqual(self.store.count("scenes"), 3, "归档只标记、不删行")
        self.assertEqual(len(self.store.query_scenes()), 1, "默认查询不含冷层")
        self.assertEqual(len(self.store.query_scenes(include_archived=True)), 3)


# ---------------------------------------------------------------------
# 九、实体索引与旁路
# ---------------------------------------------------------------------

class TestEntities(Base):
    def test_entity_bypass_hits_without_embedding(self):
        s = self.add(text="和小明吃饭", title="和小明吃饭")
        link_entities(s.id, [{"name": "小明", "kind": "person", "relation": "同事"}], self.store)

        names = match_known_entities("小明最近怎么样了", self.store)
        self.assertEqual(names, ["小明"])
        hits = recall_by_entities(names, self.store)
        self.assertEqual([h.id for h in hits], [s.id])

    def test_unknown_name_is_not_a_cue(self):
        """库里没有的名字不该成为线索——那是新信息，不是"想起什么"。"""
        self.assertEqual(match_known_entities("小红最近怎么样", self.store), [])

    def test_entity_without_relation_is_not_indexed(self):
        """`link_entities` 的第二道闸（同抽取层那条）：没 relation 不建不挂——
        直接调本函数的路径（打捞重记 / 未来的导入）同样受约束。"""
        s = self.add(text="看新闻", title="看新闻")
        link_entities(s.id, [{"name": "特朗普", "kind": "person"}], self.store)
        self.assertEqual(self.store.all_entities(), [])

    def test_relation_is_stored_per_scene(self):
        """relation 逐场景落库（实体页展示用）——「妈妈」和后来的角色可以不同。"""
        s = self.add(text="和小明吃饭", title="和小明吃饭")
        link_entities(s.id, [{"name": "小明", "kind": "person",
                              "relation": "同事"}], self.store)
        e = self.store.all_entities()[0]
        row = self.store.conn.execute(
            "SELECT relation FROM scene_entities WHERE scene_id=? AND entity_id=?",
            (s.id, e.id)).fetchone()
        self.assertEqual(row["relation"], "同事")

    def test_no_auto_merge_two_same_names(self):
        """第一版不做自动归并：宁可新建（可见、可人工合），也不猜。"""
        s1 = self.add(text="a")
        s2 = self.add(text="b")
        link_entities(s1.id, [{"name": "小明", "kind": "person", "relation": "同事"}], self.store)
        link_entities(s2.id, [{"name": "小明", "kind": "person", "relation": "同事"}], self.store)
        self.assertEqual(len(self.store.all_entities()), 1, "同名精确匹配应复用同一条")

        link_entities(s2.id, [{"name": "小明明", "kind": "person",
                              "relation": "同事"}], self.store)
        self.assertEqual(len(self.store.all_entities()), 2, "不同名不自动归并")


# ---------------------------------------------------------------------
# 十、短期窗口：预算触发 + 压缩=提取
# ---------------------------------------------------------------------

class TestShortTerm(Base):
    def test_budget_triggers_extract_and_keeps_tail(self):
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        budget = cfgmod.cfg("shortterm", "token_budget")
        keep = cfgmod.cfg("shortterm", "verbatim_messages")

        # 灌到明显超预算（中文 1 字 ≈ 1 token）
        per = budget // 5
        for i in range(keep + 5):
            st.append("user", "字" * per)
            st.append("air", "嗯")

        self.assertGreater(st.window_tokens(), budget)
        self.assertTrue(st.should_extract())

        info = st.flush_if_needed()
        self.assertIsNotNone(info)
        self.assertEqual(info["scene_id"][:2], "S1")
        self.assertEqual(self.store.count("scenes"), 1)
        # 预算触发只压最老的一批，尾部逐字留着（对话还在继续）
        self.assertGreater(len(st.messages), 0)
        self.assertLessEqual(len(st.messages), keep)

    def test_max_turns_triggers_extract(self):
        """超 M 轮未切换 → 强制提取（防一段太长一直不提取）。"""
        old = cfgmod.CONFIG["shortterm"]["max_turns_no_cut"]
        cfgmod.CONFIG["shortterm"]["max_turns_no_cut"] = 1
        try:
            st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                           session_id="t", state_path=self.root / "st.json")
            st.append("user", "嗯")
            st.append("air", "在")
            self.assertTrue(st.should_extract(), "一轮没切就该提取（阈值调成 1）")
        finally:
            cfgmod.CONFIG["shortterm"]["max_turns_no_cut"] = old

    def test_session_idle_triggers_extract(self):
        """空闲超时 = 会话自然结束——第 4 条触发的另一条到达方式。

        判的是「**他说这一句之前**隔了多久」（2026-10-09 修）：间隔在 `append` 里算。
        此前判的是"现在离最后一条消息多久"，而判定挂在 flush、flush 又在 append 之后
        ——那一刻恒为 0，这条触发在真实链路里从没生效过（L6 核对文档记过这个发现）。
        """
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        st.append("user", "在吗", ts="2026-01-01 10:00:00")
        st.append("air", "在", ts="2026-01-01 10:00:05")
        self.assertFalse(st.session_idle(), "刚聊了一轮，不算空闲")
        st.append("user", "我又来了", ts="2026-01-01 11:00:00")   # 隔了一小时
        # 她的回话紧跟其后成对入窗——**它不许把间隔刷成 0**（否则这条又永远不生效）
        st.append("air", "嗯", ts="2026-01-01 11:00:02")
        self.assertTrue(st.session_idle())
        self.assertTrue(st.should_extract())

    def test_idle_extract_keeps_the_new_turn(self):
        """空闲只切「上一段」：新段的第一轮留在窗口里（2026-10-11）。

        场景就是真实踩到的那个：隔了很久回来说一句、聊完——若把这一轮也收走，
        「重新生成」点下去只能得到"窗口是空的"（它已经进长期库了）。
        """
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        st.append("user", "第一句", ts="2026-01-01 10:00:00")
        st.append("air", "嗯", ts="2026-01-01 10:00:05")
        st.append("user", "第二句", ts="2026-01-01 10:01:00")
        st.append("air", "嗯嗯", ts="2026-01-01 10:01:05")
        st.append("user", "隔了一小时才说的第三句", ts="2026-01-01 11:00:00")
        st.append("air", "回来了", ts="2026-01-01 11:00:05")
        self.assertTrue(st.session_idle(), "隔了一小时，算空闲")

        info = st.flush_if_needed()
        self.assertIsNotNone(info, "旧段该被提取")
        self.assertEqual(self.store.count("scenes"), 1, "旧段落了场景卡")
        self.assertTrue(st.digest, "旧段进了压缩摘要")
        # 新段的第一轮（这条 user + 她的回话）留下 → 撤得动 / 重新生成得了
        self.assertEqual([m["text"] for m in st.messages],
                         ["隔了一小时才说的第三句", "回来了"])

    def test_idle_without_old_segment_extracts_nothing(self):
        """窗口里只有新段的第一轮 → 空闲触发无事可做（不白调 LLM，也不动窗口）。

        这是那个坑的最纯形态：当天第一句话说出口、聊完就进长期库——
        窗口里根本没有"上一段"可以切。
        """
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        st.append("user", "上一段的尾巴", ts="2026-01-01 09:00:00")
        st.append("air", "嗯", ts="2026-01-01 09:00:05")
        st.messages = []          # 模拟旧段已被提取走（窗口清空、`_last_active` 还留着）
        st.append("user", "隔天回来的第一句", ts="2026-01-02 10:00:00")
        st.append("air", "嗯", ts="2026-01-02 10:00:05")
        self.assertTrue(st.session_idle())

        self.assertIsNone(st.flush_if_needed(), "没有旧段可收——不该提取")
        self.assertEqual(len(st.messages), 2, "新段原封不动地留着")
        self.assertEqual(self.store.count("scenes"), 0, "没有落任何场景")

    def test_stacked_triggers_keep_the_new_turn(self):
        """多条触发叠加（超预算 + 切换 + 空闲）：仍"只切旧段"（2026-10-11 复查）。

        长消息把窗口灌到超预算，与旧段零重叠的收尾句在降级判据下又判成
        切换、且隔了一小时——三条一起成立。初版把"预算未超"和"非切换"
        当条件，这类叠加会退回整窗：刚聊完的一轮又被收走。
        """
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        per = cfgmod.cfg("shortterm", "token_budget") // 3
        for i in range(2):                       # 四条长消息：把窗口灌到超预算
            st.append("user", "字" * per, ts=f"2026-01-01 10:0{i}:00")
            st.append("air", "字" * per, ts=f"2026-01-01 10:0{i}:05")
        st.append("user", "隔了一小时才说的这句", ts="2026-01-01 11:00:00")
        st.append("air", "嗯", ts="2026-01-01 11:00:05")
        self.assertGreater(st.window_tokens(), cfgmod.cfg("shortterm", "token_budget"))
        self.assertTrue(st.session_idle())
        self.assertTrue(st._pending_cut, "零重叠的收尾句还会被判成切换——这条就是三条叠加")

        info = st.flush_if_needed()
        self.assertIsNotNone(info, "旧段该被提取")
        self.assertEqual([m["text"] for m in st.messages],
                         ["隔了一小时才说的这句", "嗯"],
                         "叠加触发也留住新段（不然「重新生成」又撤不动）")

    def test_topic_switch_extracts_old_segment_only(self):
        """话题切换也只切「上一段」——设计稿 §二：切换即提取上一段（2026-10-11）。

        原来它跟"整段收"混在一起：新话题的第一轮被一起收走，刚聊完想改 /
        重新生成就撤不动（与空闲那个坑同源）。
        """
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        for i in range(2):
            st.append("user", "字" * 300, ts=f"2026-01-01 10:0{i}:00")
            st.append("air", "字" * 300, ts=f"2026-01-01 10:0{i}:05")
        # 与前面零重叠 → 降级判据（无向量）必然判成切换；间隔很短，不沾空闲
        st.append("user", "换个话题：明天要不要带伞", ts="2026-01-01 10:01:00")
        st.append("air", "带吧", ts="2026-01-01 10:01:05")
        self.assertTrue(st._pending_cut, "降级判据该把它判成话题切换")
        self.assertFalse(st.session_idle(), "这条用例不沾空闲")

        info = st.flush_if_needed()
        self.assertIsNotNone(info, "旧段该被提取")
        self.assertEqual([m["text"] for m in st.messages],
                         ["换个话题：明天要不要带伞", "带吧"],
                         "切换点之后的这一轮要留在窗口里")

    def test_empty_window_end_session_does_not_leak(self):
        """空窗时的收尾标记不许留到下一段会话（2026-10-11 复查）。

        `end_session()` 时窗口空（上一轮刚被提取过，或「新对话」点得正好）——
        flush 会早退；若标记不清理，它会跟着同一个实例留到新会话：
        第一轮又被当"整窗收尾"提走，「重新生成」撤不动。
        """
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        st.end_session()                 # 窗口空——没有尾巴可收
        st.flush_if_needed()
        st.append("user", "新一段的第一句", ts="2026-01-01 10:00:00")
        st.append("air", "嗯", ts="2026-01-01 10:00:05")
        self.assertIsNone(st.flush_if_needed(), "新会话第一轮不该被提取")
        self.assertEqual(len(st.messages), 2, "窗口原封不动")

    def test_air_turns_are_buffered(self):
        """**air 自己的话也进缓冲**——记忆的对象是「这段互动」，不是「用户」。

        少了这一半，她就记不住自己说过什么（承诺、边界、态度），
        而「air 的一致性」正是靠它维持的。
        """
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        st.append("user", "今天很难受")
        st.append("air", "我在")
        speakers = [m["speaker"] for m in st.messages]
        self.assertIn("user", speakers)
        self.assertIn("air", speakers, "air 的话也要进缓冲，不能只记用户说的")
        self.assertIn("我在", st.build_window())

    def test_empty_window_does_not_call_llm(self):
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        self.assertIsNone(st.flush_if_needed())

    def test_window_renders_digest_and_verbatim(self):
        st = ShortTerm(self.store, scripted_llm(), emb_service=None,
                       session_id="t", state_path=self.root / "st.json")
        st.append("user", "第一句")
        st.digest = ["更早发生的事"]
        win = st.build_window()
        self.assertIn("更早发生的事", win)
        self.assertIn("第一句", win)

    def test_token_estimate_is_rough_but_ordered(self):
        self.assertGreater(estimate_tokens("我" * 100), estimate_tokens("我" * 10))
        self.assertEqual(estimate_tokens(""), 0)


# ---------------------------------------------------------------------
# 十一、写入链路：S0 落库 + 指针边 + open_loops 进 memo
# ---------------------------------------------------------------------

class WindowStateTest(unittest.TestCase):
    """窗口文件的**只读解析**（启动恢复 + 界面回填共用这一个出口）。

    `read_state` 的存在理由：`_load`（启动恢复）和 `App.history`
    （重启后对话区回填）读同一个文件——分头解析迟早漂移（老格式转换就是个坑）。
    """

    def test_missing_file_gives_empty_shape(self):
        with tempfile.TemporaryDirectory() as d:
            out = ShortTerm.read_state(Path(d) / "nope.json")
        self.assertEqual(out, {"session_id": "", "messages": [],
                               "digest": [], "last_active": ""})

    def test_legacy_string_digest_is_converted(self):
        """老格式的 digest 是一整坨字符串（那时一行 = 一批），要转成批列表。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "st.json"
            p.write_text(json.dumps(
                {"digest": "一行摘要",
                 "messages": [{"speaker": "user", "text": "甲"}]},
                ensure_ascii=False), encoding="utf-8")
            out = ShortTerm.read_state(p)
        self.assertEqual(out["digest"], ["一行摘要"])
        self.assertEqual(len(out["messages"]), 1)

    def test_broken_file_is_tolerated(self):
        """坏文件不抛、给空结构——它不该让界面或启动卡住。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "st.json"
            p.write_text("{ 不是 json", encoding="utf-8")
            out = ShortTerm.read_state(p)
        self.assertEqual(out["messages"], [])
        self.assertEqual(out["digest"], [])


class TestWriteLink(Base):
    def test_step1_writes_s0_s1_edge_and_memo(self):
        card = default_card(open_loops=[
            {"content": "下周三面试", "kind": "user_task", "due_at": "2026-09-16"},
            {"content": "我会提醒你", "kind": "air_promise", "due_at": ""},
        ])
        st = ShortTerm(self.store, scripted_llm([card]), emb_service=None,
                       session_id="s1", state_path=self.root / "st.json")
        st.append("user", "我下周三要面试")
        st.append("air", "我会提醒你")
        st.end_session()
        info = st.flush_if_needed()

        scene_id = info["scene_id"]
        raws = self.store.get_raws_by_scene(scene_id)
        self.assertEqual(len(raws), 1, "S0 原文必须留下（按天文档）")
        self.assertIn("我下周三要面试", raws[0].content)
        # 原文指针**不再存边**（2026-09-24，存储层稿 §五）：按天文档本身就是索引，
        # `get_raws_by_scene` 直接翻——这里断言"翻得到"，不查边。
        day = (self.store.get_scene(scene_id).time_record or "")[:10]
        self.assertTrue((self.store.raws_dir() / f"{day}.md").exists(),
                        "原文文档在（按场景发生日）")
        memos = self.store.memos_by_scene(scene_id)
        self.assertEqual(len(memos), 2)
        self.assertEqual({m.kind for m in memos}, {"user_task", "air_promise"})
        self.assertEqual([m.due_at for m in memos if m.kind == "user_task"], ["2026-09-16"])
        self.assertTrue(all(m.status == "pending" for m in memos))

    def test_topic_neighbors_are_derived_within_same_topic(self):
        """同 topic 的前后相邻**现算**（2026-09-24）——`causality` 边已退役。

        语义与旧边一致：旧边写的是"上一条 → 本条"、读时双向查，合计就是
        "同 topic 的相邻对"；这里 `topic_neighbors()` 现算前后各一条，等价。
        跨 topic 不连——全连等于没连，扩散会失去选择性。
        """
        st = ShortTerm(self.store, scripted_llm([
            default_card(topic="用户·工作"), default_card(topic="用户·工作"),
            default_card(topic="猫·健康"),
        ]), emb_service=None, session_id="s", state_path=self.root / "st.json")
        ids = []
        for text in ("第一段", "第二段", "第三段"):
            st.append("user", text)
            st.end_session()
            ids.append(st.flush_if_needed()["scene_id"])

        self.assertEqual(self.store.topic_neighbors(ids[0]), [ids[1]],
                         "同 topic 才相邻")
        self.assertEqual(self.store.topic_neighbors(ids[1]), [ids[0]],
                         "双向等价（旧边读时就是双向的）")
        self.assertEqual(self.store.topic_neighbors(ids[2]), [],
                         "跨 topic 不连——全连等于没连，扩散会失去选择性")


# ---------------------------------------------------------------------
# 十二、唤醒：R0 / 多维协同 / 不偏好负面 / trace
# ---------------------------------------------------------------------

class TestRecall(Base):
    def test_r0_defaults_to_searching(self):
        self.assertTrue(r0_should_recall("我最近有点累"), "含情绪词 → 翻")
        self.assertTrue(r0_should_recall("上次那个事"), "含指代 / 时间词 → 翻")
        self.assertTrue(r0_should_recall("今天天气怎么样"), "含时间词 → 翻")
        self.assertFalse(r0_should_recall("TCP 三次握手是什么"), "纯知识、无个人特征 → 不翻")

    def test_votes_are_weighted_by_strength(self):
        """票制（2026-10-05）：强维 2 票、弱维 1 票，**强信号 = 票数 ≥3 且含强维**。

        两条要分开看的老口径：原来数"维数"（≥2 维就算强）——
        「表达+情绪」与「情绪+时间」一样算 2 维，后者只是两个弱信号凑数。
        """
        weak_two = {"C2": "past", "C3": {"arousal": 1}, "C1": 0.1, "C4": 0.1,
                    "C5": 0, "C6": 1.0, "C7": 0}          # 时态 + 情绪 = 2 票（无强维）
        strong_one = {"C1": 0.9, "C2": "now", "C3": {"arousal": 0}, "C4": 0.1,
                      "C5": 0, "C6": 1.0, "C7": 0}          # 语义单维 = 2 票（一个强维单打）
        strong_plus = {"C1": 0.9, "C2": "now", "C3": {"arousal": 1}, "C4": 0.1,
                       "C5": 0, "C6": 1.0, "C7": 0}         # 强 + 弱 = 3 票
        self.assertEqual(cue_votes(weak_two)["votes"], 2)
        self.assertFalse(cue_votes(weak_two)["strong"], "两个弱维凑不出强维")
        self.assertFalse(multi_hit(weak_two), "两个弱维不再算强信号（原文最贵，不给）")
        self.assertFalse(multi_hit(strong_one), "一个强维单打也不给（2 票 < 3）")
        self.assertTrue(multi_hit(strong_plus), "强 + 弱 = 3 票 ✓")

    def test_degraded_mode_does_not_count_literal_as_a_vote(self):
        """降级时 C6 与**字面旁路**都不计票——它们和 C1 同源（都退化成字符重叠），
        一维算两遍会让"多维协同"名不副实（同 2026-09-22 对 C6 的处理）。"""
        cues = {"C1": 0.9, "C2": "now", "C3": {"arousal": 0}, "C4": 0.1,
                "C5": 0, "C6": 1.0, "C7": 0, "literal": {"脱敏": ["S1-0001"]},
                "_degraded": True}
        v = cue_votes(cues)
        self.assertEqual(v["votes"], 2, "只剩 C1 那 2 票（字面是它的同源影子）")

    def test_literal_bypass_hits_a_rare_word(self):
        """字面旁路（2026-10-05）：库里只出现一次的**罕见词**算强维命中；
        满库都有的词不算（那是噪声不是线索）。**不查词表——罕见度是数出来的。**

        两条规矩都在这里：① **长的吃掉短的**（一个术语的各种切片只算一个词，
        否则"脱敏脚本"会连同"脱敏脚""敏脚本"占掉三格名额）；
        ② df 超过 `recall.literal_max_df`（初值 3）就不算罕见。
        """
        self.store.add_scene(Scene(title="脱敏脚本", text="在写一个开源脱敏脚本"))
        for i in range(4):                       # 「面试」在 4 个场景里出现 → 满库都有
            self.store.add_scene(Scene(title=f"面试{i}", text="面试相关的事"))

        cues = compute_cues("那个脱敏脚本呢", self.store, emb=None, llm=None)
        self.assertEqual(list(cues["literal"]), ["脱敏脚本"],
                         "只在一个场景里出现过的词才算罕见词；切片被它吃掉")
        self.assertTrue(cue_hits(cues)["literal"])

        cues2 = compute_cues("面试怎么样", self.store, emb=None, llm=None)
        self.assertEqual(cues2["literal"], {},
                         f"df=4 超过 literal_max_df=3 —— 满库都有的词不算命中")

    def test_cue_hits_is_the_single_source_of_truth(self):
        """`multi_hit` / `cue_votes` 和仪表盘高亮读**同一份**判定（`cue_hits`）。

        以前前端自己抄了一份阈值（C1>0.4 / C4>0.7 / C6<0.3）：配置一改两处就漂，
        降级时（阈值低得多）前端更是全都亮不起来。判定收在后端一处。
        2026-10-05 起它多两个键：两条旁路（实体 / 字面）也各算"一个维度"——
        它们有票（强维 2 票），也要能高亮。
        """
        cues = {"C1": 0.9, "C2": "past", "C3": {"arousal": 1}, "C4": 0.7,
                "C5": 1, "C6": 0.1, "C7": 1,
                "entities": ["老白"], "literal": {"脱敏脚本": ["S1-0001"]}}
        hits = cue_hits(cues)
        self.assertEqual(set(hits), {"C1", "C2", "C3", "C4", "C5", "C6", "C7",
                                     "entity", "literal"})
        self.assertTrue(all(hits.values()))
        self.assertTrue(multi_hit(cues), "multi_hit 直接读这份判定")

    def test_degraded_mode_does_not_count_c6_as_a_hit(self):
        """降级时 C6 与 C1 同源（都是字符重叠）——「算不出来」不该计作命中。

        而且 C1 的线也要跟着降级走（0.03）：否则降级 = 一条都召不回来。
        """
        cues = {"C1": 0.05, "C2": "now", "C3": {"arousal": 0}, "C4": 0.0,
                "C5": 0, "C6": 0.9, "C7": 0, "_degraded": True}
        hits = cue_hits(cues)
        self.assertTrue(hits["C1"], "降级时 C1 的线低得多（0.03），0.05 算命中")
        self.assertFalse(hits["C6"], "降级时 C6 不计（算不出来 ≠ 命中）")

    def test_multi_path_hit_outranks_single_path(self):
        """四键排序第一键（§3.1，2026-10-05 由三键改来）：**被多路捞到的先给**。

        同一条场景被语义（R1）+ 实体旁路两条路捞到时票计 4（2 + 2）——这就是
        「票数同时是排序第一键」的意思（`recall.add` 按维记账：同维不重复计、
        不同维累加，不能像以前那样"已存在就只更新 level"，那个计数就丢了）。
        """
        both = self.add(text="小明面试", title="小明面试")
        link_entities(both.id, [{"name": "小明", "kind": "person", "relation": "同事"}], self.store)
        one = self.add(text="面试的事", title="面试的事")

        cues = {"_msg": "小明面试的事", "C1": 0.9, "C2": "now",
                "C3": {"arousal": 0, "valence": None}, "C4": 0.0, "C5": 0,
                "C6": 1.0, "C7": 0, "entities": ["小明"]}
        out = recall(cues, self.store)
        ids = [s.id for s in out["scenes"]]
        self.assertIn(both.id, ids)
        self.assertIn(one.id, ids)
        self.assertLess(ids.index(both.id), ids.index(one.id),
                        "被语义 + 实体两条路捞到的先给（hits=2 > 1）")

    def test_score_does_not_favor_negative(self):
        """同等条件下，负效价场景不比正效价排得更前（不偏好负面）。"""
        common = dict(intensity=0.5, cited_by_profile=2, created_at="2026-09-01 10:00:00")
        neg = Scene(id="S1-0001", valence=-1, **common)
        pos = Scene(id="S1-0002", valence=1, **common)
        self.assertAlmostEqual(core_score(neg), core_score(pos), places=9,
                               msg="valence 不该进核心度")

    def test_cited_saturates_so_new_memory_can_compete(self):
        """被引用次数要饱和映射：否则一条老场景永远霸榜，新记忆上不来。"""
        base = dict(intensity=0.0, created_at="2026-09-01 10:00:00")
        low = Scene(id="S1-0001", cited_by_profile=1, **base)
        high = Scene(id="S1-0002", cited_by_profile=100, **base)
        self.assertLess(core_score(low), core_score(high))
        self.assertLess(core_score(high), 1.0, "饱和后不该随引用次数无限增长")

    def test_recall_end_to_end_writes_trace(self):
        s = self.add(text="被组长当众批评", title="被组长批评",
                     keywords=["组长", "批评"], time_event="2026-09-01 10:00:00")
        llm = scripted_llm(cues={"tense": "past", "valence": -1, "arousal": 1,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("上次被批评那事我还挺难受", self.store, llm=llm, emb=None)

        self.assertTrue(result["actions"].get("R2") or result["actions"].get("R1"),
                        "过去时 + 情绪 + 语义命中，至少该触发一个检索动作")
        trace_files = list((self.root / "trace").glob("唤醒-*.jsonl"))
        self.assertEqual(len(trace_files), 1, "每次唤醒都该留痕（逻辑实验的命脉）")
        self.assertIn("cues", trace_files[0].read_text(encoding="utf-8"))
        _ = s

    def test_bump_counters_do_not_confuse_two_metrics(self):
        s = self.add(text="x")
        bump_counters(self.store, s.id, "mention")
        bump_counters(self.store, s.id, "mention")
        got = self.store.get_scene(s.id)
        self.assertEqual(got.mention_count, 2)
        self.assertEqual(got.cited_by_profile, 0, "提及不该污染重要性指标")
        self.assertTrue(got.last_mention_at, "提及要更新老化判据用的时间戳")

    def test_degraded_mode_still_recalls(self):
        """没有向量服务时，字符重叠的命中也要能触发 R1。

        拿向量的阈值（0.40）去卡字符重叠（量级 0.05）等于「降级时永不检索」——
        那就不是降级，是失忆。降级该变糙，但不该变哑。
        """
        self.add(text="下周三要面试，有点紧张", title="下周三面试", topic="用户·面试")
        llm = scripted_llm(cues={"tense": "past", "valence": None, "arousal": 0,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("上次那个面试的事", self.store, llm=llm, emb=None)
        self.assertGreater(result["actions"].get("R1", 0), 0, "降级状态也要能语义命中")
        self.assertEqual([s.id for s in result["scenes"]], ["S1-0001"])

    def test_degraded_mode_still_knows_it_was_talked_about(self):
        """降级时 C6（新鲜度）也要算得出来——不能因为没向量就当「从没聊过」。

        原先 C6 在无向量时恒为 1.0（= 全新），而「聊过」的判据是它低于
        `seen_threshold`——这一维在降级状态下**永远不命中**，
        「你们聊过这个」的提示也就永远不会出现。
        """
        self.add(text="下周三要面试，有点紧张", title="下周三面试", topic="用户·面试")
        llm = scripted_llm(cues={"tense": "past", "valence": None, "arousal": 0,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("上次那个面试的事", self.store, llm=llm, emb=None)
        self.assertTrue(result["flags"]["hint_talked_before"],
                        "降级也要能认出「这个聊过」")

    def test_time_backtrack_filters_irrelevant(self):
        """R2 要过弱相关这道门：不能因为「是过去」就把整个库捞出来。

        「过去」本身不含任何相关性信息——不过滤的话，库一大就是噪声注入。
        """
        self.add(text="下周三要面试", title="下周三面试")
        self.add(text="dataclass 和 pydantic 的区别", title="Python 工具")
        self.add(text="猫不吃饭", title="咪咪不吃饭")
        llm = scripted_llm(cues={"tense": "past", "valence": None, "arousal": 0,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("上次那个面试的事", self.store, llm=llm, emb=None)
        self.assertEqual([s.id for s in result["scenes"]], ["S1-0001"],
                         "R2 只该带回过相关的那条，不是整个库")

    def test_suppressed_is_recorded(self):
        """挤不进预算的要有名单——**「为什么没有它」和「为什么有它」一样重要**。

        抑制名单是 trace 的一半价值：只说"召回了这些"，等于把
        "本来该想起却没能想起的"藏起来了。
        """
        for i in range(9):
            self.add(text=f"面试相关的第 {i} 件事", title=f"面试{i}", topic="用户·面试")
        llm = scripted_llm(cues={"tense": "past", "valence": None, "arousal": 0,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("面试那事", self.store, llm=llm, emb=None)
        self.assertTrue(result["scenes"], "该召回到东西")
        self.assertTrue(result["suppressed"],
                        "超出注入预算的要有抑制名单，否则没法回答「为什么没它」")
        # 光有名单不够，还得说**差多少**：只给「核心度 0.41」等于没回答，
        # 「比最后一条被带上的低 0.06」才是那个问题的答案。
        over = [d for d in result["suppressed_detail"] if d["stage"] == "over_budget"]
        self.assertTrue(over)
        for d in over:
            self.assertGreaterEqual(d["gap"], 0, "gap = 与末位入选者的核心度差")

    def test_blocked_by_the_relevance_gate_is_recorded(self):
        """被相关性门挡在**门外的**也要留名——它们压根没进过排序。

        `suppressed_detail` 原先只收「挤不进前 N」的那批，于是 trace 答得出
        「谁被挤掉了」，答不出「谁没走到门口」——而后者才是「该想起却没想起」的大头。
        """
        self.add(text="下周三要面试", title="下周三面试")
        self.add(text="dataclass 和 pydantic 的区别", title="Python 工具")
        llm = scripted_llm(cues={"tense": "past", "valence": None, "arousal": 0,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("上次那个面试的事", self.store, llm=llm, emb=None)
        gated = [d for d in result["suppressed_detail"] if d["stage"] == "weak_gate"]
        self.assertTrue(gated, "被门挡住的候选要留在名单里")
        for d in gated:
            self.assertIn(d["action"], ("R1", "R2", "R3"), "要能看出是哪个动作在问")
            self.assertLess(d["rel"], d["line"], "记着它离门线差多少")
            self.assertAlmostEqual(d["gap"], d["line"] - d["rel"], places=3)

    def test_semantic_hits_outrank_pulled_in_ones(self):
        """跨动作先按**证据层级**分层、层内再按核心度。

        语义直接命中（层级 0）要压过「R2 顺手捞上来」的高核心度场景——
        核心度里**不含相关性**，这两个在核心度上分不出高下。
        """
        self.add(title="上次面试", text="面试的事", created_at="2026-09-01 10:00:00")
        for t in ("面试的事 上次聊过", "面试准备 上次面试", "面试结果 上次说的"):
            self.add(title=t, text=t, created_at="2026-09-01 10:00:00")
        b = self.add(title="别的事情", text="处理别的事情", created_at="2026-09-01 10:00:00",
                     intensity=1.0, cited_by_profile=20)

        llm = scripted_llm(cues={"tense": "past", "valence": None, "arousal": 0,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("上次那个面试的事", self.store, llm=llm, emb=None)

        ids = [s.id for s in result["scenes"]]
        self.assertTrue(ids)
        self.assertNotIn(b.id, ids, "顺手捞上来的不该压过语义直接命中")
        self.assertGreater(core_score(b), core_score(result["scenes"][0]),
                           "它的核心度确实更高——没进来是因为层级，不是因为核心度")
        entry = [d for d in result["suppressed_detail"] if d["id"] == b.id][0]
        self.assertEqual(entry["tier"], 2, "R2 的层级排在语义命中（层级 0）之后")

    def test_all_blocked_says_weak_gate_on_stage(self):
        """候选全被相关性门挡住时，整轮的 `stage` 要写成 `weak_gate`——
        「库里没有」和「有，但没走到门口」在 trace 上不能长得一样。"""
        self.add(title="Python 工具", text="asyncio 事件循环用法")
        llm = scripted_llm(cues={"tense": "past", "valence": None, "arousal": 0,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("上次那个面试的事", self.store, llm=llm, emb=None)

        self.assertEqual(result["scenes"], [], "一个都没进来")
        self.assertEqual(result["stage"], "weak_gate", "要说得清卡在门口")

    def test_r0_not_searching_is_recorded_too(self):
        """R0 判「不翻」也要留痕——否则它和「库里没有」在 trace 上一模一样。"""
        self.add(text="下周三要面试", title="下周三面试")
        llm = scripted_llm(cues={"tense": "past", "valence": None, "arousal": 0,
                                 "about_relation": False, "unresolved": False})
        result = recall_for_message("TCP 三次握手是什么", self.store, llm=llm, emb=None)
        self.assertEqual(result["actions"]["R0"], 0)
        self.assertFalse(result["scenes"])
        self.assertEqual(result["stage"], "R0", "整轮卡点要写在 `stage` 上")
        # 画像照旧注入（R4 是地板），动作表里也该有它——早退不等于没做
        self.assertEqual(result["actions"].get("R4"), 1.0)

    def test_weaving_never_auto_enters(self):
        """C5/C7 命中也不再有任何编织动作（2026-09-20 两套姿态退役）。

        原来这里断言「只置提示位、不自动进入编织层」；姿态退役后连提示位
        也删了（没有可引导的去处）。留下的那条约束还在：**呈现只能由人主动问起**，
        R6 从不自动执行——自动开始拆解 = 替用户决定「你现在需要被分析」。
        """
        llm = scripted_llm(cues={"tense": "now", "valence": None, "arousal": 0,
                                 "about_relation": True, "unresolved": True})
        result = recall_for_message("你觉得我是什么样的人", self.store, llm=llm, emb=None)
        self.assertNotIn("offer_weaving", result["flags"], "提示位已随姿态退役")
        self.assertNotIn("R6", result["actions"], "R6 不该被执行——呈现只能由人主动问起")

    def test_mention_count_is_capped(self):
        s = self.add(text="x")
        cap = cfgmod.cfg("rank", "mention_cap")
        for _ in range(cap + 5):
            bump_counters(self.store, s.id, "mention")
        self.assertEqual(self.store.get_scene(s.id).mention_count, cap,
                         "念叨多 ≠ 重要，要封顶（防反刍）")


class TestTrivial(unittest.TestCase):
    """纯寒暄**代码直接丢**——省一次 LLM，也省得「在吗」变成一张场景卡。

    为什么要代码判而不是问模型：这类内容一眼可判，而模型在这件事上偏保守
    （prompt 里强调「拿不准就存」之后尤其明显，实测「在吗」「嗯」被判成了值得存）。
    """

    def test_pure_greeting_is_trivial(self):
        self.assertTrue(is_trivial([{"speaker": "user", "text": "在吗"},
                                    {"speaker": "air", "text": "在"}]))

    def test_pure_reply_is_trivial(self):
        self.assertTrue(is_trivial([{"speaker": "user", "text": "嗯"}]))

    def test_short_but_meaningful_is_not_trivial(self):
        """短 ≠ 没内容。「面试没过」四个字，但正是最该记的那类。"""
        self.assertFalse(is_trivial([{"speaker": "user", "text": "面试没过"}]))

    def test_one_meaningful_line_saves_the_whole_segment(self):
        """只要有一句有内容，整段就不算寒暄。"""
        self.assertFalse(is_trivial([{"speaker": "user", "text": "嗯"},
                                     {"speaker": "user", "text": "我妈住院了"}]))

    def test_only_air_speaking_is_not_trivial(self):
        """只有 air 在说话 → 不判：她可能在承诺什么，那必须留。"""
        self.assertFalse(is_trivial([{"speaker": "air", "text": "嗯"}]))

    def test_punctuation_does_not_defeat_it(self):
        self.assertTrue(is_trivial([{"speaker": "user", "text": "嗯。"}]))


class TestWorthSaving(unittest.TestCase):
    """「值不值得存」——四条触发一到就无脑落库，会让纯寒暄也变成场景卡
    （「在吗」「嗯」「哈哈」各占一条）。

    判定和抽取**共用同一次 LLM 调用**（不额外花钱），所以这里测的是
    "判定结果有没有被正确执行"，不是"判得准不准"（那是实验的事）。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "t.db")
        self._old_paths = {k: cfgmod.PATHS[k]
                           # shortterm 也要：有些测试直接建 ChatSession（没传
                           # state_path），默认会落到真实的窗口文件——漏了它，
                           # 跑一次测试就往真窗口里塞几条消息（实锤过）
                           for k in ("trace_dir", "raws_dir", "backup_dir",
                                     "shortterm")}
        cfgmod.PATHS["trace_dir"] = str(Path(self._tmp.name) / "trace")
        cfgmod.PATHS["raws_dir"] = str(Path(self._tmp.name) / "raws")
        cfgmod.PATHS["backup_dir"] = str(Path(self._tmp.name) / "backups")
        # 窗口也要重定向（同上面那个 Base：直接建 ChatSession 的测试会写它）
        cfgmod.PATHS["shortterm"] = str(Path(self._tmp.name) / "shortterm.json")

    def tearDown(self):
        for k, v in self._old_paths.items():
            cfgmod.PATHS[k] = v
        self.store.close()
        self._tmp.cleanup()

    def _run(self, **card_over):
        llm = scripted_llm([default_card(**card_over)])
        msgs = [{"speaker": "user", "text": "在吗"},
                {"speaker": "air", "text": "在"}]
        return distill_step1(self.store, msgs, llm)

    def test_worthless_is_not_stored(self):
        scene, _, entities = self._run(worth_saving=False, skip_reason="纯寒暄")
        self.assertIsNone(scene)
        self.assertEqual(self.store.count("scenes"), 0, "判成不值得就不该落库")
        self.assertEqual(entities, [])

    def test_digest_survives_the_skip(self):
        """**跳过 ≠ 当没发生**：摘要要照留，否则她下一轮像是没聊过。"""
        _, digest, _ = self._run(worth_saving=False, skip_reason="纯寒暄")
        self.assertTrue(digest.strip(), "跳过了也要留摘要——窗口的连续性靠它")

    def test_skip_is_traced(self):
        """跳过要留痕：「没存什么、为什么」得答得出来。"""
        self._run(worth_saving=False, skip_reason="纯应答")
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob("跳过-*.jsonl"))
        self.assertTrue(files, "跳过必须留痕，否则查不出「我那句话怎么没记住」")
        self.assertIn("纯应答", files[0].read_text(encoding="utf-8"))

    def test_missing_field_means_worthy(self):
        """**字段缺失一律当"值得存"**——方向不能反：
        错存一条只是多占一点地方，漏存一条是永久丢记忆。"""
        card = default_card()
        card.pop("worth_saving", None)
        llm = scripted_llm([card])
        scene, _, _ = distill_step1(
            self.store, [{"speaker": "user", "text": "今天面试没过"}], llm)
        self.assertIsNotNone(scene, "拿不准就该存")
        self.assertEqual(self.store.count("scenes"), 1)

    def test_worthy_stores_normally(self):
        scene, _, _ = self._run(worth_saving=True, skip_reason="")
        self.assertIsNotNone(scene)
        self.assertEqual(self.store.count("scenes"), 1)


class ProfileInjectionTest(Base):
    """常驻画像的注入：**一半按印证、一半按情境**，最多 `inject_profiles_n`（10）条。

    画像总量没有上限（它是判断，不该因为"记得多"被删），
    所以要在**注入侧**收口——否则 topic 一直长，每轮就往里塞几百条。
    """

    def _profile(self, topic, statement, evidence, status="established", days_ago=0):
        at = ((datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
              if days_ago else "")
        p = Profile(topic=topic, subject="user", statement=statement, status=status,
                    evidence=evidence, sources=[],
                    # 三个时间戳一起给：`last_support_at` 是衰减看的那个
                    valid_at=at, created_at=at, last_support_at=at)
        self.store.add_profile(p)
        return p

    def test_caps_at_the_limit(self):
        for i in range(15):
            self._profile(f"话题{i}", f"陈述{i}", evidence=1)
        out = recall_for_message("随便说句话", self.store, llm=None, emb=None)
        self.assertLessEqual(len(out["profiles"]), 10)

    def test_weakest_evidence_drops_off_first(self):
        weak = self._profile("弱", "印证只有一次", evidence=1)
        strong = self._profile("强", "印证很多次", evidence=9)
        for i in range(10):
            self._profile(f"其他{i}", f"其他{i}", evidence=2)

        out = recall_for_message("随便说句话", self.store, llm=None, emb=None)
        ids = [p.id for p in out["profiles"]]

        self.assertIn(strong.id, ids)
        self.assertNotIn(weak.id, ids, "印证最少的排在最后，超了就轮不到它")

    def test_recent_support_outranks_equal_evidence(self):
        """**印证一样多时，最近还活着的那条在前**——这是时效衰减的全部含义。

        没有它，两条 evidence 都等于 3 的画像里，那条半年没人再印证、
        也没被提起的会一直占着前排（以前就是这样：只看 evidence，
        同分时按 `valid_at` 升序，老的反而在前）。
        """
        fresh = self._profile("新鲜", "上周还在被印证", evidence=3, days_ago=7)
        stale = self._profile("年久", "很久没人提了", evidence=3, days_ago=80)

        out = recall_for_message("随便说句话", self.store, llm=None, emb=None)
        ids = [p.id for p in out["profiles"]]
        self.assertEqual(ids[0], fresh.id, "同样印证数，最近被印证过的那条该在前")
        self.assertEqual(ids[1], stale.id)

    def test_missing_timestamp_is_not_penalised(self):
        """取不到时间戳 → 不打折。**缺失不是"很久以前"**，别拿猜测当证据。"""
        p = Profile(topic="无时间", subject="user", statement="s", status="established")
        self.assertEqual(profile_decay(p), 1.0)

    def test_context_half_picks_by_similarity(self):
        """另一半靠**情境**：印证最少的那条，跟这句话像也能进来。

        只有前半（按印证）的话，一条印证 1 次的画像永远排不上号——
        而「此刻正好相关的那条」才是"想起来"的样子。
        """
        class _FakeEmb:
            available = True

            def embed_one(self, text):
                return [1.0, 0.0]

            def embed(self, texts):
                return [[1.0, 0.0] if "面试" in x else [0.0, 1.0] for x in (texts or [])]

        old = cfgmod.CONFIG["recall"]["inject_profiles_n"]
        cfgmod.CONFIG["recall"]["inject_profiles_n"] = 2
        try:
            steady = self._profile("常驻", "印证最多的那条", evidence=9)
            mid = self._profile("一般", "别的事情", evidence=2)
            weak = self._profile("弱但相关", "跟面试有关的事", evidence=1)

            out = recall_for_message("随便说句话", self.store, llm=None, emb=_FakeEmb())
        finally:
            cfgmod.CONFIG["recall"]["inject_profiles_n"] = old

        ids = {p.id for p in out["profiles"]}
        self.assertEqual(ids, {steady.id, weak.id},
                         "一半按印证（steady）、一半按情境（weak）；mid 两边都不占")


if __name__ == "__main__":
    unittest.main(verbosity=2)
