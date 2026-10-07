"""阶段 3（备忘录 + 编织层）的测试。

对应阶段 3 的交付：`memo` 状态机 + R6 镜像呈现 + 前因后果模式统计。

这一层的测试盯的是**「不打扰」和「不越界」**：
  - 提过一次就不再提（不是忘了，是克制；**进注入即记账**）
  - 到点进注入一次，剩下的靠分寸（**"该不该提"的判定没有了**，2026-10-05 晚）
  - 敏感只调节力度：只记不提收窄到硬隐私（感冒 / 体检照提）
  - 镜像里不出现「他有过创伤」这类标签
  - 提 ≠ 端细节（分寸写在常备那一栏的提示词里）
"""
# 用例分组：
#   脚手架  Base · FakeEmb（假向量）
#   时间    TestClassify 分类不估天数 · TestDue 到点 = 有资格进注入（含退役）
#   提起    TestHit 命中判定（变更 / 完结）· TestDueIntoInjection 到点进注入（并栏）·
#           TestMemoCycle 后台维护
#   呈现    TestMirror 镜像 · TestPatterns 模式统计 · TestTopicMerge 只建议不合并 ·
#           TestEntityMerge 实体合并由人发起 · TestFactsTrace 档案留痕 ·
#           TestWriteLink 写入副产物
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core.llm import LLM
from core.memo import (classify_group_names, classify_window_kind, classify_window_kinds,
                       close, due, hit_candidates, judge_hits, mark_raised,
                       memo_cycle, retire_due, standing_memos)
from core.model import Memo, Profile, Scene
from core.store import Store
from core.weave import (merge_entities_confirmed, merge_suggestions,
                        merge_topics_confirmed, pattern_stats, render_mirror,
                        save_facts_confirmed)

TOPIC = "用户·面对被评价的反应"


def days_ago(n: float) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d %H:%M:%S")


def days_ahead(n: float) -> str:
    return (datetime.now() + timedelta(days=n)).strftime("%Y-%m-%d %H:%M:%S")


def memo_llm(classes=None, hit=None, group=None) -> LLM:
    """按 schema 名分发剧本。

    `hit`：命中判定的剧本（`{"items": [{"id", "action", "content", "due_at", "why"}]}`）。
    （`memo_raise` / `memo_opening` 两支随 K 条删了：那两次调用已经不存在。）
    """
    def fn(prompt, schema):
        name = schema.get("name")
        if name == "memo_class":
            return classes if classes is not None else {"items": []}
        if name == "memo_group":
            return group if group is not None else {"items": []}
        if name == "memo_hit":
            return hit if hit is not None else {"items": []}
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

    def add_memo(self, content="下周三面试", **over) -> Memo:
        fields = {"content": content, "kind": "user_task", "status": "pending",
                  "created_at": days_ago(0)}
        fields.update(over)
        m = Memo(**fields)
        self.store.add_memo(m)
        return m


# ---------------------------------------------------------------------
# 一、时间：分类 + 系统映射（不让模型估天数）
# ---------------------------------------------------------------------

class TestClassify(Base):
    def test_classification_maps_to_window(self):
        """模型只给类别，周期由系统按类别映射。"""
        m1 = self.add_memo("回头得交个东西")
        m2 = self.add_memo("在学吉他")
        llm = memo_llm(classes={"items": [
            {"content": "回头得交个东西", "class": "deadline"},
            {"content": "在学吉他", "class": "progress"},
        ]})
        n = classify_window_kinds(self.store, [m1, m2], llm)
        self.assertEqual(n, 2)

        defaults = cfgmod.cfg("memo", "window_defaults")
        self.assertEqual(self.store.conn.execute(
            "SELECT kind_class, window_days FROM memos WHERE id = ?", (m1.id,)
        ).fetchone()["window_days"], defaults["deadline"])
        self.assertEqual(self.store.conn.execute(
            "SELECT kind_class, window_days FROM memos WHERE id = ?", (m2.id,)
        ).fetchone()["window_days"], defaults["progress"])

    def test_no_classification_no_window(self):
        """分类失败 → 不设窗口（**不猜类别**：类别错了窗口就跟着错）。"""
        m = self.add_memo("某件说不清的事")
        self.assertEqual(classify_window_kinds(self.store, [m], memo_llm(classes={"items": []})), 0)
        self.assertEqual(self.store.get_memo(m.id).window_days, 0)

    def test_has_due_at_gets_timing_not_window(self):
        """有明确时间的：补时机类别，**不映射周期**（那是用户明说的，不是估的）。

        `due_at`（哪天到期）与 `timing=soon`（身体状况不等窗口）互不相干——
        前者是转述他的时间，后者是分类。
        """
        m = self.add_memo("下周三面试", due_at=days_ahead(5))
        # `timing` 现在只有 `soon` 还有用（2026-10-05 晚收窄）——用 soon 验"补上了"
        llm = memo_llm(classes={"items": [
            {"content": "下周三面试", "class": "deadline", "timing": "soon"}]})
        self.assertEqual(classify_window_kinds(self.store, [m], llm), 1)
        got = self.store.get_memo(m.id)
        self.assertEqual(got.timing, "soon")
        self.assertEqual(got.window_days, 0)

    def test_followup_maps_to_short_window(self):
        """`followup`（要过几天看结果的）**映射到短窗口**（2026-09-22）。

        他说"跑几天"，系统就得记几天——M-0012 曾按 idea 挂 60 天，
        差一个数量级，而它到点走的是闹钟（对话里一句 / 沉默时主动开口），
        不是常备栏（见 `standing_memos` 的过滤）。
        """
        m = self.add_memo("先跑几天观察")
        llm = memo_llm(classes={"items": [
            {"content": "先跑几天观察", "class": "followup", "timing": "later"}]})
        self.assertEqual(classify_window_kinds(self.store, [m], llm), 1)
        defaults = cfgmod.cfg("memo", "window_defaults")
        got = self.store.get_memo(m.id)
        self.assertEqual(got.kind_class, "followup")
        self.assertEqual(got.window_days, defaults["followup"])
        self.assertLess(defaults["followup"], defaults["progress"],
                        "它得比「进行中的事」短——不然又变成挂很久")

    def test_single_classify(self):
        llm = memo_llm(classes={"items": [{"content": "想试试新东西", "class": "idea"}]})
        self.assertEqual(classify_window_kind("想试试新东西", llm), "idea")
        self.assertEqual(classify_window_kind("x", memo_llm(classes={"items": []})), "")


# ---------------------------------------------------------------------
# 二、到期（到期 ≠ 该提）
# ---------------------------------------------------------------------

class TestDue(Base):
    def test_explicit_due_at(self):
        past = self.add_memo("面试", due_at=days_ago(1))
        future = self.add_memo("体检", due_at=days_ahead(3))
        ids = [m.id for m in due(self.store)]
        self.assertIn(past.id, ids)
        self.assertNotIn(future.id, ids)

    def test_window_based_due(self):
        """无具体时间 → `created_at + window_days` 才算到期。"""
        old = self.add_memo("在学吉他", kind_class="progress",
                            window_days=30, created_at=days_ago(31))
        fresh = self.add_memo("刚起意的事", kind_class="idea",
                              window_days=60, created_at=days_ago(1))
        ids = [m.id for m in due(self.store)]
        self.assertIn(old.id, ids)
        self.assertNotIn(fresh.id, ids)

    def test_hard_private_never_due_health_still_due(self):
        """档 2（硬隐私）只记不提；健康类照常到期。

        敏感不是"闭嘴"的同义词（2026-09-15）：源头（场景卡）宽、出口再一刀切，
        感冒、体检这类最该被问候的事会全被锁死。
        """
        self.add_memo("硬隐私的事", due_at=days_ago(5), sensitive=2)
        self.assertEqual(due(self.store), [])
        health = self.add_memo("体检结果", due_at=days_ago(5), sensitive=1)
        self.assertIn(health.id, [m.id for m in due(self.store)])

    def test_raise_mode_only_hard_private_locks(self):
        """提档只有两种产出：0 正常提 / 2 只记不提。

        判定**不看场景卡的 `sensitive`**（它"拿不准给 true"，健康 / 家事都标 1），
        只认硬隐私窄名单——否则"感冒好了没"这种问候会被误锁。
        """
        from core.distill import _raise_mode
        self.assertEqual(_raise_mode("下周三面试"), 0)
        self.assertEqual(_raise_mode("感冒好点没"), 0)
        self.assertEqual(_raise_mode("那件创伤的事"), 2)

    def test_raised_is_not_due_again(self):
        """提过一次就不再主动提（不是忘了，是克制）。"""
        m = self.add_memo("面试", due_at=days_ago(1))
        self.assertEqual(len(due(self.store)), 1)
        mark_raised(self.store, m.id)
        self.assertEqual(due(self.store), [])
        self.assertEqual(self.store.get_memo(m.id).status, "raised")

    def test_raised_is_still_open(self):
        """已提过的仍算「未了结」——用户可能过几天回来说结果。"""
        m = self.add_memo("面试", due_at=days_ago(1))
        mark_raised(self.store, m.id)
        self.assertIn(m.id, [x.id for x in self.store.open_memos()])

    def test_retire_closes_stale_memos(self):
        """**提的窗口**过完仍未提 → 退役（closed）。

        2026-10-05 口径：判的是"**提的窗口**过没过"（起点 `due_at` 或
        `created + window_days`，+ grace），不是"挂了多久"（原来「窗口 × 3 / 硬顶 90 天」）。
        """
        grace = int(cfgmod.cfg("memo", "retire_grace_days") or 2)
        m = self.add_memo("很久以前的事", kind_class="deadline", window_days=7,
                          created_at=days_ago(7 + grace + 1))
        retired = retire_due(self.store)
        self.assertIn(m.id, [r["id"] for r in retired])
        self.assertEqual(self.store.get_memo(m.id).status, "closed")

    def test_retire_waits_for_the_grace(self):
        """窗口内（起点 + grace 之内）不退役——她还有机会提。"""
        grace = int(cfgmod.cfg("memo", "retire_grace_days") or 2)
        m = self.add_memo("刚到期的事", kind_class="deadline", window_days=7,
                          created_at=days_ago(7 + max(grace - 1, 0)))
        self.assertEqual(retire_due(self.store), [])
        self.assertEqual(self.store.get_memo(m.id).status, "pending")

    def test_retire_raised_without_reply(self):
        """**超期未回**：提过之后一直没回音 → 也退役（2026-10-05 加）。"""
        days = int(cfgmod.cfg("memo", "retire_after_raised_days") or 3)
        m = self.add_memo("提过没人回的事", due_at=days_ago(30), status="raised",
                          raised_at=days_ago(days + 1))
        self.assertIn(m.id, [r["id"] for r in retire_due(self.store)])

    def test_raised_within_days_keeps_hanging(self):
        """刚提过的不退役——他可能过两天回来说结果。"""
        m = self.add_memo("刚提过的事", due_at=days_ago(30), status="raised",
                          raised_at=days_ago(1))
        self.assertEqual(retire_due(self.store), [])
        self.assertIn(m.id, [x.id for x in self.store.open_memos()])

    def test_retired_loop_is_not_rendered_as_pending(self):
        """退役过的钩子不再渲染「→ 未定」（2026-10-05）。

        治的是实测那个病：M-0013 的钩子在婚礼办完之后一直没标，
        S1-0013 每次被想起都带「→ 未定：10月2日妹妹结婚」。
        退役**不回流**——`closed_at` 保持空着，标的是 `retired_at`。
        """
        from core.prompts import scene_line
        s = Scene(title="婚礼", text="摘要",
                  open_loops=[{"content": "10月2日妹妹结婚", "kind": "user_task",
                               "due_at": ""}])
        self.store.add_scene(s)
        lid = self.store.get_scene(s.id).open_loops[0]["loop_id"]
        self.assertTrue(lid, "写入时该给钩子编上稳定号（loop_id，代码生成）")
        m = Memo(content="10月2日妹妹结婚", kind="user_task", scene_id=s.id,
                 loop_id=lid, created_at=days_ago(40))
        self.store.add_memo(m)

        self.assertIn("未定", scene_line(self.store.get_scene(s.id)))
        self.assertIn(m.id, [r["id"] for r in retire_due(self.store)])
        self.assertNotIn("未定", scene_line(self.store.get_scene(s.id)),
                         "退役过的钩子不再显示未定")
        loop = self.store.get_scene(s.id).open_loops[0]
        self.assertTrue(loop.get("retired_at"), "退役要留痕在钩子上")
        self.assertFalse(loop.get("closed_at"), "退役 ≠ 完成——不许写成闭合")

    def test_close_removes_from_open(self):
        m = self.add_memo("面试")
        close(self.store, m.id)
        self.assertNotIn(m.id, [x.id for x in self.store.open_memos()])


# ---------------------------------------------------------------------
# 三、命中判定（用户这一句碰到了哪几件事：变更 / 完结）
# ---------------------------------------------------------------------

class TestHit(Base):
    def test_closes_when_model_says_so(self):
        m = self.add_memo("下周三面试")
        llm = memo_llm(hit={"items": [{"id": m.id, "action": "close", "content": "",
                                       "due_at": "", "why": "他说面试过了"}]})
        out = judge_hits(self.store, "面试过了，还挺顺利", llm)
        self.assertEqual(out["closed"], m.id)
        self.assertEqual(self.store.get_memo(m.id).status, "closed")

    def test_no_open_memos_no_llm_call(self):
        """没有未关闭的事时直接短路（绝大多数轮次都走这条）。"""
        llm = memo_llm(hit={"items": [{"id": "M-9999", "action": "close",
                                       "content": "", "due_at": "", "why": "x"}]})
        self.assertEqual(judge_hits(self.store, "面试过了", llm)["closed"], "")

    def test_plain_talk_still_goes_to_the_model(self):
        """**词表退场**（2026-10-05）：日常话也进判定——判据是内容，不是字面。

        老口径（`_CLOSURE_MARKS` 预筛）正是死在这：他 00:32 说「一个结婚了」、
        00:33 说「才从老家回来」，两句都是结果，一个词都没命中、连 LLM 都没调。
        现在照调，模型说 `none` 就不动作——多花一次小调用，换掉一整类静默漏判。
        """
        self.add_memo("下周三面试")
        seen = {}

        def fn(prompt, schema):
            if schema.get("name") == "memo_hit":
                seen["called"] = True
                return {"items": [{"id": "M-0001", "action": "none", "content": "",
                                   "due_at": "", "why": "只是闲聊"}]}
            return {}

        llm = LLM()
        llm.set_mock(fn)
        out = judge_hits(self.store, "今天天气不错", llm)
        self.assertTrue(seen.get("called"), "日常话也要判")
        self.assertEqual(out["closed"], "")

    def test_unknown_id_is_ignored(self):
        """模型给了一个不存在的 id → 当没发生（误关的代价比多提一次大）。"""
        m = self.add_memo("下周三面试")
        llm = memo_llm(hit={"items": [{"id": "M-9999", "action": "close", "content": "",
                                       "due_at": "", "why": "x"}]})
        self.assertEqual(judge_hits(self.store, "面试过了", llm)["closed"], "")
        self.assertEqual(self.store.get_memo(m.id).status, "pending")

    def test_mention_is_not_closure(self):
        """提到 ≠ 了结：模型给 `none` 就不关。"""
        m = self.add_memo("下周三面试")
        llm = memo_llm(hit={"items": [{"id": m.id, "action": "none", "content": "",
                                       "due_at": "", "why": "还在准备"}]})
        self.assertEqual(judge_hits(self.store, "面试的事我还在准备", llm)["closed"], "")
        self.assertEqual(self.store.get_memo(m.id).status, "pending")

    def test_update_changes_content_and_reopens(self):
        """**变更**（2026-10-05）：他改了口径 → 内容换掉、回待提、留痕记「旧 → 新」。"""
        m = self.add_memo("月底回老家", status="raised", raised_at=days_ago(1))
        llm = memo_llm(hit={"items": [{"id": m.id, "action": "update",
                                       "content": "下个月才回老家",
                                       "due_at": "", "why": "他改了口径"}]})
        out = judge_hits(self.store, "那个不回了，下个月才回", llm)
        self.assertEqual(out["updated"], [m.id])
        cur = self.store.get_memo(m.id)
        self.assertEqual(cur.content, "下个月才回老家")
        self.assertEqual(cur.status, "pending", "内容变了 = 有新的事实要提醒")
        self.assertEqual(cur.raised_at, "")
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob("备忘-*.jsonl"))
        self.assertTrue(files, "变更要留痕（改了什么都答得出来）")
        self.assertIn("变更", files[0].read_text(encoding="utf-8"))

    def test_update_takes_due_at_only_when_stated(self):
        """明说了新时间才写 `due_at`（铁律：模型只分类、不估天数）。"""
        m = self.add_memo("面试")
        llm = memo_llm(hit={"items": [{"id": m.id, "action": "update",
                                       "content": "面试改到下周三",
                                       "due_at": days_ahead(7), "why": "他明说了"}]})
        judge_hits(self.store, "面试改到下周三", llm)
        self.assertEqual(self.store.get_memo(m.id).due_at[:10], days_ahead(7)[:10])

    def test_same_content_is_not_an_update(self):
        """新内容跟原来一模一样 → 不算变更（防"每次命中都续期"）。"""
        m = self.add_memo("面试")
        llm = memo_llm(hit={"items": [{"id": m.id, "action": "update",
                                       "content": "面试", "due_at": "", "why": "没变化"}]})
        self.assertEqual(judge_hits(self.store, "面试", llm)["updated"], [])

    def test_update_remaps_the_window(self):
        """没明说新时间 → 窗口按新内容重新映射（模型只判类别，天数由系统定）。"""
        m = self.add_memo("面试")
        llm = memo_llm(classes={"items": [{"content": "面试改到下周三",
                                           "class": "deadline", "timing": "later"}]},
                       hit={"items": [{"id": m.id, "action": "update",
                                       "content": "面试改到下周三",
                                       "due_at": "", "why": "他改了口径"}]})
        judge_hits(self.store, "面试改到下周三", llm)
        cur = self.store.get_memo(m.id)
        self.assertEqual(cur.kind_class, "deadline")
        self.assertEqual(cur.window_days,
                         cfgmod.cfg("memo", "window_defaults")["deadline"])

    def test_recently_raised_always_gets_judged(self):
        """**刚提过的必进判定**（2026-10-05 的断点）：哪怕它与这句毫无字面关系——
        "她刚提、他随口答"是最常见的闭合情形；条数多到不再全量时也要进。"""
        full_n = int(cfgmod.cfg("memo", "hit_full_n") or 20)
        for i in range(full_n + 1):
            self.add_memo(f"杂事{i}", created_at=days_ago(30))
        fresh = self.add_memo("婚礼那场", status="raised", raised_at=days_ago(0))
        cands = hit_candidates(self.store, "嗯", emb=None)
        self.assertIn(fresh.id, [m.id for m in cands])


# ---------------------------------------------------------------------
# 四、提起的时机（明确给放弃权）
# ---------------------------------------------------------------------

class TestGroupName(Base):
    """事项组（2026-09-22）：同一件事的多步共用一个短名——
    **只用于呈现与提醒收拢**，不合并、不改写 content。"""

    def test_fills_empty_group_names(self):
        """空的补上：同一件事的多步拿同一个名字，独立的事留空。"""
        a = self.add_memo("在新 API 上设置搜索")
        b = self.add_memo("设置好后跑一次验收")
        c = self.add_memo("妹妹结婚", due_at=days_ahead(9))
        llm = memo_llm(group={"items": [
            {"content": "在新 API 上设置搜索", "group_name": "搜索功能"},
            {"content": "设置好后跑一次验收", "group_name": "搜索功能"},
            {"content": "妹妹结婚", "group_name": ""},
        ]})
        self.assertEqual(classify_group_names(self.store, [a, b, c], llm), 2)
        self.assertEqual(self.store.get_memo(a.id).group_name, "搜索功能")
        self.assertEqual(self.store.get_memo(b.id).group_name, "搜索功能")
        self.assertEqual(self.store.get_memo(c.id).group_name, "",
                         "独立的事留空（宁可分成两件，也不硬塞一组）")

    def test_existing_names_are_not_rewritten(self):
        """已有的组名**一个字都不动**——只填空的（改它会让同一件事裂成两个名字）。"""
        m = self.add_memo("写脚本", group_name="开源准备")
        llm = memo_llm(group={"items": [{"content": "写脚本", "group_name": "别的名字"}]})
        self.assertEqual(classify_group_names(self.store, [m], llm), 0)
        self.assertEqual(self.store.get_memo(m.id).group_name, "开源准备")

    def test_no_targets_no_llm_call(self):
        """全都有组名 → 不调模型：会话边界不该为零收益花钱。"""
        m = self.add_memo("写脚本", group_name="开源准备")
        calls = []

        def fn(prompt, schema):
            calls.append(schema.get("name"))
            return {"items": []}

        llm = LLM()
        llm.set_mock(fn)
        self.assertEqual(classify_group_names(self.store, [m], llm), 0)
        self.assertEqual(calls, [], "一个调用都不该发生")


class TestDueIntoInjection(Base):
    """到点进注入（2026-10-05 晚并栏）：候选 / 时机表 / 主动开口都删了，
    到点那条现在落在 `standing_memos` 上——**进即记账**由 `due` 标记交给 chat。

    这几条就是设计稿那句"到点进注入一次，提过就停"的断言。
    """

    def test_due_one_gets_into_injection_with_flag(self):
        m = self.add_memo(due_at=days_ago(1))           # 昨天就该提了
        picked = standing_memos(self.store, "在吗")
        due_items = [x for x in picked if x.get("due")]
        self.assertEqual([x["id"] for x in due_items], [m.id])
        self.assertEqual(due_items[0]["why"], "到点")

    def test_not_due_never_carries_the_flag(self):
        """没到点的只是"手上得有"，不带 `due`——它不该被记账。"""
        self.add_memo(due_at=days_ahead(3))
        picked = standing_memos(self.store, "在吗")
        self.assertTrue(picked, "刚交代的事该在手上")
        self.assertFalse(any(x.get("due") for x in picked))

    def test_raised_is_not_due_again(self):
        """提过一次就不再进（`raised` 不进 `opens`）——「提一次就不再注入」那条纪律。"""
        m = self.add_memo(due_at=days_ago(1))
        mark_raised(self.store, m.id)
        picked = standing_memos(self.store, "在吗")
        self.assertFalse(any(x.get("due") for x in picked))

    def test_group_raises_once_not_per_step(self):
        """同一件事的多步：组里任何一条提过，其余步骤不再进（别分两轮催）。"""
        a = self.add_memo(content="设置搜索", due_at=days_ago(1), group_name="搜索功能")
        self.add_memo(content="跑验收", due_at=days_ago(1), group_name="搜索功能")
        mark_raised(self.store, a.id)
        picked = standing_memos(self.store, "在吗")
        self.assertFalse(any(x.get("due") for x in picked))

    def test_one_due_at_a_time(self):
        """一次一件：两件都到点时，注入里只有一件带 `due`（其余下一轮接着排）。"""
        self.add_memo(content="第一件", due_at=days_ago(2))
        self.add_memo(content="第二件", due_at=days_ago(2))
        due_items = [x for x in standing_memos(self.store, "在吗") if x.get("due")]
        self.assertEqual(len(due_items), 1)

    def test_followup_no_longer_special(self):
        """`followup` 不再单独排除（2026-10-05 晚）——它跟别的 memo 走同一条。"""
        m = self.add_memo(content="先跑几天看看", kind_class="followup",
                          window_days=4, due_at=days_ago(0.1))
        due_items = [x for x in standing_memos(self.store, "在吗") if x.get("due")]
        self.assertEqual([x["id"] for x in due_items], [m.id])

    def test_hard_private_never_injected(self):
        """硬隐私（`sensitive=2`）根本不进那一栏——比"给门"再保守一档。"""
        self.add_memo(content="那件家事", due_at=days_ago(1), sensitive=2)
        self.assertFalse(any(x.get("due") for x in standing_memos(self.store, "在吗")))


class TestMemoCycle(Base):
    def test_cycle_classifies_and_retires(self):
        m1 = self.add_memo("回头交个东西")
        grace = int(cfgmod.cfg("memo", "retire_grace_days") or 2)
        self.add_memo("陈年旧事", kind_class="deadline", window_days=7,
                      created_at=days_ago(7 + grace + 1))

        stats = memo_cycle(self.store, memo_llm(
            classes={"items": [{"content": "回头交个东西", "class": "deadline"}]},
            group={"items": [{"content": "回头交个东西", "group_name": "交东西"}]}))
        self.assertEqual(stats["classified"], 1)
        self.assertEqual(stats["grouped"], 1, "空组名在会话边界补判（2026-09-22）")
        self.assertEqual(stats["retired"], 1)
        self.assertEqual(self.store.get_memo(m1.id).kind_class, "deadline")
        self.assertEqual(self.store.get_memo(m1.id).group_name, "交东西")
        # 退役留痕（2026-10-05 补）：它是四个关闭发起方里唯一原来没留痕的
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob("备忘-*.jsonl"))
        self.assertTrue(files, "退役也要留痕——「为什么到死没被提过」要答得出来")
        self.assertIn("超期退役", files[0].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
# 六、R6 镜像呈现
# ---------------------------------------------------------------------

class TestMirror(Base):
    def _make(self, status="established", sensitive=0, statement="遇到被评价会先退出"):
        # 两条场景的敏感标记要保持一致：只要有一条不是敏感，它就该照常展示——
        # 「敏感不进镜像」是逐条的，不是「这条画像沾了敏感就整条遮掉」。
        s1 = Scene(topic=TOPIC, trigger_class="被评价", subject="user",
                   title="被批评", text="摘要", time_event="2026-08-10 09:00:00",
                   sensitive=sensitive)
        s2 = Scene(topic=TOPIC, trigger_class="被评价", subject="user",
                   title="又被批评", text="摘要", time_event="2026-08-20 09:00:00",
                   sensitive=sensitive)
        self.store.add_scene(s1)
        self.store.add_scene(s2)
        from core.model import Profile
        p = Profile(topic=TOPIC, subject="user", statement=statement, status=status,
                    evidence=2, sources=[s1.id, s2.id])
        self.store.add_profile(p)
        return p, s1, s2

    def test_pending_is_labeled_as_guess(self):
        """`pending` 必须标「还没印证够，只是猜测」——否则等于把猜测说成事实。"""
        self._make(status="pending")
        mirror = render_mirror(self.store)
        self.assertEqual(len(mirror["profiles"]), 1)
        p = mirror["profiles"][0]
        self.assertEqual(p["status"], "pending")
        self.assertIn("猜测", p["status_label"])

    def test_sensitive_sources_not_shown(self):
        """敏感内容不进镜像：只留计数，不展示具体场景。"""
        self._make(sensitive=1)
        mirror = render_mirror(self.store)
        p = mirror["profiles"][0]
        self.assertEqual(p["sources"], [])
        self.assertGreater(p["hidden_sensitive"], 0,
                           "要有计数，不能假装没有依据")

    def test_include_pending_flag(self):
        self._make(status="pending")
        self.assertEqual(len(render_mirror(self.store)["profiles"]), 1)
        self.assertEqual(len(render_mirror(self.store, include_pending=False)["profiles"]), 0)

    def test_carries_reject_entry(self):
        """呈现必须附否决入口——呈现与可否决是同一件事的两面。"""
        self._make()
        mirror = render_mirror(self.store)
        self.assertTrue(mirror["can_reject"])
        self.assertIn("不对", mirror["reject_hint"])

    def test_sources_are_traceable(self):
        p, s1, s2 = self._make()
        mirror = render_mirror(self.store)
        ids = [x["id"] for x in mirror["profiles"][0]["sources"]]
        self.assertEqual(ids, [s1.id, s2.id], "呈现要能下钻到支撑它的具体场景")

    def test_sources_carry_role(self):
        """依据要区分「出处」和「印证」——人才看得出这条判断是怎么来的。

        出处由画像自带的 `evidence_pack` 认出来（包里只装出处那几条）。
        """
        p, s1, s2 = self._make()
        pack = [{"id": s1.id, "title": s1.title, "summary": s1.text, "time": ""}]
        self.store.set_profile_sources(p.id, [s1.id, s2.id], pack=pack)

        mirror = render_mirror(self.store)
        roles = {x["id"]: x["role"] for x in mirror["profiles"][0]["sources"]}
        self.assertEqual(roles[s1.id], "forming")
        self.assertEqual(roles[s2.id], "supporting")

    def test_deleted_evidence_is_not_shown(self):
        """依据里已删的场景**不显示**（引用 = `sources ∩ 现存节点`，2026-09-24）。

        原行为是"退回追溯包的快照"（`from_pack`）——那是"永不删"时代的兜底。
        真删落地后那条路取消了：**他删掉的东西不该从快照里再露出来**。
        （`evidence_pack` 仍在库里——它是"当时凭什么"的记录，但只用来认
        「出处 / 印证」，不拿它把删掉的场景端出来。）
        """
        p, s1, s2 = self._make()
        pack = [{"id": "S1-9999", "title": "已不在库里的场景",
                 "summary": "快照摘要", "time": "2026-01-01"}]
        self.store.set_profile_sources(p.id, ["S1-9999"], pack=pack)

        mirror = render_mirror(self.store)
        self.assertEqual(mirror["profiles"][0]["sources"], [],
                         "不在库里的编号一条都不列（不兜快照）")

    def test_topic_filter(self):
        self._make()
        self.assertEqual(len(render_mirror(self.store, topic=TOPIC)["profiles"]), 1)
        self.assertEqual(len(render_mirror(self.store, topic="别的主题")["profiles"]), 0)


# ---------------------------------------------------------------------
# 七、前因后果模式统计
# ---------------------------------------------------------------------

class TestPatterns(Base):
    def _scene(self, trigger_class="被评价", title="被批评", reaction="退出",
               sensitive=0, entity="组长"):
        s = Scene(topic=TOPIC, subject="user", title=title, trigger="被说了一顿",
                  trigger_class=trigger_class, reaction=reaction, sensitive=sensitive)
        self.store.add_scene(s)
        if entity:
            eid = self.store.add_entity(entity, "person")
            self.store.link_scene_entity(s.id, eid)
        return s

    def test_counts_by_class_and_entity(self):
        """「面对【上级】的【被评价】→ 典型反应」是两个字段组合出来的精度。"""
        for i in range(3):
            self._scene(title=f"被批评{i}", reaction="先退出")
        self._scene(trigger_class="失去", title="丢了东西", entity="")

        stats = pattern_stats(self.store, min_count=2)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["trigger_class"], "被评价")
        self.assertEqual(stats[0]["entity"], "组长")
        self.assertEqual(stats[0]["count"], 3)
        self.assertIn("先退出", stats[0]["reactions"])

    def test_below_threshold_is_hidden(self):
        """单次不成模式（一次观察不下结论，同「不武断」）。"""
        self._scene()
        self.assertEqual(pattern_stats(self.store, min_count=2), [])

    def test_sensitive_scenes_excluded(self):
        """敏感场景不进统计——否则等于把他的创伤总结成一条规律呈现给他。"""
        for i in range(3):
            self._scene(title=f"敏感{i}", sensitive=1)
        self.assertEqual(pattern_stats(self.store, min_count=2), [])

    def test_scenes_without_class_ignored(self):
        """没有情境类型的场景不参与（trigger_class 是统计的索引）。"""
        for i in range(3):
            self._scene(trigger_class="", title=f"随便{i}")
        self.assertEqual(pattern_stats(self.store, min_count=2), [])


# ---------------------------------------------------------------------
# 八、写入链路接入：open_loops → 分类
# ---------------------------------------------------------------------

class FakeEmb:
    """假向量服务：按调用顺序返回预设向量。

    只测「比对逻辑对不对」，不测语义——真语义要靠真模型，那是实验的事。
    """

    def __init__(self, vecs):
        self.vecs = vecs
        self.available = True

    def embed(self, texts):
        return self.vecs[:len(texts)]


class TestTopicMerge(Base):
    """主题归并：**只建议、不自动合**（2026-09-12 接上）。

    topic 是 LLM 自由生成的字符串，裂开是常态；裂了不补，聚合与画像
    就因为「每组都不够 3 条」而永远立不起来——**而且一声不响**。
    `merge_topics()` 早就写好了，但一直没有任何地方调用它：接口等于没有。
    """

    def _scene(self, topic: str, title: str = "") -> Scene:
        s = Scene(topic=topic, title=title or f"{topic}的一件事", subject="user")
        self.store.add_scene(s)
        return s

    def test_suggests_similar_wording(self):
        """措辞相近的两个主题要被认出来——这就是裂缝的样子。"""
        self._scene("用户·被当众批评的反应")
        self._scene("用户·被批评的反应")
        emb = FakeEmb([[1.0, 0.0], [1.0, 0.02]])       # 两条名称向量几乎同向
        sug = merge_suggestions(self.store, emb=emb, threshold=0.9)
        self.assertEqual(len(sug), 1)
        self.assertIn("措辞相近", sug[0]["basis"])
        self.assertIn(sug[0]["from"], ("用户·被当众批评的反应", "用户·被批评的反应"))

    def test_unrelated_topics_are_not_suggested(self):
        """不相关的主题不该冒出来——误报会让人不敢点这个按钮。"""
        self._scene("用户·被批评的反应")
        self._scene("猫·健康问题")
        emb = FakeEmb([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(merge_suggestions(self.store, emb=emb, threshold=0.9), [])

    def test_no_embedding_means_no_guess(self):
        """没有向量就**不猜**：宁可漏报，不要瞎报。"""
        self._scene("用户·被批评的反应")
        self._scene("用户·被当众批评的反应")
        self.assertEqual(merge_suggestions(self.store, emb=None), [])

    def test_basis_prefers_the_stronger_path(self):
        """两条路径取较大者，并如实标出是哪一条。"""
        self._scene("A")
        self._scene("B")
        # 名称向量正交（不像），但场景均值向量同向（内容像）
        a = self.store.query_scenes(topic="A")[0]
        b = self.store.query_scenes(topic="B")[0]
        blob = self.store._vec_to_blob([1.0, 0.0])
        self.store.conn.execute("UPDATE scenes SET emb = ? WHERE id = ?", (blob, a.id))
        self.store.conn.execute("UPDATE scenes SET emb = ? WHERE id = ?", (blob, b.id))
        self.store.conn.commit()
        emb = FakeEmb([[1.0, 0.0], [0.0, 1.0]])
        sug = merge_suggestions(self.store, emb=emb, threshold=0.9)
        self.assertEqual(len(sug), 1)
        self.assertIn("内容相近", sug[0]["basis"])

    def test_merge_moves_all_three_tables(self):
        """合并要改 profiles / scenes / summaries 的归属，且**不删任何记录**。"""
        s = self._scene("用户·被批评的反应")
        p = Profile(topic="用户·被批评的反应", subject="user", statement="x",
                    status="pending")
        self.store.add_profile(p)
        n = self.store.merge_topics("用户·被批评的反应", "用户·被当众批评的反应")
        self.assertGreaterEqual(n, 2)
        self.assertEqual(self.store.get_scene(s.id).topic, "用户·被当众批评的反应")
        self.assertEqual(self.store.get_profile(p.id).topic, "用户·被当众批评的反应")
        self.assertIsNotNone(self.store.get_scene(s.id), "只改归属，不删记录")

    def test_confirmed_merge_is_traced(self):
        """合并必须留痕——它是**不可逆的语义断言**，事后要能复盘。"""
        self._scene("A")
        out = merge_topics_confirmed(self.store, "A", "B")
        self.assertTrue(out["ok"])
        self.assertEqual(out["merged"], 1)
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob("合并-*.jsonl"))
        self.assertTrue(files, "合并要留痕（同否决 trace 的理由）")

    def test_refuses_degenerate_input(self):
        self.assertFalse(merge_topics_confirmed(self.store, "A", "A")["ok"])
        self.assertFalse(merge_topics_confirmed(self.store, "", "B")["ok"])
        self.assertFalse(merge_topics_confirmed(self.store, "A", "")["ok"])


class TestEntityMerge(Base):
    """实体合并：**写好了没接线，接口等于没有**（2026-09-17 接上）。

    `add_entity(aliases=…)` 那个参数从第一版就在（注释还写着"留这个口子给它人工补"），
    但没有任何调用点、界面上也没有入口——两个「小明」合不起来。
    同 `merge_topics()` 的旧课：口子要能**够着**才算口子。
    这里钉住三件事：场景改挂、旧名字变别名、确认那一步留痕。
    """

    def _scene(self, title: str = "") -> Scene:
        s = Scene(topic="用户·和同事的事", title=title or "一件事", subject="user")
        self.store.add_scene(s)
        return s

    def test_moves_scenes_and_keeps_old_name_as_alias(self):
        """合并后：场景改挂到目标、旧名字变成别名、源实体不在了。"""
        s = self._scene()
        e1 = self.store.add_entity("小明", "person")
        e2 = self.store.add_entity("小明明", "person")
        self.store.link_scene_entity(s.id, e1)

        out = self.store.merge_entities(e1, e2)

        self.assertEqual(out["scenes"], 1)
        self.assertEqual(out["aliases"], 1, "「小明」要被并成别名，否则以后提到他名字命不中")
        self.assertIsNone(self.store.get_entity(e1))
        self.assertEqual(self.store.entities_of_scene(s.id), ["小明明"])
        self.assertIn("小明", self.store.get_entity(e2).aliases)

    def test_both_linked_to_same_scene_does_not_collide(self):
        """两个实体都挂同一条场景时，改归属会撞主键——那条只留一份。"""
        s = self._scene()
        e1 = self.store.add_entity("小明", "person")
        e2 = self.store.add_entity("小明（同事）", "person")
        self.store.link_scene_entity(s.id, e1)
        self.store.link_scene_entity(s.id, e2)

        out = self.store.merge_entities(e1, e2)

        self.assertEqual(out["scenes"], 0, "那条场景已经在目标名下了")
        self.assertEqual(self.store.entities_of_scene(s.id), ["小明（同事）"])

    def test_confirmed_merge_is_traced(self):
        """人确认的那一步要留痕——否则事后答不出「谁把这两个名字并了」。"""
        e1 = self.store.add_entity("小明", "person")
        e2 = self.store.add_entity("小明明", "person")
        out = merge_entities_confirmed(self.store, e1, e2)
        self.assertTrue(out["ok"])
        self.assertEqual(out["name"], "小明明")
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob("实体合并-*.jsonl"))
        self.assertTrue(files, "合并要留痕（同主题合并的理由）")

    def test_refuses_degenerate_input(self):
        e = self.store.add_entity("小明", "person")
        self.assertFalse(merge_entities_confirmed(self.store, e, e)["ok"])
        self.assertFalse(merge_entities_confirmed(self.store, "E-9999", e)["ok"])


class TestEntityActivity(Base):
    """实体页的「活跃」标记（2026-09-25）：**派生自最近交互，不落库、不影响行为**。

    判据是「关联场景里最新一条的时间」落在 `entity.active_days` 窗口内。
    它是给人看的一眼（标记 + 排序）——**不参与准入、不触发删除**：
    沉寂的名字恰恰是长程记忆该留着的（半年没提的人被提起时，索引必须在）。
    """

    def _app(self):
        from core.dashboard import App
        app = App.__new__(App)          # 不跑 __init__（那会建 LLM / emb）
        app.store = self.store
        return app

    def _link(self, name: str, when: str) -> str:
        s = Scene(topic="用户·和同事的事", title=name, time_record=when)
        self.store.add_scene(s)
        eid = self.store.add_entity(name, "person")
        self.store.link_scene_entity(s.id, eid, "同事")
        return eid

    def test_active_flag_and_ordering(self):
        self._link("小明", days_ago(90))
        self._link("小红", days_ago(3))

        rows = self._app().entities()

        by = {r["name"]: r for r in rows}
        self.assertFalse(by["小明"]["active"], "90 天没交互 = 不活跃")
        self.assertTrue(by["小红"]["active"], "3 天前交互过 = 活跃")
        self.assertEqual(rows[0]["name"], "小红", "最近交互的在最前")
        self.assertEqual(by["小红"]["relations"], ["同事"])

    def test_missing_time_is_inactive_not_crash(self):
        """场景缺时间（老库脏行）不给假活跃——空串排在最后、标记 False。"""
        s = Scene(topic="用户·和同事的事", title="无时间")
        self.store.add_scene(s)
        self.store.conn.execute("UPDATE scenes SET time_record = '' WHERE id = ?", (s.id,))
        self.store.conn.commit()
        eid = self.store.add_entity("无记录", "person")
        self.store.link_scene_entity(s.id, eid, "同事")

        rows = self._app().entities()
        row = [r for r in rows if r["name"] == "无记录"][0]
        self.assertFalse(row["active"])
        self.assertEqual(rows[-1]["name"], "无记录", "空时间排最后")


class TestFactsTrace(Base):
    """档案改动要留痕——它决定她**怎么称呼他**，是最不该说不清的一类改动。

    档案和主题合并同属「人的纠正」：都是他自己说了算的东西，所以都留痕。
    （审计时发现这里漏了：合并有 trace，档案没有。）
    """

    def _traces(self) -> str:
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob("档案-*.jsonl"))
        return "\n".join(f.read_text(encoding="utf-8") for f in files)

    def test_change_is_traced(self):
        save_facts_confirmed(self.store, {"称呼": "老张"})
        text = self._traces()
        self.assertIn("老张", text)
        self.assertIn("新增", text)

    def test_unchanged_is_not_recorded(self):
        """没变的不记账——否则每次点保存都刷一堆空改动，留痕就没法看了。"""
        save_facts_confirmed(self.store, {"称呼": "老张"})
        save_facts_confirmed(self.store, {"称呼": "老张"})
        self.assertEqual(self._traces().count("称呼"), 1)

    def test_clearing_is_recorded_as_clearing(self):
        """清空一格 = 让她忘掉，这件事必须记下来（不是"改成空字符串"）。"""
        save_facts_confirmed(self.store, {"年龄": "32"})
        save_facts_confirmed(self.store, {"年龄": ""})
        self.assertIn("清空", self._traces())
        self.assertEqual(self.store.all_user_facts(), [], "清空后档案里不该留这条")


class TestWriteLink(Base):
    def test_step1_classifies_classless_memos(self):
        """没有 due_at 的 open_loop 在写入时就被分类（否则它永远不到期）。"""
        from core.distill import distill_step1
        card = {
            "title": "面试", "keywords": [], "summary": "s", "window_digest": "d",
            "valence": None, "arousal": None, "trigger": "", "trigger_class": "",
            "reaction": "", "outcome": "", "subject": "user", "topic": TOPIC,
            "self_ref": False, "air_stance": "", "sensitive": False, "entities": [],
            "open_loops": [{"content": "在学吉他", "kind": "user_task", "due_at": ""}],
        }

        def fn(prompt, schema):
            if schema.get("name") == "scene_card":
                return card
            if schema.get("name") == "memo_class":
                return {"items": [{"content": "在学吉他", "class": "progress"}]}
            return {}

        llm = LLM()
        llm.set_mock(fn)
        scene, _, _ = distill_step1(
            self.store, [{"speaker": "user", "text": "我在学吉他"}], llm, source="t")

        memos = self.store.memos_by_scene(scene.id)
        self.assertEqual(len(memos), 1)
        self.assertEqual(memos[0].kind_class, "progress")
        self.assertEqual(memos[0].window_days,
                         cfgmod.cfg("memo", "window_defaults")["progress"])


# ---------------------------------------------------------------------
# 九、时机表与主动开口（2026-09-15）
#
# 盯两件事：**「什么时候提才自然」是分类映射出来的**（面试要等晚上、
# 婚礼要等办完、生病尽快），和**她说一次就停**（对空气说也是留言，
# 不是反复念叨——定位是"一次性的自动化任务"）。
#
# 时间全部走**固定时间线**（2026-09-15）：断言不受"跑测试的真实时刻"
# 影响（否则安静时段 / 到期判定会随昼夜飘）。
# ---------------------------------------------------------------------

T = datetime(2026, 9, 15, 15, 0, 0)     # 注入的「现在」：下午三点，不在安静时段
PAST = "2026-09-14 09:00:00"            # 一条已到期的时间戳（相对 T）


if __name__ == "__main__":
    unittest.main(verbosity=2)
