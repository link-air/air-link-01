"""阶段 5（对话层）里后来补的那些机制的测试。

这一层的风险不是"算得对不对"，而是**机制写了却没接线**——
`mode` 参数一路透传却没人读、`mark_raised` 只有测试在调、
`mention_count` 在生产链路里永远是 0。这些都是**不报错、不出异常、
只是"应该发生的事没发生"**的故障，只能靠测试钉住。

所以这里每条用例都对应一次"曾经断过的线"：
  - memo 提过一次就不再提（状态机接线）
  - 提及计数只在真被说出口时动（且内部召回不动它）
  - 压缩摘要带日期（只写时刻会被当成别的一天）
  - 没有向量服务时也要判切分（否则等于永不自动提取）
  - **失败要分得出种类**：一句"检查 LLM 配置"会把人引到错的方向
"""
# 用例分组：
#   脚手架  _NoEmbedding · Base · _StreamLLM · _FakeItem
#   接线    MemoWiringTest 备忘 · MentionWiringTest 提及 · AppWiringTest 启动备份/收尾整理 ·
#           LangTest 语言
#   渲染    WindowTierTest 窗口两档 ·
#           DigestDateTest 摘要日期 · RecallViewTest 召回视图 ·
#           FrameConventionTest 音频帧两处同步
#   降级    DegradedCutTest 无向量也要判切 · FailureClassificationTest 失败分类
#   对话    SceneEditDeleteTest 改删 · UndoTurnTest 重说 · ChatStreamTest 流式 ·
#           ReplyStreamTest 流式 reply · ContextBudgetTest 预算 · RawDocsTest 原文文档
from __future__ import annotations

import io
import json
import os
import socket
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core import net as netmod
from core.chat import ChatSession, _no_reply_hint, fit_context
from core.dashboard import App, _recall_view, _service_status, _sse_safe
from core.embedding import EmbeddingService
from core.llm import LLM, _classify_error
from core.memo import standing_memos
from core.model import (PROFILE_ESTABLISHED, Memo, Profile, Raw, Scene,
                        Summary)
from core.prompts import SCENE_SCHEMA, rel_day
from core.recall import asked_for_referent, mark_mentioned, scene_mentioned
from core.scene import should_cut_texts
from core.shortterm import ShortTerm
from core.store import Store
from core.weave import (archive_profile_confirmed, archive_summary_confirmed,
                        delete_many_confirmed, delete_scene_confirmed,
                        delete_summary_confirmed, unarchive_profile_confirmed,
                        unarchive_summary_confirmed, update_profile_confirmed,
                        update_scene_confirmed, update_summary_confirmed)


class _NoEmbedding:
    """假向量服务：**不发网络请求**。

    不传 `emb=None` 是因为 `ChatSession` 会拿它去 `build_embedding()`，
    那是照着真实配置建客户端——测试环境里会真的去连一次 ollama 然后失败。
    """
    available = False

    def embed_one(self, text):
        return None

    def embed(self, texts):
        return None


def scripted_llm(worth: bool = False,
                 hit_close: str = "", hit_update: tuple = ()) -> LLM:
    """假模型：`structured` 按 schema 名给值，`chat` 给一句固定回复。

    `worth=False`：写入路径不是这里的重点，让它跳过落库，
    免得每条用例都在断言"多出一张场景卡"。
    `hit_close`：让**命中判定**判出「这句了结了某条」（闭合接线的用例用；
    默认空串 = 什么都没了结——真实链路里绝大多数轮次走的就是这条短路）。
    `hit_update`：`(id, 新内容, 新时间)`——判出「这句变更了某条」（可空）。
    """
    defaults = dict(SCENE_SCHEMA["defaults"])

    def fn(prompt, schema):
        name = (schema or {}).get("name") or ""
        if not name:                                   # llm.chat：prompt 其实是 messages
            return "嗯，我在。"
        if name == "cue_judgement":
            return {"tense": "now", "valence": None, "arousal": 0,
                    "about_relation": False, "unresolved": False}
        if name == "memo_hit":
            items = []
            if hit_close:
                items.append({"id": hit_close, "action": "close", "content": "",
                              "due_at": "", "why": "（测试）"})
            if hit_update:
                mid, content, due = (tuple(hit_update) + ("", "", ""))[:3]
                items.append({"id": mid, "action": "update", "content": content,
                              "due_at": due, "why": "（测试）"})
            return {"items": items}
        if name == "memo_class":
            return {"items": []}
        if name == "scene_card":
            return dict(defaults, worth_saving=worth, skip_reason="（测试）")
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

    def session(self, llm=None) -> ChatSession:
        return ChatSession(self.store, llm or scripted_llm(),
                           emb=_NoEmbedding(), state_path=self.root / "st.json")


# ---------------------------------------------------------------------

class MemoWiringTest(Base):
    """memo 状态机的接线——`mark_raised` 以前只有测试在调。"""

    def _add_due_memo(self, content="面试结果"):
        from core.model import Memo
        return self.store.add_memo(Memo(content=content, due_at="2020-01-01",
                                        status="pending"))

    def test_raised_after_being_offered(self):
        """进了本轮注入 → 转「已提」。**提过一次就不再提**。"""
        mid = self._add_due_memo()
        self.session().reply("在吗")

        self.assertEqual(self.store.get_memo(mid).status, "raised")
        # 已提过的不再进"到点"池（due_memos 只取 pending）
        self.assertEqual(self.store.due_memos(), [])

    def test_not_due_is_not_marked_raised(self):
        """没到点的**不算提过**（它只是"手上得有"）——到点了照常有机会。"""
        from core.model import Memo
        mid = self.store.add_memo(Memo(content="下周的事", due_at="2099-01-01",
                                       status="pending"))
        self.session().reply("在吗")

        self.assertEqual(self.store.get_memo(mid).status, "pending")
        self.assertEqual([m.id for m in self.store.due_memos()], [],
                         "没到点，它也不该在到点池里")

    def test_trimmed_due_memo_is_not_marked_raised(self):
        """被预算裁掉的**不算「提过」**——它没进本轮注入，下轮再来正合设计。

        `fit_context` 是**就地**裁的（`_drop_one` 从 `recall["standing_memos"]`
        里 pop），`_mark_memos_raised` 读的就是裁剩下的那一份。这条链断了，
        「提过一次就不再提」会变成「看都没看到也算提过」。
        """
        mid = self._add_due_memo()
        payload = {
            "scenes": [Scene(title="面试没过", text="面试没过")],
            "profiles": [], "summaries": [], "raws": [],
            "standing_memos": [{"id": mid, "content": "面试结果", "kind": "user_task",
                                "why": "到点", "due": True}],
            "flags": {}, "cues_view": {},
        }
        old = cfgmod.CONFIG["context"]["total_budget"]
        self.addCleanup(lambda: cfgmod.CONFIG["context"].update({"total_budget": old}))
        sess = self.session()
        with patch("core.chat.recall_for_message", return_value=payload):
            cfgmod.CONFIG["context"]["total_budget"] = 1     # 必裁：小到什么都留不住
            out = sess.reply("在吗")

        self.assertFalse(payload["standing_memos"], "到点那件该被预算裁掉")
        self.assertEqual(out["memo_raised"], [], "这一轮没有一条真进了注入")
        self.assertEqual(self.store.get_memo(mid).status, "pending",
                         "被裁掉的不算提过——下一轮它还要能再进注入")
        self.assertIn("备忘录", "".join(out["recall"]["budget"]["dropped"]))

    def test_closed_this_turn_is_not_flipped_back_to_raised(self):
        """同一轮里「进了注入」+「被他这句了结」→ 保持 closed，回执也不报已提。

        时序：唤醒先算常备栏（那时还是 pending）→ 生成 → 写入（**命中判定**
        关成 closed）→ 记账（`_mark_memos_raised`）。最后这一步原来无条件写
        `raised`，会把刚关掉的那条**打回未了结**：钩子已经标了 `closed_at`，
        它却又挂在清单上（和 M-0012 同一类反刍，只是入口不同）。
        """
        mid = self._add_due_memo(content="面试结果")
        payload = {
            "scenes": [], "profiles": [], "summaries": [], "raws": [],
            "standing_memos": [{"id": mid, "content": "面试结果", "kind": "user_task",
                                "why": "到点", "due": True}],
            "flags": {}, "cues_view": {},
        }
        sess = self.session(scripted_llm(hit_close=mid))
        with patch("core.chat.recall_for_message", return_value=payload):
            out = sess.reply("面试过了，挺顺利")

        self.assertEqual(self.store.get_memo(mid).status, "closed",
                         "同一轮里被了结的，不能被记账打回未了结")
        self.assertEqual(out["memo_raised"], [], "它没真转成已提，回执不该报")
        self.assertEqual((out["memo_closed"] or {}).get("id"), mid)

    def test_updated_this_turn_is_not_marked_raised(self):
        """同一轮里被命中判定**改成「变更」**的那条，不记「已提」。

        时序同上面那条：候选是唤醒时算的（旧内容）→ 生成期间他说了新口径
        （`update` → 回 `pending`）→ 记账。记账若照样打 `raised`，
        **新内容就永远没机会再进候选**（它已经 raised 了）——内容变了
        = 有新的事实要提醒，那条机会不能被旧内容的账吞掉。
        """
        mid = self._add_due_memo(content="月底回老家")
        payload = {
            "scenes": [], "profiles": [], "summaries": [], "raws": [],
            "standing_memos": [{"id": mid, "content": "月底回老家", "kind": "user_task",
                                "why": "到点", "due": True}],
            "flags": {}, "cues_view": {},
        }
        sess = self.session(scripted_llm(hit_update=(mid, "下个月才回老家", "")))
        with patch("core.chat.recall_for_message", return_value=payload):
            out = sess.reply("那个不回了，下个月才回")

        cur = self.store.get_memo(mid)
        self.assertEqual(cur.content, "下个月才回老家")
        self.assertEqual(cur.status, "pending", "变更后的新内容还没被提过，要留着")
        self.assertEqual(out["memo_raised"], [], "它没真转成已提，回执不该报")


class MemoClosureFlowTest(Base):
    """闭合回流（2026-09-21 设计稿 D 条 2）：memo 关了，卡里的钩子一起标掉。
    （2026-10-05 补：退役走**另一个字段** `retired_at`——退役 ≠ 完成。）

    原来是"同一个事实存两份、状态只更新一份"——关掉 memo 之后，渲染和清单
    还会继续端出已经了结的钩子（M-0004 就是活例子）。
    """

    def _scene_with_loop(self, content="用户提供新闻底稿"):
        s = Scene(title="断层", text="摘要",
                  open_loops=[{"content": content, "kind": "user_task", "due_at": ""}])
        self.store.add_scene(s)
        m = Memo(content=content, kind="user_task", scene_id=s.id)
        self.store.add_memo(m)
        return s, m

    def test_closing_memo_marks_the_loop(self):
        """标，不删——内容留着，历史才回答得了「它什么时候关的」。"""
        s, m = self._scene_with_loop()
        from core.memo import close
        close(self.store, m.id)
        loop = self.store.get_scene(s.id).open_loops[0]
        self.assertTrue(loop.get("closed_at"), "钩子要被标掉")
        self.assertEqual(loop["content"], "用户提供新闻底稿", "内容不许删")

    def test_retiring_marks_retired_not_closed(self):
        """**退役 ≠ 完成**（2026-10-05）：钩子标 `retired_at`（另一个字段），
        `closed_at` 保持空着——"办完了"和"没人管了"分得开，渲染侧两者都跳过。"""
        from core.memo import retire_due
        s, m = self._scene_with_loop()
        # 40 天前挂的事、30 天窗口 → 起点在 10 天前、grace（2 天）也早过完 → 该退役
        old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
        with self.store.conn:
            self.store.conn.execute("UPDATE memos SET created_at = ? WHERE id = ?",
                                    (old, m.id))
        retired = retire_due(self.store)
        self.assertIn(m.id, [r["id"] for r in retired])
        loop = self.store.get_scene(s.id).open_loops[0]
        self.assertFalse(loop.get("closed_at"), "退役不是完成——不许写成闭合")
        self.assertTrue(loop.get("retired_at"), "但要留下退役的痕（渲染据此收口）")

    def test_retire_window_replaces_the_90_day_cap(self):
        """口径换代（2026-10-05）：**过了提的窗口就退**，不再等"窗口 × 3 / 90 天"。

        老用例钉的是 90 天硬顶——同样一条"40 天前挂的 idea"，老口径会继续挂着，
        新口径（30 天窗口 + 2 天 grace）该退役。
        """
        from core.memo import retire_due
        old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
        m = Memo(content="四十天前挂的 idea", window_days=30, created_at=old)
        self.store.add_memo(m)
        self.assertIn(m.id, [r["id"] for r in retire_due(self.store)])

    def test_closure_comes_back_in_the_receipt_and_only_once(self):
        """她判出的闭合 → 进这一轮回执（编号 + 内容），**只报一次**。

        `ShortTerm.last_closed_memo` 原来只写不读（注释写着"给调用方留痕"、
        而全项目没有那个调用方）——这条用例把那条线钉住：闭合要能被
        界面 / 实验脚本看见，但读后即清，不能被下一轮再报一次
        （同 `pending_confirm`：隔了一会儿的下句话，不该被当成对它的点头）。
        """
        _s, m = self._scene_with_loop("下周三的面试")
        out = self.session(scripted_llm(hit_close=m.id)).reply("面试过了，还挺顺利")

        self.assertEqual((out["memo_closed"] or {}).get("id"), m.id)
        self.assertIn("面试", (out["memo_closed"] or {}).get("content", ""),
                      "回执要带内容——光有编号，说'结掉了哪件'还得再查一次库")
        self.assertEqual(self.store.get_memo(m.id).status, "closed")

        # 只报一次：下一轮没有新的闭合，回执里就该是空的
        out2 = self.session(scripted_llm()).reply("嗯")
        self.assertIsNone(out2["memo_closed"])

    def test_closure_leaves_a_trace(self):
        """她判出的闭合也要留痕——四个发起方共用一份（原来这条路一条都没有）。"""
        _s, m = self._scene_with_loop("交底稿")
        self.session(scripted_llm(hit_close=m.id)).reply("交完了")

        files = list((self.root / "trace").glob("备忘-*.jsonl"))
        self.assertTrue(files, "她判的闭合要留痕——'为什么它不再提了'要答得出来")
        text = files[0].read_text(encoding="utf-8")
        self.assertIn(m.id, text)
        self.assertIn("命中判定", text, "留痕要写清是谁关的（四个发起方各有来源）")


class StandingMemosTest(Base):
    """常备备忘录（2026-09-21 设计稿 D2b / F）：**注入 ≠ 提**——她手上得有。

    07:04 的失败是"她手上没有"，不是"她没提"。
    """

    def _memo(self, content, **kw):
        m = Memo(content=content, **kw)
        self.store.add_memo(m)
        return m

    def test_recent_two_are_picked(self):
        a = self._memo("甲", created_at="2026-09-20 10:00:00")
        b = self._memo("乙", created_at="2026-09-20 11:00:00")
        c = self._memo("丙", created_at="2026-09-20 12:00:00")
        ids = [x["id"] for x in standing_memos(self.store, "", emb=None)]
        self.assertIn(c.id, ids)
        self.assertIn(b.id, ids)
        self.assertNotIn(a.id, ids, "正常轮按需挑，不是全列")
        self.assertLessEqual(len(ids), 4, "封顶 4：成本恒定")

    def test_relevant_one_is_picked(self):
        """语义相关的那条能入选（无向量时走字符重叠）——最近两条之外的名额。"""
        self._memo("用户提供新闻底稿", created_at="2026-09-20 09:00:00")
        self._memo("买牛奶", created_at="2026-09-20 10:00:00")
        self._memo("取快递", created_at="2026-09-20 11:00:00")
        contents = [x["content"] for x in
                    standing_memos(self.store, "底稿的事呢", emb=None)]
        self.assertIn("用户提供新闻底稿", contents)

    def test_full_lists_everything_for_a_followup(self):
        for i in range(6):
            self._memo(f"第{i}件")
        out = standing_memos(self.store, "哪件事", emb=None, full=True)
        self.assertEqual(len(out), 6, "他追问时要全列——此刻要的是认得出来")

    def test_hard_privacy_never_enters(self):
        self._memo("体检结果", sensitive=2)
        self._memo("普通的事")
        contents = [x["content"] for x in
                    standing_memos(self.store, "", emb=None, full=True)]
        self.assertNotIn("体检结果", contents, "硬隐私只记不提，连摊在手上也不做")

    def test_followup_walks_the_same_path_now(self):
        """`followup` **不再单独排除**（2026-10-05 晚）：

        原来它"一生不进常备栏"，靠的是候选 / 主动开口那两条**会记账**的路；
        那两条都删了，现在它跟别的 memo 走同一条——**到点进注入一次**（进即记账）、
        没到点的不占"到点"那个位子。他追问指代时照旧全列。
        """
        f = self._memo("先跑几天观察", kind_class="followup")
        n = self._memo("普通的事", kind_class="progress")
        picked = standing_memos(self.store, "", emb=None)
        ids = [x["id"] for x in picked]
        self.assertIn(n.id, ids, "别的类照常给")
        self.assertFalse(any(x.get("due") for x in picked),
                         "两条都没到点——谁都不该带 due 标记")
        ids_full = [x["id"] for x in
                    standing_memos(self.store, "哪件事", emb=None, full=True)]
        self.assertIn(f.id, ids_full, "他追问时照旧给")

    def test_carries_group_name(self):
        """常备栏带组名（2026-09-22）：注入侧靠它显示"这是一件事的第几步"。"""
        self._memo("写脚本", group_name="开源准备")
        out = standing_memos(self.store, "", emb=None)
        self.assertEqual(out[0]["group_name"], "开源准备")

    def test_due_followup_comes_once_with_the_flag(self):
        """**到期后**的 followup 跟别的 memo 一样：进注入一次、带 `due`（会被记账）。

        它不再有"单独一条路"——"每轮刷"由**进即记账**挡住：
        进了就转 raised，下一轮不在这栏里。
        """
        m = self._memo("先跑几天观察", kind_class="followup", window_days=4,
                       created_at="2026-09-01 10:00:00")      # 早就到期
        due_items = [x for x in standing_memos(self.store, "", emb=None)
                     if x.get("due")]
        self.assertEqual([x["id"] for x in due_items], [m.id],
                         "到点就进——和别的 memo 同一条路，一次一件")
        # 记账之后（chat 那一侧做的）它就不再进这一栏了
        from core.memo import mark_raised
        mark_raised(self.store, m.id)
        self.assertFalse(any(x.get("due") for x in
                             standing_memos(self.store, "", emb=None)))


class AskedForReferentTest(unittest.TestCase):
    """追问指代的判据要**窄**（2026-09-21 设计稿 F 条）——不然每轮都全列。"""

    def test_recognizes_a_followup(self):
        self.assertTrue(asked_for_referent("哪件事我一句没提"))
        self.assertTrue(asked_for_referent("你说的是哪个？"))

    def test_ignores_ordinary_talk(self):
        self.assertFalse(asked_for_referent("今天天气不错"), "没指代")
        self.assertFalse(asked_for_referent("那件事后来怎么样了"), "没在问是哪件")
        self.assertFalse(asked_for_referent("那件事" + "很长" * 40), "长消息多半在讲事情本身")


class MentionWiringTest(Base):
    """提及计数：防反刍的上限和老化的「未被提及」那一半都靠它。"""

    def test_counts_only_when_actually_said(self):
        s = Scene(title="面试没过", keywords=["面试"], text="面试没过")
        self.store.add_scene(s)

        mark_mentioned(self.store, [s], ["今天天气不错"])
        self.assertEqual(self.store.get_scene(s.id).mention_count, 0,
                         "没提到就不该动——**内部召回不算提及**")

        mark_mentioned(self.store, [s], ["今天我去面试了"])
        self.assertEqual(self.store.get_scene(s.id).mention_count, 1)

    def test_single_char_keyword_is_not_a_hit(self):
        """单字关键词在任何对话里都会命中，等于没判。"""
        s = Scene(title="猫", keywords=["猫"])
        self.assertFalse(scene_mentioned(s, ["我今天吃了饭"]))

    def test_touches_profile_last_support(self):
        """被提及也算「这条画像还活着」——老化判据的另一半。"""
        s = Scene(title="面试没过", keywords=["面试"])
        self.store.add_scene(s)
        p = Profile(topic="用户·面试", statement="面试会紧张", status="pending",
                    sources=[s.id], last_support_at="2020-01-01 00:00:00")
        self.store.add_profile(p)

        mark_mentioned(self.store, [s], ["面试怎么样了"])

        self.assertGreater(self.store.get_profile(p.id).last_support_at,
                           "2020-01-01 00:00:00")


class AppWiringTest(unittest.TestCase):
    """`App` 启动与收尾的两条线——**两条都是「写好了没接线」的旧课**：

      - `backup_daily()`：一度注释写着「每日一份」却没有任何调用点（后来接在
        `App.__init__` 上）；
      - `run_distill_cycle()`：一度只在实验脚本里跑——在仪表盘里聊多久，
        场景卡一直写、S2 和画像却永远不会形成。

    空库跑，不产生任何模型调用；路径全部重定向到临时目录。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._old_paths = {k: cfgmod.PATHS[k]
                           for k in ("trace_dir", "raws_dir", "backup_dir", "shortterm",
                                     # `App.__init__` 会调 `settings.apply()`——本地配置也
                                     # 指向临时文件（不存在 = 什么都不合），免得把开发机上
                                     # 真正的 config.local.json 合进全局 CONFIG：
                                     # 后面的用例（如 salvage 的"没配模型"）还假设它没被合过
                                     "local_config")}
        self._old_db = cfgmod.CONFIG.get("db_path")
        cfgmod.PATHS.update({"trace_dir": str(self.root / "trace"),
                             "raws_dir": str(self.root / "raws"),
                             "backup_dir": str(self.root / "backups"),
                             "shortterm": str(self.root / "st.json"),
                             "local_config": str(self.root / "config.local.json")})
        cfgmod.CONFIG["db_path"] = str(self.root / "t.db")
        # `App` 启动会调 `settings.apply()`，它有两个**真实输入源**——都要隔离，
        # 否则会把开发机上的东西合进全局 CONFIG，后面的用例（如 salvage 的
        # 「没配模型」）就会拿着真 endpoint 去发网络请求（实测：全量从 40 秒变 4 分钟）：
        #   ① 环境变量（这台机器上有求知版遗留的 `AIR2_LLM_*`）；
        #   ② config.local.json（上面已指向不存在的临时文件 = 什么都不合）。
        self._old_env = {k: os.environ.pop(k) for k in list(os.environ)
                         if k.startswith(("AIR_LINK_", "AIR2_"))}

    def tearDown(self):
        for k, v in self._old_env.items():
            os.environ[k] = v
        for k, v in self._old_paths.items():
            cfgmod.PATHS[k] = v
        if self._old_db is None:
            cfgmod.CONFIG.pop("db_path", None)
        else:
            cfgmod.CONFIG["db_path"] = self._old_db
        self._tmp.cleanup()

    def test_startup_makes_one_backup_a_day(self):
        """启动时备一份；同一天再启动不重复（幂等）。"""
        app = App(port=0)
        try:
            app2 = App(port=0)
            try:
                files = list((self.root / "backups").glob("air_link-*.db"))
                self.assertEqual(len(files), 1, "每天一份——当天第二次启动该跳过")
            finally:
                app2.store.close()
        finally:
            app.store.close()

    def test_trace_reads_the_most_recent_files_not_the_alphabet(self):
        """留痕页取文件**按修改时间**，不是按文件名（2026-10-07 修）。

        按文件名倒序时「漂移 / 档案 / 整理」的码点都比「唤醒」大——那几类一多，
        名额就被占满，**唤醒记录一条也进不来**，而"她为什么没想起那件事"
        只有唤醒那类答得出来。这条钉住两件事：最近写的那份读得到、
        记录按自己的 `ts` 倒序（不是按文件拼接顺序）。
        """
        d = self.root / "trace"
        d.mkdir(parents=True, exist_ok=True)
        writes = [
            ("漂移-20260101.jsonl", '{"ts": "2026-01-01 10:00:00", "checked": 1, "drifting": []}'),
            ("档案-20260101.jsonl", '{"ts": "2026-01-01 10:00:01", "key": "城市", "action": "新增"}'),
            ("整理-20260101.jsonl", '{"ts": "2026-01-01 10:00:02", "act": "体检"}'),
            # 唤醒写在**最后**：它的文件名字头最小，按字母排永远进不了前几名
            ("唤醒-20260101.jsonl", '{"ts": "2026-01-01 10:00:03", "input": "在吗"}'),
        ]
        for name, line in writes:
            (d / name).write_text(line + "\n", encoding="utf-8")

        app = App(port=0)
        try:
            rows = app.trace(limit=10)
        finally:
            app.store.close()

        kinds = [r.get("kind") for r in rows]
        self.assertIn("唤醒", kinds, "最近写的那份必须读得到（按字母排它就没了）")
        self.assertEqual(rows[0].get("input"), "在吗", "按记录时间倒序：最新的在最上面")
        self.assertIn("档案", kinds, "别的几类照常读得到——修的是「取哪几份」，不是「只留唤醒」")
        self.assertEqual([r["kind"] for r in rows],
                         ["唤醒", "整理", "档案", "漂移"], "四条记录按 ts 倒序")

    def test_stopping_keeps_the_half_sentence_in_the_window(self):
        """**按停止**：她说到一半的那半句要落进窗口（2026-10-07）。

        这条走**完整那条路**（不是直接建 ChatSession）：客户端断开 → 服务端写不出去
        → `App.chat_stream` 的 `finally` 把生成器 close 掉 → `reply_stream` 收到
        `GeneratorExit` → 落窗口。

        这里测的是**落没落**（且只落到窗口——**长期库不留**，见
        `PartialTurnMemoryTest`）；"必须在放锁之前落"那一半测不出来（那靠的是
        `finally` 嵌在 `with self._lock` 里面这个结构，见 `App.chat_stream` 的注释），
        它是为了让紧跟着来的 `/api/turn-undo`（重新生成）撤到正确的那一轮。
        """
        from core.chat import ChatSession
        app = App(port=0)
        try:
            app._session = ChatSession(
                store=app.store, emb=_NoEmbedding(), state_path=self.root / "st.json",
                llm=_StreamLLM([{"type": "content", "text": "抱这一下"},
                                {"type": "content", "text": "，我给不了。"}]))
            gen = app.chat_stream("我有点撑不住")
            next(gen)                     # 收到第一帧
            gen.close()                   # ≈ 客户端按了「停止」
            msgs = app.session.st.messages[-2:]
        finally:
            app.store.close()

        self.assertEqual([m["speaker"] for m in msgs], ["user", "air"])
        self.assertEqual(msgs[-2]["text"], "我有点撑不住")
        self.assertEqual(msgs[-1]["text"], "抱这一下", "只落已经说出口的那半句")
        self.assertTrue(msgs[-1].get("interrupted"), "落个标记：界面回填要标「已停止」")

    def test_end_kicks_off_a_distill_run(self):
        """收尾真的触发一次整理——这条线断过，画像就永远长不出来。"""
        app = App(port=0)
        try:
            app.end()
            self.assertTrue(app.wait_distill(), "空库，整理该很快跑完")
            self.assertIn("s2_new", app.last_distill, "跑过的结果要留在 last_distill（给界面看）")
        finally:
            app.store.close()

    def test_memo_close_marks_and_reports(self):
        """人随手划掉（`POST /api/memo-close` 的后端动作，2026-09-21 设计稿 D 条 4）。"""
        app = App(port=0)
        try:
            m = Memo(content="面试结果")
            app.store.add_memo(m)
            out = app.memo_close(m.id)
            self.assertTrue(out["ok"])
            self.assertEqual(app.store.get_memo(m.id).status, "closed")
            self.assertIn("面试结果", out["detail"])
            # 再划一次、划不存在的、没给编号：三种都拒绝且说清为什么
            self.assertFalse(app.memo_close(m.id)["ok"], "已经关过了")
            self.assertFalse(app.memo_close("M-9999")["ok"])
            self.assertFalse(app.memo_close("")["ok"])
        finally:
            app.store.close()

    def test_memo_close_marks_the_loop_and_leaves_a_trace(self):
        """划掉要带**闭合回流** + 留痕（用户随手划的唯一记录）。"""
        app = App(port=0)
        try:
            s = Scene(title="断层", text="x",
                      open_loops=[{"content": "用户提供底稿", "kind": "user_task"}])
            app.store.add_scene(s)
            m = Memo(content="用户提供底稿", scene_id=s.id)
            app.store.add_memo(m)
            self.assertTrue(app.memo_close(m.id)["ok"])
            loop = app.store.get_scene(s.id).open_loops[0]
            self.assertTrue(loop.get("closed_at"), "钩子要一起标掉（闭合回流）")
            files = list((self.root / "trace").glob("备忘-*.jsonl"))
            self.assertTrue(files, "人的划掉要留痕")
            self.assertIn(m.id, files[0].read_text(encoding="utf-8"))
        finally:
            app.store.close()


# （原来这里有两个用例类：`WeaveModeTest`（两套姿态下 pending 画像进不进注入）
#   和 `WorldDedupTest`（世界块与摘要栏去重）——2026-09-20 随姿态与
#   「关于世界的了解」一起退役，整块删除。留下的约束只有一条：
#   **pending 画像一律不进注入**（不武断），在 recall 里由 `PROFILE_ESTABLISHED`
#   写死。）


class LangTest(Base):
    """语言（中 / 英，2026-09-17）：**中文语言段零注入**；英语加一段英文指令。

    「她的输出跟着语言走」由这段提示词管；仪表盘文案由前端字典管——
    两个机制各管一边，别混。

    2026-09-26 多了一处，但**不在语言段里**：**中文模式**在**整段提示词的最后**
    加一行思维语言（`render_think_block` / `THINK_IN_CHINESE`：「思考用中文」）。
    思维链由模型自己产、API 没有语言开关，而模型默认用英文想——只能靠这一句顶，
    且只有末行押得住（见那个常量）。英文模式零注入：英文正是它的默认，顺默认不必说
    （实测英文输入 / 中文输入两种，加与不加都是 6/6 英文思维链）。
    """

    def test_zh_lang_block_is_zero_injection(self):
        """默认路径这一段一个字都不多——多一句就多一分上下文开销。"""
        from core.prompts import render_lang_block
        self.assertEqual(render_lang_block("zh"), "")
        self.assertEqual(render_lang_block(""), "")

    def test_zh_think_language_is_the_last_line(self):
        """中文模式的思维语言必须在**最后一段**、且在时间之后。

        位置就是效果本身（mimo-v2.6-flash 实测：不写 0/5、中段 2/4、末行 7/10
        ——**是概率，不是开关**）。
        """
        from core.prompts import build_system_prompt
        zh = build_system_prompt("章程", {}, lang="zh")
        parts = [p for p in zh.split("\n\n---\n\n") if p.strip()]
        self.assertIn("思考过程一律用中文", parts[-1], "必须在最后一段里")
        self.assertGreater(zh.index("思考过程一律用中文"), zh.index("当前时间"),
                           "要排在时间之后（整段的最后一件事）")

    def test_en_think_block_is_zero_injection(self):
        """英文模式**不加**思维语言——英文是模型的默认，顺默认不必说；
        而且它不在固定块预算的口径里，加了就是隐形开销。"""
        from core.prompts import build_system_prompt, render_think_block
        self.assertEqual(render_think_block("en"), "")
        en = build_system_prompt("章程", {}, lang="en")
        self.assertNotIn("思考过程一律用中文", en)
        self.assertNotIn("思考过程", en.split("\n\n---\n\n")[-1],
                         "末尾不该多出思维语言那段")

    def test_think_block_falls_back_to_chinese(self):
        """非法 / 空值按默认（中文）走——同 `lang_from_prefs` 的兜底。"""
        from core.prompts import render_think_block
        self.assertEqual(render_think_block(""), render_think_block("zh"))
        self.assertNotEqual(render_think_block(""), "")

    def test_en_block_says_english_despite_chinese_context(self):
        """指令用英文写、且明说「上面是中文也别管」——它要顶住满篇中文的上下文。"""
        from core.prompts import render_lang_block
        block = render_lang_block("en")
        self.assertIn("Reply in English", block)
        self.assertIn("Chinese", block)

    def test_prompt_carries_lang_only_in_english(self):
        from core.prompts import build_system_prompt
        zh = build_system_prompt("章程", {}, lang="zh")
        en = build_system_prompt("章程", {}, lang="en")
        self.assertNotIn("Reply in English", zh)
        self.assertIn("Reply in English", en)

    def test_default_and_illegal_value_fall_back_to_zh(self):
        from core.model import lang_from_prefs
        self.assertEqual(lang_from_prefs({}), "zh")
        self.assertEqual(lang_from_prefs({"lang": "fr"}), "zh")

    def test_save_persists_and_traces(self):
        """他定的东西，改动要能回看（语言也是她说话的一部分——切了它，提取也跟着变）。"""
        from core.weave import save_lang_confirmed
        self.assertEqual(save_lang_confirmed(self.store, "en"), "en")
        self.assertEqual(self.store.get_pref("lang"), "en")
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob("语言-*.jsonl"))
        text = "\n".join(f.read_text(encoding="utf-8") for f in files)
        self.assertIn("en", text)

    def test_save_ignores_illegal_value(self):
        from core.weave import save_lang_confirmed
        self.assertEqual(save_lang_confirmed(self.store, "xx"), "zh")
        self.assertEqual(self.store.get_pref("lang"), "")


class WindowTierTest(Base):
    """窗口只分两档：**最近 N 条逐字 + 更早的一行一条**。

    想过再加一档「100 字的摘要」，放弃了——理由在 `build_window` 的注释里。
    这里钉住的是那套理由换来的两条行为约定：
      - 逐字只有最近几条，**多的不再原样进上下文**（这就是"几千字"那个问题的本体）
      - 更早的**不能消失**：旧实现是把窗口以外的消息从渲染里直接丢掉（黑洞），
        现在它们一律降成一行
    """

    def st(self) -> ShortTerm:
        return ShortTerm(self.store, scripted_llm(), emb_service=_NoEmbedding(),
                         session_id="t", state_path=self.root / "st.json")

    def test_only_recent_is_verbatim(self):
        st = self.st()
        keep = cfgmod.cfg("shortterm", "verbatim_messages")
        for i in range(keep + 8):
            st.append("user", f"第{i}句")
            st.append("air", f"回{i}")
        win = st.build_window()

        recent = win.split("[最近对话]")[-1]
        # 逐字行现在带时间前缀（「用户（今天）2026-09-20: …」）——数行首的
        # 「用户」才是在数条数（本测试的意图），不能数「用户:」。
        n_user = sum(1 for ln in recent.splitlines() if ln.startswith("用户"))
        self.assertEqual(n_user, keep // 2, "逐字部分只该有最近这几条")
        self.assertIn(f"第{keep + 7}句", recent, "最新的话当然还在")
        self.assertNotIn("第0句", recent, "最早的那句不该占着逐字的名额")

    def test_older_messages_still_show_up_as_one_line(self):
        """**窗口以外不能是黑洞**：旧实现是把它们从渲染里直接丢掉。"""
        st = self.st()
        st.append("user", "这是我的第一句，后面还有很多内容接在它后面")
        for i in range(12):
            st.append("user", f"第{i}句")
            st.append("air", f"回{i}")
        win = st.build_window()

        self.assertIn("这是我的第一句", win, "更早的话要还在，只是压缩了")
        self.assertIn("[更早的对话", win)

    def test_persona_names_are_kept_in_the_window(self):
        """air 的话按**当时的真名**署名（2026-09-21）：切人格后"谁说的"不抹平。"""
        st = self.st()
        st.append("user", "第一句")
        st.append("air", "这是 mia 说的", persona="mia")
        st.append("user", "第二句")
        st.append("air", "这是没署名的回话")          # 旧数据 / 缺字段：回退 "air"
        win = st.build_window()
        recent = win.split("[最近对话]")[-1]
        # 逐字行带时间前缀（`mia（刚刚）2026-09-21 12:30: …`）——取行首的名字比
        self.assertIn("这是 mia 说的", recent)
        heads = [ln.split("（")[0] for ln in recent.splitlines() if ln.strip()]
        self.assertIn("mia", heads, "mia 的话要署 mia，不能写成 air")
        self.assertIn("air", heads, "没署名的（旧数据）回退 air")

    def test_gist_line_keeps_persona_name(self):
        """「更早」那一栏同样按真名署名（`gist_line`，不是只修了逐字那一栏）。"""
        st = self.st()
        st.append("air", "最早的一句", persona="xina")
        for i in range(12):
            st.append("user", f"第{i}句")
            st.append("air", f"回{i}")
        win = st.build_window()
        older = win.split("[最近对话]")[0]
        self.assertIn("xina", older)

    def test_render_conversation_uses_persona(self):
        """渲染函数本身：有 persona 用真名，没有回退 "air"（旧数据兼容）。"""
        from core.scene import render_conversation
        out = render_conversation([
            {"speaker": "user", "text": "甲"},
            {"speaker": "air", "text": "乙", "persona": "mia"},
            {"speaker": "air", "text": "丙"},
        ])
        self.assertEqual(out.splitlines(), ["用户: 甲", "mia: 乙", "air: 丙"])

    def test_gist_line_is_capped_by_code(self):
        """行长由代码截，不求模型守（模型不会数字数）。"""
        st = self.st()
        for _ in range(6):
            st.append("user", "很长" * 200)
        win = st.build_window()
        # 只看「更早」那一段：最近几条是**逐字**，本来就不该被截
        older = win.split("[最近对话]")[0]
        cap = cfgmod.cfg("shortterm", "gist_line_chars")
        longest = max(len(ln) for ln in older.splitlines() if ln.startswith("用户"))
        self.assertLessEqual(longest, cap + 30,
                             "一行再长也该被截住（+30 是「用户（23 小时前）2026-09-20 06:43: 」"
                             "前缀与省略号的余量）")

    def test_digest_stops_growing(self):
        """压缩区必须有上限：它是唯一会被持续追加、又不被「提取」清空的东西。"""
        st = self.st()
        old = cfgmod.CONFIG["shortterm"]["older_line_cap"]
        cfgmod.CONFIG["shortterm"]["older_line_cap"] = 6
        try:
            for i in range(10):
                st.digest.append(f"〔第{i}批〕\n用户: 说{i}\nair: 回{i}")
                st._trim_digest()
            total = sum(len(c.splitlines()) for c in st.digest)
            self.assertLessEqual(total, 6, "超了就要丢最早的整批")
            self.assertIn("第9批", st.digest[-1], "丢的是最早的，最近这批得留着")
            self.assertNotIn("第0批", "".join(st.digest), "最早的先走")
        finally:
            cfgmod.CONFIG["shortterm"]["older_line_cap"] = old

    def test_injected_older_block_has_a_hard_ceiling(self):
        """不管攒了多少，**注入量有常数上限**——这是这次改动要解决的根本问题。

        只压缩还不够：`messages` 在提取没触发时会一直攒，而一行最多 60 字
        只挡得住"一条特别长"，挡不住"很多条都很短"。
        """
        st = self.st()
        old = cfgmod.CONFIG["shortterm"]["older_line_cap"]
        cfgmod.CONFIG["shortterm"]["older_line_cap"] = 12
        try:
            for i in range(60):
                st.append("user", f"第{i}句")
                st.append("air", f"回{i}")
            block = st.build_window().split("[最近对话]")[0]
            self.assertLessEqual(len(block.splitlines()), 14,
                                 "12 行 + 标题与间隔，注入量不该随消息数增长")
            self.assertIn("第59句", st.build_window(), "最近几句必须还在")
        finally:
            cfgmod.CONFIG["shortterm"]["older_line_cap"] = old

    def test_long_line_in_digest_is_capped(self):
        """模型给的一行太长 → 代码截。**拖进来的一整段文档**就靠这一步顶住。"""
        st = self.st()
        block = st._dated_digest("用户: " + "很长" * 200,
                                 [{"ts": "2026-09-13 21:38:00"}])
        lines = block.splitlines()
        self.assertIn("2026-09-13", lines[0], "日期由代码加，不由模型写")
        cap = cfgmod.cfg("shortterm", "gist_line_chars")
        self.assertLessEqual(len(lines[1]), cap + 1, "超行长的一律由代码截断")


class DigestDateTest(Base):
    """压缩摘要必须带日期——只写时刻会被当成别的一天。"""

    def st(self) -> ShortTerm:
        return ShortTerm(self.store, scripted_llm(), emb_service=_NoEmbedding(),
                         state_path=self.root / "st.json")

    def test_same_day_range(self):
        msgs = [{"ts": "2026-09-13 03:30:00"}, {"ts": "2026-09-13 03:35:00"}]
        out = self.st()._dated_digest("凌晨三点半聊了天", msgs)
        self.assertIn("2026-09-13", out)
        self.assertIn("03:30", out)

    def test_crosses_midnight_shows_both_dates(self):
        """跨夜的那一段最容易出错：只有时刻的话，模型会把它整体推到前一天。"""
        msgs = [{"ts": "2026-09-12 23:40:00"}, {"ts": "2026-09-13 00:12:00"}]
        out = self.st()._dated_digest("从昨晚聊到半夜", msgs)
        self.assertIn("2026-09-12", out)
        self.assertIn("2026-09-13", out)

    def test_empty_digest_stays_empty(self):
        self.assertEqual(self.st()._dated_digest("", [{"ts": "2026-09-13 03:30:00"}]), "")

    def test_stored_digest_has_no_relative_day(self):
        """**存储**里不许出现相对日——它是持久化文本，写死的那一刻就开始过期。"""
        msgs = [{"ts": "2026-09-13 03:30:00"}, {"ts": "2026-09-13 03:35:00"}]
        out = self.st()._dated_digest("聊了天", msgs)
        self.assertNotIn("天前", out)
        self.assertNotIn("（今天）", out)

    def test_head_relative_day_is_added_at_render_time(self):
        """批头的相对日在**渲染时**补（存储只有绝对日期）——旧摘要因此也自动修好。"""
        st = self.st()
        st.digest = ["〔2026-09-13 03:30–03:35〕\n用户: 甲"]
        win = st.build_window()
        self.assertIn("2026-09-13", win)
        rel = rel_day("2026-09-13 03:30")
        self.assertIn(f"〔（{rel}）2026-09-13" if rel else "〔2026-09-13", win)


class DegradedCutTest(unittest.TestCase):
    """没有向量服务时也要判切分——不判就等于「从来不自动提取」。"""

    SAME = ["我下周要面试", "面试有点紧张", "面试准备得如何"]

    def test_same_topic_does_not_cut(self):
        self.assertFalse(
            should_cut_texts("面试的事怎么样了", self.SAME, min_distance=0.93))

    def test_new_topic_cuts(self):
        self.assertTrue(
            should_cut_texts("今天晚上吃什么", self.SAME, min_distance=0.93))

    def test_too_few_messages_never_cuts(self):
        """样本不够就不判——宁可切粗（同 `should_cut` 的取向）。"""
        self.assertFalse(should_cut_texts("今天晚上吃什么", ["面试"],
                                          min_distance=0.93))


# ---------------------------------------------------------------------

class RawDocsTest(Base):
    """S0 原文按天存文档：不在库里，只能经场景编号下钻。"""

    def _scene(self, text="面试没过", when="2026-09-13 03:30:00") -> Scene:
        s = Scene(title=text, text=text, time_record=when, time_event=when)
        self.store.add_scene(s)
        return s

    def test_lookup_with_and_without_date(self):
        s = self._scene()
        day = self.store.add_raw(Raw(scene_id=s.id, content="用户: 面试没过"),
                                 on_date="2026-09-13")
        self.assertEqual(day, "2026-09-13", "文档按天分，返回的是日期")
        self.assertEqual(len(self.store.get_raws_by_scene(s.id, on_date="2026-09-13")), 1)
        self.assertEqual(len(self.store.get_raws_by_scene(s.id)), 1,
                         "不给日期也要找得到（兜底路径）")

    def test_days_are_separate(self):
        a = self._scene("第一段", "2026-09-12 23:40:00")
        b = self._scene("第二段", "2026-09-13 00:12:00")
        self.store.add_raw(Raw(scene_id=a.id, content="用户: 第一段"), on_date="2026-09-12")
        self.store.add_raw(Raw(scene_id=b.id, content="用户: 第二段"), on_date="2026-09-13")

        day_a = (self.store.raws_dir() / "2026-09-12.md").read_text(encoding="utf-8")
        self.assertIn("第一段", day_a)
        self.assertNotIn("第二段", day_a, "跨天的两段不能串进同一份文档")

    def test_section_does_not_bleed_into_next(self):
        """分节边界要准：一节只装一条场景的原文。"""
        a = self._scene("甲", "2026-09-13 01:00:00")
        b = self._scene("乙", "2026-09-13 02:00:00")
        self.store.add_raw(Raw(scene_id=a.id, content="用户: 甲"), on_date="2026-09-13")
        self.store.add_raw(Raw(scene_id=b.id, content="用户: 乙"), on_date="2026-09-13")

        got = self.store.get_raws_by_scene(a.id, on_date="2026-09-13")[0].content
        self.assertEqual(got.strip(), "用户: 甲")

    def test_missing_scene_returns_empty(self):
        self.assertEqual(self.store.get_raws_by_scene(""), [])
        self.assertEqual(self.store.get_raws_by_scene("S1-9999"), [])


class SceneEditDeleteTest(Base):
    """改与删：**只有人能发起**，且都要留痕、都要有回退的路。"""

    def _scene_with_profile(self):
        s = Scene(title="被组长当众批评", text="被批评后想离开")
        self.store.add_scene(s)
        p = Profile(topic="用户·被评价的反应", statement="遇到批评会想退出",
                    status="pending", sources=[s.id])
        self.store.add_profile(p)
        return s, p

    def _traces(self, kind: str) -> str:
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob(f"{kind}-*.jsonl"))
        return "\n".join(f.read_text(encoding="utf-8") for f in files)

    def test_edit_changes_text_and_keeps_old_value(self):
        s, _p = self._scene_with_profile()
        out = update_scene_confirmed(self.store, s.id, "被批评后先扛了一下再决定")

        self.assertTrue(out["ok"] and out["changed"])
        self.assertEqual(self.store.get_scene(s.id).text, "被批评后先扛了一下再决定")
        self.assertIn("被批评后想离开", self._traces("场景改动"),
                      "改前的值必须留下来——不然分不清「当时」和「后来理解的」")

    def test_edit_does_not_touch_raw(self):
        s = Scene(title="甲", text="旧的理解", time_record="2026-09-13 01:00:00")
        self.store.add_scene(s)
        self.store.add_raw(Raw(scene_id=s.id, content="用户: 原话"), on_date="2026-09-13")

        update_scene_confirmed(self.store, s.id, "新的理解")

        raw = self.store.get_raws_by_scene(s.id, on_date="2026-09-13")[0]
        self.assertIn("原话", raw.content, "原文永远不动——动的只是场景卡上的理解")

    def test_edit_same_text_is_noop(self):
        s, _p = self._scene_with_profile()
        out = update_scene_confirmed(self.store, s.id, s.text)
        self.assertTrue(out["ok"])
        self.assertFalse(out["changed"])

    # ---- 三层同一套（2026-09-24，工具箱稿 §3.3 / §3.4）----

    def _summary(self, topic: str = "用户·被评价的反应") -> Summary:
        s2 = Summary(topic=topic, text="他遇到批评时会先退开，缓过来再处理")
        self.store.add_summary(s2)
        return s2

    def test_summary_edit_keeps_old_value(self):
        """S2 可改：原地改 + 旧值留痕（与 S1 同一套）。"""
        s2 = self._summary()
        out = update_summary_confirmed(self.store, s2.id, "他遇到批评先退开，但事后会回头处理")

        self.assertTrue(out["ok"] and out["changed"])
        self.assertEqual(self.store.get_summary(s2.id).text,
                         "他遇到批评先退开，但事后会回头处理")
        self.assertIn("缓过来再处理", self._traces("摘要改动"),
                      "改前的值要留下——同「场景改动」的理由")

    def test_summary_can_edit_topic_but_not_system_fields(self):
        """S2 也能改主题（标签，2026-09-24 晚）：原地改，叙述与素材都不动；
        数值 / 系统字段一律拒绝（同场景那条纪律）。"""
        s2 = self._summary()

        out = self.store.set_summary_fields(s2.id, {"topic": "用户·沟通"})
        self.assertTrue(out.get("ok") and out.get("changed"))
        got = self.store.get_summary(s2.id)
        self.assertEqual(got.topic, "用户·沟通")
        self.assertEqual(got.text, "他遇到批评时会先退开，缓过来再处理",
                         "改主题不动叙述")

        bad = self.store.set_summary_fields(s2.id, {"archived": 1})
        self.assertFalse(bad["ok"], "归档有专门动作，不走字段白名单")
        self.assertIn("不能改", bad["detail"])

        bad2 = self.store.set_summary_fields(s2.id, {"topic": "　"})
        self.assertFalse(bad2["ok"], "主题不能清空——空主题是脏数据不是标签")

    def test_topics_are_multi_with_primary(self):
        """主题标签（2026-09-24 晚）：1-3 个，**第一个 = 主主题**；
        越界与清空都拒绝（拒绝而不是静默截断）。"""
        s2 = self._summary()
        out = self.store.set_summary_topics(s2.id, ["用户·压力", "用户·工作"])

        self.assertTrue(out.get("ok") and out.get("changed"))
        got = self.store.get_summary(s2.id)
        self.assertEqual(got.topics, ["用户·压力", "用户·工作"])
        self.assertEqual(got.topic, "用户·压力", "`topic` 列 = topics[0]（物化，防漂移）")

        bad = self.store.set_summary_topics(s2.id, ["a", "b", "c", "d"])
        self.assertFalse(bad["ok"], "超过 3 个要拒绝——不静默截断")
        self.assertFalse(self.store.set_summary_topics(s2.id, [])["ok"], "不能清空")
        # 走「字段口」（她 / 界面那条路）也一样拒绝——`split_topics` 不截断，
        # 交给 `_write_topics` 明确报错（2026-09-24 检查修）
        bad2 = self.store.set_summary_fields(s2.id, {"topic": "a,b,c,d"})
        self.assertFalse(bad2["ok"], "字段口不静默截断成 3 个")

    def test_topics_backfill_and_find(self):
        """老库回填 `[topic]`（启动迁移）；按主题找是**子串 + 主附都算**。"""
        s2 = self._summary(topic="用户·压力")
        self.store.conn.execute("UPDATE summaries SET topics = '' WHERE id = ?", (s2.id,))
        self.store.conn.commit()
        self.store.close()

        store2 = Store(self.store.path)          # 重开 → `_backfill_topics`
        try:
            self.assertEqual(store2.get_summary(s2.id).topics, ["用户·压力"],
                             "老库回填成 [topic]")
            store2.set_summary_topics(s2.id, ["用户·压力", "用户·工作"])

            self.assertEqual([x.id for x in store2.summaries_with_topic("工作")], [s2.id],
                             "附加主题也要找得到")
            self.assertEqual([x.id for x in store2.summaries_with_topic("压力")], [s2.id],
                             "子串匹配（她记不全主题名）")
            self.assertIn("用户·压力", store2.all_topics())
        finally:
            store2.close()

    def test_profile_topic_is_edited_in_place(self):
        """S3 改主题 = **原地改**（2026-09-24 晚）：改的是归类——
        陈述 / 印证 / 版本链都不动（与"改陈述走修正"是两件事）。"""
        p = Profile(topic="用户·压力", statement="受挫后倾向离开",
                    status=PROFILE_ESTABLISHED, evidence=3)
        self.store.add_profile(p)

        out = self.store.set_profile_topic(p.id, "用户·边界")
        self.assertTrue(out.get("ok") and out.get("changed"))
        got = self.store.get_profile(p.id)
        self.assertEqual(got.topic, "用户·边界")
        self.assertEqual(got.statement, "受挫后倾向离开", "改主题不动陈述")
        self.assertEqual(got.evidence, 3, "印证数照旧")
        self.assertFalse(got.invalidated_at, "原地改——没有旧说法要进历史")

        bad = self.store.set_profile_topic(p.id, " ")
        self.assertFalse(bad["ok"], "主题不能清空")

    def test_profile_topic_edit_leaves_a_trace(self):
        """改标签也留痕（三层同一套）：走 `update_profile_confirmed` 的 topic 档。"""
        p = Profile(topic="用户·压力", statement="受挫后倾向离开",
                    status=PROFILE_ESTABLISHED, evidence=3)
        self.store.add_profile(p)

        out = update_profile_confirmed(self.store, p.id, "用户·边界", "topic")

        self.assertTrue(out["ok"] and out["changed"])
        text = self._traces("画像改动")
        self.assertIn("主题", text)
        self.assertIn("用户·压力", text, "改前的值要留下")

    def test_scene_fields_are_editable_but_not_everything(self):
        """场景能改的不止摘要（2026-09-24，工具箱稿 §七）：**语义描述**类都开；
        数值与系统字段一律**拒绝**（拒绝而不是静默——静默丢弃会让人以为改成了）。"""
        s = Scene(title="旧标题", text="旧摘要", topic="用户·工作",
                  trigger="被批评", reaction="退出", trigger_class="被评价",
                  time_event="2026-09-01 10:00:00")
        self.store.add_scene(s)

        out = self.store.set_scene_fields(s.id, {"title": "新标题", "topic": "用户·压力",
                                                 "trigger_class": "关系冲突"})
        self.assertTrue(out.get("ok") and out.get("changed"))
        got = self.store.get_scene(s.id)
        self.assertEqual(got.title, "新标题")
        self.assertEqual(got.topic, "用户·压力")
        self.assertEqual(got.trigger_class, "关系冲突")
        self.assertEqual(got.text, "旧摘要", "没提的字段不动")

        bad = self.store.set_scene_fields(s.id, {"valence": 1})
        self.assertFalse(bad["ok"], "情绪/强度是算出来的信号——人改它 = 伪造证据")
        self.assertIn("不能改", bad["detail"])

        bad2 = self.store.set_scene_fields(s.id, {"trigger_class": "随便写的"})
        self.assertFalse(bad2["ok"], "情境类是统计的封闭枚举")
        self.assertIn("情境类", bad2["detail"])

        bad3 = self.store.set_scene_fields(s.id, {"time_event": "上个月"})
        self.assertFalse(bad3["ok"], "相对说法算不出日期（同 memo 那条纪律）")
        self.assertIn("具体日期", bad3["detail"])

    def test_scene_field_edit_leaves_a_trace(self):
        """改字段要留痕（改前的值不可再生）——多字段一次改就一次写多条。"""
        s = Scene(title="旧", text="x", topic="用户·工作")
        self.store.add_scene(s)

        out = update_scene_confirmed(self.store, s.id, title="新")

        self.assertTrue(out["ok"] and out["changed"])
        text = self._traces("场景改动")
        self.assertIn("标题", text)
        self.assertIn("旧", text)

    def test_summary_delete_frees_its_scenes(self):
        """S2 可删：删行；它收的 S1 变回"未被覆盖"（下次提炼重聚）。

        2026-09-24 起不再手写 compose 边（那条边本来就是对 `Summary.sources`
        的冗余）——"素材回到未覆盖"由 sources 现算，这条测试因此更干净。
        """
        s2 = self._summary()
        s = Scene(title="甲", text="x", topic=s2.topic)
        self.store.add_scene(s)

        out = delete_summary_confirmed(self.store, s2.id)

        self.assertTrue(out["ok"])
        self.assertIsNone(self.store.get_summary(s2.id))
        from core.distill import _fresh_scenes
        self.assertIn(s.id, [x.id for x in _fresh_scenes(self.store, s2.topic)],
                      "素材回到「未被覆盖」——删了重聚就是它的「改」")

    def test_summary_archive_is_reversible(self):
        """S2 可归档：不召回但数据在；取消归档回热层。"""
        s2 = self._summary()
        self.assertTrue(archive_summary_confirmed(self.store, s2.id)["changed"])
        self.assertNotIn(s2.id, [x.id for x in self.store.hot_summaries(n=50)])
        self.assertIsNotNone(self.store.get_summary(s2.id), "归档不是删——记录还在")
        self.assertTrue(unarchive_summary_confirmed(self.store, s2.id)["changed"])
        self.assertIn(s2.id, [x.id for x in self.store.hot_summaries(n=50)])

    def test_profile_edit_goes_through_revision(self):
        """S3 可改：走修正——旧版进历史（`by="user"`）、新版回 pending、同 topic 串联。"""
        s, p = self._scene_with_profile()
        out = update_profile_confirmed(self.store, p.id, "遇到批评会先扛一下再决定")

        self.assertTrue(out["ok"] and out["changed"])
        old = self.store.get_profile(p.id)
        self.assertEqual(old.invalidated_by, "user", "是用户改的，不是 air 自己改的")
        new = self.store.get_profile(out["new_id"])
        self.assertIsNotNone(new)
        self.assertEqual(new.status, "pending", "新版要重新攒印证")
        self.assertEqual(new.topic, old.topic, "同一个 topic 的下一个版本")
        self.assertIn(s.id, new.sources, "证据沿用现有的——人改的是「说法」不是「依据」")

    def test_profile_archive_and_back(self):
        """S3 可归档：不召回、可取消；数据全在。"""
        _s, p = self._scene_with_profile()
        self.assertTrue(archive_profile_confirmed(self.store, p.id)["changed"])
        self.assertNotIn(p.id, [x.id for x in self.store.current_profiles(status=None)])
        self.assertIsNotNone(self.store.get_profile(p.id), "数据全在")
        self.assertTrue(unarchive_profile_confirmed(self.store, p.id)["changed"])
        self.assertIn(p.id, [x.id for x in self.store.current_profiles(status=None)])

    def test_delete_removes_the_row_and_keeps_the_record(self):
        """删 = **删一行**（2026-09-24 存储层稿 §五）：**不摘引用**。

        `sources` 是"当时凭什么"的**记录**（不可重算），所以不靠删端同步
        （那种同步必然漏——S2 就漏过）；引用改在**读取端过滤**：
        `sources ∩ 现存节点`（`store.exists`）。呈现端见 phase3 的镜像页用例。
        """
        s, p = self._scene_with_profile()
        out = delete_scene_confirmed(self.store, s.id)

        self.assertTrue(out["ok"])
        self.assertIsNone(self.store.get_scene(s.id))
        self.assertIn(s.id, self.store.get_profile(p.id).sources,
                      "记录原样保留——引用是派生的，读的时候过滤")
        self.assertFalse(self.store.exists(s.id), "读端过滤依据这个判")

    def test_delete_leaves_no_trace_and_no_snapshot(self):
        """删除**不留痕、不留快照**（2026-09-23 定）。

        留后路是确认条上的另一档（归档冷存）——删 = 真删：那个"痕"
        （标题 + 正文）对删除没有独立理由，而"察觉 LLM 行为变化再回查 trace"
        这个触发链现实中不成立。唯一残留是每日快照（`backup_daily` 的整库灾备，
        7 天自动清理）——那不是删除留下的，也不该由删除去动。
        """
        s, _p = self._scene_with_profile()
        delete_scene_confirmed(self.store, s.id)

        self.assertEqual(self._traces("场景删除"), "", "删除不再留痕")
        backups = Path(cfgmod.abspath(cfgmod.PATHS["backup_dir"]))
        self.assertEqual(list(backups.glob("before-delete-*.db")), [],
                         "删除不再存快照")

    def test_delete_many_no_snapshot(self):
        """删多条同样不留快照；"要么全删、要么不删"那道关不变。"""
        a = Scene(title="甲", text="甲的理解"); self.store.add_scene(a)
        b = Scene(title="乙", text="乙的理解"); self.store.add_scene(b)

        out = delete_many_confirmed(self.store, [a.id, b.id])

        self.assertTrue(out["ok"])
        self.assertEqual(out["ids"], [a.id, b.id])
        self.assertIsNone(self.store.get_scene(a.id))
        self.assertIsNone(self.store.get_scene(b.id))
        backups = Path(cfgmod.abspath(cfgmod.PATHS["backup_dir"]))
        self.assertEqual(list(backups.glob("before-delete-*.db")), [],
                         "不留快照——要留后路走归档那一档")

    def test_delete_many_is_all_or_nothing(self):
        """有一个编号找不到，就**一条都不删**。

        删除不可撤回，宁可让他回去确认编号，也不要删掉一部分才发现填错了。
        """
        a = Scene(title="甲", text="甲的理解"); self.store.add_scene(a)

        out = delete_many_confirmed(self.store, [a.id, "S1-9999"])

        self.assertFalse(out["ok"])
        self.assertIsNotNone(self.store.get_scene(a.id), "不能删掉一部分")
        self.assertIn("S1-9999", out["detail"], "要说清是哪个编号找不到")

    def test_delete_missing_scene_is_safe(self):
        out = delete_scene_confirmed(self.store, "S1-9999")
        self.assertFalse(out["ok"])


class FailureClassificationTest(unittest.TestCase):
    """模型没回内容时，得说得出**是哪一种**没回。

    实测发生过：拖进一篇 6 千字的设计稿，模型超时没回，
    界面上说的是「检查设置里的 LLM 配置」——而配置好好的。
    **分不清失败的种类，就只能给一句谁都不得罪的废话。**
    """

    def _http(self, code: int, body: bytes = b"") -> Exception:
        return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body))

    def test_timeout(self):
        self.assertEqual(_classify_error(socket.timeout()), "timeout")

    def test_urlerror_wrapping_timeout(self):
        """urlopen 的超时是 URLError，真正的超时藏在 `reason` 里。"""
        self.assertEqual(_classify_error(urllib.error.URLError(socket.timeout())), "timeout")

    def test_auth_and_rate(self):
        self.assertEqual(_classify_error(self._http(401)), "auth")
        self.assertEqual(_classify_error(self._http(429)), "rate")

    def test_too_long(self):
        self.assertEqual(
            _classify_error(self._http(400, b'{"message":"maximum context length"}')),
            "too_long")

    def test_other_http_still_reports_its_code(self):
        self.assertEqual(_classify_error(self._http(500)), "http_500")

    def test_hints_differ_by_kind(self):
        """不同种类的失败说不同的话——不然等于没分。"""
        self.assertIn("太长", _no_reply_hint("timeout"))
        self.assertIn("长度", _no_reply_hint("too_long"))
        self.assertIn("key", _no_reply_hint("auth"))
        self.assertNotEqual(_no_reply_hint("timeout"), _no_reply_hint("auth"))
        # 认不出来的失败退回原来那句兜底，别硬编一个
        self.assertIn("配置", _no_reply_hint(""))


class ServiceStatusTest(unittest.TestCase):
    """服务状态灯（2026-09-25）：**判据是真实调用记录**，不是"配了没有"。

    背景：向量服务死了两天，顶栏的勾一直亮着——因为那个勾看的是配置。
    这里守几件事：成功要记、失败要记（带机器码）、后成功能翻回绿、
    `_service_status` 的四态（未配 / 未测试 / 成 / 败）。
    """

    def _llm(self) -> LLM:
        llm = LLM()
        llm.endpoint, llm.api_key, llm.model = "http://x", "k", "m"
        return llm

    def _fake_urlopen(self, exc=None, payload=None) -> None:
        class Resp:
            def __init__(self, data):
                self._b = json.dumps(data).encode("utf-8")

            def read(self):
                return self._b

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(req, timeout=None):
            if exc is not None:
                raise exc
            return Resp(payload or {"choices": [{"message": {"content": "好"}}]})

        orig = urllib.request.urlopen
        urllib.request.urlopen = fake
        self.addCleanup(lambda: setattr(urllib.request, "urlopen", orig))

    def test_llm_success_is_recorded(self):
        self._fake_urlopen()
        llm = self._llm()
        self.assertEqual(llm.chat([{"role": "user", "content": "hi"}]), "好")
        self.assertTrue(llm.last_ok_at)
        self.assertEqual(llm.last_err_at, "")

    def test_llm_network_failure_records_kind(self):
        self._fake_urlopen(exc=urllib.error.URLError(OSError("boom")))
        llm = self._llm()
        self.assertEqual(llm.chat([{"role": "user", "content": "hi"}]), "")
        self.assertEqual(llm.last_error, "network")
        self.assertTrue(llm.last_err_at)
        self.assertEqual(llm.last_ok_at, "")

    def test_llm_auth_failure_records_kind(self):
        """401 记成 `auth`——"key 不对"和"连不上"要能分开说。"""
        self._fake_urlopen(exc=urllib.error.HTTPError(
            "http://x", 401, "no", {}, None))
        llm = self._llm()
        llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(llm.last_error, "auth")

    def test_llm_recovering_after_failure(self):
        """一次成功把灯翻回绿：`last_ok_at` 与 `last_err_at` 谁晚谁说了算。"""
        self._fake_urlopen(exc=urllib.error.URLError(OSError("boom")))
        llm = self._llm()
        llm.chat([{"role": "user", "content": "hi"}])
        self.assertFalse(_service_status(llm.available(), llm)["ok"])
        self._fake_urlopen()                    # 换成成功
        llm.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(_service_status(llm.available(), llm)["ok"])

    def test_service_status_three_states(self):
        class Svc:
            last_ok_at = ""
            last_err_at = ""
            last_error = ""

        svc = Svc()
        st = _service_status(False, None)       # 没配（连对象都没有）
        self.assertFalse(st["available"])
        self.assertIsNone(st["ok"], "没配不等于失败")
        self.assertIsNone(_service_status(True, svc)["ok"], "配了没调用过 = 未测试")
        svc.last_ok_at = "2026-09-25 10:00:00"
        self.assertTrue(_service_status(True, svc)["ok"])
        svc.last_err_at = "2026-09-25 11:00:00"
        svc.last_error = "network"
        out = _service_status(True, svc)
        self.assertFalse(out["ok"], "最近一次是失败")
        self.assertEqual(out["last_error"], "network")

    def test_embedding_failure_records_state(self):
        """向量服务失败：降级标志 + 时间 + 原文（连不上那句就写在里面）。"""
        svc = EmbeddingService(endpoint="http://x/v1", api_key="k", model="m")
        orig = netmod.open
        netmod.open = lambda *a, **kw: (_ for _ in ()).throw(
            urllib.error.URLError(OSError("connection refused")))
        self.addCleanup(lambda: setattr(netmod, "open", orig))
        self.assertIsNone(svc.embed_one("x"))
        self.assertTrue(svc.degraded)
        self.assertTrue(svc.last_err_at)
        self.assertIn("refused", svc.last_error)

    def test_embedding_success_clears_degraded(self):
        svc = EmbeddingService(endpoint="http://x/v1", api_key="k", model="m")
        svc.degraded = True                      # 曾经失败过

        class Resp:
            def read(self):
                return json.dumps({"data": [{"embedding": [0.1, 0.2]}]}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        orig = netmod.open
        netmod.open = lambda *a, **kw: Resp()
        self.addCleanup(lambda: setattr(netmod, "open", orig))
        self.assertEqual(svc.embed_one("x"), [0.1, 0.2])
        self.assertFalse(svc.degraded, "成功要把降级标志清掉")
        self.assertTrue(svc.last_ok_at)


class ChatStreamTest(unittest.TestCase):
    """流式：**让她想的时候你能看见**。

    复杂问题就是要想很久——干等的问题不该靠掐断解决，该靠把过程显示出来解决。
    （所以这条链路和"超时"是两件事：超时只管卡死，不管想多久。）
    """

    def _llm_with_stream(self, lines: list[str]) -> LLM:
        """假 SSE 流：把 `urlopen` 换成一串 bytes 行。"""
        class Resp:
            def __init__(self, data): self._it = iter(data)
            def __iter__(self): return self
            def __next__(self): return next(self._it)

        llm = LLM()
        llm.endpoint, llm.api_key, llm.model = "http://x", "k", "m"
        orig = urllib.request.urlopen
        urllib.request.urlopen = lambda req, timeout=None: Resp(
            [l.encode("utf-8") for l in lines])
        self.addCleanup(lambda: setattr(urllib.request, "urlopen", orig))
        return llm

    def test_thinking_and_text_are_separate_events(self):
        """思考和正文要分开：一个给你看过程，一个才是她说的话。"""
        llm = self._llm_with_stream([
            'data: {"choices":[{"delta":{"reasoning_content":"先想想"}}]}\n\n',
            'data: {"choices":[{"delta":{"reasoning_content":"再想想"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"答案是"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"42"}}]}\n\n',
            'data: [DONE]\n\n',
        ])
        evs = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual([(e["type"], e["text"]) for e in evs],
                         [("reasoning", "先想想"), ("reasoning", "再想想"),
                          ("content", "答案是"), ("content", "42")])

    def test_tool_calls_are_reassembled_from_fragments(self):
        """工具调用是**分片**来的：名字和参数都得拼起来。"""
        llm = self._llm_with_stream([
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
            '"function":{"name":"memory_","arguments":""}}]}}]}\n\n',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"name":"search","arguments":"{\\"query\\":"}}]}}]}\n\n',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"\\"面试\\"}"}}]}}]}\n\n',
            'data: [DONE]\n\n',
        ])
        evs = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        call = evs[-1]["calls"][0]
        self.assertEqual(call["name"], "memory_search")
        self.assertEqual(call["arguments"], {"query": "面试"})

    def test_tools_unsupported_is_switched_off_here_too(self):
        """服务端不支持工具 → **流式也要整块关掉**（仪表盘默认走的就是流式）。

        以前只有 `chat_with_tools` 会写 `_tools_cap`，`chat_stream` 只读不写，
        于是那句「关掉工具箱」在默认路径上永不生效：每一轮都重发 `tools`
        再吃一次 400——不崩、不报错，只是每轮多一次失败。
        """
        sent = []

        class Resp:
            def __init__(self, data): self._it = iter(data)
            def __iter__(self): return self
            def __next__(self): return next(self._it)

        def fake(req, timeout=None):
            body = json.loads(req.data.decode("utf-8"))
            sent.append("tools" in body)
            if "tools" in body:
                raise urllib.error.HTTPError(
                    "http://x", 400, "err", {"Content-Type": "application/json"},
                    io.BytesIO(b'{"error":{"message":"tools is not supported"}}'))
            # bytes 字面量只能装 ASCII，中文这句得现编码
            return Resp([s.encode("utf-8") for s in (
                'data: {"choices":[{"delta":{"content":"好"}}]}\n\n',
                'data: [DONE]\n\n')])

        llm = LLM()
        llm.endpoint, llm.api_key, llm.model = "http://x", "k", "m"
        orig = urllib.request.urlopen
        urllib.request.urlopen = fake
        self.addCleanup(lambda: setattr(urllib.request, "urlopen", orig))

        evs = list(llm.chat_stream([{"role": "user", "content": "hi"}],
                                   tools=[{"type": "function"}]))
        self.assertEqual(sent, [True, False], "第一次带 tools，重试那次不带")
        self.assertEqual(llm._tools_cap, "unsupported", "能力位要真的落下来")
        self.assertEqual([e["text"] for e in evs if e["type"] == "content"], ["好"],
                         "关掉工具箱之后这一轮仍要正常说话")


class _StreamLLM(LLM):
    """只会吐事件的假模型（流式路径不走 `set_mock`）。"""

    def __init__(self, events: list[dict]):
        super().__init__()
        self.events = events

    def chat_stream(self, messages, tools=None, max_tokens=3000, temperature=None,
                    timeout=None):
        yield from self.events


class UndoTurnTest(Base):
    """改 / 重新生成的前半步：撤掉窗口末尾的 **N 轮**（默认 1 轮）。

    语义是**回到那一句的地方重说**，不是「改一条记录」：被撤掉的这些
    从没进过长期库（窗口就是还没提取的那一段），撤了就等于没说过——
    **不留痕**也由此而来（没有记忆被改动，就没有要审计的东西）。

    `turns` 是 2026-10-07 加的：窗口里**每一轮**都该能改（不只最后一条），
    改第 2 轮 = 把第 2 轮和它后面那些一起撤掉重说。
    """

    def st(self) -> ShortTerm:
        return ShortTerm(self.store, scripted_llm(), emb_service=_NoEmbedding(),
                         session_id="t", state_path=self.root / "st.json")

    def _three_turns(self) -> ShortTerm:
        st = self.st()
        for i in (1, 2, 3):
            st.append("user", f"第{i}句")
            st.append("air", f"第{i}答")
        return st

    def test_undo_removes_the_last_turn_only(self):
        st = self.st()
        st.append("user", "第一句")
        st.append("air", "第一答")
        st.append("user", "第二句")
        st.append("air", "第二答")

        out = st.undo_turns()

        self.assertTrue(out["ok"], out)
        self.assertEqual([m["text"] for m in st.messages], ["第一句", "第一答"],
                         "只撤末尾这一轮；更早的是历史，不碰")
        self.assertEqual(out["text"], "第二句", "撤掉的那句要带出来（界面拿它换掉气泡）")
        self.assertEqual(out["air"], "第二答")

    def test_undo_many_turns_cuts_from_that_turn_on(self):
        """改第 2 轮 = 撤第 2、3 轮（**从那一轮起往后全撤**）——这就是"开新分支"。"""
        st = self._three_turns()

        out = st.undo_turns(2)

        self.assertTrue(out["ok"], out)
        self.assertEqual([m["text"] for m in st.messages], ["第1句", "第1答"],
                         "第 1 轮留着，第 2 轮起全撤")
        self.assertEqual(out["text"], "第2句", "带出的是**被撤那批的第一句**（界面认它）")
        self.assertEqual(out["turns"], 2)

    def test_undo_refuses_when_the_window_has_fewer_turns(self):
        """要撤的比窗口里的多 → 拒绝，**一个字都不撤**（装成功会让界面和后端错开）。"""
        st = self.st()
        st.append("user", "只有一轮")
        st.append("air", "一句答")

        out = st.undo_turns(2)

        self.assertFalse(out["ok"])
        self.assertIn("1 轮", out["detail"])
        self.assertEqual(len(st.messages), 2, "拒绝时窗口必须原样")

    def test_expect_guard_refuses_a_window_that_moved(self):
        """`expect` 对不上就不撤（2026-10-07）：窗口在两次点击之间可能被提取过，
        那时按 N 撤会**撤到别的轮次上**——宁可什么都不做。"""
        st = self._three_turns()

        out = st.undo_turns(2, expect="另一句话")

        self.assertFalse(out["ok"])
        self.assertEqual(len(st.messages), 6, "核对不过时一个字都不撤")

        out2 = st.undo_turns(2, expect="第2句")
        self.assertTrue(out2["ok"], "对得上就照撤")

    def test_undo_survives_reload(self):
        """撤完要**落盘**——窗口是整文件原子写，不落盘的话刷新一下它又回来了。"""
        st = self.st()
        st.append("user", "说错的")
        st.append("air", "她答的")

        st.undo_turns()

        self.assertEqual(self.st().messages, [])

    def test_undo_works_when_she_never_answered(self):
        """她没回也得能撤（那一轮失败时窗口里只有他那一句）——重说正是那时最需要的。"""
        st = self.st()
        st.append("user", "第一句")
        st.append("user", "她没回的那句")

        out = st.undo_turns()

        self.assertTrue(out["ok"])
        self.assertEqual(out["air"], "")
        self.assertEqual([m["text"] for m in st.messages], ["第一句"])

    def test_undo_is_refused_on_an_empty_window(self):
        """窗口空 = 这段已经进长期库了（提取时同一步"写 raws + 清窗口"），撤不了。

        撤不掉就如实说，**装成功会更糟**：界面会以为可以重说，而后端什么都没撤。
        """
        out = self.st().undo_turns()

        self.assertFalse(out["ok"])
        self.assertIn("窗口是空的", out["detail"])

    def test_undo_leaves_no_trace(self):
        """**不留痕**：撤的是还没进长期库的一段，没有记忆被改动，没有"谁改了什么"要回答。

        （这条钉的是语义：哪天这里冒出"对话改动"之类的留痕，说明它又漂回
        "改一条已存的记录"去了——那是另一种设计，需要另一套理由。）
        """
        st = self.st()
        st.append("user", "错字")
        st.append("air", "回话")

        st.undo_turns()

        self.assertEqual(list((self.root / "trace").glob("*.jsonl")), [])


class ReplyStreamTest(Base):
    """**流式不是另一套系统**——唤醒、写入、记账一个都不能少。"""

    def test_end_carries_everything_the_dashboard_needs(self):
        llm = _StreamLLM([{"type": "reasoning", "text": "想一下"},
                          {"type": "content", "text": "嗯，我在。"}])
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding())

        evs = list(sess.reply_stream("你好"))

        self.assertEqual([e["type"] for e in evs][:2], ["reasoning", "content"])
        end = evs[-1]
        self.assertEqual(end["type"], "end")
        self.assertEqual(end["reply"], "嗯，我在。")
        self.assertIn("recall", end, "结束事件要带这一轮唤醒了什么（右侧面板靠它）")

    def test_stream_writes_to_memory_just_like_reply(self):
        """两边共用 `_prepare` 和写入：流式下记忆不能少记。"""
        llm = _StreamLLM([{"type": "content", "text": "记下了"}])
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding())
        list(sess.reply_stream("今天面试没过"))

        # 窗口是持久的（可能带着更早会话的消息），所以只看**本轮这两条**；
        # 存的是 `speaker` / `text`（双主体：air 的话也进缓冲）
        msgs = sess.st.messages
        self.assertEqual([m["speaker"] for m in msgs[-2:]], ["user", "air"])
        self.assertEqual(msgs[-2]["text"], "今天面试没过")
        self.assertEqual(msgs[-1]["text"], "记下了")

    def test_stopping_halfway_keeps_what_she_said(self):
        """**说到一半被叫停**：已经吐出去的那半句要落进**窗口**（2026-10-07）。

        抄的是 AI SDK 的 `chatbot-resume-streams`（停止时保存 assistant 快照），
        但只抄到窗口这一步——**长期库里不留它**（提取时滤掉，见
        `PartialTurnMemoryTest`）：半句没有信息量，留着还会让她以后以为那句说完了。

        落的是**逐字原话**：不许补全成整句，也不许丢。
        """
        llm = _StreamLLM([{"type": "content", "text": "抱这一下，我给不"},
                          {"type": "content", "text": "了。能给的是"},
                          {"type": "content", "text": "听着。"}])
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding(),
                           state_path=self.root / "st.json")

        gen = sess.reply_stream("我有点撑不住")
        next(gen); next(gen)              # 只收两帧，然后按"停止"（断开 = GeneratorExit）
        gen.close()

        msgs = sess.st.messages[-2:]
        self.assertEqual([m["speaker"] for m in msgs], ["user", "air"])
        self.assertEqual(msgs[-2]["text"], "我有点撑不住")
        self.assertEqual(msgs[-1]["text"], "抱这一下，我给不了。能给的是",
                         "半句照原样落：不许补全、也不许丢")
        self.assertTrue(msgs[-1]["interrupted"],
                        "落一个 `interrupted`：界面回填时要标「（已停止）」，"
                        "刷新前后口径一致（正文仍是逐字原话）")
        self.assertNotIn("interrupted", msgs[-2], "被叫停的是她，不是他")

    def test_stopping_before_she_says_anything(self):
        """一个字都没吐出来就停：窗口里只留他那句——她没说过的，不许替她说。"""
        llm = _StreamLLM([{"type": "reasoning", "text": "想一下"},
                          {"type": "content", "text": "嗯，我在。"}])
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding(),
                           state_path=self.root / "st.json")

        gen = sess.reply_stream("在吗")
        next(gen)                         # 只拿到 thinking，还没出正文
        gen.close()

        self.assertEqual([m["speaker"] for m in sess.st.messages],
                         ["user"], "只有他那句（air 那条不落）")

    def test_normal_end_does_not_double_write(self):
        """正常跑完的那一轮**不能**被那条中断路径再写一遍（`end` 之后 close 是空操作）。"""
        llm = _StreamLLM([{"type": "content", "text": "嗯，我在。"}])
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding(),
                           state_path=self.root / "st.json")

        gen = sess.reply_stream("你好")
        for _ in gen:
            pass
        gen.close()                       # 已经跑完的生成器：close 必须什么都不做

        self.assertEqual(len(sess.st.messages), 2, "正文只落一次")
        self.assertEqual(sess.st.messages[-1]["text"], "嗯，我在。")

    def test_truncated_flag_reaches_the_end_event(self):
        """被额度掐断时 end 事件要带 `truncated`——**正文非空时也要**。

        原来的失败提示只在"一个字都没回"时出现，于是"说了一半被截断"
        看起来像正常结束（真出过）。
        """
        llm = _StreamLLM([{"type": "content", "text": "我正想说的是"},
                          {"type": "truncated"}])
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding())

        evs = list(sess.reply_stream("你在吗"))

        self.assertTrue(evs[-1]["truncated"])
        self.assertEqual(evs[-1]["reply"], "我正想说的是")

    def test_failure_hint_does_not_enter_memory(self):
        """失败提示走 `hint`，**不冒充她的正文、也不进记忆**。

        以前它被塞进 `reply`：显示上顶在"她的话"的位置（明明不是），
        还会被 `st.append("air", reply)` 写进窗口——下一轮她会"记得自己说过
        「检查 LLM 配置」"，提取时这句还会被当成对话内容。
        """
        llm = _StreamLLM([{"type": "error", "error": "timeout"}])
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding())

        evs = list(sess.reply_stream("在吗"))

        end = evs[-1]
        self.assertEqual(end["reply"], "", "失败不该伪造她的正文")
        self.assertIn("想得太久", end["hint"], "失败提示走 hint 字段（人话）")
        self.assertEqual(sess.st.messages[-1]["speaker"], "user",
                         "窗口里不该留下一条 air 的失败提示（只写了用户那句）")


class PartialTurnMemoryTest(Base):
    """「说到一半被叫停」的那半句：**窗口里留着，长期库里不留**（2026-10-07 定）。

    短期留着有用——刷新不丢、能重新生成、下一轮她知道"自己说到一半"；
    但长期记忆里留一句被掐断的话没有信息量，还会让她以后以为那句是说完了的。
    所以提取那一步把它滤掉：不进原文、不进场景卡。
    """

    def _st(self, worth: bool = True) -> ShortTerm:
        return ShortTerm(self.store, scripted_llm(worth=worth), emb_service=_NoEmbedding(),
                         session_id="t", state_path=self.root / "st.json")

    def test_the_half_sentence_is_kept_in_the_window(self):
        """窗口里**留着**（这正是它和「当没说过」的区别）。"""
        st = self._st()
        st.append("user", "我有点撑不住")
        st.append("air", "抱这一下，我给不", interrupted=True)

        self.assertEqual([m["speaker"] for m in st.messages], ["user", "air"])
        self.assertIn("我给不", st.messages[-1]["text"])

    def test_it_never_reaches_the_raw_text(self):
        """提取时滤掉：**原文里没有它**（那是长期库的第一层）。"""
        st = self._st()
        st.append("user", "我有点撑不住")
        st.append("air", "抱这一下，我给不", interrupted=True)

        out = st.compress_and_extract()

        self.assertTrue(out and out.get("scene_id"), "他那句该照常落库")
        files = list((self.root / "raws").glob("*.md"))
        self.assertTrue(files, "原文该写下来")
        text = files[0].read_text(encoding="utf-8")
        self.assertIn("我有点撑不住", text)
        self.assertNotIn("我给不", text, "被叫停的半句不许进原文")

    def test_other_messages_are_unaffected(self):
        """滤的是**那一条**，不是这一段——同一段里她完整说过的话照样落。"""
        st = self._st()
        st.append("user", "我有点撑不住")
        st.append("air", "先说一句：我在。", interrupted=True)
        st.append("user", "嗯")
        st.append("air", "那就先吃东西。")

        st.compress_and_extract()

        files = list((self.root / "raws").glob("*.md"))
        text = files[0].read_text(encoding="utf-8")
        self.assertIn("那就先吃东西。", text, "完整的那句要落")
        self.assertNotIn("先说一句", text, "被叫停的那条不落")


class RecallViewTest(unittest.TestCase):
    """唤醒结果的「前端视图」必须**可 JSON 序列化**——这里断过一条线。

    `reply_stream` 的 end 事件里 `recall` 是唤醒引擎的原始结果，
    装着 `Scene` / `Profile` / `Summary` / `Raw` 这些 dataclass 对象；
    SSE 那边 `json.dumps` 遇到它们直接抛 `TypeError`——
    表现不是"少个字段"，是**整条流在结尾处断掉**：
    右侧线索面板不更新、「提取了一段对话」的回执不出现、确认条弹不出来。
    转换落在 `dashboard._recall_view` / `_sse_safe`，这条盯住它。
    """

    def _payload(self) -> dict:
        return {
            "scenes": [Scene(id="S1-0001", title="面试没过", topic="用户·面试",
                             text="面试没过")],
            "profiles": [Profile(id="S3-0001", topic="用户·面试", statement="面试会紧张",
                                 status="established", evidence=3)],
            "summaries": [Summary(id="S2-0001", topic="用户·面试", text="聊过几次")],
            "raws": [Raw(scene_id="S1-0001", content="用户: 面试没过",
                         created_at="2026-09-13 09:30:00")],
            "standing_memos": [], "suppressed": [], "suppressed_detail": [],
            "flags": {}, "actions": {}, "cues_view": {"C1": 0.5},
            "why": {"S1-0001": "R1 语义命中"},
            "budget": {"budget": 8000, "before": 9000, "after": 7000,
                       "dropped": ["场景 S1-0002"], "alone": False},
        }

    def test_end_event_is_json_serializable(self):
        """end 事件不抛 TypeError = SSE 能发到结尾（转换过的才是能发的形状）。"""
        ev = _sse_safe({"type": "end", "reply": "嗯", "recall": self._payload()})
        text = json.dumps(ev, ensure_ascii=False)
        self.assertIn("S1-0001", text)

    def test_raw_uses_scene_id(self):
        """`Raw` 没有 id——视图里必须用 scene_id 兜（用 `r.id` 会直接 AttributeError）。"""
        view = _recall_view(self._payload())
        self.assertEqual(view["raws"][0]["id"], "S1-0001")
        self.assertIn("面试没过", view["raws"][0]["content"])

    def test_budget_report_survives(self):
        """「裁掉了什么」要带到前端——它是「应该出现的东西没出现」的答案。"""
        view = _recall_view(self._payload())
        self.assertEqual(view["budget"]["budget"], 8000)
        self.assertEqual(view["budget"]["dropped"], ["场景 S1-0002"])

    def test_non_end_events_pass_through(self):
        ev = {"type": "content", "text": "hi"}
        self.assertEqual(_sse_safe(ev), ev)


class FrameConventionTest(unittest.TestCase):
    """音频帧格式：**一份约定、两处实现**——改一处必须改另一处。

    仓库里可校验的一处是 `dashboard._pcm_stream`（把帧转发给页面）；生成端
    （`tts/server.py` 的 `_speak_stream`）**不进库**（只留 `tts/README.md` 那份契约），
    本机有它时一并校验；解码那半边在 `web/index.html`（搜 `ARAU`）。
    这个约定没有类型系统保得住，而断了的表现只是"没声音"——所以钉一条测试：
    两边必须写着同一个魔数与同一套 struct 格式。
    """

    def test_both_sides_share_the_same_header_and_frames(self):
        root = Path(__file__).resolve().parent.parent
        sides = [(root / "core" / "dashboard.py", "core/dashboard.py")]
        # 生成端（tts/server.py）**不进库**（仓库里只留 tts/README.md 那份契约）：
        # 本机有它就跟仓库里这一处一起校验，公开副本上只剩转发端这一处。
        if (root / "tts" / "server.py").exists():
            sides.append((root / "tts" / "server.py", "tts/server.py"))
        for path, where in sides:
            src = path.read_text(encoding="utf-8")
            self.assertIn('b"ARAU"', src, f"{where}: 12 字节头的魔数要一致")
            self.assertIn('"<4sIHH"', src, f"{where}: 头部的打包格式要一致")
            self.assertIn('"<i"', src, f"{where}: 帧头（样本数 i32）的打包格式要一致")


# ---------------------------------------------------------------------


class _FakeItem:
    """只有 `id` 的占位——裁掉的东西要**报得出名字**，有名字就够。"""

    def __init__(self, tag: str):
        self.id = tag


def _fake_measure(recall, blocks, item=100, block=300, fixed=500):
    """一个**可算得清**的量法：每条 100 token、每个窗口块 300、固定部分 500。

    用它而不是真的去渲染提示词：顺序是这里要测的东西，
    掺进真实长度之后，断言就变成了"碰巧是多少"。
    """

    def _m() -> int:
        # 键名跟着 `chat._TRIM_ORDER` 走（2026-10-05：`memo_candidates` → `standing_memos`）
        n = sum(len(recall.get(k) or []) for k in
                ("scenes", "profiles", "summaries", "standing_memos", "raws"))
        return fixed + n * item + len(blocks) * block

    return _m


class ContextBudgetTest(Base):
    """上下文总预算的仲裁——`context.total_budget` 以前**没有任何代码读它**。

    这三条盯的是同一件事：**分块上限加起来没有上限**。
    库一大，注入总量跟着长，长到某处就不再是「记得多」，是稀释注意力
    （真正要紧的那条被淹没在一堆次要材料里）。
    """

    def _recall(self, n_scenes=3, n_profiles=2, n_summaries=1, n_memos=1, n_raws=1):
        return {
            "scenes": [_FakeItem(f"S1-{i:04d}") for i in range(n_scenes)],
            "profiles": [_FakeItem(f"P{i}") for i in range(n_profiles)],
            "summaries": [_FakeItem(f"S2-{i}") for i in range(n_summaries)],
            # 键是 `standing_memos`（2026-10-05 晚并栏：候选 `memo_candidates` 已删）；
            # 第一条带 `due`——到点那件才是"这一轮真给过机会"的那一条
            "standing_memos": [{"id": f"M{i}", "content": "面试结果",
                                "due": i == 0} for i in range(n_memos)],
            "raws": [_FakeItem(f"R{i}") for i in range(n_raws)],
        }

    def test_trims_by_priority(self):
        """先裁场景，再画像 / 摘要 / 备忘录 / 原话，**最后才是窗口**。"""
        recall = self._recall()
        blocks = [("更早", "更早的内容"), ("最近", "最近对话")]
        report = fit_context(recall, blocks, _fake_measure(recall, blocks),
                             budget=1000)

        self.assertEqual([d.split()[0].split("「")[0] for d in report["dropped"]],
                         ["场景", "场景", "场景", "画像", "画像", "摘要",
                          "备忘录", "原话", "窗口"],
                         "顺序来自设计稿：召回内部 原文 > 摘要 > 画像 > 场景")
        self.assertEqual([b[0] for b in blocks], ["最近"],
                         "最近那几句逐字是保真下限，最后才丢")
        self.assertEqual(report["before"], 1900)
        self.assertEqual(report["after"], 800)
        self.assertFalse(report["alone"])

    def test_nothing_trimmed_within_budget(self):
        recall = self._recall(1, 0, 0, 0, 0)
        blocks = [("最近", "最近对话")]
        report = fit_context(recall, blocks, _fake_measure(recall, blocks),
                             budget=_fake_measure(recall, blocks)())

        self.assertEqual(report["dropped"], [])
        self.assertEqual(len(recall["scenes"]), 1, "没超预算就不该动它")

    def test_zero_budget_disables_arbitration(self):
        """`0` = 不仲裁。这个数还没标定过，得留一个能关的口子。"""
        recall = self._recall()
        blocks = [("更早", "x"), ("最近", "y")]
        report = fit_context(recall, blocks, _fake_measure(recall, blocks), budget=0)

        self.assertEqual(report["budget"], 0)
        self.assertEqual(len(recall["scenes"]), 3)

    def test_message_alone_over_budget_is_reported(self):
        """当前消息自己就超（拖进一篇两万字）→ 全裁光、**如实报**，不许静默。"""
        recall = self._recall()
        blocks = [("更早", "x"), ("最近", "y")]
        report = fit_context(recall, blocks, _fake_measure(recall, blocks),
                             budget=100)

        self.assertTrue(report["alone"])
        self.assertEqual(blocks, [], "可裁的都裁了")
        self.assertEqual(report["after"], 500, "剩下的全是当前消息那一份")

    def test_prepare_reads_the_budget(self):
        """`_prepare` 真的读了 `context.total_budget`（这就是那根断掉的线）。"""
        old = cfgmod.CONFIG["context"]["total_budget"]
        self.addCleanup(lambda: cfgmod.CONFIG["context"].update(
            {"total_budget": old}))
        sess = self.session()
        for i in range(6):                       # 让窗口有「更早 + 最近」两块
            sess.st.append("user", f"早些时候说的第 {i} 句，内容足够长一点")

        payload = {
            "scenes": [Scene(title="面试没过", text="面试没过",
                             time_event="2026-09-10 10:00:00")],
            "profiles": [Profile(topic="用户·面试", statement="面试会紧张",
                                 status="established", evidence=3)],
            "summaries": [Summary(topic="面试", text="面试这件事聊过好几次")],
            "raws": [Raw(content="原话" * 50, scene_id="S1-0001")],
            "standing_memos": [], "flags": {}, "cues_view": {},
        }
        with patch("core.chat.recall_for_message", return_value=payload):
            cfgmod.CONFIG["context"]["total_budget"] = 0
            _, wide, _, _ = sess._prepare("在吗")
            cfgmod.CONFIG["context"]["total_budget"] = 100
            recall, narrow, _, _ = sess._prepare("在吗")

        self.assertEqual(recall["budget"]["budget"], 100)
        self.assertTrue(recall["budget"]["dropped"], "预算这么小，必裁")
        self.assertIn("面试没过", wide)
        self.assertNotIn("面试没过", narrow,
                         "裁掉之后，系统提示词里不该还有它（否则等于没裁）")


class WebScriptSanityTest(unittest.TestCase):
    """`web/index.html` 的顶层声明不许重名——`const` 重名是**语法错误，整页脚本全挂**。

    2026-09-19 真撞过一次：加消息正文格式化时用了 `fmt`，撞上面板里已有的
    数字格式化 `fmt`——页面全僵（按钮全失效），而 python 测试全绿
    （html 的 lint 不做 JS 语法检查）。这道守卫就是那次事故的落点。
    """
    def test_no_duplicate_top_level_declarations(self):
        import re
        from collections import Counter
        src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        decls = re.findall(r"(?m)^(?:const|let|function)\s+([A-Za-z_$][\w$]*)", src)
        dup = [k for k, n in Counter(decls).items() if n > 1]
        self.assertEqual(dup, [], f"顶层重复声明（JS 会整页挂）：{dup}")

    def test_dollar_ids_are_defined(self):
        """`$("id")` 引用的元素必须真的存在（2026-09-21 加）。

        这类漂移是**静默的**：引用一个不存在的 id，平时看不出来，
        只有点到那条路时才炸（同"顶层重名"的性质——python 测试全绿、页面僵住）。
        id 可以从两处来：HTML 静态的，和 JS 动态设的（如 `memoSec`）。
        """
        import re
        src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        defined = set(re.findall(r'id="([^"]+)"', src))
        defined |= set(re.findall(r'\.id\s*=\s*"([^"]+)"', src))
        refs = set(re.findall(r'\$\("([^"]+)"\)', src))
        self.assertEqual(sorted(r for r in refs if r not in defined), [],
                         "引用了不存在的元素 id")

    def test_onclick_functions_are_defined(self):
        """`onclick="fn(…)"` 里的函数必须有 function 声明（2026-09-21 加）。

        拼错函数名 = 点了没反应，而且只在那一个入口上没反应——最容易被漏掉的一类。
        """
        import re
        src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        called = set(re.findall(r'onclick="([A-Za-z_$][\w$]*)\s*\(', src))
        defined = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", src))
        self.assertEqual(sorted(f for f in called if f not in defined), [],
                         "onclick 指向了没有声明的函数")

    def test_key_inputs_are_password_type(self):
        """key 输入框必须是 `type="password"`（2026-09-21 加）。

        `describe()` 给的是打码值（placeholder 用），但**输入框本身**若不遮挡，
        贴进去的真 key 就明晃晃摆在屏幕上——截图 / 旁人一眼抄走。改回明文要红。
        """
        import re
        src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        for i in ("llm_key", "emb_key"):
            m = re.search(rf'<input[^>]*id="{i}"[^>]*>', src)
            self.assertIsNotNone(m, f"找不到输入框 {i}")
            self.assertIn('type="password"', m.group(0),
                          f"{i} 必须是密码框（输入时自动打码）")
