"""打捞：**人主动从原文里找回一件东西**（存储层三级追溯的最后一级）。

它和每轮唤醒的下钻是两条路，别混：
  - 下钻只能经 `scene_id`——原文没有索引，「S0 最难召回」是物理实现的（有测试盯着）
  - 打捞是**人显式发起**的翻找，慢，但找回东西靠它

所以这里最有价值的一条断言是：**删掉一条之后，原文仍然找得到**。
"""
# 用例分组：
#   脚手架  Base（临时库 + 一份按天的原文文档）
#   打捞    SectionTest 分节解析 · RecentDialogueTest 重启回填 ·
#           SalvageSearchTest 按日期 + 语义找 · RebuildTest 照原文重记（不是撤销）
#           + **未了结的事跟着回来**（认领旧的 / 认不到才新记，2026-10-05）
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core.llm import LLM
from core.model import Raw, Scene
from core.salvage import _sections, rebuild_confirmed, recent_dialogue, search_raws
from core.store import Store


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = Store(self.root / "t.db")
        self._old = {k: cfgmod.PATHS[k]
                     for k in ("trace_dir", "raws_dir", "backup_dir",
                               "shortterm", "personas")}
        cfgmod.PATHS["trace_dir"] = str(self.root / "trace")
        cfgmod.PATHS["raws_dir"] = str(self.root / "raws")
        cfgmod.PATHS["backup_dir"] = str(self.root / "backups")
        # 窗口也要重定向：直接建 `ChatSession`（没传 state_path）的测试
        # 默认会落到真实窗口文件——漏了它，跑一次测试就往真窗口里塞几条消息
        cfgmod.PATHS["shortterm"] = str(self.root / "shortterm.json")
        # 人格目录也重定向：`_split_dialogue` 靠扫它认行首的说话人——
        # 建两个假的就够，别去碰开发机上真的人格文件
        pdir = self.root / "personas"
        pdir.mkdir()
        for n in ("air", "mia"):
            (pdir / f"{n}.md").write_text(f"# {n}\n", encoding="utf-8")
        cfgmod.PATHS["personas"] = str(pdir)

    def tearDown(self):
        for k, v in self._old.items():
            cfgmod.PATHS[k] = v
        self.store.close()
        self._tmp.cleanup()

    def _add(self, sid: str, content: str, day: str = "2026-09-13") -> Scene:
        s = Scene(id=sid, title="标题", text="理解", time_record=f"{day} 08:00:00")
        self.store.add_scene(s)
        self.store.add_raw(Raw(scene_id=sid, content=content), on_date=day)
        return s


class SectionTest(unittest.TestCase):
    def test_splits_a_day_into_scenes(self):
        text = ("# 原文\n\n## S1-0001 · 2026-09-13 08:00:00\n\n用户: 甲\nair: 乙\n\n"
                "## S1-0002 · 2026-09-13 09:00:00\n\n用户: 丙\n")
        got = _sections(text)
        self.assertEqual([g[0] for g in got], ["S1-0001", "S1-0002"])
        self.assertEqual(got[0][1], "2026-09-13 08:00:00")
        self.assertIn("甲", got[0][2])
        self.assertNotIn("丙", got[0][2], "节之间不能串")

    def test_pasted_headings_are_not_sections(self):
        """正文里粘贴的 `## 小标题` **不是**切分点（2026-10-07 事故）。

        他会把整份 README / 人格设定贴进对话，里面自带 `## 跑起来` 这类行。
        按「行首 `## `」一律切的话，一段对话会被切成七八节：回填（最近 8 段）
        里塞满「跑起来」「目录」这种空段，真正的 S1-0043 只剩开头三句——
        看着就是"最近两条对话没了"。只有 `## S1-xxxx · 时间` 才是标题。
        """
        text = ("## S1-0043 · 2026-10-07 02:59:57\n\n"
                "用户: 给你看看你自己\n"
                "## 核心机制\n"
                "## 跑起来\n"
                "## 许可\n"
                "xina: 我读完了\n\n"
                "## S1-0044 · 2026-10-07 03:04:03\n\n用户: 记忆肯定不会留的\n")
        got = _sections(text)
        self.assertEqual([g[0] for g in got], ["S1-0043", "S1-0044"],
                         "粘贴的小标题不许变成一节")
        self.assertIn("## 核心机制", got[0][2], "它是这一节的正文，不是另一节")
        self.assertIn("我读完了", got[0][2], "粘贴之后她的话还在同一节里")
        self.assertEqual(got[1][1], "2026-10-07 03:04:03")


class RecentDialogueTest(Base):
    """对话区回填：最近几段**已提取的原文**（"重启后记录没了"的修复点）。

    它和窗口（`shortterm.json`）互补：提取时同一步"写这里 + 清窗口"，
    两边拼起来正好是"刚才聊到哪了"。
    """

    def _add_at(self, sid: str, content: str, when: str) -> None:
        s = Scene(id=sid, title="标题", text="理解", time_record=when)
        self.store.add_scene(s)
        self.store.add_raw(Raw(scene_id=sid, content=content, created_at=when),
                           on_date=when[:10])

    def test_splits_dialogue_into_messages(self):
        self._add_at("S1-0001", "用户: 甲\nair: 乙\n\n乙的续行", "2026-09-15 08:00:00")
        out = recent_dialogue(self.store)
        self.assertEqual(len(out), 1)
        msgs = out[0]["messages"]
        self.assertEqual([m["speaker"] for m in msgs], ["user", "air"])
        self.assertIn("续行", msgs[1]["text"], "续行要并进上一条，不能丢")

    def test_splits_persona_names(self):
        """原文按**当时的真名**署名（2026-09-21）：人格名认得出，旧 "air" 照样认。"""
        self._add_at("S1-0001",
                     "用户: 甲\nair: 乙\nmia: 丙\n丙的续行", "2026-09-15 08:00:00")
        msgs = recent_dialogue(self.store)[0]["messages"]
        self.assertEqual([m["speaker"] for m in msgs], ["user", "air", "air"])
        self.assertEqual([m.get("persona") for m in msgs], [None, "air", "mia"])
        self.assertIn("续行", msgs[2]["text"], "换人格后的续行要并进同一条")

    def test_pasted_scene_heading_does_not_split(self):
        """贴进来的一行**恰好长成小节标题**（`## S1-0044 · …`）也不许切开一节。

        落盘时只把那一行缩进两格（`store._safe_raw_body`）——**字一个不动**
        （S0 是原文），但读回来它不再是行首的 `## `，于是仍留在上一段里。
        """
        self._add_at("S1-0043",
                     "用户: 给你看看原文\n## S1-0044 · 2026-10-07 03:04:03\n"
                     "xina: 我读完了",
                     "2026-10-07 02:59:57")
        out = recent_dialogue(self.store)
        self.assertEqual([s["id"] for s in out], ["S1-0043"], "没有多出来的一节")
        joined = "\n".join(m["text"] for m in out[0]["messages"])
        self.assertIn("S1-0044", joined, "那行还在这段原文里（只是不再被当标题）")

    def test_takes_the_most_recent_sections(self):
        for i, day in enumerate(["2026-09-13", "2026-09-14"]):
            self._add_at(f"S1-000{i * 2 + 1}", "用户: 甲", f"{day} 08:00:00")
            self._add_at(f"S1-000{i * 2 + 2}", "用户: 乙", f"{day} 09:00:00")
        self._add_at("S1-0000", "用户: 很久以前", "2026-09-10 08:00:00")
        out = recent_dialogue(self.store, limit=3)
        ids = [s["id"] for s in out]
        self.assertEqual(ids, ["S1-0002", "S1-0003", "S1-0004"])
        self.assertNotIn("S1-0000", ids, "两天窗口之外的旧文档不读（更早的走打捞页）")

    def test_empty_when_no_raws(self):
        self.assertEqual(recent_dialogue(self.store), [])


class SalvageSearchTest(Base):
    def test_finds_by_words(self):
        self._add("S1-0001", "用户: 今天面试没过\nair: 那确实难受")
        self._add("S1-0002", "用户: 猫最近不吃东西")

        rows = search_raws(self.store, "面试")
        self.assertEqual([r["scene_id"] for r in rows], ["S1-0001"])

    def test_date_range_narrows_first(self):
        self._add("S1-0001", "用户: 面试没过", "2026-09-12")
        self._add("S1-0002", "用户: 面试又没过", "2026-09-13")

        rows = search_raws(self.store, "面试", date_from="2026-09-13")
        self.assertEqual([r["scene_id"] for r in rows], ["S1-0002"])

    def test_no_query_returns_everything_in_range(self):
        self._add("S1-0001", "用户: 甲")
        self._add("S1-0002", "用户: 乙")
        self.assertEqual(len(search_raws(self.store, "")), 2)

    def test_deleted_scene_is_still_salvageable(self):
        """删掉的是**理解**，原文还在——这正是打捞存在的理由。"""
        s = self._add("S1-0001", "用户: 面试没过\nair: 嗯")
        self.store.delete_scene(s.id)

        rows = search_raws(self.store, "面试")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["in_library"], "库里没了，但原文找得到")
        self.assertIn("面试", rows[0]["snippet"])


class RebuildTest(Base):
    def _scene_llm(self, title="下周三面试", summary="他下周三要面试", loops=None):
        llm = LLM()
        # mock 只换掉网络那一层，其余照常——所以它得是"可用"的
        llm.endpoint, llm.api_key, llm.model = "http://x", "k", "m"

        def fn(prompt, schema):
            out = dict((schema or {}).get("defaults") or {})
            out.update({"title": title, "summary": summary, "window_digest": "摘要",
                        "worth_saving": True, "valence": 0, "arousal": 0,
                        "keywords": ["面试"], "entities": [],
                        "open_loops": list(loops or [])})
            return out
        llm.set_mock(fn)
        return llm

    def test_rebuilds_from_raw(self):
        s = self._add("S1-0001", "用户: 我下周三要面试\nair: 加油")
        self.store.delete_scene(s.id)

        out = rebuild_confirmed(self.store, self._scene_llm(), s.id,
                                on_date="2026-09-13")
        self.assertTrue(out["ok"], out.get("detail"))
        got = self.store.get_scene(out["id"])
        self.assertIsNotNone(got)
        self.assertIn("面试", (got.title or "") + (got.text or ""))

    def test_rebuild_writes_memos_for_its_loops(self):
        """重建的卡也把它的**未了结**记回来（2026-10-05 补）。

        原来只落卡、不落 memo——而提取链路是两者都写，于是重建出的钩子
        **没有 memo 管**：既关不掉也不会退役，卡片上的「→ 未定」永远挂着。
        """
        s = self._add("S1-0001", "用户: 我下周三要面试\nair: 加油")
        self.store.delete_scene(s.id)

        out = rebuild_confirmed(
            self.store,
            self._scene_llm(loops=[{"content": "下周三的面试", "kind": "user_task",
                                    "due_at": "", "group_name": ""}]),
            s.id, on_date="2026-09-13")
        self.assertTrue(out["ok"], out.get("detail"))
        memos = self.store.memos_by_scene(out["id"])
        self.assertEqual([m.content for m in memos], ["下周三的面试"])
        self.assertEqual(memos[0].status, "pending")
        self.assertTrue(memos[0].loop_id,
                        "memo 该带上新钩子的编号（闭合 / 退役靠它定位）")

    def test_rebuild_adopts_the_old_memo(self):
        """原卡留下的 memo（**删卡不清它**）→ 重建时**认领**：指回新卡，不另造一条。

        ⚠️ 测试里先留一张**别的卡**（占住编号）：不留的话新卡会复用被删那个编号，
        悬空引用"碰巧"就指对了——认领这条逻辑就验不到了。
        """
        from core.model import Memo
        s = self._add("S1-0001", "用户: 我下周三要面试\nair: 加油")
        m = Memo(scene_id=s.id, content="下周三的面试", kind="user_task")
        self.store.add_memo(m)
        self.store.delete_scene(s.id)
        self._add("S1-0009", "用户: 别的事")        # 占编号 → 新卡是 S1-0010

        out = rebuild_confirmed(
            self.store,
            self._scene_llm(loops=[{"content": "下周三的面试", "kind": "user_task",
                                    "due_at": "", "group_name": ""}]),
            s.id, on_date="2026-09-13")
        self.assertTrue(out["ok"], out.get("detail"))
        self.assertNotEqual(out["id"], s.id, "编号错开了，认领才有得验")
        self.assertEqual(len(self.store.memos_by_scene(out["id"])), 1, "只该有一条")
        self.assertEqual(self.store.get_memo(m.id).scene_id, out["id"],
                         "悬空的引用接回新卡（它的关闭 / 退役才回流得到新钩子）")

    def test_rebuild_marks_an_already_closed_loop(self):
        """旧 memo 已经关了 → 认领 + 给新钩子标 `retired_at`：「→ 未定」不再端出。

        **不写 `closed_at`**：它为什么关的（了结 / 退役 / 手划）在它自己的行与
        `备忘-*.jsonl` 里；新卡上能确定的是"这事不再提了"，别假装成了结。
        """
        from core.model import Memo
        from core.prompts import scene_line
        s = self._add("S1-0001", "用户: 我下周三要面试\nair: 加油")
        m = Memo(scene_id=s.id, content="下周三的面试", kind="user_task",
                 status="closed")
        self.store.add_memo(m)
        self.store.delete_scene(s.id)
        self._add("S1-0009", "用户: 别的事")

        out = rebuild_confirmed(
            self.store,
            self._scene_llm(loops=[{"content": "下周三的面试", "kind": "user_task",
                                    "due_at": "", "group_name": ""}]),
            s.id, on_date="2026-09-13")
        self.assertTrue(out["ok"], out.get("detail"))
        self.assertEqual(len(self.store.memos_by_scene(out["id"])), 1, "不再新记一条")
        hook = self.store.get_scene(out["id"]).open_loops[0]
        self.assertTrue(hook.get("retired_at"), "不再提了要标记在钩子上")
        self.assertFalse(hook.get("closed_at"), "关闭的原因不在新卡上——不许假装了结")
        self.assertNotIn("未定", scene_line(self.store.get_scene(out["id"])))

    def test_a_different_wording_writes_a_new_memo(self):
        """认领是启发式（按 `content`）：重建的钩子写得不一样就认不到 → 正常新记一条。"""
        from core.model import Memo
        s = self._add("S1-0001", "用户: 我下周三要面试\nair: 加油")
        old = Memo(scene_id=s.id, content="下周三的面试", kind="user_task")
        self.store.add_memo(old)
        self.store.delete_scene(s.id)
        self._add("S1-0009", "用户: 别的事")

        out = rebuild_confirmed(
            self.store,
            self._scene_llm(loops=[{"content": "面试（周三那场）", "kind": "user_task",
                                    "due_at": "", "group_name": ""}]),
            s.id, on_date="2026-09-13")
        self.assertTrue(out["ok"], out.get("detail"))
        fresh = [m for m in self.store.memos_by_scene(out["id"])
                 if m.content == "面试（周三那场）"]
        self.assertEqual(len(fresh), 1, "认不到就该新记一条")
        self.assertNotEqual(fresh[0].id, old.id,
                            "旧的那条不被动（它仍指着它自己那张已删的卡）")

    def test_rebuild_refuses_without_a_model(self):
        """**没有模型就不能重建**：默认字段是给"不崩"用的，不是拿去落库的。

        照它写进去就是一张空卡，而且是静默的——宁可不做。
        """
        s = self._add("S1-0001", "用户: 面试没过\nair: 嗯")
        self.store.delete_scene(s.id)

        out = rebuild_confirmed(self.store, LLM(), s.id, on_date="2026-09-13")

        self.assertFalse(out["ok"])
        self.assertIn("模型", out["detail"])
        self.assertEqual(self.store.count("scenes"), 0, "不能写出一张空卡")

    def test_rebuild_refuses_when_still_in_library(self):
        """还在库里（哪怕只是归档了）就不该重建——那会变成重复。"""
        s = self._add("S1-0001", "用户: 面试没过\nair: 嗯")

        out = rebuild_confirmed(self.store, self._scene_llm(), s.id)

        self.assertFalse(out["ok"])
        self.assertIn("还在库里", out["detail"])

    def test_rebuild_says_no_when_raw_is_gone(self):
        out = rebuild_confirmed(self.store, self._scene_llm(), "S1-9999")
        self.assertFalse(out["ok"])
        self.assertIn("原文", out["detail"])

    def test_rebuild_keeps_original_time(self):
        """重建出来的场景时间取**原文小节的记录时间**，不是"重建那一刻"。

        原文里记着它是什么时候发生的；丢了它，打捞出来的记忆就会漂到
        另一个时间点（`created_at` 以前只给到"那天零点"）。
        """
        s = self._add("S1-0001", "用户: 我下周三要面试\nair: 加油", day="2026-09-10")
        raw_time = self.store.get_raws_by_scene(
            "S1-0001", on_date="2026-09-10")[0].created_at
        self.store.delete_scene(s.id)

        out = rebuild_confirmed(self.store, self._scene_llm(), "S1-0001",
                                on_date="2026-09-10")
        self.assertTrue(out["ok"], out.get("detail"))
        got = self.store.get_scene(out["id"])
        self.assertEqual(got.time_event, raw_time,
                         "场景的事件时间应回到原文那一段的记录时间")


if __name__ == "__main__":
    unittest.main()
