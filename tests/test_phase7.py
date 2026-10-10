"""记忆整理行为（体检）的测试。

设计记录见本地「记忆整理行为」稿（不随仓库分发）。这一层测四件事：

  - **复核打分 → 动作**：holds 什么都不做（**不续命**）/ thin·stale 降档（不删）/
    wrong 只出提议（人点头才**真删**）——"系统的自动动作永不硬删"是硬约束
  - **防抖**：`min_days` 内不重复看、一次只看 `cap` 条、
    一次最多降 `downgrade_cap` 条（防模型集体误判）
  - **体检**：topic 封顶（钱花在料最足的地方）、留痕落盘、空闲时零调用
  - **触发**（`maintenance_tick`）：量到 / 时间到取先到；跑完清零 + 时刻落 `meta`；
    坏时间戳宁可多跑一次（不能"从此永不体检"）

用例分组：
  脚手架   _NoEmbedding · review_llm · summary_llm · Base
  复核     ReviewProfilesTest 打分映射 · ReviewGuardTest 三道防抖 ·
           ReviewProposalTest 提议生命周期
  体检     MaintenanceCycleTest 封顶/留痕/零调用 · MaintenanceTickTest 触发判定
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core.distill import maintenance_cycle, review_profiles, run_distill_cycle
from core.llm import LLM
from core.model import (PROFILE_ESTABLISHED, PROFILE_PENDING, Profile, Scene,
                        Summary)
from core.store import Store, now_str
from core.weave import user_reject_profile


class _NoEmbedding:
    available = False

    def embed_one(self, text):
        return None

    def embed(self, texts):
        return None


def review_llm(verdicts: dict, calls: list | None = None):
    """假模型：`profile_review` 按 id 回打分；其余结构化调用给保守空值。

    `verdicts`：`{画像 id: (打分, 理由)}` 或 `{画像 id: (打分, 理由, 改写)}`
    （`reword` 档要第三项）；`calls` 传列表进去就记录每次调用的 schema 名
    （用来断言"什么时候不该调模型"）。
    """
    def fn(prompt, schema):
        name = (schema or {}).get("name") or ""
        if calls is not None:
            calls.append(name)
        if name == "profile_review":
            items = [{"id": pid, "verdict": v[0], "reason": v[1],
                      "statement": v[2] if len(v) > 2 else ""}
                     for pid, v in (verdicts or {}).items()]
            return {"items": items}
        if name in ("memo_class", "memo_group"):
            return {"items": []}
        return {}

    llm = LLM()
    llm.set_mock(fn)
    return llm


def summary_llm(calls: list | None = None):
    """假模型：只答 S2 聚合（`topic_summary`），其余结构化调用给空。"""
    def fn(prompt, schema):
        name = (schema or {}).get("name") or ""
        if calls is not None:
            calls.append(name)
        if name == "topic_summary":
            return {"text": "这一段说的是同一类事。"}
        if name in ("memo_class", "memo_group"):
            return {"items": []}
        return {}

    llm = LLM()
    llm.set_mock(fn)
    return llm


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = Store(self.root / "t.db")
        self._old_paths = {k: cfgmod.PATHS[k]
                           for k in ("trace_dir", "raws_dir", "backup_dir", "shortterm")}
        cfgmod.PATHS["trace_dir"] = str(self.root / "trace")
        cfgmod.PATHS["raws_dir"] = str(self.root / "raws")
        cfgmod.PATHS["backup_dir"] = str(self.root / "backups")
        cfgmod.PATHS["shortterm"] = str(self.root / "shortterm.json")

    def tearDown(self):
        for k, v in self._old_paths.items():
            cfgmod.PATHS[k] = v
        self.store.close()
        self._tmp.cleanup()

    def profile(self, statement="遇到压力会先自己扛",
                status=PROFILE_ESTABLISHED, with_evidence: bool = True, **kw) -> Profile:
        """建一条画像（topic 用陈述派生，保证唯一）。

        `with_evidence=True`（默认）时**补三条真场景做印证**——复核会先跑一遍
        代码的硬判据（四道关：印证数 / sources / 同情境 / 跨来源），
        没有真实来源的画像会被直接降档、根本走不到模型那步（见 `review_profiles`）。
        要测那个前置降档就传 `with_evidence=False` 或事后摘掉来源。
        """
        srcs = []
        if with_evidence:
            for i in range(3):
                srcs.append(self.scene(f"用户·{statement[:6]}", i).id)
        p = Profile(topic=f"用户·{statement[:10]}", statement=statement,
                    status=status, evidence=len(srcs) or 3, sources=srcs, **kw)
        self.store.add_profile(p)
        return self.store.get_profile(p.id)

    def scene(self, topic: str, i: int, trigger_class: str = "被评价") -> Scene:
        """造一条场景。**同情境**（`trigger_class` 相同）是硬判据之一——
        复核前置会查它，所以默认给同一个合法类别（要造"情境不一致"传别的值）。"""
        s = Scene(topic=topic, title=f"{topic} 第 {i} 件", text="聊了一件相关的事",
                  trigger_class=trigger_class, trigger=f"第 {i} 次被指出问题")
        self.store.add_scene(s)
        return s


class ReviewProfilesTest(Base):
    def test_holds_changes_nothing(self):
        """`holds` 什么都不做——**尤其不 bump `last_support_at`**。

        bump 的话"复核通过"成了续命机制，90 天老化永远不触发
        （设计稿 §五点名的那个坑）。
        """
        p = self.profile()
        before = self.store.get_profile(p.id).last_support_at
        out = review_profiles(self.store, review_llm({p.id: ("holds", "还成立")}))

        got = self.store.get_profile(p.id)
        self.assertEqual(out["checked"], [p.id])
        self.assertEqual(got.status, PROFILE_ESTABLISHED, "holds 不该动状态")
        self.assertFalse(got.invalidated_at)
        self.assertEqual(got.last_support_at, before, "复核通过不续命（不 bump）")
        self.assertTrue(got.last_review_at, "复核过的要留时刻——隔离期靠它")
        self.assertEqual(self.store.unhandled_reviews(), [], "holds 不是提议")

    def test_thin_and_stale_downgrade_without_deleting(self):
        """thin / stale → 降档 `pending`。**降档不是删**：invalidated_at 仍空。"""
        a = self.profile("证据薄的一条")
        b = self.profile("过时的一条")
        out = review_profiles(self.store, review_llm({
            a.id: ("thin", "撑不住"), b.id: ("stale", "很久没人提了")}))

        self.assertEqual(sorted(out["downgraded"]), sorted([a.id, b.id]))
        for pid in (a.id, b.id):
            got = self.store.get_profile(pid)
            self.assertEqual(got.status, PROFILE_PENDING, "降档 = 退回待验证")
            self.assertFalse(got.invalidated_at,
                             "降档 ≠ 失效——invalidated_at 只给修正")

    def test_wrong_only_suggests(self):
        """`wrong` **只出提议**：她不能自己删（人的裁决最高）。"""
        p = self.profile("依据不支持的一条")
        out = review_profiles(self.store, review_llm({p.id: ("wrong", "依据不支持")}))

        self.assertEqual(out["suggested"], [p.id])
        got = self.store.get_profile(p.id)
        self.assertEqual(got.status, PROFILE_ESTABLISHED, "wrong 不动画像状态")
        self.assertFalse(got.invalidated_at)
        rows = self.store.unhandled_reviews()
        self.assertEqual(len(rows), 1, "提议要落表——它是等人处理的待办")
        self.assertEqual(rows[0].profile_id, p.id)
        self.assertEqual(rows[0].reason, "依据不支持", "理由要留（不然没法判）")

    def test_reword_revises_instead_of_downgrading(self):
        """`reword`：内容站得住、写法坏了（评价词）→ **修正**，不是降档也不是作废。

        旧版进历史（`by="revision"`——她自己更新说法，不是人的否决）、
        新版另起一条 pending，来源沿用旧的（复核没有新证据）。
        """
        p = self.profile("他是个固执的人")
        out = review_profiles(self.store, review_llm({
            p.id: ("reword", "评价词，改成取舍式", "被推着改主意时，他先坚持自己的方案")}))

        self.assertEqual(out["reworded"], [p.id])
        self.assertEqual(out["downgraded"], [], "写法坏 ≠ 证据薄，不该降档")
        old = self.store.get_profile(p.id)
        self.assertEqual(old.invalidated_by, "revision", "是她更新说法，不是人否决")
        self.assertTrue(old.invalidated_at)

        new = self.store.current_profile_by_topic(old.topic)
        self.assertIsNotNone(new)
        self.assertNotEqual(new.id, p.id, "新版本另起（版本序列，旧版不删）")
        self.assertEqual(new.statement, "被推着改主意时，他先坚持自己的方案")
        self.assertEqual(new.status, PROFILE_PENDING, "新版本回到待验证")
        self.assertEqual(self.store.unhandled_reviews(), [], "reword 不是提议")

    def test_reword_without_statement_is_skipped(self):
        """说了重写却没给改写：当没复核过（不写 `last_review_at`，下次再来）。"""
        p = self.profile()
        out = review_profiles(self.store, review_llm({p.id: ("reword", "写法坏，但我忘了写")}))
        self.assertEqual(out["checked"], [])
        self.assertFalse(self.store.get_profile(p.id).last_review_at)

    def test_reword_shares_the_change_budget(self):
        """降档与重写**共享**改动预算：一次别动太多（防模型集体误判）。"""
        a = self.profile("第一条")
        b = self.profile("第二条")
        c = self.profile("第三条")
        out = review_profiles(self.store, review_llm({
            a.id: ("thin", "薄"),
            b.id: ("reword", "写法坏", "改成取舍式"),
            c.id: ("reword", "写法坏", "改成取舍式"),
        }), downgrade_cap=2)

        self.assertEqual(len(out["downgraded"]) + len(out["reworded"]), 2,
                         "改动总数受预算封顶（降档 + 重写共享）")
        self.assertEqual(self.store.get_profile(c.id).status, PROFILE_ESTABLISHED,
                         "超预算的那条保持原状，下次体检再说")

    def test_schema_enum_matches_verdicts_constant(self):
        """schema 的 enum 与 `REVIEW_VERDICTS` 必须一致——两处手写，写漏一个就漂。

        （`reword` 就是这样加进来的：常量、schema、动作三处都得动；
        少动一处 = 模型给了分但系统不认，静默丢弃。）
        """
        from core.model import REVIEW_VERDICTS
        from core.prompts import REVIEW_SCHEMA
        enum = (REVIEW_SCHEMA["schema"]["properties"]["items"]["items"]
                ["properties"]["verdict"]["enum"])
        self.assertEqual(set(enum), set(REVIEW_VERDICTS))

    def test_unparsable_verdict_is_left_alone(self):
        """认不出的打分当没复核过——连 `last_review_at` 都不写（下次再来）。"""
        p = self.profile()
        out = review_profiles(self.store, review_llm({p.id: ("bogus", "")}))
        self.assertEqual(out["checked"], [])
        self.assertFalse(self.store.get_profile(p.id).last_review_at)
        self.assertEqual(self.store.unhandled_reviews(), [])

    def test_review_writes_a_record_with_reason(self):
        """每次复核落一行记录（不覆盖历史）——"它上次复核是什么结论"查得到。"""
        p = self.profile()
        out = review_profiles(self.store, review_llm({p.id: ("holds", "看着还行")}))
        self.assertEqual(out["details"],
                         [{"id": p.id, "verdict": "holds", "reason": "看着还行"}],
                         "逐条打分 + 理由要进返回值（留痕带着它落盘）")
        row = self.store.conn.execute(
            "SELECT * FROM profile_reviews WHERE profile_id = ?", (p.id,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["verdict"], "holds")
        self.assertEqual(row["reason"], "看着还行")
        self.assertTrue(str(row["id"]).startswith("PR-"), "编号前缀同 _ID_PREFIX")


class ReviewGuardTest(Base):
    def test_min_days_skips_recently_reviewed(self):
        """`min_days` 内不重复复核——**没候选就不该调模型**（零成本）。"""
        p = self.profile()
        calls: list = []
        llm = review_llm({p.id: ("holds", "还成立")}, calls=calls)
        review_profiles(self.store, llm)
        out2 = review_profiles(self.store, llm)

        self.assertEqual(out2["checked"], [], "7 天隔离期内不再看它")
        self.assertEqual(calls.count("profile_review"), 1, "第二次不该调模型")

    def test_downgrade_cap(self):
        """一次最多降 `downgrade_cap` 条——防模型集体误判；超出的下次再说。"""
        ps = [self.profile(f"薄证据第 {i} 条") for i in range(3)]
        out = review_profiles(self.store,
                              review_llm({p.id: ("thin", "薄") for p in ps}),
                              downgrade_cap=2)
        self.assertEqual(len(out["downgraded"]), 2)
        left = [p.id for p in ps if self.store.get_profile(p.id).status == PROFILE_ESTABLISHED]
        self.assertEqual(len(left), 1, "超出的那条原样保留（幂等，不着急）")

    def test_review_cap(self):
        """一次最多看 `cap` 条（1 次批量的规模）。"""
        ps = [self.profile(f"第 {i} 条") for i in range(6)]
        out = review_profiles(self.store,
                              review_llm({p.id: ("holds", "") for p in ps}), cap=3)
        self.assertEqual(len(out["checked"]), 3)

    def test_pending_is_not_reviewed(self):
        """只复核 `established`——待验证的本来就还没立，没什么可复核的。"""
        p = self.profile("还在猜的一条", status=PROFILE_PENDING)
        calls: list = []
        out = review_profiles(self.store, review_llm({p.id: ("holds", "")}, calls=calls))
        self.assertEqual(out["checked"], [])
        self.assertEqual(calls, [], "没有候选：一次模型都不该调")


class ReviewProposalTest(Base):
    def test_handled_proposal_disappears(self):
        """「留着」= 标记已处理（重复标记是 0）——提议不再挂出来。"""
        p = self.profile()
        review_profiles(self.store, review_llm({p.id: ("wrong", "不对")}))
        rows = self.store.unhandled_reviews()
        self.assertEqual(self.store.mark_review_handled(rows[0].id), 1)
        self.assertEqual(self.store.unhandled_reviews(), [])
        self.assertEqual(self.store.mark_review_handled(rows[0].id), 0, "幂等")

    def test_rejecting_the_profile_expires_proposals(self):
        """画像被否决（**真删**）→ 它名下的提议跟着清掉（从任何入口处理了都一样）。"""
        p = self.profile()
        review_profiles(self.store, review_llm({p.id: ("wrong", "不对")}))
        self.assertEqual(user_reject_profile(self.store, p.id), "deleted")
        self.assertEqual(self.store.unhandled_reviews(), [])

    def test_maint_reviews_expires_dead_proposals(self):
        """提议指向的画像已经不在（删了 / 旧库里的作废行）→ 列提议时顺手标过期。"""
        from core.dashboard import App
        app = App.__new__(App)
        app.store = self.store
        p = self.profile()
        review_profiles(self.store, review_llm({p.id: ("wrong", "不对")}))
        self.assertEqual(len(app.maint_reviews()), 1)

        user_reject_profile(self.store, p.id)    # 绕过 App.reject 直接删
        self.assertEqual(app.maint_reviews(), [], "已删的画像不该还挂提议")
        self.assertEqual(self.store.unhandled_reviews(), [], "提议随画像一起清了")

    def test_review_material_hides_deleted_evidence(self):
        """复核材料里**已删的依据不出现**（2026-09-24，引用 = `sources ∩ 现存节点`）。

        原行为是"退回 `evidence_pack` 快照"——真删落地后那条路取消了：
        他删掉的东西不该从复核材料里再露出来（同镜像页 / 展开的口径）。
        """
        from core.distill import _review_item
        s = Scene(id="S1-0001", title="会被删的场景", text="x")
        self.store.add_scene(s)
        p = Profile(id="S3-0001", topic="t", statement="s",
                    status=PROFILE_ESTABLISHED, sources=[s.id],
                    evidence_pack=[{"id": s.id, "title": "会被删的场景", "summary": ""}])
        self.store.add_profile(p)
        self.assertEqual(_review_item(self.store, p)["proof"], ["「会被删的场景」"])

        self.store.delete_scene(s.id)
        p2 = self.store.get_profile(p.id)
        self.assertEqual(_review_item(self.store, p2)["proof"], [],
                         "删了就一条都不列——快照也不兜底")

    def test_memory_action_is_one_door_for_three_layers(self):
        """记忆动作的统一口（2026-09-24，工具箱稿 §3.4）：**三层同一套**。

        这是台账页按钮走的路——和对话确认条（`chat.confirm`）**共用一份分派**
        （`weave.delete_by_layer` 等）：两条入口、同一批动作、同一批后果。
        """
        from core.dashboard import App
        app = App.__new__(App)              # 不跑 __init__：这里只需要 store
        app.store = self.store
        s = Scene(id="S1-0001", title="被批评", text="被批评后想离开")
        self.store.add_scene(s)
        s2 = Summary(id="S2-0001", topic="用户·压力", text="受挫后倾向离开")
        self.store.add_summary(s2)

        # 归档（混层也行）
        out = app.memory_action([s.id, s2.id], "archive")
        self.assertTrue(out["ok"])
        self.assertEqual(sorted(out["ids"]), ["S1-0001", "S2-0001"])
        self.assertTrue(self.store.get_scene(s.id).archived)

        # 取消归档
        out = app.memory_action([s.id, s2.id], "unarchive")
        self.assertEqual(sorted(out["ids"]), ["S1-0001", "S2-0001"])
        self.assertFalse(self.store.get_scene(s.id).archived)

        # 改（一次一条）
        out = app.memory_action([s2.id], "update", "改成新的叙述")
        self.assertTrue(out["ok"])
        self.assertEqual(self.store.get_summary(s2.id).text, "改成新的叙述")

        # 改字段（2026-09-24 起场景不止摘要能改；界面传中文名，后端归一）
        out = app.memory_action([s.id], "update", "用户·压力", "主题")
        self.assertTrue(out["ok"])
        self.assertEqual(self.store.get_scene(s.id).topic, "用户·压力")
        bad = app.memory_action([s.id], "update", "x", "心情")
        self.assertFalse(bad["ok"], "不认的字段要拒绝，别静默不动")
        # 三层都能改标签（2026-09-24 晚）：摘要改主题**通过**——改的是归类，
        # 叙述与它的素材都不动
        ok2 = app.memory_action([s2.id], "update", "用户·沟通", "主题")
        self.assertTrue(ok2["ok"], "摘要也能改主题")
        self.assertEqual(self.store.get_summary(s2.id).topic, "用户·沟通")
        self.assertEqual(self.store.get_summary(s2.id).text, "改成新的叙述",
                         "改主题不动叙述")
        # 多主题（2026-09-24 晚）：S2 / S3 一次给 1-3 个——第一个是主主题
        ok4 = app.memory_action([s2.id], "update", "用户·沟通，用户·工作", "主题")
        self.assertTrue(ok4["ok"])
        self.assertEqual(self.store.get_summary(s2.id).topics, ["用户·沟通", "用户·工作"])
        # 画像：改主题**原地改**，不动陈述 / 印证（与"改陈述走修正"的差别）
        p = Profile(id="S3-0001", topic="用户·压力", statement="受挫后倾向离开",
                    status=PROFILE_ESTABLISHED, evidence=2)
        self.store.add_profile(p)
        ok3 = app.memory_action([p.id], "update", "用户·边界", "主题")
        self.assertTrue(ok3["ok"], "画像也能改主题")
        got = self.store.get_profile(p.id)
        self.assertEqual(got.topic, "用户·边界")
        self.assertEqual(got.evidence, 2, "改主题不动印证数")
        self.assertEqual(got.statement, "受挫后倾向离开", "改主题不动陈述")

        # 删
        out = app.memory_action([s.id], "delete")
        self.assertTrue(out["ok"])
        self.assertIsNone(self.store.get_scene(s.id))

        # 不认的动作 / 空编号 / 改多条：诚实拒绝（不许静默做一半）
        self.assertFalse(app.memory_action([s2.id], "explode")["ok"])
        self.assertFalse(app.memory_action([], "delete")["ok"])
        self.assertFalse(app.memory_action(["S1-0001", "S2-0001"], "update", "x")["ok"])

    def test_tool_forget_then_confirm_deletes_and_expires_proposals(self):
        """她调 `forget_memory` 提议删一条画像 → 他点确认 → 真删。

        （2026-09-24：`reject_profile` 并入 `forget_memory`——三层同一套，
        处置都是"她列出来 → 他点"。统一出口仍在 `weave`。）
        """
        from core.tools import execute
        from core.weave import delete_by_layer
        p = self.profile()
        review_profiles(self.store, review_llm({p.id: ("wrong", "不对")}))

        out = execute(self.store, "forget_memory", {"id": p.id},
                      {"emb": None, "confirm_channel": True})
        self.assertTrue(out["ok"], "她只提议")
        self.assertIsNotNone(self.store.get_profile(p.id), "提议阶段一个字都不改")

        res = delete_by_layer(self.store, [p.id])        # 他点"彻底删除"
        self.assertTrue(res["ok"])
        self.assertIsNone(self.store.get_profile(p.id), "他点了才删")
        self.assertEqual(self.store.unhandled_reviews(), [],
                         "两条入口共用一处清理（界面按钮 / 她提议他点）——漏一条就挂死链")

    def test_app_reject_expires_proposals(self):
        """`App.reject`（界面否决按钮那条路）同样是真删、提议跟着清。"""
        from core.dashboard import App
        app = App.__new__(App)          # 不跑 __init__：这里只需要 store
        app.store = self.store
        p = self.profile()
        review_profiles(self.store, review_llm({p.id: ("wrong", "不对")}))

        out = app.reject(p.id)
        self.assertTrue(out["ok"])
        self.assertIsNone(self.store.get_profile(p.id))
        self.assertEqual(self.store.unhandled_reviews(), [],
                         "人否决了画像，提议不该还挂在镜像页上")


class GateRecheckTest(Base):
    def test_dead_profile_is_demoted_without_calling_llm(self):
        """可算的判据归代码：来源被摘光的画像 → 系统直接降档，**不花一次调用**。

        这是"状态与判据脱节"的出口（S1 被删、来源被摘之后，established
        可能早就不满足了）——门槛值在配置里，模型只能凭常识猜，
        所以这一层从一开始就不该交给模型（2026-09-23）。
        """
        p = self.profile()                  # 三条真场景：硬判据本来过
        self.store.set_profile_sources(p.id, [], evidence=0)   # 模拟来源被摘

        calls: list = []
        out = review_profiles(self.store,
                              review_llm({p.id: ("holds", "还成立")}, calls=calls))

        self.assertEqual(out["regressed"], [p.id])
        self.assertNotIn("profile_review", calls, "硬判据由代码复查——不该调模型")
        self.assertEqual(self.store.get_profile(p.id).status, PROFILE_PENDING)
        self.assertEqual(self.store.unhandled_reviews(), [], "系统复查降档不是提议")
        row = self.store.conn.execute(
            "SELECT * FROM profile_reviews WHERE profile_id = ?", (p.id,)).fetchone()
        self.assertIsNotNone(row, "系统复查也要留记录")
        self.assertIn("系统复查", row["reason"], "'它为什么被降'要答得出来")

    def test_mixed_situations_also_stay_out_of_the_prompt(self):
        """情境不一致（三类各一条）→ 硬判据不过 → 直接降档，也不调模型。

        这是四道关里"同情境"那条路径的覆盖；其余三条由 `test_phase3` 的
        判据测试兜着——**两边是同一份 `_gate_recheck`**，不重复实现。
        """
        srcs = [self.scene("用户·混情境", i, trigger_class=cls).id
                for i, cls in enumerate(("被评价", "目标未达", "失去"))]
        p = Profile(topic="用户·混情境", statement="他遇事就退",
                    status=PROFILE_ESTABLISHED, evidence=3, sources=srcs)
        self.store.add_profile(p)

        calls: list = []
        out = review_profiles(self.store,
                              review_llm({p.id: ("holds", "还成立")}, calls=calls))
        self.assertEqual(out["regressed"], [p.id])
        self.assertEqual(self.store.get_profile(p.id).status, PROFILE_PENDING)
        self.assertEqual(calls, [], "情境不一致也归代码判")

    def test_only_healthy_profiles_reach_the_model(self):
        """混合场景：不过硬判据的当场降档，过得去的才进模型批次（名额不浪费）。"""
        dead = self.profile("来源被摘的一条")
        self.store.set_profile_sources(dead.id, [], evidence=0)
        good = self.profile("正常的一条")

        calls: list = []
        out = review_profiles(self.store,
                              review_llm({good.id: ("holds", "还成立")}, calls=calls))

        self.assertEqual(out["regressed"], [dead.id])
        self.assertEqual(out["checked"], [good.id], "只有过硬的进了她的批次")
        self.assertEqual(calls.count("profile_review"), 1)


class MaintenanceCycleTest(Base):
    def test_topic_cap_spends_where_the_material_is(self):
        """只处理 N 个 topic——**钱花在料最足的地方**（并列时按名字稳定排序）。"""
        for topic in ("t1", "t2", "t3", "t4"):
            for i in range(3):
                self.scene(topic, i)
        stats = maintenance_cycle(self.store, summary_llm(), trigger="测试", topic_cap=2)

        self.assertEqual(len(stats["s2_new"]), 2, "封顶 2 个 topic → 最多 2 条 S2")
        self.assertEqual(stats["topic_cap"], 2)
        self.assertEqual(stats["trigger"], "测试")

    def test_idle_run_costs_nothing(self):
        """空库体检：**零次模型调用**（幂等 + 无新素材短路的设计目标）。"""
        calls: list = []
        stats = maintenance_cycle(self.store, summary_llm(calls=calls), trigger="测试")
        self.assertEqual(calls, [], "没有素材、没有画像、没有空缺：一次都不该调")
        self.assertEqual(stats["s2_new"], [])
        self.assertEqual(stats["review"]["checked"], [])

    def test_trace_lands_on_disk(self):
        """每次体检落一行 `整理-*.jsonl`——"它昨天为什么整理"要答得出来。"""
        p = self.profile()
        maintenance_cycle(self.store, summary_llm(calls=None), trigger="累计 1.2 万字")
        # 复核走 review_llm 的打分网：这里用 maintenance_cycle 内部的调用，
        # 画像在库里 → 会被复核（mock 回空 items = 认不出 → 什么都不动，安全）
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob("整理-*.jsonl"))
        self.assertTrue(files, "体检要留痕")
        rec = json.loads(files[0].read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(rec["act"], "体检")
        self.assertEqual(rec["trigger"], "累计 1.2 万字")
        self.assertIn("review", rec)
        self.assertIn("archived", rec)
        self.assertTrue(p.id)               # 画像在库（上面那步只是造素材）

    def test_boundary_cycle_still_runs_everything(self):
        """会话边界整理**不封顶**（不传 topic_cap = 全量）——体检是兜底，不是替代。"""
        for topic in ("t1", "t2", "t3", "t4"):
            for i in range(3):
                self.scene(topic, i)
        stats = run_distill_cycle(self.store, summary_llm())     # 不传 cap
        self.assertEqual(len(stats["s2_new"]), 4, "边界整理该把 4 个 topic 都做完")


class MaintenanceTickTest(Base):
    def _app(self, stub_start: bool = True):
        from core.dashboard import App
        app = App.__new__(App)          # 不跑 __init__（那会建 LLM / emb）
        app.store = self.store
        app._distill_lock = threading.Lock()   # 绕过 `__init__` 就得把它的字段补齐
        app._distilling = False
        app._chars_since_maint = 0
        app.last_distill = {}
        if stub_start:
            app.started = []
            app.start_distill = lambda maintenance=False, trigger="": (
                app.started.append((maintenance, trigger)) or True)
        else:
            app.llm = summary_llm()
            app.emb = None
            app._distill_thread = None
        return app


    def test_below_threshold_skips(self):
        app = self._app()
        self.store.set_meta("last_maint_at", now_str())    # 刚整过
        app._chars_since_maint = 100
        self.assertIn("skipped", app.maintenance_tick())
        self.assertEqual(app.started, [])

    def test_chars_trigger(self):
        """累计到量 → 触发（体检而不是边界整理）。"""
        app = self._app()
        self.store.set_meta("last_maint_at", now_str())
        app._chars_since_maint = 10001
        out = app.maintenance_tick()
        self.assertTrue(out["started"])
        self.assertEqual(app.started[0][0], True, "走 maintenance 那条")
        self.assertIn("累计", app.started[0][1], "留痕要写清为什么跑")

    def test_time_trigger_and_first_run(self):
        """时间兜底：从没记过 = 该跑；记成 13 小时前也跑。"""
        app = self._app()
        out = app.maintenance_tick()
        self.assertTrue(out["started"], "服务第一次上线：没有记录就该跑一次")
        self.assertEqual(app.started[0][1], "首次整理（没有记录）",
                         "首次的文案不能写哨兵值（`_hours_since` 的 1e9 小时）")

        app.started.clear()
        old = (datetime.now() - timedelta(hours=13)).strftime("%Y-%m-%d %H:%M:%S")
        self.store.set_meta("last_maint_at", old)
        out = app.maintenance_tick()
        self.assertTrue(out["started"])
        self.assertIn("距上次", app.started[0][1])

    def test_broken_stamp_still_runs(self):
        """坏时间戳宁可多跑一次——**不能从此永不体检**（这正是这次要修的毛病）。"""
        app = self._app()
        self.store.set_meta("last_maint_at", "垃圾时间戳")
        self.assertTrue(app.maintenance_tick()["started"])

    def test_done_clears_and_stamps(self):
        """账起跑就结：计数清零 + 时刻落 `meta`（重启后判"距上次多久"靠它）。"""
        app = self._app()
        app._chars_since_maint = 500
        before = self.store.get_meta("last_maint_at")
        app._mark_maint_done()
        self.assertEqual(app._chars_since_maint, 0)
        now = self.store.get_meta("last_maint_at")
        self.assertTrue(now)
        self.assertNotEqual(now, before)

    def test_done_is_written_by_start_distill(self):
        """真 `start_distill`（不打的桩）：起跑结账 + 后台跑完写结果。"""
        app = self._app(stub_start=False)
        self.assertTrue(app.start_distill(maintenance=True, trigger="测试"))
        self.assertEqual(app._chars_since_maint, 0, "起跑就把账结掉（失败也认）")
        self.assertTrue(app.wait_distill(), "后台体检该跑完")
        self.assertIn("review", app.last_distill, "体检结果进 last_distill（界面同一处显示）")
        self.assertEqual(app.last_distill.get("trigger"), "测试")


if __name__ == "__main__":
    unittest.main()
