"""阶段 3 演示：备忘录状态机 + 编织层镜像呈现。

跑法：
    python demo_memo.py            # 内置剧本，不需要 key
    python demo_memo.py --fresh    # 先清空演示库
    run.cmd memo                   # 等价于第一种

这个脚本想让人看见的是**「克制」怎么被写成机制**：
  - 到期 ≠ 该提：到期只让它进候选，提不提由「有没有自然时机」定
  - 没有时机就不提（模型被明确赋予放弃权）
  - 提过一次就不再提
  - 敏感只调节力度：只记不提收窄到硬隐私（感冒 / 体检照提）
  - 超期太久直接关掉，不永远挂着
  - 提的时候只能「给门」，不能端细节

后两节演示 R6：**呈现**（可追溯、pending 标注、敏感不展示）与**可否决**。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core import settings
from core.distill import distill_step1
from core.llm import LLM
from core.memo import (close, due, judge_hits, mark_raised, retire_due,
                       standing_memos)
from core.model import Profile, Scene
from core.store import Store
from core.weave import render_mirror, user_reject_profile

LINE = "=" * 72


def shift(days: float, base: datetime | None = None) -> str:
    """相对现在（或给定时刻）偏移若干天的时间戳。

    时间都用相对位移造：写死日期的脚本过几天就全过期了，而"到期 / 超期"正是要演的东西。
    """
    return ((base or datetime.now()) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def make_mock_llm(state: dict) -> LLM:
    """剧本模型：写入抽卡 / memo 分类 / 闭合判定。

    `state` 是可变的，因为闭合判定的答案（哪条 memo 被了结）要到第五节才知道。
    （「提的时机」那次调用 2026-10-05 晚随时机表一起删了——现在到点就进注入，
    提不提是提示词里的分寸，没有中间那道判定了。）
    """
    cards = [
        {   # 第一段：有明确时间 → 直接抽（不在「估时间」的禁令范围内）
            "title": "下周三面试", "keywords": ["面试"], "summary": "下周三有面试",
            "window_digest": "用户说下周三要面试。", "valence": -1, "arousal": 1,
            "trigger": "面试", "trigger_class": "悬而未决", "reaction": "紧张",
            "outcome": "", "subject": "user", "topic": "用户·面试这件事",
            "self_ref": False, "air_stance": "", "sensitive": False, "entities": [],
            "open_loops": [{"content": "下周三面试", "kind": "user_task",
                            "due_at": shift(7)}],
        },
        {   # 第二段：没时间 → 只分类，周期由系统映射
            "title": "在学吉他", "keywords": ["吉他"], "summary": "最近在学吉他",
            "window_digest": "用户说最近在学吉他。", "valence": 1, "arousal": 0,
            "trigger": "", "trigger_class": "", "reaction": "", "outcome": "",
            "subject": "user", "topic": "用户·在学的事", "self_ref": False,
            "air_stance": "", "sensitive": False, "entities": [],
            "open_loops": [{"content": "在学吉他", "kind": "user_task", "due_at": ""}],
        },
        {   # 第三段：敏感（健康类）→ 照常能提：感冒 / 体检这类就该被问候
            "title": "妈妈要做检查", "keywords": ["妈妈", "检查"],
            "summary": "妈妈下周要去医院检查", "window_digest": "用户提到妈妈要去做检查。",
            "valence": -1, "arousal": 1, "trigger": "妈妈要做检查",
            "trigger_class": "悬而未决", "reaction": "担心", "outcome": "",
            "subject": "world", "topic": "妈妈·健康问题", "self_ref": False,
            "air_stance": "", "sensitive": True,
            "entities": [{"name": "妈妈", "kind": "person", "relation": "妈妈"}],
            "open_loops": [{"content": "妈妈的检查结果", "kind": "user_task", "due_at": ""}],
        },
    ]
    memo_classes = [
        {"content": "在学吉他", "class": "progress"},
        {"content": "妈妈的检查结果", "class": "deadline"},
    ]

    def fn(prompt, schema):
        name = schema.get("name")
        if name == "scene_card":
            return cards.pop(0) if cards else {}
        if name == "memo_class":
            return {"items": memo_classes}
        if name == "memo_hit":
            cid = state.get("closure_id") or ""
            items = ([{"id": cid, "action": "close", "content": "", "due_at": "",
                       "why": "他说练成了"}] if cid and cid in prompt else [])
            return {"items": items}
        return {}

    llm = LLM()
    llm.set_mock(fn)
    return llm


def show_memos(store: Store, now: str) -> None:
    """打印备忘录（还没了结的事），重点在「时间依据」那一列：时间要么是用户明说的
    （转述），要么是系统按类别映射的窗口——**没有第三种**。"""
    print(f"    {'id':<8}{'状态':<8}{'内容':<16}{'时间依据':<22}{'敏感'}")
    for m in sorted(store.open_memos(), key=lambda x: x.id):
        if m.due_at:
            basis = f"用户明说 {m.due_at[:10]}"
        elif m.window_days:
            basis = f"{m.kind_class} + {m.window_days} 天"
        else:
            basis = "（无）"
        print(f"    {m.id:<8}{m.status:<8}{m.content:<16}{basis:<22}"
              f"{'是' if m.sensitive else '否'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="air-link-01 · 阶段 3 演示（备忘录与镜像）")
    parser.add_argument("--fresh", action="store_true", help="先清空演示库再跑")
    parser.add_argument("--real", action="store_true", help="用真实 LLM（需先配环境变量）")
    args = parser.parse_args()

    # 外部配置（config.local.json / 环境变量）**必须先合进 CONFIG**——
    # 否则 `--real` 下的 LLM 读到的还是 config.py 的空默认值，配了也等于没配。
    settings.apply()
    cfgmod.ensure_dirs()
    demo_db = cfgmod.abspath("data/demo_memo.db")
    if args.fresh and demo_db.exists():
        demo_db.unlink()
        print(f"[--fresh] 已删除 {demo_db.name}")

    store = Store(demo_db)
    state: dict = {}
    llm = LLM() if args.real else make_mock_llm(state)

    print(LINE)
    print(" air-link-01 · 阶段 3 演示：备忘录（未了结的事）与镜像呈现")
    print(LINE)
    print(f" 库文件: {demo_db}")

    # ------------------------------------------------------------------
    print("\n" + LINE)
    print(" 一、写入三件未闭合的事")
    print(LINE)
    print("  两种时间来源泾渭分明：**用户明说的直接抽**，没说的只分类（周期由系统定）")
    segments = [
        ("我下周三要面试，有点紧张", [{"speaker": "user", "text": "我下周三要面试，有点紧张"}]),
        ("最近在学吉他", [{"speaker": "user", "text": "最近在学吉他，弹得还很烂"}]),
        ("我妈下周要去医院做个检查", [{"speaker": "user", "text": "我妈下周要去医院做个检查"}]),
    ]
    for label, msgs in segments:
        scene, _, _ = distill_step1(store, msgs, llm, source="demo")
        print(f"\n   对话: {label}")
        print(f"   → {scene.id}「{scene.title}」  敏感={bool(scene.sensitive)}")
        for m in store.memos_by_scene(scene.id):
            cls = f"{m.kind_class}/{m.window_days}天" if m.kind_class else "（用了用户明说的时间）"
            tag = "【不提·只记】" if m.sensitive == 2 else ""
            print(f"     📌 {m.id} {m.content}   {cls} {tag}")

    print("\n   当前未闭合的事：")
    show_memos(store, shift(0))

    # ------------------------------------------------------------------
    now = shift(35)
    print("\n" + LINE)
    print(f" 二、时间推到 35 天后（{now[:10]}）：到点 = 有资格进注入")
    print(LINE)
    candidates = due(store, now)
    print(f"   到点了: {[m.id for m in candidates]}"
          f"（{', '.join(m.content for m in candidates)}）")
    print("   → 「妈妈的检查结果」照进——**关心不是揭开**：")
    print("      「检查结果出来了吗」是给门（答多少由他定）；")
    print("      只有硬隐私（创伤 / 家事这类）才留在「只记不提」。")
    print("   → 2026-10-05 晚起这里是**唯一一条通往对话的路**：")
    print("      时机表 / 「该不该提」的判定 / 主动开口留言，三者都删了。")

    # `now=now`：演示把时间推到 35 天后，"到点没到点"要按那个时刻算
    picked = standing_memos(store, "最近在想要不要报个班学点东西", emb=None,
                            now=now)
    print("\n   这一轮她手上有什么（【你手上还挂着的事】）：")
    for c in picked:
        mark = "   ← 到点了（这一轮可以提一句）" if c.get("due") else ""
        print(f"     · {c['id']} {c['content']}（{c['why']}）{mark}")
    print("   → **一次一件**：到点的最多给一件，其余下一轮接着排（没记账的不算提过）。")
    print("   → **进即记账**：进了这一栏就算给过机会——下一轮它不再来（见下节）。")
    print("   → 分寸在提示词里：问过程不问结果（「那边公司怎么样」优于「过了吗」），")
    print("      不端已知细节；他问起时她说得出是哪件——这一栏的另一半作用。")

    # ------------------------------------------------------------------
    due_one = next((c for c in picked if c.get("due")), None)
    if due_one:
        print("\n" + LINE)
        print(" 三、提过一次就不再提")
        print(LINE)
        target = store.get_memo(due_one["id"])
        mark_raised(store, target.id)
        print(f"   {target.id} 待提 → 已提（raised）")
        again = [c["id"] for c in standing_memos(store, "又聊到学习", emb=None,
                                                 now=now)
                 if c.get("due")]
        print(f"   再挑一次（到点名单）: {again}")
        print(f"   → {target.id} 不在了（提过就不再提）；换上来的是下一件——")
        print("      **一次一件**是这个意思：不是\"只提一件就完了\"，是排队来。")
        print("   → 用户没接话（岔开了）就不再追。追着问是把关心变成催收。")
        print("   → 但它仍在「未闭合」里：他过几天回来说结果，要能把它关掉——")
        print("      卡被唤醒时那行「→ 未了结」还在（随卡走，见设计稿「接上唤醒」）。")

    # ------------------------------------------------------------------
    print("\n" + LINE)
    print(" 四、用户回来说了结果 → 闭合")
    print(LINE)
    guitar = [m for m in store.open_memos() if "吉他" in m.content]
    if guitar:
        state["closure_id"] = guitar[0].id
        said = "吉他我练成了，已经能弹唱了"
        out = judge_hits(store, said, llm)
        print(f"   输入: {said}")
        print(f"   → 了结 {out['closed']}")
        print("   ⚠️ 结果本身**不另存**：它就是对话内容，会跟着正常提取变成一条新场景卡。")
        print("      memo 只管住「不重复提」这一件事。")
    print(f"\n   命中判定（2026-10-05 取代词表预筛）：他这句话命中的每件挂着的事")
    print(f"   都判一次「变更 / 完结 / 无关」——**判据是内容，不是字面**。")
    print(f"   （`update` 会把新说法写回那条 memo 并清掉已提状态；判不准就 `none`。）")

    # ------------------------------------------------------------------
    far = shift(200)
    print("\n" + LINE)
    print(f" 五、时间再推到 200 天后（{far[:10]}）：超期退役")
    print(LINE)
    retired = retire_due(store, far)
    print(f"   提的窗口走完仍未提 → closed: {[r['id'] for r in retired]}")
    print("   → 判的是「**提的窗口**过没过」（起点 + 2 天；提过的再过 3 天没回音），")
    print("     不是「挂了多久」——不永远挂着：候选池只涨不消，「有没有该提的事」")
    print("     这个判断本身就会失去意义。")
    print("   → **退役 ≠ 完成**：钩子标 `retired_at`（不写 `closed_at`），")
    print("     「办完了」和「没人管了」分得开；两个字段渲染侧都跳过。")

    # ------------------------------------------------------------------
    print("\n" + LINE)
    print(" 六、R6 镜像呈现：把「air 眼中的他」拉出来看")
    print(LINE)
    print("   触发条件是**用户显式请求**（「你觉得我是什么样的人」），air 不自己开——")
    print("   自己判断「你现在需要被分析」正是诱导。")

    # 造一点可供统计的材料（同情境、同对象 → 形成模式）
    for i in range(3):
        s = Scene(topic="用户·面对被评价的反应", subject="user", trigger_class="被评价",
                  title=f"被组长批评（{i + 1}）", text="摘要", reaction="先退出，不争",
                  trigger="被组长当众批评", time_event=f"2026-08-1{i} 09:00:00",
                  sensitive=0)
        store.add_scene(s)
        eid = store.find_entity("组长") or None
        if eid is None:
            eid = store.add_entity("组长", "person")
        else:
            eid = eid.id
        store.link_scene_entity(s.id, eid)

    scenes = store.query_scenes(topic="用户·面对被评价的反应")
    # 第一条算「出处」（形成这条判断的依据），其余算「印证」（后来的支撑）。
    # 分开记是因为两者的归档保护期不同：出处 365 天，印证 90 天。
    pack = [{"id": scenes[0].id, "title": scenes[0].title,
             "summary": scenes[0].text, "time": (scenes[0].time_event or "")[:10]}]
    profile = Profile(topic="用户·面对被评价的反应", subject="user",
                      statement="遇到被评价时，他会先退出、不再争取",
                      status="pending", evidence=len(scenes),
                      sources=[s.id for s in scenes], evidence_pack=pack)
    store.add_profile(profile)
    # 印证走生产路径同一个方法（2026-09-24：不再写 evidence 边——真数据是
    # 「引用起点时间」，落在 `profile.evidence_at`；role 由 `evidence_pack` 派生）
    from core.distill import _mark_evidence
    for s in scenes:
        _mark_evidence(store, s.id, profile.id)

    mirror = render_mirror(store)
    for p in mirror["profiles"]:
        print(f"\n     {p['id']}  {p['statement']}")
        print(f"          [{p['status_label']}]  印证 {p['evidence']} 条")
        for src in p["sources"][:3]:
            role = "出处" if src.get("role") == "forming" else "印证"
            tag = "（来自追溯包）" if src.get("from_pack") else ""
            print(f"          依据[{role}] {src['id']} {src['time']} 「{src['title']}」{tag}")
        if p["hidden_sensitive"]:
            print(f"          （另有 {p['hidden_sensitive']} 条依据涉及隐私，未展示）")
    print(f"\n     ⚠️ 「出处」和「印证」分开标：出处是「凭什么」的答案（少、稳定），")
    print(f"        印证是支撑量（多、随新场景一直增长）。两者归档保护期不同")
    print(f"        （{cfgmod.cfg('capacity', 'forming_grace_days')} 天 /"
          f" {cfgmod.cfg('capacity', 'citation_grace_days')} 天）——")
    print(f"        如果都给永久保护，印证会一直新增、保护集只涨不消，容量上限就废了。")
    print(f"        三级追溯：追溯包（永远，摘要）→ 场景（保护期内，即时）→ 打捞（期后，慢）。")
    print(f"\n     ⚠️ 这条是 `pending`——所以标「还没印证够，只是猜测」。")
    print(f"        不加区分地呈现，等于把猜测说成事实（那正是「不武断」要防的）。")
    print(f"     ⚠️ 敏感依据不进镜像：air 眼中的你不该含「他有过创伤」这类标签。")

    if mirror["patterns"]:
        print(f"\n     模式统计（跨时间统计这件事人做不到，air 可以）：")
        for pt in mirror["patterns"]:
            reacts = " / ".join(pt["reactions"])
            print(f"       面对【{pt['entity']}】的【{pt['trigger_class']}】→ "
                  f"{pt['count']} 次，典型反应：{reacts}")
        print(f"       统计的是**可观测模式**，不是特质标签——模式可验证、可否决；")
        print(f"       特质（「他是个消极的人」）是武断的标签，那正是要避免的。")

    print(f"\n     否决入口: {mirror['reject_hint']}（can_reject={mirror['can_reject']}）")

    # ------------------------------------------------------------------
    print("\n" + LINE)
    print(" 七、用户否决")
    print(LINE)
    last = store.current_profile_by_topic("用户·面对被评价的反应")
    if last is not None:
        print(f"   用户: 「不对，我不是这样的人」")
        outcome = user_reject_profile(store, last.id)
        print(f"   → {outcome}：{last.statement}")
        print("      真删（2026-09-24 起）——画像是推断：素材都在，")
        print("      这条判断真成立的话，以后有新证据会重新立出来。")
        print("      ⚠️ pending 被否也一样删：否则它将来会被自动扶正，")
        print("         等于系统覆盖了人的否决。")

    print("\n" + LINE)
    print(" 阶段 3 完成。剩下阶段 4：向量持久化与趋势比对")
    print("（「不滞留」的渐变检测——现在只能靠「长期没人提」，还看不出「慢慢变了」）。")
    print(LINE)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
