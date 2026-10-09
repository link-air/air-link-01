"""备忘录：`open_loops` 的可提醒子集。

一句话定位：**备忘录不是「提醒事项」，是「未了结的事」**。
所以它只有一个状态机（管住「不重复提」），记忆本身交给统一的提取链路——
「面试过了」这句话本身就是对话内容，自然会被提取成一条新场景卡，
memo 不需要独立的记忆入口。

四条纪律，都在这个文件里被写成了代码：

1. **时间锚只有一处**：有明确时间直接抽（那是转述，落 `due_at`）；没有的话只让模型
   **分类**（有期限 / 进行中 / 想法倾向 / 要过几天看结果的），窗口由系统按类别给。
   模型没有他的真实节奏，让它估天数必然给一个「平均值」——那是编的。
   ⚠️ 2026-10-05 晚：**「提的时机」整块删了**——到点只说明"**有资格进注入**"
   （见下条），不再有"什么时候提才自然"这回事。

2. **到点进注入一次，提过就停**：到点那件进常备注入**即记账**（转 `raised`）——
   下一轮不再进；他没接话就不管（不追问、不换个说法再来一次）。
   **他问起时她要答得上来**：常备那一栏本来就装着"手上挂着的事"。

3. **只有硬隐私不开口**：`sensitive >= 2`（创伤 / 家事窄名单）**不进注入**；
   其余照常——点出事情本身（「今天面试怎么样」）是给门、不是揭开
   （唤醒层 11.1）。**敏感不是"闭嘴"的同义词**（2026-09-15 起的口径，不变）。

4. **退役只管"不再提醒"**：提的窗口过完还没提成、或提过之后一直没回音，
   转 `closed` 并给钩子标 `retired_at`（**不回流**——放弃 ≠ 完成；判据见
   `retire_due`）。**数据一条不删。**

> 2026-10-05 加两条机制（设计稿 §五，正文各有注释）：
> - **退役**（`retire_due`）：判「**提的窗口**过没过」——起点 `due_at` 或
>   `created + window_days`，+ 2 天；提过之后 3 天没回音（超期未回）同样退役。
>   它替下「窗口 × 3 / 硬顶 90 天」那个钝口径（判"挂了多久"）。退役**不回流**，
>   给钩子标的是 `retired_at`（另一个字段——"办完了"和"没人管了"分得开，渲染侧两者都跳过）；
> - **命中判定**（`judge_hits`）：用户一句话命中挂着的事 → 一次 LLM 判
>   「变更 / 完结 / 无关」，替下词表预筛 `_CLOSURE_MARKS`（词表是枚举世界：
>   2026-10-05 实测「一个结婚了」「才从老家回来」两句结果话，一个词都没命中）。

> **2026-10-05 晚瘦身**（设计稿 §五；决策与证据见待优化稿 K 条）：
> 删掉**时机表**（`raise_ready_at`）· **`should_raise`**（"此刻自不自然"的单独判定）·
> **候选**（`raise_candidates`）· **主动开口整块**（`proactive_open` / `openings` /
> 安静时段 / `开口-*.jsonl`）。现在只有一条线：**到点进注入（最多一件、进即记账）→
> 她自然提 → 没接话就不管 → 窗口过完退役**。
> 动机两句话：**"什么时候提"算得准不准，对聊天几乎没有影响**（他的话："好麻烦，
> 不算了行不行"）；而闹钟那套**至今零次触发**（`openings` 0 条）——删的是一个从没响过的机制。

> 规格 §4.10 写的是一个 `MemoStore` 类，这里做成了**一组函数**：
> 表的 CRUD 已经在 `Store` 里（唯一落库出口），再包一个类只是多一层纯转发。
> 状态流转本身没有需要保持的实例状态——它是「对库做四个动作」，不是「一个东西」。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L8 备忘录
#   上游    ：config、model、prompts、store（CRUD 在那边）
#   下游    ：distill（`_write_memos` 与 `memo_cycle`）、recall（`standing_memos`）、
#             shortterm（`judge_hits`）、chat（`mark_raised`、到点那件的记账）、
#             tools（`close`，延迟 import）、dashboard（`close` / `due`）
#   对外入口：`due` / `standing_memos` / `judge_hits` / `hit_candidates` /
#             `retire_due` / `memo_cycle`
#   边界    ：**不认识对话层**——它只回答"哪些到点了、哪些该退役了"；
#             注不注入、怎么说是 recall / chat 的事
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from datetime import datetime, timedelta

from . import config as cfgmod
from .model import (MEMO_CLASSES, MEMO_PENDING, MEMO_RAISED, MEMO_TIMING_SOON,
                    MEMO_TIMINGS)
from .prompts import (MEMO_CLASS_SCHEMA, MEMO_GROUP_SCHEMA, MEMO_HIT_SCHEMA,
                      memo_class_prompt, memo_group_prompt, memo_hit_prompt)
from .store import append_trace, now_str

# ---------------------------------------------------------------------
# 一、时间分类（只分类，不估天数）
# ---------------------------------------------------------------------

def classify_window_kinds(store, memos: list, llm) -> int:
    """给备忘录补「类别窗口」与「时机类别」，返回处理的条数。

    两种缺法都补（2026-09-15 加时机后；2026-10-05 晚时机收窄到 soon / later）：
      - 缺 `timing` 的——**有具体时间的也要**（两件事互不相干：`due_at` 说"哪天到期"，
        `timing=soon` 说"身体状况，别等窗口"）；
      - 无具体时间且缺类别的——补 `kind_class`，窗口由系统映射。

    一次调用判一批（而不是一条一次）：未闭合的事常常同时有好几件，
    逐条调用的成本是线性的，而它们本来就是同一类判断。

    **拿不到分类就不设**（宁可不设，也不猜一个类别）——
    类别错了窗口就跟着错，而错窗口会让它在该提的时候不提。
    """
    targets = [m for m in (memos or [])
               if not m.timing or (not m.due_at and not m.kind_class)]
    if not targets:
        return 0

    data = llm.structured(memo_class_prompt([m.content for m in targets]), MEMO_CLASS_SCHEMA)
    # 用 content 回显来对应（而不是数组下标）：模型少回一条时不会整体错位
    by_content = {}
    for item in data.get("items") or []:
        if isinstance(item, dict) and item.get("content"):
            by_content[str(item["content"]).strip()] = item

    defaults = cfgmod.cfg("memo", "window_defaults", default={}) or {}
    done = 0
    for m in targets:
        item = by_content.get(m.content) or {}
        touched = False
        timing = item.get("timing")
        if timing in MEMO_TIMINGS:
            store.set_memo_timing(m.id, timing)
            touched = True
        # 有明确时间的不映射周期（那是用户明说的，不是估的）
        cls = item.get("class")
        if not m.due_at and cls in MEMO_CLASSES:
            days = defaults.get(cls)
            if days:
                store.set_memo_class(m.id, cls, int(days))
                touched = True
        if touched:
            done += 1
    return done


def classify_window_kind(content: str, llm) -> str:
    """单条分类。

    返回值是类别字符串，判不出来返回空串。
    """
    data = llm.structured(memo_class_prompt([content]), MEMO_CLASS_SCHEMA)
    for item in data.get("items") or []:
        if isinstance(item, dict):
            cls = item.get("class")
            if cls in MEMO_CLASSES:
                return cls
    return ""


def classify_group_names(store, memos: list, llm) -> int:
    """给**还没有组名**的备忘录补「事项组」短名，返回处理的条数（2026-09-22）。

    在会话边界（`memo_cycle`）跑一次——同 `classify_window_kinds` 的先例：
    一次调用判一批，成本恒定；**全都已有组名时不调模型**（零收益不花钱）。

    为什么需要事后补：组名是抽取时写的；本机制上线前的老数据没有，而写入那一刻
    模型也不一定看得出"这两条是一件事"。补判把两边都兜住：
      - 只填空组名的条目；
      - 已有组名只作为**参考**列进 prompt（合适就归进去），**已有的一个字都不动**——
        改它会让同一件事裂成两个名字。

    拿不到就不设（`items` 空 → 返回 0）：空组名 = 各自一条，不算坏结果。
    """
    targets = [m for m in (memos or [])
               if not (getattr(m, "group_name", "") or "").strip()]
    if not targets:
        return 0
    existing = sorted({(getattr(m, "group_name", "") or "").strip()
                       for m in (memos or [])
                       if (getattr(m, "group_name", "") or "").strip()})
    data = llm.structured(memo_group_prompt([m.content for m in targets], existing),
                          MEMO_GROUP_SCHEMA)
    # 用 content 回显来对应（同 `classify_window_kinds`）：模型少回一条时不会整体错位
    by_content = {}
    for item in data.get("items") or []:
        if isinstance(item, dict) and item.get("content"):
            by_content[str(item["content"]).strip()] = \
                str(item.get("group_name") or "").strip()
    done = 0
    for m in targets:
        g = by_content.get(m.content, "")
        if g:
            store.set_memo_group(m.id, g[:20])
            done += 1
    return done


# ---------------------------------------------------------------------
# 二、到期与提起
# ---------------------------------------------------------------------

def due(store, now: str | None = None) -> list:
    """**到点了**的备忘录（有 `due_at` 已过 / 无 `due_at` 超窗口）。

    2026-10-05 晚改口径：它就是"**有资格进注入**"——不再有时机表、不再要
    `should_raise` 再判一次"此刻自不自然"（那两道都删了）。到点那件由
    `standing_memos` 放进常备注入，提不提、怎么提是提示词里的分寸。
    """
    now = now or now_str()
    cands = list(store.due_memos(now))
    ids = {m.id for m in cands}
    # soon（身体状况）**不等常规窗口**：`created + soon_hours` 一到就算到点——
    # 「越早问越暖」的实现就在这里（常规窗口的粒度是"天"，对它太慢了）。
    soon_hours = int(cfgmod.cfg("memo", "soon_hours", default=4) or 4)
    for m in store.open_memos():
        if m.id in ids or m.status != MEMO_PENDING or (m.sensitive or 0) >= 2:
            continue
        if (m.timing or "") == MEMO_TIMING_SOON and not (m.due_at or "").strip():
            gate = _shift(m.created_at, hours=soon_hours)
            if gate and gate <= now:
                cands.append(m)
    return cands


def _group_raised(store, memo) -> bool:
    """这件事（组）**已经提过**了吗——提过就不再提它的其它步骤。

    为什么要有它（2026-09-22）：一组多步（「设置搜索」+「跑验收」）会各自到期、
    各自进候选——她被分两轮催同一件事，像催办。「提」的单位是**这件事**：
    组里任何一条被提过（`raised`），其余步骤就不再单独提；只有在那之后
    **新出现**的步骤（`created_at` 晚于那条的 `raised_at`）才重新有机会——
    那是有新进展，值得再说一次。

    无组名（独立的一件事）返回 False：那是逐条算的，跟以前一样。
    """
    g = (getattr(memo, "group_name", "") or "").strip()
    if not g:
        return False
    for m in store.open_memos():
        if m.id == memo.id or (getattr(m, "group_name", "") or "").strip() != g:
            continue
        if m.status != MEMO_RAISED:
            continue
        if (m.raised_at or "") >= (memo.created_at or ""):
            return True
    return False


def standing_memos(store, msg: str = "", emb=None,
                   full: bool = False, limit: int = 4,
                   now: str | None = None) -> list[dict]:
    """常备备忘录：**她手上得有的那几件**（2026-09-21，设计稿 D2b / F）。

    2026-10-05 晚起它是**唯一**那条"进对话"的路（候选与主动开口都删了）——
    两条通道并成这一栏（设计稿 §五「到点进注入」）：
      - **到点那 1 件**：`due()` 出来、未提过（`raised` / 同组没提过）→ 进注入
        **并记账**（返回值带 `"due": True`，由 chat 那一侧标 `raised`）；
      - **手上得有**：他问起、或话题接得上时，答得出来是哪件（07:04 的失败正是
        "她手上没有"，不是"她没提"）。

    正常轮按需挑（成本恒定）：**到点 1 + 相关先挑满 + 最近补位（≤2）**，
    去重后封顶 `limit`；`full=True`（他正**追问指代**，F 条）全列、封顶 8——
    那一刻她要的是"认得出来"，不是"省着给"。

    三个位子的判据（每一个都要说得出为什么）：
      - **到点**：见上（`due()` 已经算好"到没到"，这里只管挑）；
      - **相关**：**同一把相关性尺**——借**它那张场景卡的向量**（memo 自己没有 emb），
        没向量 / 卡没向量时退回 `char_overlap`（与 `recall` 的降级口径同源）；
      - **最近**：按 memo 的 `created_at`——**memo 没有"被提及"时间戳**（卡才有
        `last_mention_at`）；先用简单的，真不够再借卡那个。

    `sensitive >= 2` 不进：那是「只记不提」的窄名单，比"给门"更保守一档。
    `followup` **不再单独排除**（2026-10-05 晚）：它跟别的 memo 走同一条
    （到点进一次、进即记账）；原来那套"一生不进常备栏"随候选机制一起没了。
    """
    # `raised`（提过）的**不进常规轮**（2026-09-22）：「提过一次就不再提」——
    # 她在对话里提过的事，下一轮不该再被端到眼前（M-0012 被提两次，正因为
    # 常备栏这一路不认 raised）。`full=True`（他正追问指代）时照旧**全列**：
    # 那一刻要的是"答得上来"——「不再主动提」不等于「可以忘掉」（同 open_memos 的注释）。
    opens = [m for m in store.open_memos()
             if (full or m.status == MEMO_PENDING)
             and int(getattr(m, "sensitive", 0) or 0) < 2]
    if not opens:
        return []

    picked: list[dict] = []

    def add(m, why: str, due_: bool = False) -> None:
        item = {"id": m.id, "content": m.content or "",
                "kind": m.kind or "",
                # 组名带出去（2026-09-22）：注入侧与界面靠它把
                # "一件事的几步"显示成一件事
                "group_name": (getattr(m, "group_name", "") or ""),
                "why": why}
        if due_:
            # 到点的那一件：**进注入即记账**（chat 那一侧认这个标记标 `raised`）。
            # 标在数据里而不是靠口头约定——约定会被忘，字段不会（同原 `mode: "door"` 的道理）。
            item["due"] = True
        if all(x["id"] != m.id for x in picked):
            picked.append(item)

    if full:
        for m in opens:
            add(m, "追问")
        return picked[:8]

    # ① **到点的那一件**（2026-10-05 晚并栏：原来"候选 / 主动开口"两路并到这里）：
    # 未提过（`raised` 不进 `opens`）、同组也没提过才算——到点**不是"每轮可见"**，
    # 给了机会就算给过（进即记账），下一轮不再进。它同时是"到点"这条通道的全部落点：
    # 一次最多一件，其余的下一轮接着排（没记账的不算提过）。
    for m in due(store, now):
        if _group_raised(store, m):
            continue        # 这件事提过一回了——别再拿它的下一步催（分两轮催像催办）
        add(m, "到点", due_=True)
        break

    q = (msg or "").strip()
    if q:
        # ③ 语义相关：优先借场景卡的向量（memo 自己没有 emb）；
        #    没向量 / 卡没向量时退回字符重叠（同 recall 的降级口径）。
        from .embedding import cosine
        from .recall import char_overlap
        qv = emb.embed_one(q) if emb is not None else None
        vecs = dict(store.all_embeddings()) if qv else {}
        scored: list[tuple[float, object]] = []
        for m in opens:
            if any(x["id"] == m.id for x in picked):
                continue
            v = vecs.get(getattr(m, "scene_id", "") or "")
            sc = cosine(qv, v) if (qv and v) else char_overlap(q, m.content or "")
            if sc > 0:                        # 0 分不要（同 memory_search 的经验）
                scored.append((sc, m))
        scored.sort(key=lambda x: -x[0])
        for _sc, m in scored[:limit]:
            add(m, "相关")                    # ② **先挑满**（2026-09-22 改）

    # ③ 最近发生的：**补位**（最多 2 条，原「最近 2」的上限不变）。
    # 原先「最近」固定先占 2 个位子：他输入相关的内容时，真正对得上的那两条
    # 不一定都进得来。改成"相关先挑满、不够才用最近补"——相关时给的全是对得上的；
    # 一条都不相关时最近的事兜底（刚交代的事她手上总有，那是"新鲜度"那半边）。
    if len(picked) < limit:
        for m in sorted(opens, key=lambda x: x.created_at or "", reverse=True)[:2]:
            add(m, "最近")

    return picked[:limit]


def mark_raised(store, mid: str) -> bool:
    """待提 → 已提。**之后不再主动提**（提过一次就够了）。

    返回"真的转了吗"：**已关闭的不动、也不报**——同一轮里被**命中判定**
    了结掉的那条不该被记账回「已提」（它钩子已经标了 `closed_at`，
    却会重新挂回「备忘录」清单）。
    """
    return store.mark_memo_raised(mid)


def close(store, mid: str, by: str = "") -> None:
    """关闭（用户给了结果）——**顺带把场景卡里的钩子标掉**（闭合回流）。

    为什么要回流（2026-09-21，设计稿 D 条 2）：`open_loops` 与 `memos` 是
    "同一个事实存两份、状态只更新一份"——memo 关了、卡里的钩子还挂着，
    渲染和清单就会继续端出已经了结的事。**`memos` 是状态的唯一事实源**，
    这里关的时候顺手把源头标掉（`store.close_open_loop`）；渲染侧也认
    `closed_at` 字段——一处判、两处生效。

    `by`：**谁关的**（命中判定 / 她调工具 / 用户随手划）。留痕写在这里而不是
    各发起方自己写（2026-09-22）：分散写会漏——「她判」这条原来
    一条留痕都没有，「面试那句为什么把它关了」答不出来。

    ⚠️ **超期退役不走这里**（`retire_due` 直接 `store.close_memo` + 给钩子标
    `retired_at`）——退役 ≠ 完成：钩子该继续开着，只是系统不再提醒了。
    """
    m = store.get_memo(mid)
    store.close_memo(mid)
    if m is not None and m.scene_id:
        # 按 `loop_id` 定位钩子（2026-10-05）：memo 的 content 变更过也找得回；
        # 老数据没有编号 → store 那边退回 content 全等（以前唯一的匹配方式）。
        store.close_open_loop(m.scene_id, m.content,
                              loop_id=(getattr(m, "loop_id", "") or ""))
    if m is not None:
        write_memo_trace("关闭", [{"id": mid, "content": m.content, "by": by}])


def write_memo_trace(act: str, changes: list[dict]) -> None:
    """备忘录状态改动的留痕（`data/trace/备忘-YYYYMMDD.jsonl`）。

    三个关闭发起方共用这一份（2026-09-22 从 dashboard 下移——留痕不该只在
    界面层写，她判 / 她调工具那两条路同样要留下"为什么不再提了"的答案）。
    写不成不拦改动本身（同各处 trace 的兜底）。
    """
    append_trace("备忘", [{"ts": now_str(), "act": act, **c} for c in changes])


# ---------------------------------------------------------------------
# 三、命中判定（用户这一句碰到了哪几件事，各是变更还是完结）
#
# 2026-10-05 取代「了结词预筛 + 闭合判定」：
#   词表是**枚举世界**——他 00:32 说「一个结婚了」、00:33 说「才从老家回来」，
#   两句都是结果，`_CLOSURE_MARKS` 一个词都没命中，连 LLM 都没调，
#   M-0013 / M-0014 一直挂着。补词只是把漏判往后推，不是机制。
# 现在：命中粗筛（内容相关 + 刚提过的，不用字面）→ 一次 LLM 判一批
#   （`none` / `update` / `close`）——判据是内容，分类交给模型。
# ---------------------------------------------------------------------

def _minutes_ago(stamp: str, minutes: int) -> str:
    """时间戳 - N 分钟（解析不了返回空串——不猜）。"""
    return _shift(stamp, minutes=-minutes)


def hit_candidates(store, msg: str, emb=None, now: str | None = None) -> list:
    """命中粗筛：这条消息**可能碰到了哪几件**挂着的事（2026-10-05）。

    两条路并起来，都不用字面：
      - **全量**：未闭合条数少（≤ `hit_full_n`）时直接全给——判据是内容，不是阈值；
      - 超出时按**相关度**取前 `hit_top_n`（借场景卡向量 + `char_overlap` 降级，
        与 `standing_memos` 的"相关"同一套口径）；
      - 再加「**刚提过的**」（`raised_at` 在 `hit_grace_minutes` 内）——
        "她刚提、他随口答"是最常见的闭合情形，那一刻必须进判定
        （2026-10-05 的断点正是：M-0014 提完就 `raised`，他给结果时它已不在她注入里）。

    `sensitive >= 2`（硬隐私）照给：它只是"不提"，不是"不能了结 / 不能变更"。
    """
    opens = store.open_memos()
    if not opens:
        return []
    full_n = int(cfgmod.cfg("memo", "hit_full_n", default=20) or 0)
    if full_n > 0 and len(opens) <= full_n:
        return opens

    top_n = int(cfgmod.cfg("memo", "hit_top_n", default=8) or 8)
    grace_min = int(cfgmod.cfg("memo", "hit_grace_minutes", default=60) or 0)
    now_txt = now or now_str()
    cut = _minutes_ago(now_txt, grace_min) if grace_min else ""

    picked: list = []

    def add(m) -> None:
        if all(x.id != m.id for x in picked):
            picked.append(m)

    for m in opens:                      # ① 刚提过的：命中窗口里的闭合时机
        if cut and (m.raised_at or "") >= cut:
            add(m)

    q = (msg or "").strip()
    if q:
        from .embedding import cosine
        from .recall import char_overlap
        qv = emb.embed_one(q) if emb is not None else None
        vecs = dict(store.all_embeddings()) if qv else {}
        scored: list[tuple[float, object]] = []
        for m in opens:
            if any(x.id == m.id for x in picked):
                continue
            v = vecs.get(getattr(m, "scene_id", "") or "")
            sc = cosine(qv, v) if (qv and v) else char_overlap(q, m.content or "")
            if sc > 0:                   # 0 分不要（同 memory_search 的经验）
                scored.append((sc, m))
        scored.sort(key=lambda x: -x[0])
        for _sc, m in scored[:top_n]:
            add(m)                       # ② 相关度：够不够由下面那次判定兜着
    return picked


def _remap_window(store, mid: str, content: str, llm) -> None:
    """内容变更后按**新内容**重映射窗口（2026-10-05）。

    复用写入路径的同一套（`classify_window_kind` + `window_defaults`）——
    铁律不破：**模型只判类别，天数由系统定**；时间仍只在他明说时抽
    （那半由 `store.update_memo_content(due_at=)` 负责）。拿不到分类就不动。
    """
    cls = classify_window_kind(content, llm)
    days = (cfgmod.cfg("memo", "window_defaults", default={}) or {}).get(cls)
    if cls and days:
        store.set_memo_class(mid, cls, int(days))


def judge_hits(store, msg: str, llm, now: str | None = None, emb=None) -> dict:
    """命中判定：这一句碰到了哪几件挂着的事、各是什么动作（2026-10-05）。

    返回 `{"closed": "M-0014" | "", "updated": [...], "judged": [...]}`；
    `closed` 供那一轮回执用（`ShortTerm.last_closed_memo` → `memo_closed`）。

    三种动作只做两件事（`none` 什么都不做）：
      - `close` → 走带回流的那条 `close()`（钩子一起标掉 + 留痕）；
      - `update` → 内容变更（`store.update_memo_content` + 窗口重映射 + 留痕
        「旧 → 新」）——**只在模型判"实质变更"时发生**（防反复命中反复续命）。

    模型回的 `why` **不进留痕**（`write_memo_trace` 的字段里没有它的位置：编号 / 内容 /
    谁关的，变更另带旧内容 `was`、退役另带系统算的理由）——它是自述理由，当前没有任何
    读点（要留它得改这一处）。

    **判不准就 `none` / LLM 不可用什么都不做**——误关比多提贵（同名纪律：
    `close_memo` 那条"判不准就别关"）。
    """
    if not (msg or "").strip():
        return {"closed": "", "updated": [], "judged": []}
    cands = hit_candidates(store, msg, emb=emb, now=now)
    if not cands:
        return {"closed": "", "updated": [], "judged": []}

    data = llm.structured(memo_hit_prompt(msg, cands), MEMO_HIT_SCHEMA)
    by_id = {m.id: m for m in cands}
    closed, updated, judged = "", [], []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        action = str(item.get("action") or "none").strip().lower()
        m = by_id.get(mid)
        # 不在粗筛里 / 无关 / **同一条被给了两次动作**（模型偶发重复）——一律不动作；
        # 尤其是"先 close 后 update"这种自相矛盾的输出：以先到的为准，不让后者翻案。
        if m is None or action == "none" or mid in judged:
            continue
        if action == "close":
            close(store, mid, by="命中判定（用户给了结果）")
            judged.append(mid)
            if not closed:
                closed = mid
        elif action == "update":
            new = str(item.get("content") or "").strip()
            if not new or new == (m.content or "").strip():
                continue                  # 没给新话 / 跟原来一样：不算变更
            due = str(item.get("due_at") or "").strip()
            store.update_memo_content(mid, new, due_at=due or None)
            if not due:                   # 明说了新时间就不动窗口（那是转述，不是映射）
                _remap_window(store, mid, new, llm)
            write_memo_trace("变更", [{"id": mid, "content": new, "by": "命中判定",
                                       "was": m.content}])
            judged.append(mid)
            updated.append(mid)
    return {"closed": closed, "updated": updated, "judged": judged}


# ---------------------------------------------------------------------
# 四、后台维护
# ---------------------------------------------------------------------

def retire_due(store, now: str | None = None) -> list[dict]:
    """**超期退役**（2026-10-05，取代「窗口 × 3 / 90 天硬顶」）：提的窗口过完仍未提成
    → 转 `closed`（不再提醒）。返回退役的条目（调用方拿它写留痕）。

    判的是「**提的窗口**过没过」，不是"挂了多久"：

    | 情形 | 起点 | 退役时刻 |
    |---|---|---|
    | 未提过的 | `due_at`（他明说的时间）或 `created_at + window_days`（时机表 2026-10-05 晚已删） | 起点 + `retire_grace_days`(2 天) |
    | 已提过的（超期未回） | —— | `raised_at` + `retire_after_raised_days`(3 天) |

    三条纪律：
      1. **按绝对时间走**——窗口不因"她当时不在线"而停表（退役不是删除，是"不再提醒"；
         错过了就是错过了，同"像个一次性的自动化任务"的定位）；
      2. **退役不回流**（"放弃 ≠ 完成"，2026-09-21 定）：钩子继续开着，只给它标
         `retired_at`（**另一个字段**——渲染侧一样跳过，但"办完了"和"没人管了"分得开）；
      3. 留痕由调用方写（`memo_cycle`）——`by="超期退役"`。

    `sensitive >= 2`（硬隐私）**也退役**：它们从不进候选，但同样占着「未了结」的位置。
    """
    now = now or now_str()
    grace = int(cfgmod.cfg("memo", "retire_grace_days", default=2) or 0)
    no_reply = int(cfgmod.cfg("memo", "retire_after_raised_days", default=3) or 0)
    out: list[dict] = []
    for m in store.open_memos():
        if m.status == MEMO_RAISED:
            if not no_reply or not (m.raised_at or "").strip():
                continue
            deadline = _shift(m.raised_at, days=no_reply)
            why = f"超期未回：提过之后 {no_reply} 天没回音（{m.raised_at}）"
        else:
            if not grace:
                continue
            # 起点（2026-10-05 晚改）：`due_at`（他明说的时间）或 `created + window_days`。
            # 原来是 `raise_ready_at()`（时机表）——那张表已删，判据就这一条。
            start = ((m.due_at or "").strip()
                     or _shift(m.created_at, days=m.window_days or 30))
            deadline = _shift(start, days=grace) if start else ""
            why = f"提的窗口走完仍未提（{start} → {deadline}）"
        if not deadline or deadline > now:    # 时间戳格式统一，按字典序可比
            continue
        store.close_memo(m.id)
        store.retire_open_loop(m.scene_id, m.content,
                               loop_id=(getattr(m, "loop_id", "") or ""), at=now)
        out.append({"id": m.id, "content": m.content, "why": why})
    return out


def memo_cycle(store, llm, now: str | None = None) -> dict:
    """后台备忘录维护：补分类 + 补组名 + **超期退役**（并入提炼周期跑）。

    退役于 2026-10-05 替下"超期放弃"（原 `store.give_up_memos`：窗口 × 3 / 硬顶 90 天）——
    旧口径判"挂了多久"，新口径判"提的窗口过没过"（见 `retire_due`）。
    **退役留痕**（`by="超期退役"`）：它是四个关闭发起方里唯一原来没留痕的那个，
    「这件事为什么到死没被提过」答不出来。
    """
    opens = store.open_memos()
    classified = classify_window_kinds(store, opens, llm)
    # 组名补判（2026-09-22）：只填空的那几条——老数据与新出现的都在这里兜住；
    # 全都已有组名时它内部直接返回，不产生调用（见 `classify_group_names`）。
    grouped = classify_group_names(store, opens, llm)
    retired = retire_due(store, now)
    if retired:
        write_memo_trace("关闭", [{"id": r["id"], "content": r["content"],
                                   "by": "超期退役", "why": r["why"]}
                                  for r in retired])
    return {"classified": classified, "grouped": grouped,
            "retired": len(retired), "open": len(store.open_memos())}


# ---------------------------------------------------------------------
# 五、时间算术（原「五、主动开口」2026-10-05 晚整块删除——见 K 条：
#     时机表 / 安静时段 / 留言 / 留痕全不要了，只留下这个加法）
# ---------------------------------------------------------------------


def _shift(stamp: str, *, days: int = 0, hours: int = 0, minutes: int = 0) -> str:
    """时间戳 + N 天 / 小时 / 分钟（字符串进、字符串出；给负数就是往前算）。

    加法必须过 datetime（同 `store._add_days` 的理由：手写日期进位
    迟早错在月底和闰年上）。解析不了就返回空串——**不猜**。
    """
    s = (stamp or "").strip()
    for fmt, cut in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d %H:%M", 16),
                     ("%Y-%m-%d", 10)):
        try:
            t = datetime.strptime(s[:cut], fmt)
        except ValueError:
            continue
        return (t + timedelta(days=days, hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    return ""
