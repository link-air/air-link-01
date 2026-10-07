"""工具箱：她的手。

设计记录见本地「工具箱」稿（不随仓库分发）。四条在这里落地：

  1. **档位分明**：自主（auto）/ 需确认（ask）/ 只读（read）。
     **没有确认通道时，需确认档一律拒绝**（失败即拒绝）——
     少做一件事的代价，远小于自作主张地把一条记忆改坏。
  2. **红线终审单调**：只有「拒绝」没有「允许」，任何后续逻辑都不能翻案。
  3. **每次调用都留痕**（`data/trace/工具-*.jsonl`）：她改记忆的动作，
     和他改记忆一样，都要能回看。
  4. **参数认别名**：DeepSeek 没有严格的参数约束（`json_schema` 档被降级过），
     指望它把参数写对不如代码认下来——这跟 `_norm_open_loops` 认
     `item`/`type` 是同一条经验。

工具名保持英文：那是给模型看的，中文名她认不准。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L9 工具箱（她的手）
#   上游    ：config、embedding（语义排序）、store（读）
#   下游    ：chat（唯一的执行方：一个工具循环里逐次调 `execute`）
#   对外入口：`TOOLS`（工具表）/ `openai_tools` / `tool_names` / `execute`
#   边界    ：提议类动作（改 / 删）**一个字都不改**，只登记进 `ctx["proposals"]`；
#             真执行走 `weave` 那几个 `*_confirmed`（和界面按钮同一个动作）
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import re
from datetime import datetime

from . import config as cfgmod
from . import webfetch
from .embedding import cosine
from .model import (INVALIDATED_ARCHIVE,
                    MEMO_AIR_PROMISE, MEMO_PENDING, MEMO_USER_TASK,
                    PROFILE_ESTABLISHED)
from .prompts import rel_stamp, scene_line
from .store import TOPICS_MAX, layer_edit, now_str

# 三档（设计稿第二节）
AUTO, ASK, READ = "auto", "ask", "read"


# ---------------------------------------------------------------------
# 参数规整：认别名、认类型不对的写法
# ---------------------------------------------------------------------

# 每个规范名认哪些写法。她可能写 `{id: ...}` 而不是 `{scene_id: ...}`，
# 也可能把参数塞在别的键里——都认下来（同 `_norm_open_loops` 的经验）。
_ALIASES = {
    "query": ("query", "q", "keyword", "关键词", "text", "查"),
    # 「一条记忆的编号」——**一个规范名收三种编号**（三层同一套的改与删，
    # 2026-09-24）：S1 / S2 / S3 以及 M（M 由代码引导去 `close_memo`）。
    # 旧名字（`scene_id` / `memo_id` / 场景 / 画像 / 备忘 / scene / memo）照旧认。
    # ⚠️ 同一个规范名只留一条（2026-09-24 检查修）：原来有两份 `"id"`（前一份被
    # 静默覆盖，是死代码）、`"scene_id"` / `"memo_id"` 两个独立键（只被主调用的
    # `or` 冗余用到）、以及 `"profile_id"` / `"reason"`（`reject_profile` 时代的
    # 遗产，没有调用方）——现在只要编号，一律走这一条（`_pick(args, "id")`）。
    "id": ("id", "编号", "记忆编号", "scene_id", "profile_id", "memo_id",
           "场景", "画像", "备忘", "scene", "memo", "未了结"),
    "layer": ("layer", "层", "哪层", "哪一层", "which"),
    # 按主题找（2026-09-24 晚）：三层通用——S2 / S3 的「搜不到」在这条路上解决
    "topic": ("topic", "主题", "话题", "标签"),
    "entity": ("entity", "实体", "谁", "人名", "名字"),
    "when": ("when", "时间", "日期", "date", "day", "什么时候"),
    "text": ("text", "内容", "改成", "改为", "新值", "value", "new"),
    "url": ("url", "link", "href", "网址", "链接", "地址", "page"),
}

# 参数值可能是任何 JSON 类型（数字、布尔），一律转成字符串再判空
def _pick(args: dict, canonical: str) -> str:
    """按规范名取参数（**认别名**，见段前的 `_ALIASES`）。找不到返回空串。"""
    for name in _ALIASES.get(canonical, (canonical,)):
        if name in (args or {}):
            v = args[name]
            if isinstance(v, (list, dict)):
                continue
            s = str(v).strip()
            if s:
                return s
    return ""


def _split_ids(raw: str) -> list[str]:
    """一个参数里可能填了多个编号（逗号 / 顿号 / 空格都认）。

    为什么要认这么多种：她写编号是自由文本，而**参数约束不可靠**
    （DeepSeek 的 `json_schema` 档被降过级）——认下来比让她重写一遍快。
    """
    return [x.strip() for x in re.split(r"[,，、;；\s]+", raw or "") if x.strip()]


def _norm_date(raw: str) -> str:
    """把「2026年9月3日」「2026/9/3」规整成 `2026-09-03`。

    只做**格式**规整，不做语义换算：「上个月」这类相对说法仍然是她的事
    （同 memo 那条纪律：不让模型估时间）——这里一个字都不猜，
    规整不出来就由调用方**明确拒绝**，而不是当成"没给"静默放过。
    """
    s = (raw or "").strip()
    for a, b in (("年", "-"), ("月", "-"), ("日", ""), ("/", "-"), (".", "-")):
        s = s.replace(a, b)
    parts: list[str] = []
    for p in s.split("-"):
        d = "".join(ch for ch in p if ch.isdigit())
        if not d:
            break
        parts.append(d if not parts else d.zfill(2))     # 第一个是年，不补零
    return "-".join(parts[:3])


def _clip(s: str, n: int = 60) -> str:
    """截断并带 `…`：**不带标记的话，模型会把半截当完整的**。"""
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------
# 工具的实现
#
# 每个都返回 {"ok": bool, "detail": str}——
# 失败时 `detail` 里要说清「怎么才是对的」（设计稿防错第 4 条），
# 她下一轮就能自己改对，比"失败就放弃"强得多。
# ---------------------------------------------------------------------


_LAYER_ALIASES = {"s1": "s1", "scene": "s1", "场景": "s1",
                  "s2": "s2", "summary": "s2", "摘要": "s2",
                  "s3": "s3", "profile": "s3", "画像": "s3",
                  "memo": "memo", "m": "memo", "备忘": "memo"}


def _norm_layer(raw: str) -> str:
    """`layer` 参数收 s1/s2/s3/memo（认中英文别名）。**认不出的当没给**——
    比报错好：她给个奇怪的词时，列最近的比"什么都没有"有用。"""
    return _LAYER_ALIASES.get((raw or "").strip().lower(), "")


def _truthy(v) -> bool:
    """宽松认布尔（`raw` 这类参数）：模型可能给 true / "true" / "是"——认不出算假。"""
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "是", "真", "要")


def _expand_by_id(store, mid: str, args: dict, ctx: dict) -> dict:
    """编号直查 = **展开一条**（2026-09-24 合并原 `memory_read` / `memory_trace`）：

    S1 → 一行 + 被谁引用（`raw=true` 再带原话——**贵的一步显式要**）；
    S2 → 叙述 + 它的素材；S3 → 判断 + 依据链；M → 备忘一条。
    """
    up = (mid or "").strip().upper()
    if up.startswith("S1"):
        out = _expand_scene(store, mid)
        if out.get("ok") and _truthy(_pick(args, "raw")):
            raw = _read_raw(store, mid)
            if raw.get("ok"):
                # 真读了一段原话：下钻计数在这里记（提醒由 `_limit_note` 附）
                ctx["drill_used"] = (ctx.get("drill_used") or 0) + 1
                return {"ok": True, "detail": out["detail"] + "\n\n" + raw["detail"]}
        return out
    if up.startswith("S2"):
        return _expand_summary(store, mid)
    if up.startswith("S3"):
        return _expand_profile(store, mid)
    return _by_memo(store, mid)         # M 与"认不出"的兜底


def _expand_scene(store, sid: str) -> dict:
    """展开一场事：一行 + 有哪些画像引用它（含历史版本）+ 它在哪些摘要里。"""
    s = store.get_scene(sid)
    if s is None:
        return {"ok": False,
                "detail": f"没有 {sid}——列一遍用 memory_search。"}
    lines = [scene_line(s)]
    if (s.topic or "").strip():
        lines.append(f"主题：{s.topic}")
    citing = store.profiles_citing(sid, only_current=False)
    if citing:
        lines.append("引用它的画像：")
        for p in citing[:8]:
            lines.append(f"  - {p.id} {p.statement}"
                         + ("（已作废）" if p.invalidated_at else ""))
    else:
        lines.append("还没有画像引用它。")
    inside = []
    for t in store.list_topics():
        for s2 in store.summaries_by_topic(t):
            if sid in (s2.sources or []):
                inside.append(s2.id)
    if inside:
        lines.append("它在这些摘要里：" + "、".join(inside[:8]))
    return {"ok": True, "detail": "\n".join(lines)}


def _expand_profile(store, pid: str) -> dict:
    """展开一条判断：陈述 + 依据链（`evidence_pack` 快照 + 累积印证）。

    依据里**已删的不显示**（引用 = `sources ∩ 现存节点`，2026-09-24）——
    快照只用来认"出处 / 印证"，不拿它把删掉的场景端出来。
    """
    p = store.get_profile(pid)
    if p is None:
        return {"ok": False,
                "detail": f"没有 {pid}——列画像用 memory_search（layer 填 s3）。"}
    mark = "已立" if p.status == PROFILE_ESTABLISHED else "只是猜测"
    if p.invalidated_at:
        mark = "已归档冷存" if p.invalidated_by == INVALIDATED_ARCHIVE else "旧版本（进过历史）"
    lines = [f"{p.id}：{p.statement}（{mark} · {p.evidence} 次印证）",
             f"主题：{'、'.join(p.topics or [p.topic])}",
             f"形成于 {(p.valid_at or p.created_at or '')[:10]}"
             + (f"，最近印证 {(p.last_support_at or '')[:10]}"
                if p.last_support_at else "")]
    pack = [it for it in (p.evidence_pack or [])
            if isinstance(it, dict) and store.exists(str(it.get("id") or ""))]
    if pack:
        lines.append("形成时的依据（快照）：")
        for it in pack[:8]:
            lines.append(f"  - {it.get('id', '')}「{it.get('title', '')}」")
    elif p.evidence_pack:
        lines.append("形成时的依据：都删了。")
    else:
        lines.append("形成时的依据（快照）：（这份是早期记录，没留快照）")
    src = [x for x in (p.sources or [])
           if str(x).upper().startswith("S1") and store.exists(x)]
    if src:
        lines.append("后来累积的印证（场景）：")
        for x in src[:8]:
            s = store.get_scene(x)
            lines.append(f"  - {x}「{s.title or ''}」")
    return {"ok": True, "detail": "\n".join(lines)}


def _expand_summary(store, sid: str) -> dict:
    """展开一条叙述：文本 + 它的素材（由哪些场景聚合而来）。

    S2 从 2026-09-24 起也能这么看（此前只给"不单独引用"的引导）——
    三层同一套：给它编号，展开它自己的东西。
    """
    s2 = store.get_summary(sid)
    if s2 is None:
        return {"ok": False,
                "detail": f"没有 {sid}——列摘要用 memory_search（layer 填 s2）。"}
    head = f"{s2.id}（{(s2.created_at or '')[:10]}）{s2.text or ''}"
    if s2.archived:
        head += "（已归档冷存）"
    lines = [head, f"主题：{'、'.join(s2.topics or [s2.topic])}"]
    src = [i for i in (s2.sources or [])
           if str(i).upper().startswith("S1") and store.exists(i)]
    if src:
        lines.append("它的素材（场景）：")
        for x in src[:8]:
            s = store.get_scene(x)
            lines.append(f"  - {x}「{s.title or ''}」")
    elif s2.sources:
        lines.append("（素材都删了）")
    else:
        lines.append("（没有记下素材——早期摘要）")
    return {"ok": True, "detail": "\n".join(lines)}


def _by_memo(store, mid: str) -> dict:
    """备忘一条（M-xxxx）；认不出的编号给形状提示。"""
    up = (mid or "").strip().upper()
    if up.startswith("M"):
        m = {x.id: x for x in store.open_memos()}.get(mid)
        if m is None:
            return {"ok": False,
                    "detail": f"没有 {mid}（可能已经关掉了）——"
                              f"列一遍用 memory_search（layer 填 memo）。"}
        who = {MEMO_USER_TASK: "用户说的事",
               MEMO_AIR_PROMISE: "你答应的事"}.get(m.kind or "", "未了结")
        return {"ok": True, "detail": f"- {m.id}（{who}）：{m.content}"}
    return {"ok": False,
            "detail": f"认不出 {mid} 是哪一类——编号形如 S1-0003 / S2-0001 / "
                      f"S3-0001 / M-0007。"}


def _profile_layer(store, limit: int = 5) -> dict:
    """画像层：**已立与待验证都列**（待验证标「只是猜测」——镜像页同一口径）。

    2026-09-24 检查修：原来 `current_profiles()` 没传 `status=None`，默认只要
    established——pending（"只是猜测"）的列不出来，与 docstring 不符，
    下面那个 `else "只是猜测"` 也就成了死分支（她问"你对我有什么猜测"时看不见）。
    **注入侧仍然只取 established**（R4 常驻）——这里是她**主动查**，口径与镜像页一致。
    """
    cur = store.current_profiles(status=None)
    if not cur:
        return {"ok": True, "detail": "【画像】还没有。"}
    lines = []
    for p in cur[:limit]:
        mark = "已立" if p.status == PROFILE_ESTABLISHED else "只是猜测"
        tops = "、".join(p.topics or [p.topic])
        lines.append(f"- {p.id} {p.statement}"
                     f"（{mark} · {p.evidence} 次印证"
                     + (f" · {tops}" if tops else "") + "）")
    more = (f"\n（还有 {len(cur) - limit} 条——看全部去镜像页）"
            if len(cur) > limit else "")
    return {"ok": True, "detail": "【画像】\n" + "\n".join(lines) + more}


def _summary_layer(store, limit: int = 5) -> dict:
    """摘要层：最近的几条 S2（一般不单独引用，但"都记了什么"要看得到）。"""
    s2s = store.hot_summaries(n=max(1, int(limit)))
    if not s2s:
        return {"ok": True, "detail": "【主题摘要】还没有。"}
    lines = [f"- {s.id}（{(s.created_at or '')[:10]} · "
             f"{'、'.join(s.topics or [s.topic])}）：{_clip(s.text, 60)}"
             for s in s2s]
    return {"ok": True, "detail": "【主题摘要】\n" + "\n".join(lines)}


def _memo_layer(store) -> dict:
    """备忘层：还挂着的事（编号 + 内容 + 谁欠的；**按组渲染**——
    她说「那件事」时要看得出"这是一件事的几步"）。"""
    opens = store.open_memos()
    if not opens:
        return {"ok": True, "detail": "【未了结的事】没有。"}
    kind_zh = {MEMO_USER_TASK: "用户说的事", MEMO_AIR_PROMISE: "你答应的事"}
    groups: dict[str, list] = {}
    order: list[str] = []
    for m in opens:
        g = (getattr(m, "group_name", "") or "").strip()
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(m)
    lines = []
    for g in order:
        items = groups[g]
        head = f"- 「{g}」（{len(items)} 步）：" if g else ""
        if head:
            lines.append(head)
        for m in items:
            who = kind_zh.get(m.kind or "", "未了结")
            raised = "" if m.status == MEMO_PENDING else "（提过一次）"
            bullet = "  - " if head else "- "
            lines.append(f"{bullet}{m.id}（{who}）{raised}：{m.content}")
    return {"ok": True,
            "detail": f"【未了结的事】还挂着 {len(opens)} 件：\n" + "\n".join(lines)}


def _recent_layers(store) -> dict:
    """什么都没给：**把四层最近的样子摊开**（"你都记了些什么"自然支持）。"""
    parts = [_profile_layer(store, 3)["detail"],
             _recent_scenes(store, limit=5)["detail"],
             _summary_layer(store, 3)["detail"],
             _memo_layer(store)["detail"]]
    return {"ok": True, "detail": "\n\n".join(p for p in parts if p)}


def _run_memory_search(store, args: dict, ctx: dict) -> dict:
    """查记忆（只读）。**一个介质一个工具**——2026-09-23 合并了
    `recall_memory` / `list_scenes` / `list_memos`（设计稿 §3.1）。

    参数（全可选）：`query`（语义）/ `entity`（实体名，最准）/ `layer`
    （s1/s2/s3/memo）/ `when`（具体日期）/ `id`（直查某个编号）/
    `topic`（按主题跨三层找，2026-09-24 晚——S2 / S3 不能语义搜，这条是它们的"搜"）。
    **内容条件（query / entity / when）只作用于场景**——与 s2/s3/memo 同给时
    明确引导，不静默丢掉条件（2026-09-24 晚修，见下）。
    **不传条件 = 列最近的**（四层各几条，设计稿 §3.1）。
    """
    mid = _pick(args, "id").strip()
    if mid:
        return _expand_by_id(store, mid, args, ctx)
    topic = _pick(args, "topic")
    if topic:
        # 可与 layer 组合：给了层就只列那一层（2026-09-24 检查修——
        # 原来直接忽略 layer，是"静默丢弃参数"，和 `query × layer` 是同一个毛病）
        return _by_topic(store, topic, _norm_layer(_pick(args, "layer") or ""))
    layer = _norm_layer(_pick(args, "layer") or "")
    cond = _pick(args, "query") or _pick(args, "entity") or _pick(args, "when")
    if cond and layer in ("memo", "s2", "s3"):
        # 内容条件只在 S1 上有意义（只有它有向量 / 实体索引）。以前这种组合是
        # **静默丢掉条件、直接列那一层**——她以为搜过，其实只是全列（2026-09-24 晚修）。
        if layer == "memo":
            return {"ok": False,
                    "detail": "备忘不能按内容搜——要看还挂着哪些，给 layer=memo"
                              "（不要带 query / entity / when）。"}
        who = "摘要" if layer == "s2" else "画像"
        return {"ok": False,
                "detail": f"{who}不能按内容搜（语义索引只在场景上）——用 `topic` "
                          f"按主题找（跨三层、包含匹配），或给编号直查（`id`）。"}
    if layer == "memo":
        return _memo_layer(store)
    if layer == "s3":
        return _profile_layer(store)
    if layer == "s2":
        return _summary_layer(store)
    if cond:
        return _scene_search(store, args, ctx)
    if layer == "s1":
        # 只看场景层、又不给条件：列场景（给了条件的那支在上面，走检索）
        return _recent_scenes(store)
    return _recent_layers(store)


def _by_topic(store, topic: str, layer: str = "", limit: int = 8) -> dict:
    """按主题找（2026-09-24 晚）：**主主题 + 附加主题都算**，子串匹配。

    这是「S2 / S3 搜不到」的正面回答：语义检索只在 S1 上（只有它有向量 /
    实体索引），但三层都挂在主题上——给个主题名就能一次列全。
    包含匹配（不是等值）：她记不全主题名时，「压力」也要能找到「用户·压力」。
    `layer` 给了就**只列那一层**（`s1` / `s2` / `s3`；备忘没有主题）。
    """
    q = (topic or "").strip()
    if layer == "memo":
        return {"ok": True,
                "detail": "备忘（M-xxxx）没有主题——它是「还欠着的事」，不参与主题归类。"
                          "要列备忘就别给 topic（用 layer=memo）。"}
    scenes = store.scenes_with_topic(q, limit=limit) if layer in ("", "s1") else []
    s2s = store.summaries_with_topic(q)[:limit] if layer in ("", "s2") else []
    profs = store.profiles_with_topic(q)[:limit] if layer in ("", "s3") else []
    if not (scenes or s2s or profs):
        return {"ok": True,
                "detail": f"没有挂在「{q}」上的东西（这是按主题的包含匹配）。"
                          f"也可以先按内容搜（query），再顺着展开看它挂在哪。"}
    scope = {"s1": "只列场景", "s2": "只列主题摘要", "s3": "只列画像"}.get(layer, "三层")
    lines = [f"主题「{q}」（包含匹配 · {scope}）："]
    if profs:
        lines.append("【画像】")
        for p in profs:
            mark = "已立" if p.status == PROFILE_ESTABLISHED else "只是猜测"
            lines.append(f"  - {p.id} {p.statement}（{mark} · {p.evidence} 次印证）")
    if s2s:
        lines.append("【主题摘要】")
        for s2 in s2s:
            lines.append(f"  - {s2.id}（{(s2.created_at or '')[:10]}）："
                         f"{_clip(s2.text, 60)}")
    if scenes:
        lines.append("【场景】")
        for s in scenes:
            lines.append("  " + scene_line(s))
    return {"ok": True, "detail": "\n".join(lines)}


def _scene_search(store, args: dict, ctx: dict) -> dict:
    """场景检索（原 `recall_memory`）：**能给的维度越多越准**——
    和唤醒注入是同一套线索思路。

    三个维度：`query`（语义）/ `entity`（实体名，**最准**）/ `when`（日期）。
    它们的关系是**交集**，多给一个就窄一层——这才是"越准确"的意思，
    不是"给得越多越可能命中"。

    `when` 只收**具体日期**（`2026-09-13` / `2026-09`），不收"上个月"：
    相对时间的换算落到模型身上不可靠（同 memo 那条纪律：不让模型估时间）。

    能查到的和查不到的，设计稿 3.3 有诚实清单（同义不同词会漏，
    情绪只有正负粗细）——这里不掩饰：查不到就说查不到。
    """
    q = _pick(args, "query")
    ent = _pick(args, "entity")
    when = _pick(args, "when")
    if not (q or ent or when):
        return {"ok": False,
                "detail": "memory_search 要查就给一个维度：query（语义）/ "
                          "entity（实体名）/ when（日期）——都不给就是列最近的。"}

    # ① 实体先上：它最准（精确匹配，不走语义）
    if ent:
        cands = {s.id: s for s in store.scenes_by_entities([ent], limit=50)}
    else:
        cands = {s.id: s for s in store.query_scenes(limit=300)}

    # ② 时间收窄（前缀匹配：给到月就筛月，给到日就筛那天）
    day = ""
    if when:
        day = _norm_date(when)
        if not day:
            # **明确拒绝**而不是当成"没给"——静默放过会让她以为日期生效了
            return {"ok": False,
                    "detail": f"when 要写具体日期（2026-09-13 或 2026-09）；"
                              f"「{when}」这种说法算不出日期。"}
        cands = {i: s for i, s in cands.items()
                 if (s.time_record or "").startswith(day)}

    dims = " + ".join([x for x in (f"语义「{q}」" if q else "",
                                   f"实体「{ent}」" if ent else "",
                                   day) if x])
    if not cands:
        return {"ok": True, "detail": f"没有符合的（{dims}）。"}

    # ③ 排序（§3.1；2026-09-24 三键，2026-10-05 设计稿改四键）：**共同命中的维度数
    #    → 线索层级 → 核心度**（末位再看语义接近度）。维度数 = 这条查询给了几个标签
    #    （单标签时恒为 1，自动退化成"按质量排"）；层级 = 语义 / 实体直取是强线索（0）、
    #    只按日期翻出来的是弱线索（1）；核心度 = 层 + 强度 + 印证饱和
    #    （`recall.core_score`——**与注入侧同一把尺**；新鲜度 2026-10-05 已从核心度
    #    摘出去、注入侧改在排序里单列——**工具侧这一处还没跟**，见核对 L9 留账）。
    #    ⚠️ 语义分在这里**只是门**（0 分不要），不是排序键——"先给重要的"才是
    #    排序的意思；四道闸（门 → 排序 → 限量 → 浓缩）见规格 §3.1。
    emb = (ctx or {}).get("emb")
    sim: dict[str, float] = {}
    if q:
        qv = emb.embed_one(q) if emb is not None else None
        if qv:
            vecs = dict(store.all_embeddings())
            sim = {i: cosine(qv, vecs[i]) for i in cands if i in vecs}
        if not sim:
            from .recall import char_overlap
            sim = {i: char_overlap(q, f"{s.title} {s.text}")
                   for i, s in cands.items()}
        # **0 分的一个都不能要**：兜底检索会给每条都算一个分数（哪怕一个字都不重叠），
        # 不过滤的话「量子力学」也能"查到"那条面试的记录——那比查不到更糟。
        # （全被筛光时下面自然走"没找到相关的记忆"那一路）
        sim = {i: v for i, v in sim.items() if v > 0}
        cands = {i: s for i, s in cands.items() if i in sim}

    from .recall import core_score          # 与注入侧共用一把尺
    dim_n = (1 if q else 0) + (1 if ent else 0) + (1 if day else 0)
    strong = 0 if (q or ent) else 1         # 只按日期翻出来的是弱线索
    ids = sorted(cands,
                 key=lambda i: (-dim_n, strong, -core_score(cands[i]),
                                -sim.get(i, 0.0)))[:3]
    # ---- 停止条件（2026-09-21 设计稿 E 条）：全看过 = 没有新增 → 停手 ----
    # 客观判据（不靠她自评）：这一批结果她（这一轮或之前）已经看过了，
    # 再查下去只会捞回同样的东西。把剩余名额一并顶掉，别原地转圈。
    # `seen_scenes` 挂在**会话**上（chat 传进来）→ 跨轮：上轮看过的也算看过。
    seen = ctx.setdefault("seen_scenes", set())
    if ids and not [i for i in ids if i not in seen]:
        # "全看过 = 没有新增"（判据闸）：这批捞回来和上次一样——劝停并给出路。
        # 2026-09-23 起不再"顶掉名额"（名额制已撤）：反复捞旧结果时，
        # 上面这句引导 + 提示词的"够了就停"就是收敛手段。
        return {"ok": True,
                "detail": f"这些你都看过了（{'、'.join(ids)}）——没有新的了。"
                          f"用手上这些回答；要换角度，换个词再查。"}
    seen.update(ids)

    lines = []
    for sid in ids:
        s = cands.get(sid) or store.get_scene(sid)
        if s is None:
            continue
        # 一行怎么拼统一在 `prompts.scene_line`（编号 / 相对日 / 悬置标记）——
        # 与注入侧**共用一份**，格式不再各写各的（2026-09-21，设计稿 A 条）。
        lines.append(scene_line(s))
    if not lines:
        return {"ok": True, "detail": f"没找到相关的记忆（{dims}）。"
                                      f"可能本来就没记过，或者当时用的词不一样。"}
    return {"ok": True, "detail": f"查到了（按 {dims}）：\n" + "\n".join(lines)}


def _recent_scenes(store, day: str = "", limit: int = 20) -> dict:
    """列最近的场景（编号 + 相对时间 + 标题）——**看得见自己都存了什么**。

    她的原话（2026-09-13）：「recall 是查——得先知道问什么才查得到。
    现在删的钥匙给我了，可钥匙上没编号，我还是盲删；而这个动作不可撤回。」
    所以 `memory_search` **不给条件 = 先列最近的**（2026-09-23 并入；编号从
    注入起就带，这里给的是"全量 + 冷层标记"那一层）。
    """
    scenes = store.query_scenes(limit=500)
    if day:
        scenes = [s for s in scenes if (s.time_record or "").startswith(day)]
    scenes.sort(key=lambda s: s.time_record or "", reverse=True)
    if not scenes:
        return {"ok": True,
                "detail": f"【最近的场景】没有{'（' + day + '）' if day else ''}。"}

    lines = []
    for s in scenes[:limit]:
        flags = [x for x in ("冷层" if s.archived else "",
                             "敏感" if s.sensitive else "") if x]
        tail = f" [{'/'.join(flags)}]" if flags else ""
        lines.append(f"- {s.id}{rel_stamp((s.time_record or '')[:16])}："
                     f"{s.title or ''}{tail}")
    more = (f"\n（只列最近 {limit} 条，共 {len(scenes)} 条——要更早的给个日期）"
            if len(scenes) > limit else "")
    return {"ok": True,
            "detail": f"【最近的场景】共 {len(scenes)} 条：\n" + "\n".join(lines) + more}


def _run_close_memo(store, args: dict, ctx: dict) -> dict:
    """关掉一件未了结的事（自主档）：用户给了结果，或这事确实完了。

    ⚠️ **判不准就别关**——关掉之后她不再主动提它（`memo.close` 的语义）。
    关错了的代价比多挂一会儿大（同命中判定那条纪律：误关比多提更贵）。
    """
    mid = _pick(args, "id")
    if not mid:
        return {"ok": False,
                "detail": "close_memo 要 id（形如 M-0007；"
                          "先用 memory_search 拿编号）"}
    opens = {m.id: m for m in store.open_memos()}
    m = opens.get(mid)
    if m is None:
        return {"ok": False,
                "detail": f"找不到 {mid}（可能已经关掉了）——"
                          f"先 memory_search 看一眼（layer 填 memo）现在还挂着哪些。"}
    from .memo import close as _close      # 延迟 import（同 shortterm 的先例，防环）
    _close(store, mid, by="她调工具")
    return {"ok": True,
            "detail": f"关掉了 {mid}：「{m.content}」。"
                      f"用户再提就当普通话题聊——别再当成挂着的事。"}


# （原 `reject_profile`——"他明说这条不对就删掉"——2026-09-24 并入 `forget_memory`：
#   三层同一套处置（删 / 归档 / 留着都走同一张确认条，他点），不再单开一个自主档工具。
#   界面那条「不对」按钮仍在（走 `/api/reject` → `weave.user_reject_profile`）。
#   历史见 git。）


def _read_raw(store, sid: str) -> dict:
    """读一段原话（内部：`memory_search` 的 `raw=true` 分支）。

    走 `store.get_raws_by_scene`——项目里要原文的**唯一入口**（原文没有索引，
    只能经场景编号下钻）。返回按预算截断：整段灌进来不如只带该带的那截
    （同 `web_search` 的 `_clip` 思路）。
    """
    if not sid:
        return {"ok": False, "detail": "要给 S1 编号才能下钻原话"}
    up = sid.strip().upper()
    if not up.startswith("S1"):
        return {"ok": False,
                "detail": f"{sid} 不是场景编号——原文只能按 S1 编号下钻。"}
    s = store.get_scene(sid)
    if s is None:
        return {"ok": False,
                "detail": f"找不到 {sid}。编号从这一轮想起的记忆、或 memory_search 来。"}
    raws = store.get_raws_by_scene(sid, on_date=(s.time_record or "")[:10])
    blocks = [(r.content or "").strip() for r in raws if (r.content or "").strip()]
    if not blocks:
        return {"ok": True,
                "detail": f"{sid} 没留下原文（更早的记忆可能只有摘要）——只能用摘要说。"}
    text = "\n\n".join(blocks)
    limit = 2000
    clipped = (text if len(text) <= limit
               else text[:limit].rstrip() + "…（还长，先给这些）")
    return {"ok": True,
            "detail": f"{sid}「{s.title or ''}」的原话：\n\n{clipped}"}


# （原 `save_now`——"立刻把这段提取一次"——2026-09-24 删除：它只改**时机**不改结果
#  （窗口 / 话题切换 / 收尾都会落库），而"写下重要的东西"的正路是**说出来**：
#   她的话就是原文，会被提取链路判定。见工具箱稿 §3.2 / §九。）


def _web_search(q: str, ctx: dict) -> dict:
    """联网查（内部：`web` 的 `query` 分支）——走服务端搜索通道
    （哪一条由 `search.channel` 定，见 `core/search.py` 的通道表）。

    纪律（设计稿 §3.6）：**结果不进记忆**——它是这一轮说话的燃料，不是
    「关于他的记忆」（记忆的来源必须是"他说的"，不能是"网上说的"），
    所以这里只返回结果、不碰 store。
    """
    llm = (ctx or {}).get("llm")
    if llm is None:
        return {"ok": False, "detail": "这一轮没有可用的搜索通道。"}
    res = llm.search(q)
    if not res.get("ok"):
        # 「没查到」和「世界上没有」是两件事——模型很容易把工具失败说成事实不存在，
        # 所以失败文案里要把它挑明。
        return {"ok": False,
                "detail": f"没查到（{res.get('error') or '搜索没成'}）。"
                          f"跟用户说没查到就行，别当成答案。"}
    facts = (res.get("text") or "").strip()
    urls = res.get("urls") or []
    # 结构化结果里有标题——给她标题比给裸 URL 有用（她能说出"我看到有条报道说…"）；
    # URL 照样进 trace。这是官方通道多出来的那半边，别浪费。
    names = [s.get("title") or s.get("url", "")
             for s in (res.get("sources") or [])[:3]]
    tail = "\n（来源：" + "、".join(x for x in names if x) + "）" if names else ""
    return {"ok": True,
            "detail": (f"查到了：\n{_clip(facts, 1200)}{tail}" if facts
                       else "查了，但没搜到有用的内容。"),
            "searched": {"queries": res.get("queries") or [], "urls": urls}}


def _web_fetch(url: str, ctx: dict) -> dict:
    """取回一个网页（内部：`web` 的 `url` 分支）。

    实现全在 `webfetch.py`（URL 校验 / 只许公共地址 / 连接固定 / 同源重定向 /
    上限）；这里只做两件事：把参数递过去、把结果翻成她读得懂的话。

    抓取是**本机出网**——不挂 VPN / 不配代理时她只会拿到"没抓到"。
    这和她"搜不搜得到"是两回事：搜索在 DeepSeek 服务端执行，不受本机网络影响。
    """
    out = webfetch.fetch(url, opener=(ctx or {}).get("web_opener"))
    if not out.get("ok"):
        # 「没抓到」和「页面上没有」是两件事——别让她把工具失败说成事实不存在
        return {"ok": False,
                "detail": f"没抓到（{out.get('error') or '未知原因'}）。"
                          f"跟用户说没抓到就行，别当成页面里没有。"}
    head = f"{out.get('final_url') or url}（HTTP {out.get('status')}）"
    body = out.get("text") or "（这页没有可读的文本）"
    return {"ok": True, "detail": f"{head}：\n\n{body}"}


def _run_web(store, args: dict, ctx: dict) -> dict:
    """往外看（只读）：给 `query` 就查，给 `url` 就读页面（2026-09-24 合并）。

    两个能力合成一个工具（工具箱稿 §3.6）："搜到链接再读页" = 同一个工具调两次。
    该不该用是分寸（`prompts.TOOLS_LINES["web"]`），档位只管"能不能"。
    """
    url = _pick(args, "url")
    if url:
        return _web_fetch(url, ctx)
    q = _pick(args, "query")
    if q:
        return _web_search(q, ctx)
    return {"ok": False, "detail": "web 要 query（要查什么）或 url（读哪个页面）"}


def _memory_title(store, sid: str) -> str | None:
    """一条记忆的"一眼标题"（**只管记忆三层** S1 / S2 / S3）；不在三层里返回 None。

    S1 → 卡片标题；S2 → 叙述前 24 字；S3 → 陈述前 24 字。
    **M 不在这张表里**：备忘不是"改 / 删"的对象（它是"了结"，走 `close_memo`）——
    对它返回 None，好让调用方落到 `_wrong_kind_hint` 那句引导上。
    （2026-09-24 检查修：原来这里给 M 也取标题，于是**真实挂着**的 M 编号会被
    `forget_memory` / `revise_memory` 当成可提议对象——确认条弹出来，点"彻底删除"
    却报"认不出编号"（`delete_by_layer` 不认 M）。当时测试里那条 M-0001 不存在，
    所以给了假通过。）
    """
    up = (sid or "").strip().upper()
    if up.startswith("S1"):
        s = store.get_scene(sid)
        return (s.title or "") if s else None
    if up.startswith("S2"):
        s2 = store.get_summary(sid)
        return (s2.text or "")[:24] if s2 else None
    if up.startswith("S3"):
        p = store.get_profile(sid)
        return (p.statement or "")[:24] if p else None
    return None


def _wrong_kind_hint(sid: str) -> str:
    """编号不属于「记忆三层」（S1 / S2 / S3）时，说清它该走哪条路。

    2026-09-24 三层合并之后，改与删对 S1/S2/S3 是**同一套**——只剩 M 不走这条路
    （它不是删，是"了结"）。这层兜底因此瘦成一段；但它仍要存在：
    只回"找不到"，她会以为是自己编号记错（实测原话："删除接口找不到 S3 记忆"）。
    """
    up = (sid or "").strip().upper()
    if up.startswith("M"):
        return (f"{sid} 是**备忘录**（还没了结的事）——不走这个动作："
                f"要了结它用 `close_memo`。")
    return ""


def _run_revise_memory(store, args: dict, ctx: dict) -> dict:
    """改一条记忆（**三层同一套**：S1 场景 / S2 摘要 / S3 画像）。
    **只能提议**——她说了不算，等他点头。

    `field` **三层都认**（2026-09-24：三层都能改标签，不只场景）——每层的
    可改字段与别名表在 `store.LAYER_EDIT`：场景最多（标题 / 主题 / 事件时间 /
    情境 / 反应 / 情境类），摘要 / 画像 = 主文本 + 主题。
    不给 field 就改该层的**主文本**（场景 / 摘要 = 摘要与叙述，画像 = 陈述）。

    **主题字段（2026-09-24 晚）**：S2 / S3 给 **1-3 个**（逗号 / 顿号隔开，
    第一个 = 主主题；解析在 `store.split_topics`）——多出来的 0-2 个是附加主题，
    只服务"找得到"；S1 的场景主题仍是**单值**（它是聚合分组的键，多值会打乱链路）。

    这里**一个字都不改**：只把提议登记进 `ctx["proposals"]`，
    由对话层转成「待确认」，等他下一句话。
    真执行走 `weave.update_by_layer` → `update_*_confirmed`
    ——**和界面上那个按钮是同一个动作**，按钮和工具只是发起方不同。
    """
    sid = _pick(args, "id")
    text = _pick(args, "text")
    if not sid or not text:
        return {"ok": False,
                "detail": "revise_memory 要两个参数：id（形如 S1-0003 / S2-0001 / "
                          "S3-0001）和 text（改成什么）"}
    # 编号纪律在**字段校验之前**（2026-09-24 夜核对修）：M 带一个不认的 field
    # 也不该先撞上场景的字段表——当场拒绝并引导 `close_memo`。
    hint = _wrong_kind_hint(sid)
    if hint:
        return {"ok": False, "detail": hint}
    rule = layer_edit(sid)
    field_raw = _pick(args, "field")
    field = (rule["aliases"].get(field_raw.strip(), "")
             if field_raw.strip() else rule["main"])
    if not field or field not in rule["editable"]:
        return {"ok": False,
                "detail": f"不认的字段「{field_raw}」——{sid} 能改："
                          f"{'、'.join(rule['labels'].values())}"}
    if field == "topic":
        # 主题个数的校验（2026-09-24 晚）：S1 单值（分组键）；S2 / S3 是 1-3 个。
        # 在**提议之前**说清——多给的不静默丢弃（`split_topics` 会截断到 3）。
        parts = _split_ids(text)
        if not parts:
            return {"ok": False, "detail": "主题不能为空——给它 1-3 个（逗号隔开）"}
        if sid.upper().startswith("S1") and len(parts) > 1:
            return {"ok": False,
                    "detail": "场景（S1）的主题是**单值**——它决定聚合分组，"
                              "多主题只在摘要 / 画像上（S2 / S3）能挂"}
        if len(parts) > TOPICS_MAX:
            return {"ok": False,
                    "detail": f"主题最多 {TOPICS_MAX} 个（1 个主主题 + 最多 2 个附加）"
                              f"——你给了 {len(parts)} 个"}
    title = _memory_title(store, sid)
    if title is None:
        return {"ok": False,
                "detail": _wrong_kind_hint(sid)
                or f"找不到 {sid}。确认一下编号——它在这一轮想起的记忆里才有。"}
    ctx.setdefault("proposals", []).append(
        {"tool": "revise_memory", "ids": [sid], "scene_id": sid,
         "field": field, "text": text, "title": title})
    label = rule["labels"].get(field, "文本")
    return {"ok": True,
            "detail": f"已经问出口了：{sid}「{title}」的{label}要不要改成「{text}」。"
                      f"等用户点头——用户说了才算，你只能问。"}


def _run_forget_memory(store, args: dict, ctx: dict) -> dict:
    """提议处置一条或多条记忆（编号用逗号/顿号/空格隔开）。**只能提议**。

    **三层同一套**（2026-09-24，工具箱稿 §3.4）：S1 场景 / S2 摘要 / S3 画像
    走同一张确认条（彻底删除 / 归档冷存 / 不删除）——删不可撤回、归档可逆、
    留着什么都不变；**怎么处置整个由人点**，所以这里一个字都不碰。

    她也不再判"他是不是明说"：判不准、还会纠结很久——列出来就是尽到本分了。
    M 不走这条（它不是删，是"了结"：`close_memo`）。
    """
    ids = _split_ids(_pick(args, "id"))
    if not ids:
        return {"ok": False, "detail": "forget_memory 要 id（形如 S1-0003 / S2-0001 / "
                                       "S3-0001；多条就写 S1-0001,S1-0003）"}
    titles, missing = [], []
    for sid in ids:
        title = _memory_title(store, sid)
        if title is None:
            missing.append(sid)
        else:
            titles.append(title or sid)
    if missing:
        # 只会剩下两种：M（该走 `close_memo`）和真不存在的编号——
        # 只说"找不到"，她会以为是自己编号记错（实测原话："删除接口找不到 S3 记忆"）
        hints = [(_wrong_kind_hint(x), x) for x in missing]
        wrong = [h for h, _ in hints if h]
        ghost = [x for h, x in hints if not h]
        detail = "　".join(wrong)
        if ghost:
            tail = (f"另外 {'、'.join(ghost)} 也没找到——确认一下编号"
                    f"（它在这一轮想起的记忆里才有）。")
            detail = (detail + "　" + tail) if detail else (
                f"找不到 {'、'.join(ghost)}。确认一下编号——"
                f"它在这一轮想起的记忆里才有。")
        return {"ok": False, "detail": detail}
    ctx.setdefault("proposals", []).append(
        {"tool": "forget_memory", "ids": ids,
         "scene_ids": ids,          # 旧字段名留着兼容（界面 / 旧提议）
         "scene_id": ids[0], "title": "、".join(titles)})
    how = f"{len(ids)} 条（{'、'.join(ids)}）" if len(ids) > 1 else f"{ids[0]}「{titles[0]}」"
    return {"ok": True,
            "detail": f"已经问出口了：{how} 怎么处置（彻底删 / 归档冷存 / 留着）。"
                      f"用户点一下才算——你只管列出来，不用掂量用户是不是那个意思。"}


# （原 `memory_trace` / `memory_read`——2026-09-24 合并进 `memory_search`：
#   编号直查 = 展开一条（`_expand_scene` / `_expand_profile` / `_expand_summary`），
#   `raw=true` 再下钻原话（`_read_raw`）。历史见 git。）


# ---------------------------------------------------------------------
# 工具表
# ---------------------------------------------------------------------

TOOLS: dict[str, dict] = {
    # 描述只写「是什么 + 返回什么」（DeepSeek Harness 的分法：它的工具描述
    # 就一句话，如 web_search 的 "Search the web... Returns ..."）。
    # 「该不该用」的分寸不在这里——挪进系统提示词的独立一段
    # （prompts.TOOLS_LINES / render_tools_block），那边是规矩该待的地方，
    # 也省下每次请求里工具定义占的上下文。
    "memory_search": {
        "mode": READ,
        "description": "查记忆，或展开一条（给编号：依据链 / 素材 / 原话）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "想查的事（语义）"},
                "entity": {"type": "string",
                           "description": "实体名（人名/地点），精确匹配最准"},
                "layer": {"type": "string",
                          "description": "只看哪一层：s1 场景 / s2 摘要 / s3 画像 / memo 备忘"},
                "when": {"type": "string",
                         "description": "具体日期（2026-09-13 或 2026-09）"},
                "id": {"type": "string",
                       "description": "展开某个编号，如 S1-0003 / S2-0001 / S3-0001 / M-0007"},
                "topic": {"type": "string",
                          "description": "按主题跨三层找（主主题 + 附加主题，包含匹配）"
                                         "——摘要 / 画像搜不到时用这条"},
                "raw": {"type": "boolean",
                        "description": "给 S1 编号时：再把当时的原话（S0）带出来"},
            },
            "required": [],
        },
        "run": _run_memory_search,
    },
    "web": {
        "mode": READ,
        "description": "联网查一件事（给 query），或取回一个网页（给 url）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要查什么，写具体"},
                "url": {"type": "string",
                        "description": "要取的页面地址（http/https）"},
            },
            "required": [],
        },
        "run": _run_web,
    },
    "revise_memory": {
        "mode": ASK,
        "description": "提议改一条记忆的文本或标签（主题 1-3 个）。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string",
                       "description": "编号，如 S1-0003 / S2-0001 / S3-0001"},
                "text": {"type": "string", "description": "改成什么"},
                "field": {"type": "string",
                          "description": "改哪个字段（默认这条的文本；摘要 / 画像能改「主题」"
                                         "——1-3 个用逗号隔开，第一个是主主题；"
                                         "场景还能改标题 / 事件时间 / 情境 / 反应 / 情境类）"},
            },
            "required": ["id", "text"],
        },
        "run": _run_revise_memory,
    },
    "forget_memory": {
        "mode": ASK,
        "description": "提议处置一条或多条记忆——删 / 归档 / 留着由人点（编号来自 memory_search）。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string",
                       "description": "编号（S1-xxxx / S2-xxxx / S3-xxxx），多条用逗号隔开"},
            },
            "required": ["id"],
        },
        "run": _run_forget_memory,
    },
    # ---- 2026-09-21 新增（待优化设计稿 D 条 3 / D2a）----
    "close_memo": {
        "mode": AUTO,
        "description": "关掉一件未了结的事（编号来自 memory_search）。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "编号，如 M-0007"},
            },
            "required": ["id"],
        },
        "run": _run_close_memo,
    },
}

# 每轮最多调几次——**三条线分开计**（2026-09-21 加原文下钻时改）。
#
# 以前是「所有只读共享一个名额，防止她围着库转圈」。加抓取 / 下钻这类
# **两步流**之后，那条规则会把正常流程掐死（harness 的引导正是这个流程：
# 「搜一次 → 抓一页原文」「查一次 → 读一段原文」）。
# 分开计的理由：库内转圈是病，而"查到之后读原文"是活。
# ⚠️ 上限是**保险**，不是主要手段（2026-09-21 设计稿 E 条）：
#   停的判据是"本轮查到的信息够回答问题了"——落到代码是一进一退两道：
#   进（`_guard`：同一个词不许再查）+ 退（`_scene_search`：全看过 = 没有新增 → 停）。
#   **这两道是判据闸，不是配额**：拦的都是"做了也白做"（同一个词结果一样、
#   全看过没有新增），拦下还附引导。
#
# 2026-09-23（设计稿 §四"判据管病，不掐活"落地）：**库内检索 / 下钻读原文
# 不再硬拒**——"查了几次"不是"该不该停"的判据（实测把"查一次 → 读一段原文"
# 的正常两步流掐死过）；够不够回答**由她判**（提示词里的"够了就停"是主判据），
# 次数降级为**信号**：到阈值起在工具返回里附一句提醒（`_limit_note`）。
_WEB_LIMIT_PER_TURN = 2       # 网络：**仍硬限**（真钱 + 慢 + 外部世界——成本账，不是判据账）
_WEB_TOOLS = ("web",)          # 2026-09-24 合并：搜 / 抓取是同一个工具
# 提醒阈值（只提醒、不拦）
_READ_NOTE_FROM = 3           # 库内检索（memory_search）
_DRILL_NOTE_FROM = 2          # 下钻读原话（更贵，早一点提醒）


def openai_tools(names: list[str] | None = None) -> list[dict]:
    """转成 OpenAI 的 `tools` 参数格式。

    `names` 限定这一轮给哪些工具——工具多了会吃上下文，也更容易选错
    （设计稿防错第 1 条：工具要少）。
    """
    out = []
    for name, t in TOOLS.items():
        if names is not None and name not in names:
            continue
        out.append({"type": "function",
                    "function": {"name": name,
                                 "description": t["description"],
                                 "parameters": t["parameters"]}})
    return out


def tool_names() -> list[str]:
    """所有动作的名字（`chat._tool_names` 会按这一轮的情况再筛一遍）。"""
    return list(TOOLS.keys())


def execute(store, name: str, args: dict, ctx: dict | None = None) -> dict:
    """执行一次工具调用。**永远不抛异常**（失败返回 `{"ok": False, detail}`）。

    流水线（设计稿第八节）：
        档位判定 → 红线终审 → 执行 → 留痕
    """
    ctx = ctx or {}
    args = args if isinstance(args, dict) else {}

    # ---- 红线终审（单调：只有拒绝，没有允许）----
    denial = _guard(store, name, args, ctx)
    if denial is not None:
        _trace(name, args, False, denial)
        return {"ok": False, "detail": denial, "mode": "denied"}

    tool = TOOLS[name]
    try:
        out = tool["run"](store, args, ctx)
    except Exception as e:                     # 工具自己炸了也不能中断对话
        out = {"ok": False, "detail": f"这个动作没做成（{type(e).__name__}: {e}）"}
    if not isinstance(out, dict):
        out = {"ok": bool(out), "detail": str(out)}
    out.setdefault("mode", tool["mode"])
    # 次数提醒（2026-09-23）：只提醒、不拦——"够不够回答"由她判
    note = _limit_note(name, ctx)
    if note and out.get("detail"):
        out["detail"] = out["detail"] + "\n" + note
    # 留痕带的是**提醒后**的全量 detail：以后回看"她那次为什么停 / 为什么没停"，
    # trace 里看得到系统当时说了什么
    _trace(name, args, bool(out.get("ok")), out.get("detail", ""),
           extra=out.get("searched"))
    return out


def _guard(store, name: str, args: dict, ctx: dict) -> str | None:
    """红线终审。**返回字符串 = 拒绝**，返回 None = 放行。

    它只有「拒绝」没有「允许」，所以任何后续逻辑都无法把拒绝翻成放行——
    这正是「单调」的意思（设计稿第九节）。

    这里挡三件事：
      1. 不存在的动作（并把可用的列出来，她下一轮能改对）；
      2. 一轮里反复查（网络**硬限 2**；库内 / 下钻只在返回里附提醒，不拦——2026-09-23）；
      3. **需确认档没有确认通道**（失败即拒绝，不能悄悄做了）。
    """
    tool = TOOLS.get(name)
    if tool is None:
        return (f"没有 {name} 这个动作。可用的："
                + " / ".join(f"{n}（{TOOLS[n]['description']}）" for n in TOOLS))
    if tool["mode"] == READ:
        if name in _WEB_TOOLS:
            used = ctx.get("web_used") or 0
            if used >= _WEB_LIMIT_PER_TURN:
                return "这一轮外面已经查过两回了——先把手上这些说完，下一轮再查。"
            ctx["web_used"] = used + 1
        else:
            # ① 判据闸（**硬拒**，2026-09-21 设计稿 E 条）：同一个词再查，
            #    结果一样——那是原地打转，不是检索。**拒绝且不计数**。
            if name == "memory_search":
                q = " ".join((_pick(args, "query") or "").split())
                used_q = ctx.setdefault("recall_queries", [])
                if q and q in used_q:
                    return (f"「{q}」这一轮查过了——换个词再查（人名 / 日期 / "
                            f"别的说法），同一个词查出来还是同一批。")
                if q:
                    used_q.append(q)
            # ② 次数**只计数、不拒绝**（2026-09-23 定）：够不够回答由她判
            #    （"够了就停"在提示词里）；到阈值由 `_limit_note` 附一句提醒。
            # 下钻（`memory_search` 带 raw 真读了原话）在**执行处**自己记
            # `drill_used`（2026-09-24 合并后：按实际发生计数，比按工具名准）。
            ctx["read_used"] = (ctx.get("read_used") or 0) + 1
    if tool["mode"] == ASK and not ctx.get("confirm_channel"):
        # 失败即拒绝：没有能点头的人，就不能做
        return (f"「{name}」要先问用户一句才行，而这一轮没有确认的口子——"
                f"先用你的话说出来问用户。")
    return None


def _limit_note(name: str, ctx: dict) -> str:
    """次数提醒（2026-09-23 定）：**判据是"信息够不够"，不是次数**。

    次数只当信号——到阈值在返回里附一句，让**她自己判**继不继续
    （提示词里的"够了就停"才是主判据）。"做了也白做"的两件事**分两道**（2026-10-06
    复核更正：原写"仍由 `_guard` 硬拦"把两道写成一道）：同一个词再查由 `_guard`
    **硬拒**；这批结果全看过由 `_scene_search` **劝停**（照常返回"没有新的了"，
    `ok=True` 不顶名额）；网络线仍是硬限（成本账）。
    """
    # ⚠️ 只有**只读检索类**才有次数提醒（2026-09-23 检查修）：原来漏了这道检查——
    # 她查过几次之后，连 `close_memo` 这类写动作的返回都被附上"你已经查了 3 次"
    if name in _WEB_TOOLS:
        return ""                 # 网络线在 `_guard` 里硬限，不需要提醒
    if (TOOLS.get(name) or {}).get("mode") != READ:
        return ""
    n = ctx.get("drill_used") or 0
    if n >= _DRILL_NOTE_FROM:
        return (f"（这一轮你已经读过 {n} 段原话——这些够不够回答用户，你判；"
                f"不够就继续，够了就停下来把话说完。）")
    n = ctx.get("read_used") or 0
    if n >= _READ_NOTE_FROM:
        return (f"（这一轮你已经查了 {n} 次——这些够不够回答用户，你判；"
                f"不够就再查，够了就停下来把话说完。）")
    return ""


def _trace(name: str, args: dict, ok: bool, detail: str,
           extra: dict | None = None) -> None:
    """每次调用落一行（同唤醒 / 否决 / 档案留痕的理由）。

    `extra` 放**动作自己才交代得出来的东西**——比如搜索实际用了哪些词、
    打开了哪些页面。它们不在 `args` 里（`args` 是她给的），
    也不该被 `detail` 的截断吃掉，所以单独一个字段。
    """
    try:
        trace_dir = cfgmod.abspath(cfgmod.PATHS["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"工具-{datetime.now().strftime('%Y%m%d')}.jsonl"
        record = {"ts": now_str(), "tool": name, "args": args,
                  "ok": bool(ok), "detail": _clip(detail, 200)}
        if extra:
            record["extra"] = extra
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[tools] 工具留痕失败（不影响结果）: {e}")
