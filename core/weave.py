"""编织层：把散落的线索连起来呈现给他。

（原为 R6——"对话姿态"的一侧。2026-09-20 两套姿态退役后，R6 与引导提示已删、
`/api/mode` 路由已删——这个文件只保留**功能本体**：呈现与可否决（见下）。
"要不要拆、拆多深"现在是人格文件里的对话分寸，不再是这里的机制。历史见 git。）

这个文件装两件事，它们是同一件事的两面：
  - **呈现**（`render_mirror`）：画像 + 支撑场景（可追溯）+ 状态标记。
    `pending` 必须标「还没印证够，只是猜测」——不加区分地呈现，
    等于把猜测说成事实。敏感内容不进镜像。
  - **可否决**（`user_reject_profile`）：**否是织的前提**。
    一条不能说「不对」的画像，就是把 air 的判断强加给用户——
    那既不武断也不迎合，它是第三种更糟的东西（替人定义自己）。

外加一个只有 AI 才做得到的（`pattern_stats`）：跨时间统计「遇到 X 情境 →
典型反应」，人只活在当前这一刻，做不到这件事。

**第一版不做**：不主动做情绪拆解引导、不主动给整合建议、
不判断「你现在需要被分析」——那些不是「呈现」，是「诱导」。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L9 编织层（呈现与可否决；原 R6）
#   上游    ：config、embedding（主题归并要凑向量中心）、model、store
#   下游    ：chat（工具提议后的真执行）、dashboard（镜像页 / 主题页 / 改删按钮）
#   对外入口：`render_mirror` / `pattern_stats` / `merge_suggestions` / `user_reject_profile`
#             （画像的「删」走它——三层里只有它没挂 `_confirmed` 名字）
#             + 一组 `*_confirmed`（档案 / 语言 / 三层改·归档·取消归档 / 删〔场景 / 摘要〕 /
#               合并主题 / 合并实体）
#             + 三层分派 `*_by_layer`（确认条与统一口 `/api/memory-action` 共用一份）
#   边界    ：**只做呈现与人确认后的动作**——它不替人自动开口、也不自动合并主题；
#             带 `_confirmed` 的都是「界面那个按钮背后的同一个动作」
# ---------------------------------------------------------------------
from __future__ import annotations

from . import config as cfgmod
from .embedding import cosine
from .model import (INVALIDATED_ARCHIVE, INVALIDATED_USER,
                    LANGS, PROFILE_ESTABLISHED, PROFILE_PENDING,
                    lang_from_prefs)
from .store import (SCENE_FIELD_LABELS, SUMMARY_FIELD_LABELS, append_trace,
                    layer_edit, now_str, scene_ids, split_topics)


def user_reject_profile(store, pid: str) -> str:
    """用户否决一条画像：**真删**（2026-09-24 定），返回处理结果。

    返回值：`deleted` / `not_found`。

    为什么从「作废留着」改成「真删」：画像是**推断的结论**，素材全在
    （S0 原文 / S1 场景 / S2 摘要）——判断真成立的话，新证据攒够它会被
    重新立出来。所以删它的代价天然小于删场景（素材不可再生）；
    而"作废但留着"只是给系统存一份**没有消费点**的档案：
    `rejected_statements` 零调用、`否决-*.jsonl` 只写不读、
    那条「已作废」的 converge 检查在真删后由 `profile is None` 同样兜住
    （收敛候选本来就只取 `current_profiles`，作废的根本不进循环）。

    「系统的自动动作永不硬删」不变（修正 / 老化 / 复核 / 封顶都不删行）——
    这是**他点名要删**：删行 + 它的边 + 它的体检提议（`store.delete_profile`）。

    不再收 `reason`：那个"为什么不对"以前入库没人读、界面还问了他一句——
    现在不问（要留后路、要理由，都不在删除里）。
    """
    if store.get_profile(pid) is None:
        return "not_found"
    store.delete_profile(pid)
    return "deleted"


# ---------------------------------------------------------------------
# R6 镜像呈现（第一版的「做」）
# ---------------------------------------------------------------------

def render_mirror(store, topic: str | None = None, include_pending: bool = True) -> dict:
    """把「air 眼中的他」拉出来给人看（R6 的呈现动作）。

    第一版的范围：
      - **只在用户显式请求时**进入（「你觉得我是什么样的人」）——不自动开
      - 动作只有**呈现**：画像陈述 + 支撑场景（可追溯）+ 当前状态标记
      - 附否决入口（`can_reject`）：「哪条不对你直接说，我会改」

    两条硬规则：
      - **`pending` 的必须标出「还没印证够，只是猜测」**——
        不加区分地呈现，等于把猜测说成事实（那正是「不武断」要防的）
      - **敏感内容不进镜像**：air 眼中的你不该含「他有过创伤」这类标签。
        所以在 sources 里把敏感场景滤掉，只留一个计数（让人知道「有依据没展示」，
        而不是假装没有）

    每条依据带 `role`：`forming`（这条判断的**出处**）/ `supporting`（后来的印证）。
    出处由画像自带的 `evidence_pack` 认出来——**只用它的 id 集合，不拿它兜底**：
    依据里**已删的场景不显示**（引用 = `sources ∩ 现存节点`，2026-09-24）——
    他删掉的东西不该从快照里再露出来。

    第一版**不做**：不主动做情绪拆解引导、不主动给整合建议、
    不判断「你现在需要被分析」——那些都是诱导（逻辑层 §3②）。
    """
    profiles = store.current_profiles(status=None)
    if topic:
        profiles = [p for p in profiles if p.topic == topic]
    if not include_pending:
        profiles = [p for p in profiles if p.status == PROFILE_ESTABLISHED]

    shown = []
    for p in profiles:
        pack = {str(it.get("id")): it for it in (p.evidence_pack or [])
                if isinstance(it, dict) and it.get("id")}
        sources, hidden = [], 0
        for sid in scene_ids(p.sources):
            s = store.get_scene(sid)
            if s is None:
                continue        # 已删的不显示（引用 = sources ∩ 现存节点，2026-09-24）
            if s.sensitive:
                hidden += 1                 # 敏感不进镜像（只记数）
                continue
            # 区分「出处」和「印证」：出处是这条判断的形成依据（少、稳定），
            # 印证是后来累积的支撑。呈现时标出来，人才看得出这条判断是怎么来的。
            sources.append({"id": s.id, "title": s.title,
                            "time": (s.time_event or "")[:10],
                            "summary": s.text,
                            "role": "forming" if sid in pack else "supporting"})
        shown.append({
            "id": p.id,
            "topic": p.topic,                       # 主主题（兼容旧前端 / 旧脚本）
            "topics": p.topics or [p.topic],        # 多标签：1 主 + 最多 2 附
            "statement": p.statement,
            "status": p.status,
            "status_label": ("还没印证够，只是猜测" if p.status == PROFILE_PENDING else "已立"),
            "evidence": p.evidence,
            "sources": sources,
            "hidden_sensitive": hidden,
        })

    return {
        "profiles": shown,
        # 否决入口：措辞由对话层定，这里只给「能不能否」和默认说法
        # （"删掉"与行为一致：2026-09-24 起否决 = 真删，不是就地改）
        "can_reject": True,
        "reject_hint": "哪条不对你直接说，我删掉它",
        "patterns": pattern_stats(store),
    }


def pattern_stats(store, min_count: int = 2, limit: int = 8) -> list[dict]:
    """「遇到 X 情境 → 典型反应 → 结果」的统计（唤醒层 §3② 整体分析）。

    这是**人可以更好、AI 才能做**的那件事：人只活在当前这一刻，
    跨几十个场景统计自己的模式做不到；air 可以。

    组合是 `trigger_class × entity`——两个字段各自都很粗，
    但组合起来就有精度（这也正是 `trigger_class` 要做成枚举、
    实体要单独建索引的原因：**结构化的才能被统计**）。

    敏感场景**不进统计**（同「敏感不进镜像」）：这类模式一旦被呈现，
    就等于替用户把他的创伤总结成了一条规律。
    """
    buckets: dict[tuple[str, str], list] = {}
    for s in store.query_scenes(limit=1000):
        if not s.trigger_class or s.sensitive:
            continue
        names = store.entities_of_scene(s.id) or ["（无具体对象）"]
        for name in names:
            buckets.setdefault((s.trigger_class, name), []).append(s)

    out = []
    for (trigger_class, entity), group in buckets.items():
        if len(group) < min_count:
            continue
        reactions = [g.reaction for g in group if g.reaction]
        out.append({
            "trigger_class": trigger_class,
            "entity": entity,
            "count": len(group),
            "reactions": reactions[:3],
            "scene_ids": [g.id for g in group],
        })
    out.sort(key=lambda x: (-x["count"], x["trigger_class"]))
    return out[:limit]


# （原来这里有个 `world_view`：`subject='world'` 的主题 + 实体出现次数，
#   供编织模式注入「关于世界的了解」。2026-09-20 两套姿态退役后整块删除——
#   编织的做法已写进人格文件，不再有"切一个模式就换一套注入"这回事。
#   历史实现见 git。）


# （原来这里有个 `rejected_statements()`：查"被他否决过的陈述"。它自出生起
#   零生产消费（注释自认"刻意留的接口"），而 2026-09-24 起人的否决 = 真删画像
#   ——没有"被否决的陈述"这回事了，整块删除。历史见 git。）


# ---------------------------------------------------------------------
# 四、主题归并（「拿不准就新建」的补救口）
# ---------------------------------------------------------------------

def _mean_vec(vecs: list) -> list:
    """一组向量的均值。空输入、或维度不齐，都返回空列表（调用方据此跳过）。"""
    if not vecs:
        return []
    n = len(vecs[0])
    if any(len(v) != n for v in vecs):
        return []
    out = [0.0] * n
    for v in vecs:
        for k in range(n):
            out[k] += v[k]
    return [x / len(vecs) for x in out]


def merge_suggestions(store, emb=None, threshold: float | None = None) -> list[dict]:
    """找「可能是同一个主题、却被写成两种说法」的 topic 对——**只建议，不合并**。

    为什么必须有这一步：topic 是 LLM 自由生成的字符串，而「同一件事换个说法
    就断成两个主题」是这套系统最容易被写坏的地方——一个主题裂成两半，
    两边都凑不够 `step2_n` 条，**聚合与画像全废，而且一声不响**。
    设计上的对策本来是「拿不准就新建 + 留合并接口」，可 `merge_topics()`
    写完就**没有任何地方调用它**：裂缝一直敞着，接口等于没有。

    **为什么不自动合并**：合并是一次不可逆的语义断言——「这两件事是同一件」。
    相似度高只说明"看起来像"。把两次不同的经历并成一个主题，
    等于替用户断言他的生活——这正是「不武断」要避免的。
    所以这里只负责**把嫌疑列出来**，合不合由人点一下。

    两条相似路径，取较大者：
      - **措辞相近**：topic 字符串本身的向量
        （「用户·被当众批评的反应」≈「用户·被批评的反应」）
      - **内容相近**：各自名下场景向量的均值
        （措辞完全不同、却在讲同一件事时，靠这条兜住）
    """
    topics = store.list_topics()
    if len(topics) < 2:
        return []
    th = float(threshold if threshold is not None
               else cfgmod.cfg("distill", "topic_merge_threshold", default=0.86))

    info: dict[str, dict] = {}
    for t in topics:
        scenes = store.query_scenes(topic=t, limit=200)
        info[t] = {"n": len(scenes),
                   "sample": (scenes[0].title if scenes else ""),
                   "centroid": _mean_vec([s.emb for s in scenes if s.emb])}

    # 名称向量一次批量算（省调用次数）；场景均值本地算——场景向量已经存在库里，
    # 没必要为了比一次相似度再把几十条文本送出去。
    names = list(info.keys())
    name_vecs = None
    if emb is not None and getattr(emb, "available", False):
        got = emb.embed(names)
        if got and len(got) == len(names):
            name_vecs = got

    out: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            cands: dict[str, float] = {}
            if name_vecs is not None:
                cands["措辞相近"] = cosine(name_vecs[i], name_vecs[j])
            ca, cb = info[a]["centroid"], info[b]["centroid"]
            if ca and cb:
                cands["内容相近"] = cosine(ca, cb)
            if not cands:
                continue
            basis, sim = max(cands.items(), key=lambda kv: kv[1])
            if sim < th:
                continue
            # 小并大：条数少的那边更像分支，并进条数多的那边 = 把岔路接回主干
            small, big = (a, b) if info[a]["n"] <= info[b]["n"] else (b, a)
            out.append({
                "from": small, "to": big,
                "similarity": round(sim, 3), "basis": basis,
                "from_n": info[small]["n"], "to_n": info[big]["n"],
                "from_sample": info[small]["sample"], "to_sample": info[big]["sample"],
            })
    out.sort(key=lambda d: -d["similarity"])
    return out


def merge_topics_confirmed(store, from_topic: str, to_topic: str) -> dict:
    """执行主题合并——**只由人确认后调用**（仪表盘的按钮）。

    留痕的理由：合并是**不可逆的语义断言**。技术上当然还能再合回去，
    但前提是你还记得它们本来是分开的。记下「何时、把什么并进了什么、
    影响了多少条」，事后才谈得上复盘——尤其当画像因此变了的时候。
    """
    f = (from_topic or "").strip()
    t = (to_topic or "").strip()
    if not f or not t or f == t:
        return {"ok": False, "detail": "需要两个不同的主题名"}
    n = store.merge_topics(f, t)
    _write_merge_trace(f, t, n)
    return {"ok": True, "merged": n, "from": f, "to": t}


def merge_entities_confirmed(store, from_id: str, to_id: str) -> dict:
    """执行实体合并——**只由人确认后调用**（仪表盘实体页的按钮）。

    和主题合并同一性质：合并是一次**不可逆的语义断言**（"这两个名字是同一个人"）。
    系统不替人下这个判断（`entity.link_entities` 只做精确匹配 + 别名），
    但人下判断时得有个地方点——`add_entity(aliases=…)` 那个参数一直够不着
    （没有任何调用点、界面上也没有入口），这条就是补上的那个口子。
    （同 `merge_topics()` 的旧课：**写好了没接线，接口等于没有**。）

    留痕的理由同主题合并：事后要能回答「谁把哪两个名字并成了一个」——
    尤其当某人从此"合并"了经历、画像跟着变了的时候。
    """
    f = (from_id or "").strip()
    t = (to_id or "").strip()
    if not f or not t or f == t:
        return {"ok": False, "detail": "需要两个不同的实体编号"}
    src, dst = store.get_entity(f), store.get_entity(t)
    if src is None or dst is None:
        missing = "、".join(i for i, e in ((f, src), (t, dst)) if e is None)
        return {"ok": False, "detail": f"找不到实体 {missing}"}
    out = store.merge_entities(f, t)
    _write_change_trace("实体合并", [{
        "id": f, "title": f"{src.name} → {dst.name}",
        "from": src.name, "to": dst.name, "action": "合并",
        "affected": out["scenes"], "aliases": out["aliases"],
    }])
    return {"ok": True, "from": f, "to": t, "name": dst.name, **out}


def save_facts_confirmed(store, facts: dict, note: str = "") -> int:
    """人确认后写档案 + **留痕**；返回写入条数。

    档案和主题合并同属「**人的纠正**」——都是他自己说了算的东西，
    所以两者都留痕。漏了留痕的代价很具体：档案决定她**怎么称呼他**，
    而你没法回答「这个称呼谁改的、什么时候改的」——
    这是最不该说不清的一类改动。

    （空值 = 删除，见 `Store.set_user_fact`；留痕里也如实记成"清空"。）
    """
    before = {f["key"]: f["value"] for f in store.all_user_facts()}
    changes = []
    for k, v in (facts or {}).items():
        k, new = (k or "").strip(), (v or "").strip()
        if not k:
            continue
        old = before.get(k, "")
        if old == new:
            continue                       # 没变的不记账（否则每次保存都刷一堆空改动）
        changes.append({"key": k, "from": old, "to": new,
                        "action": "清空" if not new else ("新增" if not old else "修改")})
    n = store.save_user_facts(facts, note=note)
    if changes:
        _write_facts_trace(changes)
    return n


def save_lang_confirmed(store, lang: str) -> str:
    """写语言（中 / 英）+ 留痕，返回生效后的值。

    为什么留痕（同档案）：**语言也是她说话的一部分**——
    切了语言，对话原文（S0）和从它提取的东西都会跟着变，
    同属「他定的、改写入原料」的一类。非法值直接忽略（保持原值），
    不抛：这是人在界面上点的。
    """
    before = lang_from_prefs(store.all_prefs())
    after = lang if lang in LANGS else before
    if after != before:
        store.set_pref("lang", after)
        _write_change_trace("语言", [{"key": "语言", "from": before, "to": after,
                                      "action": "修改"}])
    return after


def update_scene_confirmed(store, scene_id: str, text: str = "", **fields) -> dict:
    """改一条场景的**可改字段**（2026-09-24 起不只摘要）+ 留痕。

    字段白名单与校验在 `store.set_scene_fields`（工具箱稿 §七）：能改 = 语义描述类，
    数值与系统字段不开。这里多做的事只有一件——**留痕**：
    **改前的值必须留下来**，不然改完就分不清「当时发生了什么」和「后来怎么理解的」，
    而分清这两件事正是原文存在的意义。原文永远不动，动的只是场景卡上的理解。
    """
    if text and "text" not in fields:
        fields["text"] = text            # `text` 是 `fields["text"]` 的简写（老调用点沿用）
    if not fields:
        return {"ok": False, "detail": "没说改成什么，不动它"}
    out = store.set_scene_fields(scene_id, fields)
    if not out.get("ok") or not out.get("changed"):
        return {"ok": bool(out.get("ok")), "changed": False, "id": scene_id,
                "detail": out.get("detail") or "内容没变"}
    applied = out["applied"]
    _write_change_trace("场景改动",
                        [{"id": scene_id, "field": SCENE_FIELD_LABELS.get(k, k),
                          "from": before, "to": after, "action": "修改"}
                         for k, (before, after) in applied.items()])
    fk, (before, after) = next(iter(applied.items()))
    return {"ok": True, "changed": True, "id": scene_id,
            "field": SCENE_FIELD_LABELS.get(fk, fk), "from": before, "to": after,
            "applied": {SCENE_FIELD_LABELS.get(k, k): v for k, v in applied.items()}}


def archive_scene_confirmed(store, scene_id: str) -> dict:
    """归档一条场景——**人的动作**（确认条的"归档冷存"那一档，工具箱稿 §3.4）。

    与删除的关键差别：**数据全在**。她不召回它（冷层），但画像的 `sources`
    引用不动、可打捞、可取消归档。所以它**不摘引用、不留痕**——
    归档的结果（`archived=1`）永久可查且可逆，留档没有对象。
    """
    s = store.get_scene(scene_id)
    if s is None:
        return {"ok": False, "detail": "找不到这条场景"}
    if s.archived:
        return {"ok": True, "changed": False, "id": scene_id, "detail": "它已经在冷层"}
    store.archive_scene(scene_id)
    return {"ok": True, "changed": True, "id": scene_id,
            "detail": f"{scene_id}「{s.title or ''}」已归档冷存——"
                      f"她不会再想起它，但数据都在，还能捞回来"}


def unarchive_scene_confirmed(store, scene_id: str) -> dict:
    """取消归档（**人的动作**）——"后悔"的出口（同归档，不留痕）。"""
    s = store.get_scene(scene_id)
    if s is None:
        return {"ok": False, "detail": "找不到这条场景"}
    if not s.archived:
        return {"ok": True, "changed": False, "id": scene_id, "detail": "它不在冷层"}
    store.unarchive_scene(scene_id)
    return {"ok": True, "changed": True, "id": scene_id,
            "detail": f"{scene_id}「{s.title or ''}」已回到热层"}


def delete_scene_confirmed(store, scene_id: str) -> dict:
    """真删一条场景（**人的动作**：删行；**不留快照、不留痕**）。

    **删行 + 它的边 + 实体链接**——**不摘引用**（2026-09-24：引用读成
    `sources ∩ 现存节点`，读取端过滤，见 `store.delete_scene`）。

    原来还有"① 备份""③ 留痕"，已去（理由见工具箱稿 §3.4 的 ⚠️）：
    留后路由**归档**那一档承担（人点的），删 = 真删——不再挂隐形安全网。
    （原 `reason` 参数一并删了：它只为留痕存在，留痕没了它就是死参数。）

    ⚠️ 真删数据只有人的显式调用这一条路（仪表盘按钮 / 确认条；三层同一套——
    摘要、画像的「彻底删除」同性质），**永远不会**进工具的自主档。
    """
    s = store.get_scene(scene_id)
    if s is None:
        return {"ok": False, "detail": "找不到这条场景"}
    out = store.delete_scene(scene_id)
    if not out.get("deleted"):
        return {"ok": False, "id": scene_id, "detail": "没有删成"}
    return {"ok": True, "id": scene_id}


def update_summary_confirmed(store, summary_id: str, text: str,
                             field: str = "text") -> dict:
    """改一条 S2 的**可改字段**（叙述 / 主题，2026-09-24）+ 留痕——
    同 `update_scene_confirmed` 的理由："改前的值"库里没有，是不可再生信息
    （分清"当时发生什么"和"后来怎么理解"）。

    `field` 默认 `text`（叙述）——老调用点（界面 / 她）沿用；
    白名单与校验在 `store.set_summary_fields`。
    """
    s = store.get_summary(summary_id)
    if s is None:
        return {"ok": False, "detail": "找不到这条摘要"}
    out = store.set_summary_fields(summary_id, {field: text})
    if not out.get("ok") or not out.get("changed"):
        return {"ok": bool(out.get("ok")), "changed": False, "id": summary_id,
                "detail": out.get("detail") or "内容没变"}
    before, after = out["applied"][field]
    label = SUMMARY_FIELD_LABELS.get(field, field)
    _write_change_trace("摘要改动", [{"id": summary_id, "field": label,
                                     "from": before, "to": after, "action": "修改"}])
    return {"ok": True, "changed": True, "id": summary_id, "field": label,
            "from": before, "to": after}


def delete_summary_confirmed(store, summary_id: str) -> dict:
    """真删一条 S2（**人的处置**，2026-09-24）：**只删一行**——**不留痕、不摘引用**。

    "删了重聚"：它收的那批 S1 变回"未被覆盖"，下次提炼重聚一条新的
    （`store.delete_summary` 内含）——这是它的"改"。
    """
    s = store.get_summary(summary_id)
    if s is None:
        return {"ok": False, "detail": "找不到这条摘要"}
    out = store.delete_summary(summary_id)
    if not out.get("deleted"):
        return {"ok": False, "id": summary_id, "detail": "没有删成"}
    return {"ok": True, "id": summary_id}


def archive_summary_confirmed(store, summary_id: str) -> dict:
    """归档一条 S2（**人的动作**）——她不召回它，数据全在、可取消。不留痕（可逆、查库即真相）。"""
    s = store.get_summary(summary_id)
    if s is None:
        return {"ok": False, "detail": "找不到这条摘要"}
    if s.archived:
        return {"ok": True, "changed": False, "id": summary_id, "detail": "它已经在冷层"}
    store.archive_summary(summary_id)
    return {"ok": True, "changed": True, "id": summary_id,
            "detail": f"{summary_id} 已归档冷存——她不会再想起它，数据都在"}


def unarchive_summary_confirmed(store, summary_id: str) -> dict:
    """取消归档（**人的动作**）——"后悔"的出口。"""
    s = store.get_summary(summary_id)
    if s is None:
        return {"ok": False, "detail": "找不到这条摘要"}
    if not s.archived:
        return {"ok": True, "changed": False, "id": summary_id, "detail": "它不在冷层"}
    store.unarchive_summary(summary_id)
    return {"ok": True, "changed": True, "id": summary_id,
            "detail": f"{summary_id} 已回到热层"}


def update_profile_confirmed(store, pid: str, text: str,
                             field: str = "statement") -> dict:
    """改一条画像的**可改字段**（**人的动作**，2026-09-24）：

      - `field="statement"`（陈述，默认）→ 走**修正**：旧版进历史、新版回 pending
        （与"她"的修正共用 `distill.revise_profile`：同一个 topic 的下一个版本；
        差别在 `by=INVALIDATED_USER`——统计与呈现分得清"我改主意"和"他纠正我"。
        证据用**现有 sources**：人改的是"说法"，不是"依据"，不新增证据）；
      - `field="topic"`（主题标签）→ **原地改**：改的是归类，判断本身不动
        （依据链 / 印证 / 时间戳照旧，也不产生新版本）。**1-3 个**（逗号隔开，
        第一个 = 主主题）——`text` 交给 `store.set_profile_topics` 解析。
    """
    old = store.get_profile(pid)
    if old is None:
        return {"ok": False, "detail": "找不到这条画像"}
    if field == "topic":
        out = store.set_profile_topic(pid, text)
        if out.get("ok") and out.get("changed"):
            before, after = out["applied"]["topic"]
            _write_change_trace("画像改动", [{"id": pid, "field": "主题",
                                             "from": before, "to": after,
                                             "action": "修改"}])
            return {"ok": True, "changed": True, "id": pid, "field": "主题",
                    "from": before, "to": after,
                    "detail": f"{pid} 的主题改成「{after}」——"
                              f"判断本身不动（依据 / 印证照旧）"}
        return {"ok": bool(out.get("ok")), "changed": False, "id": pid,
                "detail": out.get("detail") or "内容没变"}
    after = (text or "").strip()
    if not after:
        return {"ok": False, "detail": "没说改成什么，不动它"}
    if after == (old.statement or ""):
        return {"ok": True, "changed": False, "id": pid, "detail": "内容没变"}
    from .distill import revise_profile      # 延迟 import（防环，同 tools 的先例）
    new_id = revise_profile(store, pid, after, list(old.sources or []),
                            by=INVALIDATED_USER)
    return {"ok": True, "changed": True, "id": pid, "new_id": new_id,
            "detail": f"{pid} 改成「{after}」——旧版进了历史，新版要从证据重新攒"}


def archive_profile_confirmed(store, pid: str) -> dict:
    """归档一条画像（**人的动作**）——不召回它，数据全在、可取消；**旧版历史不动**。"""
    p = store.get_profile(pid)
    if p is None:
        return {"ok": False, "detail": "找不到这条画像"}
    if p.invalidated_at:
        return {"ok": True, "changed": False, "id": pid, "detail": "它已经不在生效中"}
    store.archive_profile(pid)
    return {"ok": True, "changed": True, "id": pid,
            "detail": f"{pid} 已归档冷存——不再影响她，但留着、能恢复"}


def unarchive_profile_confirmed(store, pid: str) -> dict:
    """取消归档（**人的动作**）——只清"归档"那一类失效，修正的历史不动。"""
    p = store.get_profile(pid)
    if p is None:
        return {"ok": False, "detail": "找不到这条画像"}
    if p.invalidated_by != INVALIDATED_ARCHIVE:
        return {"ok": True, "changed": False, "id": pid, "detail": "它不是归档状态"}
    store.unarchive_profile(pid)
    return {"ok": True, "changed": True, "id": pid, "detail": f"{pid} 已回到生效中"}


def delete_many_confirmed(store, scene_ids) -> dict:
    """一次删多条（**人的动作**）：逐条删行（**不留快照、不留痕、不摘引用**）。

    **先校验再动手**：有一个编号找不到就**一条都不删**——
    删除不可撤回，宁可让他确认一遍编号，也不要删掉一部分才发现填错了。
    （"要么全删、要么不删"这一条在所有档上都不变，工具箱稿 §3.4。）

    与 `delete_scene_confirmed` 只差"批量校验"这一步——备份与留痕
    都已去掉（2026-09-23，见那边的说明）。
    """
    ids = [str(i).strip() for i in (scene_ids or []) if str(i).strip()]
    if not ids:
        return {"ok": False, "detail": "没给编号"}

    scenes, missing = [], []
    for sid in ids:
        s = store.get_scene(sid)
        if s is None:
            missing.append(sid)
        else:
            scenes.append(s)
    if missing:
        return {"ok": False,
                "detail": f"没找到 {'、'.join(missing)}——"
                          f"一条都没删（先确认编号再删）"}

    for s in scenes:
        store.delete_scene(s.id)
    return {"ok": True, "ids": [s.id for s in scenes]}


# ---------------------------------------------------------------------
# 三层分派：**改与删，三层同一套**（2026-09-24，工具箱稿 §3.4）
#
# 按编号前缀分派——S1 场景 / S2 摘要 / S3 画像，各调各的方法；
# 流程与后果一致（"删谁谁断，上面少一条素材"是自然后果，不是规则）。
# "要么全删、要么不删"跨层统一先校验：有找不到的编号就一条都不动。
#
# 原来住在 `chat.py`（确认条的执行）——2026-09-24 搬到 `weave`：
# 这才是"人的动作"的出口，**两条入口共用**（对话确认条 / 台账页按钮）。
# ---------------------------------------------------------------------

def ids_by_layer(ids: list[str]) -> dict[str, list[str]]:
    """编号按层分组（认不出前缀的落在分组之外，由调用方报错）。"""
    out: dict[str, list[str]] = {"S1": [], "S2": [], "S3": []}
    for i in ids:
        up = str(i).upper()
        for layer in ("S1", "S2", "S3"):
            if up.startswith(layer):
                out[layer].append(i)
                break
    return out


def _unknown_ids(ids: list[str]) -> list[str]:
    """编号里认不出前缀的那些（S1 / S2 / S3 之外——M 与手滑的编号都在这里）。

    「要么全动、要么不动」的第一道闸：认不出就**整批拒绝**，不做"能认的照做、
    认不出的悄悄跳过"——那是静默的部分执行。
    （2026-09-24 检查修：原来只有 `delete_by_layer` 有这道闸，归档 / 取消归档没有——
    `archive_by_layer([M-0001])` 会返回 `ok=True`「归档了 0 条」，界面据此显示成功。）
    """
    known = ids_by_layer(ids)
    return [i for i in ids if not any(i in v for v in known.values())]


def _exists_by_layer(store, i: str) -> bool:
    up = str(i).upper()
    if up.startswith("S1"):
        return store.get_scene(i) is not None
    if up.startswith("S2"):
        return store.get_summary(i) is not None
    if up.startswith("S3"):
        return store.get_profile(i) is not None
    return False


def delete_by_layer(store, ids: list[str]) -> dict:
    """三层分派删除（**先全查再动手**：有一个找不到就一条都不删）。"""
    unknown = _unknown_ids(ids)
    if unknown:
        return {"ok": False, "detail": f"认不出编号：{'、'.join(unknown)}"}
    missing = [i for i in ids if not _exists_by_layer(store, i)]
    if missing:
        return {"ok": False,
                "detail": f"没找到 {'、'.join(missing)}——一条都没删（先确认编号再删）"}
    known = ids_by_layer(ids)
    done: list[str] = []
    if known["S1"]:
        out = delete_many_confirmed(store, known["S1"])
        if not out.get("ok"):
            return {"ok": False, "detail": out.get("detail") or "没删成"}
        done += out["ids"]
    for i in known["S2"]:
        if delete_summary_confirmed(store, i).get("ok"):
            done.append(i)
    for i in known["S3"]:
        if user_reject_profile(store, i) == "deleted":
            done.append(i)
    return {"ok": True, "ids": done,
            "detail": f"删掉了 {len(done)} 条（{'、'.join(done)}）——真删，没有备份。"}


def archive_by_layer(store, ids: list[str]) -> dict:
    """三层分派归档——不召回、数据在、可取消（与删除的差别只剩"行在不在"）。

    **先全查再动手**（2026-09-24 检查修，与删除同一纪律）：认不出前缀的、
    找不到的，**整批拒绝**——原来这两道都没有：给一个 M 编号会得到
    `ok=True`「归档了 0 条」（界面据此显示成功，其实什么都没动），
    混合编号还会"能认的照做、认不出的静默跳过"。
    本来就在冷层的记进 `skipped`、不进 `done`——**不虚报条数**。
    """
    unknown = _unknown_ids(ids)
    if unknown:
        return {"ok": False,
                "detail": f"认不出编号：{'、'.join(unknown)}"
                          f"（归档只对 S1 场景 / S2 摘要 / S3 画像）"}
    missing = [i for i in ids if not _exists_by_layer(store, i)]
    if missing:
        return {"ok": False,
                "detail": f"没找到 {'、'.join(missing)}——一条都没动（先确认编号）"}
    known = ids_by_layer(ids)
    done: list[str] = []
    skipped: list[str] = []
    for i in known["S1"]:
        out = archive_scene_confirmed(store, i)
        if not out.get("ok"):
            return {"ok": False, "detail": out.get("detail") or f"{i} 没归档成"}
        (done if out.get("changed") else skipped).append(i)
    for i in known["S2"]:
        out = archive_summary_confirmed(store, i)
        if not out.get("ok"):
            return {"ok": False, "detail": out.get("detail") or f"{i} 没归档成"}
        (done if out.get("changed") else skipped).append(i)
    for i in known["S3"]:
        out = archive_profile_confirmed(store, i)
        if not out.get("ok"):
            return {"ok": False, "detail": out.get("detail") or f"{i} 没归档成"}
        (done if out.get("changed") else skipped).append(i)
    if not done:
        return {"ok": True, "changed": False, "ids": [],
                "detail": f"{'、'.join(skipped)} 本来就在冷层——没动。"}
    tail = f"（另有 {'、'.join(skipped)} 本来就在冷层）" if skipped else ""
    return {"ok": True, "changed": True, "ids": done,
            "detail": f"归档了 {len(done)} 条（{'、'.join(done)}）——"
                      f"她不会再想起它，数据都在，还能捞回来。{tail}"}


def unarchive_by_layer(store, ids: list[str]) -> dict:
    """三层分派取消归档（"后悔了"的出口）——与归档对称，三层同一套。

    纪律同 `archive_by_layer`（2026-09-24 检查修）：认不出 / 找不到**整批拒绝**；
    本来就不在冷层的记进 `skipped`，不虚报"回热层了 N 条"。
    """
    unknown = _unknown_ids(ids)
    if unknown:
        return {"ok": False,
                "detail": f"认不出编号：{'、'.join(unknown)}"
                          f"（取消归档只对 S1 场景 / S2 摘要 / S3 画像）"}
    missing = [i for i in ids if not _exists_by_layer(store, i)]
    if missing:
        return {"ok": False,
                "detail": f"没找到 {'、'.join(missing)}——一条都没动（先确认编号）"}
    known = ids_by_layer(ids)
    done: list[str] = []
    skipped: list[str] = []
    for i in known["S1"]:
        out = unarchive_scene_confirmed(store, i)
        if not out.get("ok"):
            return {"ok": False, "detail": out.get("detail") or f"{i} 没回热层"}
        (done if out.get("changed") else skipped).append(i)
    for i in known["S2"]:
        out = unarchive_summary_confirmed(store, i)
        if not out.get("ok"):
            return {"ok": False, "detail": out.get("detail") or f"{i} 没回热层"}
        (done if out.get("changed") else skipped).append(i)
    for i in known["S3"]:
        out = unarchive_profile_confirmed(store, i)
        if not out.get("ok"):
            return {"ok": False, "detail": out.get("detail") or f"{i} 没回热层"}
        (done if out.get("changed") else skipped).append(i)
    if not done:
        return {"ok": True, "changed": False, "ids": [],
                "detail": f"{'、'.join(skipped)} 本来就不在冷层——没动。"}
    tail = f"（另有 {'、'.join(skipped)} 本来就不在冷层）" if skipped else ""
    return {"ok": True, "changed": True, "ids": done,
            "detail": f"回热层了 {len(done)} 条（{'、'.join(done)}）——她又能想起它。{tail}"}


def update_by_layer(store, sid: str, text: str, field: str = "text") -> dict:
    """三层分派改：S1 / S2 原地改留旧值；S3 的**陈述**走修正（主题是原地改）。

    **每层各有一张可改字段表**（2026-09-24：三层都能改标签，见 `store.LAYER_EDIT`）：
    场景最多（标题 / 主题 / 事件时间 / 情境 / 反应 / 情境类），摘要 / 画像 =
    主文本 + 主题；数值与系统字段不开。
    字段名按层归一（中文名 / 英文键 / 口语说法都认）——界面直接传"主题"、
    她传"topic"，到这里是同一个。
    """
    up = (sid or "").strip().upper()
    if not up.startswith(("S1", "S2", "S3")):
        # 认不出就**当场拒绝**（2026-09-24 检查修）：`layer_edit` 的约定是
        # "认不出当 S1"，于是 M（备忘）会拿场景的字段表来校验——过不了就回
        # "能改：标题 / 摘要 / 主题…"（答非所问），过得去就去找一条不存在的场景。
        hint = "；M 是备忘，要了结它用 `close_memo`" if up.startswith("M") else ""
        return {"ok": False,
                "detail": f"认不出编号：{sid or '空'}——改只对 S1 场景 / S2 摘要 / "
                          f"S3 画像{hint}"}
    rule = layer_edit(sid)
    raw = (field or "").strip()
    key = rule["aliases"].get(raw, "") if raw else rule["main"]
    if not key or key not in rule["editable"]:
        return {"ok": False,
                "detail": f"不认的字段「{field}」——{sid or '这条'} 能改："
                          f"{'、'.join(rule['labels'].values())}"}
    if up.startswith("S2"):
        out = update_summary_confirmed(store, sid, text, key)
    elif up.startswith("S3"):
        out = update_profile_confirmed(store, sid, text, key)
    else:
        if key == "topic":
            # 场景的主题是**单值**（聚合分组的键）——界面直接填多个的话，原来会
            # 原样存成一条含分隔符的"主题"（她那条路会拒绝，两边不一致；
            # 2026-09-24 检查修）。判据与她那边（`revise_memory`）同一套。
            parts = split_topics(text)
            if not parts:
                return {"ok": False, "detail": "主题不能为空——给它一个（场景是单值）"}
            if len(parts) > 1:
                return {"ok": False,
                        "detail": "场景（S1）的主题是**单值**——它决定聚合分组，"
                                  "多主题只在摘要 / 画像上（S2 / S3）能挂"}
            out = update_scene_confirmed(store, sid, topic=parts[0])
        else:
            out = update_scene_confirmed(store, sid, **{key: text})
    if out.get("ok") and out.get("changed"):
        return {"ok": True, "detail": f"改好了：{sid} →「{text}」（旧的说法留了痕）"}
    return {"ok": False, "detail": out.get("detail") or "内容没变"}


def _write_change_trace(kind: str, changes: list[dict]) -> None:
    """「他自己定的东西」的改动留痕——档案和语言共用这一个写法。

    留痕的理由见 `save_facts_confirmed`：这类改动是最不该说不清的一类。
    """
    append_trace(kind, [{"ts": now_str(), **c} for c in changes])


def _write_facts_trace(changes: list[dict]) -> None:
    """档案改动留痕（同否决 / 合并的理由：人对系统的纠正要可见）。"""
    _write_change_trace("档案", changes)


def _write_merge_trace(from_topic: str, to_topic: str, n: int) -> None:
    """合并留痕（同各处 trace：**人对系统的纠正要可见**）。"""
    append_trace("合并", {"ts": now_str(), "from": from_topic,
                          "to": to_topic, "affected": n})
