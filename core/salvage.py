"""打捞：**人主动从原文里找回一件东西**。

它和下钻是两件事，别混：

  - **下钻**（`store.get_raws_by_scene`，每轮唤醒用）：只能经 `scene_id`。
    原文**没有索引**，「S0 最难召回」是物理实现的——那是有意的像人之处
    （存储层硬约束 1，有测试盯着）。
  - **打捞**（本模块）：他要找回一件东西时，当然该能翻原文。
    这是**显式的、慢的、只由人发起**的动作，**不进自动唤醒的路径**。

设计稿的三级追溯（存储层）：追溯包（永远）→ 场景（保护期内即时）→ **打捞（期后，慢）**。
这里就是最后那个「慢」的口子——以前只在稿子里，没有实现。

**为什么删之前还是备份**：备份恢复的是**状态**（当时那张卡、它在画像里的引用位置），
打捞恢复的是**内容**（照原文重新记一遍，是一张新的卡）。两者补的不是同一个东西。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L9 打捞
#   上游    ：config、embedding（语义找）、store（读原文目录与 scene）
#   下游    ：dashboard（「打捞」页）
#   对外入口：`search_raws` / `rebuild_confirmed` / `recent_dialogue`
#   边界    ：**不进自动唤醒**。它是翻文件，慢——这正是「S0 最难召回」的物理形状
# ---------------------------------------------------------------------
from __future__ import annotations

import re

from . import persona as persona_file
from .embedding import cosine
from .store import RAW_HEAD, append_trace, now_str

_WORD = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9]+")


def _days(store, date_from: str, date_to: str) -> list:
    """日期范围内的文档（**先用时间缩范围**——翻全部原文太慢，也没必要）。"""
    files = sorted(store.raws_dir().glob("*.md"), reverse=True)
    if date_from:
        files = [p for p in files if p.stem >= date_from[:10]]
    if date_to:
        files = [p for p in files if p.stem <= date_to[:10]]
    return files


def _sections(text: str) -> list[tuple[str, str, str]]:
    """把一天那份文档切成 `(场景编号, 时间, 正文)`。

    切分点**只有场景标题**（`## S1-0043 · 2026-10-07 02:59:57`，判据见
    `store.RAW_HEAD`），不是「行首 `## `」：正文里常有大段粘贴
    （他贴过 README、贴过人格设定），那些自带的 `## 小标题` 会把一段对话
    切成七八节——2026-10-07 事故就是这样：历史回填（最近 8 段）里塞进
    「跑起来」「目录」「文档」这类空段，真正的 S1-0043 只剩开头三句，
    看着就像"最近两条对话没了"。
    """
    out: list[tuple[str, str, str]] = []
    cur_id, cur_time, buf = "", "", []
    for line in (text or "").splitlines():
        m = RAW_HEAD.match(line)
        if m:
            if cur_id and buf:
                out.append((cur_id, cur_time, "\n".join(buf).strip()))
            cur_id = m.group(1)
            cur_time = line[m.end():].strip()
            buf = []
        else:
            buf.append(line)
    if cur_id and buf:
        out.append((cur_id, cur_time, "\n".join(buf).strip()))
    return out


def _persona_names() -> list[str]:
    """`self/personas/*.md` 的名字——原文行首可能出现的说话人（加文件即生效）。

    2026-10-05 起转发到 `core.persona.names()`（原来这里单独扫一遍目录，
    理由是"不把 salvage 和 chat 绑在一起"；现在扫目录的逻辑只有一个轻量出处，
    这份重复就该收掉——**停用的人格（`.md.off`）也自动不在名单里**）。
    """
    return persona_file.names()


def _split_dialogue(body: str) -> list[dict]:
    """把一节原文拆成 `[{"speaker", "text", "persona"?}]`：
    `名字:` 开新条（全角冒号也认），续行（含空行）并进当前条。

    名字从行首取（2026-09-21）：`用户` → speaker=user；其余（air / mia / xina……）
    → speaker=air + `persona` 记原名。原文从这天起按**当时的真名**署名
    （见 `scene.render_conversation`）；旧文档一律 "air"，天然兼容。
    """
    names = ["用户"] + _persona_names()
    marks = [(f"{n}:", n) for n in names] + [(f"{n}：", n) for n in names]
    out: list[dict] = []
    for line in (body or "").splitlines():
        s = line.strip()
        if not s:
            if out:
                out[-1]["text"] += "\n"      # 段间空行留着（渲染前 trim）
            continue
        hit = next(((n, s[len(m):]) for m, n in marks if s.startswith(m)), None)
        if hit:
            name, text = hit
            msg = {"speaker": "user" if name == "用户" else "air",
                   "text": text.strip()}
            if name != "用户":
                msg["persona"] = name
            out.append(msg)
        elif out:
            out[-1]["text"] += "\n" + s
    return [m for m in out if m["text"].strip()]


def recent_dialogue(store, limit: int = 8) -> list[dict]:
    """最近几段对话原文（**已提取的那部分**）——重启后对话区回填用。

    与打捞的区别：打捞是"按一句话去找"，这里是"把最近几段拿回来"——
    重启后界面要的是"刚才聊到哪了"。窗口里**还没提取**的那段由
    `shortterm.json` 负责（`dashboard.App.history`）——两者不重叠：
    提取时同一步"写 raws + 清窗口"。读最近两天的文档就够，
    更早的走「打捞」页。
    """
    out = []
    files = sorted(store.raws_dir().glob("*.md"), reverse=True)[:2]
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for sid, when, body in _sections(text):
            out.append({"id": sid, "time": when,
                        "messages": _split_dialogue(body)})
    out.sort(key=lambda s: s["time"])
    return out[-limit:]


def _snippet(body: str, q: str, n: int = 200) -> str:
    """截一段**带命中位置**的片段：整段扔给他，不如让他看到命中的是哪句。"""
    b = (body or "").strip()
    if len(b) <= n:
        return b
    words = _WORD.findall(q or "")
    pos = max((b.lower().find(w.lower()) for w in words), default=-1)
    if pos < 0:
        return b[:n] + "…"
    start = max(0, pos - n // 3)
    return ("…" if start else "") + b[start:start + n] + "…"


def search_raws(store, query: str, date_from: str = "", date_to: str = "",
                emb=None, limit: int = 20) -> list[dict]:
    """按时间缩小范围，再在原文里找（语义优先，没有向量就用字符重叠）。

    **不建索引**——就是翻文件。它慢，但这是打捞，不是每轮要做的事。
    """
    q = (query or "").strip()
    qv = (emb.embed_one(q)
          if (q and getattr(emb, "available", False)) else None)
    out: list[dict] = []
    for p in _days(store, date_from, date_to):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for sid, when, body in _sections(text):
            if qv is not None:
                bv = emb.embed_one(body[:800])
                score = cosine(qv, bv) if bv else 0.0
            elif q:
                from .recall import char_overlap      # 降级：字符重叠兜底
                score = char_overlap(q, body)
            else:
                score = 1.0                            # 没给词 = 只要这个范围的
            if score <= 0:
                continue
            out.append({"scene_id": sid, "day": p.stem, "time": when,
                        "score": round(float(score), 3),
                        "snippet": _snippet(body, q),
                        # 库里还在不在（归档算在，**只有真删了的才是 False**）
                        "in_library": store.get_scene(sid) is not None})
    out.sort(key=lambda x: -x["score"])
    return out[:limit]


def _to_messages(content: str, ts: str = "") -> list[dict]:
    """把原文（`用户: …` / `air: …`）还原成消息列表，好喂给提取。

    `ts` 是这一节的记录时间（小节标题上的），**整段共用**——原文按条存的是
    「谁说了什么」，没有逐条时刻，能给的最细就是它。
    带上它的意义在 `extract_scene` 那边：它取「最早的消息时间」当
    `time_event`，不带的话重建出来的场景时间是**重建那一刻**——
    而原文里明明记着它是什么时候发生的。

    ⚠️ 解析**复用 `_split_dialogue`**（2026-09-21）：说话人名字认当时的真名
    （air / mia / xina……）——各写一份的话，新署名的 "mia:" 行会被当成续行
    整段吞掉，重建出来的对话少一半（还要顺带保留续行，旧实现是直接丢的）。
    """
    msgs = _split_dialogue(content)
    for m in msgs:
        m["ts"] = ts
    return msgs


def _adopt_loops(store, scene, llm, old_scene_id: str) -> dict:
    """重建的卡**认领它身上的未了结**：能认旧的认旧的，认不到才新记（2026-10-05）。

    为什么要有它：重建原来只落卡、不落 memo——而提取链路是**两者都写**
    （`distill._write_memos`），于是重建出来的钩子**没有 memo 管**：
    既关不掉也不会退役，卡片上的「→ 未定」就永远挂着。

    三条路（按 `content` 认领——旧 memo 就是照这条钩子抽出来的，文字原样；
    重建时模型若把那条钩子写得不一样，就认不到，走第三条）：

      1. 原卡留下的 memo（**删卡不清 memo**，它的 `scene_id` 悬着）里有同 `content` 的：
         **把 `scene_id` 指到新卡**——同一件事在库里只留一条，引用也接回来；
      2. 认领到的那条**已经关闭**：把"不再提了"回流到新钩子（标 `retired_at`）——
         **不写 `closed_at`**：它为什么关的（了结 / 退役 / 手划）记在它自己的行
         与 `备忘-*.jsonl` 里；新卡上能确定的是"这事不再提了"，别假装成了结；
      3. 认不到的钩子：走正常写入（`_write_memos`：新条 + 编号 + 分类）。

    认不到对应钩子的旧 memo 仍悬着——它的钩子这次没被抽出来，没什么可接的
    （悬空只让"闭合回流标钩子"落空，memo 自己的生命周期照常走）。

    ⚠️ 边角：**被删的编号会被复用**（`_next_id` 按当前最大值 +1）——重建的卡常常
    又拿到原来那个 `S1-000x`，旧 memo 的引用"碰巧"就指对了，第 1 条成为无操作。
    认领逻辑真正管用的是**编号错开**（中间又建了别的卡）与第 2、3 条。

    返回 `{"adopted", "marked", "written"}`（留痕与测试用）。
    """
    from .distill import _write_memos        # 延迟 import（同 scene→recall 的先例，防环）
    from .model import MEMO_CLOSED

    old = store.memos_by_scene(old_scene_id)
    by_content = {(m.content or "").strip(): m for m in old
                  if (m.content or "").strip()}

    adopted: list[str] = []
    marked: list[str] = []
    fresh: list = []
    for loop in scene.open_loops or []:
        content = str((loop or {}).get("content") or "").strip()
        m = by_content.get(content) if content else None
        if m is None:
            fresh.append(loop)
            continue
        store.set_memo_scene(m.id, scene.id)     # 接回新卡（悬空引用归位）
        adopted.append(m.id)
        if m.status == MEMO_CLOSED:
            store.retire_open_loop(scene.id, content,
                                   loop_id=str(loop.get("loop_id") or ""))
            marked.append(m.id)

    written: list[str] = []
    if fresh:
        import copy
        # 浅拷贝：只换 `open_loops` 交给写入——别动库里那张卡（同一对象）
        sub = copy.copy(scene)
        sub.open_loops = fresh
        written = _write_memos(store, sub, llm)
    return {"adopted": adopted, "marked": marked, "written": written}


def rebuild_confirmed(store, llm, scene_id: str, on_date: str = "",
                      emb=None) -> dict:
    """照着原文**重新记一遍**（删掉的是理解，原文还在，所以记得回来）。

    和"从备份恢复"的区别：备份给回的是当时那一张卡（含它在画像里的引用位置），
    打捞给回的是**照原文重新提取的一张新卡**。所以这不是撤销，是重记。

    重建的卡也**认领它的未了结**（`_adopt_loops`，2026-10-05）：能认旧的认旧的、
    认不到才新记——"卡"和"卡上还没完的事"一起回来，钩子从此有 memo 管
    （关得掉、退得了，不会永远挂着「→ 未定」）。
    """
    from .scene import extract_scene

    # **没有模型就不能重建**：`extract_scene` 拿不到模型时返回的是**保守默认值**
    # （那套默认值是给"不崩"用的，不是拿去落库的）——照它写进去就是一张空卡，
    # 而且**静默**。宁可不做，也不能往记忆里塞垃圾。
    if llm is None or not getattr(llm, "available", lambda: False)():
        return {"ok": False,
                "detail": "没有可用的模型，重建不了——它得把原文重新读一遍。"
                          "先去「设置」里配好 LLM。"}
    raws = store.get_raws_by_scene(scene_id, on_date=on_date)
    if not raws:
        return {"ok": False, "detail": "原文里没有这一条（也许从来没存过）"}
    if store.get_scene(scene_id) is not None:
        return {"ok": False,
                "detail": f"{scene_id} 还在库里（可能只是归档了）——它没被删，不用打捞。"}
    # 带上原文小节的时间：`extract_scene` 拿它当事件时间。
    # 不带的话，重建出来的场景时间是"重建那一刻"——而原文里记着真时间。
    messages = _to_messages(raws[0].content, ts=raws[0].created_at)
    if not messages:
        return {"ok": False, "detail": "这段原文看不出对话，没法重建"}

    scene, _digest, entities, verdict = extract_scene(
        messages, llm, emb_service=emb, source="salvage")
    # 「值不值得存」这个判据在不同层叫过 `worth` / `worth_saving`，两种都认
    if not verdict.get("worth", verdict.get("worth_saving", True)):
        return {"ok": False,
                "detail": f"重新看了一遍，这段不值得存："
                          f"{verdict.get('reason') or verdict.get('skip_reason') or ''}"}
    store.add_scene(scene)
    from .entity import link_entities
    link_entities(scene.id, entities or [], store)
    # 卡回来了，它上面的"还没完的事"也一起回来（认领旧的 / 认不到才新记）
    memos = _adopt_loops(store, scene, llm, scene_id)
    _trace({"scene_id": scene.id, "from": scene_id,
            "day": on_date or (raws[0].created_at or "")[:10],
            "title": scene.title or "",
            "memos": memos})
    note = ""
    if memos["adopted"] or memos["written"]:
        note = (f"；未了结的事跟着回来了"
                f"（认领 {len(memos['adopted'])} · 新记 {len(memos['written'])}）")
        if memos["marked"]:
            note += f"，其中 {len(memos['marked'])} 条早已了结 / 退役（不再提）"
    return {"ok": True, "id": scene.id, "title": scene.title or "",
            "detail": f"照着原文重新记下了：{scene.id}「{scene.title}」"
                      f"（原来那条是 {scene_id}）{note}"}


def _trace(record: dict) -> None:
    """打捞也留痕——**翻过什么、找回过什么**同样要能回看。"""
    append_trace("打捞", {"ts": now_str(), **record})
