"""对话实验：按脚本回放多轮对话，出一份能读的报告。

跑法：
    python run_experiment.py --script private/实验/example_stress.json --fresh
    run.cmd exp private/实验/example_stress.json

**脚本自备**：仓库不带实验素材（脚本与报告都是私人的，撤进了 `private/`，
见 `.gitignore`）——脚本格式看本文件头部与 `--help`。

为什么需要它（而不只是网页里聊）：**同一段对话要能反复跑**。
改一个阈值 → 重跑 → 对比「召回变了没有」——没有这个，参数就只能凭感觉调，
而「凭感觉调一个记忆系统的参数」基本上等于随机数。

两个刻意的设计：

1. **每次实验一个独立的库**（`data/experiments/<脚本名>.db`）。
   实验数据不该混进真实记忆里——那是两回事，混了就没法收拾。
2. **每个会话一个独立的窗口**。跨会话正是要观察的东西；
   如果窗口串在一起，就测不出「隔了几天，她还记得吗」。

报告落 `<PATHS.experiments>/reports/<脚本名>-<时间>.md`（人读）与 `.json`（机器读）——
2026-10-05 起这个路径是 `private/实验/`（**不进版本库**：报告里有对话原文）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Windows 控制台默认不是 UTF-8（GBK），而进度输出里有 `↳` 这类编不进 GBK 的符号——
# 一条 print 就能让整个回放崩在半路（**报告也写不出来**，因为崩在写报告之前）。
# 同时给 `errors="replace"`：万一还有编不出的字符，降成一个「?」也比中断强——
# 这是给人看的进度输出，不值得为它让实验跑不完。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import chat as chatmod
from core import config as cfgmod
from core import prompts as promptsmod
from core import settings
from core.chat import ChatSession, build_embedding
from core.distill import run_distill_cycle
from core.llm import LLM
from core.store import Store
from core.weave import merge_suggestions


def _cues_str(cues: dict) -> str:
    """七条线索压成一行（`C1=… C2=… …`）——报告里每一轮都会重复，所以要短。"""
    if not cues:
        return "—"
    c3 = cues.get("C3") or {}
    return (f"C1={cues.get('C1')} C2={cues.get('C2')} C3=v{c3.get('valence')}/a{c3.get('arousal')} "
            f"C4={cues.get('C4')} C5={cues.get('C5')} C6={cues.get('C6')} C7={cues.get('C7')}")


def run_script(script_path: str, db_path: str | None = None, fresh: bool = False,
               verbose: bool = True, prefs: str = "", persona_label: str = "") -> dict:
    """回放一份脚本，**返回报告原始数据**（`.json` 报告就是它的序列化）。

    每个会话一个 `ChatSession`（窗口也独立）、逐句 `reply`；全聊完跑一次提炼。
    `verbose` 只管终端——报告内容不受它影响（同一份数据永远长一样）。
    """
    cfgmod.ensure_dirs()
    settings.apply()

    script_file = Path(script_path)
    script = json.loads(script_file.read_text(encoding="utf-8"))
    stem = script_file.stem

    db = Path(db_path) if db_path else cfgmod.abspath(f"data/experiments/{stem}.db")
    db.parent.mkdir(parents=True, exist_ok=True)
    if fresh and db.exists():
        db.unlink()
        if verbose:
            print(f"[--fresh] 已删除实验库 {db.name}")
    # 窗口文件按**库名**分家：同一脚本跑多个变体（不同 --db）时窗口不能串——
    # V1 读到 V0 的窗口，测出来的就不是变体差异，是"先跑的那个"。
    # （默认库名就是 `<stem>.db`，所以老用法下 win_tag == stem，行为不变。）
    win_tag = db.stem

    store = Store(db)
    # 实验偏好：会话开跑前写入**实验库**（如 lang=en——测语言切换用）。
    # 写库而不是改全局：ChatSession 每轮从 store 读偏好，实验库一脏一净都是它自己的。
    for pair in (prefs or "").split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        store.set_pref(k.strip(), v.strip())
        if verbose:
            print(f"  偏好: {k.strip()} = {v.strip()}")
    llm = LLM()
    emb = build_embedding()
    # 她的名字（显示用）：`--persona` 的覆盖名优先，其次读实验库偏好（走真实机制），
    # 最后才回退 air。**不要把"air"写死**——人格对照实验里全标成 air，读者会以为跑的是 air。
    speaker = (persona_label or store.get_pref("persona") or "air")

    if verbose:
        print(f"实验: {script.get('name') or stem}")
        print(f"库  : {db}")
        print(f"模型: {llm.model or '(未配)'}   语义: {'on' if emb else '降级'}")
        print(f"人格: {speaker}")
        print("-" * 66)

    win_dir = cfgmod.abspath("data/experiments/windows")
    win_dir.mkdir(parents=True, exist_ok=True)

    sessions_out = []
    for si, sess in enumerate(script.get("sessions") or []):
        label = sess.get("label") or f"会话 {si + 1}"
        if verbose:
            print(f"\n══ {label} ══")
        chat = ChatSession(store, llm, emb, session_id=f"{win_tag}-s{si + 1}",
                           state_path=win_dir / f"{win_tag}-s{si + 1}.json")
        turns_out = []
        for turn in sess.get("turns") or []:
            r = chat.reply(turn)
            rec = r["recall"]
            turns_out.append({
                "user": turn,
                "reply": r["reply"],
                "cues": rec.get("cues_view") or {},
                "actions": rec.get("actions") or {},
                "recalled": [{"id": s.id, "title": s.title,
                              "why": (rec.get("why") or {}).get(s.id, "")}
                             for s in rec.get("scenes") or []],
                "suppressed": rec.get("suppressed") or [],
                # 常备备忘录（2026-10-05 晚并栏）：这一轮她手上拿到哪几件、
                # 哪一件是**到点**的（到点那件进注入即记账）。原 `memo_candidates`
                # 那一路删了——候选 / 主动开口都并进这一栏。
                "standing_memos": [x.get("id") for x in rec.get("standing_memos") or []],
                "memos_due": [x.get("id") for x in rec.get("standing_memos") or []
                              if x.get("due")],
                # `Raw` 没有 id——用 scene_id（原文按天存文档，没有独立编号）
                "raws": [x.scene_id for x in rec.get("raws") or []],
                "written": r["written"],
            })
            if verbose:
                print(f"  用户: {turn}")
                print(f"  {speaker} : {r['reply'][:110]}")
                if turns_out[-1]["recalled"]:
                    print(f"   ↳ 召回: " + "；".join(
                        f"{x['id']}「{x['title']}」{x['why']}"
                        for x in turns_out[-1]["recalled"]))
                if r["written"]:
                    print(f"   ↳ 提取: {r['written']['scene_id']}「{r['written']['title']}」")
        tail = chat.close()
        if verbose and tail:
            print(f"  ↳ 收尾提取: {tail['scene_id']}「{tail['title']}」"
                  f"  主题={tail['topic']}")
        sessions_out.append({"label": label, "turns": turns_out, "tail": tail})

    # 全部聊完 → 跑一次后台提炼（聚合 → 抽象 → 印证 → 老化 → 漂移）
    if verbose:
        print("\n" + "-" * 66)
        print("后台提炼周期…")
    cycle = run_distill_cycle(store, llm, emb=emb)
    if verbose:
        print(f"  S2 聚合: {cycle['s2_new']}   S3 抽象: {cycle['s3_new']}")
        print(f"  收敛为已立: {cycle['established']}   老化: {cycle['aged']}")
        # 没收上去的画像**要说得出为什么**——不然「优化」只能靠反复重跑碰运气
        for b in cycle.get("blocked") or []:
            print(f"  ⚠️ 未收敛 {b['id']}「{b['topic']}」: {b['reason']}")
        print(f"  漂移: {cycle['drift'].get('drifting') and len(cycle['drift']['drifting']) or 0} 个主题")

    # 主题归并建议：**只建议，不自动合**——合并是不可逆的语义断言
    # （「这两件事是同一件」），而相似度只说明"看起来像"。裂缝要补，
    # 但得由人点这一下（仪表盘「主题」页）。
    topic_merge = merge_suggestions(store, emb)
    if verbose and topic_merge:
        print(f"  主题归并建议 {len(topic_merge)} 对（可在仪表盘「主题」页合并）:")
        for x in topic_merge[:5]:
            print(f"    「{x['from']}」→「{x['to']}」（{x['basis']} {x['similarity']}）")

    report = {
        "name": script.get("name") or stem,
        "note": script.get("note") or "",
        "script": str(script_file),
        "db": str(db),
        "model": llm.model,
        "persona": speaker,
        "vector": bool(emb),
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": sessions_out,
        "cycle": {k: v for k, v in cycle.items() if k != "drift"},
        "drift": cycle.get("drift") or {},
        "topic_merge": topic_merge,
        "final": _snapshot(store),
    }
    store.close()
    return report


def _snapshot(store: Store) -> dict:
    """实验结束时把库里所有东西摊平成纯数据（给报告和 `.json` 用）。

    「这一轮之后库里变成什么样了」才是改一个阈值之后真正值得对比的东西。
    全部是原生类型——这份要能直接 `json.dump`。
    """
    return {
        "scenes": [{"id": s.id, "title": s.title, "topic": s.topic, "subject": s.subject,
                    "summary": s.text, "valence": s.valence, "arousal": s.arousal,
                    "intensity": round(s.intensity or 0, 3),
                    "trigger_class": s.trigger_class, "sensitive": bool(s.sensitive),
                    "entities": store.entities_of_scene(s.id),
                    "has_vec": s.emb is not None,
                    "mention": s.mention_count, "cited": s.cited_by_profile}
                   for s in store.query_scenes(include_archived=True)],
        "summaries": [{"id": x.id, "topic": x.topic, "text": x.text, "sources": x.sources}
                      for x in store.hot_summaries(50)],
        "profiles": [{"id": p.id, "topic": p.topic, "statement": p.statement,
                      "status": p.status, "evidence": p.evidence,
                      "sources": p.sources, "invalidated_at": p.invalidated_at,
                      "invalidated_by": p.invalidated_by}
                     for p in store.current_profiles(status=None)],
        "memos": [{"id": m.id, "content": m.content, "kind": m.kind,
                   "status": m.status, "kind_class": m.kind_class,
                   "due_at": m.due_at, "raise_mode": int(m.sensitive or 0)}
                  for m in store.open_memos()],
        # 原文已经不在库里（`raws` 表随 S0 改成按天文档一起废了），
        # `store.count("raws")` 会直接抛「未知表名」——这里按文档份数报。
        # ⚠️ `raws` 表随 S0 改成按天文档废了、`edges` 表随四类边改派生退役（2026-09-24）
        # ——`store.count()` 对不存在的表会直接抛，这里按**现存**的表报。
        "counts": {t: store.count(t) for t in
                   ("scenes", "summaries", "profiles", "memos", "entities")},
        "raw_days": store.raw_doc_days(),
    }


def render_markdown(rep: dict) -> str:
    """把报告数据渲染成人读的 Markdown。**只读 dict、不碰 store**——
    同一份 JSON 可以反复渲染（这是 `--no-report` 之后还能补写报告的前提）。

    每轮都写「线索 → 动作 → 召回 → 提取」：改一个阈值带来了什么变化，
    只有看见中间状态才回答得了。
    """
    L: list[str] = []
    # 她的名字：报告里**不写死 "air"**（人格对照实验会误导）——与回复标签同一个来源。
    who = rep.get("persona") or "air"
    L.append(f"# 对话实验：{rep['name']}\n")
    if rep.get("note"):
        L.append(f"> {rep['note']}\n")
    L.append(f"- 脚本：`{rep['script']}`")
    L.append(f"- 库：`{rep['db']}`")
    L.append(f"- 模型：`{rep['model']}`　语义：{'已启用' if rep['vector'] else '降级（字符重叠）'}")
    L.append(f"- 人格：**{who}**")
    if rep.get("variant"):
        L.append(f"- 变体：**{rep['variant']}**")
    L.append(f"- 时间：{rep['started']}\n")

    for sess in rep["sessions"]:
        L.append(f"\n---\n\n## {sess['label']}\n")
        for t in sess["turns"]:
            L.append(f"**用户**：{t['user']}\n")
            L.append(f"**{who}**：{t['reply']}\n")
            debug = [f"线索 `{_cues_str(t['cues'])}`",
                     f"动作 `{json.dumps(t['actions'], ensure_ascii=False)}`"]
            if t["recalled"]:
                debug.append("召回 " + "；".join(
                    f"`{x['id']}`「{x['title']}」*{x['why']}*" for x in t["recalled"]))
            else:
                debug.append("召回 （无）")
            if t["suppressed"]:
                debug.append(f"抑制 `{' '.join(t['suppressed'])}`")
            if t["standing_memos"]:
                due = "、到点：" + " ".join(t["memos_due"]) if t["memos_due"] else ""
                debug.append(f"手上有 `{' '.join(t['standing_memos'])}`{due}")
            if t["raws"]:
                debug.append(f"下钻原文 `{' '.join(t['raws'])}`")
            L.append("> " + "　\n> ".join(debug) + "\n")
            if t["written"]:
                w = t["written"]
                L.append(f"**提取** → `{w['scene_id']}`「{w['title']}」"
                         f"　主题={w['topic']}　强度={w['intensity']}"
                         f"　效价={w['valence']}　唤醒={w['arousal']}\n")
        if sess.get("tail"):
            w = sess["tail"]
            L.append(f"**收尾提取** → `{w['scene_id']}`「{w['title']}」\n")

    fin = rep["final"]
    L.append("\n---\n\n## 跑完之后的库\n")
    L.append("| 场景 | 主题 | 摘要 | 情绪 | 情境 | 实体 | 向量 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in fin["scenes"]:
        L.append(f"| `{s['id']}` {s['title']} | {s['topic']} | {s['summary']} "
                 f"| v{s['valence']}/a{s['arousal']} | {s['trigger_class'] or '—'} "
                 f"| {','.join(s['entities']) or '—'} | {'✓' if s['has_vec'] else '✗'} |")

    if fin["summaries"]:
        L.append("\n## 主题摘要（S2）\n")
        for x in fin["summaries"]:
            L.append(f"- `{x['id']}`（{x['topic']}）：{x['text']}\n  原料 {x['sources']}")

    L.append("\n## 画像（S3）\n")
    if not fin["profiles"]:
        L.append("（还没形成——画像要 ≥3 次同情境印证才立）")
    for p in fin["profiles"]:
        mark = ("已失效" if p["invalidated_at"] else
                ("已立" if p["status"] == "established" else "待验证（不进常驻）"))
        L.append(f"- `{p['id']}` **{p['statement']}**　[{mark}]　印证 {p['evidence']} 次")
        L.append(f"  依据 {p['sources']}")

    L.append("\n## 备忘录\n")
    if not fin["memos"]:
        L.append("（无）")
    for m in fin["memos"]:
        tag = "【不提·只记】" if m.get("raise_mode", 0) >= 2 else ""
        L.append(f"- `{m['id']}` {m['content']}　[{'air 的承诺' if m['kind'] == 'air_promise' else '用户的事'}]"
                 f"　状态={m['status']}　时间依据={m['due_at'] or m['kind_class'] or '—'} {tag}")

    c = fin["counts"]
    L.append(f"\n## 总量\n")
    L.append(f"场景 {c['scenes']} · 原文 {fin.get('raw_days', 0)} 天 · 摘要 {c['summaries']} · "
             f"画像 {c['profiles']} · 备忘录 {c['memos']} · 实体 {c['entities']}")

    cy = rep.get("cycle") or {}
    L.append(f"\n## 提炼周期\n")
    L.append(f"- S2 新增 `{cy.get('s2_new')}`　S3 新增 `{cy.get('s3_new')}`")
    L.append(f"- 收敛为已立 `{cy.get('established')}`　老化降级 `{cy.get('aged')}`")
    blocked = cy.get("blocked") or []
    if blocked:
        L.append("- **未收敛**（画像为什么没立——卡在哪一关说在这里）：")
        for b in blocked:
            L.append(f"  - `{b.get('id')}`「{b.get('topic')}」：{b.get('reason')}")
    else:
        L.append("- 未收敛：（无）")

    tm = rep.get("topic_merge") or []
    if tm:
        L.append(f"\n## 主题归并建议\n")
        L.append("> 疑似「同一个主题被写成两种说法」。**系统不会自动合并**——"
                 "「这两件事是同一件」不该由代码替你断言。"
                 "在仪表盘的「主题」页点一次即可（只改归属、不删记录、会留痕）。\n")
        for x in tm:
            L.append(f"- 「{x['from']}」→「{x['to']}」"
                     f"（{x['basis']} {x['similarity']}；{x['from_n']} 条 → {x['to_n']} 条）")
    drift = rep.get("drift") or {}
    if drift.get("drifting"):
        L.append(f"- 漂移检测：{len(drift['drifting'])} 个主题超阈值")
        for d in drift["drifting"]:
            L.append(f"  - {d.get('topic')}：{d.get('reason')} → **{d.get('verdict')}**")

    L.append("\n% 每次唤醒都另存了 trace（`data/trace/*.jsonl`），"
             "想看「为什么召回了这个」可以去那儿逐行查。")
    return "\n".join(L)


# 变体开关（A/B 实验）：把某个常驻块**置空**再回放。
# 为什么置空模块常量、而不是穿透一个参数：`build_system_prompt` 读的就是
# 这些模块级名字——置空 = 那条规则从未存在过。这才测得出「删掉它会怎样」；
# 「要求模型忽略它」测的是另一件事（指令服从）。
_DROPABLE = {
    "progress": ("PROGRESS_DISCIPLINE", "推进纪律"),
    "memory": ("MEMORY_DISCIPLINE", "记忆纪律"),
    "weave": ("WEAVE_DISCIPLINE", "编织分寸"),
}


def apply_drop(name: str) -> str:
    """置空指定常驻块；返回置空的块名（空串 = 不置空）。名字不认识就报错退出。"""
    key = (name or "").strip().lower()
    if not key:
        return ""
    if key not in _DROPABLE:
        raise SystemExit(f"--drop 只认：{' / '.join(_DROPABLE)}（收到「{name}」）")
    attr, label = _DROPABLE[key]
    setattr(promptsmod, attr, "")
    print(f"[--drop] 已置空：{label}（{attr} = \"\"）——变体实验")
    return key


def apply_inject(path: str) -> str:
    """把文件内容放进实验槽位（`prompts.EXTRA_DISCIPLINE`）；返回文件名（空串 = 没注入）。

    槽位默认是空的、**不在固定块预算里**——候选块先在这里过 A/B，证明有效再谈转正。
    """
    if not (path or "").strip():
        return ""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--inject 的文件不存在：{path}")
    promptsmod.EXTRA_DISCIPLINE = p.read_text(encoding="utf-8").strip()
    print(f"[--inject] 实验槽位 <- {p.name}（{len(promptsmod.EXTRA_DISCIPLINE)} 字符）")
    return p.stem


def apply_persona(path: str) -> str:
    """用文件内容**整体替换** air 的人格层（patch `chat.load_persona`——纪律时代
    它替换的是宪章 + 纪律，现在替换的就是人格本身），返回 persona 名。

    测的是「换灵魂」：同样的输入，不同的人格提示词会怎么接。替换面刻意放大——
    要看的就是"不是 air 的别的东西"；工具分寸 / 记忆 / 时间这些**环境件**保留
    （那是场地，不是人格——mia 和 xina 本来也没有这套场地）。
    """
    if not (path or "").strip():
        return ""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--persona 的文件不存在：{path}")
    text = p.read_text(encoding="utf-8").strip()
    chatmod.load_persona = lambda name="": text   # 对话每轮读的就是这个名字
    print(f"[--persona] 人格已被替换为 {p.name}（{len(text)} 字符）")
    return p.stem


def main() -> int:
    """命令行入口：跑脚本 → 写两份报告（`.md` 给人、`.json` 给机器）。"""
    ap = argparse.ArgumentParser(description="air-link-01 对话实验（脚本回放）")
    # 没有默认脚本：**仓库不带实验素材**（撤进 `private/`，不进版本库）——
    # 给一个不存在的默认路径，只会让人以为"跑起来但坏了"。
    ap.add_argument("--script", default="",
                    help="实验脚本（JSON，格式见本文件头部；仓库不自带）")
    ap.add_argument("--db", default=None, help="指定实验库（默认 data/experiments/<脚本名>.db）")
    ap.add_argument("--fresh", action="store_true", help="先清空实验库（**只删实验库**）")
    ap.add_argument("--drop", default="",
                    help="A/B 变体：置空一个常驻块再回放（progress / memory / weave）。"
                         "变体务必配 --db 或 --fresh——别和基线共用一个库")
    ap.add_argument("--prefs", default="",
                    help="开跑前写入实验库的偏好，如 \"lang=en\"")
    ap.add_argument("--inject", default="",
                    help="把文件内容放进实验槽位（prompts.EXTRA_DISCIPLINE）——测候选块用")
    ap.add_argument("--persona", default="",
                    help="用文件整体替换人格（换人格对照实验）")
    ap.add_argument("--no-report", action="store_true", help="不写报告文件")
    a = ap.parse_args()
    if not a.script:
        print("要一个实验脚本：--script <你的脚本.json>\n"
              "（仓库不带实验素材——脚本是私人的，放在 private/实验/ 下）")
        return 2

    dropped = apply_drop(a.drop)
    injected = apply_inject(a.inject)
    persona = apply_persona(a.persona)
    rep = run_script(a.script, db_path=a.db, fresh=a.fresh, prefs=a.prefs,
                     persona_label=persona)
    variant_bits = []
    if dropped:
        variant_bits.append(f"去 {dropped}")
    if a.prefs:
        variant_bits.append(f"偏好 {a.prefs}")
    if injected:
        variant_bits.append(f"注入 {injected}")
    if persona:
        variant_bits.append(f"persona {persona}")
    rep["variant"] = "；".join(variant_bits)

    if not a.no_report:
        stem = Path(a.script).stem
        out_dir = cfgmod.abspath(cfgmod.PATHS["experiments"]) / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name_bits = [b for b in (dropped,
                                 "prefs" if a.prefs else "",
                                 Path(a.inject).stem if a.inject else "",
                                 persona) if b]
        suffix = ("-" + "-".join(name_bits)) if name_bits else ""
        md = out_dir / f"{stem}{suffix}-{ts}.md"
        md.write_text(render_markdown(rep), encoding="utf-8")
        (out_dir / f"{stem}{suffix}-{ts}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告: {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
