"""阶段 1 演示：从「对话」走到「记住」再走到「想起来」。

跑法：
    python demo.py            # 用内置剧本（假模型），不需要任何 API key
    python demo.py --real     # 用真实 LLM / embedding（先配环境变量，见 README）
    python demo.py --fresh    # 先清空演示库（**显式**清空，不是静默覆盖）

为什么默认不需要 API key：一条链路如果只有配好服务才能跑，
那它实际上从没被真正跑通过。假模型跑通的是**同一条代码路径**
（`LLM.structured` 只是返回值换了来源），不是另写一条。
这样任何人在任何机器上 clone 下来都能先看见它动起来。

演示的库是 `data/demo.db`（与真实记忆库分开）——演示数据不该混进真记忆。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认不是 UTF-8，中文会乱码。显式切一下编码。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core import settings
from core.embedding import EmbeddingService
from core.llm import LLM
from core.recall import recall_for_message
from core.shortterm import ShortTerm
from core.store import Store

LINE = "=" * 66

# 演示剧本：三段时间上连续的对话 + 两次唤醒。
# 每段都设计了一个不同的机制，跑完就能看见它们各自在干什么：
#   段 1 —— 情绪场景（valence/arousal/trigger_class/强度）
#   段 2 —— air 的承诺 → open_loops → memo（air 的一致性靠它管）
#   段 3 —— 实体（宠物）→ 实体索引 → 旁路召回
SCRIPT = [
    {
        "label": "段 1｜情绪场景（前因后果 + 情绪字段 + 强度）",
        "turns": [
            ("user", "今天又被组长当众说了一顿"),
            ("air", "听着挺难受的"),
            ("user", "算了，我不想争了，干完这段就想走"),
        ],
        "card": {
            "title": "被组长当众批评后想离开", "keywords": ["组长", "当众批评", "想走"],
            "summary": "被当众批评后产生逃避念头",
            "window_digest": ("用户说被组长当众批评，air 表示理解；用户说不想争了、"
                              "打算干完这段就走。情绪负面、唤醒较高，出现逃避倾向。"),
            "valence": -1, "arousal": 1, "trigger": "被组长当众批评", "trigger_class": "被评价",
            "reaction": "不想争、想离开", "outcome": "", "subject": "user",
            "topic": "用户·面对工作压力的反应", "self_ref": False, "air_stance": "",
            "sensitive": False, "entities": [{"name": "组长", "kind": "person",
                                              "relation": "组长"}],
            "open_loops": [],
        },
    },
    {
        "label": "段 2｜air 的承诺（open_loops → memo）",
        "turns": [
            ("user", "对了，我下周三要面试，有点紧张"),
            ("air", "记下了，到那天我会提醒你"),
        ],
        "card": {
            "title": "下周三面试", "keywords": ["面试", "紧张", "下周三"],
            "summary": "下周三有面试，有点紧张",
            "window_digest": "用户提到下周三要面试、有点紧张；air 承诺到那天会提醒他。",
            "valence": -1, "arousal": 1, "trigger": "下周三面试", "trigger_class": "悬而未决",
            "reaction": "紧张", "outcome": "", "subject": "user",
            "topic": "用户·面试这件事", "self_ref": False,
            "air_stance": "承诺在面试当天提醒用户",
            "sensitive": False, "entities": [],
            "open_loops": [
                {"content": "下周三面试", "kind": "user_task", "due_at": "2026-09-16"},
                {"content": "面试当天提醒用户", "kind": "air_promise", "due_at": "2026-09-16"},
            ],
        },
    },
    {
        "label": "段 3｜实体（宠物）→ 实体索引",
        "turns": [
            ("user", "我家咪咪这两天也不吃饭，有点担心"),
            ("air", "猫不吃东西要留意，超过两天就得看看了"),
        ],
        "card": {
            "title": "咪咪不吃饭", "keywords": ["咪咪", "猫", "不吃饭"],
            "summary": "家里的猫两天没吃饭，用户担心",
            "window_digest": "用户说家里的猫咪咪两天不吃饭、有点担心；air 提醒超过两天要就医。",
            "valence": -1, "arousal": 0, "trigger": "咪咪不吃饭", "trigger_class": "悬而未决",
            "reaction": "担心", "outcome": "", "subject": "world",
            "topic": "咪咪·健康问题", "self_ref": False,
            "air_stance": "建议超过两天不进食就要就医",
            "sensitive": False, "entities": [{"name": "咪咪", "kind": "pet",
                                              "relation": "家里的猫"}],
            "open_loops": [],
        },
    },
]

# 唤醒用例：分别打 R2/R1（时间回溯）和实体旁路（不经语义检索）。
QUERIES = [
    ("上次那个面试的事怎么样了", {"tense": "past", "valence": -1, "arousal": 1,
                                 "about_relation": False, "unresolved": True}),
    ("咪咪最近吃东西了吗", {"tense": "now", "valence": -1, "arousal": 1,
                            "about_relation": False, "unresolved": False}),
    ("TCP 三次握手是什么", {"tense": "now", "valence": 0, "arousal": 0,
                            "about_relation": False, "unresolved": False}),
]


def make_mock_llm() -> LLM:
    """内置剧本模型：场景卡按段顺序取；线索判断按查询用 keyword 匹配。

    这就是「测试与 demo 跑在同一条代码路径」的意思——`LLM.structured()`
    的实现一行没变，只是返回值的来源从 HTTP 换成了字典。
    """
    cards = [dict(seg["card"]) for seg in SCRIPT]
    expect = [seg["card"]["summary"] for seg in SCRIPT]

    def fn(prompt: str, schema: dict) -> dict:
        name = schema.get("name")
        if name == "scene_card":
            return cards.pop(0) if cards else dict(SCRIPT[-1]["card"])
        if name == "cue_judgement":
            for q, cues in QUERIES:
                if q in prompt:
                    return dict(cues)
            return {"tense": "now", "valence": None, "arousal": None,
                    "about_relation": False, "unresolved": False}
        return {}

    llm = LLM()
    llm.set_mock(fn)
    llm._demo_expect = expect          # 仅供 demo 展示，不参与逻辑
    return llm


def build_embedding() -> EmbeddingService | None:
    """按配置建向量服务；没配就返回 None（上层据此走字符重叠降级）。"""
    conf = cfgmod.cfg("embedding", default={}) or {}
    svc = EmbeddingService(endpoint=conf.get("endpoint", ""),
                           api_key=conf.get("api_key", ""),
                           model=conf.get("model", "text-embedding-3-small"),
                           insecure_ssl=bool(conf.get("insecure_ssl", False)))
    return svc if svc.available else None


def show_write(seg: dict, info: dict, store) -> None:
    """打印一次提取的结果：场景卡的字段 + 顺带发生的副产物（原文 / 实体 / memo）。
    提取不只是「存了一句话」——看不见的那几件等于不知道。"""
    print(f"\n  ▸ 触发提取（会话结束——真实运行中，话题切换同样会触发）")
    print(f"    ✔ 场景 {info['scene_id']}  「{info['title']}」")
    print(f"       主题={info['topic']}   效价={info['valence']} 唤醒={info['arousal']} "
          f"强度={info['intensity']}  情境={info['trigger_class'] or '(无)'}")
    print(f"       实体={info['entities'] or '(无)'}   未闭合线索={info['open_loops']} 条")
    print(f"       摘要（供检索）= {seg['card']['summary']}")
    digest = seg["card"]["window_digest"]
    print(f"       窗口摘要（供短期窗口，与被压掉的逐字等长）= {digest[:42]}…")

    memos = store.memos_by_scene(info["scene_id"])
    for m in memos:
        when = m.due_at or "(无具体时间)"
        who = "air 的承诺" if m.kind == "air_promise" else "用户的事"
        print(f"       📌 memo[{who}] {m.content}  时间={when}  状态={m.status}")


def show_recall(query: str, result: dict, store, records: list) -> None:
    """打印一次唤醒的完整推理链：线索 → 动作 → 召回（含 why）→ 两个 flag。

    `records` 是 trace 记录（每轮那行）——**要显示真正跑过的那一次**，
    重算会让画面说谎。
    """
    rec = records[-1]
    cues = rec["cues"]
    print(f"\n  ▸ 输入: {query}")
    print(f"    线索: C1={cues['C1']}  C2={cues['C2']}  C3={cues['C3']}  "
          f"C4={cues['C4']}  C5={cues['C5']}  C6={cues['C6']}  C7={cues['C7']}")
    print(f"          实体命中={cues.get('entities') or '(无)'}")
    print(f"    动作: {rec['actions']}")
    if not result["scenes"]:
        print("    召回: （无）—— 只注入常驻画像")
    recall_info = {r["id"]: r for r in rec.get("recalled", [])}
    for s in result["scenes"]:
        info = recall_info.get(s.id, {})
        print(f"    召回: {s.id} 「{s.title}」  core={info.get('core')}  "
              f"why={info.get('why', '')}")
    if result.get("raws"):
        # `Raw` 没有 id——它的编号就是 scene_id（原文没有独立索引，见 model.Raw）
        print(f"    R5 下钻原文: {[r.scene_id for r in result['raws']]}（多维协同命中才给）")
    flags = result.get("flags", {})
    if flags.get("hint_talked_before"):
        print("    提示: 「你们聊过这个」（C6 新鲜度低）")


def main() -> int:
    """按剧本跑三段对话 + 三次唤醒。**返回退出码**（0 = 跑完，非 0 = 中断）。"""
    parser = argparse.ArgumentParser(description="air-link-01 记忆系统 · 阶段 1 演示")
    parser.add_argument("--real", action="store_true",
                        help="用真实 LLM / embedding（需先配环境变量）")
    parser.add_argument("--fresh", action="store_true",
                        help="先清空演示库与窗口状态再跑（显式操作，不会静默覆盖；"
                             "trace 不删——留痕是审计数据）")
    args = parser.parse_args()

    # 外部配置（config.local.json / 环境变量）**必须先合进 CONFIG**：
    # 否则 `--real` 下的 LLM、以及 embedding 读到的还是 config.py 的空默认值，
    # 配了也等于没配，而且不会报错——只是安静地全程走降级。
    settings.apply()
    cfgmod.ensure_dirs()
    demo_db = cfgmod.abspath("data/demo.db")
    demo_window = cfgmod.abspath("data/demo_shortterm.json")
    if args.fresh:
        for p in (demo_db, demo_window):
            if p.exists():
                p.unlink()
                print(f"[--fresh] 已删除 {p.name}")

    store = Store(demo_db)
    llm = LLM() if args.real else make_mock_llm()
    emb = build_embedding()

    print(LINE)
    print(" air-link-01 记忆系统 · 阶段 1 演示")
    print(LINE)
    print(f" 库文件      : {demo_db}")
    print(f" 已有场景    : {store.count('scenes')} 条")
    print(f" 模式        : {'真实 LLM' if args.real else '内置剧本（假模型，不需要 key）'}")
    print(f" 向量服务    : {'已配置' if emb else '未配置（C1/C4 走字符重叠兜底）'}")
    print(f" 人格        : {cfgmod.PATHS['personas']}"
          f"（按他选的注入；记忆层只负责放进去，不负责执行它）")

    st = ShortTerm(store, llm, emb_service=emb, session_id="demo",
                   state_path=demo_window)

    print("\n" + LINE)
    print(" 一、写入：对话 → 场景卡（短期窗口的「压缩 = 提取」）")
    print(LINE)
    for seg in SCRIPT:
        print(f"\n—— {seg['label']} ——")
        for speaker, text in seg["turns"]:
            who = "用户" if speaker == "user" else "air "
            print(f"   {who}: {text}")
        for speaker, text in seg["turns"]:
            st.append(speaker, text)
        st.end_session()
        info = st.flush_if_needed()
        if info:
            show_write(seg, info, store)

    print("\n" + LINE)
    print(" 二、库内状态")
    print(LINE)
    for s in store.query_scenes(include_archived=True):
        print(f"   {s.id}  [{s.subject:5s}] {s.topic:24s} {s.title}")
    print(f"   实体索引: {[e.name for e in store.all_entities()]}")
    print(f"   memo    : {store.count('memos')} 条（含 air 自己的承诺）")
    # S0 按天存文档（`data/raws/`），**不在库里**——这里报的是文档份数（天数）。
    # `count("raws")` 是迁移前那张表的写法：那张表已改名 `raws_migrated`，
    # 这么查会直接抛「未知表名」（同 demo_distill.py 的说明）。
    print(f"   原文 S0 : {store.raw_doc_days()} 份（按天文档，**无索引，只能经 S1 下钻**）")

    print("\n" + LINE)
    print(" 三、唤醒：一句话 → 七条线索 → 动作 → 抑制取前 N → 注入")
    print(LINE)
    records: list = []

    # 包一层：recall_for_message 自己会写 trace，这里顺手把 trace 内容捞回来展示
    from core import recall as recall_mod
    orig_trace = recall_mod.write_trace

    def capture(record):
        records.append(record)
        return orig_trace(record)

    recall_mod.write_trace = capture
    try:
        for query, _ in QUERIES:
            result = recall_for_message(query, store, llm=llm, emb=emb)
            show_recall(query, result, store, records)
    finally:
        recall_mod.write_trace = orig_trace

    print("\n" + LINE)
    print(" 四、可观测：每次唤醒都留痕（记忆系统最难的是「为什么召回了这个」不可见）")
    print(LINE)
    trace_dir = cfgmod.abspath(cfgmod.PATHS["trace_dir"])
    for f in sorted(trace_dir.glob("唤醒-*.jsonl")):
        print(f"   {f}  （{len(f.read_text(encoding='utf-8').strip().splitlines())} 行）")
    print("   每行包含：输入 / 七条线索 / 动作权重 / 召回(含 why 与核心度) / 抑制名单")

    print("\n" + LINE)
    print(" 阶段 1 完成。下一步（阶段 2）：提炼 step2/step3 —— 同主题聚合 S2")
    print(" 与跨主题抽象 S3（印证 ≥3 且同情境才升 established）。")
    print(LINE)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
