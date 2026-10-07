"""阶段 4 演示：渐变检测（人慢慢变了，但没有哪一条算得上反例）。

跑法：
    python demo_trend.py            # 内置剧本，不需要 key
    python demo_trend.py --fresh    # 先清空演示库
    run.cmd trend                   # 等价于第一种

这个脚本想让人看见的核心是**「数字漂了 ≠ 人变了」**：

  两个 topic 的漂移指标几乎一样（效价都移动了 1.0），但结论相反——
  一个确实变了，一个只是最近聊的事不一样。

所以流程必须是两段：**算数提出疑问 → 判断回答疑问**。
数一变就改画像的话，画像会随噪声来回翻。

（这也是「不滞留」的最后一块：突变归 `revise_profile`、
冷掉归 `age_out_profiles`，中间那条「一直在提但内容在慢慢变」的缝归这里。）
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core.llm import LLM
from core.model import Profile, Scene
from core.store import Store
from core.trend import detect_drift, topic_drift

LINE = "=" * 74
TOPIC_A = "用户·面对被评价的反应"
TOPIC_B = "用户·对咖啡的偏好"


def vec(deg: float) -> list[float]:
    """一个二维单位向量（按角度给，余弦一眼能估）。

    漂移看的是"中心方向转了多少度"——二维足够表达，而且数值直接看得懂。
    """
    r = math.radians(deg)
    return [math.cos(r), math.sin(r)]


# 每个 topic：早期 3 条 + 近期 3 条。
# 近期那批的**情绪与语义都移动了**，但两个 topic 的真相不同。
CASES = [
    {
        "topic": TOPIC_A,
        "statement": "遇到被评价的情境时，他会先退出、不再争取",
        "early": [(vec(0), -1, 1, "被组长当众批评", "不想争，算了"),
                  (vec(6), -1, 1, "方案被主管否掉", "改了三版，没说话"),
                  (vec(12), -1, 1, "被同事当面质疑", "回去自己查")],
        "recent": [(vec(55), 0, 0, "又被批评", "当场把话说清楚了"),
                   (vec(60), 0, 0, "方案又被否", "直接找主管对了需求"),
                   (vec(65), 0, 0, "被质疑", "当场回应，没有回避")],
        "truth": "人真的变了：从退出到正面回应",
    },
    {
        "topic": TOPIC_B,
        "statement": "他很喜欢咖啡，会花时间在这件事上",
        "early": [(vec(180), 1, 0, "买了台手冲设备", "研究水温粉水比"),
                  (vec(186), 1, 0, "研究豆子产地", "做笔记"),
                  (vec(192), 1, 0, "每天早上磨豆子", "当成仪式")],
        "recent": [(vec(200), 0, 0, "最近忙，改喝速溶", "对付一口"),
                   (vec(205), 0, 0, "办公室只有挂耳", "不挑了"),
                   (vec(210), 0, 0, "随便喝喝", "没时间弄")],
        "truth": "只是最近忙，不是不喜欢了——**聊的事不一样，人没变**",
    },
]


def make_mock_llm() -> LLM:
    """剧本模型：按 prompt 里的 topic 给不同结论。

    这正是演示的重点——同一组数字，**结论可以不同**，区别只在把两段材料
    放在一起看之后，判断「如果早期那几件事发生在今天，他还会那样反应吗」。
    """
    def fn(prompt, schema):
        if schema.get("name") == "profile_revision":
            if "咖啡" in prompt:
                return {"verdict": "holds", "statement": ""}
            return {"verdict": "revise",
                    "statement": "遇到被评价时，他会先扛一下再当场回应"}
        return {}

    llm = LLM()
    llm.set_mock(fn)
    return llm


def build(store: Store) -> None:
    """造两个 topic：**漂移指标几乎一样，结论却相反**——这是演示的全部要点。
    只造一个"确实变了"的例子，就说明不了为什么必须"算数提疑问 → 判断答疑问"。
    """
    for case in CASES:
        for i, (emb, val, aro, title, reaction) in enumerate(case["early"]):
            store.add_scene(Scene(topic=case["topic"], subject="user", title=title,
                                  text=reaction, trigger=title, trigger_class="被评价",
                                  reaction=reaction, valence=val, arousal=aro, emb=emb,
                                  time_event=f"2026-02-{10 + i:02d} 09:00:00"))
        for i, (emb, val, aro, title, reaction) in enumerate(case["recent"]):
            store.add_scene(Scene(topic=case["topic"], subject="user", title=title,
                                  text=reaction, trigger=title, trigger_class="被评价",
                                  reaction=reaction, valence=val, arousal=aro, emb=emb,
                                  time_event=f"2026-08-{10 + i:02d} 09:00:00"))
        store.add_profile(Profile(
            topic=case["topic"], subject="user", statement=case["statement"],
            status="established", evidence=3,
            sources=[s.id for s in store.query_scenes(topic=case["topic"])]))


def main() -> int:
    parser = argparse.ArgumentParser(description="air-link-01 · 阶段 4 演示（渐变检测）")
    parser.add_argument("--fresh", action="store_true", help="先清空演示库再跑")
    args = parser.parse_args()

    cfgmod.ensure_dirs()
    demo_db = cfgmod.abspath("data/demo_trend.db")
    if args.fresh and demo_db.exists():
        demo_db.unlink()
        print(f"[--fresh] 已删除 {demo_db.name}")

    store = Store(demo_db)
    llm = make_mock_llm()
    build(store)

    print(LINE)
    print(" air-link-01 · 阶段 4 演示：渐变检测（没有反例的那种变化）")
    print(LINE)
    print(f" 库文件: {demo_db}")
    print("\n 要处理的问题（逻辑层第 ⑤ 条）：")
    print("   「矛盾证据触发降级」只处理**突变**——出现明确反例。")
    print("   处理不了**渐变**：人慢慢变了，但没有任何一条可指认的反例事件。")

    print("\n" + LINE)
    print(" 一、情况：每个 topic 早期 3 条、近期 3 条")
    print(LINE)
    for case in CASES:
        print(f"\n   【{case['topic']}】")
        for label, rows in (("早期", case["early"]), ("近期", case["recent"])):
            for emb, val, aro, title, reaction in rows:
                print(f"     {label}: {title}  （效价 {val:+d} / 唤醒 {aro}）"
                      f"  反应：{reaction}")
    print("\n   ⚠️ 注意：**没有任何一条是反例**。"
          "「当场回应」不算反例——它只是这一次不太一样。")
    print("      单看每一条，一次 revise 都不会被触发。这正是渐变难被发现的原因。")

    print("\n" + LINE)
    print(" 二、算数：只给指标，不下结论")
    print(LINE)
    for case in CASES:
        d = topic_drift(store, case["topic"])
        vd = "（无向量服务）" if d["vector_drift"] is None else d["vector_drift"]
        print(f"\n   【{case['topic']}】{d['n']} 条场景")
        print(f"     语义中心漂移: {vd}")
        print(f"     效价: {d['valence_early']} → {d['valence_recent']}"
              f"（移动 {d['valence_shift']}）")
        print(f"     唤醒度: {d['arousal_early']} → {d['arousal_recent']}"
              f"（移动 {d['arousal_shift']}）")
        flag = "⚠️ 值得再看一眼" if d["drifting"] else "稳定"
        print(f"     判定: {flag}"
              + (f"  —— {d['reason']}" if d["reason"] else ""))

    print("\n   两条都触发了疑问，但**原因完全不同**：")
    print("     被评价那条——语义和情绪**都动了**（真的换了反应方式）")
    print("     咖啡那条——**只有情绪动了**（语气变淡），语义几乎没动")
    print("\n   而真相是：前者人变了，后者只是最近忙。")
    print("   **数字只能提出疑问，回答不了「是人变了，还是最近聊的不一样」。**")
    print("   这就是为什么必须再过一遍判断——数一变就改画像，画像会随噪声来回翻。")
    print("\n   顺便：咖啡那条**没有向量也能被发现**——情绪一半存在库里，")
    print("   不依赖 embedding 服务。降级时少的只是「语义」那只眼睛，不是全瞎。")

    print("\n" + LINE)
    print(" 三、判断：把早期和近期放在一起看")
    print(LINE)
    print("   问的是一个很具体的问题：")
    print("     「如果早期那几件事发生在**今天**，他还会那样反应吗？」")
    print("     会   → 只是情境不同（holds）")
    print("     不会 → 确实变了（revise）")

    stats = detect_drift(store, llm)
    for entry in stats["drifting"]:
        case = next(c for c in CASES if c["topic"] == entry["topic"])
        verdict = entry["verdict"]
        print(f"\n   【{entry['topic']}】")
        print(f"     漂移指标: {entry['reason']}")
        print(f"     判断结果: {verdict}")
        print(f"     真相: {case['truth']}")
        if verdict == "holds":
            print("     → 画像**一动不动**。数字漂了，人没变——")
            print("       「最近聊的事不一样」不是「人变了」。")
        else:
            print(f"     → 画像修正: {entry.get('new_profile_id','')}")

    print("\n" + LINE)
    print(" 四、结果对比")
    print(LINE)
    for case in CASES:
        topic = case["topic"]
        hist = store.profile_history(topic)
        print(f"\n   【{topic}】")
        for i, p in enumerate(hist):
            mark = (f"已失效（{p.invalidated_at[:10]} 起，进历史）"
                    if p.invalidated_at else f"当前有效（{p.status}）")
            tag = "旧版本" if i == 0 and len(hist) > 1 else "当前"
            print(f"     [{tag}] {p.statement}")
            print(f"            {mark}")
        if len(hist) == 1:
            print("     → 只有一条：判断没被改动。")

    print("\n   ⚠️ 两条路走下来，差别不在数字，在判断。")
    print("      而这个判断有据可依：**两段材料都摆在那儿，人能复核**。")

    print("\n" + LINE)
    print(" 五、留痕")
    print(LINE)
    trace_dir = cfgmod.abspath(cfgmod.PATHS["trace_dir"])
    for f in sorted(trace_dir.glob("漂移-*.jsonl")):
        print(f"   {f.name}（{len(f.read_text(encoding='utf-8').strip().splitlines())} 行）")
    print("   每次检测都落一行：漂移指标 + 判断结果 + 修正后的画像 id。")
    print("   单看每一次都很平淡，但把几周的记录排起来就看得出")
    print("   「是一直在漂，还是只抖了一下」——这正是二期做趋势曲线的素材。")

    print("\n" + LINE)
    print(" 四个阶段到此走完：S0–S3 分级 → 线索唤醒 → 三次提炼 → 备忘录 → 镜像 → 渐变检测。")
    print(" 整套「不滞留」现在是三条路：突变（revise）/ 冷掉（age_out）/ 渐变（drift）。")
    print(LINE)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
