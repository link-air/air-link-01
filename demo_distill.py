"""阶段 2 演示：三次提炼 + 印证收敛 + 修正 + 老化 + 可否决。

跑法：
    python demo_distill.py            # 内置剧本，不需要 key
    python demo_distill.py --fresh    # 先清空演示库
    run.cmd distill                   # 等价于第一种

和 `demo.py` 的分工：那个演示**写入与唤醒**（S0→S1、C1–C7 → 动作），
这个演示**提炼**（S1→S2→S3 以及围绕它的那套约束）。
两个脚本都不需要 API key，跑的是同一条代码路径。

这个脚本最想让人看见的，是**「不武断」被写成了可计算的判据**：
  - 证据不够 → 停在 pending，不许 established
  - 情境不同 → 同类反应也不算同向收敛
  - 用户否了 → 立刻生效（这是能被证伪的承诺，不是态度声明）
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
from core.chat import build_embedding
from core.distill import age_out_profiles, run_distill_cycle
from core.llm import LLM
from core.model import PROFILE_ESTABLISHED, Scene
from core.store import Store
from core.weave import user_reject_profile

LINE = "=" * 70
TOPIC = "用户·面对被评价的反应"

# 三条**同主题、同情境**（trigger_class 都是「被评价」）的场景。
# 这是能收敛的最小配置：证据够 3 条 + 情境同类。
SITUATIONS = [
    ("被组长当众批评后想离开", "组长当众批评了他", "不想争、想走"),
    ("方案被主管否掉", "主管说方案不行", "当晚改了三版"),
    ("被同事当面质疑能力", "同事质疑他的方案", "没反驳，回去自己查"),
]
# 第 4 条：支持既有判断（holds）
SUPPORTING = ("又被客户挑刺，他还是选择先退一步", "客户挑刺", "先退一步")
# 第 5 条：显示人变了（revise）
CHANGING = ("这次被批评，他当场就把话说清楚了", "被批评", "当场回应，没有回避")


def make_mock_llm() -> LLM:
    """剧本模型：一次周期里会依次用到四种 prompt，按 schema 名分发。"""
    q_summary = [{"text": ("他先后在三个场合被评价（被组长当众批评、方案被主管否掉、"
                           "被同事当面质疑），当时的反应都是不再争取、往后退一步。")}]
    q_profile = [{"statement": "遇到被评价的情境时，他会先退出、不再争取"}]
    q_revision = [
        {"verdict": "holds", "statement": ""},                              # 第 4 条：支持
        {"verdict": "revise", "statement": "遇到被评价时，他会先扛一下再决定"},
    ]
    q_topic = [{"topic": TOPIC}]

    def pick(q):
        return q.pop(0) if len(q) > 1 else q[0]

    def fn(prompt, schema):
        name = schema.get("name")
        if name == "topic_summary":
            return pick(q_summary)
        if name == "profile_abstraction":
            return pick(q_profile)
        if name == "profile_revision":
            return pick(q_revision)
        if name == "topic_choice":
            return pick(q_topic)
        return {}

    llm = LLM()
    llm.set_mock(fn)
    return llm


def add_scene(store: Store, title: str, trigger: str, reaction: str) -> Scene:
    """造一张场景卡（**不走提取**——这一阶段要看的是 S1→S2→S3）。

    日期按条数往下排：切成两段要求时间有先后（`trend.split_by_time` 靠它）。
    """
    s = Scene(topic=TOPIC, trigger_class="被评价", subject="user",
              title=title, text=f"{title}（摘要）", trigger=trigger, reaction=reaction,
              valence=-1, arousal=1, intensity=0.6,
              time_event=f"2026-08-{10 + store.count('scenes'):02d} 09:00:00")
    store.add_scene(s)
    return s


def show_cycle(tag: str, stats: dict) -> None:
    """打印一个提炼周期的结果——`（无）` 也要打：分不清"没跑"和"跑了没产出"
    是排查后台任务时最常见的一次卡壳。"""
    print(f"\n  ▸ {tag}")
    print(f"    S2 新增: {stats['s2_new'] or '（无）'}      "
          f"S3 新增: {stats['s3_new'] or '（无）'}")
    print(f"    维持(holds): {stats.get('held') or '（无）'}   "
          f"修正(revise/overturn): {stats['revised'] or '（无）'}")
    print(f"    本轮收敛为已立: {stats['established'] or '（无）'}   "
          f"老化降级: {stats['aged'] or '（无）'}")
    print(f"    归档: {stats['archived']}")


def show_profiles(store: Store, note: str = "") -> None:
    """打印某个 topic 的**整条画像历史**（含已失效的版本）。

    双时间戳存在的意义就是能回答「某个月 air 眼中的他是什么样」。
    """
    print(f"\n  ── 画像状态{'（' + note + '）' if note else ''} ──")
    hist = store.profile_history(TOPIC)
    if not hist:
        print("    （该主题下还没有画像）")
        return
    for p in hist:
        if p.invalidated_at:
            mark = f"已失效（{p.invalidated_at[:10]} 起，进历史）"
        elif p.status == "established":
            mark = "已立（当前有效）"
        else:
            mark = "待验证（不进常驻）"
        print(f"    {p.id}  {p.statement}")
        print(f"          印证 {p.evidence} 条 · {mark}")


def show_chain(store: Store) -> None:
    """打印加工链——**派生**出来的（2026-09-24 起没有边表）。

    关系不再存一份：画像的依据看 `sources`、"被引用那一刻"看 `evidence_at`、
    摘要的素材看 `Summary.sources`、原文按天翻文档。
    """
    print("\n  ── 加工链（派生，不再存边）──")
    for p in store.current_profiles(status=None)[:2]:
        print(f"    {p.id}「{(p.statement or '')[:20]}」")
        print(f"      依据 {len(p.sources or [])} 条："
              f"{'、'.join((p.sources or [])[:6])}")
        if p.evidence_at:
            sid, at = sorted(p.evidence_at.items(), key=lambda kv: kv[1])[0]
            print(f"      最早引用：{sid} @ {at[:16]}")
    for s2 in store.hot_summaries(2):
        print(f"    {s2.id}「{(s2.text or '')[:16]}…」素材 {len(s2.sources or [])} 条")


def main() -> int:
    parser = argparse.ArgumentParser(description="air-link-01 · 阶段 2 演示（提炼与印证）")
    parser.add_argument("--fresh", action="store_true", help="先清空演示库再跑")
    args = parser.parse_args()

    # 外部配置（config.local.json / 环境变量）**必须先合进 CONFIG**，
    # 否则下面建 LLM / embedding 读到的还是 config.py 里的空默认值——
    # 配了也等于没配（脚本会安静地全程走降级）。
    settings.apply()
    cfgmod.ensure_dirs()
    demo_db = cfgmod.abspath("data/demo_distill.db")
    if args.fresh and demo_db.exists():
        demo_db.unlink()
        print(f"[--fresh] 已删除 {demo_db.name}")

    store = Store(demo_db)
    llm = make_mock_llm()
    # 向量服务：配了就接上——「同情境判定」的「trigger 语义相近」那一半要用它，
    # 以前这里没接，于是那一半从来没被演示过（只比 trigger_class）。
    # 没配就降级，脚本照样跑得完。
    emb = build_embedding()

    print(LINE)
    print(" air-link-01 · 阶段 2 演示：S1 → S2 → S3 与围绕它的那些约束")
    print(LINE)
    print(f" 库文件: {demo_db}")
    print(f" 主题  : {TOPIC}")
    print(f" 向量  : {'已配置' if emb else '未配置（同情境判定退化为只查 trigger_class——降级）'}")

    print("\n" + LINE)
    print(" 一、写入 3 条同主题、同情境的场景（trigger_class 都是「被评价」）")
    print(LINE)
    for title, trigger, reaction in SITUATIONS:
        s = add_scene(store, title, trigger, reaction)
        print(f"    {s.id}  {title}")
        print(f"           情境：{trigger} → 反应：{reaction}")

    print("\n" + LINE)
    print(" 二、第一次后台提炼")
    print(LINE)
    print("  S1→S2 聚合（只叙述，不下结论）→ S2→S3 抽象（下判断，要过四道关）")
    stats = run_distill_cycle(store, llm, emb=emb)
    show_cycle("run_distill_cycle", stats)

    s2 = store.summaries_by_topic(TOPIC)
    for x in s2:
        print(f"\n    {x.id}（聚合叙述）: {x.text}")
        print(f"          原料 = {x.sources}")

    prof = store.current_profile_by_topic(TOPIC)
    if prof:
        print(f"\n    {prof.id}（画像）: {prof.statement}")
        print(f"          状态 = {prof.status} · 印证 {prof.evidence} 条 "
              f"· 来源 {prof.sources}")
    show_profiles(store, "三条同情境证据 → 直接收敛为已立")
    show_chain(store)

    print("\n" + LINE)
    print(" 三、新证据 1：支持既有判断（holds——追加证据，不换版本）")
    print(LINE)
    s = add_scene(store, *SUPPORTING)
    print(f"    {s.id}  {SUPPORTING[0]}")
    stats = run_distill_cycle(store, llm, emb=emb)
    show_cycle("run_distill_cycle", stats)
    show_profiles(store, "版本没变，印证数上升")

    print("\n" + LINE)
    print(" 四、新证据 2：人变了（revise——旧版本进历史，新版本另起）")
    print(LINE)
    s = add_scene(store, *CHANGING)
    print(f"    {s.id}  {CHANGING[0]}")
    stats = run_distill_cycle(store, llm, emb=emb)
    show_cycle("run_distill_cycle", stats)
    show_profiles(store, "同一主题的两条 = 一条变化轨迹（双时间戳）")
    hist = store.profile_history(TOPIC)
    if len(hist) >= 2:
        old, new = hist[0], hist[-1]
        print(f"\n    双时间戳的用处：能查「某个时刻 air 眼中的他是什么样」。")
        print(f"        查 {old.valid_at[:10]} 那天（它失效于 {old.invalidated_at[:10]}）：")
        print(f"            {old.statement}")
        print(f"        现在（{new.valid_at[:10]} 起）：")
        print(f"            {new.statement}")
        print("    ⚠️ 旧版本只是被填了 invalidated_at，**没有删除**——素材不丢。")

    print("\n" + LINE)
    print(" 五、老化：把时间推到 100 天后（被动版「不滞留」）")
    print(LINE)
    future = datetime.now() + timedelta(days=100)
    aged = age_out_profiles(store, now=future)
    print(f"    降级: {aged or '（无）'}")
    show_profiles(store, "超 stale_days 未被印证/提及 → 降回待验证")
    print("\n    ⚠️ 注意 invalidated_at 依然是空的：")
    print("       老化是「过期」，不是「作废」——记录还在、还能被召回，")
    print("       只是不再常驻、不再当作事实。invalidated_at 只留给「修正」。")

    print("\n" + LINE)
    print(" 六、可否决：用户说「不对，我不是这样的人」")
    print(LINE)
    pending = store.current_profile_by_topic(TOPIC)
    if pending is not None:
        # 上一段刚把它老化成 pending。先恢复成已立，才能演示「否掉一条已立的判断」。
        store.set_profile_status(pending.id, PROFILE_ESTABLISHED)
        target = store.get_profile(pending.id)
        print(f"    否决 {target.id}（{target.status}）：{target.statement}")
        outcome = user_reject_profile(store, target.id)
        print(f"    → 结果: {outcome}（**真删**——行不在了）")
        print(f"       {target.id} 现在: {store.get_profile(target.id)}")
        print("       删的是一条不成立的推断：素材（原文/场景/摘要）都在——")
        print("       判断真成立的话，以后有新证据会重新立出来。")
        print("       ⚠️ 不分 pending / established 都删：否则它将来会被自动扶正，")
        print("          等于系统覆盖了人的否决。")
    show_profiles(store, "否决后当前有效画像为空")

    print("\n" + LINE)
    print(" 七、库内总量")
    print(LINE)
    # S0 按天存文档（`data/raws/`），**不在库里**——这里报的是文档份数（天数），
    # 不是条数（同 `dashboard.state()`）。`count("raws")` 是迁移前那张表的写法，
    # 那张表已经改名为 `raws_migrated`，这么查会直接抛「未知表名」。
    print(f"    S0 原文 {store.raw_doc_days()} 份（按天）· S1 场景 {store.count('scenes')} "
          f"· S2 摘要 {store.count('summaries')} · S3 画像 {store.count('profiles')} "
          f"· 实体 {store.count('entities')}")
    print("    （「边」不再入库：2026-09-24 起引用 / 相邻 / 原文全改派生，"
          "`edges` 表已退役——见存储层稿 §五）")
    print("    每处 sources 引用都能反向查到具体场景——画像可追溯。")

    print("\n" + LINE)
    print(" 阶段 2 完成。下一步（阶段 3）：memo 提醒状态机 + 编织层镜像呈现。")
    print(" 再往后（阶段 4）：向量持久化与趋势比对（完整的「不滞留」的渐变检测）。")
    print(LINE)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
