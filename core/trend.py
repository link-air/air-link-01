"""渐变检测——「不滞留」里最难的那一半。

逻辑层第 ⑤ 条把话说得很准：
> 现有的「矛盾证据触发降级」只处理**突变**（出现明确反例），处理不了**渐变**——
> 人慢慢变了，但没有任何一个可指认的反例事件。

所以「不滞留」有三条路，各管一段：

| 路径 | 管什么 | 机制 |
|---|---|---|
| `revise_profile` | **突变**：明确的矛盾证据 | LLM 判 holds / revise / overturn |
| `age_out_profiles` | **冷掉**：长期没人提 | 超 `stale_days` → 降回 pending |
| **`detect_drift`（本模块）** | **渐变**：一直在提，但内容在变 | 按时间切两段比向量与情绪 |

中间那条缝正是本模块要补的：画像「活跃」（`last_support_at` 很新，不会被老化），
也没有单条场景算得上反例（LLM 逐条判都是 holds），
但把早期和近期**放在一起看**，已经不是一个样子了。

两个刻意的设计：

1. **检测只提出疑问，不自己动手**。漂移超阈值 → 交给 LLM 看两段材料判
   「是人变了，还是只是最近聊的事不一样」。因为**漂移 ≠ 变了**：
   最近正好聊了几件不同的事，向量中心一样会动。这一层的判断不能省。
2. **没有向量也要能跑**。情绪（valence / arousal）存在库里，不依赖 embedding 服务——
   降级时只失去「语义」那一半，情绪那一半照常工作。

> `trend` 的检测结果落 trace（`data/trace/漂移-*.jsonl`）而不入库：
> 一期没有消费方（呈现和统计都还不需要它）。哪天要做「漂移曲线」——
> 连着几周的漂移度画成一条线，看它是一次性的还是持续的——再升级成表。
> **先别为想象中的需求建表。**
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L9 渐变检测
#   上游    ：config、embedding（余弦）、prompts（`drift_prompt`）、store
#   下游    ：distill（`run_distill_cycle` 在老化之后带它跑一轮）
#   对外入口：`topic_drift`（只给数）/ `detect_drift`（带着 LLM 判断）/ `split_by_time`
#   边界    ：**提疑问，不动手**——它不自己改画像，改是 `distill.revise_profile` 的事
# ---------------------------------------------------------------------
from __future__ import annotations

from . import config as cfgmod
from .embedding import cosine
from .prompts import REVISION_SCHEMA, drift_prompt
from .store import append_trace, now_str


# ---------------------------------------------------------------------
# 一、纯计算：切两段、算漂移
# ---------------------------------------------------------------------

def split_by_time(scenes: list, ratio: float = 0.5) -> tuple[list, list]:
    """按时间把场景切成前 / 后两段。

    时间取 `time_event`（事件发生时间）优先、退回 `created_at`：
    「人什么时候变的」问的是事件发生在何时，不是什么时候被记下来的。
    """
    ordered = sorted(scenes or [],
                     key=lambda s: (s.time_event or s.created_at or ""))
    if len(ordered) < 2:
        return ordered, []
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * ratio)))
    return ordered[:cut], ordered[cut:]


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _center(vecs: list[list[float]]) -> list[float] | None:
    """一组向量的中心（逐维平均）。

    用平均而不是「取最典型的一条」：平均能吸收个别噪声，
    而渐变恰恰是一堆「单看都正常」的样本累积出来的偏移——
    逐条看会被每一条自己骗过去。
    """
    vecs = [v for v in (vecs or []) if v]
    if not vecs:
        return None
    dim = len(vecs[0])
    vecs = [v for v in vecs if len(v) == dim]
    if not vecs:
        return None
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def vector_drift(early_vecs: list, recent_vecs: list) -> float | None:
    """语义中心的漂移度 = 1 − 余弦（0 = 没变，越大越远）。"""
    a, b = _center(early_vecs), _center(recent_vecs)
    if not a or not b:
        return None
    return round(1.0 - cosine(a, b), 4)


def emotion_shift(early_scenes: list, recent_scenes: list) -> dict:
    """情绪的移动量：早期均值 → 近期均值。

    `valence` 拿不准的（NULL）**跳过**，不当作 0——0 是「确定的中性」，
    含义不同。
    """
    out: dict = {}
    for name in ("valence", "arousal"):
        e = _mean([getattr(s, name) for s in early_scenes or []])
        r = _mean([getattr(s, name) for s in recent_scenes or []])
        out[f"{name}_early"] = None if e is None else round(e, 3)
        out[f"{name}_recent"] = None if r is None else round(r, 3)
        out[f"{name}_shift"] = (None if (e is None or r is None)
                                else round(abs(r - e), 3))
    return out


# ---------------------------------------------------------------------
# 二、某个 topic 漂没漂
# ---------------------------------------------------------------------

def topic_drift(store, topic: str) -> dict:
    """算一个 topic 的漂移指标（**只给数，不做判断**）。

    返回里 `drifting` 是「值得拿给人/模型再看一眼」的意思，
    不是「已经变了」——那是 `detect_drift` 里 LLM 的事。
    """
    conf = cfgmod.cfg("trend", default={}) or {}
    min_scenes = int(conf.get("min_scenes", 6))
    ratio = float(conf.get("split_ratio", 0.5))

    scenes = store.query_scenes(topic=topic)
    result = {"topic": topic, "n": len(scenes), "drifting": False,
              "vector_drift": None, "reason": "", "early_ids": [], "recent_ids": []}
    if len(scenes) < min_scenes:
        result["reason"] = f"场景不足（{len(scenes)} < {min_scenes}），切不成两段"
        return result

    early, recent = split_by_time(scenes, ratio)
    result["early_ids"] = [s.id for s in early]
    result["recent_ids"] = [s.id for s in recent]

    reasons = []

    # 语义那一半：需要向量（没有就退化为只用情绪）
    vd = vector_drift([s.emb for s in early], [s.emb for s in recent])
    result["vector_drift"] = vd
    if vd is not None and vd >= float(conf.get("vector_drift_threshold", 0.15)):
        reasons.append(f"语义中心漂移 {vd}")

    # 情绪那一半：不需要向量服务，降级时照常工作
    shift = emotion_shift(early, recent)
    result.update(shift)
    if (shift.get("valence_shift") is not None
            and shift["valence_shift"] >= float(conf.get("valence_shift_threshold", 1.0))):
        reasons.append(f"效价移动 {shift['valence_shift']}"
                       f"（{shift['valence_early']} → {shift['valence_recent']}）")
    if (shift.get("arousal_shift") is not None
            and shift["arousal_shift"] >= float(conf.get("arousal_shift_threshold", 0.5))):
        reasons.append(f"唤醒度移动 {shift['arousal_shift']}"
                       f"（{shift['arousal_early']} → {shift['arousal_recent']}）")

    result["drifting"] = bool(reasons)
    result["reason"] = "；".join(reasons)
    return result


# ---------------------------------------------------------------------
# 三、扫一遍：把「值得再看一眼」的交给 LLM 判
# ---------------------------------------------------------------------

def detect_drift(store, llm) -> dict:
    """扫所有 topic，对漂移的做一次判定；判到「确实变了」就修正画像。

    为什么还要过一遍 LLM：**漂移 ≠ 变了**。
    最近正好聊了几件不同的事，向量中心一样会动；情绪均值也会动。
    所以流程是「**算数提出疑问 → 判断回答疑问**」，
    而不是「数一变就改画像」——后者会让画像随噪声来回翻。

    返回 `{"checked", "drifting", "revised"}`，全量结果落 trace。
    """
    from .distill import revise_profile       # 延迟 import：distill 依赖本模块的调用方

    conf = cfgmod.cfg("trend", default={}) or {}
    checked, drifting, revised, details = 0, [], [], []

    for topic in store.list_topics():
        profile = store.current_profile_by_topic(topic)
        if profile is None:
            continue                          # 没有画像就没有「变没变」的问题
        checked += 1
        d = topic_drift(store, topic)
        if not d["drifting"]:
            continue

        scenes = store.query_scenes(topic=topic)
        by_id = {s.id: s for s in scenes}
        early = [by_id[i] for i in d["early_ids"] if i in by_id]
        recent = [by_id[i] for i in d["recent_ids"] if i in by_id]

        data = llm.structured(
            drift_prompt(topic, profile.statement, early, recent), REVISION_SCHEMA)
        verdict = data.get("verdict") or "holds"
        new_statement = (data.get("statement") or "").strip()

        entry = {"topic": topic, "profile_id": profile.id, **{
            k: d[k] for k in ("n", "vector_drift", "valence_shift", "arousal_shift",
                              "reason")}, "verdict": verdict}
        if verdict in ("revise", "overturn") and new_statement:
            from .distill import _merge_ids
            merged = _merge_ids(profile.sources, [s.id for s in recent])
            entry["new_profile_id"] = revise_profile(
                store, profile.id, new_statement, merged)
            revised.append(entry["new_profile_id"])
        drifting.append(entry)
        details.append(entry)

    _write_trace({"ts": now_str(), "checked": checked,
                  "drifting": details, "revised": revised})
    return {"checked": checked, "drifting": drifting, "revised": revised}


def _write_trace(record: dict) -> None:
    """漂移检测留痕（与唤醒 trace 同目录、同理由：判断的来龙去脉要可见）。

    尤其这类「慢变化」——单看每一次检测都很平淡，
    但把几周的记录排起来就看得出「是一直在漂，还是抖了一下」。
    """
    append_trace("漂移", record)
