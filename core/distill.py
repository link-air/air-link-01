"""提炼流水线。

三次提炼 + 一个持续修正（存储层 §6）：

  | 提炼 1 | S0 → S1 | 切场景 + 抽字段            | 结构化，**不下结论** |
  | 提炼 2 | S1 → S2 | 同主题场景**聚合成叙述**    | 聚合，可逆，**不下结论** |
  | 提炼 3 | S2 → S3 | 跨主题抽**共性/模式**       | 抽象，不可逆，**下了结论** |
  | 持续修正 | S3 → S3' | 新证据修正画像           | 旧记录填 invalidated_at |

**这条区分决定了约束压在哪**：印证 / 收敛这类「防止武断」的机制
**只该作用在抽象层**——聚合不下结论所以不需要，抽象在判断一个人所以必须有。

写入单元是**话题段**不是轮次（设计稿二）：一句话构不成场景，太碎；
一整段有头有尾的互动才是场景的自然边界。所以 step1 收的是 `messages`（一段）。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L5 提炼层
#   上游    ：config、embedding、model、prompts、scene、entity、store
#   下游    ：shortterm（`distill_step1` = 它的压缩）、trend（渐变检测由这里带动）、
#             memo（`memo_cycle` 挂在提炼周期末尾）、dashboard（手动/周期跑/体检）
#   对外入口：`distill_step1` / `distill_step2` / `distill_step3` / `run_distill_cycle` /
#             `converge_detail` / `revise_profile` / `age_out_profiles` / `cap_profiles` /
#             `maintenance_cycle`（体检：封顶主整理 + S3 复核 + 留痕）/ `review_profiles`
#   边界    ：**改画像前一律从库里重读**（调用方手里那个对象可能是旧的，
#             拿它判「是否已升级」会重复计数、重复写库）
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta

from . import config as cfgmod
from .embedding import cosine
from .entity import link_entities
from .model import (INVALIDATED_REVISION,
                    MEMO_KINDS, MEMO_PENDING, MEMO_USER_TASK,
                    PROFILE_ESTABLISHED, PROFILE_PENDING, REVIEW_VERDICTS,
                    Memo, Profile, Raw, Scene, Summary)
from .prompts import (PROFILE_SCHEMA, REVISION_SCHEMA, REVIEW_SCHEMA,
                      SUMMARY_SCHEMA, TOPIC_SCHEMA,
                      profile_prompt, review_prompt, revision_prompt,
                      summary_prompt, topic_prompt)
from .scene import behavior_intensity, extract_scene, render_conversation
from .store import append_trace, now_str, scene_ids


def distill_step1(store, messages: list[dict], llm, emb_service=None,
                  source: str = "") -> tuple[Scene | None, str, list[dict]]:
    """S0→S1：把一段对话落成场景卡，并处理它的副产品。

    **`scene` 可能是 `None`**：这段被判定为"没有信息量"（纯寒暄 / 纯应答 /
    与刚聊完的完全重复），于是**不落库**。但 `window_digest` 照常返回——
    摘要要继续累积，否则被跳过的这几句会在短期记忆里凭空消失，
    她下一轮就"忘了刚才聊过"。**跳过只该意味着「不进长期库」，不该意味着「当没发生」。**

     把签名写成 `distill_step1(raw_id) -> scene_id`，那是抽象说法
    （「从原文提炼出场景」）。实际链路里原文就是这段对话本身，
    先抽字段才知道要写什么，所以这里收 `messages`、返回落库后的 `Scene`。

    一次调用产出的两样东西（这是「压缩=提取」的全部含义）：
      - `window_digest` → 回到短期窗口，替代被压掉的逐字（**不进 DB**）
      - `Scene` + open_loops → 进长期记忆

    顺带处理的四件事，都是设计稿定死的、缺一不可：
      1. **S0 原文** 落库（按天文档；S0 无独立索引，下钻按日期翻——2026-09-24
         起不再建 `raw` 指针边）
      2. **实体** 挂索引（含消歧）
      3. **同 topic 相邻**：R1 扩散现算（`store.topic_neighbors`）——
         2026-09-24 起不再建 `causality` 边
      4. **open_loops → memos**（含 air 自己的承诺——它管着 air 的一致性）
    """
    # 候选清单注入：topic 要稳定，先给已有主题让模型挑（同求知版 list_taglib 的教训）
    k = cfgmod.cfg("distill", "topic_candidates_k", default=5)
    topic_candidates = store.list_topics()[:k] if k else []

    scene, digest, entities, verdict = extract_scene(
        messages, llm, emb_service=emb_service,
        topic_candidates=topic_candidates, source=source)

    # 值不值得存——这一段没信息量就不落库（判定和抽取是同一次调用，不额外花钱）。
    # 注意**不是"什么都不做"**：摘要照留（窗口的连续性），只是少一张场景卡。
    if not verdict["worth"]:
        reason = verdict["reason"] or "未说明原因"
        print(f"[distill] 这一段不值得存，只留摘要、不落库：{reason}")
        write_skip_trace(messages, reason)
        return None, digest, []

    # 强度用行为信号算（不靠 LLM）。「此前被提及次数」按同 topic 的场景数近似——
    # 同一个主题聊过几次，就是这件事在生活里出现了几次。
    prev = store.query_scenes(topic=scene.topic) if scene.topic else []
    scene.intensity = behavior_intensity(messages, prev_mentions=len(prev))

    store.add_scene(scene)

    # S0：原文按天写成文档（不在库里）。**没有索引、没有关键词**——
    # 这是存储层的物理约束，不要在这里给它加任何"便于检索"的字段。
    # `on_date` 用**场景发生日**：跨天提取时，这段对话属于它发生的那一天。
    raw = Raw(scene_id=scene.id, content=render_conversation(messages))
    store.add_raw(raw, on_date=(scene.time_record or "")[:10])
    # （原「raw 边」——记录"这条场景的原文在哪天那份文档里"——已去，2026-09-24：
    #   **读取端早就不用它**（`get_raws_by_scene` 按 scene_id 直接翻按天文档），
    #   只写不读的边是死数据——存储层稿 §五"能派生的不存"。）

    link_entities(scene.id, entities, store)

    # （原「causality 边」已去，2026-09-24：R1 的扩散改**派生**——
    #   `store.topic_neighbors()` 按"同 topic + 时间相邻"现算（存储层稿 §五
    #   "能派生的不存"）。原来那份克制仍成立：相邻 = 前后各一条、不是全链——
    #   全链会让 R1 扩散退化成「把所有记忆都捞出来」，扩散失去选择性。）

    _write_memos(store, scene, llm)
    return scene, digest, entities


def write_skip_trace(messages: list[dict], reason: str) -> None:
    """跳过落库也要留痕——**「没存什么、为什么」和「存了什么」一样重要**。

    两道判断（代码判寒暄 / 模型判没信息量）**都要调它**：
    只留一道，另一道就成了「说不清为什么没记住」的黑洞，
    而那正是这类系统最难查的故障（应该出现的东西没出现）。

    这个函数公开（不叫 `_write_skip_trace`）就是因为**两个模块都要用**：
    `distill` 走模型判那条路，`shortterm` 走代码判那条路。
    留痕的格式与位置必须一致，否则查的时候要翻两个地方。
    """
    append_trace("跳过", {"ts": now_str(), "reason": reason,
                          "preview": render_conversation(messages)[:200]})


def _merge_ids(*groups) -> list[str]:
    """合并 id 列表（保序去重）——`sources` 是列表，重复项会让印证计数虚高。"""
    out: list[str] = []
    for g in groups:
        for i in g or []:
            if i and i not in out:
                out.append(i)
    return out


def _dominant_subject(scenes: list[Scene]) -> str:
    """一组场景的归属（取多数）。画像要沿用它做「跨来源计数」的判据。"""
    if not scenes:
        return "user"
    counts: dict[str, int] = {}
    for s in scenes:
        counts[s.subject or "user"] = counts.get(s.subject or "user", 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _has_sources(profile: Profile) -> bool:
    """可追溯（四道关第 2 关）：`sources` 必填。

    air 说不出依据的画像不该存在——这既是防虚构的兜底，
    也是「用户能否决」能落地的前提（没有依据，否决权是空的）。
    """
    return bool(profile.sources)


def _same_situation(scenes: list[Scene], emb=None) -> tuple[bool, str]:
    """同情境判定（四道关第 1 关的前半）。返回 `(是否通过, 为什么)`。

    两层（存储层 §6）：`trigger_class` 相同 **且** `trigger` 语义相近。
    不同情境下的同类反应**不算**同向收敛——
    「被批评后退出」和「被表白后退出」是两件事，不能拼成「他一遇事就退」。

    ⚠️ **比的是 `trigger`（触发情境），不是整个场景。**
    这一点最初写错了（用 `scene.emb` 比整体），后果是跑实验才发现的：
    三条场景都是「被评价」（class 同、topic 同、印证 3 次），
    可它们的整体向量掺着"谁、什么反应、什么结果"——越具体相似度越低，
    于是**同类的具体情境被误判成不同情境，画像永远升不了级**。
    规格写的就是「用前因后果的 `trigger` 比对 + 语义相近」，是这里偷懒了。

    降级说明：拿不到 trigger 向量时退化为**只查 `trigger_class`**（少了语义那一层）。
    宁可更保守的另一面：**没有情境标签（class 为空）一律判不通过**。

    **为什么返回原因**：这一关有四种失败方式，而它们的对策完全不同——
    「标签太少」是模型整体没在分类（查抽取 prompt）；
    「类别不一致」是分类噪声（查类别边界写清没有）；
    「语义不够近」要么是阈值偏高、要么是情境真的不同（看 avg 与阈值的距离）。
    以前这四种都只表现为「画像没立」，只能翻库逐条猜——那是最贵的调试成本。
    """
    # **多数同类**，不是「完全相同」（2026-09-12 改）。
    # 原因：trigger_class 是 LLM 判的，会有单条波动——实测里「被组长当众批评」
    # 和「开会时被打断」被分到了两类，一条噪声就把整条画像卡死在 pending。
    # 要求「多数」（≥60% 且至少 2 条）既能容忍个别误判，
    # 又守住了设计意图：真·不同情境（各说各的）凑不出多数，照样不通过。
    classes = [s.trigger_class for s in scenes if s.trigger_class]
    if len(classes) < 2:
        return False, (f"情境标签不足（{len(classes)}/2 条有 trigger_class）"
                       f"——模型整体没在分类，先查抽取 prompt")

    dist = Counter(classes).most_common()
    need = max(2, int(len(classes) * 0.6 + 0.999))
    if dist[0][1] < need:
        detail = " / ".join(f"{k}×{v}" for k, v in dist)
        return False, f"情境不一致（{detail}，要求同一类 ≥{need} 条）——分类噪声"

    triggers = [s.trigger for s in scenes if s.trigger]
    if len(triggers) < 2:
        return True, "只靠 trigger_class 判定（没有可比的情境文本）"

    if emb is not None and getattr(emb, "available", False):
        vecs = emb.embed(triggers)      # 现算：trigger 不像场景那样存了向量
        if vecs and len(vecs) == len(triggers):
            sims = [cosine(vecs[i], vecs[j])
                    for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
            avg = sum(sims) / len(sims)
            th = float(cfgmod.cfg("distill", "trigger_sim_threshold", default=0.5))
            if avg < th:
                return False, (f"trigger 语义不够近（avg {avg:.2f} < {th}）"
                               f"——阈值偏高，或情境确实不同")
            return True, f"同情境（class 多数一致，trigger 语义 avg {avg:.2f}）"
    return True, "只靠 trigger_class 判定（无向量服务）"


def _cross_source_ok(profile: Profile, scenes: list[Scene]) -> bool:
    """跨来源计数（四道关第 3 关）。

    「关于用户本人」的画像**只认用户本人的直接表达**（别人转述的不算）；
    「关于世界 / 关于 air」的可以跨来源（不同场景从不同角度印证同一件事）。
    规则值来自 `CONFIG.distill.cross_source`，不是这里的硬编码。
    """
    rule = cfgmod.cfg("distill", "cross_source", default={}) or {}
    if rule.get(profile.subject):
        return True
    return all((s.subject or "user") == "user" for s in scenes)


def _evidence_scenes(store, profile: Profile) -> list[Scene]:
    """这条画像的全部印证场景（顺着 sources 里的 S1 下钻）。"""
    out = []
    for sid in scene_ids(profile.sources):
        s = store.get_scene(sid)
        if s is not None:
            out.append(s)
    return out


def _gate_recheck(store, profile: Profile, emb=None) -> tuple[bool, str]:
    """四道关里**可算**的部分：复查一条画像还满不满足（**不写库、不改状态**）。

    为什么单拎出来（2026-09-23，记忆整理稿 §五）：这四条——印证数 ≥ `evidence_min`、
    `sources` 必填、同情境、跨来源——都是**有确定答案**的，代码判得比模型准
    （门槛值在配置里，模型只能凭常识猜）。所以复核**不该把它们交给模型**。
    收敛（判 pending 能不能立）与复核（查 established 还成不成立）共用这一份。

    复核时它是"**状态与判据脱节**"的出口：`_evidence_scenes` 只数**现存**场景
    （引用 = `sources ∩ 现存节点`，2026-09-24 起不再靠删端摘引用），所以
    S1 被删之后一条画像可能还挂着 established 却早就不满足了——
    那时降档不需要过模型、不用花钱。
    """
    scenes = _evidence_scenes(store, profile)
    need = int(cfgmod.cfg("distill", "evidence_min", default=3))
    if len(scenes) < need:
        return False, f"印证不足（{len(scenes)}/{need}）"
    if not _has_sources(profile):
        return False, "缺 sources（无法追溯）"
    ok, why = _same_situation(scenes, emb)
    if not ok:
        return False, why
    if not _cross_source_ok(profile, scenes):
        return False, "跨来源规则不过（user 类只认本人直接表达）"
    return True, f"印证 {len(scenes)} 次；{why}"


def resolve_topic(store, statement: str, subject: str, llm) -> str:
    """为画像决定 topic（存储层 §3「新建 vs 复用」）。

    ① 候选生成：按 `subject` 硬过滤（「用户·X」只在 user 类里找），再语义 Top-K
    ② 二元判断：把「新陈述 + 候选」给 LLM，问「属于哪个，还是新建？」
    ③ **拿不准 → 新建**（复用污染隐蔽、难恢复；新建裂缝可见，用 `merge_topics` 补救）

    返回空串表示「这次定不了」——上层据此**放弃本次抽象**，而不是编一个 topic。

    ⚠️ 现状（2026-10-06 全量复核）：**生产链路零调用点**（AST 实测 0 处，只有
    测试在用）。topic 实际由 step1 抽取时的候选注入（`topic_candidates`）随场景
    落库自组织，S3 只认现成的 topic 键——本函数是旧口，留着，不代表在跑。
    """
    k = int(cfgmod.cfg("distill", "topic_candidates_k", default=5) or 5)
    candidates = store.list_topics(subject)[:k]
    data = llm.structured(topic_prompt(statement, subject, candidates), TOPIC_SCHEMA)
    topic = (data.get("topic") or "").strip()
    # 防御：模型偶尔会回一整句话当 topic。topic 是索引键，长了就没法当序列纽带。
    if len(topic) > 40:
        topic = topic[:40]
    return topic


def _extra_candidates(store, main: str, k: int = 12) -> list[str]:
    """附加主题的候选：已有主题去掉主主题（按名字排序，取前 k 个——可复现）。

    为什么给候选而不许它自由发挥：附加主题会进检索——编出来的近义主题
    （「工作压力」vs「用户·工作压力」）会让"按主题找"变糊。同 `resolve_topic` 的
    候选注入：那边保的是版本序列，这边保的是检索，收敛的都是措辞。
    """
    return [t for t in store.all_topics() if t and t != main][:k]


def _clean_extra_topics(raw, main: str, candidates: list[str]) -> list[str]:
    """把 LLM 给的附加主题清一遍（2026-09-24 晚）：**只留候选清单里的**、
    去主主题、去重、最多 2 个。

    模型偶尔不守规矩（给整句话 / 编个新的 / 把主主题又抄一遍）——这里按
    **精确匹配**过滤：宁可少挂，也不让检索被野主题污染。
    """
    if not isinstance(raw, list):
        return []
    allowed = {t for t in candidates if t and t != main}
    out: list[str] = []
    for x in raw:
        t = str(x or "").strip()
        if t and t in allowed and t not in out:
            out.append(t)
    return out[:2]


def _fresh_scenes(store, topic: str) -> list[Scene]:
    """该 topic 里**还没被任何 S2 收进去**的场景（增量判据，一处定义）。

    两个用途：step2 判"攒够没有"、体检判"哪些 topic 最该先跑"
    （见 `_busiest_topics`）——同一套判据不写两份。
    """
    covered: set[str] = set()
    for s2 in store.summaries_by_topic(topic):
        covered |= set(s2.sources or [])
    return [s for s in store.query_scenes(topic=topic) if s.id not in covered]


def distill_step2(store, topic: str, llm, n: int | None = None) -> Summary | None:
    """提炼 2：S1 → S2 **聚合**（攒够 n 条同主题才做）。

    增量聚合：只聚「还没被任何 S2 收进去的」场景，落到一条新的 S2 上。
    这样 S2 是**可逆**的（同一批 S1 能重建出同一批 S2），
    也不会因为「重跑一次提炼」就把旧摘要覆盖掉。

    聚合**不下结论**（prompt 里写死了）——下了结论的东西要过印证，
    而这里没有印证机制，所以它不能下结论。
    """
    n = int(n or cfgmod.cfg("distill", "step2_n", default=3))
    fresh = _fresh_scenes(store, topic)
    if len(fresh) < n:
        return None                 # 幂等：没有攒够新场景就不再聚合

    fresh.sort(key=lambda s: s.time_event or s.created_at or "")
    cands = _extra_candidates(store, topic)
    data = llm.structured(summary_prompt(topic, fresh, candidates=cands), SUMMARY_SCHEMA)
    text = (data.get("text") or "").strip()
    if not text:
        return None
    # 附加主题（0-2 个）：只服务"找得到"——主主题（聚合键）仍然是 topic
    extra = _clean_extra_topics(data.get("extra_topics"), main=topic, candidates=cands)

    summary = Summary(topic=topic, topics=[topic] + extra, text=text,
                      sources=[s.id for s in fresh])
    store.add_summary(summary)
    # （原「compose 边」S1→S2 已去，2026-09-24：`Summary.sources` 就是它——
    #   追溯全走 sources（`_fresh_scenes` / 展开里的"它在哪些摘要里"），边是重复数据。）
    return summary


def converge(store, profile: Profile, emb=None) -> bool:
    """印证收敛：判定一条画像能否从 `pending` 升为 `established`（提炼 3 的四道关）。

    **不满足就保持 pending**——这是「不武断」在实现里的样子：
    证据不够时，宁可什么都不说。

    想知道「为什么没升级」用 `converge_detail()`；这里只取它的第一项，
    好让旧调用方不用改。
    """
    return converge_detail(store, profile, emb)[0]


def converge_detail(store, profile: Profile, emb=None) -> tuple[bool, str]:
    """带原因的收敛判定：四道关**卡在哪一关、差多少**。

    四道关：
    1. **印证数 ≥ `evidence_min`**（默认 3）**且同情境**（trigger_class 相同 +
       trigger 语义相近）——不同情境下的同类反应不算同向收敛
    2. `sources` 必填（可追溯）
    3. 跨来源计数规则（user 只认本人表达；world/air 可跨来源）
    4. 长期未印证 → 由 `age_out_profiles` 反向处理（降回 pending）

    **为什么要有这个函数**：四道关原本全是 `return False`，而失败的**表现只有一个**
    ——「画像没立」。至于是印证不够、类别漂了、还是阈值太高，只能翻库逐条猜。
    把原因说出来，调 prompt / 调阈值才有靶子；否则「优化」就只是反复重跑碰运气。
    """
    if profile is None or not profile.id:
        return False, "空画像"
    # **以库里的当前状态为准**，不信传进来的对象：调用方手里的对象可能是
    # 上一轮读出来的（status 还没更新），拿它判「是否已升级」会重复判定，
    # 而收敛会写库——重复判定意味着重复计数、重复写库。
    profile = store.get_profile(profile.id)
    if profile is None:
        return False, "画像不存在"
    if (profile.invalidated_at or ""):
        return False, "已作废"
    if profile.status == PROFILE_ESTABLISHED:
        return False, "已是已立"

    scenes = _evidence_scenes(store, profile)
    ok, why = _gate_recheck(store, profile, emb)
    if not ok:
        return False, why

    store.set_profile_status(profile.id, PROFILE_ESTABLISHED, evidence=len(scenes))
    return True, f"升级（{why}）"


def _forming_ids(store, summaries) -> set[str]:
    """哪些场景是画像的**出处**（聚合成那些 S2 的 S1）。

    出处和印证要分开（2026-09-11 定）：
      - **出处**在画像形成时就定了，少而稳定 → 保护期长、进追溯包
      - **印证**随新场景不断累积，多而持续增长 → 保护期短、不进包

    不分开的后果：印证的场景会一直新增，「被引用 = 永久免死」会让保护集
    只涨不消，`s1_cap` 形同虚设。
    """
    out: set[str] = set()
    for s2 in summaries or []:
        out |= set(scene_ids(s2.sources or []))
    return out


def _build_evidence_pack(scenes) -> list[dict]:
    """形成依据的快照（**追溯的底线保障**）。

    只装出处那几条，理由同上：包跟画像走，装全部印证会让它无限膨胀，
    而「依据是哪几条」这个问题，出处就回答得了。

    敏感场景**不进包**——同「敏感不进镜像」：air 眼中的你不该含
    「他有过创伤」这类标签，哪怕是以快照的形式。
    """
    return [{"id": s.id, "title": s.title, "summary": s.text,
             "time": (s.time_event or "")[:10], "trigger": s.trigger or ""}
            for s in (scenes or []) if not s.sensitive]


def _mark_evidence(store, scene_id: str, profile_id: str) -> None:
    """把一条场景记为对某条画像的印证：记「引用起点时间」+ 计数 +1（2026-09-24 改）。

    幂等（先看 `evidence_at` 里有没有）：同一条证据被重复记两次，会让印证数虚高——
    而印证数正是「能不能确立」的判据，虚高等于放水。

    为什么不再写边、也不再收 `role`（2026-09-24，存储层稿 §五）：evidence 边是对
    `sources` 的冗余存储，唯一的真数据是**"被引用那一刻"**（保护期起算）——
    它落进 `profiles.evidence_at`；而 role（forming / supporting）由
    `evidence_pack` 的 id 集合派生就够了（出处那几条进包，
    `protected_scene_ids` 认包里 = forming，同一套判法）。

    ⚠️ 这里调 `bump_counters(..., "cited")` 而不是 "mention"：
    画像引用场景是**重要性**；场景在对话里被说出来才是「提及」。
    """
    p = store.get_profile(profile_id)
    if p is None or scene_id in (p.evidence_at or {}):
        return
    p.evidence_at[scene_id] = now_str()
    store.set_profile_evidence_at(profile_id, p.evidence_at)
    from .recall import bump_counters                             # 延迟 import：避免环形依赖
    bump_counters(store, scene_id, "cited")


def revise_profile(store, pid: str, new_statement: str, evidence_ids: list[str],
                   by: str = INVALIDATED_REVISION, reason: str = "") -> str:
    """**修正**：旧记录填 `invalidated_at`，新记录 `valid_at`，同 topic 串联。

    `by` 区分发起者（见 `model.INVALIDATED_*`）：
    `INVALIDATED_REVISION` = air 自己拿新证据改；`INVALIDATED_USER` = **人改**
    （界面 / 确认条那条路，2026-09-24 起）。
    两者走同一条修正路径（旧版进历史、新版回 pending）——差别只在统计与呈现
    （见 `store.invalidate_profile`）。

    「空气不说依据的画像不该存在」在修正里的体现：新记录同样带 sources，
    而且是**追加**了这次的新证据后的完整来源列表——修正不是重开一条序列，
    是同一个 topic 上的下一个版本。
    """
    old = store.get_profile(pid)
    if old is None:
        raise ValueError(f"画像不存在: {pid}")
    store.invalidate_profile(pid, at=now_str(), by=by,
                             reason=reason or ("新证据修正" if by == "revision"
                                               else "用户改的"))

    # 新版本重新算一遍出处与快照：出处由「当前的 S2 聚合了哪些场景」决定，
    # 不随版本累积（否则每次修正都会把整条话题的场景都变成出处）。
    forming = _forming_ids(store, store.summaries_by_topic(old.topic))
    forming_scenes = [s for s in (store.get_scene(i) for i in forming) if s is not None]

    new = Profile(
        topic=old.topic,
        # 主题标签随版本走（2026-09-24 晚）：**附加主题不该因为改一次说法就丢**——
        # 主主题是版本序列的键，附加主题是检索标签，人改"说法"时它们都没变。
        topics=list(old.topics or [old.topic]),
        subject=old.subject, statement=new_statement,
        status=PROFILE_PENDING, evidence=len(scene_ids(evidence_ids)),
        sources=list(evidence_ids),
        evidence_pack=_build_evidence_pack(forming_scenes),
    )
    store.add_profile(new)
    for sid in scene_ids(evidence_ids):
        _mark_evidence(store, sid, new.id)
    return new.id


def distill_step3(store, topic: str, llm, emb=None) -> Profile | None:
    """提炼 3：S2 → S3 **抽象**（唯一在「下判断」的一步）。

    分两条路：
      - **没有当前画像** → 抽一条新的（`pending`），再跑一次收敛判定
      - **已有当前画像** → 把新证据给它，判 holds / revise / overturn
        （`holds` 就追加证据；`revise` / `overturn` 走 `revise_profile` 留下旧版本）

    为什么必须过 LLM 判一次「还是不是成立」：人会变。这是「不滞留」在
    证据出现时的主动版本，`age_out_profiles` 是被动版本（长期没人提）。
    """
    summaries = store.summaries_by_topic(topic)
    if not summaries:
        return None                     # 没有聚合就没有抽象（S2→S3 的次序不能倒）
    scenes = store.query_scenes(topic=topic)
    if not scenes:
        return None

    current = store.current_profile_by_topic(topic)
    if current is None:
        cands = _extra_candidates(store, topic)
        data = llm.structured(profile_prompt(topic, summaries, scenes, candidates=cands),
                              PROFILE_SCHEMA)
        statement = (data.get("statement") or "").strip()
        if not statement:
            return None                 # 证据不足 → 不写（prompt 里要求证据不足给空串）
        # 附加主题（0-2 个）：只服务"找得到"——版本序列仍认主主题 topic
        extra = _clean_extra_topics(data.get("extra_topics"), main=topic, candidates=cands)
        # 出处（形成依据）与印证分开记：出处进追溯包、享受长保护期；
        # 印证随新场景累积，只享受短保护期。混在一起会让保护集无限增长。
        forming = _forming_ids(store, summaries)
        profile = Profile(
            topic=topic, topics=[topic] + extra, subject=_dominant_subject(scenes),
            statement=statement, status=PROFILE_PENDING, evidence=len(scenes),
            sources=_merge_ids([s.id for s in summaries], [s.id for s in scenes]),
            evidence_pack=_build_evidence_pack([s for s in scenes if s.id in forming]),
        )
        store.add_profile(profile)
        # （原「compose 边」S2→S3 已去，2026-09-24：`Profile.sources` 就是它。）
        for s in scenes:
            _mark_evidence(store, s.id, profile.id)
        converge(store, profile, emb=emb)
        # **重读**：status 是 `converge` 在库里改的，手里这个对象还是旧的。
        # 不重读的话调用方拿到「pending」，统计里会凭空多出一条未收敛，
        # 而画像其实已经立了——报告说谎比没有报告更糟。
        return store.get_profile(profile.id) or profile

    # ---- 已有画像：只处理「新证据」 ----
    covered = set(current.sources or [])
    fresh = [s for s in scenes if s.id not in covered]
    if not fresh:
        return None                     # 幂等：没有新东西就不打扰

    data = llm.structured(revision_prompt(topic, current.statement, fresh), REVISION_SCHEMA)
    verdict = data.get("verdict") or "holds"
    new_statement = (data.get("statement") or "").strip()
    merged = _merge_ids(current.sources, [s.id for s in fresh])

    if verdict in ("revise", "overturn") and new_statement:
        new_id = revise_profile(store, current.id, new_statement, merged)
        return store.get_profile(new_id)

    store.set_profile_sources(current.id, merged, evidence=len(scene_ids(merged)))
    for s in fresh:
        # 新证据是**印证**（不是出处）：出处是画像形成时定下的那几条，不会再变。
        # 这样包和长保护期都只跟固定的一小批走（role 由包派生，不再单独记）。
        _mark_evidence(store, s.id, current.id)
    updated = store.get_profile(current.id)
    converge(store, updated, emb=emb)
    return store.get_profile(current.id) or updated     # 同上：重读才拿得到升级后的 status


def age_out_profiles(store, now: datetime | None = None) -> list[str]:
    """**老化降级**：长期未印证**且**未被提及 → 降回 `pending`。

    **不填 `invalidated_at`**：记录仍在、仍可召回。
    「不滞留」是「老结论会过期」，不是「老结论被删除」。
    `last_support_at` 由「被印证」和「被提及」两条路更新（见 `bump_counters`），
    所以这里判的是两者之中**较晚**的那次——只要求「没人再提起它」。
    """
    days = int(cfgmod.cfg("distill", "stale_days", default=90))
    cutoff = ((now or datetime.now()) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    aged = []
    for p in store.current_profiles(status=PROFILE_ESTABLISHED):
        stamp = p.last_support_at or p.valid_at or p.created_at or ""
        if stamp and stamp < cutoff:
            store.downgrade_profile(p.id)
            aged.append(p.id)
    return aged


def cap_profiles(store, cap: int | None = None) -> list[str]:
    """画像总数有上限：超了**印证最少的降回待验证**。

    **不是删**。画像永不硬删是硬约束（`invalidated_at` 只留给修正 / 归档——
    人的否决是真删、不填字段）——降档之后它还在、还可召回，证据攒够了还能再升回来。

    为什么要有上限：一个人身上"成立"的判断不该无限增长，多了就是不重要的；
    而全量注入会把上下文塞满（`_pick_profiles` 只管注入侧，这一条管总量）。
    """
    cap = cap if cap is not None else int(cfgmod.cfg("capacity", "profile_cap", default=100))
    cur = store.current_profiles(status=PROFILE_ESTABLISHED)
    if len(cur) <= cap:
        return []
    # 印证最少的先降；印证一样多时，老的先降（它被支撑的机会已经给过了）
    worst = sorted(cur, key=lambda p: ((p.evidence or 0), p.valid_at or ""))[:len(cur) - cap]
    out = []
    for p in worst:
        store.downgrade_profile(p.id)
        out.append(p.id)
    return out


def _busiest_topics(store, topics: list[str], cap: int) -> list[str]:
    """按「未覆盖的新场景数」从多到少取前 cap 个（体检的主整理用）。

    为什么按这个排：体检的钱要花在料最足的地方——攒了 5 条新场景的 topic
    比只有 1 条的更该先跑（1 条根本过不了 `step2_n` 的门槛，跑了也是空转）。
    并列时按 topic 名升序，保证同一份库上顺序稳定（可复现）。
    """
    scored = sorted(((-len(_fresh_scenes(store, t)), t) for t in topics))
    return [t for _k, t in scored[:max(0, int(cap))]]


def run_distill_cycle(store, llm, emb=None, topic_cap: int | None = None) -> dict:
    """后台提炼周期：一次跑完该跑的事。

    顺序不能乱：**先聚合再抽象**（S2 是 S3 的原料），
    最后才是收敛 / 老化 / 归档（它们作用在已经存在的画像上）。
    每一步都是幂等的——重复跑同一个周期不会产生重复数据，
    这是后台任务的基本要求（它会被定时器反复触发）。

    `topic_cap`（2026-09-23，记忆整理稿 §四）：只处理"最该处理的 N 个 topic"
    ——体检（`maintenance_cycle`）用它把主整理的钱封顶；
    会话边界整理**不传** = 全量（边界是精确时机，该做的都做完）。
    """
    from .recall import core_score        # 延迟 import：archive_s1 需要核心度

    stats: dict = {"s2_new": [], "s3_new": [], "held": [], "revised": [],
                   "established": [], "blocked": [], "aged": [], "capped": [],
                   "drift": {}, "memo": {}, "archived": {}}
    topics = store.list_topics()
    if topic_cap is not None:
        topics = _busiest_topics(store, topics, int(topic_cap))

    for topic in topics:
        s2 = distill_step2(store, topic, llm)
        if s2 is not None:
            stats["s2_new"].append(s2.id)

    for topic in topics:
        before = store.current_profile_by_topic(topic)
        before_status = before.status if before is not None else None
        p = distill_step3(store, topic, llm, emb=emb)
        if p is None:
            continue
        if before is None:
            stats["s3_new"].append(p.id)            # 新建了一条判断
        elif p.id != before.id:
            stats["revised"].append(p.id)           # 修正：旧版本进历史，新版本另起
        else:
            stats["held"].append(p.id)              # 维持：只追加了证据
        # **step3 内部也会尝试收敛**，所以「这一轮升级了谁」必须在这里一并记下。
        # 只靠下面那个收敛循环会漏掉它们——画像明明立了，报告却说
        # 「收敛为已立 []」，看报告的人会以为收敛逻辑没工作。
        if p.status == PROFILE_ESTABLISHED and before_status != PROFILE_ESTABLISHED:
            stats["established"].append(p.id)

    # 收敛：可能有画像在上一轮拿到足够证据却还没升级（幂等补跑）。
    # **没升级的也要回报原因**：「画像没立」本身不含任何可行动信息，
    # 而「卡在印证不足 2/3」和「卡在情境不一致」要做的事完全不同。
    for p in store.current_profiles(status=PROFILE_PENDING):
        ok, why = converge_detail(store, p, emb=emb)
        if ok:
            if p.id not in stats["established"]:
                stats["established"].append(p.id)
        else:
            stats["blocked"].append({"id": p.id, "topic": p.topic, "reason": why})

    stats["aged"] = age_out_profiles(store)
    stats["capped"] = cap_profiles(store)      # 总量收口：超了降档，不删

    # 渐变检测（阶段 4）：**突变**归 revise_profile、**冷掉**归 age_out_profiles，
    # 中间那条缝——「一直在提，但内容在慢慢变」——归这里。
    # 放在老化之后：刚被老化的画像已经没有「当前版本」，不需要再判漂移。
    from .trend import detect_drift
    stats["drift"] = detect_drift(store, llm)

    # 备忘录：补分类（写入时漏掉的）+ **超期退役**（2026-10-05 起：判"提的窗口过没过"，
    # 替下"窗口 × 3 / 硬顶 90 天"）。
    # 「不永远挂着」很重要：候选池只涨不消，迟早让「有没有该提的事」失去意义。
    from .memo import memo_cycle
    stats["memo"] = memo_cycle(store, llm)
    stats["archived"] = store.archive_all(score_fn=core_score)
    return stats


# ---------------------------------------------------------------------
# 体检（记忆整理稿，2026-09-23）：不靠会话边界的那次整理
# ---------------------------------------------------------------------

def _review_item(store, p: Profile) -> dict:
    """复核材料：陈述 + 印证数 + 最近时间 + **依据**（至少要标题）。

    依据优先取 `evidence_pack`（形成时的快照，最稳）；老数据包为空时退回
    `sources` 现查场景标题（归档的也查得到——`get_scene` 不看 archived）。
    **已删的不列**（引用 = `sources ∩ 现存节点`，2026-09-24）：快照也不再兜底
    ——他删掉的东西不该从复核材料里再露出来（同镜像页 / 展开的口径）。
    只给前 6 条：复核判的是"撑不撑得住"，不是逐条核对。
    """
    proof = [f"「{it.get('title') or it.get('id')}」"
             for it in (p.evidence_pack or [])
             if isinstance(it, dict) and store.exists(str(it.get("id") or ""))]
    if not proof:
        for sid in list(p.sources or []):
            if not str(sid).startswith("S1") or not store.exists(sid):
                continue
            s = store.get_scene(sid)
            if s is not None:
                proof.append(f"「{s.title or sid}」")
    return {"id": p.id, "statement": p.statement, "evidence": p.evidence,
            "last": p.last_support_at or p.valid_at or p.created_at or "",
            "proof": proof[:6]}


def review_profiles(store, llm, cap: int | None = None, min_days: float | None = None,
                    downgrade_cap: int | None = None, emb=None) -> dict:
    """S3 复核（记忆整理稿 §五）：定期回头看「已立」的画像还站不站得住。

    为什么需要（两条现有路都不覆盖的那一类）：`distill_step3` 的
    holds / revise 只在**有新证据**时判、`trend.detect_drift` 只在**算数报警**时判
    ——"没人提它、也没新证据、向量也没动，内容却悄悄不对了"的画像会一直挂着，
    没有出口。这里就是那个出口。

    打分 → 动作（**永不硬删**，L1 硬约束）：
      - `holds` → 什么都不做。**尤其不 bump `last_support_at`**：复核 ≠ 印证，
        去 bump 的话"复核通过"成了续命机制，90 天老化永远不触发
      - `thin` / `stale` → 降档 `pending`（`downgrade_profile`，不删）
      - `reword`（同日补）→ 内容站得住、**写法坏了**（评价词 / 特质标签 / 诊断词）：
        走**修正**（`revise_profile`：旧版进历史、新版 pending，`by="revision"`）
        ——她更新的是"说法"，不是推翻判断；写法判据见 `PROFILE_STATEMENT_RULES`
      - `wrong` → **只记提议**（`profile_reviews` 里 handled=0），人点头才**真删**

    **可算的判据归代码**（2026-09-23）：四道关（印证数 / sources / 同情境 / 跨来源）
    先由 `_gate_recheck` 复查一遍——不过的**直接降档**（记 `regressed`，与她的
    `downgraded` 分开），不占模型名额、不花一次调用。模型只判**语义**
    （依据撑不撑得住 / 写法 / 时间）——那是代码做不了的那半。
    这也是"状态与判据脱节"的出口：S1 被删、来源被摘后，established 可能早已不成立。

    防抖三件：`min_days` 内不重复复核（`last_review_at`）、一次最多看 `cap` 条、
    一次最多**改动** `downgrade_cap` 条（降档 + 重写共享这个预算——都是改库，
    一次别动太多，防模型集体误判；超出的下轮再说——幂等，不着急）。
    """
    cap = int(cap if cap is not None else
              (cfgmod.cfg("maintenance", "review_cap", default=5) or 5))
    min_days = float(min_days if min_days is not None else
                     (cfgmod.cfg("maintenance", "review_min_days", default=7) or 7))
    downgrade_cap = int(downgrade_cap if downgrade_cap is not None else
                        (cfgmod.cfg("maintenance", "review_downgrade_cap", default=2) or 2))

    now = datetime.now()
    cutoff = (now - timedelta(days=min_days)).strftime("%Y-%m-%d %H:%M:%S")
    # 候选：已立 + 距上次复核够久（空 = 从没复核过，最该先看）
    cands = [p for p in store.current_profiles(status=PROFILE_ESTABLISHED)
             if not p.last_review_at or p.last_review_at < cutoff]
    cands.sort(key=lambda p: (p.last_review_at or ""))
    items = cands[:cap] if cap > 0 else []
    if not items:
        return {"checked": [], "downgraded": [], "reworded": [], "suggested": [],
                "verdicts": {}, "details": [], "regressed": []}

    # 前置：**可算的判据让代码复查**（四道关）——门槛值在配置里，模型只能凭常识猜。
    # 不过的直接降档（那是**系统**的动作，记 `regressed` 与她的 `downgraded` 分开），
    # 而且不占这一批的模型名额（省下的留给真正需要判语义的）。
    live: list[Profile] = []
    regressed: list[str] = []
    for p in items:
        ok, why = _gate_recheck(store, p, emb)
        if ok:
            live.append(p)
            continue
        store.downgrade_profile(p.id)
        store.set_profile_reviewed(p.id)
        store.add_profile_review(p.id, "thin", f"系统复查（不过硬判据）：{why}")
        regressed.append(p.id)
    items = live
    if not items:
        return {"checked": [], "downgraded": [], "reworded": [], "suggested": [],
                "verdicts": {}, "details": [], "regressed": regressed}

    data = llm.structured(review_prompt([_review_item(store, p) for p in items]),
                          REVIEW_SCHEMA)
    # 用 id 回显来对应（而不是数组下标）：模型少回一条时不会整体错位
    # （同 classify_window_kinds 的先例）
    by_id = {str(it.get("id") or "").strip(): it
             for it in (data.get("items") or []) if isinstance(it, dict)}

    checked: list[str] = []
    downgraded: list[str] = []
    suggested: list[str] = []
    reworded: list[str] = []
    verdicts: dict = {}
    details: list[dict] = []        # 逐条的「打分 + 理由」——进留痕（"为什么被改"）
    stamp = now_str()
    for p in items:
        it = by_id.get(p.id) or {}
        verdict = it.get("verdict") if it.get("verdict") in REVIEW_VERDICTS else ""
        if not verdict:
            continue        # 没回 / 回了没用的值：当没复核过（下次体检再来）
        reason = str(it.get("reason") or "").strip()
        new_statement = str(it.get("statement") or "").strip()
        if verdict == "reword" and not new_statement:
            continue        # 说重写却没给改写：当没复核过（不写 last_review_at，下次再来）
        checked.append(p.id)
        verdicts[p.id] = verdict
        details.append({"id": p.id, "verdict": verdict, "reason": reason})
        store.set_profile_reviewed(p.id, at=stamp)
        store.add_profile_review(p.id, verdict, reason)
        # 降档 / 重写**共享一个预算**（`downgrade_cap`）：都是改库，一次别动太多
        changed = len(downgraded) + len(reworded)
        if verdict in ("thin", "stale") and changed < downgrade_cap:
            # 降档不是删：记录仍在、仍可召回，只是不再进常驻注入
            store.downgrade_profile(p.id)
            downgraded.append(p.id)
        elif verdict == "reword" and changed < downgrade_cap:
            # 内容对、写法坏了：**修正**（旧版进历史、新版 pending）——
            # 同 step3 / drift 走的那条路（by="revision"），不删任何东西。
            # 来源沿用旧的：复核没有新证据，只是换个说法。
            revise_profile(store, p.id, new_statement, list(p.sources or []))
            reworded.append(p.id)
        elif verdict == "wrong":
            # 她不能自己作废——提议放着，人点一下才走 user_reject_profile
            suggested.append(p.id)
    return {"checked": checked, "downgraded": downgraded, "reworded": reworded,
            "suggested": suggested, "verdicts": verdicts, "details": details,
            "regressed": regressed}


def write_maint_trace(stats: dict) -> None:
    """体检留痕（`data/trace/整理-YYYYMMDD.jsonl`）。

    为什么单独一份（不复用唤醒 trace）：这份回答的是"库里的东西为什么
    被动了"——降了谁、归档了几条、复核打了几分、为什么跑（trigger）。
    `profile_reviews` 存的是**状态**（提议等人处理），这里存的是**日志**。
    写不成不拦改动本身（同各处 trace 的兜底）。
    """
    append_trace("整理", {
        "ts": now_str(), "act": "体检", "trigger": stats.get("trigger") or "",
        "topic_cap": stats.get("topic_cap"),
        "s2_new": stats.get("s2_new") or [], "s3_new": stats.get("s3_new") or [],
        "held": len(stats.get("held") or []),
        "established": stats.get("established") or [],
        "aged": stats.get("aged") or [], "capped": stats.get("capped") or [],
        "drift": stats.get("drift") or {}, "memo": stats.get("memo") or {},
        "archived": stats.get("archived") or {},
        "review": stats.get("review") or {}})


def maintenance_cycle(store, llm, emb=None, trigger: str = "",
                      topic_cap: int | None = None) -> dict:
    """体检（记忆整理稿 §四）：不靠会话边界的那次整理。

    三档里的①（零成本维护：归档 / 老化 / 收口 / 超期放弃 / 补分类 / 补组名）
    与②（主整理）都在 `run_distill_cycle` 里——**直接复用它**（幂等，不复制
    逻辑），只是把逐 topic 的部分收窄成"最该处理的 N 个"；
    ③ S3 复核在这里补上（`review_profiles`）。

    `trigger`：这次为什么跑（"累计 1.2 万字" / "距上次 13 小时"），进留痕——
    没有它，"它昨天为什么整理"就答不出来。
    """
    cap = int(topic_cap if topic_cap is not None else
              (cfgmod.cfg("maintenance", "topic_cap", default=3) or 3))
    stats = run_distill_cycle(store, llm, emb=emb, topic_cap=cap)
    stats["topic_cap"] = cap
    stats["trigger"] = trigger
    stats["review"] = review_profiles(store, llm, emb=emb)
    write_maint_trace(stats)
    return stats


# 硬隐私（**窄名单**）：真的不该由系统主动提的。宁可少锁——
# memo 是"未发生的事"，天然极少是创伤（那些已经写进场景卡了）。
# 2026-09-15 补医疗重症词（「大病要提吗」的答案是**不提**，只记）：
# 提错的风险不对称（他正难受 / 不想谈），而且大病不是"结果没出"——
# 是沉重，没有"问过程"能化解的姿势。他主动说起时她在场、记得，就够了。
_HARD_PRIVATE = ("自杀", "自残", "抑郁", "家暴", "虐待", "创伤",
                 "确诊", "癌", "病危", "去世", "葬礼",
                 "住院", "手术", "化疗", "重症", "病重")


def _raise_mode(content: str) -> int:
    """这条未了结的事「能提到哪一档」：0 正常提（点事）/ 2 只记不提。

    判定**不看场景卡的 sensitive**（2026-09-15 改）：那张卡"拿不准给 true"
    （健康、家事都会标 1），而这个出口原先是"一刀切闭嘴"——源头宽、出口再保守，
    感冒、体检这类最该被问候的事全被锁死。
    现在只认**硬隐私**（窄名单）：其余一律能提——"点出事情本身"
    （「今天面试怎么样」「感冒好了没」）本来就在允许的"主动关心"里
    ——那是给门，不是揭开。敏感只调节**措辞的力度**（问过程不问结果、
    不端已知细节），这个分寸是渲染层的通用指引，不做 per-memo 降档。
    """
    if any(w in (content or "") for w in _HARD_PRIVATE):
        return 2
    return 0


def _write_memos(store, scene: Scene, llm=None) -> list[str]:
    """open_loops → memos（可提醒子集），并给「无具体时间」的补上分类与窗口。

    `due_at` 只在用户**明说了时间**时才有值（那是转述，不是估算）；
    没有的话交给 `classify_window_kinds` **只分类**
    （deadline / progress / idea / followup），周期由系统按类别映射——
    **不让模型估天数**（见 memo.py 的纪律 1）。
    """
    ids = []
    created: list[Memo] = []
    for loop in scene.open_loops or []:
        content = str((loop or {}).get("content") or "").strip()
        if not content:
            continue
        kind = (loop or {}).get("kind")
        if kind not in MEMO_KINDS:
            kind = MEMO_USER_TASK
        memo = Memo(
            scene_id=scene.id, content=content, kind=kind,
            # 钩子的稳定编号（2026-10-05）：闭合/退役/变更靠它定位，不再靠文字全等
            # （编号由 `store.add_scene` 生成）；老数据为空 → store 那边退回全等匹配。
            loop_id=str((loop or {}).get("loop_id") or "").strip(),
            due_at=str((loop or {}).get("due_at") or "").strip(),
            # 组名原样带过来（同一件事的多步共用一个名字，2026-09-22）——
            # 它只影响呈现与提醒收拢，判重/闭合仍按单条走。
            group_name=str((loop or {}).get("group_name") or "").strip(),
            status=MEMO_PENDING,
            sensitive=_raise_mode(content),   # 提档：0 点事 / 2 只记不提
            created_at=now_str(),
        )
        ids.append(store.add_memo(memo))
        created.append(memo)

    # 分类只对「没有 due_at」的那些做；没有这类 memo 时函数会直接返回，
    # 不产生 LLM 调用——写入路径上不该为「没事发生」多付一次钱。
    if llm is not None and created:
        from .memo import classify_window_kinds    # 延迟 import：写入路径不必拖上整个 memo 模块
        classify_window_kinds(store, created, llm)
    return ids
