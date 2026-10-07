"""唤醒引擎——「该不该翻记忆、翻什么、翻多深」。

像人一样被唤起（被线索触发，不是主动搜索），但是**正常人的唤起**
（有选择、受控、会衰减，不是闪回和反刍）。

流程：`compute_cues`（算七条线索）→ `recall`（线索耦合出动作）→ `rank`（核心度排序）
→ 抑制取前 N → 注入。每一次唤醒落一行 trace（§5.5）——记忆系统最难的是
「为什么召回了这个」不可见，不留痕就没法调试，也看不出 air 的判断是否正常。

三条容易写歪的地方，先在这里说清：
  1. **R0 默认倾向「翻」**：该翻没翻（失忆、答非所问）不可挽回；
     不该翻翻了只是注入噪音，有抑制机制兜底。代价不对称，所以默认翻。
  2. **不给 valence 额外权重**：情绪只管强弱（arousal），不偏好负面——
     抑郁式唤醒的特征正是"只想起坏的"，那是要排除的病态模式。
  3. **规则版不引用 `theta` / 耦合权重矩阵**：两者数学等价，规则是它的可读展开。
     `theta` / `w` 是二期拟合的入口，第一版别去调它们。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L7 读取侧（唤醒）
#   上游    ：config、embedding、entity（旁路）、model、prompts（线索判定）、store
#   下游    ：chat（每轮唯一入口）、dashboard（把线索和抑制名单画出来）、
#             distill / shortterm（延迟 import `core_score` 与 `bump_counters`）
#   对外入口：`recall_for_message`（一个函数走完全程）/ `compute_cues` / `recall` /
#             `core_score` / `rank` / `mark_mentioned` / `cue_hits`（给前端算高亮）
#             / `cue_votes`（票制：强 2 / 弱 1，2026-10-05）
#   边界    ：**读取侧**——唯一会写的是两个计数器，且由调用方判断该不该记
# ---------------------------------------------------------------------
# 本文件分段
#   段 0  纯函数 —— 字符重叠兜底 / 衰减 / core_score / rank
#   段 1  compute_cues —— 七条线索（C1–C7）+ 两条旁路（实体 / 字面）
#   段 2  r0_should_recall / cue_hits / multi_hit —— 门控与分级
#   段 3  recall —— 规则版耦合（C1–C7 → R1–R6）
#   段 4  bump_counters —— 两个指标 + 老化时间戳
#   段 5  trace —— 唤醒留痕 + 高层入口 recall_for_message
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from datetime import datetime

from . import config as cfgmod
from .embedding import cosine
from .entity import match_known_entities, recall_by_entities
from .model import PROFILE_ESTABLISHED, Scene
from .prompts import CUE_SCHEMA, cue_prompt
from .store import now_str

# R0 触发特征（第一版规则）。
# 为什么用「特征词」而不是小模型：规则的可解释性在这里比准确率重要——
# 误判时你打开 trace 就知道是哪条规则点的火，小模型只给一个分。
_PRONOUNS = ("它", "那个", "那件", "上次", "这次", "他", "她", "他们", "她们", "这事", "那事")
_TIME_WORDS = ("昨天", "今天", "明天", "上周", "下周", "上个月", "下个月", "之前", "后来",
               "当时", "那天", "刚才", "以前", "后来", "最近", "当年", "前阵子")
_EMOTION_WORDS = ("难受", "开心", "烦", "累", "焦虑", "生气", "高兴", "害怕", "委屈",
                  "压力", "崩溃", "松了", "难过", "沮丧", "兴奋", "担心", "放心")
_FIRST_PERSON = ("我", "咱", "我们")


# ---- 段 0：纯函数 ----

def char_overlap(a: str, b: str) -> float:
    """字符 bigram Jaccard 相似度——embedding 不可用时的兜底。

    为什么不是「包含关系」或「词重叠」：bigram 对中文更稳（中文没有空格分词），
    且对语序不敏感。它不如向量准，但**方向是对的**（同主题的文本重叠更多），
    够 R0/R1 在降级状态下做出不比"完全失忆"更差的选择。
    """
    if not a or not b:
        return 0.0

    def grams(s: str) -> set:
        s = "".join(s.split())
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return inter / len(ga | gb)


def _activity_time(scene: Scene) -> str:
    """这条场景的「最近活跃时间」：被提及 > 事件时间 > 记录时间。"""
    return scene.last_mention_at or scene.time_event or scene.created_at or ""


def _decay_since(ts: str, halflife_key: str, default_days: float,
                 now: datetime | None = None) -> float:
    """按半衰期算「还剩多少新鲜」（0–1）。

    用指数衰减而不是线性：淡忘是前快后慢的，线性会让"三年前"和"三个月前"
    的差别小得没有意义。

    **取不到时间戳 → 返回 1.0（不打折）**：时间戳缺失是"不知道什么时候"，
    不是"很久以前"。按最老处理会静默地把一批记录压到队尾，那是对缺失数据的猜测。
    """
    if not ts:
        return 1.0
    try:
        t = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 1.0
    days = max(0.0, ((now or datetime.now()) - t).total_seconds() / 86400)
    hl = float(cfgmod.cfg("rank", halflife_key, default=default_days) or default_days)
    return 0.5 ** (days / hl) if hl > 0 else 1.0


def _recency_decay(scene: Scene, now: datetime | None = None) -> float:
    """时间衰减：半衰期 `recency_halflife_days`（默认 30 天）。"""
    return _decay_since(_activity_time(scene), "recency_halflife_days", 30, now)


def core_score(scene: Scene) -> float:
    """核心度（存储层 §4）——决定**同一动作内**谁优先、谁先进冷层。

        core = w_layer + w_int×intensity + w_imp×cited_by_profile

    三个刻意的处理：
      - `cited_by_profile` 是"不设上限的重要性"，但进公式要**饱和映射**：
        不饱和的话一条被引用 50 次的场景会永远霸占第一位，
        等于把"重要"写死成"历史累计量"，新记忆再也没机会上来。
      - **不给 valence 任何权重**，**arousal 也不进**（它只触发 R3 与深度预算）。
      - **recency 不进这里**（2026-10-05 摘出去）：新鲜度改在排序侧**单列一个键**
        （`order()` 四键的第三键，见下）。理由：**一个信号只在一处表达**——
        埋在核心度里它和"强度/重要性"混成一个数，「差一票时谁赢、赢多少」答不出来。
        核心度从此只回答"这条有多重"，"有多新"由新鲜度那一键回答（存储层 §四 同日注）。
    """
    w = cfgmod.cfg("rank", default={}) or {}
    w_layer = (w.get("w_layer") or {}).get("S1", 0.3)
    cited = max(0, int(scene.cited_by_profile or 0))
    cited_sat = 1.0 - pow(2.718281828, -cited / 3.0) if cited else 0.0
    return (w_layer
            + float(w.get("w_intensity", 0.4)) * float(scene.intensity or 0.0)
            + float(w.get("w_imp", 0.3)) * cited_sat)


def rank(scenes: list[Scene]) -> list[Scene]:
    """按核心度从高到低排序（同一动作内的排序，见存储层 §4）。

    `mention_count` **不参与排序**——它只用于防反刍（上限 + 连续注入降权）。
    把念叨次数当重要性，正是设计稿要排除的"反刍"。
    """
    return sorted(scenes, key=core_score, reverse=True)


# ---- 段 1：七条线索 ----


def _literal_hits(msg: str, store) -> dict[str, list[str]]:
    """**字面旁路**：这句话里的「罕见词」命中了哪些场景（2026-10-05 加，唤醒层 §五/§八）。

    与实体旁路并列的第二条旁路。补的洞：C1 是向量、**不含字面**；实体旁路只认
    库里"带个人关系的名字"——一般术语（「验收」「脱敏脚本」）原来没有任何字面通道，
    只在向量服务挂掉时才被 `char_overlap` 兜住。

    两条规矩：

    - **取词**：2–4 字的连续片段（中文不分词，与 `char_overlap` 的 bigram 同思路、
      片段加长）。不查词表——**"哪些词重要"是数出来的，不是列出来的**。
    - **罕见度**：片段在库里的**出现场景数 ≤ `recall.literal_max_df`**（初值 3）才算命中。
      这一条同时替掉了停用词表：「我们」「的时候」满库都是，自然被挡在门外；
      「面试」也一样（满库都有），而「脱敏脚本」只有一个场景有 → 算。
      这是"关键词要真强"的实现——不稀有的词命中一堆场景，那是噪声不是线索。

    一条消息最多带 `recall.literal_max_words`（初值 3）个词出去：防"整句话都命中"
    把候选池灌满。常见词在数到 `max_df+1` 时就提前放弃（不必扫完整个库）。

    ⚠️ **已知边界**（先说清，别等出问题才查）：扫的是 `query_scenes(limit=500)`
    ——**最近 500 条**。库比这更大时，只有很老的场景里出现过的词捞不到
    （"罕见度"也只在窗口内数）。先跑着看：真有这个量级，再谈按需扩窗或建
    字面索引（现在加索引是给一个还没出现的问题付钱）。
    """
    text = " ".join((msg or "").split())
    if len(text) < 2:
        return {}
    max_df = int(cfgmod.cfg("recall", "literal_max_df", default=3) or 3)
    max_words = int(cfgmod.cfg("recall", "literal_max_words", default=3) or 3)
    frags: list[str] = []
    for n in (4, 3, 2):                 # 长的优先：长的更具体、更可能罕见
        for i in range(len(text) - n + 1):
            frags.append(text[i:i + n])
    frags = list(dict.fromkeys(frags))[:120]      # 去重 + 封顶（长消息不做全文扫描）
    scenes = store.query_scenes(limit=500)
    out: dict[str, list[str]] = {}
    for frag in frags:
        if len(out) >= max_words:
            break
        sids: list[str] = []
        for s in scenes:
            hay = f"{s.title or ''} {s.text or ''} {s.topic or ''}"
            if frag in hay:
                sids.append(s.id)
                if len(sids) > max_df:       # 常见词：不是"罕见"，放弃
                    break
        if sids and len(sids) <= max_df:
            # **长的吃掉短的**：同一个词的各种切片别各算一个"词"——否则一个术语
            # 就吃掉三格名额（"脱敏脚本"命中后，"脱敏脚"/"敏脚本"/"脱敏"都是它），
            # 还会把候选池里同一件事连记三次。`out` 里只有先前收下的（更长）片段。
            if any(frag in w for w in out):
                continue
            out[frag] = sids
    return out


def compute_cues(msg: str, store, emb=None, llm=None) -> dict:
    """算查询侧七条线索（C1–C7）+ **两条旁路**（实体 / 字面）。

    LLM 只做**一次**轻量调用判 C2/C3/C5/C7（四次调用换四个封闭式判断不值当）；
    C1/C4 用 embedding；C6 查重。**C7 未决性不落存储字段**——
    它是「当前这句话」的属性，不是历史场景的属性（两侧对称的含义）。
    """
    emb_map = store.all_embeddings()
    msg_vec = emb.embed_one(msg) if (emb is not None and msg is not None) else None

    # C1 内容/语义：与全库的最高相似度。embedding 不可用 → 字符重叠兜底。
    c1 = 0.0
    if msg_vec:
        for _, vec in emb_map:
            c = cosine(msg_vec, vec)
            if c > c1:
                c1 = c
    else:
        for s in store.query_scenes(limit=200):
            c1 = max(c1, char_overlap(msg, f"{s.title} {s.text}"))

    # C4 自我相关性：与「关于这个人」的共鸣度，**设上限**防自我相关过度主导。
    # 一期用 subject='user' 的场景近似画像库（画像本身没存向量，二期再补）。
    c4 = 0.0
    user_ids = {s.id for s in store.query_scenes(subject="user", limit=500)}
    if msg_vec:
        for sid, vec in emb_map:
            if sid in user_ids:
                c4 = max(c4, cosine(msg_vec, vec))
    cap = float(cfgmod.cfg("recall", "self_relevance_cap", default=0.7))
    c4 = min(c4, cap)

    # C6 新鲜度：这条消息讲的东西，库里有过没有（1 − 最大相似度）。
    # 降级时用字符重叠兜底（与 C1 同源）：不兜底它恒为 1.0，而「聊过」的判据是
    # 它低于 `seen_threshold`——降级时这一维永远不命中，于是
    # 「你们聊过这个」这个提示在没有向量服务时彻底消失（降级变哑，不是变糙）。
    c6 = 1.0
    if msg_vec and emb_map:
        c6 = min(1.0, 1.0 - max((cosine(msg_vec, v) for _, v in emb_map), default=0.0))
    else:
        # 库里没有向量时 `c1` 是 0（上面的循环没跑），这里自然还是 1.0 = 全新。
        c6 = min(1.0, 1.0 - c1)

    judge = llm.structured(cue_prompt(msg), CUE_SCHEMA) if llm is not None else {}
    valence = judge.get("valence")
    if valence not in (-1, 0, 1):
        valence = None
    arousal = judge.get("arousal")
    if arousal not in (0, 1):
        arousal = None

    return {
        "C1": round(float(c1), 4),
        "C2": judge.get("tense") or "now",
        "C3": {"valence": valence, "arousal": arousal},
        "C4": round(float(c4), 4),
        "C5": 1 if judge.get("about_relation") else 0,
        "C6": round(float(c6), 4),
        "C7": 1 if judge.get("unresolved") else 0,
        "entities": match_known_entities(msg, store),
        # 字面旁路（2026-10-05）：{罕见词: [场景号…]}——强维，与实体旁路同级
        "literal": _literal_hits(msg, store),
        "msg_emb": msg_vec,
        # 原话留在 cues 里（内部字段，不进 trace）：降级时相关性只能靠它现算
        # 字符重叠，**不能指望调用方记得再塞一次**。以前只有 `recall_for_message`
        # 在外面补这一刀，于是任何直接调 `recall(cues, ...)` 的路径在降级态下
        # 相关性恒为 0、R1/R2/R3 全空——静默失忆，且没有断言拦得住。
        "_msg": msg or "",
        # 降级标记：后面所有阈值判断都要看它。带下划线 = 内部字段，不进 trace。
        "_degraded": msg_vec is None,
    }


# ---- 段 2：门控与分级 ----
#   `_c1_line` / `_c6_line` 阈值收在一处（跟着降级状态走）｜`cue_hits` 逐维判命中
#   `cue_votes` 票制（强 2 / 弱 1）｜`multi_hit` 票数 ≥k 且含强维 → 强信号（才给 S0 原文）
#   ｜`_weak_ok` 是 R2/R3 的相关门

def _c1_line(cues: dict) -> float:
    """C1「明确命中」的阈值——**降级时用低得多的线**。

    字符重叠的量级天生比余弦低：短查询命中一两个 bigram 就到 0.05 量级，
    拿向量的 0.40 去卡它，等于「降级时 R1 永远不触发」——
    那降级就不是降级，是失忆。降级状态下相关性变糙，但方向还在，
    宁可多点噪声（有 rank + 预算兜底），也不要一条都召不回来。
    """
    degraded = bool(cues.get("_degraded"))
    key = "degraded_c1_threshold" if degraded else "c1_threshold"
    return float(cfgmod.cfg("recall", key, default=0.03 if degraded else 0.4))


def _c6_line(cues: dict) -> float:
    """C6「聊过」的线——**降级时也跟着降**。

    字符重叠量级天生低（见 `_c1_line`），`1 − 重叠` 因此挤在 1.0 附近：
    拿向量的 0.30 去卡它，降级时这一维永远不会命中。
    降级时唯一诚实的尺子就是 C1 那条线——**重叠到了「明确命中」的程度就算聊过**。

    ⚠️ 降级时 C6 与 C1 是**同一个信号**（都来自字符重叠）。所以它只用于
    「你们聊过这个」那个 flag；`multi_hit` 里降级时**不计这一维**——
    把一维算两遍，「多维协同」就名不副实了。
    """
    if cues.get("_degraded"):
        return 1.0 - _c1_line(cues)
    return float(cfgmod.cfg("recall", "seen_threshold", default=0.3))


def _relevance(scene: Scene, cues: dict) -> float:
    """这条场景与当前消息的相关度（降级时走字符重叠）。

    它是「弱相关」这道门的尺子，不参与排序：
    排序仍然用核心度（存储层 §4），这里只负责把「八竿子打不着」的挡在外面。
    """
    msg_vec = cues.get("msg_emb")
    if msg_vec and scene.emb:
        return cosine(msg_vec, scene.emb)
    return char_overlap(cues.get("_msg", ""),
                        f"{scene.title} {scene.text} {scene.topic}")


def _weak_ok(scene: Scene, cues: dict) -> bool:
    """弱相关过滤（R2/R3 用）。

    为什么需要它：`C2=过去` 会拉高 R2，但「过去」本身不含任何"相关"信息——
    不过滤的话 R2 就是「把库里最近的 N 条全捞出来」，库一大就变成噪声注入。
    设计稿说 R2 是「按时间戳找过去**的相关**场景」，这个函数就是那句"相关"的落点。
    """
    return _relevance(scene, cues) >= _c1_line(cues)


def r0_should_recall(msg: str, entities: list[str] | None = None) -> bool:
    """R0 轻量感知：要不要翻记忆（所有输入都跑的地板）。

    **默认倾向翻**：含指代 / 时间词 / 情绪词 / 实体名 / 第一人称 → 翻；
    一条个人特征都没有 → 不翻（那多半是纯知识问答）。
    代价不对称（该翻没翻不可挽回，多翻有抑制兜底），所以门槛要低。
    """
    text = msg or ""
    if entities:
        return True
    if any(w in text for w in _PRONOUNS):
        return True
    if any(w in text for w in _TIME_WORDS):
        return True
    if any(w in text for w in _EMOTION_WORDS):
        return True
    if any(w in text for w in _FIRST_PERSON):
        return True
    return False


def cue_hits(cues: dict) -> dict[str, bool]:
    """逐维的「明确命中」判定——**阈值的唯一落点**。

    各维判据（唤醒层 §8）：
      C1 语义 > c1_threshold；C2 非 now（明确指向过去/未来/假设）；
      C3 arousal=1；C4 > cap；C5 为真；C6 < seen_threshold（聊过）；C7 为真。
    阈值的取法跟着降级状态走（见 `_c1_line`），否则降级时永远凑不出多维协同。

    **为什么单独抽出来**：这个判定有两个消费方——`multi_hit`（要不要给 S0）
    和仪表盘（哪几维该高亮）。以前仪表盘在前端**抄了一份阈值**
    （`C1>0.4 / C4>0.7 / C6<0.3`）：配置一改两处就漂，降级时（0.03）
    前端更是全都亮不起来。判定收在这里，两边读同一份结果。
    """
    c1_line = _c1_line(cues)
    seen_line = _c6_line(cues)
    cap = float(cfgmod.cfg("recall", "self_relevance_cap", default=0.7))
    return {
        "C1": cues.get("C1", 0) > c1_line,
        "C2": cues.get("C2") != "now",
        "C3": (cues.get("C3") or {}).get("arousal") == 1,
        "C4": cues.get("C4", 0) > cap - 1e-9,
        "C5": bool(cues.get("C5")),
        # 降级时 C6 **不算命中**：它与 C1 同源（都是字符重叠），
        # 算进来等于把一维当两维，「多维协同」这个判据就失真了。
        # 这一维此时不是"没命中"，是"算不出来"——算不出来就该不计，而不是计 0。
        "C6": (not cues.get("_degraded")) and cues.get("C6", 1) < seen_line,
        "C7": bool(cues.get("C7")),
        # 两条旁路也各算"一个维度"（2026-10-05）：它们有票（强维 2 票），
        # 也要能高亮——仪表盘读的就是这份判定。
        "entity": bool(cues.get("entities")),
        "literal": bool(cues.get("literal")),
    }


# ---- 票制（2026-10-05，唤醒层 §八）----
# 强维 = 语义 / 字面命中 / 实指名（2 票）；弱维 = C2–C7（1 票）。
# 为什么要有票、不做七档系数：规则版要可读可调试（`recall()` 注释：加权和与规则版
# 数学等价，权重矩阵等有数据再拟合）；整数两档是这条路的延续。
_STRONG_DIMS = (("C1", 2), ("entity", 2), ("literal", 2))
_WEAK_DIMS = (("C2", 1), ("C3", 1), ("C4", 1), ("C5", 1), ("C6", 1), ("C7", 1))
# ⚠️ 「编号直查」不在这一套里：它是**工具侧**的直取（`memory_search` 给 id，
# 不排序、不比较），强维票只描述"唤醒这一路"。


def cue_votes(cues: dict) -> dict:
    """这一轮的**票数**与"有没有强维"（唤醒层 §八，2026-10-05）。

    原来数"维数"（≥2 维就算强）：「语义 + 情绪」和「实体 + 时间」一样算 2 维——
    前者是一条证据链，后者只是两个弱信号凑数。

    - **票按维去重**：这里按维度名记账，同一维重复命中只计一次
      （"沿连边扩散"是连带想起，不加票）；
    - 降级时 **C6 与字面旁路都不计票**：它们和 C1 同源（都退化成字符重叠），
      一维算两遍会让"多维协同"名不副实（同 2026-09-22 对 C6 的处理）。
    """
    h = cue_hits(cues)
    degraded = bool(cues.get("_degraded"))
    votes, strong = 0, False
    for dim, w in _STRONG_DIMS + _WEAK_DIMS:
        if not h.get(dim):
            continue
        if degraded and dim in ("C6", "literal"):
            continue
        votes += w
        if w == 2:
            strong = True
    return {"votes": votes, "strong": strong}


def multi_hit(cues: dict) -> bool:
    """强信号判定（唤醒层 §8）：**票数 ≥ `recall.signal_votes` 且含至少一个强维**。

    2026-10-05 由"命中维度数 ≥ k（等权）"改成票制——两个弱维就能凑出"多维"。
    现在门槛 3 票：2 票可以是"一个强维单打"（语义命中一条就走），也可以是
    "两个弱维凑数"（情绪 + 时间）——两种都不该给原文（R5 是最贵的东西）；
    3 票把这两种都排除，只剩"强 + 弱"与"强 + 强"。

    **门槛只从配置读**（不接参数）：上一版的 `k` 参数在改票制时含义悄悄从
    "维数"变成"票数"——没人传它，留着就是个坑（同「一个旋钮只写一处」）。
    """
    v = cue_votes(cues)
    k = int(cfgmod.cfg("recall", "signal_votes", default=3) or 3)
    return v["votes"] >= k and v["strong"]


# ---- 段 3：规则版耦合 ----

def _top_by_embedding(cues: dict, store, n: int) -> list[tuple[Scene, float]]:
    """R1 的主题检索：语义 top-n（降级时走字符重叠）。"""
    msg_vec = cues.get("msg_emb")
    scored: list[tuple[Scene, float]] = []
    if msg_vec:
        for sid, vec in store.all_embeddings():
            scored.append((sid, cosine(msg_vec, vec)))
    else:
        for s in store.query_scenes(limit=200):
            scored.append((s.id, char_overlap(cues.get("_msg", ""), f"{s.title} {s.text}")))
    scored.sort(key=lambda x: x[1], reverse=True)

    out = []
    for sid, score in scored[:n]:
        scene = store.get_scene(sid)
        if scene is not None:
            out.append((scene, score))
    return out


def profile_decay(p, now: datetime | None = None) -> float:
    """画像的时效衰减（0–1）：半衰期 `rank.profile_recency_halflife_days`（默认 45 天）。

    时间的起点是「上一次它还被印证或被提及」（`last_support_at`），没有才退到
    `valid_at` / `created_at`——**"最近有没有人再提它"比"它什么时候立的"更能说明它活着**。

    取老化期（`distill.stale_days`=90）的一半当半衰期，是一条刻意的安排：
    **衰减先挪位置，老化才请出常驻**。一条画像 90 天没人印证也没被提到，
    在这 90 天里它的排位权重一路降到 1/4（让更新、更活的那几条上来），
    到第 90 天才由 `age_out_profiles` 降回 pending。
    两条路串行，不会出现"还没来得及被挤到后面、就已经被请出去了"那种断档。
    """
    ts = p.last_support_at or p.valid_at or p.created_at or ""
    return _decay_since(ts, "profile_recency_halflife_days", 45, now)


def _pick_profiles(store, msg: str, emb, status: str | None,
                   n: int | None = None) -> list:
    """挑这一轮常驻的画像：**一半靠印证，一半靠情境**。

    全量注入是不行的——topic 会一直长，画像跟着长，迟早把上下文塞满。
    但**只按印证排**又会漏掉「此刻正好相关的那条」（那才是"想起来"的样子）。
    所以一半一半：
      - 前半：**印证最多的**——长期稳定，回答"他是谁"
      - 后半：**跟这句话最像的**——此刻相关，回答"此刻该想起哪条"

    上限 `recall.inject_profiles_n`（默认 10）。**没被选中的不是丢掉**：
    它们仍在库里，镜像页看得到、`memory_search` 查得到，只是这一轮不常驻。

    **前半（靠印证的那几个名额）按「衰减后的印证数」排**（`profile_decay`）：一条半年没人印证、
    也没被提到的画像不该一直占着前排——它不是错的，只是"上一次它还活着"是很久以前。
    **另一半（靠情境）不衰减**：那半是按「此刻相不相关」挑的，
    相关性本身就是新鲜的证明，再乘衰减等于同一件事罚两次。
    """
    n = n or int(cfgmod.cfg("recall", "inject_profiles_n", default=10))
    all_p = store.current_profiles(status)
    if len(all_p) <= n:
        return all_p

    by_ev = sorted(all_p, key=lambda p: (-((p.evidence or 0) * profile_decay(p)),
                                        p.valid_at or ""))
    qv = emb.embed_one(msg) if (emb is not None and msg) else None
    if qv is None:
        # 没有情境可比 → 就按印证取前 n（**绝不能按创建顺序**，
        # 那样"印证最少"的会因为建得早混进来）
        return by_ev[:n]

    steady = by_ev[:n // 2]
    taken = {p.id for p in steady}
    rest = [p for p in by_ev if p.id not in taken]
    want = n - len(steady)
    # **批量算一次**，不是逐条：`embed_one` 一次一条的话，几条画像就是几次往返。
    # `embed` 是"全或无"——失败就整批拿不到，此时情境这半落空（退回只按印证），
    # 与逐条全失败的结果一样（`embed_one` 走的是同一个服务）。
    pvs = emb.embed([p.statement or "" for p in rest]) if rest else None
    scored = []
    if pvs and len(pvs) == len(rest):
        for p, pv in zip(rest, pvs):
            if pv:
                scored.append((p, cosine(qv, pv)))
    scored = [x for x in scored if x[1] > 0]
    scored.sort(key=lambda x: -x[1])
    return steady + [p for p, _ in scored[:want]]


def recall(cues: dict, store, emb=None) -> dict:
    """按线索执行唤醒动作（规则版，逐条对应唤醒层 §7 的耦合表）。

    耦合公式 `W_r = base_r + Σ(score_c × w_{c,r})` 与这里的规则**数学等价**：
    规则版每条自带阈值，可读、可调试、可观测；有数据后再拟合权重矩阵。
    **第一版不引用 `theta` / `w`**（见模块 docstring 第 3 条）。

    返回结构给上层（对话层 / demo）用：命中的场景、常驻画像、要不要下钻原文，
    以及「你们聊过这个」这类提示 flag。
    """
    # `stage` = **整轮**卡在哪一步（`""` / `"R0"` / `"weak_gate"`）。
    # 它和 `suppressed_detail` 是互补的两层：后者逐条说"哪条没进来、为什么"，
    # 前者在**一条都没进来**时一句话说清卡在哪——而早退那一路连候选都没产生，
    # 逐条的名单答不了它。
    result = {"scenes": [], "profiles": [], "summaries": [], "raws": [], "actions": {},
              "flags": {"hint_talked_before": False},
              "suppressed": [], "suppressed_detail": [], "stage": "", "why": {}}

    r0 = r0_should_recall(cues.get("_msg", ""), cues.get("entities"))
    result["actions"]["R0"] = 1 if r0 else 0
    if not r0:
        # 信号不足：只注入常驻画像就收工（R4 是地板，不是可选项）
        # 判据不写进 trace：它是 `r0_should_recall` 那几条规则，每次都一样，
        # 每行重复一遍只是把噪声写进日志。要查去看那个函数。
        result["stage"] = "R0"
        # 常驻画像**只带「已立」**：`pending`（还没印证够的猜测）不进注入（不武断）
        result["profiles"] = _pick_profiles(store, cues.get("_msg", ""), emb,
                                            PROFILE_ESTABLISHED)
        # R4 既然真的执行了，动作表里就该有它（早退不等于没做）
        result["actions"]["R4"] = 1.0
        return result

    # 这两处的兜底值须与 `config` 同值（只在键缺失时生效；不一致即漂移——
    # 曾经是 5 / 2 两个旧值，与 config 的 2 / 3 已经对不上）
    budget = int(cfgmod.cfg("recall", "inject_n", default=2))
    if (cues.get("C3") or {}).get("arousal") == 1:
        # R3 命中 → 「上调深度预算」：条数上限加 arousal_extra_n
        budget += int(cfgmod.cfg("recall", "arousal_extra_n", default=3))

    # 候选扫描宽度（R2/R3 共用）：先宽进、再过门
    scan_n = int(cfgmod.cfg("recall", "scan_n", default=60))
    c1_line = _c1_line(cues)
    found: dict[str, Scene] = {}
    why: dict[str, str] = {}
    tier: dict[str, int] = {}
    # 票数（§3.1 排序第一键）：**强维 2 / 弱维 1，按维去重**（2026-10-05 由"被几条
    # 线索捞到"的等权计数改来）。`votes` 是那个场景当前的票，`dims` 记它已经计过票的
    # 维度名——同一维重复命中只计一次（扩散算"连带想起"，vote=0）。
    votes: dict[str, int] = {}
    dims: dict[str, set] = {}
    # 被相关性门挡在外面的候选（**不进 `found`**，所以也不在任何名单里）
    blocked: list[dict] = []

    def gate(scene: Scene, action: str) -> bool:
        """过弱相关这道门——**过没过都要留痕**。

        为什么不直接用 `_weak_ok`：它是纯函数（好测，也不该知道 trace 这种事），
        而留痕要"这次是哪个动作在问"这个上下文。挡住的东西不留名，
        「卡在哪一步」就永远只能答后半句。
        """
        if _weak_ok(scene, cues):
            return True
        # 比较规则仍然只有 `_weak_ok` 一处；这里重算一次只为了把"差多少"写进留痕。
        rel = _relevance(scene, cues)
        blocked.append({
            "id": scene.id, "title": scene.title, "stage": "weak_gate",
            "action": action,
            "why": f"{action} 候选，相关度 {rel:.3f} < 门线 {c1_line:.3f}",
            "rel": round(rel, 4), "line": round(c1_line, 4),
            "gap": round(c1_line - rel, 4),
            "core": round(core_score(scene), 3),
            # 没进过排序就没有证据层级。给 `None` 不给 9——9 是"查不到"的默认值，
            # 写进留痕就成了"它是最弱那一层"这种假事实（§四：不用 0 顶替）。
            "tier": None,
        })
        return False

    def add(scene: Scene, reason: str, level: int = 9,
            dim: str = "", vote: int = 0):
        """收一个候选。`level` 越小 = 证据越强；**票数记在这一处**。

        排序四键（2026-10-05 由三键改来，工具箱稿 §3.1）：**票数 → 线索层级
        → 新鲜度 → 核心度**。第一键靠 `votes` 记账——**票按维去重**（`dims` 记
        这个场景已经计过票的维度名）：同一条被语义 + 情绪两条路捞到，比只被
        一条路捞到更该先给；而"沿连边扩散"是连带想起，**`vote=0`、不加票**。
        票的权重（强维 2 / 弱维 1）在 `cue_votes` 那一套里定义，这里只按调用方
        给的 `vote` 记账——一处定义、一处使用。

        **跨动作的排序为什么按 level 分层**：多个动作的候选最后会并成一个池子，
        而核心度（存储层 §4）里**不含相关性**——「被语义直接命中」和
        「因为同是负面情绪被顺手捞上来」在核心度上分不出高下。
        所以跨动作用「证据层级」先分层（层内按新鲜度 → 核心度排）。
        层级依据唤醒层 §7 的耦合强度：语义 / 字面 / 实体是直取，时态 / 情绪是辅助联想。
        """
        if scene is None:
            return
        if dim and dim not in dims.setdefault(scene.id, set()):
            dims[scene.id].add(dim)
            votes[scene.id] = votes.get(scene.id, 0) + vote
        if scene.id not in found:
            found[scene.id] = scene
            why[scene.id] = reason
            tier[scene.id] = level
        elif level < tier.get(scene.id, 9):
            tier[scene.id] = level
            why[scene.id] = reason

    def order() -> list[Scene]:
        """四键排序（§3.1，2026-10-05）：票数 → 线索层级 → 新鲜度 → 核心度。

        新鲜度单列的理由（他要的两条互搏规则）：**票多者先**（一条旧记忆被多个
        维度同时命中，赢过刚发生的一条）、**同票比新鲜**（同一个关键词命中，
        新的排在旧的前面）。它原来埋在核心度里、和"强度/重要性"混成一个数，
        差一票时谁赢、赢多少，回答不出来。
        """
        return sorted(found.values(),
                      key=lambda s: (-votes.get(s.id, 0), tier.get(s.id, 9),
                                     -_recency_decay(s), -core_score(s)))

    # R1 主题检索（C1 明确命中；C2 未来/假设也走它）+ 沿 causality 扩散一层
    c1_hit = cues.get("C1", 0) > c1_line
    if c1_hit or cues.get("C2") in ("future", "hypo"):
        for scene, score in _top_by_embedding(cues, store, budget * 2):
            if gate(scene, "R1"):
                add(scene, f"R1 语义命中({score:.2f})", level=0, dim="C1", vote=2)
        # 扩散：只在已命中项上走一层（不加预算，扩散是"连带想起"不是"多搜一轮"）。
        # 相邻关系**现算**（`topic_neighbors`，2026-09-24）——原 causality 边已退役：
        # 它是"同 topic 上下一条"的冗余存储，能派生就不存（存储层稿 §五）。
        # **不加票**（2026-10-05）：连带想起不是一条独立证据，给了票就等于给扩散开了后门。
        for sid in list(found.keys()):
            for nb in store.topic_neighbors(sid):
                nb_scene = store.get_scene(nb)
                if nb_scene is not None and not nb_scene.archived:
                    add(nb_scene, "R1 沿 causality 扩散", level=0)
        result["actions"]["R1"] = round(min(1.0, cues.get("C1", 0)), 3)

    # R2 时间回溯（C2 = past）。**要过弱相关这道门**——「过去」本身不含相关性，
    # 不过滤就是「把最近的 N 条全捞出来」，库一大就成了噪声注入。
    if cues.get("C2") == "past":
        # 宽度用 `scan_n` 不跟 `inject_n` 走（「看多少」≠「带几条」，理由见 config）
        for scene in store.query_scenes(until=now_str(), limit=scan_n):
            if gate(scene, "R2"):
                add(scene, "R2 时间回溯", level=2, dim="C2", vote=1)
        result["actions"]["R2"] = 1.0

    # R3 情绪匹配（arousal 高；valence 拿不准就**跳过**——不确定就不作为）
    c3 = cues.get("C3") or {}
    if c3.get("arousal") == 1 and c3.get("valence") is not None:
        for scene in store.query_scenes(valence=c3["valence"], limit=scan_n):
            if gate(scene, "R3"):
                add(scene, "R3 情绪匹配", level=3, dim="C3", vote=1)
        result["actions"]["R3"] = 1.0

    # 实体旁路（唤醒层 §5）：不经语义检索，直接按实体定位。**强维**（2 票）。
    if cues.get("entities"):
        for scene in recall_by_entities(cues["entities"], store, limit=budget):
            add(scene, "实体旁路命中", level=1, dim="entity", vote=2)
        result["actions"]["entity"] = 1.0

    # 字面旁路（唤醒层 §5，2026-10-05 加）：命中**罕见词** → 直接定位，与实体旁路同级。
    # 强维（2 票）：一个只在一个场景里出现过的词被说出来，那几乎就是"在说这件事"。
    for word, sids in (cues.get("literal") or {}).items():
        got = False
        for sid in sids:
            sc = store.get_scene(sid)
            if sc is not None and not sc.archived:
                add(sc, f"字面命中({word})", level=1, dim="literal", vote=2)
                got = True
        if got:
            result["actions"]["literal"] = 1.0

    # R5 原文下钻：**多维协同才给**（原文最长，全量注入会撑爆上下文）
    ranked = order()
    if multi_hit(cues):
        result["actions"]["R5"] = 1.0
        if ranked:
            # 顺手把发生日期传进去：原文按天存文档，给了日期只翻那一天的文件
            result["raws"] = store.get_raws_by_scene(
                ranked[0].id, on_date=(ranked[0].time_record or "")[:10])

    # （原来这里有一条 R6 提示：C5 自指 / C7 未决时提示「编织层候选」——
    #   2026-09-20 两套姿态退役后，它没有可引导的去处，整块删除。）
    # 「你们聊过这个」提示（C6 新鲜度低）——阈值跟着降级状态走，
    # 否则没有向量服务时这句提示永远不会出现（它本来是降级时最该靠得住的一维）。
    if cues.get("C6", 1) < _c6_line(cues):
        result["flags"]["hint_talked_before"] = True

    # R4 常驻画像（**只含 established**——`pending` 是还没印证够的猜测，不进注入）
    # + 活跃 S2（热层）
    result["profiles"] = _pick_profiles(store, cues.get("_msg", ""), emb,
                                        PROFILE_ESTABLISHED)
    # 活跃 S2（热层）：`inject_summaries_n` 是"这轮带几条"，**0 = 不注入**。
    # 显式短路——不能直接调 `hot_summaries(0)`：那边的 `n or 兜底` 会把 0 吃成 20。
    s2_n = int(cfgmod.cfg("recall", "inject_summaries_n", default=0))
    result["summaries"] = store.hot_summaries(s2_n) if s2_n > 0 else []
    result["actions"]["R4"] = 1.0

    # 抑制：排序后取前 N，其余记进 suppressed（trace 要能看到"为什么没它"）
    #
    # `suppressed` 保持 id 列表（报告 / 前端都在按字符串拼它）；
    # **原因另开 `suppressed_detail`**——「名单」和「为什么」是两件事，
    # 硬塞进一个结构会让所有旧调用方一起改，而它们只想要那串 id。
    result["scenes"] = ranked[:budget]
    result["suppressed"] = [s.id for s in ranked[budget:]]
    # 「差多少」= 与**末位入选者**的核心度差。只说"核心度 0.41"答不出问题，
    # 说"比最后一条被带上的低 0.06"才是在回答「它为什么没进来」。
    # 一条都没入选时没有参照系，`gap` 给 None（不给 0 顶替）。
    last = core_score(result["scenes"][-1]) if result["scenes"] else None
    over = []
    for s in ranked[budget:]:
        c = core_score(s)
        over.append({"id": s.id, "title": s.title, "why": why.get(s.id, ""),
                     "stage": "over_budget", "core": round(c, 3),
                     "tier": tier.get(s.id, 9),
                     "gap": None if last is None else round(last - c, 3)})
    # 被相关性门挡住的那批接在后面：它们连排序都没进过，
    # 所以只能带着自己的相关度 / 门线 / 差额出现在这里。
    result["suppressed_detail"] = over + blocked
    result["why"] = {s.id: why.get(s.id, "") for s in result["scenes"]}
    if not result["scenes"] and blocked:
        result["stage"] = "weak_gate"
    return result


# ---- 段 4：计数器 ----

def bump_counters(store, scene_id: str, by: str, at: str = "") -> None:
    """两个指标的递增入口（**递增时机不同，别合并**）。

      by="cited"   → 画像引用这条场景（提炼 3 / 印证收敛）
      by="mention" → 这条场景在对话中被提及（air 说出口或用户复述）

    ⚠️ **内部召回不动 mention_count**——召回是理解的需要，不是「提及」。
    两个方向都顺带 `touch_profile`：被引用 / 被提及都算「这条画像还活着」，
    这是老化判据（`last_support_at`）唯一的数据来源。
    """
    at = at or now_str()
    if by == "cited":
        store.bump_cited(scene_id)
    elif by == "mention":
        store.bump_mention(scene_id, at)
    else:
        raise ValueError(f"未知计数类型: {by}")
    for p in store.profiles_citing(scene_id):
        store.touch_profile(p.id, at)


def scene_mentioned(scene: Scene, texts: list[str]) -> bool:
    """这条场景在对话里被**真的提到**了吗（用户复述 / air 说出口都算）。

    判据是离散的字面命中（关键词 / 标题出现在文本里），不用相似度：
      - 「提及」是**发生过还是没发生过**的事实，不是程度问题——
        在这里设阈值只会把噪声当成信号；
      - 中文短文本的字符重叠量级天生低（见 `_c1_line`），拿它判命中必然漂。

    关键词限 ≥2 字：单字（「猫」「他」）在任何对话里都会命中，等于没判。
    """
    blob = " ".join(t for t in (texts or []) if t)
    if not blob:
        return False
    for kw in scene.keywords or []:
        k = str(kw).strip()
        if len(k) >= 2 and k in blob:
            return True
    title = (scene.title or "").strip()
    return len(title) >= 2 and title in blob


def mark_mentioned(store, scenes: list[Scene], texts: list[str],
                   at: str = "") -> list[str]:
    """给「这一轮真的被提到」的场景记一次 `mention_count`（返回记了哪些）。

    **这是 `mention_count` 在生产链路里唯一的入口**，接线在对话层
    （`ChatSession.reply`）：它手里同时有「注入了哪些场景」和「双方说了什么」。

    两个刻意的限制：
      - **只参评被注入的场景**：召回但没进上下文的，连被说出口的机会都没有；
      - **必须真的在文本里出现**：**内部召回不算提及**——召回是理解的需要，
        不是念叨（对照表第三节）。所以不能放在召回路径上自己数。

    没有它，`mention_count` / `last_mention_at` 永远是 0，连带两件事失效：
    防反刍的上限（`rank.mention_cap`）成了死配置；
    老化的「长期未印证**且**未被提及」只剩「未印证」这一半。
    """
    hit = []
    for s in scenes or []:
        if scene_mentioned(s, texts):
            bump_counters(store, s.id, "mention", at=at)
            hit.append(s.id)
    return hit


# ---- 段 5：trace 与高层入口 ----

def write_trace(record: dict) -> None:
    """每次唤醒落一行 JSONL 到 `data/trace/唤醒-YYYYMMDD.jsonl`。

    这份日志是**逻辑实验的命脉**：记忆系统最难的是「为什么召回了这个」不可见，
    不留痕就只能靠猜。同时它也是二期「向量持久化 / 渐变检测」的数据源——
    C1–C7 的每次取值都在里面，攒够了就是原始素材。
    """
    try:
        trace_dir = cfgmod.abspath(cfgmod.PATHS["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"唤醒-{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[recall] trace 写入失败（不影响唤醒结果）: {e}")


# 他是不是在追问一个指代（2026-09-21，设计稿 F 条）。
# 判据要**窄**：指代词 + 疑问/指认词 + 短消息——不然每轮都会变成"全列"，
# 常备栏就不省了；追问本来就是短的，长消息里的"那个"多半在讲事情本身。
_REFERENT_PRONOUNS = ("那件", "那事", "这事", "这个", "那个",
                      "哪件", "哪事",          # 「哪件事我一句没提」——问的也是某一件事
                      "上次说的", "你说的")
_REFERENT_ASKS = ("哪件", "哪个", "什么", "指的是", "说的是", "指谁")


def asked_for_referent(msg: str) -> bool:
    """他在追问「你刚才说的那个指什么」吗——D2b 常备栏的 `full` 开关（F 条）。"""
    text = (msg or "").strip()
    if not text or len(text) > 60:
        return False
    return (any(w in text for w in _REFERENT_PRONOUNS)
            and any(w in text for w in _REFERENT_ASKS))


def recall_for_message(msg: str, store, llm=None, emb=None) -> dict:
    """高层入口：一句话进 → 唤醒结果出（含 trace）。

    对话层将来只需要调这一个函数。把 cues 和 trace 一起包进来，
    是为了让「每次唤醒都有痕迹」成为默认行为，而不是靠调用方记得写。
    """
    # `_msg` 由 `compute_cues` 自己填（降级时相关性靠它），这里不用再塞一遍。
    cues = compute_cues(msg, store, emb=emb, llm=llm)
    result = recall(cues, store, emb=emb)

    # 常备备忘录（2026-09-21 立；2026-10-05 晚并栏）：**它现在是唯一那条进对话的路**——
    # 候选（`memo_candidates`）与主动开口都删了，合并进这一栏：
    #   ① 到点那 1 件（返回值带 `"due": True`）→ **进注入即记账**（chat 那一侧标 raised）；
    #   ② 其余三个位子给"她手上得有"的（相关先挑满 + 最近补位）。
    # 键名仍是 `standing_memos`：装的是**备忘录的常备子集**，不是卡上的 `open_loops` 字段。
    from .memo import standing_memos
    result["standing_memos"] = standing_memos(
        store, msg, emb=emb, full=asked_for_referent(msg))

    # 线索也回给调用方一份（仪表盘要显示"这轮为什么翻记忆"）——
    # 顺手剥掉内部字段与向量：向量又大又没人看，内部字段以 `_` 开头。
    cues_view = {k: v for k, v in cues.items() if not k.startswith("_") and k != "msg_emb"}
    # 逐维「命中否」**由后端算好**（`cue_hits`）：前端照着高亮就行，
    # 阈值不在页面里抄一份——抄了就会和配置漂移（降级时更明显）。
    cues_view["hits"] = cue_hits(cues)
    # 票数与强信号（2026-10-05）：**给原文（R5）那道门的完整答案**——
    # 只看 hits 数不出票（强维 2 / 弱维 1、按维去重），「为什么这轮没下钻原文」
    # 要答得出来（同「应该发生但没发生必须留痕」）。也进 trace。
    cues_view["votes"] = cue_votes(cues)
    result["cues_view"] = cues_view

    write_trace({
        "ts": now_str(),
        "input": msg,
        "cues": cues_view,
        "actions": result.get("actions", {}),
        "recalled": [{"id": s.id, "core": round(core_score(s), 3),
                      "why": result.get("why", {}).get(s.id, "")}
                     for s in result.get("scenes", [])],
        "profiles": [{"id": p.id, "topic": p.topic, "status": p.status}
                     for p in result.get("profiles", [])],
        "injected": [s.id for s in result.get("scenes", [])],
        "suppressed": result.get("suppressed", []),
        # 带原因的那份也进 trace：「为什么没有它」要能当场答，
        # 只留 id 等于把答案留在了别处（那处通常是没人记得去翻的代码）。
        "suppressed_detail": result.get("suppressed_detail", []),
        # 整轮卡在哪一步（`""` / `"R0"` / `"weak_gate"`）：
        # 「应该发生但没发生」这一类故障全靠它——没有它，trace 上
        # 「库里没有」和「有，但没走到门口」长得一模一样。
        "stage": result.get("stage", ""),
        # 原文按天存文档，分节就是**场景编号**——trace 里记它（Raw 没有 id 了）
        "raws": [r.scene_id for r in result.get("raws", [])],
        # 常备备忘录（D2b；2026-10-05 晚并栏）：这轮她手上有哪几件——
        # 「她为什么答得出/答不出」看这里。**到点那件带 `due`**（另记一个名单，
        # 它是"这一轮真的给过机会"的那条，留痕里要一眼分得开）。
        "standing_memos": [m["id"] for m in result.get("standing_memos", [])],
        "memos_due": [m["id"] for m in result.get("standing_memos", [])
                      if m.get("due")],
        "flags": result.get("flags", {}),
    })
    return result
