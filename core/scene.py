"""场景切分与字段抽取。

两个函数各管一件事：
  - `should_cut`  决定「这段对话讲到哪儿算一个场景」——**切分即提取触发**，
    两者是同一件事（设计稿：不引入新机制）。
  - `extract_scene` 一次 LLM 调用，把一整段对话抽成场景卡 **+ 窗口压缩摘要**。

为什么用「语义距离突增」而不是让 LLM 判边界：边界是**每轮都要判**的成本敏感动作，
LLM 判一次就是一次调用；embedding 距离是免费的，且是 EM-LLM surprise 的轻量代理
（那篇是 token 级、基于模型预测；这里是消息级、用语义距离代理，无 GPU 依赖）。

**保守策略：宁可切粗。** 切碎了（一段完整的互动被拆成三条）比切大了更糟：
场景不完整，后面的聚合和抽象都是在错的原料上做。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L4 写入侧（切分与抽取）
#   上游    ：config、embedding（距离）、model、prompts（抽字段的 prompt 与 schema）
#   下游    ：shortterm（什么时候该切）、distill（它来调这一层的抽取）
#   对外入口：`should_cut` / `should_cut_texts`（降级版）/ `extract_scene` /
#             `render_conversation` / `behavior_intensity` / `is_trivial`
#   边界    ：**不落库**。它给出"这一段是一张什么卡"，写是 distill 的事
# ---------------------------------------------------------------------
from __future__ import annotations

from . import config as cfgmod
from .embedding import cosine
from .model import (ENTITY_KINDS, MEMO_AIR_PROMISE, MEMO_KINDS,
                    MEMO_USER_TASK, SUBJECTS, TRIGGER_CLASSES, Scene)
from .prompts import SCENE_SCHEMA, extract_scene_prompt, rel_stamp

# 消息结构（全项目统一）：{"speaker": "user" | "air", "text": str, "ts": str | ""}
# 用 dict 而不是 dataclass 是因为它跨越 shortterm / scene / demo 的边界，
# 且在落库前不需要类型保证；一旦进入长期记忆就变成 Scene，那才是强类型。
Message = dict

# 追问 / 延续的标志词（第一版规则，实测后再调）。
# 它对应 intensity 的第三个信号「用户是否主动追问」——
# 追问意味着这个话题对用户还有拉力，是强度而非情绪。
_FOLLOWUP_MARKS = ("然后", "那", "后来", "继续", "还有", "接下来", "接着说", "再")


def render_conversation(messages: list[Message], with_date: bool = False) -> str:
    """把消息列表渲染成 prompt 里的对话文本。

    `with_date`：给消息加时间前缀——`（刚刚）2026-09-20 20:58` /
    `（3 小时前）2026-09-20 18:00` / `（一天前）2026-09-19 08:30`。

    为什么要它：短期窗口会攒着跨天的旧消息（提取没触发时尤其明显），
    而渲染出来的对话原本**一个时间戳都没有**——模型看着一堆裸文本、
    加上提示词末尾那个「当前时间」，只能猜这些话是什么时候说的。
    实测后果：把今天刚聊完的事说成「昨晚」。

    为什么今天也标：不给标签时「这是刚才」全靠位置暗示，而位置会被
    压缩、被裁、被挪进「更早」栏——标签跟着消息走，挪到哪儿都还在。
    相对日由 `prompts.rel_stamp` 现算（内部走 `rel_day`）：差几天这一步不该让模型自己数。
    """
    lines = []
    for m in messages or []:
        # 署名用**当时的真名**（`persona`，2026-09-21）：记忆共享、人格是外壳，
        # 但"谁说的"不能抹平——切到 mia 后历史里全是 "air"，她会把别人的话
        # 当成自己说过的。旧的 / 缺字段的消息回退 "air"（兼容）。
        who = ("用户" if (m.get("speaker") or "user") == "user"
               else (m.get("persona") or "air"))
        tag = ""
        if with_date:
            tag = rel_stamp((m.get("ts") or "")[:16])
        lines.append(f"{who}{tag}: {(m.get('text') or '').strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# 一、切场景
# ---------------------------------------------------------------------

def should_cut(new_msg_emb: list | None,
               window_embs: list,
               gamma: float | None = None,
               min_distance: float | None = None,
               min_window: int | None = None) -> bool:
    """这条新消息是否开启了一个新场景。

    判据：新消息与「窗口内最后一条」的语义距离，是否突增到
    `max(mean + gamma × std, min_distance)` 之上。

    两个保险都是刻意的：
      - `min_distance` 绝对下限：短对话里相邻距离都很小，光看相对突增会把
        正常起伏判成边界（一段五分钟的闲聊被切成四段）。
      - `min_window` 最少样本数：少于 3 条消息时 mean/std 没有统计意义，
        直接不切——**宁可切粗**。

    向量缺失（embedding 服务不可用）时返回 False：降级状态下宁可不切，
    也不要凭猜切。这一段会在「会话结束」时被提取，不会丢。
    """
    if not new_msg_emb or not window_embs:
        return False
    gamma = cfgmod.cfg("cut", "gamma", default=1.0) if gamma is None else gamma
    min_distance = cfgmod.cfg("cut", "min_distance", default=0.35) if min_distance is None else min_distance
    min_window = cfgmod.cfg("cut", "min_window", default=3) if min_window is None else min_window

    if len(window_embs) < min_window:
        return False

    dists = [1.0 - cosine(window_embs[i], window_embs[i + 1])
             for i in range(len(window_embs) - 1)]
    dists = [d for d in dists if d >= 0]
    if len(dists) < 2:
        return False

    mean = sum(dists) / len(dists)
    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    std = var ** 0.5

    d_new = 1.0 - cosine(window_embs[-1], new_msg_emb)
    threshold = max(mean + gamma * std, min_distance)
    return d_new > threshold


def should_cut_texts(new_text: str, window_texts: list[str],
                     gamma: float | None = None,
                     min_distance: float | None = None,
                     min_window: int | None = None) -> bool:
    """**没有向量服务时**的切分判据：用字符重叠代替余弦。

    为什么必须有它：原来降级（没配 embedding）就**完全不判切分**，
    于是自动提取只剩「攒够 60 条消息 / 4000 字 / 空闲 30 分钟」这几个兜底——
    实际表现就是「从来不自动提取，只能点『新对话』」。
    **降级变糙可以，变哑不行**：这跟 `recall._c1_line` 是同一条要求
    （降级时阈值跟着走，否则等于失忆）。

    距离用 `1 - char_overlap`（bigram Jaccard），判据结构完全同 `should_cut`：
    `> max(mean + gamma×std, min_distance)`，同样「宁可切粗」。

    ⚠️ 字符重叠的分布和余弦完全不同（同主题文本的重叠也在 0.05–0.3，
    距离挤在 0.7–0.95），所以**不能复用 `cut.min_distance`（0.35）**——
    那条线对字符重叠来说低到每条都会切。用单独的 `min_distance_text`。
    """
    if not new_text or not window_texts:
        return False
    from .recall import char_overlap        # 延迟 import：char_overlap 住在 recall 里

    gamma = cfgmod.cfg("cut", "gamma", default=1.0) if gamma is None else gamma
    min_distance = (cfgmod.cfg("cut", "min_distance_text", default=0.93)
                    if min_distance is None else min_distance)
    min_window = cfgmod.cfg("cut", "min_window", default=3) if min_window is None else min_window

    if len(window_texts) < min_window:
        return False
    dists = [1.0 - char_overlap(window_texts[i], window_texts[i + 1])
             for i in range(len(window_texts) - 1)]
    if len(dists) < 2:
        return False
    mean = sum(dists) / len(dists)
    std = (sum((d - mean) ** 2 for d in dists) / len(dists)) ** 0.5
    d_new = 1.0 - char_overlap(window_texts[-1], new_text)
    return d_new > max(mean + gamma * std, min_distance)


# ---------------------------------------------------------------------
# 二、行为强度（不靠 LLM）
# ---------------------------------------------------------------------

def _is_followup(messages: list[Message]) -> bool:
    """用户是否在主动追问 / 延续（取最后一条用户消息判断）。

    第一版是规则，不引 LLM：强度是**行为信号**，而「问句 + 延续词」这个规则
    虽然粗，但它的错误是随机分布的（不像 LLM 会有系统性偏差），不会带偏排序。
    """
    last_user = None
    for m in messages or []:
        if (m.get("speaker") or "user") == "user":
            last_user = m.get("text") or ""
    if not last_user:
        return False
    if "?" in last_user or "？" in last_user:
        return True
    return any(w in last_user for w in _FOLLOWUP_MARKS)


def behavior_intensity(messages: list[Message], prev_mentions: int = 0) -> float:
    """强度 = 行为信号计数，归一化到 0–1。**第一版只用三个信号**。

      w_len      消息总字数 / `len_ref`（截断到 1）
      w_repeat   该话题此前被提及次数（`prev_mentions`）
      w_followup 用户是否主动追问 / 延续

    为什么不由 LLM 给强度：强度要的是**可比**，不同次调用之间必须同尺度；
    LLM 的判断会漂（同一个对话换个问法给不同分），那排序就成了随机数。
    行为信号不依赖模型，是可复现的。
    """
    conf = cfgmod.cfg("intensity", default={}) or {}
    text = "".join((m.get("text") or "") for m in messages or [])
    len_ref = float(conf.get("len_ref") or 200)
    repeat_ref = float(conf.get("repeat_ref") or 3)

    s_len = min(len(text) / len_ref, 1.0) if len_ref else 0.0
    s_rep = min(prev_mentions / repeat_ref, 1.0) if repeat_ref else 0.0
    s_fol = 1.0 if _is_followup(messages) else 0.0

    val = (conf.get("w_len", 0.4) * s_len
           + conf.get("w_repeat", 0.4) * s_rep
           + conf.get("w_followup", 0.2) * s_fol)
    return max(0.0, min(1.0, val))


# ---------------------------------------------------------------------
# 三、字段抽取（一次调用出场景卡 + 窗口摘要）
# ---------------------------------------------------------------------

def _norm_open_loops(raw) -> list[dict]:
    """归一化 open_loops：只留 `{content, kind, due_at, group_name}`。

    **为什么要认别名**：`json_schema` 档能锁死键名，但服务端一旦不支持
    （DeepSeek 就是），降级到 `json_object` 后**键名全靠模型自觉**。
    实测它会写 `item` 而不是 `content`、写 `type` 而不是 `kind`——
    那时如果只认规范键名，这一条就被**静默丢掉**：不报错、不留痕，
    只是「备忘录莫名其妙少了一条」，极难查（真查了一轮）。

    所以：规范键名优先，另认几个常见别名；实在认不出来才丢，
    而且**丢的时候一定要出声**——静默丢数据的代价比多打一行日志大得多。

    未闭合线索是**备忘录的唯一来源**，垃圾进垃圾出——宁可少存一条，
    也不要让「空字符串」「kind 拼错」这类东西进 memos 表。
    """
    content_keys = ("content", "item", "title", "text", "name")
    kind_keys = ("kind", "type", "category", "source")
    # 组名也认几个常见写法（同 content / kind 的理由：降级时键名靠模型自觉）
    group_keys = ("group_name", "group", "task", "plan", "组名", "组")
    out = []
    for idx, item in enumerate(raw or []):
        if not isinstance(item, dict):
            print(f"[scene] open_loops[{idx}] 不是对象，跳过: {str(item)[:60]}")
            continue
        content = next((str(item[k]).strip() for k in content_keys
                        if str(item.get(k) or "").strip()), "")
        if not content:
            print(f"[scene] open_loops[{idx}] 认不出内容，跳过（键: {list(item.keys())}）")
            continue
        kind = next((item[k] for k in kind_keys
                     if item.get(k) in MEMO_KINDS), None) or MEMO_USER_TASK
        # 客套话不是承诺——prompt 里写了反例，实测仍会漏，代码再兜一道。
        if kind == MEMO_AIR_PROMISE and is_pleasantry(content):
            print(f"[scene] open_loops[{idx}] 看着是 air 的客套话，跳过: {content[:30]}")
            continue
        # 组名**自由生成、不收敛**（2026-09-22）：同一件事的多步共用一个短名，
        # 截到 20 字——它只用于呈现与提醒收拢，长了没意义。
        gname = next((str(item[k]).strip() for k in group_keys
                      if str(item.get(k) or "").strip()), "")
        out.append({"content": content, "kind": kind,
                    "due_at": str(item.get("due_at") or "").strip(),
                    "group_name": gname[:20]})
    return out


# air 的客套话：**说出来就完了，不该进提醒池**。
# 这不是"讨厌寒暄"，而是 memo 的定义——它是「未了结的**事**」的候选，
# 而客套话没有"事"，只有态度。混进来的后果不是多一条数据，
# 是将来某天 air 一本正经地提醒「我答应过你随时可以聊」——很怪。
_PLEASANTRIES = ("随时", "有需要", "有想聊", "加油", "早点休息", "别客气",
                 "慢慢来", "注意休息", "路上小心", "保重", "开心点")


def is_pleasantry(content: str) -> bool:
    """这条 `air_promise` 是不是其实只是客套话。

    **为什么要在代码里也兜一道**：prompt 里已经写了「客套话不是承诺」
    并且给了反例，实测仍然会漏——`air 邀请用户随时聊天` 被抽成了
    `air_promise`。prompt 管的是"多数情况判断对"，管不了"错的那一次"。

    判据刻意保守：**短 + 含客套词**，两条都满足才算。
    真承诺会带上具体的事（「下次给你看那份文档」），长度和用词都不同；
    只含态度、没有事项的短句，才是客套。

    宁可少收一条：memo 是**提醒的候选池**，多一条噪声就多一次尴尬提醒；
    少一条的代价只是"这次没提"（用户不提也就过去了）。
    """
    text = (content or "").strip()
    if not text or len(text) > 20:
        return False                    # 有具体内容的，不是客套
    return any(w in text for w in _PLEASANTRIES)


# 纯应答 / 纯寒暄的词表（先去掉句末标点再比）。
_TRIVIAL_REPLIES = {
    "嗯", "嗯嗯", "好的", "好", "是", "是的", "对", "对的", "啊", "哦", "噢", "哈哈",
    "在", "在吗", "在不在", "知道了", "收到", "没事", "谢谢", "谢谢啦", "行", "行吧",
    "ok", "OK", "Ok", "没有", "不知道", "睡吧", "晚安", "早", "早上好",
}


def is_trivial(messages: list[Message]) -> bool:
    """整段是不是「纯寒暄 / 纯应答」——**代码判，不问模型**。

    为什么要代码判：
      1. 这类内容一眼可判（很短 + 全在词表里），让模型判要白花一次调用；
      2. 模型在这件事上**偏保守**——prompt 里反复强调「拿不准就存」之后尤其明显。
         实测「在吗」「嗯」被模型判成了值得存（它宁可多存，这是对的，
         但这段确实没有任何东西值得留）。

    判据刻意严：**所有**用户消息都 ≤6 字、且**都**在词表里才算。
    只要有一句带实质内容——哪怕很短（「面试没过」）——就不算。
    """
    users = [(m.get("text") or "").strip() for m in messages
             if m.get("speaker") == "user"]
    users = [t for t in users if t]
    if not users:
        return False                    # 只有 air 在说话 → 不判（可能是她在承诺什么）
    return all(len(t) <= 6 and t.strip("。！？!?.,，~～ ") in _TRIVIAL_REPLIES
               for t in users)


def extract_scene(messages: list[Message],
                  llm,
                  emb_service=None,
                  topic_candidates: list[str] | None = None,
                  source: str = "") -> tuple[Scene, str, list[dict], dict]:
    """把一段对话抽成场景卡。

    返回 `(scene, window_digest, entities, verdict)`——比 `tuple[Scene, str]` 多两项：
    实体要写 `scene_entities` 关联表
    （存储层 §5 用**表**不用列）；`verdict` 是「**这段值不值得存**」，
    得从这一层带出来，让调用方决定落不落库。

    **关于 verdict**：四条触发一到就无脑落库，会让纯寒暄也变成场景卡
    （「在吗」「嗯」「哈哈」各占一条）。所以让模型在**同一次抽取**里顺手判一下
    ——不额外花一次调用。默认 `true`，只有明确没信息量才 false：
    **错存一条只是多占一点地方，漏存一条是永久丢记忆。**

    关键约定：
      - **summary 与 window_digest 粒度不同**，不共用（一个供检索、一个补窗口）。
      - **时间不由 LLM 抽**：`time_event` 取这段对话的记录时刻。
        LLM 的时间感不可靠——同 memo「不让 LLM 估天数」一个道理。
        例外是用户**明说**的时间（如 memo 的 due_at），那是转述不是估算。
      - `valence` / `arousal` 非法或拿不准 → **None**（不确定就不作为）。
      - `trigger_class` 不在 7 类枚举内 → 空串（不猜；它是统计索引，污染了整列都废）。
    """
    conversation = render_conversation(messages)
    data = llm.structured(extract_scene_prompt(conversation, topic_candidates), SCENE_SCHEMA)

    scene = Scene()
    scene.title = str(data.get("title") or "").strip()[:60]
    scene.keywords = [str(k).strip() for k in (data.get("keywords") or [])
                      if str(k).strip()][:8]
    scene.text = str(data.get("summary") or "").strip()
    scene.topic = str(data.get("topic") or "").strip()

    v = data.get("valence")
    scene.valence = v if v in (-1, 0, 1) else None
    a = data.get("arousal")
    scene.arousal = a if a in (0, 1) else None

    scene.trigger = str(data.get("trigger") or "").strip()
    tc = data.get("trigger_class")
    scene.trigger_class = tc if tc in TRIGGER_CLASSES else ""
    scene.reaction = str(data.get("reaction") or "").strip()
    scene.outcome = str(data.get("outcome") or "").strip()

    subj = data.get("subject")
    scene.subject = subj if subj in SUBJECTS else "user"
    scene.self_ref = 1 if data.get("self_ref") else 0
    scene.air_stance = str(data.get("air_stance") or "").strip()
    # 保守取敏感：它只影响「说出口的力度」与镜像呈现，不影响召回，
    # 所以误标 1 的代价远小于漏标。
    scene.sensitive = 1 if data.get("sensitive", True) else 0
    scene.open_loops = _norm_open_loops(data.get("open_loops"))
    scene.source = source
    scene.time_record = scene.time_event = _event_time(messages)

    # 向量用「标题 + 关键词 + 摘要」拼的检索文本算，而不是整段原文：
    # 检索要的是「这条场景在说什么」，不是它的措辞细节（细节在 S0）。
    if emb_service is not None:
        search_text = " ".join([scene.title] + scene.keywords + [scene.text]).strip()
        if search_text:
            scene.emb = emb_service.embed_one(search_text)

    entities = _norm_entities(data.get("entities"))
    verdict = {"worth": bool(data.get("worth_saving", True)),
               "reason": str(data.get("skip_reason") or "").strip()}
    return scene, str(data.get("window_digest") or "").strip(), entities, verdict


def _event_time(messages: list[Message]) -> str:
    """事件发生时间：取这段对话里最早的一条消息时间戳，没有就用当前时间。"""
    stamps = [m.get("ts") for m in messages or [] if m.get("ts")]
    if stamps:
        return str(min(stamps))
    from .store import now_str
    return now_str()


def _norm_entities(raw) -> list[dict]:
    """归一化 entities：留 `{name, kind, relation}`，丢掉无名项与**没关系的项**。

    2026-09-25 收窄：实体只记**和用户有个人关系的**（家人 / 朋友 / 宠物 /
    地点 / 用户的项目）。`relation` 是这段话里说出来的关系或态度，写不出关系
    的名字（公共人物、技术名词、话题词）**在这里就被丢掉**——它不进场景卡，
    更进不了索引。判据为什么放在抽取层：关系词就在原文里（「我妈」），
    让模型认结构比事后猜"这个名字重不重要"可靠得多。

    kind 不在枚举内时**保留原值**而不是硬塞一个人为默认，只放行短词——
    **超 20 字符**的自由长句兜回 `person`（那多半是模型把一句话写进了
    kind，不是类别名）。kind 只用于展示和轻统计，保留模型给的原词保留了信息，
    硬塞「person」反而会污染统计（把「某个抽象概念」记成人）。
    """
    out = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        relation = str(item.get("relation") or "").strip()
        if not relation:
            continue
        kind = str(item.get("kind") or "").strip() or "person"
        if kind not in ENTITY_KINDS and len(kind) > 20:
            kind = "person"
        out.append({"name": name, "kind": kind, "relation": relation})
    return out
