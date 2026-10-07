"""数据模型。

为什么用 dataclass 而不是裸 dict：字段名打错一个字母，dict 要跑到底才炸、
而且可能静默写进一个谁都不认的键；dataclass 在构造时就报错。
对一个「记忆不可重建」的系统，这类静默错误比崩溃危险得多。

字段顺序与建表一致，方便对照审计（改一处记得改两处：建表 / 本文件）。
`from_row` 负责把 SQLite 的 TEXT 列（JSON 数组）反序列化回来，
`to_row` 反过来——**所有序列化只走这两个出口**，不散在业务代码里。

每个数据类的 `to_row` / `from_row` 都是这一对约定的实例，
所以它们**不再逐个写 docstring**——职责就是那两个箭头，
写一遍约定比把同一句话抄十四遍有用。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L0 基础层
#   上游    ：无
#   下游    ：store（读写行）、以及一切需要造 `Scene` / `Profile` 的层
#   对外入口：Entity / Scene / Raw / Summary / Profile / ProfileReview / Memo
#              + 各枚举常量 + `lang_from_prefs`（`Opening` 随主动开口退役，2026-10-05）
#   边界    ：**纯数据结构**——不写「为空就取那个」的兜底，那是逻辑层的判断
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------
# 枚举常量（设计稿里定死的取值域，不接受自由文本）
# ---------------------------------------------------------------------

# 归属三类（存储层 §1）：关于用户本人 / 关于世界（含实体与话题）/ 关于 air
SUBJECTS = ("user", "world", "air")

# trigger_class 的 7 类粗枚举（存储层 §3）。
# 它是**统计的索引**，不是理解的载体——所以必须封闭，才能聚合。
TRIGGER_CLASSES = (
    "被评价", "目标未达", "关系冲突", "失去", "得到达成", "转折改变", "悬而未决",
)

# （`EDGE_KINDS` / `EDGE_ROLES`——`edges` 表的取值域——已随边表退役，2026-09-24：
#   四种边全改派生（引用 = `sources`，相邻 = topic + 时间，原文 = 按天文档，
#   引用起点 = `Profile.evidence_at`）。见存储层稿 §五。）

# 实体类别（2026-09-25 收窄：只记**和用户有个人关系的**人 / 宠物 / 地点 / 作品）。
# 「事件」桶已去——它把「9月20日自然醒」「剪葫芦」这类发生过的事也吞成实体，
# 而那些是场景本身的内容，不是标签。判据与闸门在 `prompts._SCENE_PROMPT`
# 与 `entity.link_entities`（没 relation 不记）。
ENTITY_KINDS = ("person", "pet", "place", "work")

# 画像状态（存储层 §3）：待验证 / 已立
PROFILE_PENDING, PROFILE_ESTABLISHED = "pending", "established"
PROFILE_STATUS = (PROFILE_PENDING, PROFILE_ESTABLISHED)

# 画像失效的发起者（`Profile.invalidated_by`）——三个值各有专门写口：
#   revision  她自己拿新证据修正（`distill.revise_profile`，默认）
#   user      人改画像陈述（界面 / 确认条那条路，`weave.update_profile_confirmed`）
#   archive   归档进冷层（`store.archive_profile`，可 `unarchive_profile` 撤回）
# ⚠️ 「用户**否决**」自 2026-09-24 起 = 真删（`store.delete_profile`）、不落这个字段；
#    旧库里按旧语义留下的 user 行仍读得出（见 `store.invalidate_profile` 的说明）。
INVALIDATED_REVISION, INVALIDATED_USER, INVALIDATED_ARCHIVE = (
    "revision", "user", "archive")

# S3 复核的打分（记忆整理稿 §五，2026-09-23；同日补 reword）：
#   holds 还成立（**什么都不做**）/ thin 证据薄 / stale 过时了
#   / reword 内容站得住但**写法坏了**（评价词 / 特质标签——给改写，走修正）
#   / wrong 判断本身不成立（只出「提议」，人点头才作废）
# 系统直接做的只有降档与重写（都不删任何东西）；wrong 是唯一要人点头的。
REVIEW_VERDICTS = ("holds", "thin", "stale", "reword", "wrong")

# 基础档案的常见键（界面上预填这几项；键不限于此，可自定义）。
#
# 为什么单独立一块、而不是塞进 profiles：那是 air 的**推断**（要印证、会过期、可否决），
# 这是用户**明说的事实**（无需印证、不会老化）。语义不同，存储、注入、修改权都不同。
FACT_KEYS = ("称呼", "年龄", "性别", "城市", "职业")

# 用户偏好的存放处是 `user_prefs`（key/value），装的是**他想要什么**，不是他是什么。
# 有哪些键由使用点决定（`lang` / `persona`），**不在这里列一遍**——
# 列了没人用就是死常量，还多一处会跟实现漂移的地方。

# 2026-09-20 退役：两套对话姿态（BASELINE / WEAVE / CHAT_MODES）整块删除——
# 编织的做法已写进人格文件（`self/personas/*.md`），不再单独做这个模式。
# 老 `weave_mode` 偏好键留库、不再被读（无害）；历史实现见 git。

# 2026-09-19 退役：说话方式五档（STYLE_DIMS / DEFAULT_STYLE / style_from_prefs）
# 整块删除——和三个人格不搭，「怎么说话」只在人格文件里（`self/personas/*.md`）。
# 老 `style_*` 偏好键留库不再被读（无害）；历史实现见 git。


# 界面与她的输出语言（2026-09-17）：中 / 英。
# `zh` 是默认，语言段**零注入**（见 prompts.render_lang_block）；`en` 时系统提示词里
# 多一段英文指令。另有一处跟"想"有关、**不在那一段里**：**中文模式**在提示词
# **末尾**多一行「思考过程一律用中文」（prompts.render_think_block；2026-09-26：
# 模型默认用英文想，只有末行押得住）；英文模式零注入——顺默认。
LANGS = ("zh", "en")
DEFAULT_LANG = "zh"


def lang_from_prefs(prefs: dict) -> str:
    """从 `user_prefs` 读语言；没设过、或值非法，回中文（默认不打扰）。"""
    v = (prefs or {}).get("lang") or ""
    return v if v in LANGS else DEFAULT_LANG

# 备忘录来源两类（短期记忆 §4）：用户的未完成事 / air 自己的承诺
MEMO_USER_TASK, MEMO_AIR_PROMISE = "user_task", "air_promise"
MEMO_KINDS = (MEMO_USER_TASK, MEMO_AIR_PROMISE)

# 备忘录分类（短期记忆 §5）：无明确时间时只分类，周期由系统映射
# 2026-09-22 加 `followup`：**要过几天回头看结果的**（观察期 / 等反馈）——
# 它的窗口给得短（4 天），到点进注入一次（进即记账，见 `memo.due` / `standing_memos`）。
# ⚠️ 2026-10-06 改注：原来这里写着"不挂在常备栏每轮刷（走候选 / 主动开口）"——
# 那两条路 2026-10-05 晚删了；现在"每轮刷"由**进即记账 + 一次一件 + 封顶 4** 挡住，
# 没到点的它照别的 memo 一样，只在"相关 / 最近"里兜底出现。
MEMO_CLASSES = ("deadline", "progress", "idea", "followup")

# 备忘录状态机（短期记忆 §5）：待提 → 已提 → 关闭
MEMO_PENDING, MEMO_RAISED, MEMO_CLOSED = "pending", "raised", "closed"
# ⚠️ `MEMO_STATUS` 目前**只作取值域对照**（`Memo.status` 的注释指向它）：
#    三态**没有从外部传值的写入口**——`store.mark_memo_raised()` / `close_memo()`
#    内部用成员常量写死，判断侧（`open_memos` 等）也只用成员常量；
#    没有校验点可接，如实写明（不为"让它有人用"造一个）。
MEMO_STATUS = (MEMO_PENDING, MEMO_RAISED, MEMO_CLOSED)

# 备忘录的「时机类别」（2026-09-15 立；2026-10-05 晚**收窄到两值**）。
# 原来四值（result / after / soon / later）配一张「什么时候提才自然」的时机表；
# 那张表删了（K 条：算不准，也不算），四个值里只剩 `soon` 还有用——
# **身体状况不等常规窗口**（`created + memo.soon_hours` 就算到点，越早问越暖）。
# 其余一律 `later`；老库里的 result / after 原样躺着（读侧只认 `soon`，不报错）。
#
# 两个值**建成员常量**（2026-10-06）：`soon` 是**参与比较的值**
# （`memo.due()`：`timing == MEMO_TIMING_SOON`），裸字面量改名就静默失效——
# 同 `MEMO_PENDING` / `INVALIDATED_*` 的处理。
MEMO_TIMING_SOON, MEMO_TIMING_LATER = "soon", "later"
# ⚠️ `later` **没有读点**：它是"照常"的取值域对照（判断侧只认 `soon`）——
#    同 `MEMO_STATUS` 的处境，如实写明，不为"让它有人用"造一个读点。
MEMO_TIMINGS = (MEMO_TIMING_SOON, MEMO_TIMING_LATER)


def _json_list(raw: Any) -> list:
    """TEXT 列 → list。容忍空 / 坏数据（存储层出错时降级成空列表，不抛）。

    这里「不抛」是刻意的：读取侧崩掉会让整条链路停摆，
    而一个字段坏掉只该影响这一个字段（素材不丢，但也不因它陪葬）。
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (ValueError, TypeError):
        return []


def _json_dict(raw: Any) -> dict:
    """TEXT 列 → dict（`_json_list` 的同型兜底：空 / 坏数据降级成空 dict，不抛）。"""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except (ValueError, TypeError):
        return {}


def _json_dumps(val: Any) -> str:
    """list → TEXT 列（`_json_list` 的反向）。空值写成 `[]` 而不是 `null`——
    读回来时一律是 list，调用方不需要先判 None。
    """
    return json.dumps(val if val is not None else [], ensure_ascii=False)


# ---------------------------------------------------------------------
# 实体
# ---------------------------------------------------------------------

@dataclass
class Entity:
    """实体索引里的一条（人物 / 宠物 / 地点 / 作品——与 `ENTITY_KINDS` 同）。

    单用户场景下取消「按人分区」，第三方人物与宠物、地点一样只是**具体元素**
    （存储层 §1）。aliases 是消歧用的别名表。
    """
    id: str = ""
    name: str = ""
    kind: str = "person"
    aliases: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_row(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "aliases": _json_dumps(self.aliases), "created_at": self.created_at}

    @staticmethod
    def from_row(row) -> "Entity":
        return Entity(id=row["id"], name=row["name"], kind=row["kind"],
                      aliases=_json_list(row["aliases"]), created_at=row["created_at"])


# ---------------------------------------------------------------------
# S1 场景卡（主体资产）
# ---------------------------------------------------------------------

@dataclass
class Scene:
    """S1 场景卡——**所有字段都由七条唤醒线索反推**（存储层 §3）。

    改字段前先问：它对应哪条线索？对不上就不该存
    （「能客观定的才存，需要理解的留给对话」）。
    """
    id: str = ""
    title: str = ""
    keywords: list[str] = field(default_factory=list)
    text: str = ""                     # 一句话摘要（供检索 / 画像），≠ window_digest
    emb: list[float] | None = None     # 语义向量（C1/C4 用）；无服务时为 None
    time_event: str = ""               # 事件发生时间（C2）
    time_record: str = ""              # 记录时间（C2）
    valence: int | None = None         # 效价 1/-1/0；NULL = 不确定（LLM 拿不准）
    arousal: int | None = None         # 唤醒度 1/0；NULL = 不确定
    intensity: float = 0.0             # 强度 0–1（**行为信号计数，非 LLM**）
    subject: str = "user"              # user / world / air
    topic: str = ""                    # 本条场景的主主题（S2 按它分组、S3 沿用它）
    self_ref: int = 0                  # 元交流标记（C5）
    cited_by_profile: int = 0          # 被画像引用（**重要性**，不设上限）
    mention_count: int = 0             # 对话中被提及（**防反刍**，设上限）
    sensitive: int = 0                 # 敏感标记（调节说出口的力度，**不排除召回**）
    trigger: str = ""                  # 具体触发（自由文本，人可读）
    trigger_class: str = ""            # 情境类型（粗枚举 7 类，统计索引）
    open_loops: list = field(default_factory=list)  # 未闭合线索；memos 是它的可提醒子集
    air_stance: str = ""               # air 的实质性表态（承诺/立场/边界/信息），可空
    reaction: str = ""
    outcome: str = ""
    source: str = ""                   # 来源对话 id
    last_mention_at: str = ""          # 最近一次被提及（老化判据）
    archived: int = 0                  # 1 = 已进冷层（仍可打捞，非删除）
    created_at: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["keywords"] = _json_dumps(self.keywords)
        d["open_loops"] = _json_dumps(self.open_loops)
        # ⚠️ 必须 **pop** 掉 emb，不能设成 None：
        # 设成 None 时键还在，`add_scene` 又会拼一个 ", emb" 列 →
        # INSERT 里出现两个 emb，SQLite 取第一个（NULL），
        # 向量就永远是空的。这个 bug 藏了很久，因为所有测试都在
        # 「没配向量服务」的降级状态跑 —— 向量本来就是 None，
        # 「存不进去」和「本来就是空的」看起来一模一样。
        d.pop("emb", None)
        return d

    @staticmethod
    def from_row(row, emb: list[float] | None = None) -> "Scene":
        return Scene(
            id=row["id"], title=row["title"], keywords=_json_list(row["keywords"]),
            text=row["text"], emb=emb, time_event=row["time_event"],
            time_record=row["time_record"], valence=row["valence"], arousal=row["arousal"],
            intensity=row["intensity"], subject=row["subject"], topic=row["topic"],
            self_ref=row["self_ref"], cited_by_profile=row["cited_by_profile"],
            mention_count=row["mention_count"], sensitive=row["sensitive"],
            trigger=row["trigger"], trigger_class=row["trigger_class"],
            open_loops=_json_list(row["open_loops"]), air_stance=row["air_stance"],
            reaction=row["reaction"], outcome=row["outcome"], source=row["source"],
            last_mention_at=row["last_mention_at"], archived=row["archived"],
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------
# S0 原文
# ---------------------------------------------------------------------

@dataclass
class Raw:
    """S0 原文：对话原文 + 反指向它的 S1。

    **S0 自身没有关键词和索引**——这正是「S0 最难召回」的本质：
    它没有独立可检索性，只能通过 S1 下钻（存储层 §3）。
    所以这里除了 `scene_id` 没有任何用于检索的字段，这是刻意的，别加。

    ⚠️ **它不进库**：原文按天写成文档（`data/raws/YYYY-MM-DD.md`），
    这个类只在内存里流转（写文档前、读文档后）。
    所以它**没有 `id` / `archived` / `to_row` / `from_row`**——
    那些是"表里的行"才需要的东西，留着就是死代码。
    """
    scene_id: str = ""
    content: str = ""
    created_at: str = ""


# ---------------------------------------------------------------------
# S2 主题摘要
# ---------------------------------------------------------------------

@dataclass
class Summary:
    """S2：同主题多个 S1 **聚合**成的叙述（可逆，可从 S1 重建）。

    聚合不下结论，所以 S2 不需要印证机制——印证只压在抽象层（S3）。
    """
    id: str = ""
    topic: str = ""                    # 主主题（**S2 按它分组**——聚合与版本的键）
    topics: list[str] = field(default_factory=list)
    # ↑ 主题标签（1-3 个，2026-09-24 晚）：**第一个 = 主主题**（与 `topic` 同步写）。
    #   多出来的 0-2 个是**附加主题**（这条叙述顺带沾到的话题），只用于「按主题找」
    #   与展示——聚合 / 版本序列只认主主题（链路键不能多值）。
    #   老库为 NULL → `from_row` 兜 `[topic]`，回填见 `store._backfill_topics`。
    text: str = ""
    sources: list[str] = field(default_factory=list)
    archived: int = 0
    created_at: str = ""

    def to_row(self) -> dict:
        return {"id": self.id, "topic": self.topic,
                "topics": _json_dumps(self.topics), "text": self.text,
                "sources": _json_dumps(self.sources), "archived": self.archived,
                "created_at": self.created_at}

    @staticmethod
    def from_row(row) -> "Summary":
        keys = row.keys() if hasattr(row, "keys") else []
        topic = row["topic"] or ""
        return Summary(id=row["id"], topic=topic, text=row["text"],
                       # 老库补列后是 NULL（回填由 store 的迁移做）——读端先兜 `[topic]`
                       topics=((_json_list(row["topics"]) if "topics" in keys else [])
                               or ([topic] if topic else [])),
                       sources=_json_list(row["sources"]), archived=row["archived"],
                       created_at=row["created_at"])


# ---------------------------------------------------------------------
# S3 画像
# ---------------------------------------------------------------------

@dataclass
class Profile:
    """S3 画像：跨主题的模式陈述（**唯一在「下判断」的一层**）。

    双时间戳取代版本链（存储层 §3）：
      - valid_at       从什么时候起成立
      - invalidated_at 什么时候失效（为空 = 当前有效）
    **invalidated_at 只由两件事写：新证据修正、用户否决**。
    老化过期是另一条路——只把 status 从 established 降回 pending，
    不填 invalidated_at。
    """
    id: str = ""
    topic: str = ""                    # 主主题（**版本序列的键**，resolve_topic 维护）
    topics: list[str] = field(default_factory=list)
    # ↑ 主题标签（1-3 个，2026-09-24 晚）：同 `Summary.topics`——第一个 = 主主题，
    #   其余是附加主题（只用于「按主题找」与展示）。
    subject: str = "user"              # topic 的主语归属（resolve_topic 硬过滤用）
    statement: str = ""
    evidence: int = 0                  # 印证数
    status: str = PROFILE_PENDING      # 待验证 / 已立（见 PROFILE_STATUS）
    valid_at: str = ""
    invalidated_at: str = ""           # 为空 = 当前有效
    invalidated_by: str = ""           # revision / user / archive（空 = 当前有效，见 INVALIDATED_*）
    invalidated_reason: str = ""       # 否决时用户的原话（修正时为空）
    last_support_at: str = ""          # 最近一次被印证 / 被提及（老化判据）
    last_review_at: str = ""           # 最近一次「复核」的时刻（记忆整理稿 §五，2026-09-23）
    # ↑ 复核 ≠ 印证：复核**不 bump** `last_support_at`——否则「复核通过」成了续命机制，
    #   90 天老化永远不会触发。这个字段只服务于「复核间隔」防抖。
    sources: list[str] = field(default_factory=list)   # 支撑它的 S1/S2（可追溯）
    evidence_at: dict[str, str] = field(default_factory=dict)
    # ↑ 「这条依据是哪一刻被引用的」——id → 被引用时刻（`{"S1-0003": "2026-09-24 12:00:00"}`）。
    #   有它才算得出保护期（`store.protected_scene_ids`：forming 365 天 / 印证 90 天，
    #   起点是**被引用那一刻**，不是场景发生那天）。
    #   2026-09-24 之前它住在 `edges` 表的 evidence 边里——那条边是对 `sources` 的冗余
    #   存储，只有"引用起点时间"是真数据；搬到这里后边表可退役（存储层稿 §五）。
    #   role（forming / supporting）**不存**：由 `evidence_pack` 的 id 集合派生即可
    #   （forming 那几条进包，见 `weave.render_mirror` 的同款判法）。
    evidence_pack: list[dict] = field(default_factory=list)
    # ↑ 形成依据的**快照**（id / 标题 / 摘要 / 时间）。
    #   它是「追溯的底线保障」：原始场景被归档后仍能说出「凭哪几条」，
    #   只是看不到完整原文（那是 S0 的事，仍可打捞）。
    #   只装 forming 那几条（少而稳定），不装后来的印证——印证会一直新增，装进去包会膨胀。
    created_at: str = ""

    def to_row(self) -> dict:
        return {"id": self.id, "topic": self.topic,
                "topics": _json_dumps(self.topics), "subject": self.subject,
                "statement": self.statement, "evidence": self.evidence,
                "status": self.status, "valid_at": self.valid_at,
                "invalidated_at": self.invalidated_at,
                "invalidated_by": self.invalidated_by,
                "invalidated_reason": self.invalidated_reason,
                "last_support_at": self.last_support_at,
                "last_review_at": self.last_review_at,
                "sources": _json_dumps(self.sources),
                "evidence_at": _json_dumps(self.evidence_at),
                "evidence_pack": _json_dumps(self.evidence_pack),
                "created_at": self.created_at}

    @staticmethod
    def from_row(row) -> "Profile":
        keys = row.keys() if hasattr(row, "keys") else []
        topic = row["topic"] or ""
        return Profile(
            id=row["id"], topic=topic, subject=row["subject"],
            # 老库补列后是 NULL（回填由 store 的迁移做）——读端先兜 `[topic]`
            topics=((_json_list(row["topics"]) if "topics" in keys else [])
                    or ([topic] if topic else [])),
            statement=row["statement"], evidence=row["evidence"], status=row["status"],
            valid_at=row["valid_at"], invalidated_at=row["invalidated_at"],
            # 两个新列在老库里可能还没补上（迁移是懒执行的），取不到就当空
            invalidated_by=(row["invalidated_by"] if "invalidated_by" in keys else "") or "",
            invalidated_reason=((row["invalidated_reason"]
                                 if "invalidated_reason" in keys else "") or ""),
            last_support_at=row["last_support_at"],
            # 新列在老库里可能还没补上（同 invalidated_by 的容错）
            last_review_at=(row["last_review_at"] if "last_review_at" in keys else "") or "",
            sources=_json_list(row["sources"]),
            # 老库补列时旧行 NULL → 兜成空 dict（回填由 store 的迁移负责）
            evidence_at=(_json_dict(row["evidence_at"])
                         if "evidence_at" in keys else {}),
            evidence_pack=(_json_list(row["evidence_pack"])
                           if "evidence_pack" in keys else []),
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------
# S3 复核记录（记忆整理稿 §五，2026-09-23）
# ---------------------------------------------------------------------

@dataclass
class ProfileReview:
    """一次复核的打分**记录**（每次复核一行，不覆盖历史）。

    为什么落表而不只写 trace：`wrong` 的提议需要一个**有状态的**载体——
    「人处理过没有」是状态，而 trace 是 append-only 的日志，标不了。
    （同 `openings` 的先例：一条记录 + 一个标记。）

    `handled` 只对 `wrong` 有意义——其余打分当轮就已执行完，没有待办。
    """
    id: str = ""
    profile_id: str = ""
    verdict: str = ""                  # 见 REVIEW_VERDICTS
    reason: str = ""                   # 她给的理由（一句话，原样留痕）
    handled: int = 0                   # 1 = 这条提议人处理过了（作废 / 留着）
    created_at: str = ""

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row) -> "ProfileReview":
        return ProfileReview(id=row["id"], profile_id=row["profile_id"],
                             verdict=row["verdict"] or "",
                             reason=row["reason"] or "",
                             handled=row["handled"] or 0,
                             created_at=row["created_at"] or "")


# （`Edge` 模型与「边（四种）」这一段已删，2026-09-24：`edges` 表退役——
#   四种边全改派生，见存储层稿 §五。历史见 git。）


# ---------------------------------------------------------------------
# 备忘录
# ---------------------------------------------------------------------

@dataclass
class Memo:
    """备忘录：`open_loops` 的可提醒子集（短期记忆 §5）。

    两个来源：用户的未完成事 + **air 自己的承诺**（后者管 air 的一致性）。
    时间处理的关键约定：**不让 LLM 估天数**——有明确时间直接抽，
    没明确时间只让 LLM 分类（deadline/progress/idea/followup），周期由系统映射；
    附加的 `timing`（时机类别）同样只分类——2026-10-05 晚收窄到两值：
    `soon`（身体状况：`created + memo.soon_hours` 就算到点）/ `later`（照常）。
    （原来还有 result / after，配一张"什么时候提才自然"的时机表——表删了，
    那两个值不再产出；老数据里留着，读侧不认它、也不报错。）
    """
    id: str = ""
    scene_id: str = ""
    content: str = ""
    # 钩子的**稳定编号**（2026-10-05）：形如 `S1-0042#2`，由代码在写入时生成
    # （`store.add_scene`；指针纪律——模型不写指针）。
    # 它是「变更内容」的前提：闭合/退役靠它定位，不再靠文字全等。
    # 老数据为空 → 退回 content 全等匹配（那正是以前唯一的匹配方式）。
    loop_id: str = ""
    # 同一件事的多步共用一个组名（2026-09-22）：**只用于呈现与提醒收拢**，
    # 闭合仍逐条（哪个了结划掉哪个）；空 = 独立的一件事。
    # 组名**自由生成、不做候选收敛**（他说"随便取"）：未了结的事量小、了结就散，
    # 跨卡裂成两个组名的代价远低于为它上一条 topic 那样的收敛链。
    group_name: str = ""
    kind: str = MEMO_USER_TASK         # 用户的事 / air 的承诺（见 MEMO_KINDS）
    due_at: str = ""                   # 具体时间（用户明说了才有；空 = 无具体时间）
    kind_class: str = ""               # 无具体时间时的分类
    window_days: int = 0               # 由 kind_class 映射（非 LLM 估）
    timing: str = ""                   # 时机类别（MEMO_TIMINGS；空 = 未判，按常规窗口走）
    status: str = MEMO_PENDING         # 待提 / 已提 / 关闭（见 MEMO_STATUS）
    raised_at: str = ""
    # （`last_tried_at`：主动开口的"今天试过"标记——2026-10-05 晚随那套机制删除。
    #   老库那一列还在，只是没人读写了。）
    sensitive: int = 0
    created_at: str = ""

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row) -> "Memo":
        keys = row.keys() if hasattr(row, "keys") else []
        return Memo(
            id=row["id"], scene_id=row["scene_id"], content=row["content"],
            # 新列按「老库可能还没补上」容错（同 timing 的写法）
            loop_id=(row["loop_id"] if "loop_id" in keys else "") or "",
            group_name=(row["group_name"] if "group_name" in keys else "") or "",
            kind=row["kind"], due_at=row["due_at"], kind_class=row["kind_class"],
            window_days=row["window_days"],
            # 两个新列按「老库可能还没补上」容错（同 Profile.from_row 的写法）
            timing=(row["timing"] if "timing" in keys else "") or "",
            status=row["status"],
            raised_at=row["raised_at"],
            sensitive=row["sensitive"],
            created_at=row["created_at"],
        )


# （原 Opening 数据类：主动开口的留言——2026-10-05 晚随那套机制整块删除，
#   表退役留底见 store._retire_openings_table。）
