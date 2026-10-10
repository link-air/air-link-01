"""SQLite 存储封装——**唯一的落库出口**。

为什么所有写操作都要过这里：记忆系统的失败模式里最致命的一条是
「静默丢数据」。把建表、事务、序列化、备份、归档收在一个类里，
才能保证每条写路径都走事务、每个坏文件都不被清空。

三条硬约束，写代码时别绕开：
  1. **S0 不建索引**：原文按天存文档，只能经 `scene_id` 下钻。
     「S0 最难召回」是物理实现的，不是靠调用方自觉。
  2. **系统的自动动作永不硬删画像**：修正填 `invalidated_at`，老化 / 复核 / 封顶
     只降 `status`、不删行；**人否决一条画像 = 真删**（`delete_profile`，2026-09-24）
     ——画像是推断的结论，素材（原文 / 场景 / 摘要）全在，删了能重立。
  3. **归档只标记不删行**：`archived=1` 即进冷层，仍可打捞。
     （S0 例外——它本来就在文件里，见段 4。删除**只**经
     `delete_scene`：摘引用、删行——**不留快照、不留痕**（删干净）：
     留后路是人的另一条路 `archive_scene`，见工具箱稿 §3.4 / §五。）

事务约定：所有写操作走 `with self.conn:`（退出即 commit，异常即 rollback）。
不手动 BEGIN / COMMIT——手写的事务迟早会有一处漏掉 commit。

并发约定：**每线程一个连接**（thread-local）+ WAL。
sqlite3 的连接跨线程用会直接报 `ProgrammingError`——而仪表盘是多线程 HTTP。
所以这里不是「共享一个连接再加锁」，而是各用各的：WAL 让读写不互相阻塞
（没有它，一个长查询会把所有写卡住，而仪表盘是边聊边刷新的）。
并发写的冲突交给 SQLite 自己的锁 + `timeout` 兜底。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L1 存储层
#   上游    ：config（路径与容量参数）、model（数据模型与序列化）
#   下游    ：除 L0 外几乎全部——它是**唯一落库出口**
#   对外入口：`Store` 类；`now_str`（全项目的时间源）、`atomic_write_json`、
#             `append_trace`（留痕的唯一写法）、`scene_ids`（sources 里的 S1）、
#             `RAW_HEAD`（原文文档的小节标题判据——写与读共用，salvage 也认它）
#   边界    ：只管"怎么存"，不判断"该不该存"；不认识 scene/topic/profile 的业务含义
# ---------------------------------------------------------------------
# 本文件分段
#   段 0  模块函数（在类外）—— atomic_write_json / append_trace / scene_ids / 原文分节解析
#   段 1  Store.__init__ / 连接 / 建表 —— 启动与自检（损坏不覆盖）
#   段 2  备份 —— backup_daily（每日一份，留 7 天）
#   段 3  S1 场景卡 CRUD + 计数器 + 全量向量
#   段 4  S0 原文 —— **按天文档，不在库里**（含老库的一次性迁移）
#   段 5  S2 摘要
#   段 6  S3 画像（双时间戳 + topic 版本序列）
#   段 7  边 / 实体索引
#   段 8  备忘录
#   段 9  归档（容量有界，只标记不删行，含被引用保护）
#   段 10 改与删（**只有人能发起**）——删除 = 真删（不留快照，2026-09-23 起
#          也已无"删除前备份"；备份只有段 2 那份每日快照）
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os
import secrets
import shutil
import sqlite3
import threading
import time
from array import array
import re
from datetime import datetime, timedelta
from pathlib import Path

from . import config as cfgmod
from .model import (_json_list, TRIGGER_CLASSES,
                    INVALIDATED_ARCHIVE, INVALIDATED_REVISION,
                    MEMO_CLOSED, MEMO_PENDING, MEMO_RAISED,
                    PROFILE_ESTABLISHED, PROFILE_PENDING, PROFILE_STATUS,
                    Entity, Memo, Profile, ProfileReview, Raw, Scene, Summary)

# ---------------------------------------------------------------------
# 段 0：原子写 JSON
# ---------------------------------------------------------------------

def atomic_write_json(path: Path, data) -> None:
    """原子替换地写一个 JSON 文件。

    借的是求知版 `core/lock.py` 的思路（那套是踩过坑的）：
    随机后缀临时文件 + fsync + os.replace 带退避重试。三件事都有理由——
      - 随机后缀：两个进程同时写同一个目标也各写各的，不会内容交错；
      - fsync：宁可慢几十毫秒，也不要「rename 成功但内容还在缓存」，
        断电后读到空文件被当成记忆清零；
      - 退避重试：Windows 上有人正打开着目标文件时 replace 会直接拒绝访问，
        这是瞬时冲突（实测稳定复现），重试即可。

    调用点不止短期窗口：`shortterm` 落窗口（`data/shortterm.json`）、`settings` 写
    `config.local.json`、`dashboard` 写语音配置——凡是「写坏就丢东西」的 JSON 都走它。
    """
    p = str(path)
    tmp = f"{p}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        delay = 0.005
        for i in range(7):
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                if i == 6:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.15)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def now_str() -> str:
    """当前时间戳（本地时间，秒精度）。全项目统一从这里取时间。

    不用 UTC 的理由：这是「关系记忆」，时间戳是要给人看、
    也是要跟用户说的「上周三」对得上的——本地时间才对得上。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_trace(kind: str, record) -> None:
    """往 `data/trace/<kind>-YYYYMMDD.jsonl` 追加留痕——**全项目 trace 的唯一写法**。

    为什么收在这里：文件名前缀 `kind` 就是仪表盘分派渲染的键
    （`dashboard.App.trace` 按它认这条记录该长什么样），日期决定"同一天一份"——
    两样都不能各模块各写一遍，写歪一处就是「应该有但没显示」那类故障。
    `record` 给一条 dict，或给一列 dict（一次写多行，如批量改动）。

    **永不抛**：留痕失败不拦它记录的那个动作本身。
    """
    try:
        trace_dir = cfgmod.abspath(cfgmod.PATHS["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{kind}-{datetime.now().strftime('%Y%m%d')}.jsonl"
        rows = record if isinstance(record, list) else [record]
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[store] {kind}留痕失败（不影响流程）: {e}")


# 原文文档的小节标题（`## S1-0003 · 2026-09-13 09:30:00`）——**只有它算切分点**。
#
# 判据必须是这个形状，不能是「行首 `## `」：正文里常有大段粘贴（他贴过 README、
# 贴过人格设定、贴过原文文档本身），那些自带的 `## 小标题` 会把一段对话切成七八节
# （2026-10-07 事故：历史回填里冒出「跑起来」「目录」这种空段，真正的 S1-0043
# 只剩开头三句）。写（`add_raw`）与读（`_raw_doc_section` / `salvage._sections`）
# 共用这一条——各写一份就会漂移。
RAW_HEAD = re.compile(r"^## (S\d+-\d+) · ", re.M)


def _raw_doc_section(text: str, scene_id: str) -> tuple[str, str] | None:
    """从一份按天文档里取出某个场景的那一节，返回 `(时间, 正文)`。

    时间来自小节标题（`## S1-0003 · 2026-09-13 09:30:00`）。
    它以前被丢掉，于是打捞重建出来的场景时间只能取"重建那一刻"——
    而原文里明明写着它是什么时候发生的。正文为空时返回 None
    （有标题没内容 = 这一节没有可用的原文；同一个编号有好几节时，
    取**有内容的**那一节——空节挡不住后面那段）。

    S0 没有索引，所以「查」就是把文档读进来找那一节——文件很小（一天一份），
    而这是 S0 唯一的检索方式，慢一点是设计的一部分（「S0 最难召回」）。

    切分只认 `RAW_HEAD`：正文里粘贴的 `## 小标题` 不是标题
    （见 `RAW_HEAD` 那一段，2026-10-07 事故）。
    """
    heads = list(RAW_HEAD.finditer(text or ""))
    for i, m in enumerate(heads):
        if m.group(1) != scene_id:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        head = text[m.end():end]
        when = head.split("\n", 1)[0].strip()      # 标题剩下的部分就是时间戳
        # 第一行是标题剩下的时间戳，内容从第二行起
        body = head.split("\n", 1)[1] if "\n" in head else ""
        if not body.strip():
            continue
        return when, body.strip()
    return None


def _safe_raw_body(content: str) -> str:
    """正文里**长得像小节标题的那几行**缩进两个空格再落盘。

    他会把整份 README、整份人格设定贴进对话（2026-10-07 那次贴的就是本仓库的
    README），里面自带 `## 小标题`；按 `RAW_HEAD` 判的话，只有恰好写成
    `## S1-0043 · …` 这种形状的才会被误认——真撞上了就把一节切两半。

    **只缩进、不改字**：缩进后不再是行首 `## `，读的时候不当标题；
    而 markdown 里两格缩进仍然是同一个标题，显示不变。改字不行——S0 是原文，
    贴进来是什么就得存什么（唯一动的那两格空格，是把"存不下"这件事本身
    记进格式里，不是改内容）。
    """
    return "\n".join(("  " + ln if RAW_HEAD.match(ln) else ln)
                     for ln in (content or "").splitlines())


def _add_days(stamp: str, days: int) -> str:
    """时间戳 + N 天（字符串进、字符串出，格式统一 YYYY-MM-DD HH:MM:SS）。

    直接在字符串上比大小是可行的（这个格式是按字典序可比设计的），
    但**加法必须过 datetime**——手写日期进位迟早错在月底和闰年上。
    """
    try:
        base = datetime.strptime((stamp or "")[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        base = datetime.now()
    return (base + timedelta(days=int(days))).strftime("%Y-%m-%d %H:%M:%S")


# 建表 SQL。
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scenes (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    keywords      TEXT,
    text          TEXT,
    emb           BLOB,
    time_event    TEXT,
    time_record   TEXT,
    valence       INTEGER,
    arousal       INTEGER,
    intensity     REAL,
    subject       TEXT,
    topic         TEXT,
    self_ref      INTEGER DEFAULT 0,
    cited_by_profile INTEGER DEFAULT 0,
    mention_count    INTEGER DEFAULT 0,
    sensitive     INTEGER DEFAULT 0,
    trigger       TEXT,
    trigger_class TEXT,
    open_loops    TEXT,
    air_stance    TEXT,
    reaction      TEXT,
    outcome       TEXT,
    source        TEXT,
    last_mention_at TEXT,
    archived      INTEGER DEFAULT 0,
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_scenes_time ON scenes(time_event);
CREATE INDEX IF NOT EXISTS idx_scenes_subject ON scenes(subject);
CREATE INDEX IF NOT EXISTS idx_scenes_archived ON scenes(archived);

-- S0 原文**不在库里**：它按天写成文档（`data/raws/2026-09-13.md`，追加写、按场景分节）。
-- 理由（存储层稿"S0 的落点"，2026-09-13 补）：原文是最大的数据，不该跟结构化记忆挤在一个库里；
-- 而且**冷层本该是文件**——可直接看、可直接删，占空间的那部分当场就能清。
-- 库里只留一个指针边（kind='raw'），指到那天的文档。
-- 老库里已有的 `raws` 表会在启动时被导成文档并改名留底（见 `_migrate_raws_to_files`）。

CREATE TABLE IF NOT EXISTS summaries (
    id        TEXT PRIMARY KEY,
    topic     TEXT,
    text      TEXT,
    sources   TEXT,
    archived  INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS profiles (
    id             TEXT PRIMARY KEY,
    topic          TEXT,
    subject        TEXT,
    statement      TEXT,
    evidence       INTEGER DEFAULT 0,
    status         TEXT,
    valid_at       TEXT,
    invalidated_at TEXT,
    invalidated_by TEXT,
    invalidated_reason TEXT,
    last_support_at TEXT,
    sources        TEXT,
    evidence_at    TEXT,
    evidence_pack  TEXT,
    created_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_profiles_topic ON profiles(topic);
CREATE INDEX IF NOT EXISTS idx_profiles_subject ON profiles(subject);
CREATE INDEX IF NOT EXISTS idx_profiles_valid ON profiles(invalidated_at);

-- 基础档案：**用户明说的事实**（称呼 / 年龄 / 性别 / 城市 / 职业）。
-- 和 `profiles` 严格分开，因为两者性质不同：
--   profiles 是 **air 的推断** —— 要印证、会过期、可否决、带不确定性
--   user_facts 是 **他自己说的** —— 无需印证、不会老化、就是事实
-- 混在一起的后果很具体：「他叫老张」会被当成一条待验证的假设等 3 次印证，
-- 而且会因为"长期未被提及"被 age_out_profiles 降级回待验证——
-- 称呼会因为你几天没提而失效。
CREATE TABLE IF NOT EXISTS user_facts (
    key        TEXT PRIMARY KEY,   -- 称呼 / 年龄 / 性别 / 城市 / 职业（可自定义）
    value      TEXT,
    source     TEXT,               -- 恒为 user_stated（档案只由人手工填；
                                   -- 旧设计里 air 自己推断那一路随 `remember_fact` 一起撤了）
    note       TEXT,               -- 出处（哪句话说的，便于回看与纠错）
    updated_at TEXT
);

-- 用户**偏好**：他想要什么，而不是他是什么。
-- 与 `user_facts` 分开的理由同档案与画像：`facts` 是事实（他叫老张），
-- 这里是他定的设置（语言 / 人格 / 声音这些）。
-- 混进 facts 会在注入时渲染错位置——事实排在画像前，偏好不该在那儿。
--
-- ⚠️ **系统不自动改这里的值**：他定了就定了，关掉页面、重启都还在。
-- 自作主张地改等于系统覆盖人的选择（同否决那条理由）。
CREATE TABLE IF NOT EXISTS user_prefs (
    key        TEXT PRIMARY KEY,   -- lang / persona 等（有哪些键由使用点决定）
    value      TEXT,
    updated_at TEXT
);

-- （`edges` 表已退役，2026-09-24：四种边全部改派生，见存储层稿 §五——
--   引用 = `sources`，相邻 = `topic` + 时间现算，原文 = 按天文档，
--   保护期的"引用起点时间" = `profiles.evidence_at`。
--   老库的边在启动时改名 `edges_retired` 留底，不删——见 `_retire_edges_table`。）

CREATE TABLE IF NOT EXISTS entities (
    id      TEXT PRIMARY KEY,
    name    TEXT,
    kind    TEXT,
    aliases TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS scene_entities (
    scene_id  TEXT,
    entity_id TEXT,
    relation  TEXT,
    PRIMARY KEY (scene_id, entity_id)
);

CREATE TABLE IF NOT EXISTS memos (
    id          TEXT PRIMARY KEY,
    scene_id    TEXT,
    content     TEXT,
    -- 钩子的稳定编号（2026-10-05）：`S1-0042#2`，写入时由代码生成（见 add_scene）
    loop_id     TEXT,
    group_name  TEXT,
    kind        TEXT,
    due_at      TEXT,
    kind_class  TEXT,
    window_days INTEGER,
    timing      TEXT,
    status      TEXT,
    raised_at   TEXT,
    -- （`last_tried_at`：主动开口的"今天试过"标记，2026-10-05 晚删——老库里那列还在）
    sensitive   INTEGER DEFAULT 0,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_memos_status ON memos(status);

-- （主动开口的 `openings` 表 2026-10-05 晚退役：那套"她先开一句 + 留言"
--   整块删了，见待优化稿 K 条。老库的表在启动时改名 `openings_retired` 留底，
--   不删——见 `_retire_openings_table`。新库不建这张表。）

-- S3 复核记录（记忆整理稿 §五，2026-09-23）：每次复核一行；wrong 的提议等人处理
CREATE TABLE IF NOT EXISTS profile_reviews (
    id          TEXT PRIMARY KEY,
    profile_id  TEXT,
    verdict     TEXT,
    reason      TEXT,
    handled     INTEGER DEFAULT 0,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_profile_reviews_pending ON profile_reviews(verdict, handled);

-- 系统自己的状态（体检时刻这类；2026-09-23 加）。
-- 和 `user_prefs` 分开：那边装「他想要什么」（只有人能改，见 set_pref 的注释），
-- 这里装系统的账——系统自己读写，两边不混。
CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);
"""

# 表名 → id 前缀。只允许白名单里的表参与 `_next_id`，
# 表名会拼进 SQL，绝不能来自外部输入。
_ID_PREFIX = {
    "scenes": "S1", "summaries": "S2", "profiles": "S3",
    "entities": "EN", "memos": "M",
    "profile_reviews": "PR",
    # （`openings`: "OP" 随主动开口一起删，2026-10-05 晚）
}


def scene_ids(ids) -> list[str]:
    """从一组编号里挑出 S1 的——`sources` 的下钻口径（S2 是聚合、S3 是判断，
    都不是直接素材）。放这里是因为它认的正是上面那张前缀表。"""
    return [i for i in (ids or []) if str(i).startswith(_ID_PREFIX["scenes"])]


# 场景**可改字段**的白名单与中文名（2026-09-24，工具箱稿 §七）——留痕 / 界面 / 工具共用一份。
#
#   - 可改 = **语义描述**类：他纠正"不是这样"有明确含义的那些。
#   - **不开**的（各有理由）：`time_record`（S0 原文按天文档的落点、`memory_search`
#     的 `when` 过滤键——改了原文就下钻不到）；`valence` / `arousal` / `intensity`
#     （"当时的行为 / 情绪信号"是**算出来的**，人改它等于伪造证据，不是纠正理解）；
#     `mention_count` / `cited_by_profile`（系统账）；`archived` / `subject`
#     （各有专门动作）；实体关联（正路是实体页的合并 / 别名）。
SCENE_EDITABLE = ("title", "text", "topic", "time_event",
                  "trigger", "reaction", "trigger_class")
SCENE_FIELD_LABELS = {"title": "标题", "text": "摘要", "topic": "主题",
                      "time_event": "事件时间", "trigger": "情境",
                      "reaction": "反应", "trigger_class": "情境类"}
# 名字 → 字段键（**她说的和界面填的都走这一份**，2026-09-24）：中文名、英文键、
# 以及几个口语说法（"说法" = 摘要、"时间" = 事件时间）。
SCENE_FIELD_ALIASES = {v: k for k, v in SCENE_FIELD_LABELS.items()}
SCENE_FIELD_ALIASES.update({
    "说法": "text", "时间": "time_event", "when": "time_event",
    "summary": "text", "trigger": "trigger", "text": "text",
    "标签": "topic",
})
# 规范键也认自己（2026-09-24 夜核对修）：`revise_memory` 的提议里存的 `field`
# 就是规范键（S3 默认 `statement`），确认条把它**原样**递给 `weave.update_by_layer`
# ——别名表认不回自己的键，那个提议就永远改不成（实测：S3 陈述经确认条报
# 「不认的字段「statement」」；英文键 `title` / `topic` 同理）。三张表统一补齐。
SCENE_FIELD_ALIASES.update({k: k for k in SCENE_EDITABLE})

# 摘要（S2）与画像（S3）能改的**语义字段**（2026-09-24，三层同一套的「改」）：
# 主文本（S2 叙述 / S3 陈述）+ 主题标签（topic）。数值 / 系统字段一律不开——
# S3 的 status / evidence 是算出来的，人改它等于伪造印证（同场景那条纪律）。
SUMMARY_EDITABLE = ("text", "topic")
SUMMARY_FIELD_LABELS = {"text": "叙述", "topic": "主题"}
PROFILE_EDITABLE = ("statement", "topic")
PROFILE_FIELD_LABELS = {"statement": "陈述", "topic": "主题"}

# 字段名 → 键：中文标签、英文键、以及口语说法。"**说法**"这类词是**按层**认的
# （场景里 = 摘要，画像里 = 陈述）——同一句口语在不同层指的不是同一个字段，
# 所以这张表按层各一份，不能合成全局一份。
SUMMARY_FIELD_ALIASES = {v: k for k, v in SUMMARY_FIELD_LABELS.items()}
SUMMARY_FIELD_ALIASES.update({"摘要": "text", "说法": "text", "summary": "text",
                              "text": "text", "标签": "topic", "topic": "topic"})
SUMMARY_FIELD_ALIASES.update({k: k for k in SUMMARY_EDITABLE})
PROFILE_FIELD_ALIASES = {v: k for k, v in PROFILE_FIELD_LABELS.items()}
PROFILE_FIELD_ALIASES.update({"说法": "statement", "摘要": "statement",
                              "summary": "statement", "text": "statement",
                              "标签": "topic", "topic": "topic"})
PROFILE_FIELD_ALIASES.update({k: k for k in PROFILE_EDITABLE})

# 三层的「改」规则（一个入口收三种编号）：可改字段 / 展示名 / 字段别名 /
# 主文本字段（不给 field 时的默认）。`weave.update_by_layer` 与
# `tools._run_revise_memory` 都读这一份——两处各写一套必然漂移（2026-09-24）。
LAYER_EDIT = {
    "S1": {"editable": SCENE_EDITABLE, "labels": SCENE_FIELD_LABELS,
           "aliases": SCENE_FIELD_ALIASES, "main": "text"},
    "S2": {"editable": SUMMARY_EDITABLE, "labels": SUMMARY_FIELD_LABELS,
           "aliases": SUMMARY_FIELD_ALIASES, "main": "text"},
    "S3": {"editable": PROFILE_EDITABLE, "labels": PROFILE_FIELD_LABELS,
           "aliases": PROFILE_FIELD_ALIASES, "main": "statement"},
}


def layer_edit(sid: str) -> dict:
    """某层的「改」规则（认不出前缀当 S1——**默认层兜底**而已；
    编号纪律的闸在调用方：M 在 `tools._run_revise_memory` /
    `weave.update_by_layer` 被当场拒绝，走不到这里）。"""
    up = (sid or "").strip().upper()
    for p in ("S1", "S2", "S3"):
        if up.startswith(p):
            return LAYER_EDIT[p]
    return LAYER_EDIT["S1"]


# ---- 主题标签（1-3 个，2026-09-24 晚）----
#
# S2 / S3 的「主题」是**多标签**：第一个是**主主题**（链路键——S2 按它聚合、
# S3 的版本序列认它），其余 0-2 个是**附加主题**（这条记忆顺带沾到的话题）。
# 附加主题只用于「按主题找」与展示；链路（聚合 / 印证 / 版本）只认主主题。

TOPICS_MAX = 3          # 1 主 + 最多 2 附：给多了是标签泛滥，检索反而变糊


def split_topics(raw: str) -> list[str]:
    """把「她 / 他给的主题串」切成主题列表（逗号 / 顿号 / 分号 / 空格都认）。

    **第一个 = 主主题**（写入端 `_write_topics` 按它同步 `topic` 列）。
    **这里不截断**：个数上限交给 `_write_topics` 拒绝（> `TOPICS_MAX` 报错）——
    截断是静默丢弃，会让人以为"四个都挂上了"（2026-09-24 检查修）。
    去重按原样字符串（主题是中文 /「主语·主题」，不做大小写折叠）。
    """
    out: list[str] = []
    for x in re.split(r"[,，、;；\s]+", (raw or "").strip()):
        x = x.strip()
        if x and x not in out:
            out.append(x)
    return out


def load_str_list(raw) -> list[str]:
    """把一列 JSON 数组文本读成 `list[str]`（NULL / 坏值 → `[]`）。"""
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except Exception:
        return []
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()]

# 建表之后补的列（表名 → {列名: 类型}）。
#
# 为什么不直接改 `SCHEMA_SQL` 了事：`CREATE TABLE IF NOT EXISTS` 对**已存在的**
# 表什么都不做——老库不会因为改了建表语句就多出一列，然后在 INSERT 时炸掉。
# 记忆库是不可重建的，能加列就绝不重建表，所以走 ALTER。
_MIGRATIONS = {
    "profiles": {
        # 区分「被新证据修正」和「被用户否决」——两者的后续处理完全不同：
        # 前者是 air 自己更新认知，后者是人对系统的纠正，得能分开统计与呈现。
        "invalidated_by": "TEXT",
        "invalidated_reason": "TEXT",
        # 形成依据的快照（追溯的底线保障，见 Profile.evidence_pack）
        "evidence_pack": "TEXT",
        # 「引用起点时间」（id → 时刻，2026-09-24）：原来住在 `edges` 的 evidence 边里
        # ——那条边是对 `sources` 的冗余存储，只有这个时间戳是真数据（保护期起算用）。
        # 老库补列后为 NULL → `from_row` 兜空 dict，**回填**由 `_backfill_evidence_at` 做。
        "evidence_at": "TEXT",
        # 最近一次「复核」的时刻（记忆整理稿 §五）：只做复核间隔防抖用；
        # 老库补列时旧行是 NULL → `Profile.from_row` 兜成空串（= 从没复核过，优先复核）。
        "last_review_at": "TEXT",
        # 主题标签（2026-09-24 晚）：同 `summaries.topics`——1-3 个，
        # **第一个 = 主主题**（版本序列认它）；回填见 `_backfill_topics`。
        "topics": "TEXT",
    },
    "memos": {
        # 钩子的稳定编号（2026-10-05）：`S1-0042#2`，写入时由代码生成。老库补列后
        # 旧行是 NULL → `Memo.from_row` 兜成空串 → 闭合/退役退回「content 全等」。
        "loop_id": "TEXT",
        # 时机类别（他明说的时间 / 状况类的小时粒度——**原来的时机表已删**，2026-10-05 晚）
        "timing": "TEXT",
        # （原 `last_tried_at`：主动开口的「试过」时间——随那套机制删除，2026-10-05 晚。
        #   这里不再补列；老库里那一列还在，只是没人读写。）
        # 同一件事的多步共用一个组名（2026-09-22）：只用于呈现与提醒收拢，
        # 闭合仍逐条。老库补这一列时旧行是 NULL → `Memo.from_row` 兜成空串。
        "group_name": "TEXT",
    },
    "summaries": {
        # 主题标签（2026-09-24 晚）：1-3 个，**第一个 = 主主题**（与 `topic` 列同步：
        # topic 是聚合 / 版本序列的键，`topics` 是多标签，供「按主题找」与展示）。
        # 老库补列后为 NULL → `Summary.from_row` 兜 `[topic]`，回填见 `_backfill_topics`。
        "topics": "TEXT",
    },
    "scene_entities": {
        # 这条场景里他和该实体的关系（2026-09-25，如「妈妈」「同事」）：
        # 逐场景存（同一人不同场景角色不同），实体页展示用；老库补列后为 NULL
        # → 空关系，不影响旁路（旁路只认名字）。
        "relation": "TEXT",
    },
}


class Store:
    """SQLite 封装。构造即建表；库文件损坏时**报错而不是清空**（见 `_verify_not_corrupt`）。

    这里同时也是**唯一的落库出口**：业务代码不写 SQL、不碰 sqlite3，
    表的形状变了只改这一处。
    """

    def __init__(self, db_path: Path | str | None = None):
        """打开（必要时新建）库并建表。`db_path` 只给测试用，业务代码别拼路径。"""
        cfgmod.ensure_dirs()
        self.path = Path(db_path) if db_path else cfgmod.db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # 向量表缓存（2026-09-24，存储层稿 §七"相似度的规模"）：原来是**每次唤醒
        # 全量拉两遍**（`compute_cues` 算 C1 一遍、`_top_by_embedding` 算 R1 一遍）
        # 且逐条解码 BLOB。写时失效（`_bump_emb`），版本号防"加载途中被改写"的竞态。
        self._emb_cache: list[tuple[str, list[float]]] | None = None
        self._emb_stamp = 0
        self._verify_not_corrupt()
        self._ensure_schema()

    @property
    def conn(self):
        """当前线程的连接（**thread-local**）。

        sqlite3 的连接不能跨线程用（默认 `check_same_thread=True`）——
        而仪表盘是多线程 HTTP，每个请求都在新线程里。所以这里是「每线程一个连接」
        而不是共享一个。

        单线程用法（测试 / 批处理 / demo）完全不受影响：只会创建那一个连接。
        `timeout=10` 是给并发写留的余地；WAL 模式让读写不互相阻塞
        （没有它，一个长查询会把所有写卡住——而仪表盘会边聊边刷新）。
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass          # 只读介质等情况：WAL 开不了也能继续跑
            self._local.conn = conn
        return conn

    # ---- 段 1：连接与自检 ----

    def _verify_not_corrupt(self) -> None:
        """开机自检：库文件坏了就**备份 + 报错**，绝不静默重建。

        静默重建 = 记忆清零，而记忆恰恰是这个项目里唯一不可重建的东西。
        宁可启动失败让人来处理，也不要「跑起来了但她在失忆」。
        """
        try:
            self.conn.execute("PRAGMA schema_version").fetchone()
            self.conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as e:
            corrupt = self.path.with_suffix(self.path.suffix + f".corrupt-{int(time.time())}")
            try:
                self.conn.close()
                shutil.move(str(self.path), str(corrupt))
            except Exception:
                pass
            raise RuntimeError(
                f"[store] 数据库损坏，已原样保留为 {corrupt.name}，拒绝重建（不覆盖）。原始错误: {e}"
            ) from e

    def _ensure_schema(self) -> None:
        """建表 + 补列 + 把老库的 S0 段落迁出去。顺序不能换：先补列才能迁移。"""
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)
            self._migrate()
        self._migrate_raws_to_files()
        self._backfill_evidence_at()
        self._backfill_topics()
        self._retire_edges_table()      # 顺序不能反：先回填、再退役（见那个方法）
        self._retire_openings_table()   # 2026-10-05 晚：主动开口整块删了，表留底

    def _migrate(self) -> None:
        """给老库补列（幂等：已经有的列跳过）。

        **只加列，不改列、不重建**：记忆库不可重建，任何"重建表"的方案
        都得先把数据搬出来——那正是最容易丢数据的一步。
        """
        for table, columns in _MIGRATIONS.items():
            existing = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns.items():
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def _backfill_topics(self) -> None:
        """老库回填 `topics`（2026-09-24 晚）：`[topic]`——主主题本来就是它。

        只填**空**的（幂等）：新写入由 `_write_topics` 同步两列，重跑不会覆盖；
        `topic` 为空的脏行跳过（没什么可回填的）。
        """
        for table in ("summaries", "profiles"):
            rows = self.conn.execute(
                f"SELECT id, topic, topics FROM {table} "
                f"WHERE COALESCE(topic, '') <> ''").fetchall()
            pending = [(r["id"], r["topic"]) for r in rows
                       if not load_str_list(r["topics"])]
            if not pending:
                continue
            with self.conn:
                for rid, topic in pending:
                    self.conn.execute(
                        f"UPDATE {table} SET topics = ? WHERE id = ?",
                        (json.dumps([topic], ensure_ascii=False), rid))

    def close(self) -> None:
        """关掉**当前线程**的连接（其他线程的由它们自己关，或进程退出时释放）。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        """context manager 退出时**只关当前线程**的连接（多线程下另外几个还活着）。"""
        self.close()

    # ---- 段 2：备份 ----

    def backup_daily(self, keep_days: int = 7) -> Path | None:
        """每天一份快照，留 keep_days 天。当天已有则跳过（幂等）。

        用 `VACUUM INTO`（SQLite 原生在线备份）：它在事务一致点生成副本，
        不像文件拷贝那样可能拷到「写了一半」的库。失败退回**在线备份 API**
        （`sqlite3.Connection.backup`）——同样不退回文件拷贝，理由见下面那段。

        调用点只有一个：仪表盘启动（`dashboard.App.__init__`）。
        「每日一份」靠这个方法自己的幂等性（当天已有就跳过），不需要定时器。
        """
        stamp = datetime.now().strftime("%Y%m%d")
        backup_dir = cfgmod.abspath(cfgmod.PATHS["backup_dir"])
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / f"air_link-{stamp}.db"
        if target.exists():
            return None
        try:
            self.conn.execute("VACUUM INTO ?", (str(target),))
        except Exception:
            # 兜底用标准库的**在线备份 API**——不要退回 `shutil.copy2`：
            # WAL 模式下已提交的事务可能还在 `-wal` 文件里没并进主库，
            # 只拷 `.db` 会拷出一个"少了最近事务"的旧库（而备份存在的全部意义就是完整）。
            try:
                if target.exists():
                    target.unlink()
                dest = sqlite3.connect(str(target))
                try:
                    self.conn.backup(dest)
                finally:
                    dest.close()
            except Exception as e:
                print(f"[store] 备份失败（不阻塞主流程）: {e}")
                return None
        # 清理过期备份
        cutoff = datetime.now() - timedelta(days=keep_days)
        for old in backup_dir.glob("air_link-*.db"):
            try:
                if datetime.strptime(old.stem.split("-")[-1], "%Y%m%d") < cutoff:
                    old.unlink()
            except Exception:
                continue
        return target

    # ---- 内部：id 生成 ----

    def _next_id(self, table: str) -> str:
        """生成形如 `S1-0001` 的顺序 id（按当前最大值 +1，归档不删行也不冲突）。

        ⚠️ 排序必须按**数值**，不能按字符串：`id` 是等宽补零的
        （`S1-0001`…`S1-9999`），字典序一直是对的——直到第 10000 条：
        `"S1-10000" < "S1-9999"`（逐字符比到第 4 位时 `1` < `9`），
        `ORDER BY id DESC` 会取到 `S1-9999`，算出 `S1-10000` → 主键重复、
        INSERT 直接炸。归档只标记不删行，所以总行数会一直涨——
        这条线迟早会被跨过去，修在这里一次到位。
        """
        prefix = _ID_PREFIX[table]
        row = self.conn.execute(
            f"SELECT id FROM {table} WHERE id LIKE ?"
            f" ORDER BY CAST(substr(id, {len(prefix) + 2}) AS INTEGER) DESC LIMIT 1",
            (prefix + "-%",),
        ).fetchone()
        n = 1
        if row:
            try:
                n = int(str(row["id"]).split("-")[-1]) + 1
            except ValueError:
                n = self.count(table) + 1
        return f"{prefix}-{n:04d}"

    def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        """表行数（table 走白名单校验，防止拼 SQL 被注入）。"""
        if table not in _ID_PREFIX:
            raise ValueError(f"未知表名: {table}")
        sql = f"SELECT count(*) AS n FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.conn.execute(sql, params).fetchone()["n"])

    # ---- 段 3：S1 场景卡 ----

    @staticmethod
    def _vec_to_blob(vec: list[float] | None) -> bytes | None:
        """float 列表 → BLOB（float32 序列化）。

        用 float32 而不是 JSON 文本：几千条 × 1536 维，JSON 会膨胀数倍，
        而且每次读都要 parse——向量只用来算余弦，不需要精确到双精度。
        """
        if not vec:
            return None
        return array("f", [float(x) for x in vec]).tobytes()

    @staticmethod
    def _blob_to_vec(blob) -> list[float] | None:
        """BLOB → float 列表（`_vec_to_blob` 的反向）。"""
        if not blob:
            return None
        a = array("f")
        a.frombytes(blob)
        return list(a)

    def add_scene(self, scene: Scene) -> str:
        """写入一条 S1。若 scene.id 为空则分配新 id 并回填。"""
        if not scene.id:
            scene.id = self._next_id("scenes")
        if not scene.created_at:
            scene.created_at = now_str()
        if not scene.time_record:
            scene.time_record = scene.created_at
        # 给钩子编稳定号（2026-10-05）：`S1-0042#2`——**由代码生成**（指针纪律：
        # 模型不写指针）。它是「变更内容」的前提：闭合/退役按号定位，不再靠文字全等。
        # 只补空缺（已有编号的不动）；老钩子没有编号，退回 content 全等匹配。
        for i, loop in enumerate(scene.open_loops or [], start=1):
            if isinstance(loop, dict) and not str(loop.get("loop_id") or "").strip():
                loop["loop_id"] = f"{scene.id}#{i}"
        row = scene.to_row()
        cols = ", ".join(row.keys()) + ", emb"
        ph = ", ".join(["?"] * (len(row) + 1))
        with self.conn:
            self.conn.execute(
                f"INSERT INTO scenes ({cols}) VALUES ({ph})",
                tuple(row.values()) + (self._vec_to_blob(scene.emb),),
            )
        self._bump_emb()        # `archived=0` 的集合变了（哪怕这条没向量）
        return scene.id

    def get_scene(self, scene_id: str) -> Scene | None:
        """取一条 S1（含向量）；没有则返回 None。"""
        row = self.conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
        return self._scene_from_row(row) if row else None

    def _scene_from_row(self, row) -> Scene:
        """行 → Scene。向量单独带出来（它不在 `Scene.to_row` 里，被故意 pop 掉了）。"""
        return Scene.from_row(row, emb=self._blob_to_vec(row["emb"]))

    def query_scenes(self, subject: str | None = None, topic: str | None = None,
                     since: str | None = None, until: str | None = None,
                     valence: int | None = None, arousal: int | None = None,
                     min_intensity: float | None = None,
                     include_archived: bool = False,
                     limit: int | None = None) -> list[Scene]:
        """按维度查场景卡（C1–C6 的存储侧比对都走这里）。

        **刻意不提供 `sensitive` 过滤**：敏感场景照常参与召回——
        召回是理解的前提；「敏感」只在说出口时调节力度、在镜像呈现时排除。
        加这个参数就等于把「不提」做成「不召回」，那是设计稿推翻过的错误。

        时间过滤用 `COALESCE(NULLIF(time_event,''), created_at)`：
        有的场景 LLM 没抽出明确事件时间，退回记录时间比直接漏掉它好。
        """
        sql = ["SELECT * FROM scenes WHERE 1=1"]
        params: list = []
        if not include_archived:
            sql.append("AND archived = 0")
        if subject is not None:
            sql.append("AND subject = ?")
            params.append(subject)
        if topic is not None:
            sql.append("AND topic = ?")
            params.append(topic)
        tcol = "COALESCE(NULLIF(time_event,''), created_at)"
        if since:
            sql.append(f"AND {tcol} >= ?")
            params.append(since)
        if until:
            sql.append(f"AND {tcol} <= ?")
            params.append(until)
        if valence is not None:
            sql.append("AND valence = ?")
            params.append(valence)
        if arousal is not None:
            sql.append("AND arousal = ?")
            params.append(arousal)
        if min_intensity is not None:
            sql.append("AND intensity >= ?")
            params.append(min_intensity)
        sql.append("ORDER BY " + tcol + " DESC")
        if limit:
            sql.append("LIMIT ?")
            params.append(limit)
        rows = self.conn.execute(" ".join(sql), tuple(params)).fetchall()
        return [self._scene_from_row(r) for r in rows]

    def scenes_by_entities(self, names: list[str], limit: int = 10) -> list[Scene]:
        """实体**旁路**：按实体名 / 别名直接定位场景（唤醒层 §5）。

        不走语义检索——消息里出现「小明」时，最可靠的线索就是「小明」这两个字。
        同一个场景命中多个实体也只返回一次（DISTINCT）。

        ⚠️ **别名必须在 Python 端比**：`aliases` 列存的是 JSON 数组文本
        （`["小明","明明"]`），拿单个名字去 `IN` 一整个 JSON 串永远不相等
        ——原先那句 `e.aliases IN (...)` 是**静默失效**的：不报错、只是查不到。
        实体表很小（几十条），全表取回来逐个比对，比在 SQL 里拼 LIKE 更直白。
        """
        clean = [n.strip() for n in (names or []) if n and n.strip()]
        if not clean:
            return []
        wanted = {n.lower() for n in clean}

        matched: set[str] = set()
        for r in self.conn.execute("SELECT id, name, aliases FROM entities").fetchall():
            if (r["name"] or "").strip().lower() in wanted:
                matched.add(r["id"])
                continue
            for alias in _json_list(r["aliases"]):
                if (alias or "").strip().lower() in wanted:
                    matched.add(r["id"])
                    break
        if not matched:
            return []

        marks = ",".join(["?"] * len(matched))
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT s.* FROM scenes s
            JOIN scene_entities se ON se.scene_id = s.id
            WHERE s.archived = 0 AND se.entity_id IN ({marks})
            ORDER BY s.time_record DESC
            LIMIT ?
            """,
            tuple(matched) + (limit,),
        ).fetchall()
        return [self._scene_from_row(r) for r in rows]

    def all_embeddings(self, include_archived: bool = False) -> list[tuple[str, list[float]]]:
        """全量加载向量——**进程内缓存**（2026-09-24）。

        几千条 × 1536 维约几十 MB，完全可接受，省掉一个依赖；量大再换 ANN 时
        接口形状不用变（这条不变）。

        缓存为什么值得做：一次唤醒会调它**两遍**（`compute_cues` 算 C1、
        `_top_by_embedding` 算 R1），每遍都把全表 BLOB 解成 float 列表——
        库上千条之后，那是每轮对话都在白烧的时间。
        `include_archived=True`（调试 / 打捞视角）**不走缓存**——形状不同。
        """
        if self._emb_cache is not None and not include_archived:
            return self._emb_cache
        seen = self._emb_stamp
        sql = "SELECT id, emb FROM scenes WHERE emb IS NOT NULL"
        if not include_archived:
            sql += " AND archived = 0"
        out = []
        for r in self.conn.execute(sql).fetchall():
            vec = self._blob_to_vec(r["emb"])
            if vec:
                out.append((r["id"], vec))
        # 加载途中若有人写过（`_bump_emb` 动过版本号），这一份就不入缓存——
        # 否则会把"旧快照"盖回去，检索持续看不到刚写的那条。
        if not include_archived and seen == self._emb_stamp:
            self._emb_cache = out
        return out

    def _bump_emb(self) -> None:
        """向量表缓存失效——**任何会改 `scenes.emb` 或 `archived` 的写**之后都要调。

        为什么宁可多失效：漏失效的后果是"检索到已删 / 已归档的，或漏掉刚写入的"
        ——静默且难查；多失效的代价只是下次重拉一遍（毫秒级）。
        """
        self._emb_stamp += 1
        self._emb_cache = None

    def bump_mention(self, scene_id: str, at: str = "") -> None:
        """`mention_count` +1（**这条场景在对话中被提及**），并更新 last_mention_at。

        设上限（`rank.mention_cap`）是设计约束：念叨多 ≠ 重要，
        不封顶的话一条被反复提及的场景会永远霸占预算（反刍）。
        注意：**内部召回不算提及**，别在召回路径上调这个。
        """
        at = at or now_str()
        cap = cfgmod.cfg("rank", "mention_cap", default=10)
        with self.conn:
            self.conn.execute(
                "UPDATE scenes SET mention_count = MIN(mention_count + 1, ?),"
                " last_mention_at = ? WHERE id = ?",
                (cap, at, scene_id),
            )

    def bump_cited(self, scene_id: str, at: str = "") -> None:
        """`cited_by_profile` +1（**画像引用这条场景**）——重要性，**不设上限**。

        它和 mention_count 是两个不同指标，递增时机也不同，别合并
        。
        这里只管计数：画像的 `last_support_at` 由调用方用 `touch_profile` 更新——
        计数和「谁获得支持」是两件事，混在一起 store 就得反查画像，职责就乱了。
        """
        with self.conn:
            self.conn.execute(
                "UPDATE scenes SET cited_by_profile = cited_by_profile + 1 WHERE id = ?",
                (scene_id,),
            )

    # （`set_scene_archived` 原来在这里——2026-09-26 复核发现全库零调用，且
    #   是无条件改 `archived`、**不做 `_bump_emb` 失效**（唯一的漏网点）——
    #   已删；程序化开关用 `archive_scene()` / `unarchive_scene()`，那两个带
    #   幂等条件（`archived = 0/1`）也会让向量缓存失效。）

    # ---- 段 4：S0 原文（按天文档，不在库里）----

    def raws_dir(self) -> Path:
        """原文文档目录（`data/raws/`），确保存在后返回。"""
        p = cfgmod.abspath(cfgmod.PATHS["raws_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    def add_raw(self, raw: Raw, on_date: str = "") -> str:
        """把一段原文追加进**那天**的文档，返回日期（`2026-09-13`）。

        `on_date` 用**场景发生日**（`scene.time_record`）而不是写入时刻：
        跨天提取时，这段对话属于它发生的那一天，不是被整理的那一天。

        为什么是文档不是表：原文是最大的数据，不该跟结构化记忆挤在一个库里
        （备份也跟着变重）；而且**冷层本该是文件**——可直接看、可直接删。
        见本地设计记录「存储层」（S0 的落点：原文按天存文件，不进库）。
        """
        if not raw.created_at:
            raw.created_at = now_str()
        day = (on_date or raw.created_at)[:10]
        path = self.raws_dir() / f"{day}.md"
        block = (f"\n## {raw.scene_id} · {raw.created_at}\n\n"
                 f"{_safe_raw_body(raw.content).rstrip()}\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
        return day

    def get_raws_by_scene(self, scene_id: str, on_date: str = "") -> list[Raw]:
        """经 scene_id 下钻原文（R5 的唯一入口，S0 没有别的检索方式）。

        给了日期就只翻开那一天的文件（快）；没给就按天倒序翻（兜底，慢）。
        调用方手上有场景对象，顺手把 `time_record` 传进来即可。
        """
        if not scene_id:
            return []
        if on_date:
            files = [self.raws_dir() / f"{on_date[:10]}.md"]
        else:
            files = sorted(self.raws_dir().glob("*.md"), reverse=True)
        for p in files:
            if not p.exists():
                continue
            try:
                found = _raw_doc_section(p.read_text(encoding="utf-8"), scene_id)
            except OSError:
                continue
            if found:
                when, body = found
                # 标题上的时间优先（那是原文真实的记录时刻）；
                # 取不到才退回「那天零点」兜底
                return [Raw(scene_id=scene_id, content=body,
                            created_at=when or f"{p.stem} 00:00:00")]
        return []

    def raw_doc_days(self) -> int:
        """有原文的文档份数（仪表盘显示用——它按天，不是按条）。"""
        try:
            return len(list(self.raws_dir().glob("*.md")))
        except OSError:
            return 0

    def _migrate_raws_to_files(self) -> int:
        """把老库里 `raws` 表的原文导成按天文档（一次性、幂等）。

        跑完把表**改名**成 `raws_migrated` 留底——不删：
        迁移也是代码写的，它也可能有 bug，表还在就能重来。
        """
        # 先看表在不在（新库根本没有这张表——**不该靠异常来判**：
        # 每次启动都抛一次 DatabaseError 虽然被接住了，但那是噪声）
        has = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raws'").fetchone()
        if not has:
            return 0
        rows = self.conn.execute(
            "SELECT scene_id, content, created_at FROM raws"
            " ORDER BY created_at").fetchall()
        for r in rows:
            self.add_raw(Raw(scene_id=r["scene_id"], content=r["content"] or "",
                             created_at=r["created_at"] or ""),
                         on_date=(r["created_at"] or "")[:10])
        if rows:
            with self.conn:
                self.conn.execute("ALTER TABLE raws RENAME TO raws_migrated")
            print(f"[store] 已把 {len(rows)} 条原文导出到 data/raws/ "
                  f"（旧表改名 raws_migrated 留底，不删）")
        return len(rows)

    def _backfill_evidence_at(self) -> int:
        """把老 `evidence` 边里的「引用起点时间」搬进 `profiles.evidence_at`
        （一次性、幂等）。

        2026-09-24：evidence 边退役（存储层稿 §五）——它唯一的真数据就是这个
        时间戳（保护期起算用），from / to 是 `sources` 的重复。搬完不再写边，
        这里也就自然不再有活干（表不在 / 没有 evidence 边 → 直接返回）。
        """
        has = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='edges'").fetchone()
        if not has:
            return 0
        rows = self.conn.execute(
            "SELECT from_id, to_id, created_at FROM edges"
            " WHERE kind = 'evidence'").fetchall()
        moved = 0
        for r in rows:
            p = self.get_profile(r["to_id"])
            if p is None:
                continue
            if r["from_id"] in (p.evidence_at or {}):
                continue        # 已经有了（回填过 / 新数据）——幂等
            p.evidence_at[r["from_id"]] = r["created_at"] or ""
            self.set_profile_evidence_at(p.id, p.evidence_at)
            moved += 1
        if moved:
            print(f"[store] evidence 边回填：{moved} 条「引用起点时间」"
                  f"搬进 profiles.evidence_at（边退役，2026-09-24）")
        return moved

    def _retire_edges_table(self) -> None:
        """退役 `edges` 表（2026-09-24，存储层稿 §五"能派生的不存"）。

        四种边全部停了写、也都换了派生替代（见 `# ---- 段 7` 的清单）。
        **顺序不能反：先回填 `evidence_at`（唯一的真数据），再动表**——
        回填没跑完就退役 = 保护期凭空丢失。

        这里只**改名留底**，不 DROP（同 `_migrate_raws_to_files` 的先例：
        迁移也是代码写的，它也可能有 bug——表还在就能重来）。
        老库的边数据留在 `edges_retired` 里，想回看直接查它。
        """
        has = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='edges'").fetchone()
        if not has:
            return                      # 新库根本没有这张表
        left = self.conn.execute("SELECT count(*) AS n FROM edges").fetchone()["n"]
        with self.conn:
            self.conn.execute("ALTER TABLE edges RENAME TO edges_retired")
        print(f"[store] edges 表退役（{left} 条历史边改名 edges_retired 留底，不删）"
              f"——四类边全改派生，2026-09-24")

    def _retire_openings_table(self) -> None:
        """退役 `openings` 表（2026-10-05 晚，待优化稿 K 条）。

        主动开口整块删了（她不再先开口、留言与送达也不要了），表里那些
        "她说过、还没送达的话"**改名留底、不删**——同 `_retire_edges_table`
        的先例（迁移也是代码写的，它也可能有 bug；表还在就能重来）。
        新库不再建这张表（见 `SCHEMA_SQL` 里的注）。
        """
        has = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='openings'").fetchone()
        if not has:
            return                      # 新库根本没有这张表
        left = self.conn.execute("SELECT count(*) AS n FROM openings").fetchone()["n"]
        with self.conn:
            self.conn.execute("ALTER TABLE openings RENAME TO openings_retired")
        print(f"[store] openings 表退役（{left} 条历史留言改名 openings_retired 留底，不删）"
              f"——主动开口整块删除，2026-10-05")

    # ---- 段 5：S2 摘要 ----

    def add_summary(self, summary: Summary) -> str:
        """写一条 S2；空 id / 空时间戳就地补。返回 id。"""
        if not summary.id:
            summary.id = self._next_id("summaries")
        if not summary.created_at:
            summary.created_at = now_str()
        row = summary.to_row()
        with self.conn:
            self.conn.execute(
                f"INSERT INTO summaries ({', '.join(row.keys())})"
                f" VALUES ({', '.join(['?'] * len(row))})",
                tuple(row.values()),
            )
        return summary.id

    def get_summary(self, sid: str) -> Summary | None:
        """取一条 S2（**含归档的**）。没有则返回 None——同 `get_scene` / `get_profile`。"""
        row = self.conn.execute("SELECT * FROM summaries WHERE id = ?", (sid,)).fetchone()
        return Summary.from_row(row) if row else None

    def summaries_by_topic(self, topic: str, include_archived: bool = False) -> list[Summary]:
        """某个 topic 下的 S2（提炼 3 做抽象时的原料）。

        **必须能拿到 sources**：S2 是 S1 的聚合，抽象要能顺着它下钻到具体场景，
        否则「印证」就成了无源之水（拿什么证明这条画像？）。
        """
        sql = "SELECT * FROM summaries WHERE topic = ?"
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY created_at"
        return [Summary.from_row(r) for r in self.conn.execute(sql, (topic,)).fetchall()]

    def hot_summaries(self, n: int | None = None) -> list[Summary]:
        """热层「活跃 S2」：最近创建的前 n 条；不传 n 才用 `capacity.hot_s2_n` 兜底。

        ⚠️ **`n=0` 就是 0 条**——不能写 `n or 兜底`：那会把 0 吃成 20，
        而 0 是"这一轮不带"的意思（`recall.inject_summaries_n` 用它关闭注入）。
        """
        n = int(cfgmod.cfg("capacity", "hot_s2_n", default=20) if n is None else n)
        rows = self.conn.execute(
            "SELECT * FROM summaries WHERE archived = 0 ORDER BY created_at DESC LIMIT ?",
            (n,)).fetchall()
        return [Summary.from_row(r) for r in rows]

    def archive_s2(self, cap: int) -> int:
        """S2 归档：超上限时最旧的先走（同 S1，只标记不删行）。"""
        total = self.count("summaries", "archived = 0")
        if total <= cap:
            return 0
        with self.conn:
            cur = self.conn.execute(
                "UPDATE summaries SET archived = 1 WHERE id IN ("
                " SELECT id FROM summaries WHERE archived = 0 ORDER BY created_at LIMIT ?)",
                (total - cap,))
        return cur.rowcount

    def set_summary_fields(self, summary_id: str, fields: dict) -> dict:
        """改一条 S2 的**可改字段**（白名单：叙述 / 主题，2026-09-24）——
        `set_scene_fields` 的同构版（三层同一套"改"）。

        主题（topic）是**标签**：改它是重新归类，不动它收的素材（`sources` 固定）；
        但**不能清空**——空主题在聚合 / 检索里是"没有归类"，那是脏数据，不是改标签。
        "改前的值"库里没有，是不可再生信息——留痕由调用方（weave）做。

        返回 `{"ok", "changed", "applied": {字段: (旧值, 新值)}}`。
        """
        bad = [k for k in fields if k not in SUMMARY_EDITABLE]
        if bad:
            return {"ok": False,
                    "detail": f"这些字段不能改：{'、'.join(bad)}"
                              f"（可改：{'、'.join(SUMMARY_FIELD_LABELS.values())}）"}
        s = self.get_summary(summary_id)
        if s is None:
            return {"ok": False, "detail": "找不到这条摘要"}
        applied: dict[str, tuple[str, str]] = {}
        for k, v in fields.items():
            if k == "topic":
                # 主题走自己的口（要同步 `topics` 两列，见 `_write_topics`）——
                # 调用约定是一次只改一个字段（`weave.update_*_confirmed`），
                # 所以不必为跨字段原子性操心。
                out = self.set_summary_topics(summary_id, split_topics(str(v or "")))
                if not out.get("ok"):
                    return out
                if out.get("changed"):
                    applied["topic"] = out["applied"]["topic"]
                continue
            new = ("" if v is None else str(v)).strip()
            old = str(getattr(s, k) or "")
            if old != new:
                applied[k] = (old, new)
        if not applied:
            return {"ok": True, "changed": False, "detail": "内容没变"}
        with self.conn:
            for k, (_old, new) in applied.items():
                # 字段名来自上面的白名单校验（不是外部直接输入）——拼 SQL 安全
                self.conn.execute(f"UPDATE summaries SET {k} = ? WHERE id = ?",
                                  (new, summary_id))
        return {"ok": True, "changed": True, "applied": applied}

    def set_summary_text(self, summary_id: str, text: str) -> bool:
        """改一条 S2 的叙述——`set_summary_fields` 的单字段简写（旧调用点与测试沿用）。"""
        return bool(self.set_summary_fields(summary_id, {"text": text}).get("ok"))

    def archive_summary(self, summary_id: str) -> bool:
        """把一条 S2 放进冷层（**人的动作**，2026-09-24）——她不召回它，数据全在、可取消。"""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE summaries SET archived = 1 WHERE id = ? AND archived = 0",
                (summary_id,))
        return cur.rowcount > 0

    def unarchive_summary(self, summary_id: str) -> bool:
        """取消归档（**人的动作**）——"后悔"的出口。"""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE summaries SET archived = 0 WHERE id = ? AND archived = 1",
                (summary_id,))
        return cur.rowcount > 0

    def delete_summary(self, summary_id: str) -> dict:
        """真删一条 S2（**人的处置**，2026-09-24）——**只删一行**（边表已退役）。

        与 `delete_scene` 的差别：**不摘引用**——引用是派生的（读取端过滤，
        存储层稿 §五）。它收的那批 S1 会变回"未被覆盖"，下次提炼重聚一条新的
        （`_fresh_scenes` 按现存 S2 算覆盖）——"删了重聚"就是它的"改"。
        """
        with self.conn:
            cur = self.conn.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
        return {"deleted": cur.rowcount > 0}

    # ---- 段 6：S3 画像 ----

    def add_profile(self, p: Profile) -> str:
        """写一条 S3；空 id 与三个时间戳就地补（都退到 `created_at`）。

        那三个时间戳是老化判据的全部输入，留空 = 这条画像永远不会老化。
        """
        if not p.id:
            p.id = self._next_id("profiles")
        if not p.created_at:
            p.created_at = now_str()
        if not p.valid_at:
            p.valid_at = p.created_at
        if not p.last_support_at:
            p.last_support_at = p.created_at
        row = p.to_row()
        with self.conn:
            self.conn.execute(
                f"INSERT INTO profiles ({', '.join(row.keys())})"
                f" VALUES ({', '.join(['?'] * len(row))})",
                tuple(row.values()),
            )
        return p.id

    def current_profiles(self, status: str | None = PROFILE_ESTABLISHED) -> list[Profile]:
        """当前有效的画像。

        `invalidated_at` 为空 = 当前有效；**默认只取 established**——
        未验证的（pending）不进常驻。
        传 status=None 可取到全部当前有效画像（含 pending），
        供 R1/R2/R3 检索与 R6 呈现使用。
        """
        sql = "SELECT * FROM profiles WHERE COALESCE(invalidated_at, '') = ''"
        params: tuple = ()
        if status is not None:
            sql += " AND status = ?"
            params = (status,)
        sql += " ORDER BY last_support_at DESC"
        return [Profile.from_row(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_profile(self, pid: str) -> Profile | None:
        """取一条画像（**含已失效的**）。没有则返回 None。"""
        row = self.conn.execute("SELECT * FROM profiles WHERE id = ?", (pid,)).fetchone()
        return Profile.from_row(row) if row else None

    def invalidate_profile(self, pid: str, at: str = "", by: str = INVALIDATED_REVISION,
                           reason: str = "") -> None:
        """**修正**：填 `invalidated_at`（旧记录进历史，不删）。

        这是 invalidated_at 的**修正**这一路（另两路：`by=user` 人改陈述、
        `by=archive` 归档——归档走 `archive_profile()`，别从这里走）。
        老化降级走 `downgrade_profile`，两者的区别在——别混。

        `by` 会收到 `model.INVALIDATED_*` 里的**两个**：
          - `revision`：air 自己拿到新证据后更新认知（默认）；
          - `user`：**人改画像陈述**（界面 / 确认条那条路，
            `weave.update_profile_confirmed`）。
        （`archive` 不从这里走——归档由 `archive_profile()` 自己写。）
        ⚠️ 人的「否决」自 2026-09-24 起**不再走这里**：他否决 = 真删
        （`delete_profile`）——画像是推断，素材全在，判断真成立会被重新立出来，
        不需要留档。（旧库里 by='user' 的行可能是更早的「否决」语义。）
        """
        at = at or now_str()
        with self.conn:
            self.conn.execute(
                "UPDATE profiles SET invalidated_at = ?, invalidated_by = ?,"
                " invalidated_reason = ? WHERE id = ?",
                (at, by, reason or "", pid))

    def downgrade_profile(self, pid: str, at: str = "") -> None:
        """**老化降级**：established → pending，**不填 `invalidated_at`**。

        记录仍在、仍可召回——「不滞留」是「老结论会过期」，不是「被删除」。
        `at` 是降级时刻；一期没有「降级时间」字段所以不落库，
        保留参数是为二期审计留口（别把它误用成 invalidated_at）。
        """
        with self.conn:
            self.conn.execute(
                "UPDATE profiles SET status = ? WHERE id = ? AND status = ?",
                (PROFILE_PENDING, pid, PROFILE_ESTABLISHED))

    def touch_profile(self, pid: str, at: str = "") -> None:
        """更新 `last_support_at`（被印证 / 被提及都算）——老化判据的另一半。"""
        at = at or now_str()
        with self.conn:
            self.conn.execute("UPDATE profiles SET last_support_at = ? WHERE id = ?", (at, pid))

    # ---- S3 复核（记忆整理稿 §五，2026-09-23）----

    def set_profile_reviewed(self, pid: str, at: str = "") -> None:
        """记一次「复核过了」——**只服务于复核间隔防抖**。

        刻意不动 `last_support_at`：复核 ≠ 印证。复核通过就去 touch 的话，
        这条画像会被自己的"体检通过"永久续命，90 天老化永远不触发。
        """
        with self.conn:
            self.conn.execute("UPDATE profiles SET last_review_at = ? WHERE id = ?",
                              (at or now_str(), pid))

    def add_profile_review(self, pid: str, verdict: str, reason: str = "") -> str:
        """写一条复核记录（每次复核一行，不覆盖历史）。返回 id。"""
        r = ProfileReview(profile_id=pid, verdict=verdict, reason=reason or "",
                          created_at=now_str())
        r.id = self._next_id("profile_reviews")
        row = r.to_row()
        with self.conn:
            self.conn.execute(
                f"INSERT INTO profile_reviews ({', '.join(row.keys())})"
                f" VALUES ({', '.join(['?'] * len(row))})",
                tuple(row.values()),
            )
        return r.id

    def unhandled_reviews(self) -> list[ProfileReview]:
        """还没被人处理的提议（`wrong` 且 `handled = 0`），新的在前。"""
        rows = self.conn.execute(
            "SELECT * FROM profile_reviews WHERE verdict = 'wrong' AND handled = 0"
            " ORDER BY created_at DESC, id DESC").fetchall()
        return [ProfileReview.from_row(r) for r in rows]

    def mark_review_handled(self, rid: str) -> int:
        """标记一条提议已处理（作废 / 留着）——返回改动条数（重复标记为 0）。"""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE profile_reviews SET handled = 1 WHERE id = ? AND handled = 0", (rid,))
            return int(cur.rowcount)

    def profile_history(self, topic: str) -> list[Profile]:
        """同一 topic 的版本序列（按 valid_at 排序）——双时间戳的「轨迹」用法。"""
        rows = self.conn.execute(
            "SELECT * FROM profiles WHERE topic = ? ORDER BY valid_at", (topic,)).fetchall()
        return [Profile.from_row(r) for r in rows]

    def current_profile_by_topic(self, topic: str) -> Profile | None:
        """该 topic 当前有效的那条画像（无效的进历史，不算「当前」）。

        一个 topic 同时只该有一条有效画像——这是「版本序列」的前提：
        同 topic 的记录按 valid_at 排就是这条画像的变化轨迹。
        """
        row = self.conn.execute(
            "SELECT * FROM profiles WHERE topic = ? AND COALESCE(invalidated_at, '') = ''"
            " ORDER BY valid_at DESC LIMIT 1", (topic,)).fetchone()
        return Profile.from_row(row) if row else None

    def set_profile_status(self, pid: str, status: str, evidence: int | None = None) -> None:
        """改画像状态（pending ↔ established）。**不动 `invalidated_at`**——
        状态与失效是两回事，混在一起就会把「还没验证」写成「已经作废」。

        `status` 过 `PROFILE_STATUS` 白名单：**拼错当场报错**（状态是统计与
        准入的索引，静默写进一个谁都不认的值，等于这条画像两头都不算）。
        这也是该域元组的唯一读点（2026-09-25 接线，见 L0 核对）。
        """
        if status not in PROFILE_STATUS:
            raise ValueError(
                f"不认识的画像状态「{status}」（只能是：{'、'.join(PROFILE_STATUS)}）")
        with self.conn:
            if evidence is None:
                self.conn.execute("UPDATE profiles SET status = ? WHERE id = ?", (status, pid))
            else:
                self.conn.execute(
                    "UPDATE profiles SET status = ?, evidence = ? WHERE id = ?",
                    (status, evidence, pid))

    def set_profile_sources(self, pid: str, sources: list[str],
                            evidence: int | None = None,
                            pack: list[dict] | None = None) -> None:
        """更新画像的可追溯来源（追加印证证据时用）。

        `pack` 只在要顺带刷新快照时传（一般不用——包只装 forming 那几条，
        它们在画像形成时就定了，后来的印证不进包）。
        """
        payload = json.dumps(list(sources or []), ensure_ascii=False)
        sets, params = ["sources = ?"], [payload]
        if evidence is not None:
            sets.append("evidence = ?")
            params.append(evidence)
        if pack is not None:
            sets.append("evidence_pack = ?")
            params.append(json.dumps(list(pack), ensure_ascii=False))
        params.append(pid)
        with self.conn:
            self.conn.execute(f"UPDATE profiles SET {', '.join(sets)} WHERE id = ?",
                              tuple(params))

    def profiles_citing(self, scene_id: str, only_current: bool = True) -> list[Profile]:
        """反查：哪些画像的 `sources` 引用了这条场景。

        用途有两个——场景被提及时连带 touch 画像的 last_support_at；
        以及归档时判断「这条场景是不是某条画像的依据」。
        `sources` 是 JSON 数组文本，用 LIKE 匹配带引号的 id：
        够用且不引入 JSON 扩展依赖（id 形如 S1-0001，不会撞上子串误匹配）。
        """
        sql = "SELECT * FROM profiles WHERE sources LIKE ?"
        params: list = [f'%"{scene_id}"%']
        if only_current:
            sql += " AND COALESCE(invalidated_at, '') = ''"
        return [Profile.from_row(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def list_topics(self, subject: str | None = None) -> list[str]:
        """已有 topic 清单（候选来源：场景写入与 `resolve_topic` 都要用）。

        同时收 `scenes` 与 `profiles` 的 topic：**topic 是贯通三层的**，
        只从画像里取会漏掉「攒够场景但还没抽象成画像」的那些主题。
        包含历史（已失效）记录的 topic——老 topic 认不出来，
        版本序列就会断成两半，双时间戳再优雅也没用。
        """
        sql = ("SELECT DISTINCT topic FROM ("
               " SELECT topic, subject FROM scenes WHERE COALESCE(topic,'') <> ''"
               " UNION"
               " SELECT topic, subject FROM profiles WHERE COALESCE(topic,'') <> ''"
               ") WHERE 1=1")
        params: tuple = ()
        if subject is not None:
            sql += " AND subject = ?"
            params = (subject,)
        return [r["topic"] for r in self.conn.execute(sql, params).fetchall()]

    def all_topics(self) -> list[str]:
        """三层**全部**主题（含附加主题、含 S2 独有的）——排序后返回。

        与 `list_topics` 的分工：那条是**链路册子**（scenes + profiles，带 subject
        过滤——`resolve_topic` 的"新建 vs 复用"要用它，只该看主主题且要认主语）；
        这条是**候选池 / 找东西**：附加主题的候选（`distill._extra_candidates`）
        要从"库里真出现过的所有主题"里挑——漏掉 summaries 的话，
        S2 独有的主题（含它自己的附加主题）永远进不了候选，池子只会越用越窄。
        """
        seen: set[str] = {t for t in self.list_topics() if t}
        for table in ("summaries", "profiles"):
            for r in self.conn.execute(f"SELECT topic, topics FROM {table}"):
                for t in (load_str_list(r["topics"])
                          or ([r["topic"]] if r["topic"] else [])):
                    if t:
                        seen.add(t)
        for r in self.conn.execute(
                "SELECT topic FROM scenes WHERE COALESCE(topic, '') <> ''"):
            seen.add(r["topic"])
        return sorted(seen)

    # ---- 按主题找（**主主题 + 附加主题都算**，2026-09-24 晚）----
    #
    # 与 `summaries_by_topic` / `query_scenes(topic=)` 的分工：那两条是**链路**
    # （只认主主题、等值匹配——"哪些场景还没被收进去"这类判断靠它们）；
    # 这里三条是**找东西**（认全部标签、子串匹配——她记不全主题名时
    # 「压力」也要能找到「用户·压力」）。

    def summaries_with_topic(self, topic: str,
                             include_archived: bool = False) -> list[Summary]:
        """按主题找 S2（子串匹配，认主主题 + 附加主题）。"""
        q = (topic or "").strip()
        if not q:
            return []
        sql = "SELECT * FROM summaries WHERE topics LIKE ?"     # 先粗筛（SQL）
        params: list = [f"%{q}%"]
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY created_at DESC"
        out: list[Summary] = []
        for r in self.conn.execute(sql, tuple(params)).fetchall():
            s2 = Summary.from_row(r)                            # 再精筛（解析 JSON）
            if any(q.lower() in t.lower() for t in (s2.topics or [s2.topic])):
                out.append(s2)
        return out

    def profiles_with_topic(self, topic: str,
                            only_current: bool = True) -> list[Profile]:
        """按主题找画像（子串匹配，认主主题 + 附加主题；默认只找当前有效的）。"""
        q = (topic or "").strip()
        if not q:
            return []
        sql = "SELECT * FROM profiles WHERE topics LIKE ?"
        params: list = [f"%{q}%"]
        if only_current:
            sql += " AND COALESCE(invalidated_at, '') = ''"
        sql += " ORDER BY last_support_at DESC"
        out: list[Profile] = []
        for r in self.conn.execute(sql, tuple(params)).fetchall():
            p = Profile.from_row(r)
            if any(q.lower() in t.lower() for t in (p.topics or [p.topic])):
                out.append(p)
        return out

    def scenes_with_topic(self, topic: str, include_archived: bool = False,
                          limit: int = 200) -> list[Scene]:
        """按主题找场景（S1 是**单主题**——只有主主题，没有多标签）——子串匹配。"""
        q = (topic or "").strip()
        if not q:
            return []
        sql = "SELECT * FROM scenes WHERE topic LIKE ?"
        params: list = [f"%{q}%"]
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY COALESCE(NULLIF(time_event, ''), created_at) DESC LIMIT ?"
        params.append(int(limit))
        return [self._scene_from_row(r)
                for r in self.conn.execute(sql, tuple(params)).fetchall()]

    # ---- 基础档案（用户明说的事实，与画像严格分开）----

    def all_user_facts(self) -> list[dict]:
        """全部档案（按 key 排序），带 `source` 与出处。

        常驻注入，所以不做过滤——档案本来就少（几条），
        而且它**不该有"待验证"这种状态**：用户说的事实不需要印证。
        返回 source 是因为注入时要说清哪条是「他说的」、哪条是「我推的」——
        两者的分量不一样。
        """
        rows = self.conn.execute(
            "SELECT key, value, source, note, updated_at FROM user_facts ORDER BY key"
        ).fetchall()
        return [{"key": r["key"], "value": r["value"],
                 "source": r["source"] or "user_stated",
                 "note": r["note"] or "", "updated_at": r["updated_at"] or ""}
                for r in rows if (r["value"] or "").strip()]

    def set_user_fact(self, key: str, value: str, source: str = "user_stated",
                      note: str = "") -> None:
        """写一条档案。**空值 = 删除这条**——档案里没有"空"这个状态。

        空值当删除（而不是存空串）：空串会在注入时渲染出一行
        「- 年龄：（空）」，看着像"她记得，但记的是空"。
        """
        k = (key or "").strip()
        v = (value or "").strip()
        if not k:
            return
        with self.conn:
            if not v:
                self.conn.execute("DELETE FROM user_facts WHERE key = ?", (k,))
                return
            self.conn.execute(
                "INSERT INTO user_facts (key, value, source, note, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " source=excluded.source, note=excluded.note,"
                " updated_at=excluded.updated_at",
                (k, v, source or "user_stated", note, now_str()))

    def save_user_facts(self, facts: dict, note: str = "") -> int:
        """批量写（界面保存用）。返回写入条数。

        **空值即删除**，所以界面上把一格清空 = 让她忘掉这条
        （而不是留着一条空的）。
        """
        n = 0
        for k, v in (facts or {}).items():
            self.set_user_fact(k, v, note=note)
            n += 1
        return n

    # ---- 用户偏好（他想要什么，不是他是什么）----

    def get_pref(self, key: str, default: str = "") -> str:
        """读一条偏好。没存过给 `default`（**默认给最少的加工**，保守侧）。"""
        row = self.conn.execute(
            "SELECT value FROM user_prefs WHERE key = ?", (key or "",)).fetchone()
        return row["value"] if (row and row["value"] is not None) else default

    def set_pref(self, key: str, value: str) -> None:
        """写一条偏好。**只有人能改它**——系统不自动重置（见建表注释）。"""
        k, v = (key or "").strip(), (value or "").strip()
        if not k:
            return
        with self.conn:
            self.conn.execute(
                "INSERT INTO user_prefs (key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " updated_at=excluded.updated_at",
                (k, v, now_str()))

    def all_prefs(self) -> dict:
        """全部用户偏好（`key → value`，**都是字符串**）。

        类型转换统一在各自的 `*_from_prefs` 里做（那里会对非法值降级）。
        """
        return {r["key"]: (r["value"] or "")
                for r in self.conn.execute("SELECT key, value FROM user_prefs").fetchall()}

    # ---- 系统状态（`meta`：系统的账，不是他的偏好）----

    def get_meta(self, key: str, default: str = "") -> str:
        """读一条系统状态（体检时刻这类）。没存过给 `default`。"""
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key or "",)).fetchone()
        return row["value"] if (row and row["value"] is not None) else default

    def set_meta(self, key: str, value: str) -> None:
        """写一条系统状态。与 `set_pref` 分开：那边装「他想要什么」（只有人能改），
        这里装系统的账——体检时刻必须能**自动**更新，否则"距上次多久"没有判据。"""
        k, v = (key or "").strip(), value or ""
        if not k:
            return
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta (key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " updated_at=excluded.updated_at",
                (k, v, now_str()))

    def top_entities(self, limit: int = 8) -> list[dict]:
        """出现次数最多的实体（「用户反复提到的那些人和地方」）。

        ⚠️ **当前无调用点**：原先供编织层的 `world_view` 用——2026-09-20 两套姿态
        退役（`world_view` 删除）后悬空。**先留着不删**：实体页（`dashboard.entities`）
        现在用的是 `all_entities` + 逐条计数，"最常出现"这个现成接口将来要接就用它
        （同 `theta` / `r0_floor` 那类"注释写明的刻意接口"——没有读点，
        但处境注明在案，比悄悄挂着强）。
        """
        rows = self.conn.execute(
            "SELECT e.name, e.kind, count(*) AS n FROM scene_entities se"
            " JOIN entities e ON e.id = se.entity_id"
            " GROUP BY e.id ORDER BY n DESC, e.name LIMIT ?", (limit,)).fetchall()
        return [{"name": r["name"], "kind": r["kind"], "count": r["n"]} for r in rows]

    def merge_topics(self, from_topic: str, to_topic: str) -> int:
        """topic 合并（「拿不准就新建」的补救口）：把裂开的两个序列接回去。

        返回受影响条数。不删任何记录——只改归属。
        `topics` 多标签列**一并改名**（2026-09-24 晚）：只改 `topic` 的话，
        标签里会留着旧名——"按主题找"还能找到它，但主主题已经不叫那个了，
        正是那种"两份数据必然漂移"的坑（`topic` 与 `topics[0]` 必须始终一致）。
        """
        with self.conn:
            n = self.conn.execute(
                "UPDATE profiles SET topic = ? WHERE topic = ?", (to_topic, from_topic)).rowcount
            n += self.conn.execute(
                "UPDATE scenes SET topic = ? WHERE topic = ?", (to_topic, from_topic)).rowcount
            n += self.conn.execute(
                "UPDATE summaries SET topic = ? WHERE topic = ?", (to_topic, from_topic)).rowcount
            for table in ("profiles", "summaries"):
                rows = self.conn.execute(
                    f"SELECT id, topics FROM {table} WHERE topics LIKE ?",
                    (f"%{from_topic}%",)).fetchall()
                for r in rows:
                    tags = load_str_list(r["topics"])
                    if from_topic not in tags:
                        continue
                    new: list[str] = []
                    for t in tags:
                        x = to_topic if t == from_topic else t
                        if x not in new:
                            new.append(x)
                    if new and new[0] != to_topic:      # 不变量：topics[0] == topic
                        new = [to_topic] + [x for x in new if x != to_topic]
                    self.conn.execute(
                        f"UPDATE {table} SET topics = ? WHERE id = ?",
                        (json.dumps(new, ensure_ascii=False), r["id"]))
        return n

    # ---- 段 7：实体（`edges` 表已退役，2026-09-24）----
    #
    # （`add_edge` / `all_edges` / `has_edge` / `neighbors` 四个方法原来在这里——
    #   随 `edges` 表一起退役，历史见 git。四种边全部改了派生：
    #     compose    → `Summary.sources` / `Profile.sources`（本就是它的冗余）
    #     causality  → `topic_neighbors()` 现算（同 topic + 时间相邻）
    #     raw        → `get_raws_by_scene()` 按日期翻文档（读取端一直没用过边）
    #     evidence   → `Profile.evidence_at`（真数据：引用起点时间，保护期起算）
    #   见存储层稿 §五"能派生的不存"。）

    def find_entity(self, name: str) -> Entity | None:
        """精确匹配 name / aliases（大小写不敏感）。

        第一版不做自动语义归并（「两个小明是不是同一个人」需要判断，
        宁可存疑新建 + 人工合并，也不猜）。
        """
        if not name or not name.strip():
            return None
        target = name.strip().lower()
        for r in self.conn.execute("SELECT * FROM entities").fetchall():
            e = Entity.from_row(r)
            if e.name.strip().lower() == target:
                return e
            if any((a or "").strip().lower() == target for a in e.aliases):
                return e
        return None

    def add_entity(self, name: str, kind: str = "person",
                   aliases: list[str] | None = None) -> str:
        """建一个新实体并返回 id。**不做去重**——找旧的是 `find_entity` 的事。

        只有「确认它是个新实体」之后才该走到这里（两个同名条目算不算同一个
        是一次语义断言，系统不替人下，见 `entity.link_entities`）。
        """
        eid = self._next_id("entities")
        with self.conn:
            self.conn.execute(
                "INSERT INTO entities (id, name, kind, aliases, created_at) VALUES (?,?,?,?,?)",
                (eid, name, kind, json.dumps(aliases or [], ensure_ascii=False), now_str()))
        return eid

    def all_entities(self) -> list[Entity]:
        """全部实体（**全量**）——实体旁路每轮都拿它做匹配，几百条的量级，不需要索引。"""
        return [Entity.from_row(r) for r in self.conn.execute("SELECT * FROM entities").fetchall()]

    def entities_of_scene(self, scene_id: str) -> list[str]:
        """一条场景涉及哪些实体（名字列表）。

        用途：编织层的模式统计要按 `trigger_class × entity` 组合——
        「面对【上级】的【被评价】→ 典型反应」这种精度是靠两个字段组合出来的，
        而实体这一半得从关联表查。
        """
        rows = self.conn.execute(
            "SELECT e.name FROM scene_entities se JOIN entities e ON e.id = se.entity_id"
            " WHERE se.scene_id = ?", (scene_id,)).fetchall()
        return [r["name"] for r in rows]

    def link_scene_entity(self, scene_id: str, entity_id: str,
                          relation: str = "") -> None:
        """挂一条「场景 ↔ 实体」关系（同一对重复挂时**更新关系**、不重复插）。

        `relation` 是这条场景里他和该实体的关系（「妈妈」「同事」）。
        重复挂只在**带新关系**时覆盖（`excluded.relation <> ''`）——
        第二次不带关系的调用不该把已有关系抹成空。
        """
        with self.conn:
            self.conn.execute(
                "INSERT INTO scene_entities (scene_id, entity_id, relation)"
                " VALUES (?,?,?)"
                " ON CONFLICT(scene_id, entity_id) DO UPDATE SET"
                " relation = excluded.relation WHERE excluded.relation <> ''",
                (scene_id, entity_id, (relation or "").strip()))

    def get_entity(self, eid: str) -> Entity | None:
        """按编号取一个实体；没有则返回 None。"""
        row = self.conn.execute("SELECT * FROM entities WHERE id = ?", (eid,)).fetchone()
        return Entity.from_row(row) if row else None

    def merge_entities(self, from_id: str, to_id: str) -> dict:
        """把两个实体合成一个（「两个小明其实是同一个人」的人工补救口）。

        返回 `{"scenes", "aliases"}`：改挂了多少条场景、并过去几个别名。
        **只改归属、不删场景**（同 topic 合并那条纪律）。

        做的三件事，缺一不可：
          1. `from` 的场景关联改挂到 `to`；
          2. **`from` 的名字与别名并进 `to` 的 aliases**——不并的话，
             以后提到他原来的叫法就命中不了，等于白合；
          3. 删掉 `from` 那一行（索引里不该留着同一个人的旧条目）。

        ⚠️ `scene_entities` 的主键是 `(scene_id, entity_id)`：两个实体都挂在
        同一条场景上时，直接改 `entity_id` 会**主键冲突**。所以走
        「先 `INSERT OR IGNORE` 复制、再删旧行」——重复的那条被忽略，
        正是想要的（同一条场景只需挂一次）。
        """
        if not from_id or not to_id or from_id == to_id:
            return {"scenes": 0, "aliases": 0}
        src, dst = self.get_entity(from_id), self.get_entity(to_id)
        if src is None or dst is None:
            return {"scenes": 0, "aliases": 0}

        old = [str(a).strip() for a in (dst.aliases or []) if str(a).strip()]
        merged = list(old)
        seen = {str(dst.name or "").strip().lower()} | {a.lower() for a in old}
        for cand in [src.name] + list(src.aliases or []):
            c = str(cand or "").strip()
            if c and c.lower() not in seen:
                merged.append(c)
                seen.add(c.lower())

        with self.conn:
            moved = self.conn.execute(
                "INSERT OR IGNORE INTO scene_entities (scene_id, entity_id, relation)"
                " SELECT scene_id, ?, relation FROM scene_entities WHERE entity_id = ?",
                (to_id, from_id)).rowcount
            self.conn.execute("DELETE FROM scene_entities WHERE entity_id = ?", (from_id,))
            self.conn.execute("UPDATE entities SET aliases = ? WHERE id = ?",
                              (json.dumps(merged, ensure_ascii=False), to_id))
            self.conn.execute("DELETE FROM entities WHERE id = ?", (from_id,))
        return {"scenes": moved, "aliases": len(merged) - len(old)}

    # ---- 段 8：备忘录 ----

    def add_memo(self, memo: Memo) -> str:
        """写一条备忘录；空 id / 空时间戳就地补，返回 id。

        此时 `kind_class` / `window_days` 可能还空着——要等模型分类后回填
        （`distill._write_memos`）。
        """
        if not memo.id:
            memo.id = self._next_id("memos")
        if not memo.created_at:
            memo.created_at = now_str()
        row = memo.to_row()
        with self.conn:
            self.conn.execute(
                f"INSERT INTO memos ({', '.join(row.keys())})"
                f" VALUES ({', '.join(['?'] * len(row))})",
                tuple(row.values()),
            )
        return memo.id

    def get_memo(self, mid: str) -> Memo | None:
        """取一条备忘录（**任意状态**）；没有则返回 None。"""
        row = self.conn.execute("SELECT * FROM memos WHERE id = ?", (mid,)).fetchone()
        return Memo.from_row(row) if row else None

    def memos_by_scene(self, scene_id: str) -> list[Memo]:
        """这条场景带出来的备忘录（任意状态）。"""
        rows = self.conn.execute("SELECT * FROM memos WHERE scene_id = ?", (scene_id,)).fetchall()
        return [Memo.from_row(r) for r in rows]

    def open_memos(self) -> list[Memo]:
        """所有没关闭的备忘录（pending = 待提，raised = 已提过一次）。

        已提过的也留着：用户可能过几天回来说结果，那时要能把它闭合掉
        （「提过一次就不再主动提」不等于「可以忘掉」）。
        """
        rows = self.conn.execute(
            "SELECT * FROM memos WHERE status <> ? ORDER BY created_at",
            (MEMO_CLOSED,)).fetchall()
        return [Memo.from_row(r) for r in rows]

    def due_memos(self, now: str | None = None) -> list[Memo]:
        """**到期待提**的备忘录。

        两种到期判据（对应「时间不靠 LLM 估」那条约定）：
          - **有具体时间**（`due_at` 用户明说的）→ 时间过了就到期
          - **无具体时间** → `created_at + window_days`（周期由分类映射，非 LLM 估）

        `memo.sensitive` 是**提档**（0 正常提 / 2 只记不提）：
        只有 2（硬隐私）不进候提——其余照进候选。
        （2026-09-15 前是二值：敏感就闭嘴——那会把感冒、体检这类全锁死。）
        """
        now = now or now_str()
        out = []
        for m in self.open_memos():
            if m.status != MEMO_PENDING or m.sensitive >= 2:
                continue
            if m.due_at:
                if m.due_at <= now:
                    out.append(m)
            elif _add_days(m.created_at, m.window_days or 30) <= now:
                out.append(m)
        return out

    def close_open_loop(self, scene_id: str, content: str,
                        at: str | None = None, loop_id: str = "") -> bool:
        """把某张卡里对应的那条 `open_loops` 标成**已闭合**——标，不删。

        这是「闭合回流」的一半（设计稿 D 条 2）：`memos` 是状态的唯一事实源，
        它关的时候顺手把场景卡里的钩子标掉，渲染/清单就不会再端出已经了结的事。
        标而不删（Zep 的"失效不删"）：一边是「这条关了吗」，另一边是
        「它曾经开着、什么时候关的」——历史留着，才回答得了
        「为什么它后来不提了」。

        **匹配（2026-10-05 改）**：优先按 `loop_id`（钩子的稳定编号，形如 `S1-0042#2`），
        没有 / 对不上时退回 `content` 全等（老数据没有编号，那正是以前唯一的匹配方式）。
        改用编号是「变更内容」的前提：memo 的 content 改了，文字匹配就断、编号不会断——
        "改一个字，闭合回流静默断掉"那个坑从此填上。
        找不到对应项返回 False——**不报错**，memo 那边的关闭照常算数。
        """
        return self._mark_open_loop(scene_id, at=at, loop_id=loop_id,
                                    content=content, field="closed_at")

    def retire_open_loop(self, scene_id: str, content: str, loop_id: str = "",
                         at: str | None = None) -> bool:
        """给钩子标 `retired_at`——**退役 ≠ 完成，所以另起一个字段**（2026-10-05）。

        为什么不复用 `closed_at`：「退役」是"系统不再提醒了"，不是"事情了结了"。
        两个字段分开，才答得了「它是办完了，还是没人管了」；渲染侧两者都跳过
        （`prompts._pending_of`）——「→ 未定」行不再挂着一件早已过去的、或不再提的事。
        """
        return self._mark_open_loop(scene_id, at=at, loop_id=loop_id,
                                    content=content, field="retired_at")

    def _mark_open_loop(self, scene_id: str, *, at: str | None, loop_id: str,
                        content: str, field: str) -> bool:
        """给一条钩子打时间戳（`closed_at` / `retired_at` 共用这一份）。

        - 幂等：已经打过这个戳的不再动；
        - **退役跳过已闭合的**（办完了 ≠ 没人管了）；
        - 匹配**编号优先、文字兜底**（见 `close_open_loop` 的说明）。
        """
        s = self.get_scene(scene_id)
        if s is None:
            return False
        lid = (loop_id or "").strip()
        target = (content or "").strip()
        if not lid and not target:
            return False
        changed = False
        for loop in s.open_loops or []:
            if not isinstance(loop, dict) or loop.get(field):
                continue
            if field == "retired_at" and loop.get("closed_at"):
                continue
            hit = ((lid and str(loop.get("loop_id") or "").strip() == lid)
                   or (target and str(loop.get("content") or "").strip() == target))
            if hit:
                loop[field] = at or now_str()
                changed = True
        if not changed:
            return False
        with self.conn:
            self.conn.execute("UPDATE scenes SET open_loops = ? WHERE id = ?",
                              (json.dumps(s.open_loops, ensure_ascii=False), scene_id))
        return True

    def mark_memo_raised(self, mid: str, at: str | None = None) -> bool:
        """待提 → 已提。**提过一次就不再主动提**（用户没接话也不追）。

        ⚠️ **只对 `pending` 生效**（2026-09-22）：同一轮里"它进了候选"和
        "他这句话把它了结"可以同时发生——唤醒先算候选（那时还是 pending）、
        写入时才被命中判定关掉。无条件写 `raised` 会把刚关掉的那条
        **打回未了结**（钩子已经标了 `closed_at`，它却还挂在清单上）。
        返回"真的转了吗"——回执只报真进账的那些。
        """
        with self.conn:
            cur = self.conn.execute(
                "UPDATE memos SET status = ?, raised_at = ? WHERE id = ? AND status = ?",
                (MEMO_RAISED, at or now_str(), mid, MEMO_PENDING))
            return bool(cur.rowcount)

    def close_memo(self, mid: str) -> None:
        """已提 / 待提 → 关闭（用户给了结果，或超期放弃）。

        规格里「结果本身不用单独存」——它跟着上下文提取走，会变成一条新场景卡。
        memo 只需要管住「不重复提」这件事。
        """
        with self.conn:
            self.conn.execute("UPDATE memos SET status = ? WHERE id = ?",
                              (MEMO_CLOSED, mid))

    def set_memo_class(self, mid: str, kind_class: str, window_days: int) -> None:
        """补写无具体时间备忘条的分类与窗口（写入时算好，供到期判定用）。"""
        with self.conn:
            self.conn.execute(
                "UPDATE memos SET kind_class = ?, window_days = ? WHERE id = ?",
                (kind_class, int(window_days), mid))

    def set_memo_timing(self, mid: str, timing: str) -> None:
        """补写时机类别（soon / later）——**只认 `soon`**：身体状况不等常规窗口
        （`memo.due()` 用 `created + memo.soon_hours`）；`later` 就是"照常"。"""
        with self.conn:
            self.conn.execute("UPDATE memos SET timing = ? WHERE id = ?", (timing, mid))

    def set_memo_group(self, mid: str, group_name: str) -> None:
        """补写事项组名（同一件事的多步共用一个短名，2026-09-22）。

        只在**空组名**上补（`memo.classify_group_names` 挑的）——改已有的组名
        会让同一件事裂成两个名字，所以这条 SQL 的调用方要先判空。
        """
        with self.conn:
            self.conn.execute("UPDATE memos SET group_name = ? WHERE id = ?",
                              (group_name, mid))

    def update_memo_content(self, mid: str, content: str,
                            due_at: str | None = None) -> bool:
        """内容变更（`memo.judge_hits` 的 `update` 动作，2026-10-05）——**回待提**。

        内容变了 = 有新的事实要提醒，所以回到 `pending` 并清掉 `raised_at`
        （重新有机会进注入）。时间**只在用户明说时**才写（`due_at=None` 表示
        没明说，保持原样；铁律不破：模型只分类、不估天数）。
        """
        if not (content or "").strip():
            return False
        sets = ["content = ?", "status = ?", "raised_at = ''"]
        vals: list = [content.strip(), MEMO_PENDING]
        if due_at:
            sets.append("due_at = ?")
            vals.append(str(due_at).strip())
        vals.append(mid)
        with self.conn:
            cur = self.conn.execute(
                f"UPDATE memos SET {', '.join(sets)} WHERE id = ?", tuple(vals))
            return bool(cur.rowcount)

    def set_memo_scene(self, mid: str, scene_id: str) -> None:
        """把一条 memo 的 `scene_id` 指到另一张卡（2026-10-05）。

        唯一的调用方是打捞重建（`salvage._adopt_loops`）：原卡被删时 memo 留了下来、
        `scene_id` 悬着，重建后**认领**它——引用接回新卡，它的关闭 / 退役
        就能照常回流到新卡的钩子上。
        """
        with self.conn:
            self.conn.execute("UPDATE memos SET scene_id = ? WHERE id = ?",
                              (scene_id, mid))

    # （`mark_memo_tried` / `add_opening` / `unread_openings` /
    #   `mark_openings_delivered` 2026-10-05 晚删除——主动开口与留言整块不要了，
    #   见待优化稿 K 条；`openings` 表退役留底见 `_retire_openings_table`。）

    # ---- 段 9：归档 ----

    def set_profile_evidence_at(self, pid: str, mapping: dict) -> bool:
        """写「引用起点时间」（id → 时刻，2026-09-24）——同 `set_profile_sources` 的写法。

        它原来是 `edges` 表 evidence 边的 `created_at`；搬进画像本身后边表可退役
        （存储层稿 §五："唯一真数据搬进记录"）。
        """
        with self.conn:
            cur = self.conn.execute(
                "UPDATE profiles SET evidence_at = ? WHERE id = ?",
                (json.dumps(mapping or {}, ensure_ascii=False), pid))
        return cur.rowcount > 0

    def protected_scene_ids(self, now: str | None = None) -> set[str]:
        """当前**不能归档**的场景 id 集合。

        保护 = 「被**当前有效**画像引用」**且**「在保护期内」。

        两个角色、两个保护期（2026-09-11 定）：
          - `forming`（画像的出处）→ `forming_grace_days`（长，默认 365 天）
          - `supporting`（后来的印证）→ `citation_grace_days`（短，默认 90 天）
        role **不单独存**：由 `evidence_pack` 的 id 集合派生（出处那几条进包，
        与 `weave.render_mirror` 认「出处 / 印证」同一套判法）。

        **计时起点是"被引用那一刻"**（`profile.evidence_at`；2026-09-24 前是
        evidence 边的 `created_at`），不是场景的创建时刻：
        保护的意义是「用户此刻质疑，air 得能当场拿出依据」——
        这个需求从被引用时开始衰减，而不是从场景发生那天算。

        为什么必须限时：**印证的场景会一直新增**。如果「被引用 = 永久免死」，
        保护集只涨不消，`s1_cap` 形同虚设。限时保护是滑动的：
        今天新增的印证，N 天后自动放开——不需要为「新增」再写一条规则。

        过期之后不是「看不见」，是「慢一点看」：归档只标记不删行，仍可打捞。
        追溯的底线保障是画像自带的 `evidence_pack`（形成依据的快照）。
        """
        now = now or now_str()
        forming_days = int(cfgmod.cfg("capacity", "forming_grace_days", default=365) or 365)
        citing_days = int(cfgmod.cfg("capacity", "citation_grace_days", default=90) or 90)

        # 只保护「当前有效」的画像引用的场景：画像被修正/否决之后，
        # 它旧版本的依据不必继续占着保护位（新版本有自己的）。
        out: set[str] = set()
        for p in self.current_profiles(status=None):
            pack_ids = {str(it.get("id")) for it in (p.evidence_pack or [])
                        if isinstance(it, dict)}
            for sid, at in (p.evidence_at or {}).items():
                days = forming_days if sid in pack_ids else citing_days
                # 缺时间戳的条目不保护——保守一侧（保护集宁可小，归档可打捞）
                if at and _add_days(at, days) > now:
                    out.add(sid)
        return out

    def archive_s1(self, cap: int, score_fn=None, now: str | None = None) -> int:
        """S1 归档：超上限时把**核心度最低的**先送进冷层。

        两条硬保护：
          ① **被当前有效画像引用、且在保护期内的场景不归档**
             （见 `protected_scene_ids`）——用户当场否决画像时，
             air 得能立刻把依据拿出来；依据进了冷层，追溯就变成一句空话。
          ② 只标记、不删行。

        `score_fn(scene) -> float` 由调用方给（算核心度要用 `recall.core_score`，
        而 store 不该反向依赖 recall）；不传则退回「最旧的先走」，那只是兜底不是设计。
        """
        candidates = [self._scene_from_row(r) for r in self.conn.execute(
            "SELECT * FROM scenes WHERE archived = 0").fetchall()]
        if len(candidates) <= cap:
            return 0
        protected = self.protected_scene_ids(now)
        # 保护对象也占容量：能动的只有剩下的那些
        movable = [s for s in candidates if s.id not in protected]
        excess = len(candidates) - cap
        if excess <= 0 or not movable:
            return 0
        if score_fn:
            movable.sort(key=score_fn)            # 核心度从低到高
        else:
            movable.sort(key=lambda s: s.created_at or "")
        victims = movable[:excess]
        with self.conn:
            for s in victims:
                self.conn.execute("UPDATE scenes SET archived = 1 WHERE id = ?", (s.id,))
        self._bump_emb()
        return len(victims)

    # ---- 段 10：改 / 归档 / 删（**只有人能发起**，见本地设计记录「工具箱」§3.4 / §五）----

    def set_scene_fields(self, scene_id: str, fields: dict) -> dict:
        """改场景的**可改字段**（白名单，2026-09-24）——**人的纠正**的一条口。

        **不碰原文**：原文永远不动，动的只是场景卡上的理解——
        否则改完就分不清「当时发生了什么」和「后来怎么理解的」。
        改前的值由调用方（weave）留痕。

        校验**拒绝**而不是静默丢弃（静默丢弃会让人以为改成了）：
          - 字段不在 `SCENE_EDITABLE` → 整体拒绝，说清可改的有哪些；
          - `trigger_class` 必须 ∈ `TRIGGER_CLASSES`（或空串 = 清掉）——
            它是 `pattern_stats` 的统计索引，自由文本会静默长出新桶；
          - `time_event` 只收 `YYYY-MM-DD` 开头（相对说法算不出日期，同 memo 那条纪律）。

        返回 `{"ok", "changed", "applied": {字段: (旧值, 新值)}}`。
        """
        bad = [k for k in fields if k not in SCENE_EDITABLE]
        if bad:
            return {"ok": False,
                    "detail": f"这些字段不能改：{'、'.join(bad)}"
                              f"（可改：{'、'.join(SCENE_EDITABLE)}）"}
        s = self.get_scene(scene_id)
        if s is None:
            return {"ok": False, "detail": "找不到这条场景"}
        applied: dict[str, tuple[str, str]] = {}
        for k, v in fields.items():
            new = ("" if v is None else str(v)).strip()
            if k == "trigger_class" and new and new not in TRIGGER_CLASSES:
                return {"ok": False,
                        "detail": f"情境类只能是这些之一：{'、'.join(TRIGGER_CLASSES)}"
                                  f"（给的是「{new}」）"}
            if k == "time_event" and new and not re.match(r"^\d{4}-\d{2}-\d{2}", new):
                return {"ok": False,
                        "detail": f"事件时间要写具体日期（2026-09-13 或 2026-09-13 10:30:00）；"
                                  f"「{new}」这种说法算不出日期"}
            old = str(getattr(s, k) or "")
            if old != new:
                applied[k] = (old, new)
        if not applied:
            return {"ok": True, "changed": False, "detail": "内容没变"}
        with self.conn:
            for k, (_old, new) in applied.items():
                # 字段名来自上面的白名单校验（不是外部直接输入）——拼 SQL 安全
                self.conn.execute(f"UPDATE scenes SET {k} = ? WHERE id = ?", (new, scene_id))
        return {"ok": True, "changed": True, "applied": applied}

    def set_scene_text(self, scene_id: str, text: str) -> bool:
        """改一条场景的摘要——`set_scene_fields` 的单字段简写（旧调用点与测试沿用）。"""
        return bool(self.set_scene_fields(scene_id, {"text": text}).get("ok"))

    def _write_topics(self, table: str, rid: str, topics: list[str]) -> dict:
        """写主题标签（1-3 个）——**两列只有一个写口**：`topic` 是 `topics[0]` 的
        物化副本（聚合 / 版本序列的 SQL 认它），在这里同步，漂移不可能。

        `topics[0]` 就是主主题；多出来的 0-2 个是附加主题。
        返回 `{"ok", "changed", "applied": {"topic": (旧, 新)}}`（旧 / 新是
        「、」连接的串——留痕与回显直接用）。
        """
        clean: list[str] = []
        for t in (topics or []):
            t = str(t or "").strip()
            if t and t not in clean:
                clean.append(t)
        if not clean:
            return {"ok": False, "detail": "主题不能清空——它是这条记忆的归类"}
        if len(clean) > TOPICS_MAX:
            return {"ok": False,
                    "detail": f"主题最多 {TOPICS_MAX} 个（1 个主主题 + 最多 2 个附加）"}
        row = self.conn.execute(f"SELECT id, topic, topics FROM {table} WHERE id = ?",
                                (rid,)).fetchone()
        if row is None:
            return {"ok": False, "detail": "找不到这条记忆"}
        old_topic = row["topic"] or ""
        old = load_str_list(row["topics"]) or ([old_topic] if old_topic else [])
        if old == clean:
            return {"ok": True, "changed": False, "detail": "内容没变"}
        with self.conn:
            self.conn.execute(
                f"UPDATE {table} SET topic = ?, topics = ? WHERE id = ?",
                (clean[0], json.dumps(clean, ensure_ascii=False), rid))
        return {"ok": True, "changed": True,
                "applied": {"topic": ("、".join(old), "、".join(clean))}}

    def set_summary_topics(self, summary_id: str, topics: list[str]) -> dict:
        """改一条 S2 的**主题标签**（1-3 个，第一个 = 主主题）——原地改
        （改归类：叙述与它收的素材都不动）。"""
        return self._write_topics("summaries", summary_id, topics)

    def set_profile_topics(self, pid: str, topics: list[str]) -> dict:
        """改一条画像的**主题标签**（1-3 个，第一个 = 主主题）——原地改，**不走修正**。

        与"改陈述"（`weave.update_profile_confirmed` → `distill.revise_profile`）的
        差别：改的是**归类**，不是"这话对不对"——判断自己的依据链（`sources` /
        `evidence` / 时间戳）一个字都不动；它只是落进新主题的分组（镜像 / 统计
        按主主题走）。所以旧版不需要进历史（没有"旧说法"要留）。
        """
        return self._write_topics("profiles", pid, topics)

    def set_profile_topic(self, pid: str, topic: str) -> dict:
        """改一条画像的主题——`set_profile_topics` 的**单值简写**（改成一个主题）。

        注意它是**替换**语义：给定一个就把标签集设成这一个（附加主题一并清掉）——
        "改主题"是重新设定归类，不是往上一层层叠。
        """
        return self.set_profile_topics(pid, split_topics(topic))

    def archive_scene(self, scene_id: str) -> bool:
        """把一条场景放进冷层（`archived=1`）——**人的动作**（工具箱稿 §3.4 的中间档）。

        这就是"留后路"那一档：她不召回它（`query_scenes` / 向量都不含归档），
        但数据全在——可打捞、可取消归档，画像的 `sources` 引用**不动**
        （归档不是删，不摘引用）。

        刻意不并进 `archive_s1`：那条是**系统自动**归档（超容量时挑最不重要的），
        这条是**人指定**的（他点了"归档冷存"）。
        """
        with self.conn:
            cur = self.conn.execute(
                "UPDATE scenes SET archived = 1 WHERE id = ? AND archived = 0",
                (scene_id,))
        self._bump_emb()
        return cur.rowcount > 0

    def unarchive_scene(self, scene_id: str) -> bool:
        """取消归档（**人的动作**）——"后悔没删干净"和"后悔删了"都该有出口。"""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE scenes SET archived = 0 WHERE id = ? AND archived = 1",
                (scene_id,))
        self._bump_emb()
        return cur.rowcount > 0

    def exists(self, sid: str) -> bool:
        """这个编号在不在（跨三层 + 备忘）——**读端过滤"引用"用**（2026-09-24）。

        引用 = `sources ∩ 现存节点`（存储层稿 §五）：删一个节点 = 删一行，
        指它的关系在**读取时**自然断。这个方法是那道"读"——
        `render_mirror` / 展开 / 台账给依据列表时都过它。
        """
        up = (sid or "").strip().upper()
        if up.startswith("S1"):
            return self.get_scene(sid) is not None
        if up.startswith("S2"):
            return self.get_summary(sid) is not None
        if up.startswith("S3"):
            return self.get_profile(sid) is not None
        if up.startswith("M"):
            return self.get_memo(sid) is not None
        return False

    def delete_scene(self, scene_id: str) -> dict:
        """真删一条场景：**删行 + 实体链接**（不留快照、不留痕，2026-09-23）。

        **不摘引用**（2026-09-24，存储层稿 §五）：引用读成 `sources ∩ 现存节点`
        ——读取端过滤，**删一个节点 = 删一行**，其余全部自动。
        为什么取消"删的时候摘"：那种同步必然漏（S2 的 `sources` 就漏过），
        而过滤漏不了；而且引用本来就是派生的——少一个节点，指它的关系自然断
        （画像是少一次印证、S2 是少一条素材，都不用手动同步）。

        为什么不留快照 / 不留痕（工具箱稿 §3.4 / §五）：
        留后路由**归档**那一档承担（人点的），删 = 真删——不再挂隐形安全网；
        留痕（标题 + 正文）对删除没有独立理由，也不再写。
        唯一残留是每日快照（`backup_daily` 的整库灾备，7 天自动清理）——
        那是防整库写坏的，不是给单条删除的恢复通道。

        返回 `{"deleted"}`——"它被哪些画像引用"不再是删的动作，
        要看用 `profiles_citing()`（反查现存引用）。
        """
        with self.conn:
            self.conn.execute("DELETE FROM scene_entities WHERE scene_id = ?", (scene_id,))
            cur = self.conn.execute("DELETE FROM scenes WHERE id = ?", (scene_id,))
        self._bump_emb()
        return {"deleted": cur.rowcount > 0}

    def delete_profile(self, pid: str) -> dict:
        """真删一条画像：**删行 + 清它的体检提议**（**人的否决**，2026-09-24）。

        与 `delete_scene` 的对称与差别：
          - 场景是**素材**（不可再生），画像是**推断的结论**——素材全在，
            判断真成立的话新证据攒够它会重新立出来，删的代价天然小；
            要留后路走 `archive_profile`（2026-09-24 起**三层都有归档档**）。
          - 场景删了**不摘引用**（引用读端过滤：`sources ∩ 现存节点`）；
            画像删了更不用管——`sources` / `evidence_at` 是画像自己的字段，
            跟着行一起走，没有第二张表要清。
          - 体检提议要一起清：留着就是悬空编号。

        「系统的自动动作永不硬删」不变（修正 / 老化 / 复核 / 封顶都不删行）——
        这个方法只给**他点名要删**用（界面「不对」/ `forget_memory` 提议他点）。
        """
        with self.conn:
            self.conn.execute("DELETE FROM profile_reviews WHERE profile_id = ?", (pid,))
            cur = self.conn.execute("DELETE FROM profiles WHERE id = ?", (pid,))
        return {"deleted": cur.rowcount > 0}

    def topic_neighbors(self, scene_id: str) -> list[str]:
        """同 topic 里**时间上紧邻**的两条（前一条 / 后一条）——R1 扩散的派生依据。

        2026-09-24：取代 `causality` 边（存储层稿 §五"能派生的不存"）。
        语义与旧边一致：写入端连的是"同 topic 的上一条"，读取端双向查——
        合计就是"同 topic 的相邻对"；这里现算前后各一条，等价。
        **只取相邻、不取全链**：全链会让 R1 扩散退化成"把所有记忆都捞出来"。

        时间列同 `query_scenes` 的口径：优先 `time_event`，空则退 `created_at`。
        不过滤归档——调用方（`recall`）自己过滤，与旧路径一致。
        """
        tcol = "COALESCE(NULLIF(time_event, ''), created_at)"
        me = self.conn.execute(
            f"SELECT topic, {tcol} AS t FROM scenes WHERE id = ?",
            (scene_id,)).fetchone()
        if me is None or not (me["topic"] or "").strip():
            return []
        out: list[str] = []
        for op, order in (("<=", "DESC"), (">=", "ASC")):
            row = self.conn.execute(
                f"SELECT id FROM scenes WHERE topic = ? AND id != ?"
                f" AND {tcol} {op} ? ORDER BY {tcol} {order} LIMIT 1",
                (me["topic"], scene_id, me["t"])).fetchone()
            if row is not None and row["id"] not in out:
                out.append(row["id"])
        return out

    def archive_profile(self, pid: str) -> bool:
        """把一条画像放进冷层（**人的动作**，2026-09-24）——"别再让它影响你，但留着"。

        落法用 `invalidated_at` + `by=INVALIDATED_ARCHIVE`：`current_profiles` 本来就按它
        过滤（不进注入 / 镜像 / 召回），与"修正旧版"共用同一个开关——但语义不同
        （那个被新版取代，这个只是收起来）。数据全在、可取消。
        """
        with self.conn:
            cur = self.conn.execute(
                "UPDATE profiles SET invalidated_at = ?, invalidated_by = ?,"
                " invalidated_reason = ''"
                " WHERE id = ? AND COALESCE(invalidated_at, '') = ''",
                (now_str(), INVALIDATED_ARCHIVE, pid))
        return cur.rowcount > 0

    def unarchive_profile(self, pid: str) -> bool:
        """取消归档（**人的动作**）——只清"归档"那一类失效，修正的历史不动。"""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE profiles SET invalidated_at = '', invalidated_by = '',"
                " invalidated_reason = ''"
                " WHERE id = ? AND invalidated_by = ?", (pid, INVALIDATED_ARCHIVE))
        return cur.rowcount > 0

    def archive_all(self, score_fn=None) -> dict:
        """按配置跑一遍三个归档（后台维护任务的入口）。

        `score_fn` 透传给 `archive_s1`（核心度最低的先走）。
        由调用方给而不是在这里 import：算核心度要用 `recall`，
        而 `recall` 依赖 `store`——反向 import 会成环。
        """
        # 没有 s0 这一项：原文按天存文档，**文档本身就是冷层**，
        # 不需要「超期 → 标记归档」那套（这也是删掉 s0_retain_days 那组旋钮的原因）。
        return {
            "s1": self.archive_s1(cfgmod.cfg("capacity", "s1_cap", default=5000),
                                  score_fn=score_fn),
            "s2": self.archive_s2(cfgmod.cfg("capacity", "s2_cap", default=500)),
        }
