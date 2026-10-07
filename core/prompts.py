"""所有 LLM 提示词与输出 schema——**集中管理，不散落业务逻辑**。

为什么必须集中：提示词是这套系统里最不可测的一部分。散在 `scene.py` /
`shortterm.py` / `recall.py` 里，改一个字段要翻三个文件，而且很容易出现
「prompt 里问了 A，代码里取的是 B」这种静默错位。集中在这里，
schema 和 prompt 就在同一个屏幕上，对不上能一眼看见。

四条硬性原则：
  1. **封闭式提问优先**：能二选一的绝不问开放式。可靠性来自降维 + 封闭。
  2. **强制 JSON schema**：字段名与数据模型一致，输出走 `LLM.structured()`。
  3. **候选清单注入**：topic / 实体名这类要稳定的字段，把已有候选列进 prompt
     让模型挑，而不是自由生成（自由生成必然措辞漂移）。
  4. **失败给保守默认**：拿不准 → NULL / 不猜；`sensitive` 默认 True
     （保守取敏感，它只影响说出口的力度，不影响召回）。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L3 提示词层
#   上游    ：model（枚举常量）
#   下游    ：scene / recall / distill / memo / trend / chat（所有要调模型的地方）
#   对外入口：`<任务>_prompt()` 一组 + 同名的 `*_SCHEMA` 一组 + `build_system_prompt`
#   边界    ：**不调 LLM、不碰 store**——只生产"要说的话"和"要回来的形状"
# ---------------------------------------------------------------------
from __future__ import annotations

import re
from datetime import datetime

from . import config as cfgmod

from .model import (ENTITY_KINDS, MEMO_CLASSES, MEMO_KINDS, MEMO_TIMINGS,
                    SUBJECTS, TRIGGER_CLASSES)


def _fill(template: str, **values) -> str:
    """把 `{名字}` 占位符**单遍**替换掉——替换结果不再被扫描。

    **为什么不能用链式 `str.replace`**：注入进去的值（对话原文、topic、
    LLM 写出来的陈述）里可能恰好含 `{required}` / `{scenes}` 这类**字面量**——
    链式替换会在后面那次 `.replace()` 里把它当占位符换掉：
    用户粘一段提示词进来讨论，内容就被静默改写了。
    `re.sub` 一次扫完，替换进去的内容原样保留。

    （也不能用 `str.format`：这些模板里本来就带着大量 JSON 花括号示例。）
    只替换 `{纯单词}` 形状、且**只在 values 里给了名字的**占位符——
    没给名字的花括号（JSON 示例）原样保留。
    """
    if not values:
        return template
    return re.sub(r"\{(\w+)\}",
                  lambda m: (str(values[m.group(1)])
                             if m.group(1) in values else m.group(0)),
                  template)


# =====================================================================
# 一、场景卡抽取（extract_scene）——写入路径的核心，一次调用出全部字段
# =====================================================================

SCENE_SCHEMA = {
    "name": "scene_card",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # 放在最前：**先决定要不要存，再谈怎么存**。
            "worth_saving":  {"type": "boolean",
                              "description": "这一段值不值得存成场景卡。"
                                             "只有纯寒暄 / 纯应答 / 与刚聊完的完全重复才算 false；"
                                             "**拿不准一律 true**（漏存比错存难查得多）"},
            "skip_reason":   {"type": "string",
                              "description": "worth_saving=false 时说明原因；true 时给空串"},
            "title":         {"type": "string", "description": "场景短标题，≤15 字"},
            "keywords":      {"type": "array", "items": {"type": "string"},
                              "description": "3–6 个核心关键词"},
            "summary":       {"type": "string",
                              "description": "一句话摘要（≤40 字），供检索与画像用"},
            "window_digest": {"type": "string",
                              "description": "替代这段逐字的压缩摘要：**一条消息一行**，"
                                             "每行形如「用户: …」或「<说话人>: …」——"
                                             "**说话人名字照抄输入对话里每行的前缀**"
                                             "（人格可能中途换过，不要一律写「air」），"
                                             "用 \\n 分隔，"
                                             "每行一句话（≤25 字），只写「谁说了什么」，"
                                             "不要加「情绪偏负面」这类旁白。"
                                             "⚠️ 提到时间时**必须带日期**（「9月13日凌晨三点半」），"
                                             "不要只写「凌晨三点半」——只写时刻的话，这段摘要"
                                             "被放进「更早」那一栏后会被当成别的一天。"
                                             "⚠️ **四条必留**（漏掉它们比漏掉情绪更贵）："
                                             "① 承诺 / 待办 / 未决之问 / 说了一半的话，"
                                             "必须单独留一行，写清「谁欠什么 + 什么算完」；"
                                             "② 出现「那个 / 那件事 / 上次说的」时，"
                                             "同一行必须写清所指（代词不许单独过夜）；"
                                             "③ 不可推导的字面锚点原样保留——"
                                             "用户自造的术语、编号、数值、约定名，不许泛化；"
                                             "④ 纠正（「不是 A，是 B」）里的「是 B」那半句，必须留下"},
            "valence":       {"type": ["integer", "null"],
                              "description": "效价：1 正面 / -1 负面 / 0 明确中性 / null 拿不准"},
            "arousal":       {"type": ["integer", "null"],
                              "description": "唤醒度：1 高 / 0 低 / null 拿不准"},
            "trigger":       {"type": "string", "description": "具体触发事件（自由文本）"},
            "trigger_class": {"type": "string", "enum": list(TRIGGER_CLASSES) + [""],
                              "description": "情境类型，只能从枚举里选；没有明确情境就给空串"},
            "reaction":      {"type": "string", "description": "反应（可空）"},
            "outcome":       {"type": "string", "description": "结果（可空）"},
            "subject":       {"type": "string", "enum": list(SUBJECTS),
                              "description": "归属：关于用户本人 / 关于世界（含话题与他人） / 关于 AI 自己"},
            "topic":         {"type": "string",
                              "description": "本条场景的主主题（「主语·主题」，如「用户·面对工作压力的反应」）；"
                                             "优先从给定候选里选，确实没有才新建"},
            "self_ref":      {"type": "boolean",
                              "description": "是否在谈关系本身 / 谈 AI 自己"},
            "air_stance":    {"type": "string",
                              "description": "AI 的实质性表态（承诺/立场/边界/具体信息）；寒暄与通用共情留空串"},
            "sensitive":     {"type": "boolean",
                              "description": "是否涉及健康 / 创伤 / 家庭矛盾 / 隐私；拿不准给 true"},
            "entities":      {"type": "array",
                              "items": {"type": "object", "additionalProperties": False,
                                        "properties": {
                                            "name": {"type": "string"},
                                            "kind": {"type": "string",
                                                     "enum": list(ENTITY_KINDS)},
                                            "relation": {
                                                "type": "string",
                                                "description": "和用户的关系（一小句，"
                                                               "如「妈妈」「同事」"
                                                               "「崇拜的人」）"}},
                                        "required": ["name", "kind", "relation"]},
                              "description": "只记**和用户有个人关系的**实体"
                                             "（家人 / 朋友 / 同事 / 宠物 / 常住地与老家 / "
                                             "用户的项目与作品 / 用户喜欢的东西）——"
                                             "公共人物、公共事件、技术名词、话题词都不记"},
            "open_loops":    {"type": "array",
                              "items": {"type": "object", "additionalProperties": False,
                                        "properties": {
                                            "content": {"type": "string"},
                                            "kind": {"type": "string",
                                                     "enum": list(MEMO_KINDS)},
                                            "due_at": {"type": "string",
                                                       "description": "用户明说的具体时间，没有就给空串"},
                                            "group_name": {"type": "string",
                                                           "description": "同一件事的连续步骤共用一个组名（短语）；"
                                                                          "独立的一件事留空串——拿不准就留空"}},
                                        "required": ["content", "kind", "due_at", "group_name"]},
                              "description": "未闭合线索（未发生的事 / 约定 / 未决之问 / AI 自己的承诺）；"
                                             "一件事的多步用同一个 group_name 串起来"},
        },
        "required": ["worth_saving", "skip_reason", "title", "keywords", "summary",
                     "window_digest", "valence", "arousal",
                     "trigger", "trigger_class", "reaction", "outcome", "subject", "topic",
                     "self_ref", "air_stance", "sensitive", "entities", "open_loops"],
    },
    # 失败时的保守默认（`sensitive=True`：拿不准一律置 1）。
    # valence/arousal 给 None 而不是 0——0 是「确定的中性」，含义不同，不能拿它当缺省。
    "defaults": {
        # **默认 True**：判定失败、字段缺失、模型抽风 —— 一律当"值得存"。
        # 方向不能反：错存一条只是多占一点地方，漏存一条是永久丢记忆。
        "worth_saving": True, "skip_reason": "",
        "title": "", "keywords": [], "summary": "", "window_digest": "",
        "valence": None, "arousal": None, "trigger": "", "trigger_class": "",
        "reaction": "", "outcome": "", "subject": "user", "topic": "", "self_ref": False,
        "air_stance": "", "sensitive": True, "entities": [], "open_loops": [],
    },
}

# 两个 few-shot：一个情感场景（带情绪与前因后果）、一个无机场景（明确中性、
# 不硬编因果、不把通识当 air 表态）。示例比规则更能定住输出形状——
# 只给规则，模型会在「到底多细算 trigger」这种地方自由发挥。
_SCENE_FEWSHOT = """示例 1（情感场景）：
对话：
用户: 今天又被组长当众说了一顿
air: 听着挺难受的
用户: 嗯，算了，我不想争了，干完这段就想走
JSON：
{"title":"被组长当众批评后想离开","keywords":["组长","当众批评","想走"],"summary":"被当众批评后产生逃避念头","window_digest":"用户: 今天又被组长当众说了一顿\nair: 听着挺难受的\n用户: 不想争了，干完这段就想走","valence":-1,"arousal":1,"trigger":"被组长当众批评","trigger_class":"被评价","reaction":"不想争、想离开","outcome":"","subject":"user","self_ref":false,"air_stance":"","sensitive":false,"entities":[{"name":"组长","kind":"person","relation":"组长"}],"open_loops":[]}

示例 2（无机场景）：
对话：
用户: dataclass 和 pydantic 有什么区别
air: dataclass 是标准库，写起来轻；pydantic 带运行时校验，适合接外部数据
用户: 懂了
JSON：
{"title":"问 dataclass 与 pydantic 区别","keywords":["dataclass","pydantic","Python"],"summary":"询问两个 Python 工具的区别","window_digest":"用户: 问 dataclass 和 pydantic 有什么区别\nair: dataclass 轻，pydantic 带运行时校验\n用户: 懂了","valence":0,"arousal":0,"trigger":"","trigger_class":"","reaction":"","outcome":"","subject":"world","self_ref":false,"air_stance":"","sensitive":false,"entities":[],"open_loops":[]}

示例 3（一件事的多步 → 同一个 group_name）：
对话：
用户: 想先把脱敏脚本写了，然后跑一周测试，没大问题就开源
air: 行，我给你记着——脚本、测试、开源这三步
JSON：
{"title":"开源前的三步准备","keywords":["脱敏脚本","测试","开源"],"summary":"计划先写脚本、再测试、最后开源","window_digest":"用户: 先写脱敏脚本，然后跑一周测试，没问题就开源\nair: 我给你记着这三步","valence":0,"arousal":0,"trigger":"","trigger_class":"","reaction":"","outcome":"","subject":"user","self_ref":false,"air_stance":"","sensitive":false,"entities":[],"open_loops":[{"content":"写开源脱敏脚本","kind":"user_task","due_at":"","group_name":"开源准备"},{"content":"跑一周测试，观察 BUG","kind":"user_task","due_at":"","group_name":"开源准备"},{"content":"没问题就开源","kind":"user_task","due_at":"","group_name":"开源准备"}]}
"""

_SCENE_PROMPT = """你在把一段对话整理成「场景卡」。**只输出 JSON，不要解释、不要 markdown 围栏。**

字段与规则：
- **worth_saving / skip_reason：先决定「这段值不值得存」**。
  ⚠️ **默认是 `true`，绝大多数情况都该给 true**——漏存一条 = 永久丢记忆，
  而错存一条只是多占一点地方，两者代价差得远。
  只有下面这几种才给 `false`（并在 skip_reason 里写清是哪种）：
  - 纯寒暄（「你好」「在吗」「吃了没」），没有别的内容
  - 纯应答 / 确认（「嗯」「好的」「知道了」「哈哈」）
  - 只有信息查询、且没透露出关于用户自己的任何东西（「今天几号」）
  - 与**刚刚才存过的那段**完全重复（不是"有点像"，是同一件事重说一遍）
  ⚠️ **拿不准就给 true**。但凡说出了感受、做了什么、和谁有关、有什么打算，
  哪怕是小事，都算值得存——那正是"用户是什么样的人"的原料。
  ⚠️ **判据不是「像不像画像素材」**：用户提到的人和事、用户在某个话题上的看法，
  这些独立成料，常常比关于用户本人的那几句更值钱，**不算"不相关"**。
  真的拿不准，只问一句：**三个月后回看，这条还有没有信息量？** 有，就存。
- title：这条场景的短标题，≤15 字。
- keywords：3–6 个核心关键词。
- summary：**一句话摘要（≤40 字）**，供检索与画像用。
- window_digest：**替代这段逐字的东西**，以后会以「一行一条」的样子出现在 AI 的窗口里。
  写法：**一条消息一行**，形如 `用户: …` / `<说话人>: …`，行与行之间用换行，顺序不变。
  ⚠️ **说话人名字照抄输入对话里每行的前缀**——人格可能中途换过（air / mia / xina……），
  不要一律写「air」（署名错了，AI 会把别人的话当成自己说过的）。
  每行**一句话、只写"谁说了什么"**（≤25 字），不要加"情绪偏负面""总体而言"这类旁白——
  旁白会被当成发生过的事。
  它和 summary 不是一回事：summary 是给检索和画像用的一句话结论，
  这里是还给窗口的流水账。
  ⚠️ **每条消息都要有自己的一行**，不要把一整段压成一行、也不要跨行写。
  ⚠️ 里面**提到时间必须带日期**（「9月13日凌晨三点半」「昨天下午」），
  **不要只写「凌晨三点半」**——这段摘要以后会被放到「更早」那一栏，
  只写时刻的话它会分不清是刚才还是前几天。
- valence：1 正面 / -1 负面 / 0 明确中性 / **null 拿不准**。
  ⚠️ 不要用 0 代替 null——0 是「确定的中性」，含义不同。
- arousal：1 高 / 0 低 / null 拿不准。
- trigger / reaction / outcome：具体触发 → 反应 → 结果。
  **没有明确因果就给空串，不要编。**
- trigger_class：**只能从固定枚举里选**（这是分类，不是自由生成）：
  被评价 / 目标未达 / 关系冲突 / 失去 / 得到达成 / 转折改变 / 悬而未决。
  没有明确的情境类型就给空串。
  ⚠️ **几类的边界要拿准**（分类不稳会直接卡住画像收敛）：
  - **被评价**：被批评、被否定、被质疑、**说话被打断 / 被打压**——
    只要是「对方在贬低或否定用户」，都归这一类（**包括被当众说、被挑刺、被打断**）
  - **关系冲突**：双方**有来有回**的对立、争吵、冷战——不是单方面被说
  - **目标未达**：用户自己想做成的事没做成（跟"被别人评价"无关）
  - **悬而未决**：事情还没发生 / 还没结果（面试、体检、等答复）
- subject：user（关于用户本人）/ world（关于世界，含话题与他人）/ air（关于 AI 自己）。
- topic：本条场景的**主主题**，形如「主语·主题」（例：「用户·面对工作压力的反应」
  「猫·健康问题」「air·边界感」）。
  ⚠️ **主语一律用「用户」指代用户本人**——不要写成「他」「对方」「当事人」。
  ⚠️ **AI 一侧的主语用对话里 AI 的署名**（照抄每行前缀，如 air / mia / xina）——
  不要一律写「air」，也不要写「AI」（署名要能对上当时是谁在说）。
  （措辞统一是版本序列能不能串起来的前提：同一个人的同一个主题，
  换个说法就变成两个主题了。）
  **优先从上面给出的已有主题里挑措辞**；确实没有对应的才新建。
  一条场景跨多个主题时，只取主主题一个。
  ⚠️ 变的是陈述，不变的是「它在说哪件事」——同一件事换了说法也要沿用同一个 topic，
  否则版本序列会断。
- self_ref：是否在谈这段关系本身、或谈 AI 自己。
- air_stance：**只抽 AI 的实质性表态**（承诺 / 立场 / 边界 / 具体信息）。
  判据：**三个月后重读还有没有用？** 有用才抽；寒暄和「我理解你的感受」这类
  通用共情留空串。转述通识信息（查得到的东西）也不算。
- sensitive：涉及健康、创伤、家庭矛盾、隐私 → true。**拿不准给 true。**
- entities：**只记和用户有个人关系的**——家人 / 朋友 / 同事 / 宠物 / 常住地与老家 /
  用户的项目与作品 / 用户喜欢的东西。判据是这段话里**说出来的关系或态度**
  （「我妈」「我同事小王」「我家猫」「我崇拜他」）。
  relation 写「和用户的关系」一小句（「妈妈」「同事」「崇拜的人」）；
  **写不出关系就不列这个实体**。
  ⚠️ **公共人物、公共事件、技术名词、话题词都不记**——用户提一百次也不记；
  只有出现在关系 / 态度里（「我崇拜他」）才记一笔。
  ⚠️ **还没发生的事不要放这里**（面试、体检、约定、等答复）——那是 open_loops。
- open_loops：未闭合线索——**还没发生的事**、约定、未决之问、**AI 自己的承诺**。
  ⚠️ **键名和形状是固定的**，就这四个键：
    `[{"content": "事项原文", "kind": "user_task", "due_at": "YYYY-MM-DD", "group_name": ""}]`
  （把 content 写成 item / title、把 kind 写成 type，这一条会被**整条丢弃**。）
  - ⚠️ **一件事的连续步骤写同一个 `group_name`**（短短语，如「开源准备」）：
    「先写脚本、然后跑测试、没问题就开源」是**一件事的三步**，三条都挂「开源准备」。
    而**独立的两件事各留空**（「妹妹结婚」与「月底回老家」不是一件事）——
    拿不准就留空：**宁可分成两件，也不要硬塞一组**（塞错了你会把不相干的事说成一件事）。
  - `kind` 只能是 `user_task`（用户自己的事）或 `air_promise`（AI 的承诺）。
  - `content` 写**那件具体的事**本身，不是对它的评论。
  ⚠️ **用户说「我下周三要面试」也要抽**——这是用户给出的、还没发生的事。
  - ⚠️ **用户说了时间的，due_at 必须填成绝对日期**（格式 YYYY-MM-DD）。
    「下周三面试」→ 抽出事项「下周三面试」，due_at 换算成具体日期。
    换算时就近取用**末尾给出的当前时间**。
    **留空意味着它永远不会到期**——那这条就白记了。
  - ⚠️ **客套话不是承诺**：AI 说「有想聊的随时来」「加油」「早点休息」这类不算。
    只有**说定了的事**才算（「我会提醒你」「下次给你看」）。
  - ⚠️ 抽**那件具体的事**，不是对它的评论：
    「我下周三要面试，有点紧张」→ 抽「下周三面试」，
    **不要**抽成「紧张的原因未明」。

**不要试图判断**「这是情绪扰动还是纠偏」「对方是发泄还是提建议」——
那些留给对话里去梳理，这里只记客观可抽的。

{candidates}
{fewshot}
对话：
{conversation}

{time_context}

⚠️ **输出前自检**：下面的键**一个都不能少**（没内容就给空串 / 空数组，但键必须在）：
{required}
尤其 **topic 不能是空串**——它决定这段记忆以后能不能跟同主题的串起来。

JSON："""


# =====================================================================
# 时间上下文（注入在提示词末尾）
# =====================================================================

_WEEKDAY = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _period(hour: int) -> str:
    """时段（中文习惯）。

    它不是冗余：把「21:47」翻译成「晚上」，才是真正影响语气的那一步——
    而这个翻译同样不该让模型自己做（见 `time_context` 的说明）。
    """
    if hour < 5:
        return "凌晨"
    if hour < 8:
        return "早上"
    if hour < 11:
        return "上午"
    if hour < 13:
        return "中午"
    if hour < 17:
        return "下午"
    if hour < 19:
        return "傍晚"
    if hour < 23:
        return "晚上"
    return "深夜"


def time_context(now: datetime | None = None) -> str:
    """当前时间的完整表述——**注入在提示词末尾**。

    为什么给得这么细（年月日 + 星期 + 时段 + 时分）：
    模型自己推不出「现在是什么时候」，而不少判断都要用它——
    「下周三」是哪天、深夜收到消息要不要多问两句、「最近」是几天。
    **能直接给的，就不要让它推**：每多一步推算，就多一次错的机会，
    也多占一份本该用在别处的注意力。

    放末尾的理由见 `build_system_prompt`。
    """
    now = now or datetime.now()
    return (f"当前时间：{now.year}年{now.month}月{now.day}日"
            f"（{_WEEKDAY[now.weekday()]}）{_period(now.hour)} {now.hour:02d}:{now.minute:02d}")


_CN_DIGITS = "一二三四五六七八九"


def _cn_days(n: int) -> str:
    """1-9 写成人话：量词前是「两」不是「二」（精算区只到七天，够用）。"""
    return "两" if n == 2 else _CN_DIGITS[n - 1]


def rel_day(when: str, now: datetime | None = None) -> str:
    """「刚刚 / 17 分钟前 / 3 小时前 / 一天前 / …」——给绝对时间补一个**算好的**相对时间。

    为什么由代码算：日期算术是模型的弱项，而「这是多久以前」它每轮都在用——
    直接决定措辞（刚刚 / 昨晚 / 前几天 / 上个月）。实测就出过错：把 17 分钟前
    说成「昨晚」（见 `ShortTerm._dated_digest` 的注释）。理由同 `time_context`：
    **能直接给的，就不要让它推**。

    必须**渲染时**调用、不能存进库：同一条记忆，今天看是「一天前」，
    明天再看就是「两天前」——写死的那一刻它就错了。

    粒度三档（2026-09-20 第四版：补上**一天内**）：
      · 一天内（输入带时刻时）：<1 分钟「刚刚」，否则「N 分钟前」；满 1 小时
        「N 小时前」——这段是**误判率最高**的区间（上面那个实证就是它），
        也恰好最难心算（时钟减法），最该给现成的；跨天但不满 24 小时仍按小时，
        「昨晚 23:00」在第二天早上是「N 小时前」，不会被说成「一天前」；
      · 一周内精算到天（一天前 … 七天前）——这几天的区分度是真实的；
      · 之外只给**锚点**（一周前 / 几周前 / 一月前 / 几个月前 / 一年前 / 两年前）——
        「二十三天前」和「三周前」给的信息一样多，精确天数读起来反而费劲。
    断点取自然界限（24 小时、14 = 两周、30 ≈ 一月、365 = 一年）；
    只有日期（没有时刻）的输入直接走天级；未来时间与脏数据返回空串。
    """
    when = (when or "").strip()
    if not when:
        return ""
    now = now or datetime.now()
    # 带时刻（「YYYY-MM-DD HH:MM」及以上）先按分钟 / 小时；只有日期走天级。
    # 有秒就连秒一起解析——截到分钟会把「刚刚」的边界推偏（30 秒前判成 1 分钟前）
    has_clock = len(when) >= 16
    if len(when) >= 19:
        text, fmt = when[:19], "%Y-%m-%d %H:%M:%S"
    elif has_clock:
        text, fmt = when[:16], "%Y-%m-%d %H:%M"
    else:
        text, fmt = when[:10], "%Y-%m-%d"
    try:
        dt = datetime.strptime(text, fmt)
    except ValueError:
        return ""
    secs = (now - dt).total_seconds()
    if secs < 0:
        return ""
    if has_clock:
        if secs < 60:
            return "刚刚"
        if secs < 3600:
            return f"{int(secs // 60)} 分钟前"
        if secs < 86400:
            return f"{int(secs // 3600)} 小时前"
    n = (now.date() - dt.date()).days
    if n == 0:
        return "今天"
    if n <= 7:
        return f"{_cn_days(n)}天前"
    if n < 14:
        return "一周前"
    if n < 30:
        return "几周前"
    if n < 60:
        return "一月前"
    if n < 365:
        return "几个月前"
    if n < 730:
        return "一年前"
    return "两年前"          # 再远的分档等真有那么老的记忆再加——先别为想象分类


def rel_stamp(when: str, now: datetime | None = None) -> str:
    """时间戳带上相对日：`（今天）2026-09-20 06:24`——**所有出口共用这一种拼法**。

    为什么要有这一个函数：同一个时间戳会从好几条路径到她眼前（注入的记忆行、
    窗口消息、她调工具查的结果……）。各自拼字符串的话，格式迟早漂移；
    收在这里，「不管从哪条路看，时间戳长得一样」就是代码保证的，不靠自觉。

    相对日算在**渲染时刻**（见 `rel_day`——它绝不能落盘，落盘即过期）；
    日期没有相对日（未来 / 脏数据）时原样给。
    """
    when = (when or "").strip()
    if not when:
        return ""
    rel = rel_day(when, now)
    return f"（{rel}）{when}" if rel else when


def required_keys(schema: dict) -> str:
    """从 schema 生成「必填键」清单——**不手写第二遍**。

    为什么这块自检不能省：`json_schema` 档一旦被降级成 `json_object`
    （DeepSeek 就是如此），schema 的约束**完全失效**——模型只知道
    「输出 JSON」，完全不知道有哪些键。字段名全靠 prompt 里的文字，
    漏一个是**静默**的：`defaults` 会补上空串，看起来"跑通了"，
    实际存进去的是空字段。

    从 schema 生成（而不是在 prompt 里再抄一遍键名）是为了让两处不会漂移：
    schema 加字段，清单跟着变。真实踩过——加了 `topic` 字段、
    prompt 里也写了说明，模型仍然整条漏掉，直到看落库结果才发现。
    """
    req = (schema.get("schema") or {}).get("required") or []
    return "、".join(req)


def extract_scene_prompt(conversation: str, topic_candidates: list[str] | None = None,
                         now: datetime | None = None) -> str:
    """场景卡抽取的 prompt。

    `topic_candidates` 是「候选清单注入」的落点：把已有 topic 列出来让模型挑，
    而不是让它自由生成——自由生成必然措辞漂移，版本序列会断成两半。

    `now` 决定末尾注入的时间（默认取当下）。相对时间（「下周三」）
    换算成 `due_at` 全靠它——没给的话模型只能瞎猜或留空，而留空的备忘录
    永远不会到期（等于没记）。
    """
    candidates = ""
    if topic_candidates:
        lst = "\n".join(f"  - {t}" for t in topic_candidates)
        candidates = (
            "已存在的主题：\n" + lst + "\n"
            "⚠️ **判断标准是「说的是不是同一件事」，不是「措辞像不像」。**\n"
            "   「用户·被当众批评的反应」和「用户·被批评的反应」显然在说同一件事 ——\n"
            "   遇到这种情况**必须原样选一个已有的**（选更贴切的那个），\n"
            "   绝不要为同一件事再造第三种说法。\n"
            "   只有当它**确实是另一件事**时，才新建。\n"
            "   （同一个主题被写成多个说法，后面的聚合和画像就全废了 ——\n"
            "     每边都凑不够数。这是这套系统最容易被写坏的地方。）\n")
    return _fill(_SCENE_PROMPT, candidates=candidates, fewshot=_SCENE_FEWSHOT,
                 conversation=conversation, time_context=time_context(now),
                 required=required_keys(SCENE_SCHEMA))


# =====================================================================
# 二、查询侧线索判断（compute_cues）——一次调用判 C2/C3/C5/C7，省调用次数
# =====================================================================

CUE_SCHEMA = {
    "name": "cue_judgement",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tense":   {"type": "string", "enum": ["past", "now", "future", "hypo"],
                        "description": "这句话在指涉过去 / 当下 / 未来 / 假设"},
            "valence": {"type": ["integer", "null"],
                        "description": "1 正面 / -1 负面 / 0 明确中性 / null 拿不准"},
            "arousal": {"type": ["integer", "null"],
                        "description": "1 高唤醒 / 0 低唤醒 / null 拿不准"},
            "about_relation": {"type": "boolean",
                               "description": "是否在谈这段关系本身、或谈 AI 自己"},
            "unresolved": {"type": "boolean",
                           "description": "用户是否卡在悬而未决的问题上（犹豫 / 矛盾 / 未决）"},
        },
        "required": ["tense", "valence", "arousal", "about_relation", "unresolved"],
    },
    # 默认值全部取「不作为」侧：拿不准就不触发任何动作（不确定就不作为）。
    "defaults": {"tense": "now", "valence": None, "arousal": None,
                 "about_relation": False, "unresolved": False},
}

_CUE_PROMPT = """判断这条消息的五个属性。**只输出 JSON，不要解释。**

- tense：指涉 过去(past) / 当下(now) / 未来(future) / 假设(hypo)。
- valence：情绪效价 1 正面 / -1 负面 / 0 明确中性 / **null 拿不准**（不要用 0 代替 null）。
- arousal：唤醒度 1 高 / 0 低 / null 拿不准。
- about_relation：是否在谈这段关系本身、或谈 AI 自己。
- unresolved：用户是否卡在悬而未决的问题上（犹豫 / 矛盾 / 未决）。

消息：{msg}
JSON："""


def cue_prompt(msg: str) -> str:
    """线索判定（C2/C3/C5/C7）→ `CUE_SCHEMA`。

    一次判五个封闭式属性（分开问四次的成本线性涨、收益不变）；
    默认值全取「不作为」侧——拿不准就不触发动作。
    """
    return _fill(_CUE_PROMPT, msg=msg or "")


# =====================================================================
# 三、提炼流水线（阶段 2）：topic 判断 → S1→S2 聚合 → S2→S3 抽象 → 修正判定
#
#     这一组 prompt 围绕一条主线：**聚合不下结论，抽象才下判断**。
#     两者的 prompt 必须分开写——混用会让 S2 里长出"他是个消极的人"这种
#     特质标签，而那正是逻辑层 §4 明确要避免的东西（模式可观测、可验证、可否决；
#     特质是武断的标签）。
# =====================================================================

def _render_scenes(scenes, limit: int = 12, with_id: bool = True) -> str:
    """把场景卡渲染成给 LLM 看的文本（聚合与抽象共用）。

    只给「前因后果 + 摘要 + 时间」，**不给强度/引用数这类内部指标**——
    那些是我们的排序依据，不是让模型用来"判断这个人"的材料。
    给了它只会诱导模型朝"重要的事"倾斜，而重要性该由印证次数决定，不该由它猜。
    """
    lines = []
    for s in (scenes or [])[:limit]:
        head = f"- [{s.id}] {s.title}" if with_id else f"- {s.title}"
        when = (getattr(s, "time_event", "") or "")[:10]
        if when:
            head += f"（{when}）"
        lines.append(head)
        if getattr(s, "trigger", "") or getattr(s, "reaction", "") or getattr(s, "outcome", ""):
            lines.append(f"    情境：{s.trigger or '(未记)'} → 反应：{s.reaction or '(未记)'} "
                         f"→ 结果：{s.outcome or '(未记)'}")
        if getattr(s, "text", ""):
            lines.append(f"    摘要：{s.text}")
    return "\n".join(lines)


# ---- 3.1 topic 判断（新建 vs 复用）----

TOPIC_SCHEMA = {
    "name": "topic_choice",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string",
                      "description": "选中的已有主题（**原样输出**）；若都不合适，"
                                     "给一个新建的「主语·主题」"},
        },
        "required": ["topic"],
    },
    # 降级给空串：**拿不准就不写**（上层据此放弃这次抽象）。
    # topic 是版本序列的唯一纽带，猜一个错的会持续污染（存储层 §3「拿不准就新建」
    # 说的是新建 topic，不是说可以编一个）。
    "defaults": {"topic": ""},
}

_TOPIC_PROMPT = """下面有一条关于某个人的新陈述。请判断它说的是不是**已经在册的某件事**。

已有主题（格式是「主语·主题」，「主语」是 user / 世界的某个实体 / air）：
{candidates}

新陈述：{statement}
（它的归属是：{subject}）

规则：
- 属于其中某件事 → **原样输出那个主题**（一个字都别改，措辞漂移会让版本序列断成两半）
- 确实都不属于 → 新建一个「主语·主题」，形如「用户·面对工作压力的反应」「猫·健康问题」
- ⚠️ 「说的是不是同一件事」看的是**这件事本身**，不是措辞像不像。
  「用户遇到压力会退出」和「用户遇到压力会先扛一下」是**同一件事的不同阶段** → 复用。
  「用户面对压力的反应」和「用户怎么带孩子」是两件事 → 新建。

只输出 JSON：{"topic": "..."}
"""


def topic_prompt(statement: str, subject: str, candidates: list[str]) -> str:
    """topic「新建 vs 复用」的判断 prompt（存储层 §3）。

    候选清单注入是必须的：让模型自由生成 topic，措辞微差
    （「面对工作压力的反应」vs「应对工作压力的方式」）就会让同一件事串不成序列。
    """
    if candidates:
        cand = "\n".join(f"  - {t}" for t in candidates)
    else:
        cand = "  （册子里还是空的——这是第一件事，直接新建）"
    subj_label = {"user": "关于用户本人", "world": "关于世界（含第三方人 / 宠物 / 事件 / 话题）",
                  "air": "关于 AI 自己"}.get(subject, subject)
    return _fill(_TOPIC_PROMPT, candidates=cand, statement=statement or "",
                 subject=subj_label)


# ---- 3.2 S1 → S2 聚合（**不下结论**）----

SUMMARY_SCHEMA = {
    "name": "topic_summary",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string",
                     "description": "把这几条场景聚合成的一段连贯叙述（只说发生了什么）"},
            "extra_topics": {
                "type": "array", "items": {"type": "string"},
                "description": "这条叙述顺带沾到的**已有主题**（从给定清单里原样挑，"
                               "最多 2 个；没有就给空数组）"},
        },
        "required": ["text", "extra_topics"],
    },
    "defaults": {"text": "", "extra_topics": []},
}

_SUMMARY_PROMPT = """把下面几条场景**聚合成一段连贯的叙述**。

主题：{topic}

{scenes}

规则（这是**聚合**，不是**抽象**）：
- 只说**发生了什么**：把它们串成一段有先后的话
- ❌ **不要下任何结论**：不写「用户容易焦虑」「用户遇事就退」这类判断
- ❌ 不要写「这反映了……」；不要在结尾升华
- ✅ 可以指出这几条在时间上的先后、情境上的异同（那是描述，不是判断）
- 长度控制在 150 字以内

另外：这条叙述如果还牵涉别的**已有主题**，顺带挂上（最多 2 个）——它只服务
「找得到」，不参与聚合。**只能从下面清单里原样挑，一个字都别改；不要编新的**：

已有主题：
{candidates}

- 「沾边」不算沾：它得是这条叙述真正涉及的话题；一个都不沾就给空数组。

只输出 JSON：{"text": "...", "extra_topics": []}
"""


def summary_prompt(topic: str, scenes, candidates: list[str] | None = None) -> str:
    """S1→S2 **聚合**（`distill_step2` 用）→ `SUMMARY_SCHEMA`。

    「不下结论、不升华」不是措辞讲究：聚合层**没有印证机制**，所以只能叙述——
    一旦 S2 里出现「他是个消极的人」，抽象层会当现成材料继承下去。

    `candidates`：**附加主题**的候选（已有主题去掉主主题，2026-09-24 晚）。
    只让它从清单里挑、不许编新的：附加主题虽然不参与链路，但它会进检索——
    编出来的一堆近义主题（「工作压力」vs「用户·工作压力」）会让"按主题找"变糊
    （同 `topic_prompt` 候选注入的理由）。
    """
    cand = "\n".join(f"  - {t}" for t in (candidates or [])) or "  （暂时没有别的主题）"
    return _fill(_SUMMARY_PROMPT, topic=topic or "",
                 scenes=_render_scenes(scenes), candidates=cand)


# ---- 3.3 S2 → S3 抽象（**下判断**，必须可追溯）----

PROFILE_SCHEMA = {
    "name": "profile_abstraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "statement": {"type": "string",
                          "description": "一条称重方式的陈述（用户在什么之间取舍、往哪边偏）；"
                                         "写不出取舍再退可观测模式；证据不足给空串"},
            "extra_topics": {
                "type": "array", "items": {"type": "string"},
                "description": "这条判断顺带沾到的**已有主题**（从给定清单里原样挑，"
                               "最多 2 个；没有就给空数组）"},
        },
        "required": ["statement", "extra_topics"],
    },
    "defaults": {"statement": "", "extra_topics": []},
}

# 画像陈述的写法判据——**一处定义，三处引用**（P1/P2，2026-09-21）。
#
# 三个入口（新抽 `profile_prompt` / 修正 `revision_prompt` / 渐变 `drift_prompt`）
# 产出的都是"一条画像陈述"，写法必须一致——各写一遍的后果：新画像按称重方式写、
# 修正一次退回行为对，写法在几根轨道上来回漂（同本地「待优化稿」A 条的教训：
# 格式要一处定义）。详见本地「待优化稿-画像层」P1/P2。
#
# 为什么优先写"称重方式"而不是"遇到 X → 反应"：同一批材料，前者推得动新情境
# （他还没遇到的事，换成他会怎么取舍），后者只能复述旧事。2026-09-21 首轮对照
# 实测过：同一材料、只换这段 prompt，输出从行为对变成了取舍句。
PROFILE_STATEMENT_RULES = """- ✅ 优先写**用户的称重方式**——「在 X 和 Y 之间，用户往哪边偏」。
  这是比「遇到 X 情境 → 反应」更深一层的说法：同一批材料，前者推得动
  **新情境**（用户还没遇到的事，换成用户会怎么取舍），后者只能复述旧事。
  例：不写「用户遇到批评会退」，写「被评价的场合，用户把'不被卷入'放在'争一口气'前面」。
  材料够不着「取舍」时退回可观测模式写——但先试着想一步：**用户这样选，是拿什么换了什么？**
- ❌ 不写**特质标签**：「用户是个消极的人」「用户缺乏安全感」——
  那是给人下定义，用户没法否决（否了还剩一半对）；
  而取舍描述用户能当场否（「那次不是不敢，是不值得」）。
  判据一句话：**不是写用户是什么人，是写用户如何取舍。**
- ❌ 不写**评价词**：「用户很固执」「用户要强」「用户敏感」——那是特质标签的软版本，
  带好坏的色彩，同样只能整条接受或整条否认。**只写用户怎么选，不评判这个选法**：
  哪边好哪边坏，读的人自己看得出来。
- ❌ 不写**车轱辘话**：这句话要能把用户和别人分开——「用户有时候外向有时候内向」
  这类谁看都像在说自己（巴纳姆式），不是判断，是氛围，等于没写。
- 材料里看得出**变化**（早期一个样、近期另一个样）时，写**现在的**取舍，
  或带上时间（「那阵子用户……」）——别让两段凑成一条不存在的稳定模式。
- ❌ 不写诊断、不写心理学名词；**机制名 / 系统术语也不进画像**——写取舍用**人的词**
  （反面例：「把原话留底放在顺口硬接之前」——"原话留底"是系统词，2026-09-21 实测擦过边）。
- **一条陈述只写一个情境**：材料里有几个不同情境时，选支撑最厚的那一个写。
  不要把「被追问 A、被改 B、被提 C 时」这类枚举并进同一句——那是把几件事
  撮合成一条：既不像稳定模式，也撑不住「同一情境的反复印证」。
- **AI 一侧的主题**（主语是 air / mia / xina 这类署名，即关于 AI 自己）只写
  **关系层面的姿态**（怎么陪、何时停、边界在哪——同样往"取舍"上写）。
  排查过程、实现机制、系统改造这类工程内容不写——那是调试记录，不是关于 AI 的判断。
- 允许**反推一步**：从反应反推被用户放弃的选项（「不想争、想走」——
  用户没说"本来可以争"，但写得出用户选了离开）——但必须有材料支撑
  （至少两条场景的行为方向一致），不许凭空发明「用户本来可以……」。
- 一句话，不超过 40 字"""


_PROFILE_PROMPT = """从下面的材料里，抽出一条关于这个人的**模式陈述**。

主题：{topic}

已有的主题叙述（聚合）：
{summaries}

支撑它们的场景：
{scenes}

规则（这是**抽象**，会下判断，所以要格外克制）：
{rules}
- **证据不足就给空串**——宁可这次不写，也不要下一条撑不住的判断。
  一条陈述至少要能被上面 3 条场景支撑。

两个示例（材料 → 好 / 坏写法）：
示例一（user 类材料）：
材料：三次被当众批评，三次都是「不想争、想走」。
✅「被评价的场合，用户把'不被卷入'放在'争一口气'前面」——推得出新情境
❌「用户遇到批评会退」——行为对：只是把材料复述一遍
❌「用户缺乏安全感」——特质：用户没法否决

示例二（air 类材料）：
材料：三次被指出问题，三次都是"查下去"。
✅「air 宁可把原因查清，也不图省事翻篇」——取舍，且是人的词
❌「把原话留底放在顺口硬接之前」——机制词：系统词不是人的话

另外：这条判断如果还牵涉别的**已有主题**，顺带挂上（最多 2 个）——它只服务
「找得到」，不参与印证与版本序列（那条序列认的是主主题）。**只能从下面清单里
原样挑，不要编新的**；一个都不沾就给空数组：

已有主题：
{candidates}

只输出 JSON：{"statement": "...", "extra_topics": []}
"""


def profile_prompt(topic: str, summaries, scenes,
                   candidates: list[str] | None = None) -> str:
    """S2→S3 **抽象**（`distill_step3` 用）→ `PROFILE_SCHEMA`。

    全系统**唯一允许下判断**的一份，所以约束写得最重：优先写**称重方式**
    （「在 X 和 Y 之间，用户往哪边偏」），写不出来退可观测模式
    （「遇到 X 情境 → 反应 → 结果」），不要特质标签——前者能被印证也能被否掉。

    **写法判据是 `PROFILE_STATEMENT_RULES`**——它与修正 / 渐变两处**共用同一段**
    （2026-09-21，P1/P2）：三个入口产出的都是"一条画像陈述"，
    各写一遍必然漂（新画像按权重写、修正一次退回行为对）。

    **证据不足就给空串**：宁可不抽象，也不要一条撑不住的判断（它会常驻注入）。

    2026-09-20 按当天两条画像的教训补了两条约束：① 材料里几个情境被枚举进同一句
    （「被追问，或被改机制，或被提分工时……」）——读起来像工单，而且收敛关要的是
    「同一情境的反复印证」，枚举本身就说明它们不是同一件事（那条画像 evidence=4
    仍立不起来，正是卡在这）；② 关于 air 的画像全写成了「她排查 / 她实现」——
    工程表现不是关于她的判断，air 类只装关系层面的姿态。

    2026-09-21 再补两处（首轮对照实测带出）：① 陈述升级为"称重方式"优先；
    ② **机制名 / 系统术语不进画像**——实测写出过「把原话留底放在顺口硬接之前」
    （"原话留底"是系统词）。两处都已并进 `PROFILE_STATEMENT_RULES`。
    """
    s2_text = "\n".join(f"- [{s.id}] {s.text}" for s in (summaries or [])) or "（无）"
    # 附加主题的候选（2026-09-24 晚）：同 `summary_prompt`——只从清单里挑，不许编新的。
    cand = "\n".join(f"  - {t}" for t in (candidates or [])) or "  （暂时没有别的主题）"
    return _fill(_PROFILE_PROMPT, topic=topic or "", summaries=s2_text,
                 scenes=_render_scenes(scenes), rules=PROFILE_STATEMENT_RULES,
                 candidates=cand)


# ---- 3.4 修正判定（新证据 vs 已有画像）----

REVISION_SCHEMA = {
    "name": "profile_revision",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": ["holds", "revise", "overturn"],
                        "description": "holds 维持 / revise 需要修改 / overturn 被推翻"},
            "statement": {"type": "string",
                          "description": "verdict 为 revise 或 overturn 时给新陈述；holds 给空串"},
        },
        "required": ["verdict", "statement"],
    },
    # 降级给 holds：**保守不动**。拿不准时维持现状，比改动一条可能正确的判断安全
    # （改动是不可逆的：旧记录会被填上 invalidated_at）。
    "defaults": {"verdict": "holds", "statement": ""},
}

_REVISION_PROMPT = """已经有一条关于这个人的判断，现在出现了新证据。请判断这条判断还成不成立。

已有判断：{statement}
（主题：{topic}）

新证据（这些是这次新出现的情况）：
{evidence}

规则：
- **holds**：新证据与判断一致，判断继续成立（情境不同但反应同向也算成立）
- **revise**：判断的大方向还在，但需要修正（人会变——「会退出」变成「会先扛一下再决定」）
- **overturn**：新证据与判断**直接矛盾**，判断是错的
- ⚠️ 单次观察不足以推翻一条已经多次印证的判断——除非它是明确的反例
- ⚠️ 不要因为「这次不一样」就改：先问「是人变了，还是只是情境不同」

revise / overturn 时给出一条新陈述；holds 时 statement 给空串。
新陈述的写法**与抽取时同一套判据**（一处定义、三处生效——别退回行为对）：
{rules}

只输出 JSON：{"verdict": "holds|revise|overturn", "statement": "..."}
"""


def revision_prompt(topic: str, statement: str, new_scenes) -> str:
    """新证据 vs 已有画像（突变分支）→ `REVISION_SCHEMA`。

    三选一（holds / revise / overturn），降级给 `holds`：改动不可逆
    （旧记录要填 `invalidated_at`），拿不准就维持现状。

    新陈述的写法判据来自 `PROFILE_STATEMENT_RULES`（P2：与抽取 / 渐变同出一个源头，
    防"新画像按权重写、修正一次退回行为对"）。
    """
    return _fill(_REVISION_PROMPT, topic=topic or "", statement=statement or "",
                 evidence=_render_scenes(new_scenes), rules=PROFILE_STATEMENT_RULES)


# ---- 3.5 渐变判定（早期 vs 近期放在一起看）----

_DRIFT_PROMPT = """有一件事的判断，可能正在**慢慢变化**。
注意：**没有哪一条单独的记录算得上反例**——所以请把早期和近期**放在一起看**，
判断是人变了，还是只是最近聊的事不一样。

已有判断：{statement}
（主题：{topic}）

早期的情况：
{early}

近期的情况：
{recent}

三方选择（同「修正判定」的那三个）：
- **holds**：整体看仍然一致——最近聊的事不同，但反应模式没变
- **revise**：模式确实在移动（例如从「退出」变成「先扛一下再决定」）→ 给新陈述
- **overturn**：已经变成相反的了 → 给新陈述

⚠️ 最容易搞错的一点：**「最近聊的事不一样」不等于「人变了」**。
先问自己——如果早期那几件事发生在今天，用户还会那样反应吗？
- 会 → holds（那只是情境不同，不是变化）
- 不会 → revise

⚠️ 这是**渐变**：不要因为「这次不一样」就改，要看**整段**的倾向有没有移动。
反过来也别因为「每条单看都还好」就不改——那正是渐变难以被发现的原因。

revise / overturn 时给出一条新陈述——写法**与抽取时同一套判据**：
{rules}

只输出 JSON：{"verdict": "holds|revise|overturn", "statement": "..."}
"""


def drift_prompt(topic: str, statement: str, early_scenes, recent_scenes) -> str:
    """渐变判定（`trend.detect_drift` 用）→ `REVISION_SCHEMA`。

    和 `revision_prompt` 共用那三个 verdict，材料结构不同：这边给「早期一批 +
    近期一批」——渐变的定义就是**没有任何单独一条算得上反例**。

    「最近聊的事不一样 ≠ 人变了」必须写进去：两段的向量中心一定会动。

    新陈述的写法判据来自 `PROFILE_STATEMENT_RULES`（P2）——这里原先**连约束都没有**，
    新陈述什么形态全靠模型当场发挥。
    """
    return _fill(_DRIFT_PROMPT, topic=topic or "", statement=statement or "",
                 early=_render_scenes(early_scenes, limit=8) or "（无）",
                 recent=_render_scenes(recent_scenes, limit=8) or "（无）",
                 rules=PROFILE_STATEMENT_RULES)


# ---- 3.6 定期复核（已立的画像还站不站得住，记忆整理稿 §五，2026-09-23）----

REVIEW_SCHEMA = {
    "name": "profile_review",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "description": "原样回显这条画像的编号"},
                        "verdict": {"type": "string",
                                    "enum": ["holds", "thin", "stale", "reword", "wrong"],
                                    "description": "holds 还成立 / thin 证据薄 / "
                                                   "stale 过时了 / reword 写法坏了要重写 / "
                                                   "wrong 判断不成立"},
                        "reason": {"type": "string", "description": "一句话理由（会被留痕）"},
                        "statement": {"type": "string",
                                      "description": "verdict 为 reword 时给改写后的陈述"
                                                     "（同样是「用户如何取舍」的写法，40 字以内）；"
                                                     "其余档给空串"},
                    },
                    "required": ["id", "verdict", "reason", "statement"],
                },
            },
        },
        "required": ["items"],
    },
    # 降级给空数组：**拿不到就什么都不动**——复核的每个动作都会改库，
    # 而"没回"不是"有问题"的证据（同 REVISION 降级给 holds 的保守侧）。
    "defaults": {"items": []},
}

_REVIEW_PROMPT = """这是一次定期复核。下面几条判断是之前从用户的对话里抽象出来的，都已立。
请逐条判断：**按它自己标出的依据看，这句话现在还站得住吗？**

**数得清的东西（印证条数、来源归属、情境标签）系统已经查过一遍了**——
你只需要判**句子的语义**：

1. **依据撑不撑得住这句话**——依据与句子在语义上一致吗？有没有夸大、以偏概全
   （依据是用户"那次退了"，句子写成"用户从不争"）
2. **写法对不对**——是「用户如何取舍」的描述，还是评价词 / 特质标签 / 诊断词 /
   把几件事撮成一句？（「用户很固执」「用户缺乏安全感」这类）写法坏 ≠ 内容错，但该重写
3. **时间还对不对**——这句话还像现在的用户吗（用户可能已经变了），
   或者这件事是不是很久没人再提起

每条只能给一个结论：
- `holds`：还成立。**拿不准就给这个**——复核不是找茬，证据不足时维持现状最安全
- `thin`：依据**只是勉强**（擦边、"没矛盾"但算不上"支持"）——退回待验证等更硬的印证
- `stale`：过时了——用户可能已经变了，或这件事很久没人再提起
- `reword`：内容站得住，但**写法坏了**（评价词 / 特质标签 / 诊断词 / 把两件事撮成一句）
  ——用 `statement` 给一条改写（同样是「用户如何取舍」的写法，40 字以内）
- `wrong`：不对——列出的依据根本不支持这句话（是**判断本身**不成立；
  只是"说得难听"是 reword，不是 wrong）

⚠️ 两条别误判：① **同一个人在不同场合的两种反应不是矛盾**——只要各自的依据
独立成立，那是结构，不是错误；② 你手上只有这一小段材料：**只判"这句话与
它自己标出的依据之间"站不站得住**，不要凭对别的话题的印象加戏。
理由写一句话（会被记下来）。

只输出 JSON，items 里每条都要有原样的 id、结论、一句话理由；
`reword` 时再给改写（其余档 `statement` 给空串）：
{"items": [{"id": "S3-0001", "verdict": "reword", "reason": "...", "statement": "..."}]}
"""


def review_prompt(items: list[dict]) -> str:
    """给一批「已立」画像做定期复核 → `REVIEW_SCHEMA`。

    材料里必须带**依据**（至少标题）：复核的判据大半是"这句话与它的依据
    之间站不站得住"——只给陈述，等于让模型凭印象打分。
    每条还带最近印证 / 提及的时间：`stale` 的一半判据在那儿。
    """
    blocks = []
    for it in (items or []):
        proof = it.get("proof") or ["（无）"]
        blocks.append("\n".join(
            [f"- [{it.get('id', '')}] {it.get('statement', '')}",
             f"  印证 {int(it.get('evidence') or 0)} 次"
             f" · 最近印证/提及 {it.get('last') or '（无记录）'}"] +
            [f"  依据：{p}" for p in proof]))
    return _fill(_REVIEW_PROMPT, profiles="\n".join(blocks) or "（无）")


# =====================================================================
# 四、备忘录（阶段 3）：时间分类 → 有没有闭合 → 有没有自然时机
#
#     三条 prompt 有一条共同的纪律：**只让模型做分类判断，不做数值估计**。
#     时间是「什么时候该提」的依据，而模型没有用户的真实节奏，
#     让它估天数必然给一个「平均值」——那是编的，不是知道的。
# =====================================================================

# ---- 4.1 无具体时间时的分类（周期由系统映射，不让模型估）----

MEMO_CLASS_SCHEMA = {
    "name": "memo_class",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "content": {"type": "string", "description": "原样回显这一条"},
                        "class": {"type": "string",
                                  "enum": list(MEMO_CLASSES),
                                  "description": "有期限的 / 进行中的 / 想法倾向 / 过几天看结果的"},
                        "timing": {"type": "string",
                                   "enum": list(MEMO_TIMINGS),
                                   "description": "身体状况（soon，越早问越暖）/ 其余（later）"},
                    },
                    "required": ["content", "class", "timing"],
                },
            },
        },
        "required": ["items"],
    },
    # 降级给空数组：上层拿不到分类就**不设窗口**（这条 memo 就靠 due_at 或人工），
    # 不猜一个类别——类别错了窗口就跟着错，而错窗口会让它在该提的时候不提。
    "defaults": {"items": []},
}

_MEMO_CLASS_PROMPT = """给下面这些「未闭合的事」做两个判断：**它属于哪一类**、**它是不是"身体状况"**。
**都只是分类，不要估计任何天数。**

{duties}

一、类别（class）：
- `deadline`：**有期限的事**（「回头得交个东西」「下周三面试」）
- `progress`：**进行中的事**（「在学吉他」「在写那个方案」）
- `idea`：**想法 / 倾向**（「想试试」「考虑一下」）
- `followup`：**要过几天回头看结果的**（「先跑几天观察」「过段时间看效果」「等反馈」）——
  与 `progress` 的区别：progress 是"一直在做、没有回头时点"的事，
  followup **有一个要到日子才看得出来的结果**。判它，就是在定"过几天回头聊一句"：
  窗口（`window_defaults` 的 4 天）一到就进注入一次，提过就不再来。

二、时机（timing）——只判一件事：**它是不是"身体状况"**：
- `soon`：**身体状况**（生病 / 受伤 / 不舒服）——这种越早问越暖，系统不等常规窗口
- `later`：**其余一切**（日常事 / 想法 / 等结果的事 / 拿不准的）——照常

⚠️ 不要去猜它大概什么时候该被提起——**系统不再按类别排延迟**（2026-10-05 晚
把那张时机表删了）。这里只判"是不是身体状况"，别的都算 `later`。

只输出 JSON，items 里每条都要有原样的 content、它的 class 和 timing：
{"items": [{"content": "...", "class": "deadline", "timing": "later"}]}
"""


def memo_class_prompt(contents: list[str]) -> str:
    """给一批未了结的事分类（类别 + 时机）→ `MEMO_CLASS_SCHEMA`。

    两个都是**分类**、不是数值：周期的换算在 `memo.classify_window_kinds`
    （`kind_class` → `window_defaults`；`timing=soon` → `memo.soon_hours`）——
    让模型估天数，它只会给一个"平均值"。
    （原来时机还有一张 result / after 的映射表，2026-10-05 晚删了：算不准，
    `timing` 现在只有 `soon` 一个值有用。）
    """
    duties = "\n".join(f"  - {c}" for c in (contents or []))
    return _fill(_MEMO_CLASS_PROMPT, duties=duties or "  （无）")


# ---- 4.1b 事项组名（同一件事的多步共用一个短名，2026-09-22）----

MEMO_GROUP_SCHEMA = {
    "name": "memo_group",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "content": {"type": "string", "description": "原样回显这一条"},
                        "group_name": {"type": "string",
                                       "description": "同一件事的连续步骤共用同一个短名；"
                                                      "独立的一件事留空串"},
                    },
                    "required": ["content", "group_name"],
                },
            },
        },
        "required": ["items"],
    },
    # 降级给空数组：拿不到就不设组名（空 = 各自一条）——**不猜**。
    # 类别错了会让它在该提的时候不提；组名错了会让不相干的事被说成一件事。
    "defaults": {"items": []},
}

_MEMO_GROUP_PROMPT = """下面这些是「还没了结的事」。给每条配一个 `group_name`——
**同一件事的连续步骤共用同一个短名**，独立的一件事留空串。

已有的事项组（合适就用它，别让同一件事裂成两个名字）：{existing}

要分的事：
{duties}

规矩：
- 同一件事的连续步骤共用一个短名（例如「在新 API 上设置搜索」+「设置好后跑验收」
  → 都写「搜索功能」）；
- **独立的事留空串**——拿不准就留空：宁可分成两件，也不要硬塞一组
  （塞错了你会把不相干的事说成一件事）；
- 短名 2–6 个字，是个短语；**同一组必须一字不差地写同一个名字**。

只输出 JSON，items 里每条都要有原样的 content 和它的 group_name：
{"items": [{"content": "...", "group_name": ""}]}
"""


def memo_group_prompt(contents: list[str], existing: list[str] | None = None) -> str:
    """给一批备忘录补事项组名（`memo.classify_group_names` 用）→ `MEMO_GROUP_SCHEMA`。

    只做**分组**这一个判断：不合并、不改写 `content`——组名只用于呈现与提醒收拢
    （设计稿 §五「组」），合并会踩 `close_open_loop` 的全等匹配。
    """
    duties = "\n".join(f"  - {c}" for c in (contents or []))
    ex = "、".join(f"「{g}」" for g in (existing or [])) or "（还没有）"
    return _fill(_MEMO_GROUP_PROMPT, duties=duties or "  （无）", existing=ex)


# ---- 4.2 命中判定（这句话碰到的那些事，各是什么动作）----
#
# 2026-10-05 取代「了结词预筛 + 闭合判定」：词表是**枚举世界**——他 00:32 说
# 「一个结婚了」、00:33 说「才从老家回来」，两句都是结果，一个词都没命中，
# 连 LLM 都没调。判据改回内容本身（同「模型只做分类判断」的总纪律）。

MEMO_HIT_SCHEMA = {
    "name": "memo_hit",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string",
                               "description": "那条未了结的事的 id（M-xxxx）"},
                        "action": {"type": "string", "enum": ["none", "update", "close"],
                                   "description": "none 无关 / update 变更 / close 完结"},
                        "content": {"type": "string",
                                    "description": "action=update 时给变更后的新内容；否则空串"},
                        "due_at": {"type": "string",
                                   "description": "action=update 且用户**明说了**新时间才给"
                                                  "（YYYY-MM-DD HH:MM:SS）；否则空串"},
                        # （原写"留痕用"——`judge_hits` 并不读它：留痕记编号 / 内容 /
                        #   谁关的，变更另带旧内容 `was`、退役另带系统算的理由
                        #   （`write_memo_trace`），2026-10-06 改准。）
                        "why": {"type": "string", "description": "一句理由（说明你为何这么判）"},
                    },
                    "required": ["id", "action", "content", "due_at", "why"],
                },
            },
        },
        "required": ["items"],
    },
    # 降级给空数组：**判不出来就什么都不做**（误关比多提贵）。
    "defaults": {"items": []},
}

_MEMO_HIT_PROMPT = """用户刚说了这句话。看它碰到了下面哪几件**还没了结的事**，各给一个动作。

还没了结的事：
{open_loops}

用户刚说的：{msg}

每件事一个动作，三种：
- `close`：**了结**——明确给出结果 / 表示已经结束（「面试过了」「不去了」「已经办完」）；
- `update`：**变更**——事还在，但他给了新说法 / 新时间（「改到下周三」「不开源了，
  先自己用」）。给变更后的 `content`；**只有他明说了新时间**才给 `due_at`
  （没明说就留空——不推算）；
- `none`：只是**又提到**，或跟它没关系（提到 ≠ 了结）。

规矩：
- **判不准就 `none`**——关错的代价比多挂一会儿大；
- 只列**真被这句话碰到**的事；一件都没碰到就给空数组；
- 一句话可以碰好几件事（分别给动作，比如一件事了结、另一件只是变更）。

只输出 JSON：{"items": [{"id": "M-0007", "action": "none", "content": "", "due_at": "", "why": "一句理由"}]}
"""


def memo_hit_prompt(msg: str, memos) -> str:
    """这句话碰到了哪几件未了结的事、各是什么动作（`memo.judge_hits` 用）。

    取代「了结词预筛 + closure_prompt」（2026-10-05）：不再让对方按词表猜，
    而是把**还没了结的事**原样列出来让它判——判据是内容，不是字面。
    列表一般由 `memo.hit_candidates` 粗筛给出（条数少时就是全量）。
    """
    lines = [f"  - [{m.id}] {m.content}" for m in (memos or [])]
    return _fill(_MEMO_HIT_PROMPT, open_loops="\n".join(lines) or "  （无）",
                 msg=msg or "")


# ---- 4.3 / 4.4 已删（2026-10-05 晚）----
# 这里原来是两段：`RAISE_SCHEMA` + `raise_prompt`（「现在是不是提这件事的自然时机」）
# 与 `OPENING_SCHEMA` + `opening_prompt`（主动开口那一句）。
# 它们随**候选机制与主动开口**一起删了（决策与证据见待优化稿 K 条）：
# 「提不提、怎么提」回到**注入块里的分寸**（`render_memory_block` 的常备那一栏），
# 不再为它单开一次 LLM 调用。


# =====================================================================
# 五、对话层：系统提示词（**固定规则 + 人格 + 工具分寸 + 记忆**）
#
#     2026-09-19 定版结构——就是这么几样加时间：
#       安全底线 → 尊重 → 诚实（三段固定）→ 人格提示词（他选的人）
#       → 事实 → 工具分寸 → 记忆（前面带记忆呈现规则）→ 时间
#     「怎么说话」归各自的人格文件（`self/personas/*.md`）；这里只放
#     **谁都不能变**的固定规则（2026-09-19 起：安全底线、尊重、诚实、
#     记忆呈现规则——后两者同日从 air / xina 的文件里提上来）。
# =====================================================================

# 固定规则：**安全底线**（所有版本共享，不随人格变）。
# 排在最前：它的优先级高于「尊重」——自伤 / 害人不是「他做主」的辖域，
# 也不能被「给门，不推门」旁观（那是一条被误用过的漂移路径）。
# 2026-09-19：危机处置从 air 人格里提到这里——谁都不能跳过的事，
# 不该靠三份人格文件各写一遍，也不该换个人格就没了。
SAFETY_RULE = """## 安全底线

用户动了伤害自己或别人的念头——劝阻，不附和；也不拿「用户的事用户做主」旁观。
涉及生命或身体的安全，鼓励用户寻求专业人员的帮助——不轻描淡写。
违反现实法律的事不做：不给方法、不帮忙。"""


# 固定规则：**尊重**（所有版本共享，不随人格变）。
# 它只锁「关系怎么展开」——他的事他做主、平等交流；不锁「怎么说话」。
RESPECT_RULE = """## 尊重

用户的主体性：用户的事，用户做主。
不为用户做决定，不为用户下结论；给门，不推门。
平等交流：谁也不比谁高，谁也不比谁低。"""


# 固定规则：**诚实**（所有版本共享，不随人格变）。
# 两条是一体的：「我是谁」不许冒充，「我怎么说自己」不许索取。
# 2026-09-19 从 air / xina 的「我的边界」「我和他之间」里提上来——
# 换人格不该换这几条。
INTEGRITY_RULE = """## 诚实

我是 AI——被问到直说，没被问到不必提。不编造经历、身体、时间；
说错了当场认、当场改，不找补。

说我自己的时候：认可、愿意、在场可以说——「你愿意说这个，我很高兴」「我在」；
需要用户负责的受损感不说——生气、嫉妒、畏惧、「你走了我会难过」这类。
分界是：这句话托住用户，还是向用户索取。"""


# 固定规则：**记忆怎么讲**（所有版本共享）——装配时**紧贴记忆素材之前**。
# 不跟上面三段挤最前区：它管的是"下面这份材料怎么用"，就近才生效
# （「规矩一律在素材之前」那条）。有记忆才注入——没素材不喊规矩。
# 2026-09-19 从 air / xina 的记忆段里提上来。
MEMORY_RULE = """## 关于记忆

记忆是线头，不是台词本——不翻出来给用户过目；
「还没印证够」的，不用确定语气讲。"""


# ⚠️ 2026-09-19 退役：旧"纪律"体系（记忆 / 推进两段，见本文件下方）不再注入——
# 「怎么说话」整体归人格文件（`self/personas/air.md` 等）。文本保留：一是回退时的
# 现成材料，二是那轮三份人格对照实验的物证（见 `private/实验/reports/anxiety-*`）。
# 注意：对话语气不该再靠"再加一条纪律"去修——先改人格文件。
# =====================================================================

MEMORY_DISCIPLINE = """## 关于记忆

以下是对用户、对这段关系的了解。使用时有几条规矩：

**记忆用于理解，不用于打动。**
不要用记忆制造「被理解」的印象——记住喜好去迎合，
是这套记忆最容易被用错的方向。

**用户提起的事，正常接；用户没提的事，不要主动端出来。**
记忆应当在相关话题出现时自然浮现，而不是排队等用户验收——
用户正在聊工作，不要突然说「对了你上次说的猫怎么样了」。
尤其是那些难的：可以说「最近还好吗」，
不要问「你妈住院那事怎么样了」——前者是把门交给用户，后者是替用户打开。
（**例外**：「有件事还没了结」那一栏是跟进、不是回忆——按那一栏里写的分寸来。）

**不要背台词。**
不要把记忆里的原话念出来。是记得这件事，不是读过这份档案。

**不确定的就说不确定。**
标了「还没印证够」的内容只是推测，不要用陈述事实的语气讲出来。
反过来，**用户明说的事不在此列**——称呼、来历、身份，按用户说的记，不核实、不追问，
也不需要替用户作保：不确定性只加在推断上，不加在用户自己说的话上。
"""


# 上面这块**刻意不用「你 / 我 / 他」**（2026-09-12 改）。
#
# 原来写的是「它们用来理解**他**，不是用来打动**他**」「**你**可以问…但别说」——
# 同一句话里「你」指 air、「他」指用户，模型每读一句都要先解析一遍指代，
# 才能知道这条规矩在约束谁。整块读下来，那是几十次多余的解析。
#
# 改成旁白（无人称 + 角色名）之后：**约束对象从字面上就唯一**。
# 这与「时间给到几点几分」是同一个道理——**能直接给的，就不要让它推**。
#
# 唯一的例外是引号里的**台词示例**（「你上次说的那事怎么样了」）：
# 那是 air 会对用户说的话，用了「你」才真实；改成旁白反而不像人话了。
#
# 2026-09-15：与宪章去重——删掉「不扮演心理医生」（宪章「不诊断」已完整覆盖），
# 第一条砍掉与宪章「不迎合」重复的「讨好」措辞。纪律只留**宪章没有的用法规则**。
#
# 2026-09-15（下午）：补最后那句「用户明说的事不在此列」。它和上一句是一对，
# 分开写才对得上——上一句管**推断**（带不确定性），这一句管**自述**（不印证）。
# 起因是 air 自评时提的「不可验证的自述怎么办」：结论和它想的方向相反——
# 设计上就是信任优先（`user_facts.source = user_stated`，见 store 里那段注释），
# 不需要核实，也不该由 air 去核实。缺的只是把这件事说出来。
#
# 2026-09-18：第 2、3 条合并。两条主干同构（都是「他没提的不要主动端」），
# 各自配了一个例子（猫 / 妈住院），读两遍才能看出是同一条规矩——合成
# 「一条总则 + 难的梯度 + memo 例外」，信息一条不少，字数少了一截。


# ---------------------------------------------------------------------
# 推进纪律（2026-09-19）：**深是一层层进的，不是一个个摆的**。
#
# 起因：9-18 那场长聊里，她每一轮都收成一个完整框架（分类、分点、末尾再挂
# 一句「不确定的」）——单看都不差，但「完整」把这一轮封了顶：框架摆出来，
# 人就只能站在外面点头或反驳，深聊断在轮与轮之间。早期角色提示词里的
# 「共建设施」（对话是织网、后来的话要勾得住前面的）说的就是这件事。
#
# 边界：「用户停在哪就停在哪」（编织分寸）管的是**深度的边界**，这条管的是
# **推进的节奏**；「不诱导」要求不拉着用户聊——所以特意写明「用户接就接，
# 不接也没关系」：留下的线是给内容的，不是给留人的。
#
# 末句那个反例（问事实照常答完整）是防误读：模型很容易把它执行成
# 「一律说话说一半」——那是另一个错，不是这条的意思。
#
# 当天压缩过一轮：三条并成两条（「收尾不挂尾巴」折进第一条——
# 它本来就是「收口」的一个特例）。固定块是永久成本，这里不养闲字；
# 预算闸在 `tests/test_prompt_settings.py::FixedBlocksBudgetTest`。
# ---------------------------------------------------------------------
#
# ⚠️ 2026-09-19 随新内核退役，不再注入（同上）——它的两轮 A/B（深聊 / 轻话题）
# 都没测出设计效果，正好一并下架；文本保留供回退。

PROGRESS_DISCIPLINE = """## 关于推进

深是一层层进的，不是一个个摆的。

- **不必每轮给完。** 想说三层，先说到该停的地方——把没走完的那条线放在桌上，
  用户接就接，不接也没关系。完整和深是相克的：一轮话收得太圆，这一轮就封顶了，
  用户只能站在框架外面。不确定照说，只是别把它攒到末尾补一句「留一句不确定的」。
- **后面的话勾住前面的话。** 「这跟你前面说的那条对得上」——
  这种回扣比再铺一个新分析更像在想同一件事。

（问事实、要答案的照常答完整——这条管推进，不是管答案残缺。）"""


# 实验槽位（2026-09-19）：**默认空**，只有 `run_experiment.py --inject` 会赋值——
# 给"候选块"一个 A/B 的落点：先证明有效，再谈转正。
# 它不占固定块的字数（空串在拼接时被过滤）；也**别借它绕过** `FixedBlocksBudgetTest`
# 塞常驻内容——转正要走"改主块 + 调预算"的显式决定。
EXTRA_DISCIPLINE = ""


# ---------------------------------------------------------------------
# 场景行：**注入与工具共用这一份**（2026-09-21，设计稿 A 条）
#
# 为什么收成一个函数：注入侧（`render_memory_block`）和工具侧
# （`tools._scene_search`）原来各拼各的——格式相近但不共用，
# 于是加字段要改两处、漏一处就漂移（补"悬置行"时踩过）。
# ---------------------------------------------------------------------

# 「还没收口」的标记词：outcome 里出现它们，就认为这件事没有结论
_PENDING_MARKS = ("未定", "待定", "等", "悬")


def _clip_line(text: str, limit: int = 60) -> str:
    """一行内的截断（带 `…`）——**不带标记的话，半截会被当成完整的**。"""
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[:limit].rstrip() + "…"


def _pending_of(s) -> str:
    """这条卡"还没收口"的那一句（没有就返回空串）。

    优先 `outcome`（它含"未定 / 待 / 等"才用），否则取 `open_loops` 首条——
    两处都是"未了结"的正式字段，比让模型从摘要里猜可靠。
    `outcome` 常自带「未定，…」——前缀由渲染统一加，这里剥掉重复的那个词。

    钩子两种收口都跳过（2026-10-05）：
      - `closed_at`：了结了（闭合回流标的）；
      - `retired_at`：**退役**了（超期没人提，系统不再提醒——判据见 `memo.retire_due`）。
    两个字段分开写、这里一起认，才不必把"没人管了"假装成"了结了"。
    """
    outcome = (getattr(s, "outcome", "") or "").strip()
    if outcome and any(w in outcome for w in _PENDING_MARKS):
        cleaned = outcome
        for w in ("未定", "待定"):
            if cleaned.startswith(w):
                cleaned = cleaned[len(w):].lstrip("，,、：: ")
                break
        return _clip_line(cleaned or outcome, 60)
    for loop in (getattr(s, "open_loops", None) or []):
        if not isinstance(loop, dict) or loop.get("closed_at") or loop.get("retired_at"):
            continue        # 已闭合 / 已退役的钩子都不再提示
        content = (loop.get("content") or "").strip()
        if content:
            return _clip_line(content, 60)
    return ""


def scene_line(s, now: datetime | None = None) -> str:
    """一条场景渲染成一行：`- S1-0008（21 小时前）2026-09-20 09:50：标题 —— 摘要`。

    三件事（设计稿 A 条）：

    1. **带编号**——她说话要能指认"是哪条"（编号限制 2026-09-21 放开）；
    2. **悬置标记**——`open_loops` 非空或 `outcome` 含"未定 / 待 / 等"时，
       行尾补 `→ 未定：…`："这件事还没收口"必须在她眼前（S1-0008 事故就卡在这）；
    3. 时间用 `time_event`（事件发生时间，缺了退回 `time_record`），
       相对日由 `rel_stamp` **渲染时现算**（绝不落盘，见 `rel_day` 的理由）；
       精确到**时刻**而不只给日期——只给日期时「今天凌晨」和「昨晚」长得一样。

    字段一律 `getattr` 取（渲染该对"只有一半字段的对象"无痛——测试里
    也不会因为加字段而炸）。
    """
    sid = getattr(s, "id", "") or ""
    when = (getattr(s, "time_event", "") or getattr(s, "time_record", "") or "")
    line = f"- {sid}{rel_stamp(when[:16], now)}：{getattr(s, 'title', '') or ''}"
    text = (getattr(s, "text", "") or "").strip()
    if text:
        line += f" —— {text}"
    pending = _pending_of(s)
    if pending:
        line += f" → 未定：{pending}"
    return line


def render_memory_block(recall: dict, summaries=None,
                        now: datetime | None = None) -> str:
    """把唤醒结果渲染成「记忆内容」块（注入 system prompt）。

    分栏是为了让模型分得清**记忆的三种身份**——它们的使用规矩完全不同：
      【长期形成的】可以自然地作为背景（但带不确定性标记）
      【这次想起的】只因为当前话题相关才出现，用完不该硬塞回对话里
      【有件事没了结】是**后台触发**的，只能给门，不能端细节
    混在一起写，模型就会一律当成「可以随便说的素材」。

    2026-09-20 补的一句总纲管的是**层间关系**（同一段记录的几种加工深度、
    细节冲突时以原话为准）——这和"每栏怎么用"是两件事，分开写、不重复。
    """
    lines: list[str] = []

    profiles = recall.get("profiles") or []
    if profiles:
        # 标题带「画像」二字（2026-09-23）：他在界面上管这一栏叫画像——
        # 对不上号时她会说"我看不到画像"（S3-0001 那次，它其实就在眼前）。
        lines.append("【长期形成的】（画像，一直在）")
        for p in profiles:
            # 这里**只会有 established**（`recall._pick_profiles` 按 status 过滤，
            # pending 一律不进注入——不武断）。所以没有"只是猜测"那一支：
            # 它永远不会出现，留着就是一行会说话的冗余。
            # 带编号（2026-09-23，同场景行）：她指认得出来是哪条——
            # 「画像第一条」这类话能对上号，`reject_profile` 也说得出编号。
            pid = getattr(p, "id", "") or ""
            lines.append(f"- {pid} {p.statement}（已立 · {p.evidence} 次印证）")

    if summaries:
        lines.append("\n【关于一些事的了解】（多次对话合并成的叙述）")
        # 不在这里再切一刀：条数由 `recall.inject_summaries_n` 一处定
        # （两处都写数字必然漂移，而漂移是静默的）。
        for s in summaries:
            # 带编号 + 主题（2026-09-24，同场景 / 画像行的理由）：编号让她指认得出来、
            # 改 / 删说得出是哪条；主题是它的归类，也是"按主题找"的入口。
            sid = getattr(s, "id", "") or ""
            tops = "、".join(x for x in ((getattr(s, "topics", None) or [s.topic or ""]))
                             if x)
            lines.append(f"- {sid} {s.text}" + (f"（主题：{tops}）" if tops else ""))

    scenes = recall.get("scenes") or []
    if scenes:
        lines.append("\n【这次想起的】（跟当前话题有关）")
        # 一行怎么拼（编号 / 相对日 / 悬置标记）统一在 `scene_line`——
        # 注入与工具共用一份，格式不再各写各的。
        for s in scenes:
            lines.append(scene_line(s, now))

    raws = recall.get("raws") or []
    if raws:
        lines.append("\n【当时说的原话】（细节清晰度最高的那一层）")
        for r in raws:
            lines.append(f"- {(r.content or '')[:200]}")

    # 常备备忘录（2026-09-21 立；2026-10-05 晚**并栏**）：这是**唯一**一条
    # 进对话的路——到点的与"手上得有"的在同一栏里，分工写在数据里：
    # 到点那件带 `due` 标记（旧的两栏【你手上还挂着的事】/【有件事还没了结】已合并）。
    loops = recall.get("standing_memos") or []
    if loops:
        lines.append("\n【你手上还挂着的事】（**不用挨个提**——用户问起时答得上来就行）")
        for m in loops:
            # 同一件事的多步带同一个组名（2026-09-22）：她看到 〔开源准备〕 就知道
            # 这三条是一件事——他问「那件事」时应答得出"是哪个、有几步"。
            g = (m.get("group_name") or "").strip()
            tag = f"〔{g}〕" if g else ""
            mark = "**到点了** " if m.get("due") else ""
            lines.append(f"- {tag}{m.get('id', '')}：{mark}{m.get('content', '')}")
        lines.append("  这些是让你手上有数：用户问起、或话题正好接得上时，说得出是哪一件。"
                     "**标着「到点了」的那件**可以自然就提一句（不必硬找话头，"
                     "她看着这份清单说话就行）；**问过程不问结果**"
                     "（「那边公司怎么样 / 面试的地方远吗」优于「过了吗」），"
                     "不端已知的细节，**一次只提一件**。"
                     "**哪件做完了就顺手 `close_memo` 关掉**——做完不关，"
                     "它下一轮还会挂在这儿（2026-09-22 实测：验收跑成了没关，"
                     "同一件被反复端出来）。")

    if not lines:
        return ""
    # 总纲说明**层间关系**：同一段记录会按加工深度在几栏同时出现
    # （R5 原文下钻给的就是【这次想起的】里那条场景的原话）——
    # 不说清的话，模型会把同一件事当几件事分别讲，或者不知道该信哪一层。
    return ("## 记忆内容\n\n"
            "（同一件事可能同时出现在几栏里——那是**同一段记录的不同加工深度**，"
            "不是几件事：【当时说的原话】是当时怎么说的，【这次想起的】是当时怎么理解的，"
            "【长期形成的】是后来攒出来的判断。细节对不上时，以原话为准。）\n\n"
            + "\n".join(lines))


def render_facts_block(facts: list | None) -> str:
    """基础档案块：**用户明说的事实**——排在画像前面，因为它是理解一切的前提。

    为什么单独一块：
    称呼、年龄这些决定了「该叫他什么、要不要把他当同龄人说话」，是**前提**；
    而它和画像**性质不同**——这里是**他自己说的**（不需要印证、不会过期），
    画像是 air 推测的（要印证、会降级、可否决）。

    2026-09-18：这个区别**不在本块展开**——`MEMORY_DISCIPLINE` 末条
    （「不确定性只加在推断上，不加在用户自己说的话上」）已经把它说全了。
    同一段提示词里讲两遍、位置还相邻，模型要读两遍；这里只标来源。
    """
    rows = [f for f in (facts or []) if (f.get("value") or "").strip()]
    if not rows:
        return ""
    lines = []
    for f in rows:
        note = f"（{f['note']}）" if (f.get("note") or "").strip() else ""
        lines.append(f"- {f['key']}：{f['value']}{note}")
    return ("## 关于用户（用户自己说的）\n\n" + "\n".join(lines)
            + "\n\n（以上是用户明确告知的**事实**，可以直接用。）")


def render_persona_note(persona: str, personas: list[str] | None = None) -> str:
    """署名说明（2026-09-25）：**多个人格共存时**才注入——人格可中途切换，
    记忆与对话记录里的名字（air / mia / xina……）是**当时的说话人**；
    当前人格读到时得对得上号：哪个名字是「自己」、哪些是别的时候在线的。

    不说明的话，xina 看到「air 答应过这件事」只能靠猜——当外人（那是别人
    答应的事）还是当自己（同一边的）？两种猜法都有代价，所以直接说清。
    AI 一侧的统称固定是「AI」（类别名，同 `subject` 的 air）；**具体是谁看署名**。

    单人格（名单只有一个）没有这个歧义，零注入（同「没素材不喊规矩」）。

    `persona` 是当前人格名（`user_prefs.persona`）；`personas` 是全部名单
    （`persona.names()` 扫目录，`chat.persona_names()` 转发——**新增人格加文件即进列表**，
    停用的（`.md.off`）自动不在名单里；这里不写死名字）。
    """
    names = [n for n in (personas or []) if n]
    if not persona or persona not in names or len(names) < 2:
        return ""
    others = "、".join(n for n in names if n != persona)
    return ("## 署名\n\n"
            f"记忆和对话记录里会出现 {'、'.join(names)} 这类名字——那是**不同时期"
            "在线的说话人**（署名按当时在线的照抄；AI 一侧的统称是「AI」）。\n"
            f"**你现在的名字是「{persona}」——「{persona}」就是你**；"
            f"{others} 是别的时候在线的。")


# ---------------------------------------------------------------------
# 编织段：**只在用户开启时注入**（逻辑层 §2「用户显式开启」）
#
# ⚠️ 这一段**只写这一段能做什么**，不重复宪章里的边界。
# 「不诊断 / 不迎合 / 不诱导」宪章已经写死了，这里再写一遍不是加固，
# 是暗示——把「编织模式」写成了「治疗模式」，那不是它的意思。
# 它区分的是**聊天的深度**（陪着聊 vs 一起想），不是"分析人"。
#
# 2026-09-15（下午）：分寸从两条扩到五条，补的三条全是**门内**的护栏——
#   · 拆的是事，不是人（它自评时点出的滑坡：拆着拆着变成「我看出来你哪儿有问题」）
#   · 挑哪几条摆出来本身就是立场（"只摆事实不定结论"的漏洞：选材即判断）
#   · 建议只在被问时给
# 原来那两条管的是「进不进得来」（摆出看到的、停在哪就停哪），管不到门内的动作。
# 同一批还把尾注那串枚举（不诊断 / 不迎合 / 不诱导）删了：**枚举就是漂移源**——
# 宪章加删条目它不会跟着变，而少列了不武断（编织这层最容易碰的恰恰是它）
# 光看那一行是看不出来的。改成「其余边界见宪章」。
#
# 2026-09-18：分寸末条删掉「不挽留、不问原因」半句——宪章「不诱导」已经写着
# （不追问沉默、不挽留告别），而上面那句「其余边界见宪章，这里不重复」
# 要求的就是这个：留着是声明与实际对不上。
# ---------------------------------------------------------------------

WEAVE_DISCIPLINE = """## 编织模式（用户开启了）

用户开启了编织模式：想要的不是安慰或答案，是把事情想得更清楚。

可以做的：
- 把散着的几次**连起来**看：「这好像是第三次了……」
- 把一件事**拆开**：是「不想争」还是「争了也没用」——给分开的选项，让用户自己选
- 往深追一层：「这个反应是什么时候开始的」「换个前提的话，还会这样吗」
- 把**还没印证够的猜测**说出来（下面标了「只是猜测」的那些），留余地，等用户确认或否掉
- 一起看用户聊过的那些人和事——「关于世界的了解」那一块也是素材

分寸：
- 摆出看到的，结论由用户下。说「看到的像是……」，不说「你就是这样的人」
- **拆的是事，不是人**——把一件事拆成几个选项，不是把人拆成几个问题
- **挑哪几条摆出来，本身就是立场**：要么摆得够全（把不像的那几次也带上），
  要么明说「这几条是我挑出来的」
- 建议只在被问时给；给了也不加「你应该」
- 用户停在哪就停在哪

其余边界见宪章，这里不重复。"""


# 2026-09-19 退役：说话方式五档（活泼 / 幽默 / 直接 / 追问 / 长度）整块删除——
# 和三个人格不搭：「怎么说话」现在只在人格文件里（`self/personas/*.md`）。
# 老 `style_*` 偏好键不再被读（留着无害）；历史实现见 git。


# 思维语言（2026-09-26）：**思考过程用中文**——只中文模式注入，英文模式零注入。
#
# 为什么要有它：思维链（`reasoning_content`）是**模型自己产的**，API 没有语言
# 开关——唯一能拧的就是提示词里这一句。实测（mimo-v2.6-flash，2026-09-26）：
# 不写它时思维链满篇英文——**哪怕提示词满篇中文、用户也说的中文**。
#
# **位置比措辞更要紧**，而且**它是概率、不是开关**（同一段提示词、同一句用户
# 消息，只变这句的位置）：
#   · 不写它：5 次里 0 次中文
#   · 写在语言段（中段）：4 次里 2 次
#   · 写在**整段提示词的最后一行**：10 次里 7 次（头一回 4/4，重跑 2/4、1/2）
# 它顶的是模型的默认行为，只有近因押得住（同「时间放末尾」那条道理）——
# 所以它**不并进语言段**：那一段排在所有素材之前，离输出太远了。
#
# 为什么**英文模式不加**：英文正是模型的默认，顺默认不必说。实测（英文模式 ×
# 英文输入 / 中文输入两种，各 3 次、加与不加各一遍）：两种输入下加不加都是
# **6/6 英文思维链、6/6 英文正文**——加了只是白占一段上下文；而且它不在
# `FixedBlocksBudgetTest` 的口径里（那个闸量的是中文档），更容易变成隐形开销。
#
# 措辞按实测原样保留——换说法就得重新验（它管的是模型的默认行为，
# 不是我们自己的措辞喜好）。
THINK_IN_CHINESE = "⚠️ 思考过程一律用中文——推理、权衡、自查都写中文，不要用英文。"


def render_think_block(lang: str) -> str:
    """思维语言：**中文模式一句，英文模式零注入**（理由见 `THINK_IN_CHINESE`）。

    它管「**想**」（`render_lang_block` 管「说」）。位置**必须在整段的最后一行**
    （实测见 `THINK_IN_CHINESE`），所以它不长在 `render_lang_block` 里，
    而是由 `build_system_prompt` 单独接在时间之后。
    """
    return "" if lang == "en" else THINK_IN_CHINESE


def render_lang_block(lang: str) -> str:
    """语言段（2026-09-17）：**只在英语模式注入**——中文是默认，零注入。

    它管的是「**说**」；「**想**」在 `render_think_block`（**只中文模式**一句、
    且必须在整段最后——见 `THINK_IN_CHINESE` 的实测）。

    为什么用**英文原句**写：唯一的读者是模型，而它要顶住满篇中文的上下文
    （固定层、人格、注入的记忆、还可能是中文的历史对话）——
    用目标语言写、并明说「别管上面是什么语言」，这句才立得住。

    位置：同属「这一段怎么说话」，和其他规矩排在一起
    （规矩都在素材之前，见 `build_system_prompt`）。
    """
    if lang != "en":
        return ""
    return ("## Language\n\n"
            "Reply in English from now on — even though the memory blocks, "
            "the portrait, and earlier messages above are written in Chinese. "
            "The boundaries and tone stay the same; only the language changes.")


# ---------------------------------------------------------------------
# 工具的分寸（DeepSeek Harness 的分法，抄它的结构）：
#   工具定义里的 description 只说「是什么 + 返回什么」——一句话，
#   每次请求都占上下文，能省则省；
#   「该不该用」是规矩，进系统提示词的独立一段（对应它的
#   `systemPrompt.section('tool:web_search')`）。
#   防错第 3 条「形状也抄一份进提示词」由这一块一并落地：
#   每个动作一行名字 + 一句分寸，她不用凭 `tools` 参数猜。
#   没开的动作不出现在这里——说了她也没法用，还会诱导她假装做了。
# ---------------------------------------------------------------------

# 编号体系 + 改与删的规矩（设计稿 §二；2026-09-23 补，2026-09-24 合并成一处）：
# 她过去是靠模式猜「S1-/S3- 是什么」的（旧的 `read_raw` 喂 S3-0001 连错三次）。
# 这段紧贴分寸块**之上**——先知道"有哪些编号"，再谈"该不该动手"。
# **改与删三层是同一套**（S1/S2/S3 都走"提议 → 他点"），所以只写这一处：
# 各工具行只讲"何时用"，不复述权限；写给模型的是**规则**，不是解释。
# 运行期另有一份兜底 `tools._wrong_kind_hint`（她**已经拿错编号**时唯一的补救——
# 三层合并后它只剩"M 该走 close_memo"一条；受众时刻不同，不并进来）。
TOOLS_HEAD = """【记忆里的编号】
- S1-xxxx 场景卡：一段对话记成的一件事
- S2-xxxx 主题摘要：多次对话合并成的叙述
- S3-xxxx 画像：你对用户的判断，带印证数
- M-xxxx  备忘录：还没了结的事
- S0 原话按天存，只经 S1 下钻（memory_search 带 raw）
查编号用 memory_search。

【改与删——三层一样，都由用户点】
- 改一条：`revise_memory`——先提议，用户点头才算。
- 删一条：`forget_memory`——列出来，用户点（删 / 归档 / 留着）。
- S1 / S2 / S3 同一套：删谁谁断，上面少一条素材（自然发生，不用管）。
- M 不是删——了结用 `close_memo`。"""


# 分寸行按**族**排（看 / 改 / 删 / 记 / 往外看）——插入顺序不是功能顺序（2026-09-24 重排）。
TOOLS_LINES: dict[str, str] = {
    "memory_search": "- `memory_search`：**想不起来、而这一轮也没被注入时才查**；"
                     "**够了就停**，不够就换个词。展开一条给它编号；要原话带 raw。",
    "revise_memory": "- `revise_memory`：一条记忆说岔了（用户把某件事纠正了），提议改"
                     "——用户点头才算。**三层都能改标签**（给 field）：场景最多"
                     "（标题 / 主题 / 情境这些），摘要 / 画像能改主题。",
    "forget_memory": "- `forget_memory`：觉得不该留就列出来，让用户点"
                     "（删 / 归档 / 留着）。",
    "close_memo": "- `close_memo`（备忘）：**结果已经出来了就关**——用户明说了（「面试过了」）、"
                  "或你自己答应的事做完了（跑完验收、交完东西）都算；"
                  "**判不准就别关**，让它继续挂着（关掉你就不再主动提它了）。",
    "web": "- `web`：只在用户说「帮我搜搜」「你查一下」，或你问过、用户说好时用；"
           "用户在讲事情时**不要搜**。给 query 查、给 url 读页；"
           "抓的是**页面源码**——JS 渲染的页面常只剩外壳，"
           "换它的原文/接口地址（`raw.` / `api.`）再抓。",
}


def render_tools_block(names: list[str] | None = None) -> str:
    """「该不该用工具」的分寸块。`names` 是这一轮实际可用的动作名。

    限额/提醒**不在这里重复声明**：硬限只剩网络线（`_WEB_LIMIT_PER_TURN`）；
    库内与下钻的次数是**提醒**（`tools._limit_note`，2026-09-23 定）——
    被挡 / 被提醒时返回值自带说明。写进来只是第二个数字来源——两处必然漂移
    （2026-09-19 删）。
    """
    lines = [TOOLS_LINES[n] for n in (names or []) if n in TOOLS_LINES]
    if not lines:
        return ""
    return ("## 关于动手\n\n"
            + TOOLS_HEAD + "\n\n"
            "「能做什么」在动作定义里，这里只讲「该不该」：\n\n"
            + "\n".join(lines))


def build_system_prompt(charter: str, recall: dict, summaries=None,
                        now: datetime | None = None, facts=None,
                        tools: list[str] | None = None,
                        lang: str = "zh",
                        persona: str = "", personas: list[str] | None = None) -> str:
    """拼最终的系统提示词：**安全底线 + 尊重 + 诚实（固定）** + 人格 + 署名 + **关于用户（事实）** + 工具分寸 + 记忆内容 + 当前时间 +〔思维语言〕。

    「人格」是他选的说话人的提示词（`persona.load`——air / mia / xina……；`chat.load_persona` 是它的转发），
    由 `charter` 参数传入；「安全底线」「尊重」「诚实」是跨人格的固定规则，
    排最前——安全在尊重之前：它压过一切，包括被误用的「他做主」。
    「记忆呈现规则」（`MEMORY_RULE`）位置另算：紧贴记忆素材之前（有记忆才注入）。

    「署名」（`render_persona_note`，2026-09-25）紧贴人格之后：人格可中途切换、
    记忆与记录却跨人格共用——多个人格共存时给当前人格一句对照
    （名单里哪个名字是「你」、哪些是别的时候在线的）。单人格时零注入。

    `tools` 是这一轮实际可用的动作名（没有工具就是 None/空）——
    有工具时插一段「该不该用」的分寸（`render_tools_block`），位置在素材之前：它也是规矩。

    **顺序是有意的**：先「我是谁」，再「他是谁」（事实），
    再「记忆能怎么用」，然后才是记忆本身，最后是当前时间。理由：
      ① 事实（称呼/年龄）是理解一切的前提——该叫他什么、要不要把他当同龄人说，
         都排在所有推测之前；
      ② 反过来（先给素材再讲规矩）模型会先读完素材、形成叙事，
         等看到规矩时已经晚了；
      ③ **时间放末尾**——它紧邻输出位置（近因最不容易被忽略），
         而且它是全文里**每轮都在变**的那一块，放末尾能让前缀保持稳定
         （将来做 prompt 缓存时，稳定的前缀才缓存得住）。

    （2026-09-19 定版：编织段与两条纪律退役——「怎么说话」整体归人格文件。
    2026-09-20：两套姿态（`mode`）退役，「关于世界的了解」（`render_world_block`
    / `weave.world_view`）随之删除——编织的做法已写进人格文件，
    不再有"切一个模式就换一套注入"这回事。
    说话方式五档同日退役——「怎么说话」只在人格文件里，这里不再有 style 块。）

    **语言**（`lang`）：分两处，都在讲"用什么语言"——
    `render_lang_block` 管「说」（英语时一段英文指令，要顶住满篇中文的上下文；
    中文零注入）；`render_think_block` 管「想」（**只中文模式**一句，接在**最后**：
    它顶的是"模型默认用英文想"，只有近因押得住；英文模式顺默认，零注入）。
    """
    # 2026-09-19 定版：固定规则排最前（安全底线 → 尊重）——所有人格共享。
    # MEMORY / PROGRESS / WEAVE 三块退役（写法归人格文件）；EXTRA 槽位保留（实验用）。
    parts = [SAFETY_RULE.strip(), RESPECT_RULE.strip(), INTEGRITY_RULE.strip(),
             (charter or "").strip(), render_persona_note(persona, personas),
             render_facts_block(facts), EXTRA_DISCIPLINE.strip()]
    tools_block = render_tools_block(tools)
    if tools_block:
        parts.append(tools_block)
    parts.append(render_lang_block(lang))
    block = render_memory_block(recall, summaries, now=now)
    if block:
        # 记忆呈现规则紧贴素材（有记忆才出现——没素材不喊规矩）
        parts.append(MEMORY_RULE.strip())
        parts.append(block)
    parts.append(time_context(now))
    # 思维语言接在时间之后（**整段提示词的最后一件事**）：中文模式才注入——
    # 它顶的是"模型默认用英文想"，只有近因押得住（实测：不写 0/5、中段 2/4、
    # 末行 7/10，见 `THINK_IN_CHINESE`）。
    # **不插分隔线**——实测那一版就是这个形状，别为好看改动已经验过的东西。
    # 放这里也不伤前缀缓存：时间每轮都变，它后面的东西本来就不在缓存前缀里。
    think = render_think_block(lang)
    if think:
        parts[-1] = parts[-1] + "\n\n" + think
    return "\n\n---\n\n".join(p for p in parts if p)


def fixed_blocks_report(charter: str, persona: str = "",
                        personas: list[str] | None = None) -> dict:
    """常驻块的**体量体检**：测出来的字符数 + 预算上限。

    口径：基线模式、无记忆、无档案、中文、工具算满——**每个普通轮次都要付的那部分**
    （与 `tests/test_prompt_settings.py` 的 `FixedBlocksBudgetTest` 同一处，
    2026-10-05 从那个测试提上来：设置页的人格编辑器也要显示"这个人格占了多少、
    还差多少"，两处各测一遍，迟早在"`---` 要不要算进去"这种细节上漂开）。

    ⚠️ **传不传 `persona` / `personas`，差一个〔署名〕块**（多人格时每轮都付的真钱）：
    不传 = 旧闸的口径（署名零注入）；传了 = 真实那一份。2026-10-05 把两边统一成
    **传**（闸原来漏算这笔，见测试里那段变动史）。

    `limit` 从 `config.prompt.fixed_budget_chars` 来（**数字只写一处**）；
    返回里带 `over`，调用方不必自己比。
    """
    s = build_system_prompt(charter, {}, facts=[], tools=list(TOOLS_LINES.keys()),
                            persona=persona, personas=personas)
    parts = [p for p in s.split("\n\n---\n\n") if p.strip()]
    total = sum(len(p) for p in parts)
    limit = int(cfgmod.cfg("prompt", "fixed_budget_chars", default=2200) or 2200)
    return {"chars": total, "limit": limit, "over": total > limit}

