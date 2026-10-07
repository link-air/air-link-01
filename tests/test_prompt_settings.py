"""第四轮修复的回归测试：提示词、时间注入、配置优先级、打码 key。

**每一条都对应一个真实踩过的坑**（不是凑覆盖率）：
  1. 时间太粗 → 模型自己推「下周三」是哪天，推错就白记
  2. 打码 key 被存回文件 → 之后每次请求都带着一串圆点，报 UnicodeEncodeError
  3. 求知版遗留的环境变量压过本版配置 → 界面改了没反应，看不出原因
  4. json_schema 降级后键名靠模型自觉 → 用错键名则整条备忘录静默消失
"""
# 用例分组：
#   脚手架  _LocalConfigTestCase（临时的 config.local.json）
#   提示词  TimeContextTest 时间粗细 · RelDayTest 相对日 · RequiredKeysTest 必填键 ·
#           PromptPlaceholderTest 占位符只替换一遍 · OpenLoopsAliasTest 认别名 ·
#           PleasantryTest 客套话不是承诺 · FixedBlocksBudgetTest 常驻块预算
#   配置    MaskedKeyTest 打码值 · LocalConfigMergeTest 合并 · EmbeddingSslTest ·
#           SearchToggleTest 搜索开关 · PriorityTest 四层优先级 ·
#           DashboardBindingTest 改完要生效
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config as cfgmod
from core import settings
from core import search as searchmod
from core.chat import load_persona, persona_names
from core.embedding import EmbeddingService
from core.prompts import (PROFILE_STATEMENT_RULES, SCENE_SCHEMA, TOOLS_LINES,
                          build_system_prompt, drift_prompt, extract_scene_prompt,
                          fixed_blocks_report, profile_prompt, rel_day, rel_stamp,
                          render_memory_block, render_persona_note,
                          required_keys, revision_prompt, time_context)
from core.scene import _norm_open_loops


class TimeContextTest(unittest.TestCase):
    def test_full_format(self):
        """时间要给全：年月日 + 星期 + 时段 + 时分——少一项模型就得自己推。

        「凌晨两点」和「下午两点」对语气的影响完全不同，这个换算不该交给模型。
        """
        s = time_context(datetime(2026, 9, 12, 2, 15))
        for part in ("2026年", "9月", "12日", "周六", "凌晨", "02:15"):
            self.assertIn(part, s, f"缺少 {part}")
        self.assertIn("晚上", time_context(datetime(2026, 9, 12, 21, 47)))
        self.assertIn("深夜", time_context(datetime(2026, 9, 12, 23, 50)))
        self.assertIn("上午", time_context(datetime(2026, 9, 12, 9, 5)))

    def test_injected_near_end(self):
        """时间注入在**末尾**（紧邻输出位置），不是开头。"""
        p = extract_scene_prompt("用户: 测试", ["用户·X"])
        self.assertIn("当前时间", p[-400:], "时间应当在提示词末尾附近")
        self.assertNotIn("{time_context}", p, "占位符没被替换")


class RelDayTest(unittest.TestCase):
    """相对日（一周内精算到天，之外给锚点）——由代码算，不让模型数「差几天」。

    起因：模型把 17 分钟前说成「昨晚」（见 `ShortTerm._dated_digest` 的注释）。
    往时间戳前挂一个算好的相对日，把这一步从它手里拿走——
    同 `time_context` 的道理：**能直接给的，就不要让它推**。

    2026-09-20（第三版）：一周内到天；之外「一周前 / 几周前 / 一月前 /
    几个月前 / 一年前 / 两年前」——越远的记忆越只给量级。
    """

    NOW = datetime(2026, 9, 20, 21, 0)

    def test_within_a_week_counts_days(self):
        """一周内精算到天：中文数字（量词前是「两」不是「二」）+ 今天 + 带时刻输入。"""
        self.assertEqual(rel_day("2026-09-20", self.NOW), "今天")
        self.assertEqual(rel_day("2026-09-19 08:30:00", self.NOW), "一天前",
                         "带时刻、但已满 24 小时：回天级")
        for days, want in ((1, "一天前"), (2, "两天前"), (3, "三天前"), (4, "四天前"),
                           (5, "五天前"), (6, "六天前"), (7, "七天前")):
            d = (self.NOW - timedelta(days=days)).strftime("%Y-%m-%d")
            self.assertEqual(rel_day(d, self.NOW), want, f"{days} 天前")

    def test_within_a_day_counts_minutes_and_hours(self):
        """一天内到分钟 / 小时——误判率最高的区间（时钟减法归代码做）。"""
        self.assertEqual(rel_day("2026-09-20 20:59:35", self.NOW), "刚刚")
        self.assertEqual(rel_day("2026-09-20 20:45:10", self.NOW), "14 分钟前")
        self.assertEqual(rel_day("2026-09-20 18:30:00", self.NOW), "2 小时前")
        self.assertEqual(rel_day("2026-09-20 06:24:00", self.NOW), "14 小时前")
        self.assertEqual(rel_day("2026-09-19 22:00:00", self.NOW), "23 小时前",
                         "跨天但不满 24 小时：仍是小时级（「昨晚」不会被说成「一天前」）")
        self.assertEqual(rel_day("2026-09-19 08:30:00", self.NOW), "一天前",
                         "满 24 小时才接天级")

    def test_beyond_a_week_gives_anchors(self):
        """锚点档的边界——数字都是自然断点（两周 / 一月 / 一年）。"""
        for days, want in ((8, "一周前"), (13, "一周前"),
                           (14, "几周前"), (29, "几周前"),
                           (30, "一月前"), (59, "一月前"),
                           (60, "几个月前"), (364, "几个月前"),
                           (365, "一年前"), (729, "一年前"),
                           (730, "两年前"), (3000, "两年前")):
            d = (self.NOW - timedelta(days=days)).strftime("%Y-%m-%d")
            self.assertEqual(rel_day(d, self.NOW), want, f"{days} 天前")

    def test_future_and_dirty_are_blank(self):
        """未来时间、脏输入：不硬算、不崩、返回空串。"""
        self.assertEqual(rel_day("2026-09-25", self.NOW), "", "未来时间不硬算")
        self.assertEqual(rel_day("", self.NOW), "")
        self.assertEqual(rel_day("不是什么日期", self.NOW), "")

    def test_recomputed_per_call(self):
        """同一条记忆，换一个「现在」就是另一个相对日——必须现算、不能存。"""
        self.assertEqual(rel_day("2026-09-19", datetime(2026, 9, 20, 8, 0)), "一天前")
        self.assertEqual(rel_day("2026-09-19", datetime(2026, 9, 21, 8, 0)), "两天前")
        self.assertEqual(rel_day("2026-09-13", datetime(2026, 10, 25, 8, 0)), "一月前",
                         "同一日期，隔六周再看已换档")

    def test_stamp_is_shared_by_all_outlets(self):
        """时间戳的拼法只有一处（所有出口共用）——`（今天）2026-09-20 06:24`。

        同一个时间戳会从注入 / 窗口 / 工具等好几条路到她眼前；这里钉住
        「拼法唯一」，防止哪条路自己拼出另一种样子（漂移过一次——工具侧）。
        """
        self.assertEqual(rel_stamp("2026-09-20", self.NOW), "（今天）2026-09-20")
        self.assertEqual(rel_stamp("2026-09-20 06:24", self.NOW),
                         "（14 小时前）2026-09-20 06:24")
        self.assertEqual(rel_stamp("2026-09-20 20:50", self.NOW),
                         "（10 分钟前）2026-09-20 20:50")
        self.assertEqual(rel_stamp("2026-09-25", self.NOW), "2026-09-25",
                         "未来日期没有相对日：原样给，不硬算")
        self.assertEqual(rel_stamp("", self.NOW), "")

    def test_memory_line_carries_the_label(self):
        """注入的记忆行：`- S1-0009（一天前）2026-09-19 09:16：标题`（带编号）。"""
        class S:                      # 渲染只用到这几个字段
            id = "S1-0009"
            time_event = "2026-09-19 09:16:00"
            title = "标题"
            text = ""
        blk = render_memory_block({"scenes": [S()]}, now=self.NOW)
        self.assertIn("S1-0009（一天前）2026-09-19 09:16：标题", blk)

    def test_summary_line_carries_id_and_topic(self):
        """注入的摘要行也带编号 + 主题（2026-09-24）：`- S2-0001 叙述（主题：A、B）`。

        编号的理由同场景 / 画像行——她指认得出来是哪条、改 / 删说得出编号；
        主题是它的归类，也是"按主题找"的入口。
        """
        class S2:
            id = "S2-0001"
            text = "遇到批评会先退开"
            topic = "用户·被评价的反应"
            topics = ["用户·被评价的反应", "air·工作"]

        blk = render_memory_block({}, summaries=[S2()], now=self.NOW)
        self.assertIn("- S2-0001 遇到批评会先退开（主题：用户·被评价的反应、air·工作）", blk)

        class Old:                     # 老对象 / 只有单主题：退回 topic
            id = "S2-0002"
            text = "叙述"
            topic = "用户·X"

        blk = render_memory_block({}, summaries=[Old()], now=self.NOW)
        self.assertIn("- S2-0002 叙述（主题：用户·X）", blk)

    def test_pending_loop_is_marked(self):
        """没收口的卡带 `→ 未定：…`（2026-09-21 设计稿 A 条）。

        起因：S1-0008 的 outcome 里写着"等用户提供新闻线索"，但那半句
        不注入——她手上只有一句摘要，认不出"那件事"是什么。
        """
        class S:
            id = "S1-0008"
            time_event = "2026-09-20 09:50:00"
            title = "air 的断层与不硬接底线"
            text = "拒编造并请用户补料"
            outcome = "未定，等用户提供新闻线索或点头授权搜索"
            open_loops = []
        blk = render_memory_block({"scenes": [S()]}, now=self.NOW)
        self.assertIn("→ 未定：等用户提供新闻线索或点头授权搜索", blk,
                      "outcome 自带的「未定」不该叠成「未定：未定」")

    def test_closed_loop_is_not_marked(self):
        """闭合回流标过的钩子不再提示（2026-09-21 设计稿 D 条 2）——
        `closed_at` 非空 = 已经了结（标，不删）。"""
        class S:
            id = "S1-0007"
            time_event = "2026-09-20 09:10:00"
            title = "门槛塌了"
            text = "还压着更凉推论"
            outcome = ""
            open_loops = [{"content": "用户提供底稿",
                           "closed_at": "2026-09-21 07:00:00"}]
        blk = render_memory_block({"scenes": [S()]}, now=self.NOW)
        self.assertNotIn("→ 未定", blk)

    def test_open_loop_without_outcome_is_marked(self):
        """没有 outcome 时，`open_loops` 首条顶上（两处都是正式字段）。"""
        class S:
            id = "S1-0007"
            time_event = "2026-09-20 09:10:00"
            title = "门槛塌了"
            text = "还压着更凉推论"
            outcome = ""
            open_loops = [{"content": "用户提供 air 预告过的那条新闻的大概内容（底稿）"}]
        blk = render_memory_block({"scenes": [S()]}, now=self.NOW)
        self.assertIn("→ 未定：用户提供 air 预告过的那条新闻的大概内容（底稿）", blk)


class RequiredKeysTest(unittest.TestCase):
    def test_generated_from_schema(self):
        """必填清单要**从 schema 生成**，不能手写第二遍（两处必然漂移）。"""
        s = required_keys(SCENE_SCHEMA)
        for k in SCENE_SCHEMA["schema"]["required"]:
            self.assertIn(k, s)
        self.assertIn("topic", s, "topic 漏填过一次，必须在清单里")

    def test_prompt_has_no_placeholder_left(self):
        p = extract_scene_prompt("用户: 测试")
        self.assertNotIn("{required}", p)
        self.assertNotIn("{candidates}", p)


class PromptPlaceholderTest(unittest.TestCase):
    """占位符是**单遍**替换的——注入值里的 `{xxx}` 字面量不该被二次换掉。

    链式 `str.replace` 的坑：对话正文里恰好出现 `{required}`（比如用户在
    讨论这套提示词本身），它会被后面那次 `.replace()` 当成占位符——
    内容被静默改写，查都没法查（`_fill` 存在的理由）。
    """

    def test_literal_placeholder_in_conversation_survives(self):
        p = extract_scene_prompt("用户: 这里的 {required} 是什么意思")
        self.assertIn("{required} 是什么意思", p, "对话里的字面量必须原样保留")

    def test_user_text_containing_time_placeholder_is_kept(self):
        p = extract_scene_prompt("用户: {time_context} 会被换成什么")
        self.assertIn("{time_context} 会被换成什么", p)


class OpenLoopsAliasTest(unittest.TestCase):
    """降级到 json_object 档后，键名靠模型自觉——**写错键名不能等于丢数据**。"""

    def test_accepts_model_aliases(self):
        out = _norm_open_loops([{"type": "user_task", "item": "下周三面试",
                                 "due_at": "2026-09-16"}])
        self.assertEqual(len(out), 1, "item/type 这类别名必须认（DeepSeek 就这么写）")
        self.assertEqual(out[0]["content"], "下周三面试")
        self.assertEqual(out[0]["kind"], "user_task")
        self.assertEqual(out[0]["due_at"], "2026-09-16")

    def test_canonical_shape_still_works(self):
        out = _norm_open_loops([{"content": "交房租", "kind": "air_promise", "due_at": ""}])
        self.assertEqual(out[0], {"content": "交房租", "kind": "air_promise",
                                  "due_at": "", "group_name": ""})

    def test_group_name_is_kept_and_clipped(self):
        """组名（一件事的多步共用一个短名，2026-09-22）：认别名、超长截断、
        没给就空串——空 = 独立一条，跟以前一样。"""
        out = _norm_open_loops([
            {"content": "写脚本", "kind": "user_task", "due_at": "", "group": "开源准备"},
            {"content": "跑测试", "kind": "user_task", "group_name": "开" * 30},
            {"content": "独立的事", "kind": "user_task"},
        ])
        self.assertEqual(out[0]["group_name"], "开源准备", "别名 group 也要认")
        self.assertEqual(len(out[1]["group_name"]), 20, "组名截到 20 字（只用于显示）")
        self.assertEqual(out[2]["group_name"], "", "没给组名 → 空串（各自一条）")

    def test_bad_kind_falls_back_not_dropped(self):
        out = _norm_open_loops([{"content": "某事", "kind": "乱七八糟"}])
        self.assertEqual(len(out), 1, "kind 拼错只该回落，不该丢掉整条")
        self.assertEqual(out[0]["kind"], "user_task")

    def test_unrecognizable_is_dropped(self):
        self.assertEqual(_norm_open_loops([{"foo": "bar"}]), [])
        self.assertEqual(_norm_open_loops(["不是对象"]), [])
        self.assertEqual(_norm_open_loops(None), [])


class PleasantryTest(unittest.TestCase):
    """客套话不是承诺——prompt 管不住的那一次，代码兜住。

    prompt 里已经写了「客套话不算承诺」并给了反例，但实测仍会漏：
    `air 邀请用户随时聊天` 被抽成了 `air_promise`。判据刻意保守（短 + 含客套词），
    宁可少收一条：memo 是提醒的候选池，多一条就多一次尴尬提醒。
    """

    def test_pleasantry_is_dropped(self):
        out = _norm_open_loops([{"content": "有想聊的随时来", "kind": "air_promise"}])
        self.assertEqual(out, [], "客套话不该进提醒池")

    def test_real_promise_is_kept(self):
        out = _norm_open_loops([{"content": "下次给你看那份文档", "kind": "air_promise"}])
        self.assertEqual(len(out), 1, "带具体事项的承诺不能被误杀")

    def test_long_text_containing_pleasantry_word_survives(self):
        """客套词只是顺带出现、句子又长 → 不算客套（判据是**短 + 含词**）。"""
        content = "我会在下周三之前把那份关于面试的整理文档发给你，随时可以问我"
        out = _norm_open_loops([{"content": content, "kind": "air_promise"}])
        self.assertEqual(len(out), 1, "有具体事项的承诺不该被误杀")

    def test_only_applies_to_air_promise(self):
        """判据只作用于 `air_promise`——用户的事不归它管。"""
        out = _norm_open_loops([{"content": "随时找他聊", "kind": "user_task"}])
        self.assertEqual(len(out), 1)


class FixedBlocksBudgetTest(unittest.TestCase):
    """常驻块有**软预算**——提示词里唯一「只涨不跌」的账。

    记忆与窗口有 `context.total_budget` 仲裁（超了按优先级裁），而常驻块
    （安全 / 尊重 / 人格 / 工具分寸 / 时间）**裁不动**：每加一个字，都是对所有轮次的
    永久征税（「库一大是稀释注意力」那条原则，第一次用到提示词自己身上，
    2026-09-19）。这个测试就是那道闸：超线就红，逼一次「删还是调」的显式决定。

    口径：基线模式、无记忆、无风格、中文——即每个普通轮次都要付的那部分。
    变动史：2026-09-19 早 2217（宪章 1037 + 记忆纪律 455 + 推进纪律 294 +
    工具分寸 404 + 时间 27）→ 压缩至 2140 → 换新内核（纪律退役）1160 →
    内核复核 1118 → 定版（尊重 63 + air 人格 793 + 工具 377 + 时间 27）1260
    → 安全底线提为固定（+98，air 里危机段移出）→ **诚实块提为固定**（+151，
    air / xina 同删三段：我的边界 / 我和他之间 / 记忆）→ **1413**
    （安全 98 + 尊重 63 + 诚实 151 + air 697 + 工具 377 + 时间 27）。
    口径 = 默认人格 air（「记忆怎么讲」有记忆才注入，不进口径）；
    mia / xina 更长是刻意的（各自做极致，不在闸内——闸盯的是"人格之外"
    与 air 自己的体量）。
    → 2026-09-23：加第 11 个工具（`reject_profile` 否决画像——工具箱稿第二节
    早就写了"她该直接做"，代码一直缺）→ 工具分寸 640 → 735，**同轮精简约 76 字符**
    （`reject_profile` / `memory_search` / `web_fetch` / `memory_search` / `save_now`），
    守住 1700——**余额已用尽**：下一个工具该走"按需加载"（工具箱稿待确认第 1 条），
    而不是再精简或调大这个数。
    → 2026-09-23 同日：**工具箱稿 §二/§三 落地**（检索三件套 11→10 个工具 +
    编号体系段进提示词）——这是"余额已用尽"后**该走的那条路的例外**：
    段子不是新工具，是"她终于知道编号是什么"（原来靠猜）。
    阈值 1700 → **2000**（实际 1900），理由：稿子 §九 第 5 条明写
    "合并后若仍超，**先放宽预算**，按需加载排后"——这轮正是那个情形。
    **按需加载仍排后**：它是下一个真需要省的地方。
    → 2026-09-26：中文那档加一行思维语言（`THINK_IN_CHINESE`，33 字符，
    接在时间之后——**整段提示词的最后一行**）：思维链由模型自己产、
    没有参数可调，实测不写 0/5、中段 2/4、末行 7/10 次中文
    （**是概率不是开关**，位置本身就是效果）。英文那档**零注入**（顺默认，
    实测 6/6 英文；它也不在本闸口径里，加了就是隐形开销）。
    → **1943**（阈值未动；余额只剩 57——下一个想常驻的东西，先想清楚删哪句）。
    → 2026-09-27：air 加「我怎么说话」（语气采样 / 长度 / 收尾，78 字符）。
    为什么只动 air：三个人格里只有 air 是**采样型**——xina 明写「我不接用户的
    情绪，只做认知共情」（她那一格是**否定的**）、mia 的裁判是「好玩 / 熵变」
    （她那一格是**对冲**）；而 xina / mia 各自早已写全了长度与收尾
    （「力气跟问题的分量走」「句号比问号有力」「熵不变最坏」），**只有 air
    没写采样规则**。所以这不是共享脚手架该去的地方（同 2026-09-19「怎么说话
    整体归人格文件」）。同轮在 air 内部删 23 字符（「不预设好与不好」
    「以用户为中心」、三连「鼓励用户」、思考段尾句——都是下游已展开过的重复）。
    **1999**（阈值未动；余额 1——再往 air 里加东西，就得先删。
    xina 1990 / mia 2214，仍不在口径里：闸盯的是"人格之外"与 air 自己的体量）。
    → 2026-09-28：air 补两处规则，管的是**没被邀请的点评**。那套「先说好、后说不好」的
    补充说明不是失灵，是训练目标本身（平衡性被奖励 → 给结论必须覆盖反面）；
    抠字眼 / 装严谨同源（"严谨"被过度实例化）。**它顶的是强默认，禁令顶不掉**，
    只能写成**分岔口 + 近因**（同 `THINK_IN_CHINESE` 那条的性质：是概率、不是开关）：
    ① 行「先分清用户这次要什么」补第三态**只是分享**（+5）——原来只有
    "想倾诉 / 要解决"两态，"只是在说自己的看法"没有位置；
    ② 新增「判断：用户没问就不给——分享不是请人点评；要补就轻着补，别摆成
    一份利弊表」（+37）。原则不是新的：`RESPECT_RULE`「不为用户下结论」就是它，
    缺的只是人格层那一句实例化。
    **只给 air**（2026-09-28 定）：xina 的放大镜若加准入门槛会拆掉她的人设
    （她认的是真，那一格是主动的）；mia 的裁判是"好玩 / 熵变"，不出这个症状。
    **2043**（阈值 2000 → **2100**；余额 57）。为什么这次是调阈值而不是再删：
    2026-09-27 已把 air 里真正的重复合过一遍，剩下的是承载内容的句子——
    这轮加的是"少做一件事"的规则，挤掉别的话等于拿内容换空位。
    **调阈值要写理由，别静默调大。**
    → **2026-10-05：口径修正 + 阈值 2100 → 2200（2170，余额 30）**。
    上面所有数字都是**旧口径**——它漏算了〔署名〕块（`render_persona_note`，
    2026-09-25 加：多人格时每轮都注入的那一段，三个名字 127 字符）。
    漏的原因很实在：测试调的 `build_system_prompt` 没传 `persona` / `personas`
    （署名零注入），而线上每轮都传——**闸量的是"除署名外的常驻块"**，
    这条注释记了半个月没人发现，因为数字一直在余额里。
    这次是设置页的人格编辑器要显示同一个数（`fixed_blocks_report`）才把两边对齐的：
    **量真实那一份**（含署名）。阈值 +100 = 补上这笔一直没入账的钱，
    **不是给新内容松绑**——想加新东西仍然要先删（余额 30）。
    顺带定死：阈值只在 `config.prompt.fixed_budget_chars` 一处，本文件不再留副本。
    """

    def test_fixed_blocks_within_budget(self):
        rep = fixed_blocks_report(load_persona("air"), "air", persona_names())
        self.assertLessEqual(
            rep["chars"], rep["limit"],
            f"常驻块共 {rep['chars']} 字符，超了 {rep['limit']} 的预算——"
            "先考虑合并/删除，再考虑调大阈值（并在注释里写理由）")

    def test_fixed_rules_present_and_ordered(self):
        """固定层三段都在，且按 安全 → 尊重 → 诚实 排——漏接、乱序都该红。

        2026-09-19：诚实块加进 parts 时漏挂过一次（常量写了、装配忘了），
        组装验证抓出来的——这道断言就是那次教训。
        """
        s = build_system_prompt(load_persona("air"), {}, facts=[], tools=[])
        for name in ("## 安全底线", "## 尊重", "## 诚实"):
            self.assertIn(name, s)
        self.assertLess(s.index("## 安全底线"), s.index("## 尊重"))
        self.assertLess(s.index("## 尊重"), s.index("## 诚实"))


class PersonaNoteTest(unittest.TestCase):
    """署名说明（2026-09-25）：**多个人格共存**时注入——人格可中途切换，
    记忆与记录里的名字是"当时的说话人"，当前人格得能对上号
    （哪个名字是「自己」、哪些是别的时候在线的）。
    单人格没有歧义、零注入（同「没素材不喊规矩」的节省）。
    """

    def test_injected_with_multiple_personas(self):
        note = render_persona_note("xina", ["air", "mia", "xina"])
        self.assertIn("你现在的名字是「xina」", note)
        self.assertIn("air、mia", note, "名单里的其他名字要列出来")

    def test_single_persona_is_silent(self):
        """只有一个人格时没有"认错自己"的空间——零注入。"""
        self.assertEqual(render_persona_note("air", ["air"]), "")

    def test_unknown_persona_is_silent(self):
        """当前人格不在名单里（异常状态）不硬写——宁可没有说明，也不写错。"""
        self.assertEqual(render_persona_note("ghost", ["air", "mia"]), "")

    def test_system_prompt_carries_note(self):
        """装配挂上了：`build_system_prompt` 传 persona/personas 就有这一段。"""
        s = build_system_prompt("（人格占位）", {}, persona="mia",
                                personas=["air", "mia", "xina"])
        self.assertIn("你现在的名字是「mia」", s)


class ProfileStatementRulesTest(unittest.TestCase):
    """画像陈述的写法判据**三处共用**（2026-09-21，P1/P2）。

    新抽（`profile_prompt`）/ 修正（`revision_prompt`）/ 渐变（`drift_prompt`）
    产出的都是"一条画像陈述"——各写一遍的后果：新画像按称重方式写、
    修正一次退回行为对，写法在几根轨道上来回漂（同本地「待优化稿」A 条教训）。
    所以这里断言：同一段判据**原文**出现在三个入口的成品提示词里。
    """

    def _scenes(self) -> list:
        return [SimpleNamespace(id="S1-0001", title="被当众批评", time_event="2026-09-01",
                                trigger="被当众批评", reaction="不想争、想走",
                                outcome="", text="被批评后想走")]

    def test_rules_shared_by_all_three_entrypoints(self):
        """三个入口都要引用同一段判据原文——缺一个就是在留漂移的口子。"""
        s = self._scenes()
        s2 = [SimpleNamespace(id="S2-0001", text="聚合叙述")]
        prompts = {
            "新抽": profile_prompt("用户·被评价的反应", s2, s),
            "修正": revision_prompt("用户·被评价的反应", "遇到被评价会先退出", s),
            "渐变": drift_prompt("用户·被评价的反应", "遇到被评价会先退出", s, s),
        }
        for name, p in prompts.items():
            self.assertIn(PROFILE_STATEMENT_RULES, p,
                          f"{name}入口没有引用共用判据——写死了就会漂")

    def test_rules_carry_the_two_p1_landings(self):
        """判据里必须装着 P1 的两条落点：称重方式优先、机制词不进画像。"""
        self.assertIn("称重方式", PROFILE_STATEMENT_RULES)
        self.assertIn("机制名", PROFILE_STATEMENT_RULES)

    def test_rules_cover_evaluation_and_distinctiveness(self):
        """通用画像判据的落点（2026-09-23）：不写评价词、要有区分度、变化带时间。

        这三条是"人格画像该有的样子"那套标准里，原判据没覆盖到的部分
        （评价性语言、巴纳姆式车轱辘话、状态/变化）。
        """
        self.assertIn("评价词", PROFILE_STATEMENT_RULES)
        self.assertIn("分开", PROFILE_STATEMENT_RULES)
        self.assertIn("变化", PROFILE_STATEMENT_RULES)


class _LocalConfigTestCase(unittest.TestCase):
    """把 `config.local.json` 指到临时目录，绝不碰用户的真文件。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old_path = cfgmod.PATHS["local_config"]
        cfgmod.PATHS["local_config"] = str(Path(self.tmp.name) / "config.local.json")

    def tearDown(self):
        cfgmod.PATHS["local_config"] = self._old_path
        self.tmp.cleanup()

    def _saved(self) -> dict:
        return json.loads(settings.local_config_path().read_text(encoding="utf-8"))


class MaskedKeyTest(_LocalConfigTestCase):
    """打码值和空值都**不能**落盘覆盖真 key——这两种都会毁掉已存的 key。"""

    def test_masked_key_does_not_overwrite(self):
        settings.save_local({"llm": {"endpoint": "https://a",
                                     "api_key": "sk-real-key-1234567890"}})
        settings.save_local({"llm": {"endpoint": "https://b", "api_key": "••••••••"}})
        data = self._saved()
        self.assertEqual(data["llm"]["api_key"], "sk-real-key-1234567890",
                         "打码值不是 key，不能覆盖真 key")
        self.assertEqual(data["llm"]["endpoint"], "https://b", "该改的还是要改")

    def test_empty_key_keeps_existing(self):
        settings.save_local({"llm": {"endpoint": "https://a",
                                     "api_key": "sk-real-key-1234567890"}})
        settings.save_local({"llm": {"endpoint": "https://c", "api_key": ""}})
        self.assertEqual(self._saved()["llm"]["api_key"], "sk-real-key-1234567890",
                         "只改 endpoint 时不该把 key 抹成空")

    def test_is_masked(self):
        # 当前掩码（ASCII 星号）：整串，以及**夹在首尾明文之间**的长 key 打码形状
        self.assertTrue(settings._is_masked(settings._MASK))
        self.assertTrue(settings._is_masked("sk-abc" + settings._MASK + "wxyz"))
        # 旧圆点掩码仍要认（老配置里可能存着它）
        self.assertTrue(settings._is_masked("••••••••"))
        self.assertTrue(settings._is_masked("••••"))
        self.assertFalse(settings._is_masked("sk-abc"))
        self.assertFalse(settings._is_masked(""))
        self.assertFalse(settings._is_masked(None))

    def test_mask_is_ascii(self):
        """打码串必须是纯 ASCII——非 ASCII（如圆点 U+2022）在 GBK 控制台一 print 就炸。"""
        self.assertTrue(all(ord(c) < 128 for c in settings._MASK),
                        f"掩码含非 ASCII 字符: {settings._MASK!r}")

    def test_embedded_mask_does_not_overwrite(self):
        """长 key 的打码形状（首尾留明文）也不能覆盖真 key——这条以前漏判过。"""
        settings.save_local({"llm": {"endpoint": "https://a",
                                     "api_key": "sk-real-key-1234567890"}})
        settings.save_local({"llm": {"endpoint": "https://b",
                                     "api_key": "sk-r" + settings._MASK + "7890"}})
        self.assertEqual(self._saved()["llm"]["api_key"], "sk-real-key-1234567890",
                         "夹着掩码的串不是 key，不能覆盖真 key")


class LocalConfigMergeTest(_LocalConfigTestCase):
    """保存动作**不该顺手删配置**：表单没覆盖到的键原样保留。

    `insecure_ssl` 这类手写键最容易中招——页面上没有它，
    一次「只改了 endpoint」的保存就会把它抹掉，而且没人会立刻发现。
    """

    def test_unknown_keys_are_preserved(self):
        settings.save_local({"embedding": {"endpoint": "http://127.0.0.1:11434/v1",
                                           "model": "bge-m3", "api_key": "ollama",
                                           "insecure_ssl": True}})
        # 模拟从设置页保存：表单里没有 insecure_ssl，key 框留空
        settings.save_local({"embedding": {"endpoint": "http://127.0.0.1:11435/v1",
                                           "model": "bge-m3", "api_key": ""}})
        data = self._saved()
        self.assertEqual(data["embedding"]["endpoint"], "http://127.0.0.1:11435/v1")
        self.assertEqual(data["embedding"]["api_key"], "ollama", "key 的防呆照旧")
        self.assertTrue(data["embedding"].get("insecure_ssl"),
                        "表单没覆盖到的键不该被保存动作丢掉")

    def test_other_sections_are_preserved(self):
        """**段一级**的同一道理：表单只传 llm / embedding，磁盘上手写的
        `search`（或 `tts`）段不该因为一次「改 LLM 设置」的保存被抹掉。
        """
        settings.save_local({"search": {"model": "deepseek-v4-pro"}})
        # 模拟从设置页保存：只有 llm 段
        settings.save_local({"llm": {"endpoint": "https://api.deepseek.com/v1"}})
        data = self._saved()
        self.assertEqual((data.get("search") or {}).get("model"), "deepseek-v4-pro",
                         "表单没覆盖到的段不该被保存动作丢掉")
        self.assertEqual(data["llm"]["endpoint"], "https://api.deepseek.com/v1")


class EmbeddingSslTest(_LocalConfigTestCase):
    """证书校验**默认开启**——「服务只读所以低危」不成立，被截的是 key 本身。

    自签 / 内网端点靠显式 `insecure_ssl` 放开；它只影响 embedding 一条路。
    """

    def test_default_is_verify(self):
        svc = EmbeddingService(endpoint="https://x/v1", api_key="k")
        self.assertFalse(svc.insecure_ssl, "默认必须校验证书")

    def test_flag_is_carried(self):
        svc = EmbeddingService(endpoint="https://x/v1", api_key="k",
                               insecure_ssl=True)
        self.assertTrue(svc.insecure_ssl)

    def test_config_file_can_turn_it_on(self):
        """`insecure_ssl` 写在文件里要能生效（存在即生效，不信真值）。

        ⚠️ 这里调 `_apply_file()` 而不是 `apply()`：`apply()` 还会顺带应用
        环境变量——开发机上真设着求知版遗留的 `AIR2_LLM_*`，它会把全局
        `CONFIG["llm"]` 换成真实配置（之后所有用例里的 `LLM()` 都"可用"了，
        连"没有模型就不能重建"那条都会被带翻车）。测哪一段就调哪一段。
        """
        old = dict(cfgmod.CONFIG["embedding"])
        self.addCleanup(lambda: cfgmod.CONFIG.update({"embedding": old}))
        settings.save_local({"embedding": {"insecure_ssl": True}})
        settings._apply_file()
        self.assertTrue(cfgmod.cfg("embedding", "insecure_ssl"),
                        "`insecure_ssl` 写在文件里要能生效（存在即生效，不看真值）")


class SearchToggleTest(_LocalConfigTestCase):
    """`search` 段的文件键：开关存在即生效 + 通道改名不淘汰老配置。

    `enabled` 同 `insecure_ssl` 的道理：`false` 是一个**决定**，不是"没填"。
    """

    def test_config_file_can_turn_it_off(self):
        old = dict(cfgmod.CONFIG["search"])
        self.addCleanup(lambda: cfgmod.CONFIG.update({"search": old}))
        settings.save_local({"search": {"enabled": False}})
        settings._apply_file()
        self.assertFalse(cfgmod.cfg("search", "enabled"),
                         "`enabled: false` 写在文件里要能生效（存在即生效）")

    def test_old_style_key_still_selects_the_channel(self):
        """**改名不淘汰老配置**（2026-09-26 搜索通道重构）。

        2026-09-22 ~ 09-26 期间写的是 `search.style`，之后读的是
        `search.channel`——旧名要被**搬进**新键，不是并排留着：
        并排读"新名优先"的话，新键的默认值（anthropic）会挡住旧名写的
        openai，老配置静默失效（症状：拿新家的 key 去打旧家的端点，401）。
        """
        old = dict(cfgmod.CONFIG["search"])
        self.addCleanup(lambda: cfgmod.CONFIG.update({"search": old}))
        settings.save_local({"search": {"style": "openai"}})
        settings._apply_file()
        self.assertEqual(cfgmod.cfg("search", "channel"), "openai")
        ch, why = searchmod.selected()
        self.assertIsNotNone(ch, why)
        self.assertEqual(ch.name, "openai")


class WebFetchToggleTest(_LocalConfigTestCase):
    """`web.fetch_enabled` 写在文件里要能生效——同 `search.enabled` 的道理：
    开关的 `false` 是一个**决定**，不是"没填"，也绝不能静默失效
    （文档承诺了它，失效就是「配置在说谎」）。
    """

    def test_config_file_can_turn_it_off(self):
        old = dict(cfgmod.CONFIG["web"])
        self.addCleanup(lambda: cfgmod.CONFIG.update({"web": old}))
        settings.save_local({"web": {"fetch_enabled": False}})
        settings._apply_file()
        self.assertFalse(cfgmod.cfg("web", "fetch_enabled"),
                         "`fetch_enabled: false` 写在文件里要能生效（存在即生效）")


class PriorityTest(_LocalConfigTestCase):
    """legacy 环境变量只做兜底，**不能压过本版配置文件**。

    踩过的坑：文件里明明改成了 DeepSeek，生效的还是求知版留下的
    `AIR2_LLM_*`，而且界面上完全看不出来——「改了没用」最难查。
    """

    ENV_KEYS = ("AIR2_LLM_ENDPOINT", "AIR_LINK_LLM_ENDPOINT")

    def setUp(self):
        super().setUp()
        self._env = {k: os.environ.get(k) for k in self.ENV_KEYS}
        for k in self.ENV_KEYS:
            os.environ.pop(k, None)
        self._saved_conf = dict(cfgmod.CONFIG["llm"])

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cfgmod.CONFIG["llm"] = self._saved_conf
        super().tearDown()

    def test_file_beats_legacy_env(self):
        settings.save_local({"llm": {"endpoint": "https://api.deepseek.com"}})
        os.environ["AIR2_LLM_ENDPOINT"] = "https://api.xiaomimimo.com/v1"
        settings.apply()
        self.assertEqual(cfgmod.cfg("llm")["endpoint"], "https://api.deepseek.com",
                         "本版文件应当压过求知版遗留的环境变量")

    def test_new_env_beats_file(self):
        settings.save_local({"llm": {"endpoint": "https://api.deepseek.com"}})
        os.environ["AIR_LINK_LLM_ENDPOINT"] = "https://override.example/v1"
        settings.apply()
        self.assertEqual(cfgmod.cfg("llm")["endpoint"], "https://override.example/v1",
                         "本版环境变量仍应能覆盖（部署场景）")

    def test_legacy_env_still_works_as_fallback(self):
        cfgmod.CONFIG["llm"]["endpoint"] = ""
        os.environ["AIR2_LLM_ENDPOINT"] = "https://fallback.example/v1"
        settings.apply()
        self.assertEqual(cfgmod.cfg("llm")["endpoint"], "https://fallback.example/v1",
                         "文件没配时 legacy 仍该兜底")

    def test_env_overrides_reports_only_effective(self):
        settings.save_local({"llm": {"endpoint": "https://api.deepseek.com"}})
        os.environ["AIR2_LLM_ENDPOINT"] = "https://ignored.example"
        self.assertNotIn("endpoint", settings.env_overrides("llm"),
                         "文件压住了 legacy，就不该再提示「来自环境变量」")


class DashboardBindingTest(unittest.TestCase):
    """仪表盘只绑本机——那是「能看别人记忆」的东西，不该有外部口子。

    这条没法用行为测（要真起服务），所以直接断言源码里的绑定地址：
    它是一行代码，将来谁改成 `0.0.0.0`，这个测试会立刻红。
    """

    def test_binds_loopback_only(self):
        src = (Path(__file__).resolve().parents[1] / "core" / "dashboard.py"
               ).read_text(encoding="utf-8")
        self.assertIn('ThreadingHTTPServer(("127.0.0.1", port)', src,
                      "仪表盘必须绑 127.0.0.1")
        self.assertNotIn('"0.0.0.0"', src, "不该监听所有网卡")


if __name__ == "__main__":
    unittest.main()
