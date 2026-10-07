"""工具箱的测试。

设计记录见本地「工具箱」稿（不随仓库分发）。这一层测的不是"工具能不能用"，
是**它在不该动的时候动不动**：

  - 需确认档没有确认通道时，**必须拒绝**（失败即拒绝）——
    少做一件事的代价，远小于自作主张地把一条记忆改坏
  - 只读工具一轮限量：**库内 1 / 下钻 1 / 网络 2**（防止围着库或外面转圈，
    但「查一次 → 读一段原文」和「搜一次 → 抓一页」是活，不该被掐死）
  - 未知动作要把「可用的」说出来（她下一轮能自己改对）
  - 参数认别名（DeepSeek 没有严格参数约束，指望它写对不如代码认下来）
  - **每次调用都留痕**——她改记忆的动作，和他改记忆一样，都要能回看
  - 工具循环最多两轮（air 是聊天，不是跑长循环的代码 agent）
"""
# 用例分组：
#   脚手架  _NoEmbedding · Base · _FakeLLM · _CaptureLLM
#   注册与门 ToolRegistryTest 工具表 · GuardTest 红线终审 · ToolTraceTest 留痕 ·
#           ToolLoopTest 循环上限 · QuotaGroupsTest 三条限流线
#   只读    RecallMemoryTest 回忆 · RecallDimensionsTest 维度取交集 ·
#           ListScenesTest 列编号 · ListMemosTest 列备忘录 ·
#           ReadRawTest 读原文 · WebSearchTest 搜索 · WebFetchTest 取网页 ·
#           SearchToolVisibilityTest 不支持/关掉就摘掉
#   写入    SaveNowTest 立刻记 · CloseMemoTest 闭合备忘 · ConfirmFlowTest 确认流 ·
#           ConfirmJudgementTest 点头判定
from __future__ import annotations

import email.message
import io
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core.chat import ChatSession, _RECEIPT_TURNS, _judge_confirmation
from core.weave import (archive_by_layer, delete_by_layer, unarchive_by_layer,
                        update_by_layer)
from core.llm import LLM
from core.model import (MEMO_AIR_PROMISE, MEMO_CLOSED, MEMO_PENDING,
                        MEMO_RAISED, MEMO_USER_TASK, PROFILE_ESTABLISHED,
                        PROFILE_PENDING, Memo, Profile, Raw, Scene, Summary)
from core.prompts import SCENE_SCHEMA, TOOLS_LINES, render_tools_block
from core.shortterm import ShortTerm
from core.store import Store
from core.entity import link_entities
# 搜索的形态知识在通道表里（core/search.py）——`llm.search()` 只跑通用流程
from core import search as searchmod
from core.search import _parse_anthropic, _parse_openai
from core.tools import ASK, AUTO, READ, TOOLS, execute, openai_tools, tool_names
from core import webfetch


class _NoEmbedding:
    available = False

    def embed_one(self, text):
        return None

    def embed(self, texts):
        return None


def tool_llm(calls_per_round, final="好。"):
    """假模型：前几轮返回工具调用，之后回一句人话。

    `calls_per_round`：第 n 轮要调的工具列表（`[{"name":..., "arguments":{...}}]`），
    没有工具调用就走普通回复。
    """
    state = {"n": 0}

    def fn(prompt, schema):
        name = (schema or {}).get("name") or ""
        if name:                                   # 结构化调用（抽取 / 线索）
            if name == "cue_judgement":
                return {"tense": "now", "valence": None, "arousal": 0,
                        "about_relation": False, "unresolved": False}
            if name == "memo_raise":
                return {"raise": False}
            if name == "memo_class":
                return {"items": []}
            if name == "scene_card":
                return dict(SCENE_SCHEMA["defaults"], worth_saving=False)
            return {}
        i = state["n"]
        state["n"] += 1
        if i < len(calls_per_round):
            calls = [dict(c, id=c.get("id") or f"call_{i}_{j}")
                     for j, c in enumerate(calls_per_round[i])]
            return {"content": "", "tool_calls": calls}
        return final

    llm = LLM()
    llm.set_mock(fn)
    return llm


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = Store(self.root / "t.db")
        self._old_paths = {k: cfgmod.PATHS[k]
                           # shortterm 也要：有些测试直接建 ChatSession（没传
                           # state_path），默认会落到真实的窗口文件——漏了它，
                           # 跑一次测试就往真窗口里塞几条消息（实锤过）
                           for k in ("trace_dir", "raws_dir", "backup_dir",
                                     "shortterm")}
        cfgmod.PATHS["trace_dir"] = str(self.root / "trace")
        cfgmod.PATHS["raws_dir"] = str(self.root / "raws")
        cfgmod.PATHS["backup_dir"] = str(self.root / "backups")
        # 窗口也要重定向：直接建 `ChatSession`（没传 state_path）的测试
        # 默认会落到真实窗口文件——漏了它，跑一次测试就往真窗口里塞几条消息
        cfgmod.PATHS["shortterm"] = str(self.root / "shortterm.json")
        self.ctx = {"emb": None}

    def tearDown(self):
        for k, v in self._old_paths.items():
            cfgmod.PATHS[k] = v
        self.store.close()
        self._tmp.cleanup()

    def traces(self, kind="工具") -> str:
        files = list(Path(cfgmod.abspath(cfgmod.PATHS["trace_dir"])).glob(f"{kind}-*.jsonl"))
        return "\n".join(f.read_text(encoding="utf-8") for f in files)


class ToolRegistryTest(unittest.TestCase):
    def test_every_tool_declares_a_mode_and_description(self):
        for name, t in TOOLS.items():
            self.assertIn(t["mode"], (AUTO, ASK, READ), f"{name} 的档位不合法")
            self.assertTrue(t["description"].strip(), f"{name} 没有描述")

    def test_openai_shape(self):
        spec = openai_tools(["memory_search"])[0]
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "memory_search")
        self.assertIn("parameters", spec["function"])

    def test_names_are_distinguishable(self):
        """名字之间不能像——约束不可靠时，语义距离就是唯一的约束。"""
        names = list(TOOLS.keys())
        self.assertEqual(len(names), len(set(names)))
        for a in names:
            for b in names:
                if a != b:
                    self.assertNotIn(a, b, f"{a} 和 {b} 长得太像，她会选错")

    def test_every_tool_is_fully_wired(self):
        """每个工具**六件齐**（设计稿 §九 第 4 件）：mode / description /
        parameters / run / `TOOLS_LINES` 分寸 / openai 形状。

        为什么要有这条：**"分寸漏一条"以前靠人眼**——加了个工具、`TOOLS_LINES`
        忘了配，提示词里就少一句"该不该用"，而她照旧会调它（选错工具的源头）。
        参数形状也要真能转成 openai 那份（空 properties 合法，键必须在）。
        """
        for name, t in TOOLS.items():
            self.assertIn(t["mode"], (AUTO, ASK, READ), f"{name} 档位不合法")
            self.assertTrue(t["description"].strip(), f"{name} 没有描述")
            self.assertIn(name, TOOLS_LINES,
                          f"{name} 少了分寸（提示词里就没有「该不该用」这一句）")
            self.assertTrue(callable(t["run"]), f"{name} 没有 run")
            self.assertIsInstance(t["parameters"].get("properties"), dict,
                                  f"{name} 的 parameters 形状不对")
            spec = openai_tools([name])[0]
            self.assertEqual(spec["function"]["name"], name)
            self.assertEqual(spec["function"]["parameters"], t["parameters"])


class GuardTest(Base):
    def test_unknown_tool_lists_what_is_available(self):
        """未知动作要告诉她「怎么才是对的」——她下一轮能自己改对。"""
        out = execute(self.store, "delete_everything", {}, self.ctx)
        self.assertFalse(out["ok"])
        self.assertIn("没有 delete_everything", out["detail"])
        self.assertIn("memory_search", out["detail"])

    def test_ask_without_channel_is_denied(self):
        """**失败即拒绝**：没有能点头的人，就不能做。"""
        # 造一个需确认档的工具来验红线（现有工具里没有 ask 档的）
        from core import tools as tools_mod
        tools_mod.TOOLS["_probe_ask"] = {
            "mode": ASK, "description": "测试用",
            "parameters": {"type": "object", "properties": {}},
            "run": lambda store, args, ctx: {"ok": True, "detail": "不该走到这里"},
        }
        try:
            out = execute(self.store, "_probe_ask", {}, self.ctx)
            self.assertFalse(out["ok"], "没有确认通道时必须拒绝")
            self.assertIn("要先问用户一句", out["detail"])
            # 有通道就该放行
            out2 = execute(self.store, "_probe_ask", {}, {"confirm_channel": True})
            self.assertTrue(out2["ok"])
        finally:
            tools_mod.TOOLS.pop("_probe_ask", None)

    def test_read_tool_counts_without_blocking(self):
        """库内只读**不再硬拒**（2026-09-23）：次数只当信号——到阈值附一句提醒，
        继不继续**由她判**（"够了就停"在提示词里）。判据闸（同一个词）另测。"""
        ctx = {"emb": None}
        for q in ("甲", "乙", "丙"):
            execute(self.store, "memory_search", {"query": q}, ctx)
        out = execute(self.store, "memory_search", {"query": "丁"}, ctx)
        self.assertTrue(out["ok"], "第 4 次不该被拒（旧行为是拒）")
        self.assertIn("你已经查了 4 次", out["detail"])

    def test_same_query_is_refused_without_spending_quota(self):
        """同一个词再查 = 原地打转——拒绝且**不占计数**（E 条的"换词"要求）。"""
        ctx = {"emb": None}
        self.assertTrue(execute(self.store, "memory_search", {"query": "面试"}, ctx)["ok"])
        again = execute(self.store, "memory_search", {"query": "面试"}, ctx)
        self.assertFalse(again["ok"])
        self.assertIn("换个词", again["detail"])
        # 那次拒绝不占计数：再查两次才到提醒线（若占了，这里会显示 4 次）
        out = execute(self.store, "memory_search", {"query": "甲"}, ctx)
        self.assertNotIn("你判", out["detail"], "第 2 次还没到提醒线")
        out = execute(self.store, "memory_search", {"query": "乙"}, ctx)
        self.assertIn("你已经查了 3 次", out["detail"],
                      "被换词拒的那次不该占计数（否则这里会是 4 次）")


class RecallMemoryTest(Base):
    def test_finds_and_says_so_when_not_found(self):
        s = Scene(title="面试没过", text="面试没过，再找找")
        self.store.add_scene(s)

        hit = execute(self.store, "memory_search", {"query": "面试"}, {"emb": None})
        self.assertTrue(hit["ok"])
        self.assertIn("面试没过", hit["detail"])

        miss = execute(self.store, "memory_search", {"query": "量子力学"}, {"emb": None})
        self.assertTrue(miss["ok"])
        self.assertIn("没找到", miss["detail"])

    def test_accepts_q_alias(self):
        self.store.add_scene(Scene(title="面试", text="面试"))
        out = execute(self.store, "memory_search", {"q": "面试"}, {"emb": None})
        self.assertIn("面试", out["detail"])

    def test_importance_is_ranked_above_mere_similarity(self):
        """三键排序（§3.1，2026-09-24）：**同档内先给重要的**（核心度）。

        语义分在这里**只是门**（0 分不要），不是排序键——"先给重要的"才是排序的
        意思（与注入侧共用 `recall.core_score`）。这条用例的构造正好把新旧行为分开：
        兜底检索下短文本的字符重叠率**更高**（旧排序会把它排前），但它没有被引用、
        没有行为强度——新排序该让另一条（intensity + cited）排前。
        """
        thin = Scene(title="面试", text="面试", intensity=0.0)
        self.store.add_scene(thin)
        rich = Scene(title="面试复盘", text="面试没过，聊了很久",
                     intensity=1.0, cited_by_profile=5)
        self.store.add_scene(rich)

        out = execute(self.store, "memory_search", {"query": "面试"}, {"emb": None})
        self.assertIn(rich.id, out["detail"])
        self.assertIn(thin.id, out["detail"])
        self.assertLess(out["detail"].index(rich.id), out["detail"].index(thin.id),
                        "核心度高的先说——排序看质量，不看谁的字面重叠更像")


# （原 `SaveNowTest`——三个用例测的是 `save_now`——2026-09-24 随工具一起删除；
#   历史见 git。）


class ToolTraceTest(Base):
    def test_every_call_is_traced(self):
        execute(self.store, "memory_search", {})
        execute(self.store, "memory_search", {"query": "面试"}, {"emb": None})
        execute(self.store, "不存在的工具", {})

        text = self.traces()
        self.assertIn("memory_search", text)
        self.assertIn("memory_search", text)
        self.assertIn("不存在的工具", text)


class ToolLoopTest(Base):
    """对话层里的工具循环：她调了工具 → 执行 → 再问一次（**最多两轮**）。"""

    def _session(self, llm) -> ChatSession:
        return ChatSession(self.store, llm, emb=_NoEmbedding(),
                           state_path=self.root / "st.json")

    def test_tool_result_goes_back_and_final_reply_is_used(self):
        self.store.add_scene(Scene(title="面试没过", text="面试没过，再找找"))
        llm = tool_llm([[{"name": "memory_search",
                          "arguments": {"query": "面试"}}]],
                       final="好，我想起来了。")
        out = self._session(llm).reply("面试怎么样了")

        self.assertEqual(out["reply"], "好，我想起来了。")
        self.assertEqual(len(out["tools"]), 1)
        self.assertTrue(out["tools"][0]["ok"])
        self.assertIn("面试没过", out["tools"][0]["detail"])

    def test_tool_detail_does_not_enter_the_window(self):
        """工具结果只回给她看：**进窗口的是她的话，不是工具的输出**。

        工具结果是那一轮的临时消息（`messages` 里 role=tool，下一轮就没了）。
        写进窗口的只有 `st.append("user"…)` / `st.append("air", reply)`——
        哪天它跟着混进 `reply`，那句工具输出就会变成"她说过的话"，
        之后还会被提取进长期库（同 `hint` 不冒充正文的那条坑）。
        """
        self.store.add_scene(Scene(title="面试没过", text="面试没过，再找找"))
        llm = tool_llm([[{"name": "memory_search",
                          "arguments": {"query": "面试"}}]],
                       final="好，我想起来了。")
        sess = self._session(llm)
        out = sess.reply("面试怎么样了")

        spoken = "\n".join(m["text"] for m in sess.st.messages)
        self.assertIn("好，我想起来了。", spoken, "她的话要进窗口")
        self.assertNotIn("面试没过", spoken,
                         "工具输出不该进窗口——否则她下一轮会'记得自己说过'")
        self.assertIn("面试没过", out["tools"][0]["detail"],
                      "工具输出只走 `tools` 那一路（给仪表盘）")

    def test_loop_stops_after_two_rounds(self):
        """她要是每轮都调工具，两轮就停——air 是聊天，不是长循环。"""
        llm = tool_llm([[{"name": "memory_search", "arguments": {"query": "甲"}}],
                        [{"name": "memory_search", "arguments": {"query": "甲"}}],
                        [{"name": "memory_search", "arguments": {"query": "甲"}}]],
                       final="好。")
        out = self._session(llm).reply("帮我查查")

        self.assertLessEqual(len(out["tools"]), 2, "最多两轮")
        # 第二轮同一个词会被挡下（换词要求）——两道闸都生效：循环上限 + 换词
        self.assertFalse(out["tools"][-1]["ok"])
        self.assertIn("查过了", out["tools"][-1]["detail"])

    def test_plain_reply_when_no_tool_called(self):
        llm = tool_llm([], final="嗯，我在。")
        out = self._session(llm).reply("在吗")
        self.assertEqual(out["reply"], "嗯，我在。")
        self.assertEqual(out["tools"], [])


def _search_reply(text: str = "要点一二三", query: str = "搜的词",
                  url: str = "https://example.com/x", title: str = "示例页") -> dict:
    """造一份 Anthropic Messages 搜索响应（结构化结果块 + 汇报文本）。"""
    return {"content": [
        {"type": "server_tool_use", "name": "web_search",
         "input": {"query": query}},
        {"type": "web_search_tool_result",
         "content": [{"type": "web_search_result", "url": url, "title": title}]},
        {"type": "text", "text": text},
    ]}


class _FakeLLM(LLM):
    """假 Messages 通道：只替换**网络那一层**（`_post_json`），解析与降级照常走真代码。

    `last` 记下最后一次请求（URL / 头 / 载荷）——"这个形态到底发去哪、带什么头"
    是通道表**被正确使用**的证据，得看得见（以前这两件事分别埋在
    `_post_messages` / `_post_chat` 里，测试碰不到）。
    """

    def __init__(self, data: dict | None = None, err: Exception | None = None):
        super().__init__()
        self.endpoint = "https://api.deepseek.com"     # 让它 `available()`
        self.api_key, self.model = "sk-test", "deepseek-flash"
        self.data, self.err, self.calls = data or {}, err, 0
        self.last = None

    def _post_json(self, url, headers, payload, timeout=None):
        self.calls += 1
        self.last = (url, headers, payload)
        if self.err is not None:
            raise self.err
        return self.data


class _FakeChatLLM(LLM):
    """假 chat 通道（openai 形态搜索）：只替换**网络那一层**，解析照真代码走。"""

    def __init__(self, data: dict | None = None, err: Exception | None = None):
        super().__init__()
        self.endpoint = "https://api.mimo.example/v1"   # 让它 `available()`
        self.api_key, self.model = "sk-test", "mimo-v2.6-flash"
        self.data, self.err, self.calls = data or {}, err, 0
        self.last = None

    def _post_json(self, url, headers, payload, timeout=None):
        self.calls += 1
        self.last = (url, headers, payload)
        if self.err is not None:
            raise self.err
        return self.data


class _CaptureLLM(LLM):
    """抓住「这一轮给了她哪些工具」，用来看搜索被摘掉没有。"""

    def __init__(self):
        super().__init__()
        self.seen = None

    def chat_with_tools(self, messages, tools=None, max_tokens=3000, temperature=None,
                        timeout=None):
        self.seen = tools
        return {"content": "好", "tool_calls": []}


class _FakeResponse:
    """假 HTTP 响应：够 `fetch` 用的那四样（status / headers / read / geturl）。"""

    def __init__(self, body: bytes = b"", status: int = 200,
                 ct: str = "text/plain; charset=utf-8",
                 url: str = "https://example.com/x"):
        self._body = body
        self.status = status
        self.headers = email.message.Message()
        if ct:
            self.headers["Content-Type"] = ct
        self._url = url

    def read(self, n=-1):
        return self._body[:n] if (n is not None and n >= 0) else self._body

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    """假网络层：`open()` 返回给定响应、或抛给定异常。"""

    def __init__(self, resp=None, err=None):
        self.resp, self.err, self.calls = resp, err, 0

    def open(self, req, timeout=None):
        self.calls += 1
        if self.err is not None:
            raise self.err
        return self.resp


class WebSearchTest(Base):
    """联网搜索（工具箱第十一节）。

    测的重点和别的工具不同：**它在不该搜的时候搜不搜**，
    以及黑盒那一半——拿得到的进 trace，拿不到的别假装拿得到。
    """

    def test_registered_as_read_only_with_the_discipline_in_the_prompt(self):
        t = TOOLS["web"]
        self.assertEqual(t["mode"], READ)
        # 「该不该搜」是分寸问题，档位拦不住——写在系统提示词的分寸块里
        # （同 DeepSeek Harness：description 只说是什么，规矩进 systemPrompt 的独立 section）
        self.assertNotIn("不要搜", t["description"])
        block = render_tools_block(["web"])
        self.assertIn("不要搜", block)

    def test_tools_block_only_lists_available_tools(self):
        # 没开的动作不进块——说了她也没法用，还会诱导她假装做了
        full = render_tools_block(tool_names())
        for name in tool_names():
            self.assertIn(f"`{name}`", full)
        no_web = render_tools_block([n for n in tool_names() if n != "web"])
        self.assertNotIn("`web`", no_web)
        self.assertEqual(render_tools_block(None), "")
        self.assertEqual(render_tools_block([]), "")

    def test_parse_anthropic_picks_text_queries_and_sources(self):
        data = {"content": [
            {"type": "server_tool_use", "name": "web_search",
             "input": {"query": "deepseek v4 联网搜索"}},
            {"type": "thinking", "thinking": "想了一下"},
            {"type": "web_search_tool_result", "content": [
                {"type": "web_search_result", "url": "https://github.com/x",
                 "title": "GitHub 页面"}]},
            {"type": "text", "text": "要点一二三"},
        ]}
        text, queries, sources = _parse_anthropic(data)
        self.assertEqual(text, "要点一二三")
        self.assertEqual(queries, ["deepseek v4 联网搜索"])
        self.assertEqual(sources, [{"url": "https://github.com/x",
                                    "title": "GitHub 页面"}])

    def test_search_returns_facts_and_what_was_searched(self):
        llm = _FakeLLM(data=_search_reply(text="要点：先复盘", query="面试 复盘 方法"))
        out = llm.search("面试复盘怎么做")
        self.assertTrue(out["ok"], out.get("error"))
        self.assertIn("先复盘", out["text"])
        self.assertEqual(out["queries"], ["面试 复盘 方法"])
        self.assertEqual(out["urls"], ["https://example.com/x"])

    def test_no_result_blocks_is_a_failure_not_a_prose_result(self):
        """**没有结果块 = 没搜**——模型没工具时照样会写文本（实测：「当前环境
        未提供可用的联网搜索工具」），那段自白绝不能当结果带出去。
        """
        llm = _FakeLLM(data={"content": [
            {"type": "text",
             "text": "当前环境未提供可用的联网搜索工具，无法执行所要求的搜索。"}]})
        out = llm.search("GitHub 某个仓库")
        self.assertFalse(out["ok"])
        self.assertEqual(out["text"], "")
        self.assertIn("没有真的", out["error"])

    def test_unsupported_probing_turns_it_off_for_good(self):
        """探测到没有就**永久关掉**（同工具调用那次的做法）。"""
        llm = _FakeLLM(err=urllib.error.HTTPError("u", 404, "Not Found", {}, None))
        self.assertFalse(llm.search("查个东西")["ok"])
        self.assertFalse(llm.search_capable())
        used = llm.calls
        self.assertFalse(llm.search("再查一次")["ok"])
        self.assertEqual(llm.calls, used, "关掉之后不该再发请求")

    def test_network_error_is_not_mistaken_for_unsupported(self):
        """误判的代价是**永久**的（要重启才恢复），所以网络抖动不算。"""
        llm = _FakeLLM(err=OSError("connection reset"))
        self.assertFalse(llm.search("查个东西")["ok"])
        self.assertTrue(llm.search_capable(), "网络抖动不能被记成永久不可用")

    def test_parse_openai_picks_annotations_and_usage(self):
        """OpenAI 形态（MiMo）的解析：正文 + `message.annotations` 的结构化引用。
        「真搜了没有」看 `usage.web_search_usage` 或 annotations（2026-09-22 加）。"""
        data = {"choices": [{"message": {
            "content": "要点一二三",
            "annotations": [
                {"type": "url_citation", "url": "https://a.example/x",
                 "title": "甲页", "site_name": "某站"},
                "not-a-dict",
                {"type": "url_citation", "url": "", "title": "空的跳过"},
            ]}}],
            "usage": {"web_search_usage": {"tool_usage": 3, "page_usage": 9}}}
        text, queries, sources = _parse_openai(data)
        self.assertEqual(text, "要点一二三")
        self.assertEqual(sources, [{"url": "https://a.example/x", "title": "甲页"}])
        self.assertEqual(queries, [], "这条通道不回报搜了什么词——调用方兜底")

    def test_parse_openai_no_search_is_a_failure(self):
        """没有 web_search_usage、也没有 annotations = 没搜（同 Anthropic 那条纪律）：
        文本还在，但 sources 置空——调用方据此判失败，不把自白当结果。"""
        text, _, sources = _parse_openai(
            {"choices": [{"message": {"content": "我直接答了，没搜"}}]})
        self.assertIn("直接答", text)
        self.assertEqual(sources, [])

    def test_parse_openai_bad_usage_is_not_a_crash(self):
        """usage 里是非数字（没在实测里见过）：当没搜，不崩。
        ——`search()` 现在也兜住通道的解析异常，但**别依赖那条兜底**：
        它给的是一句"读不懂这次的响应"，而这里判"没搜"更准。"""
        text, _, sources = _parse_openai(
            {"choices": [{"message": {"content": "x"}}],
             "usage": {"web_search_usage": {"tool_usage": "五次"}}})
        self.assertEqual(sources, [])

    def test_search_openai_channel_goes_through_chat(self):
        """`search.channel=openai`：走 chat 通道（**不读** search.endpoint），
        sources 来自 annotations，queries 用请求词兜底。"""
        cfgmod.CONFIG["search"]["channel"] = "openai"
        self.addCleanup(lambda: cfgmod.CONFIG["search"].update(channel="anthropic"))
        llm = _FakeChatLLM(data={"choices": [{"message": {
            "content": "要点", "annotations": [
                {"type": "url_citation", "url": "https://m.example/1",
                 "title": "页一"}]}}],
            "usage": {"web_search_usage": {"tool_usage": 2}}})
        out = llm.search("小米 MiMo 最近发布什么")
        self.assertTrue(out["ok"], out.get("error"))
        self.assertEqual(out["urls"], ["https://m.example/1"])
        self.assertEqual(out["queries"], ["小米 MiMo 最近发布什么"])
        self.assertEqual(out["sources"],
                         [{"url": "https://m.example/1", "title": "页一"}])

    def test_not_subscribed_is_not_a_permanent_shutdown(self):
        """MiMo 的「没开通」400（`webSearchEnabled`）**不能**记成"服务没这个能力"——
        开通后直接重试就该能用，不该逼人重启（误判的代价是永久的）。"""
        body = io.BytesIO(
            b'{"error": {"message": "web search tool found in the request body, '
            b'but webSearchEnabled is false"}}')
        err = urllib.error.HTTPError("u", 400, "Bad Request", {}, body)
        cfgmod.CONFIG["search"]["channel"] = "openai"
        self.addCleanup(lambda: cfgmod.CONFIG["search"].update(channel="anthropic"))
        llm = _FakeChatLLM(err=err)
        out = llm.search("查个东西")
        self.assertFalse(out["ok"])
        self.assertIn("开通", out["error"])
        self.assertTrue(llm.search_capable(), "没开通只是暂时的，不能永久关掉")

    def test_tool_executes_and_traces_what_was_searched(self):
        llm = _FakeLLM(data=_search_reply(text="论文要点", query="q1",
                                          url="https://arxiv.org/abs/1",
                                          title="某篇论文"))
        out = execute(self.store, "web", {"query": "某篇论文"},
                      {"emb": None, "llm": llm})
        self.assertTrue(out["ok"])
        self.assertIn("论文要点", out["detail"])
        self.assertIn("某篇论文", out["detail"])       # 标题进了素材（她能说出处）
        self.assertEqual(out["searched"]["urls"], ["https://arxiv.org/abs/1"])
        # 「搜了什么」和「打开过哪些页面」进 trace
        self.assertIn("q1", self.traces())

    def test_web_and_library_budgets_are_separate_lines(self):
        """库内与网络**两条线分开计**（2026-09-18 加 web_fetch 时改）。

        网络仍硬限 2、库内是提醒（2026-09-23）——分开计的意思不变：
        网络用掉名额，记忆库照查（那边只在返回里带提醒）。
        """
        llm = _FakeLLM(data=_search_reply(text="要点"))
        ctx = {"emb": None, "llm": llm}
        self.assertTrue(execute(self.store, "web", {"query": "x"}, ctx)["ok"])
        # 两条线互不挤占：网络用掉名额，库内那几次照样能用
        for q in ("y", "z", "w"):
            self.assertTrue(execute(self.store, "memory_search", {"query": q}, ctx)["ok"],
                            "网络的名额不该挤库内")
        out = execute(self.store, "memory_search", {"query": "v"}, ctx)
        self.assertTrue(out["ok"], "库内不再硬拒（2026-09-23）")
        self.assertIn("你判", out["detail"], "次数到线只提醒")

    def test_web_limit_is_still_two(self):
        """网络线**仍硬限 2**（2026-09-23 改限流时唯一没动的一条）：
        第三次被拒——它的理由是**成本**（真钱 + 慢 + 外部世界），不是"信不过她"。"""
        llm = _FakeLLM(data=_search_reply(text="要点"))
        ctx = {"emb": None, "llm": llm}
        for q in ("a", "b"):
            self.assertTrue(execute(self.store, "web", {"query": q}, ctx)["ok"])
        out = execute(self.store, "web", {"query": "c"}, ctx)
        self.assertFalse(out["ok"])
        self.assertIn("两回", out["detail"])

    def test_failure_says_not_found_not_that_it_does_not_exist(self):
        """工具失败 ≠ 世界上没有。这句要说清，否则她会把「没查到」说成「没有」。"""
        llm = _FakeLLM(err=urllib.error.HTTPError("u", 404, "Not Found", {}, None))
        out = execute(self.store, "web", {"query": "x"}, {"emb": None, "llm": llm})
        self.assertFalse(out["ok"])
        self.assertIn("没查到", out["detail"])

    def test_tool_says_not_found_when_the_server_did_not_search(self):
        """服务端没执行搜索时，她收到的是「没查到」——2026-09 的真事故：
        模型的自白被当成搜索结果，她说出「查到了：当前环境未提供可用的联网搜索工具」。
        """
        llm = _FakeLLM(data={"content": [
            {"type": "text", "text": "当前环境未提供可用的联网搜索工具。"}]})
        out = execute(self.store, "web", {"query": "x"}, {"emb": None, "llm": llm})
        self.assertFalse(out["ok"])
        self.assertIn("没查到", out["detail"])
        self.assertNotIn("联网搜索工具", out["detail"])


class SearchChannelTest(Base):
    """搜索通道表（`core/search.py`，2026-09-26）。

    这一组测的不是"搜得对不对"（那是上面 `WebSearchTest`），而是**那次重构要买到的东西**：
    换家 = 换配置、加形态 = 加一条描述。所以最值钱的两条是——
    ① 端点从哪来由**通道自己**声明（错用另一条的默认值 = 拿新家的 key 打旧家的端点）；
    ② 往表里注册一条描述，`llm.search()` **一行不改**就能走通。
    """

    def setUp(self):
        super().setUp()
        self._orig = dict(cfgmod.CONFIG["search"])
        self.addCleanup(self._restore)

    def _restore(self):
        cfgmod.CONFIG["search"].clear()
        cfgmod.CONFIG["search"].update(self._orig)

    def test_aliases_resolve_to_one_canonical_channel(self):
        """俗名只管认人，不产生第二条维护线：
        deepseek / messages / claude → anthropic，mimo / chat → openai。"""
        for alias in ("anthropic", "deepseek", "messages", "claude", "Anthropic"):
            ch, why = searchmod.resolve(alias)
            self.assertIsNotNone(ch, alias)
            self.assertEqual(ch.name, "anthropic", alias)
            self.assertEqual(why, "", alias)
        for alias in ("openai", "mimo", "chat", "chat-completions"):
            ch, _ = searchmod.resolve(alias)
            self.assertEqual(ch.name, "openai", alias)

    def test_unknown_channel_is_named_out_loud(self):
        """认不出**不回退默认**——静默回退的症状是"配置写着 A、实际走了 B"，
        比一次明确的失败贵得多。报错要带：写错的那个名字 + 可用的清单。"""
        ch, why = searchmod.resolve("kimi")
        self.assertIsNone(ch)
        self.assertIn("kimi", why)
        self.assertIn("anthropic", why)

    def test_every_channel_declares_where_its_endpoint_comes_from(self):
        """端点来源是**通道自己的属性**：DeepSeek 要求搜索与对话分开配、
        MiMo 就是同一个端点——两种接法都得表达得了，且不能串。"""
        self.assertEqual(searchmod.resolve("anthropic")[0].base_from, "search")
        self.assertEqual(searchmod.resolve("openai")[0].base_from, "llm")

    def test_anthropic_channel_goes_to_the_search_endpoint(self):
        """anthropic 通道：`search.endpoint` + `/messages`，头带 `x-api-key` 那套。"""
        cfgmod.CONFIG["search"]["channel"] = "anthropic"
        cfgmod.CONFIG["search"]["endpoint"] = "https://api.deepseek.com/anthropic/v1"
        llm = _FakeLLM(data=_search_reply())
        self.assertTrue(llm.search("x")["ok"])
        url, headers, payload = llm.last
        self.assertEqual(url, "https://api.deepseek.com/anthropic/v1/messages")
        self.assertEqual(headers.get("x-api-key"), "sk-test")
        self.assertEqual(payload["tools"][0]["type"], "web_search_20250305")

    def test_openai_channel_ignores_the_search_endpoint(self):
        """openai 通道：与对话同端（`llm.endpoint` + `/chat/completions`），
        **不读** `search.endpoint`——那条默认值是别人家的端点（拿新家的 key
        打旧家的端点，401；这条坑 2026-09-22 实测踩过）。"""
        cfgmod.CONFIG["search"]["channel"] = "openai"
        cfgmod.CONFIG["search"]["endpoint"] = "https://wrong.example/anthropic/v1"
        llm = _FakeChatLLM(data={"choices": [{"message": {
            "content": "要点", "annotations": [
                {"type": "url_citation", "url": "https://m.example/9",
                 "title": "页九"}]}}],
            "usage": {"web_search_usage": {"tool_usage": 1}}})
        self.assertTrue(llm.search("x")["ok"])
        url, headers, _ = llm.last
        self.assertEqual(url, "https://api.mimo.example/v1/chat/completions")
        self.assertNotIn("x-api-key", headers, "Anthropic 那套头不该出现在这条通道上")

    def test_a_new_channel_is_a_description_not_a_rewrite(self):
        """**这次重构的目的本身**：加一种新形态 = 往 `CHANNELS` 放一段描述
        （端点从哪来 / 请求怎么拼 / 响应怎么读），`llm.search()` 一行不改就能走通。
        下面这条"探针通道"就是一份最小描述。
        """
        def build(q, model, key):
            return {"X-Key": key}, {"model": model, "query": q}

        def parse(data):
            return data.get("answer", ""), ["表里写的词"], data.get("hits") or []

        searchmod.CHANNELS["probe"] = searchmod.Channel(
            name="probe", label="探针（测试用）", base_from="search",
            path="/probe", build=build, parse=parse)
        self.addCleanup(lambda: searchmod.CHANNELS.pop("probe", None))

        cfgmod.CONFIG["search"]["channel"] = "probe"
        cfgmod.CONFIG["search"]["endpoint"] = "https://probe.example/v1"
        llm = _FakeLLM(data={"answer": "要点", "hits": [
            {"url": "https://p.example/1", "title": "丙页"}]})
        out = llm.search("随便问问")
        self.assertTrue(out["ok"], out.get("error"))
        self.assertEqual(out["text"], "要点")
        self.assertEqual(out["urls"], ["https://p.example/1"])
        self.assertEqual(out["queries"], ["表里写的词"])
        url, headers, _ = llm.last
        self.assertEqual(url, "https://probe.example/v1/probe")
        self.assertEqual(headers, {"X-Key": "sk-test"})

    def test_a_channel_without_results_is_still_a_failure(self):
        """**通用纪律对所有通道一样**：没有结构化结果 = 没搜（哪怕正文写得像模像样）。
        它由主流程兜底判死，新通道不用自己记得这条。"""
        def build(q, model, key):
            return {}, {"q": q}

        def parse(data):
            return data.get("answer", ""), [], data.get("hits") or []

        searchmod.CHANNELS["probe2"] = searchmod.Channel(
            name="probe2", label="探针二（测试用）", base_from="search",
            path="/probe", build=build, parse=parse)
        self.addCleanup(lambda: searchmod.CHANNELS.pop("probe2", None))
        cfgmod.CONFIG["search"]["channel"] = "probe2"
        llm = _FakeLLM(data={"answer": "我查到了很多"})
        out = llm.search("x")
        self.assertFalse(out["ok"])
        self.assertIn("没有真的联网查", out["error"])

    def test_a_broken_new_channel_does_not_escape_the_main_flow(self):
        """通道是**扩展点**，而 `build` / `parse` 是新写的一条描述里最容易出错的地方；
        `search()` 的契约是**绝不抛**（工具那边只接得住返回值）。出格的响应
        （顶层不是对象、长成别的形状）该**指名报出来**，不是穿透出去——
        穿透出去工具侧只剩一句"没查到"，真因（表里那条描述想错了）就看不见了。
        """
        def build(q, model, key):
            return {"X-Key": key}, {"q": q}

        def parse(data):
            return data["content"][0]["text"], [], []      # 按 Anthropic 的形状读

        searchmod.CHANNELS["probe3"] = searchmod.Channel(
            name="probe3", label="探针三（测试用）", base_from="search",
            path="/probe", build=build, parse=parse)
        self.addCleanup(lambda: searchmod.CHANNELS.pop("probe3", None))
        cfgmod.CONFIG["search"]["channel"] = "probe3"
        cfgmod.CONFIG["search"]["endpoint"] = "https://probe.example/v1"
        out = _FakeLLM(data=[{"text": "假装是响应"}]).search("x")   # 顶层是数组
        self.assertFalse(out["ok"])
        self.assertIn("probe3", out["error"], "报错要指名是哪条通道")

    def test_a_channel_handing_back_half_baked_results_is_a_failure(self):
        """结果**成不成形**（`{url, title}`）是通用判据，不由通道自己记得：
        新通道忘了给 `url` 时，主流程判"没真搜"，而不是拿一份半成品去凑
        （`urls` 是从 `sources` 推的——形状不齐以前会在这儿炸）。"""
        def build(q, model, key):
            return {}, {"q": q}

        def parse(data):
            return "查到了", [], ["https://没有标题.example"]      # 不是 {url, title}

        searchmod.CHANNELS["probe4"] = searchmod.Channel(
            name="probe4", label="探针四（测试用）", base_from="search",
            path="/probe", build=build, parse=parse)
        self.addCleanup(lambda: searchmod.CHANNELS.pop("probe4", None))
        cfgmod.CONFIG["search"]["channel"] = "probe4"
        llm = _FakeLLM(data={})
        out = llm.search("x")
        self.assertFalse(out["ok"])
        self.assertIn("没有真的联网查", out["error"])

    def test_no_channel_configured_means_the_documented_default(self):
        """没配过（空名）= 缺省，不是错：走 `DEFAULT` 那条，且与配置默认值一致。"""
        self.assertEqual(cfgmod.CONFIG["search"]["channel"], searchmod.DEFAULT)
        ch, why = searchmod.resolve("")
        self.assertEqual(ch.name, searchmod.DEFAULT)
        self.assertEqual(why, "")


class WebFetchTest(Base):
    """取网页（工具箱第十一节的另一半）：他给链接时她读得动。

    测的重点是**边界**——它挡什么：本机 / 内网地址一律拒绝（她读不到本地
    服务）、二进制与不明类型不收、跨源重定向不跟、4xx 是结果不是错误。
    全程**不打真网络**：解析与代理都打桩，网络层是假 opener。
    """

    def setUp(self):
        super().setUp()
        self._orig_resolve = webfetch._resolve_public
        self._orig_proxy = webfetch.proxied
        webfetch._resolve_public = lambda host, port: (["93.184.216.34"], "")
        webfetch.proxied = lambda url: False
        self.addCleanup(lambda: setattr(webfetch, "_resolve_public", self._orig_resolve))
        self.addCleanup(lambda: setattr(webfetch, "proxied", self._orig_proxy))

    def _fetch(self, url, resp=None, err=None, **kw):
        op = _FakeOpener(resp=resp, err=err)
        return webfetch.fetch(url, opener=op, **kw), op

    def test_registered_as_read_only(self):
        self.assertEqual(TOOLS["web"]["mode"], READ)
        self.assertIn("`web`", render_tools_block(tool_names()))

    def test_bad_urls_are_refused_before_any_request(self):
        for bad in ("ftp://example.com/x", "https://user:pw@example.com/",
                    "没有地址", "https://example.com/" + "x" * 3000):
            out, op = self._fetch(bad)
            self.assertFalse(out["ok"], bad)
            self.assertEqual(op.calls, 0, f"{bad} 不该发起请求")

    def test_local_and_private_targets_are_blocked(self):
        """她读不到本地服务——127.0.0.1 上的奥拉马 / 语音 / 仪表盘都在那儿。"""
        for bad in ("http://127.0.0.1:11434/api/tags", "http://192.168.1.1/",
                    "http://10.0.0.5/", "http://[::1]/", "http://169.254.1.1/"):
            out, op = self._fetch(bad)
            self.assertFalse(out["ok"], bad)
            self.assertEqual(op.calls, 0, f"{bad} 不该发起请求")

    def test_resolve_rejects_local_hosts(self):
        """域名这条路也拦：解析结果里有一个非公共地址，整体拒绝。"""
        ips, err = self._orig_resolve("localhost", 80)
        self.assertEqual(ips, [])
        self.assertIn("本机或内网", err)

    def test_fetches_and_turns_html_into_readable_text(self):
        html = ("<html><head><style>body{}</style></head><body>"
                "<h1>标题</h1><p>正文&amp;段落</p><ul><li>甲</li></ul>"
                "<script>var secret = 1;</script></body></html>").encode("utf-8")
        out, _ = self._fetch("https://example.com/x",
                             resp=_FakeResponse(body=html, ct="text/html"))
        self.assertTrue(out["ok"])
        self.assertEqual(out["status"], 200)
        self.assertIn("# 标题", out["text"])
        self.assertIn("正文&段落", out["text"])
        self.assertIn("- 甲", out["text"])
        self.assertNotIn("var secret", out["text"], "script 里的不是内容")

    def test_4xx_is_a_result_not_an_error(self):
        """状态码是被抓资源的一部分——404 也是一种回答（同 harness）。"""
        h = email.message.Message()
        h["Content-Type"] = "text/html; charset=utf-8"
        err = urllib.error.HTTPError("https://example.com/x", 404, "Not Found",
                                     h, io.BytesIO("不见了".encode("utf-8")))
        out, _ = self._fetch("https://example.com/x", err=err)
        self.assertTrue(out["ok"], out.get("error"))
        self.assertEqual(out["status"], 404)
        self.assertIn("不见了", out["text"])

    def test_cross_origin_redirect_tells_her_to_call_again(self):
        err = urllib.error.HTTPError("https://example.com/x", 302, "Found", None, None)
        out, _ = self._fetch("https://example.com/x", err=err)
        self.assertFalse(out["ok"])
        self.assertIn("别的站点", out["error"])

    def test_redirect_error_carries_the_target_url(self):
        """跨源跳转的文案要**带上跳转目标**——"换最终那个地址"得有个地址可换
        （jsDelivr → fastly 节点这类跳转很常见，她拿着地址就能接着抓）。"""
        h = email.message.Message()
        h["Location"] = "https://elsewhere.example/final"
        err = urllib.error.HTTPError("https://example.com/x", 302, "Found", h, None)
        out, _ = self._fetch("https://example.com/x", err=err)
        self.assertFalse(out["ok"])
        self.assertIn("https://elsewhere.example/final", out["error"])

    def test_same_origin_is_scheme_host_and_port(self):
        """同源判定 = scheme + host + port 全同（端口按 scheme 补默认值）。"""
        self.assertTrue(webfetch._same_origin("https://example.com/a",
                                              "https://example.com/b"))
        self.assertFalse(webfetch._same_origin("http://example.com/",
                                               "https://example.com/"))
        self.assertFalse(webfetch._same_origin("https://example.com/",
                                               "https://other.com/"))
        self.assertFalse(webfetch._same_origin("https://example.com/",
                                               "https://example.com:8443/"))

    def test_binary_is_refused(self):
        out, _ = self._fetch("https://example.com/a.png",
                             resp=_FakeResponse(body=b"\x89PNG", ct="image/png"))
        self.assertFalse(out["ok"])
        self.assertIn("不是文本", out["error"])

    def test_long_body_is_truncated_with_a_note(self):
        out, _ = self._fetch("https://example.com/x",
                             resp=_FakeResponse(body=("字" * 200).encode("utf-8")),
                             max_chars=20)
        self.assertTrue(out["ok"])
        self.assertTrue(out["truncated"])
        self.assertIn("截断", out["text"])

    def test_base64_json_content_is_unwrapped(self):
        """GitHub 内容 API 的坑（2026-09-18 实证）：正文在 JSON 的 `content`
        字段里、base64 编码——不拆开她读到的就是编码后的乱码。"""
        import base64 as b64
        body = ('{"name":"README.md","size":9,"encoding":"base64","content":"'
                + b64.b64encode("# 标题\n正文".encode("utf-8")).decode() + '"}')
        out, _ = self._fetch("https://api.github.com/repos/x/y/readme",
                             resp=_FakeResponse(body=body.encode("utf-8"),
                                                ct="application/json"))
        self.assertTrue(out["ok"], out.get("error"))
        self.assertIn("# 标题", out["text"])
        self.assertNotIn("encoding", out["text"], "拆完不该还带着编码外壳")

    def test_base64_is_unwrapped_before_truncation(self):
        """拆包在截断之前——不然 max_chars 截的是 base64 串。"""
        import base64 as b64
        body = ('{"encoding":"base64","content":"'
                + b64.b64encode(("字" * 100).encode("utf-8")).decode() + '"}')
        out, _ = self._fetch("https://example.com/api",
                             resp=_FakeResponse(body=body.encode("utf-8"),
                                                ct="application/json"),
                             max_chars=20)
        self.assertTrue(out["ok"])
        self.assertTrue(out["truncated"])
        self.assertIn("字", out["text"])

    def test_plain_or_undecodable_json_is_left_alone(self):
        """普通 JSON、或解不出内容的 base64 壳 → 原样放行（不做猜测）。"""
        cases = [('{"name": "x", "count": 2}', '"count": 2'),
                 ('{"encoding": "base64", "content": "%%%"}', '"content": "%%%"'),
                 ('{"encoding": "base64", "content": ""}', '"content": ""')]
        for body, marker in cases:
            out, _ = self._fetch("https://example.com/api",
                                 resp=_FakeResponse(body=body.encode("utf-8"),
                                                    ct="application/json"))
            self.assertTrue(out["ok"], body)
            self.assertIn(marker, out["text"], body)

    def test_tool_executes_and_says_what_it_got(self):
        out = execute(self.store, "web", {"url": "https://example.com/x"},
                      {"web_opener": _FakeOpener(resp=_FakeResponse(body=b"hello"))})
        self.assertTrue(out["ok"])
        self.assertIn("hello", out["detail"])
        self.assertIn("web", self.traces())

    def test_missing_url_tells_her_the_shape(self):
        out = execute(self.store, "web", {}, {})
        self.assertFalse(out["ok"])
        self.assertIn("url", out["detail"])

    def test_search_then_fetch_is_allowed_but_capped(self):
        """「先搜 → 再抓」是正常两步流（harness 的引导），网络线给两个名额。
        （2026-09-24 合并后是**同一个工具调两次**。）"""
        llm = _FakeLLM(data=_search_reply(text="要点", url="https://example.com/x"))
        ctx = {"emb": None, "llm": llm,
               "web_opener": _FakeOpener(resp=_FakeResponse(body=b"page"))}
        self.assertTrue(execute(self.store, "web", {"query": "q"}, ctx)["ok"])
        self.assertTrue(execute(self.store, "web",
                                {"url": "https://example.com/x"}, ctx)["ok"])
        third = execute(self.store, "web", {"url": "https://example.com/y"}, ctx)
        self.assertFalse(third["ok"])
        self.assertIn("两回", third["detail"])


class SearchToolVisibilityTest(Base):
    """搜索关掉时（人关的，或探测到服务没有）**连工具都别塞给她**——
    留一个用不了的工具在提示词里，只会让她选错。

    2026-09-24 合并后 `web` = 搜 + 抓两个能力：**两个都关**才摘掉它；
    只关一边时工具留着（另一边还走得通）。
    """

    def _names(self, cap: str | None) -> list[str]:
        llm = _CaptureLLM()
        if cap:
            llm._search_cap = cap
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding())
        sess._generate([{"role": "user", "content": "hi"}], 500)
        return [t["function"]["name"] for t in (llm.seen or [])]

    def test_present_when_capable(self):
        self.assertIn("web", self._names(None))

    def test_kept_when_search_unsupported_but_fetch_is_on(self):
        """搜索探测失败不挡抓取——那是本机出网的事，和搜索端点没关系。"""
        self.assertIn("web", self._names("unsupported"))

    def test_dropped_when_both_sides_off(self):
        cfgmod.CONFIG["web"]["fetch_enabled"] = False
        cfgmod.CONFIG["search"]["enabled"] = False
        self.addCleanup(lambda: cfgmod.CONFIG["web"].update(fetch_enabled=True))
        self.addCleanup(lambda: cfgmod.CONFIG["search"].update(enabled=True))
        self.assertNotIn("web", self._names("unsupported"))

    def test_kept_when_only_fetch_switched_off(self):
        """只关抓取不该摘掉它——搜索还开着，那条路走得通。"""
        cfgmod.CONFIG["web"]["fetch_enabled"] = False
        self.addCleanup(lambda: cfgmod.CONFIG["web"].update(fetch_enabled=True))
        self.assertIn("web", self._names(None))


class RecallDimensionsTest(Base):
    """`memory_search` 的维度：**给得越多越准**——

    它们之间是**交集**，不是"多给一个就多一次机会"。
    （和唤醒注入是同一套线索思路：线索越多指向越窄。）
    """

    def _scene(self, title, text, when, entity=None):
        s = Scene(title=title, text=text, time_record=when)
        self.store.add_scene(s)
        if entity:
            link_entities(s.id, [{"name": entity, "kind": "pet",
                                  "relation": "家里的宠物"}], self.store)
        return s

    def test_entity_is_exact_and_narrows(self):
        a = self._scene("猫不吃东西", "咪咪不吃", "2026-09-10 08:00:00", "咪咪")
        self._scene("狗不吃东西", "旺财不吃", "2026-09-11 08:00:00", "旺财")

        hit = execute(self.store, "memory_search",
                      {"query": "不吃", "entity": "咪咪"}, {"emb": None})
        self.assertIn(a.id, hit["detail"])
        self.assertNotIn("旺财", hit["detail"], "实体是精确匹配，不该带出别的")

    def test_unknown_entity_matches_nothing(self):
        self._scene("猫不吃东西", "咪咪不吃", "2026-09-10 08:00:00", "咪咪")
        out = execute(self.store, "memory_search",
                      {"query": "不吃", "entity": "查无此人"}, {"emb": None})
        self.assertIn("没有符合", out["detail"])

    def test_when_narrows_to_that_day(self):
        a = self._scene("甲", "面试没过", "2026-09-10 08:00:00")
        b = self._scene("乙", "面试没过", "2026-09-12 08:00:00")

        out = execute(self.store, "memory_search",
                      {"query": "面试", "when": "2026-09-12"}, {"emb": None})
        self.assertIn(b.id, out["detail"])
        self.assertNotIn(a.id, out["detail"], "日期是交集，不是提示")

    def test_chinese_date_is_normalized(self):
        """她写日期的样式不固定——只做**格式**规整，不做语义换算。"""
        a = self._scene("甲", "面试没过", "2026-09-12 08:00:00")
        out = execute(self.store, "memory_search",
                      {"query": "面试", "when": "2026年9月12日"}, {"emb": None})
        self.assertIn(a.id, out["detail"])

    def test_relative_time_is_refused_not_ignored(self):
        """「上个月」算不出日期——**明确拒绝**，不能当成"没给"静默放过。"""
        self._scene("甲", "面试没过", "2026-09-12 08:00:00")
        out = execute(self.store, "memory_search",
                      {"query": "面试", "when": "上个月"}, {"emb": None})
        self.assertFalse(out["ok"])
        self.assertIn("具体日期", out["detail"])

    def test_no_dimension_lists_recent(self):
        """**不给维度 = 列最近的**（2026-09-23 合并后）：四层各摊几条——
        "你都记了些什么"由此自然支持（旧行为是报错要维度）。"""
        a = self._scene("甲", "面试没过", "2026-09-12 08:00:00")
        out = execute(self.store, "memory_search", {}, {"emb": None})
        self.assertTrue(out["ok"])
        self.assertIn("【最近的场景】", out["detail"])
        self.assertIn(a.id, out["detail"])
        self.assertIn("【画像】", out["detail"])
        self.assertIn("【未了结的事】", out["detail"])


class ListScenesTest(Base):
    """`memory_search`——**看得见自己在存什么**。

    她的原话：「recall 是查，得先知道问什么才查得到；删的钥匙给了，
    可钥匙上没编号，我还是盲删；而这个动作不可撤回。」
    （「编号刻意不进注入」的旧立场 **2026-09-21 作废**——见设计稿 A 条：
    既然允许她说编号，她的手上就得有编号。）
    """

    def _scene(self, title, when):
        s = Scene(title=title, text="理解", time_record=when)
        self.store.add_scene(s)
        return s

    def test_lists_ids(self):
        a = self._scene("甲", "2026-09-10 08:00:00")
        out = execute(self.store, "memory_search", {}, {"emb": None})
        self.assertTrue(out["ok"])
        self.assertIn(a.id, out["detail"], "拿不到编号就只能盲删")

    def test_when_narrows(self):
        self._scene("甲", "2026-09-10 08:00:00")
        b = self._scene("乙", "2026-09-12 08:00:00")
        out = execute(self.store, "memory_search", {"when": "2026-09-12"}, {"emb": None})
        self.assertIn(b.id, out["detail"])
        self.assertNotIn("甲", out["detail"])

    def test_refuses_relative_time(self):
        out = execute(self.store, "memory_search", {"when": "上个月"}, {"emb": None})
        self.assertFalse(out["ok"])
        self.assertIn("具体日期", out["detail"])


class ListMemosTest(Base):
    """`memory_search`——**说「那件事」之前先查这个**（设计稿 D 条 3，2026-09-21）。

    起因：她说"那件事你一句没提"，被追问时手上没有对象——查到卡也认不出
    （07:04 事故）。这一份把「我手上挂着哪几件」直接摆到眼前：编号 + 内容 + 谁欠的。
    """

    def _memo(self, content, kind=MEMO_USER_TASK, status=MEMO_PENDING, **kw):
        m = Memo(content=content, kind=kind, status=status, **kw)
        self.store.add_memo(m)
        return m

    def test_groups_steps_under_one_head(self):
        """同一件事的多步挂在一个组头下（2026-09-22）——她说「那件事」时
        要看得出"这是一件事的几步"，而不是几条不相干的条目。"""
        self._memo("写开源脱敏脚本", group_name="开源准备")
        self._memo("跑一周测试，观察 BUG", group_name="开源准备")
        self._memo("独立的一件事")
        out = execute(self.store, "memory_search", {}, self.ctx)
        d = out["detail"]
        self.assertIn("「开源准备」（2 步）", d)
        self.assertIn("  - ", d, "组内步骤缩进一层")
        self.assertIn("独立的一件事", d)

    def test_lists_ids_content_and_who(self):
        a = self._memo("用户提供新闻底稿", kind=MEMO_USER_TASK)
        b = self._memo("air 去搜那条新闻", kind=MEMO_AIR_PROMISE)
        out = execute(self.store, "memory_search", {}, self.ctx)
        self.assertTrue(out["ok"])
        self.assertIn(a.id, out["detail"])
        self.assertIn(b.id, out["detail"])
        self.assertIn("用户说的事", out["detail"])
        self.assertIn("你答应的事", out["detail"])

    def test_closed_are_gone_raised_is_marked(self):
        """关掉的不列；提过一次的列出来但标一下。"""
        closed = self._memo("已经做完的事")
        self.store.close_memo(closed.id)
        raised = self._memo("提过一回的事", status=MEMO_RAISED)
        out = execute(self.store, "memory_search", {}, self.ctx)
        self.assertNotIn(closed.id, out["detail"])
        self.assertIn(raised.id, out["detail"])
        self.assertIn("提过一次", out["detail"])

    def test_empty_is_honest(self):
        out = execute(self.store, "memory_search", {"layer": "memo"}, self.ctx)
        self.assertTrue(out["ok"])
        self.assertIn("没有", out["detail"])


class MemorySuiteTest(Base):
    """检索三件套（2026-09-23 合并落地，工具箱稿 §3.1）：**一个介质一个工具**。

    这里覆盖**合并进来的新能力**：layer 过滤 / id 直查 / 非 S1 引导 / 血缘两条路；
    场景检索那半在 `RecallDimensionsTest` / `ListScenesTest`（改名前就有的那批）。
    """

    def _scene(self, title="被组长当众批评", text="被批评后想离开"):
        s = Scene(title=title, text=text)
        self.store.add_scene(s)
        return s

    def _profile(self, sid, statement="受挫后倾向离开"):
        p = Profile(id="S3-0001", topic="用户·压力", statement=statement,
                    status=PROFILE_ESTABLISHED, evidence=1, sources=[sid])
        self.store.add_profile(p)
        return p

    def test_search_by_layer(self):
        """layer 过滤：给哪层只出哪层（认中英文别名）。"""
        s = self._scene()
        self._profile(s.id)
        m = Memo(content="面试结果", kind=MEMO_USER_TASK)
        self.store.add_memo(m)

        out = execute(self.store, "memory_search", {"layer": "s3"}, self.ctx)
        self.assertIn("【画像】", out["detail"])
        self.assertNotIn("【未了结的事】", out["detail"])
        # pending 的也要列出来（2026-09-24 检查修）：原来默认只取 established，
        # 与「已立与待验证都列」不符——她问「你对我有什么猜测」时会看不见
        pend = Profile(id="S3-0009", topic="用户·表达方式", statement="只在深夜长谈",
                       status=PROFILE_PENDING, evidence=1)
        self.store.add_profile(pend)
        out = execute(self.store, "memory_search", {"layer": "s3"}, self.ctx)
        self.assertIn(pend.id, out["detail"])
        self.assertIn("只是猜测", out["detail"])
        out = execute(self.store, "memory_search", {"layer": "memo"}, self.ctx)
        self.assertIn(m.id, out["detail"])
        self.assertNotIn("【画像】", out["detail"])
        out = execute(self.store, "memory_search", {"layer": "s1"}, self.ctx)
        self.assertIn(s.id, out["detail"])
        self.assertNotIn("【画像】", out["detail"])
        # ⚠️ 干净 ctx：库内检索"一轮 3 次"是硬限，这一条是第 4 次调用
        out = execute(self.store, "memory_search", {"层": "画像"}, {"emb": None})
        self.assertIn("【画像】", out["detail"], "认中文别名（防错第 5 条）")

        # 内容条件 × 非场景层：**明确引导，不静默丢条件**（2026-09-24 晚修）
        out = execute(self.store, "memory_search", {"layer": "s2", "query": "记忆"},
                      self.ctx)
        self.assertFalse(out["ok"])
        self.assertIn("不能按内容搜", out["detail"])
        out = execute(self.store, "memory_search", {"layer": "s3", "entity": "谁"},
                      self.ctx)
        self.assertFalse(out["ok"], "S3 也不能按实体搜")
        self.assertIn("topic", out["detail"], "引导要指明正路")
        out = execute(self.store, "memory_search",
                      {"layer": "memo", "when": "2026-09-20"}, self.ctx)
        self.assertFalse(out["ok"], "备忘不能按日期搜")
        # 对照：场景层带条件照旧走检索（不受上面那道闸影响）
        out = execute(self.store, "memory_search", {"layer": "s1", "query": "批评"},
                      self.ctx)
        self.assertTrue(out["ok"])

    def test_search_by_id(self):
        """id 直查：S1 出场景行、S3 出画像行、M 出备忘行。"""
        s = self._scene()
        p = self._profile(s.id)
        m = Memo(content="面试结果", kind=MEMO_USER_TASK)
        self.store.add_memo(m)

        out = execute(self.store, "memory_search", {"id": s.id}, self.ctx)
        self.assertIn(s.id, out["detail"])
        out = execute(self.store, "memory_search", {"id": p.id}, self.ctx)
        self.assertIn(p.statement, out["detail"])
        out = execute(self.store, "memory_search", {"id": m.id}, self.ctx)
        self.assertIn(m.content, out["detail"])

    def test_expand_non_s1(self):
        """给非 S1 编号 = **展开它自己**（2026-09-24 合并后）——
        画像出依据链，不再往别的工具引导（`memory_read` 已并入 `memory_search`）。"""
        s = self._scene()
        p = self._profile(s.id)

        out = execute(self.store, "memory_search", {"id": p.id}, self.ctx)
        self.assertTrue(out["ok"])
        self.assertIn("形成时的依据", out["detail"])

        # S1 + raw：再下钻原话（旧参数名 `scene_id` 仍认——别名）
        s2 = Scene(title="有原文", text="摘要",
                   time_event="2026-09-20 09:00:00",
                   time_record="2026-09-20 09:00:00")
        self.store.add_scene(s2)
        self.store.add_raw(Raw(scene_id=s2.id, content="用户: 原话在"),
                           on_date="2026-09-20")
        out = execute(self.store, "memory_search",
                      {"scene_id": s2.id, "raw": True}, {"emb": None})
        self.assertTrue(out["ok"])
        self.assertIn("原话在", out["detail"])

    def test_layer_s1_with_and_without_when(self):
        """`layer=s1` 的两条支路：不给条件 = 列场景；给日期 = 按日期筛。

        （2026-09-23 检查补：这是新加的分支，先前没人看着——它曾漏掉、
        落到"列全层"去，是检查时抓出来的。）
        """
        a = Scene(title="甲", text="理解", time_record="2026-09-12 08:00:00")
        b = Scene(title="乙", text="理解", time_record="2026-09-20 08:00:00")
        self.store.add_scene(a)
        self.store.add_scene(b)
        self._profile(a.id)

        out = execute(self.store, "memory_search", {"layer": "s1"}, self.ctx)
        self.assertIn("【最近的场景】", out["detail"])
        self.assertNotIn("【画像】", out["detail"], "给了层就只出这层")

        out = execute(self.store, "memory_search",
                      {"layer": "s1", "when": "2026-09-12"}, {"emb": None})
        self.assertIn(a.id, out["detail"])
        self.assertNotIn(b.id, out["detail"], "日期筛生效")

    def test_expand_both_directions(self):
        """展开两条路：S3 → 依据链；S1 → 反向血缘（谁引用它）。"""
        s = self._scene()
        p = self._profile(s.id)

        out = execute(self.store, "memory_search", {"id": p.id}, self.ctx)
        self.assertTrue(out["ok"])
        self.assertIn("形成时的依据", out["detail"])

        out = execute(self.store, "memory_search", {"id": s.id}, self.ctx)
        self.assertTrue(out["ok"])
        self.assertIn(p.id, out["detail"], "反向血缘要说出谁引用了它")


class CloseMemoTest(Base):
    """`close_memo`——用户给了结果就关；**判不准别关**（关掉她就不再主动提）。"""

    def test_closes_and_reports(self):
        m = Memo(content="面试", kind=MEMO_USER_TASK)
        self.store.add_memo(m)
        out = execute(self.store, "close_memo", {"memo_id": m.id}, self.ctx)
        self.assertTrue(out["ok"])
        self.assertEqual(self.store.get_memo(m.id).status, MEMO_CLOSED)
        self.assertIn("面试", out["detail"])

    def test_wrong_id_points_at_the_right_tool(self):
        out = execute(self.store, "close_memo", {"memo_id": "M-9999"}, self.ctx)
        self.assertFalse(out["ok"])
        self.assertIn("memory_search", out["detail"])


class ThreeLayerMemoryToolsTest(Base):
    """改与删的**三层同一套**（2026-09-24，工具箱稿 §3.3 / §3.4）：

    她一个入口（`forget_memory` / `revise_memory`），编号给 S1 / S2 / S3 都认——
    只提议、只列出来；**动手的永远是人**（确认条）。
    （原 `reject_profile` 并入 `forget_memory`："他明说不对"曾走自主档，
    三层统一后都是"她列出来 → 他点"。）
    """

    def _profile(self, pid: str = "S3-0001"):
        p = Profile(id=pid, topic="用户·被评价的反应", statement="遇到批评会想退出",
                    status=PROFILE_ESTABLISHED, evidence=3)
        self.store.add_profile(p)
        return p

    def _summary(self, sid: str = "S2-0001"):
        s2 = Summary(id=sid, topic="用户·被评价的反应", text="遇到批评会先退开")
        self.store.add_summary(s2)
        return s2

    def _scene(self, sid: str = "S1-0001"):
        s = Scene(id=sid, title="被组长批评", text="被批评后想离开")
        self.store.add_scene(s)
        return s

    def test_forget_accepts_all_three_layers(self):
        """一个入口收三种编号：S1 / S2 / S3 都进同一条提议。"""
        s, s2, p = self._scene(), self._summary(), self._profile()
        ctx = {"confirm_channel": True}
        out = execute(self.store, "forget_memory",
                      {"id": f"{s.id},{s2.id},{p.id}"}, ctx)

        self.assertTrue(out["ok"], "三层都认——不再有「画像不走这条路」")
        self.assertEqual(len(ctx["proposals"][0]["ids"]), 3)
        self.assertIsNotNone(self.store.get_scene(s.id), "提议阶段一个字都不动")
        self.assertIsNotNone(self.store.get_summary(s2.id))
        self.assertIsNotNone(self.store.get_profile(p.id))

    def test_forget_proposal_carries_titles(self):
        """提议里要带"一眼能认出"的标题（他核对用）——三层各有各的取法。"""
        p = self._profile()
        ctx = {"confirm_channel": True}
        execute(self.store, "forget_memory", {"id": p.id}, ctx)
        self.assertIn("遇到批评会想退出", ctx["proposals"][0]["title"])

    def test_aliases_and_old_param_names_still_work(self):
        """参数认别名：旧名字 `scene_id`、以及「场景」这类都照认（`_pick`）。"""
        p = self._profile("S3-0002")
        ctx = {"confirm_channel": True}
        out = execute(self.store, "forget_memory", {"场景": p.id}, ctx)
        self.assertTrue(out["ok"], "「场景」这类别名也要认")

    def test_revise_can_target_a_scene_field(self):
        """改标签（2026-09-24，工具箱稿 §七）：`field` 认中文名，提议带上它；
        不认的字段**拒绝**（拒绝而不是静默丢弃——静默丢弃会让人以为改成了）。"""
        s = Scene(id="S1-0001", title="被批评", text="被批评后想离开")
        self.store.add_scene(s)
        ctx = {"confirm_channel": True}

        out = execute(self.store, "revise_memory",
                      {"id": s.id, "field": "主题", "text": "用户·压力"}, ctx)
        self.assertTrue(out["ok"])
        self.assertEqual(ctx["proposals"][0]["field"], "topic")

        bad = execute(self.store, "revise_memory",
                      {"id": s.id, "field": "心情", "text": "x"}, ctx)
        self.assertFalse(bad["ok"], "不认的字段要拒绝，别静默丢弃")
        self.assertIn("不认的字段", bad["detail"])
        self.assertIn("主题", bad["detail"], "错误里要说清这层能改什么")

        # 字段名**按层认**："陈述"是画像的词，到场景来不认（不静默按 text 改）
        bad2 = execute(self.store, "revise_memory",
                       {"id": s.id, "field": "陈述", "text": "x"}, ctx)
        self.assertFalse(bad2["ok"])

    def test_revise_topics_across_all_three_layers(self):
        """**三层都能改标签**（2026-09-24 晚）：摘要 / 画像也能改主题——
        各自的主文本（叙述 / 陈述）是默认字段，给「主题」就是改归类。"""
        s2 = self._summary()
        p = self._profile("S3-0002")
        ctx = {"confirm_channel": True}

        out = execute(self.store, "revise_memory",
                      {"id": s2.id, "field": "主题", "text": "用户·沟通"}, ctx)
        self.assertTrue(out["ok"], "摘要能改主题")
        self.assertEqual(ctx["proposals"][0]["field"], "topic")

        out = execute(self.store, "revise_memory",
                      {"id": p.id, "field": "主题", "text": "用户·边界"}, ctx)
        self.assertTrue(out["ok"], "画像能改主题")
        self.assertEqual(ctx["proposals"][0]["field"], "topic")

        # 默认字段 = 各层的主文本（她的旧写法：不给 field）
        execute(self.store, "revise_memory",
                {"id": s2.id, "text": "换个说法"}, ctx)
        self.assertEqual(ctx["proposals"][-1]["field"], "text")
        execute(self.store, "revise_memory",
                {"id": p.id, "text": "遇到批评会先退开再处理"}, ctx)
        self.assertEqual(ctx["proposals"][-1]["field"], "statement")

    def test_search_by_topic(self):
        """按主题找（2026-09-24 晚）：跨三层、主附都算、子串匹配——S2 / S3 的「搜」。"""
        s = self._scene()
        s2 = self._summary()
        p = self._profile()
        self.store.set_scene_fields(s.id, {"topic": "用户·压力"})       # S1：单主题
        self.store.set_summary_topics(s2.id, ["用户·压力", "用户·工作"])
        self.store.set_profile_topics(p.id, ["用户·压力"])

        out = execute(self.store, "memory_search", {"topic": "压力"}, {})
        self.assertTrue(out["ok"])
        for sid in (s.id, s2.id, p.id):
            self.assertIn(sid, out["detail"], f"{sid} 应该被主题搜到")

        out = execute(self.store, "memory_search", {"topic": "工作"}, {})
        self.assertIn(s2.id, out["detail"], "附加主题也要命中")

        # topic × layer 可组合（2026-09-24 检查修）：给了层就只列那一层
        out = execute(self.store, "memory_search", {"topic": "压力", "layer": "s2"}, {})
        self.assertIn(s2.id, out["detail"])
        self.assertNotIn(s.id, out["detail"], "给了 layer=s2 就不该再列场景")

        out = execute(self.store, "memory_search", {"topic": "没这个主题"}, {})
        self.assertTrue(out["ok"], "找不到不是错误——如实说没有")
        self.assertIn("没有", out["detail"])

    def test_revise_topic_bounds(self):
        """主题个数（2026-09-24 晚）：S2 / S3 是 1-3 个；S1 是单值——越界都拒绝。"""
        s2 = self._summary()
        ctx = {"confirm_channel": True}

        ok = execute(self.store, "revise_memory",
                     {"id": s2.id, "field": "主题", "text": "用户·压力, 用户·工作"}, ctx)
        self.assertTrue(ok["ok"])
        self.assertEqual(ctx["proposals"][-1]["field"], "topic")

        bad = execute(self.store, "revise_memory",
                      {"id": s2.id, "field": "主题", "text": "a,b,c,d"}, ctx)
        self.assertFalse(bad["ok"], "4 个要拒绝（不静默截断）")

        s = self._scene()
        bad2 = execute(self.store, "revise_memory",
                       {"id": s.id, "field": "主题", "text": "a,b"}, ctx)
        self.assertFalse(bad2["ok"], "S1 的主题是单值（聚合分组的键）")

    def test_revise_accepts_all_three_layers(self):
        """改也是三层同一套：`revise_memory` 收 S1 / S2 / S3。"""
        s2 = self._summary()
        ctx = {"confirm_channel": True}
        out = execute(self.store, "revise_memory",
                      {"id": s2.id, "text": "改成新的叙述"}, ctx)
        self.assertTrue(out["ok"])
        self.assertEqual(ctx["proposals"][0]["ids"], [s2.id])

    def test_missing_id_says_the_shape(self):
        """缺编号时说清形状——但它得先过确认通道（需确认档的老规矩）。"""
        out = execute(self.store, "forget_memory", {}, {"confirm_channel": True})
        self.assertFalse(out["ok"])
        self.assertIn("S1-", out["detail"], "缺编号时说清形状")

    def test_without_confirm_channel_it_refuses(self):
        """**没有确认通道 = 拒绝**（失败即拒绝）——她不能悄悄删东西。"""
        p = self._profile()
        out = execute(self.store, "forget_memory", {"id": p.id}, {"emb": None})
        self.assertFalse(out["ok"])
        self.assertIsNotNone(self.store.get_profile(p.id), "一个字都没动")

    def test_leaves_a_tool_trace(self):
        """提议照样留痕——她的每次动作都可查（和界面那条路同一待遇）。"""
        p = self._profile()
        execute(self.store, "forget_memory", {"id": p.id}, self.ctx)
        self.assertIn(p.id, self.traces("工具"))


class ReadRawTest(Base):
    """下钻原话（`memory_search` 带 `raw=true`）——引用原话前对一次（D2a，2026-09-21）。

    走「经场景编号下钻」的口（`store.get_raws_by_scene` 是唯一入口）；
    07:04 她的失败正是"查到卡片、读不到原文"。
    """

    def _scene_with_raw(self, content="用户: 你说更凉的东西是什么\nair: 缺两块料：一是底稿"):
        s = Scene(title="断层", text="摘要", time_event="2026-09-20 09:50:00")
        self.store.add_scene(s)
        self.store.add_raw(Raw(scene_id=s.id, content=content),
                           on_date=(s.time_record or "")[:10])
        return s

    def test_reads_the_original_words(self):
        s = self._scene_with_raw()
        out = execute(self.store, "memory_search",
                      {"scene_id": s.id, "raw": True}, self.ctx)
        self.assertTrue(out["ok"])
        self.assertIn("底稿", out["detail"])

    def test_no_raw_says_so_instead_of_pretending(self):
        """没有原文就如实说——**不能拿摘要冒充原话**。"""
        s = Scene(title="只有摘要", text="一句话")
        self.store.add_scene(s)
        out = execute(self.store, "memory_search",
                      {"scene_id": s.id, "raw": True}, self.ctx)
        self.assertTrue(out["ok"])
        self.assertIn("没留下原文", out["detail"])

    def test_unknown_scene_is_refused(self):
        out = execute(self.store, "memory_search",
                      {"scene_id": "S1-9999", "raw": True}, self.ctx)
        self.assertFalse(out["ok"])
        self.assertIn("没有", out["detail"])


class QuotaGroupsTest(Base):
    """限流的**现在**（2026-09-23 改）：只有网络线是硬限，库内 / 下钻是提醒。

    为什么撤掉那两条硬限：**"查了几次"不是"该不该停"的判据**——实测把
    "查一次 → 读一段原文"这种正常两步流掐死过（她连读第二条原文被拒两次）。
    够不够回答**由她判**（提示词里"够了就停"是主判据）；次数到阈值只附一句提醒。
    仍然硬拦的是**判据闸**（都是"做了也白做"）：同一个词再查、这批结果全看过。
    （网络线硬限没动——真钱 + 慢 + 外部世界，成本账不是判据账；它要真联网
    才能测，这里不覆盖，回归由 `WebSearchTest` 那组守着。）
    """

    def test_library_search_is_not_hard_limited(self):
        """库内检索**不再**因为次数被拒——从第 3 次起在返回里带提醒。"""
        ctx = {"emb": None}
        for i in range(4):
            out = execute(self.store, "memory_search", {}, ctx)
            self.assertTrue(out["ok"], f"第 {i + 1} 次不该被拒（旧行为是第 4 次拒）")
        self.assertIn("你已经查了 4 次", out["detail"], "第 3 次起要附提醒")
        self.assertIn("你判", out["detail"], "提醒是把判断交回给她")

    def test_drill_is_not_hard_limited(self):
        """下钻读原话同样不再硬拒——第 2 段起提醒（它更贵）。

        （2026-09-24 合并后：下钻 = `memory_search` 带 `raw=true`，
        计数在**真读到原话**时记——按实际发生，不按工具名。）
        """
        s = Scene(title="甲", text="理解")
        self.store.add_scene(s)
        self.store.add_raw(Raw(scene_id=s.id, content="用户: 甲的原话"), on_date="")
        ctx = {"emb": None}

        out = execute(self.store, "memory_search", {"id": s.id, "raw": True}, ctx)
        self.assertTrue(out["ok"])
        self.assertNotIn("你判", out["detail"], "第 1 段不提醒")

        out = execute(self.store, "memory_search", {"id": s.id, "raw": True}, ctx)
        self.assertTrue(out["ok"], "第 2 次不该被拒（旧行为就是拒这一次）")
        self.assertIn("读过 2 段原话", out["detail"])

    def test_same_query_is_still_blocked(self):
        """判据闸没变：**同一个词再查**照旧被拒（结果一样，做了也白做）。"""
        ctx = {"emb": None}
        self.assertTrue(execute(self.store, "memory_search",
                                {"query": "面试"}, ctx)["ok"])
        out = execute(self.store, "memory_search", {"query": "面试"}, ctx)
        self.assertFalse(out["ok"])
        self.assertIn("换个词", out["detail"])

    def test_note_never_attaches_to_write_tools(self):
        """次数提醒只出现在**检索类**的返回里。

        （2026-09-23 检查修：原来漏了档位检查——她查过几次之后，连
        `close_memo` 这类写动作的返回都被附上"你已经查了 3 次"。）
        """
        m = Memo(content="面试", kind=MEMO_USER_TASK)
        self.store.add_memo(m)
        ctx = {"emb": None, "confirm_channel": True}
        for q in ("甲", "乙", "丙"):
            execute(self.store, "memory_search", {"query": q}, ctx)
        self.assertIn("你判", execute(self.store, "memory_search",
                                      {"query": "丁"}, ctx)["detail"],
                      "检索类该有提醒")

        out = execute(self.store, "close_memo", {"memo_id": m.id}, ctx)
        self.assertNotIn("你判", out["detail"], "写类工具不该被附检索提醒")


class ThreeLayerConfirmTest(Base):
    """确认条的三层分派（2026-09-24，工具箱稿 §3.4）：**一个入口、按编号前缀走各自的路**。

    S1 场景 / S2 摘要 / S3 画像——流程与后果一致；"要么全删、要么不删"是**跨层**的。
    """

    def _scene(self) -> Scene:
        s = Scene(title="甲", text="x")
        self.store.add_scene(s)
        return s

    def _summary(self) -> Summary:
        s2 = Summary(topic="用户·X", text="叙述")
        self.store.add_summary(s2)
        return s2

    def _profile(self) -> Profile:
        p = Profile(topic="用户·Y", statement="判断", status=PROFILE_ESTABLISHED, evidence=3)
        self.store.add_profile(p)
        return p

    def test_delete_mixed_layers(self):
        s, s2, p = self._scene(), self._summary(), self._profile()
        out = delete_by_layer(self.store, [s.id, s2.id, p.id])

        self.assertTrue(out["ok"])
        self.assertEqual(len(out["ids"]), 3)
        self.assertIsNone(self.store.get_scene(s.id))
        self.assertIsNone(self.store.get_summary(s2.id))
        self.assertIsNone(self.store.get_profile(p.id))

    def test_one_missing_none_touched(self):
        """**要么全删、要么不删**跨层有效：有一个找不到就一条都不动。"""
        s = self._scene()
        out = delete_by_layer(self.store, [s.id, "S2-9999"])
        self.assertFalse(out["ok"])
        self.assertIsNotNone(self.store.get_scene(s.id), "别的层也一条不动")

    def test_unknown_prefix_is_refused(self):
        out = delete_by_layer(self.store, ["X-0001"])
        self.assertFalse(out["ok"])
        self.assertIn("认不出", out["detail"])

    def test_archive_dispatch(self):
        s, s2, p = self._scene(), self._summary(), self._profile()
        out = archive_by_layer(self.store, [s.id, s2.id, p.id])

        self.assertTrue(out["ok"])
        self.assertTrue(self.store.get_scene(s.id).archived)
        self.assertTrue(self.store.get_summary(s2.id).archived)
        self.assertEqual(self.store.get_profile(p.id).invalidated_by, "archive")
        # 三层都不再"当前有效"——但数据全在
        self.assertIsNotNone(self.store.get_scene(s.id))
        self.assertIsNotNone(self.store.get_summary(s2.id))
        self.assertIsNotNone(self.store.get_profile(p.id))

    def test_update_dispatch_by_layer(self):
        """S2 原地改（留痕）；S1 走老的原地改——**同一入口、按层分派**。"""
        s, s2 = self._scene(), self._summary()
        self.assertTrue(update_by_layer(self.store, s2.id, "新的叙述")["ok"])
        self.assertEqual(self.store.get_summary(s2.id).text, "新的叙述")
        self.assertTrue(update_by_layer(self.store, s.id, "新的理解")["ok"])
        self.assertEqual(self.store.get_scene(s.id).text, "新的理解")

    def test_confirm_path_passes_canonical_field(self):
        """确认条路径回归（2026-09-24 夜核对修）：提议里存的 `field` 是**英文规范键**
        （S3 默认 `statement`），`chat.confirm` 把它原样递给 `update_by_layer`——
        别名表认不回自己的键，S3 的陈述经确认条就**永远改不成**（实测复现过：
        「不认的字段「statement」」）。别名表补规范键后，三层都要通。"""
        s, s2, p = self._scene(), self._summary(), self._profile()
        for iid, new in ((s.id, "新的理解"), (s2.id, "新的叙述"), (p.id, "新的陈述")):
            ctx = {"confirm_channel": True}
            out = execute(self.store, "revise_memory", {"id": iid, "text": new}, ctx)
            self.assertTrue(out["ok"], out.get("detail"))
            prop = ctx["proposals"][0]
            res = update_by_layer(self.store, prop["scene_id"], prop["text"],
                                  prop.get("field") or "text")
            self.assertTrue(res["ok"], f"{iid} 经确认条改不成：{res.get('detail')}")
        self.assertEqual(self.store.get_scene(s.id).text, "新的理解")
        self.assertEqual(self.store.get_summary(s2.id).text, "新的叙述")
        # S3 走修正：旧版进历史（新版另起、回 pending）
        self.assertIsNotNone(self.store.get_profile(p.id).invalidated_at)

        # 英文键也要能直接给（她给 field：§3.3「两处都归一到键」）
        s_b = Scene(id="S1-0009", title="旧标题", text="旧摘要")
        self.store.add_scene(s_b)
        self.assertTrue(update_by_layer(self.store, s_b.id, "新标题", "title")["ok"])
        self.assertEqual(self.store.get_scene(s_b.id).title, "新标题")

    def test_archive_refuses_unknown_and_missing(self):
        """归档守「要么全动、要么不动」（2026-09-24 检查修）：认不出 / 找不到
        **整批拒绝**——原来会返回 `ok=True`「归档了 0 条」（界面据此显示成功）。"""
        s2 = self._summary()
        out = archive_by_layer(self.store, ["M-0001"])
        self.assertFalse(out["ok"], "M 编号不是归档的对象")
        self.assertIn("认不出", out["detail"])
        out = archive_by_layer(self.store, ["S2-9999"])
        self.assertFalse(out["ok"])
        self.assertIn("没找到", out["detail"])
        # 混合：认得出的那一条也不许动（不做部分执行）
        out = archive_by_layer(self.store, [s2.id, "M-0001"])
        self.assertFalse(out["ok"])
        self.assertFalse(self.store.get_summary(s2.id).archived, "整批拒绝，一条都不动")

    def test_archive_twice_reports_no_change(self):
        """归档幂等：第二次点不再虚报「归档了 1 条」（同一次检查修）。"""
        s2 = self._summary()
        self.assertTrue(archive_by_layer(self.store, [s2.id]).get("changed"))
        out = archive_by_layer(self.store, [s2.id])
        self.assertTrue(out["ok"])
        self.assertFalse(out.get("changed"))
        self.assertIn("本来就在冷层", out["detail"])
        self.assertTrue(unarchive_by_layer(self.store, [s2.id]).get("changed"))

    def test_unarchive_refuses_unknown(self):
        out = unarchive_by_layer(self.store, ["M-0001"])
        self.assertFalse(out["ok"])
        self.assertIn("认不出", out["detail"])

    def test_update_refuses_non_memory_ids(self):
        """改也守编号纪律：M 是备忘，当场引导 `close_memo`（2026-09-24 检查修）。"""
        out = update_by_layer(self.store, "M-0001", "x", "标题")
        self.assertFalse(out["ok"])
        self.assertIn("close_memo", out["detail"])
        out = update_by_layer(self.store, "X-0001", "x")
        self.assertFalse(out["ok"])
        self.assertIn("认不出", out["detail"])

    def test_scene_topic_rejects_multi_value(self):
        """场景主题是**单值**：界面那条路也拦（2026-09-24 检查修）——
        判据与她那条（`revise_memory`）同一套，不静默存成含分隔符的怪主题。"""
        s = self._scene()
        out = update_by_layer(self.store, s.id, "a,b", "主题")
        self.assertFalse(out["ok"])
        self.assertIn("单值", out["detail"])
        out = update_by_layer(self.store, s.id, "  ", "主题")
        self.assertFalse(out["ok"], "空主题也拦")
        out = update_by_layer(self.store, s.id, " 用户·压力 ", "主题")
        self.assertTrue(out["ok"])
        self.assertEqual(self.store.get_scene(s.id).topic, "用户·压力",
                         "通过校验的按规整后的单值存")


class ConfirmFlowTest(Base):
    """她只能提议，**他说了才算**（工具箱 3.4 / 3.5 / 3.6）。

    这一层测的是"什么时候真的动手"：提议阶段一个字都不能改；
    而"嗯"这类含糊的话**必须不能**触发删除——
    删除和改动都不该由一句语气词触发。
    """

    def _scene(self):
        s = Scene(title="被组长当众批评", text="被批评后想离开")
        self.store.add_scene(s)
        return s

    def _ask(self, call: dict) -> ChatSession:
        """她提议一次（返回会话，等他下一句）。"""
        llm = tool_llm([[call]])
        sess = ChatSession(store=self.store, llm=llm, emb=_NoEmbedding())
        sess.reply("这条能不能处理一下")
        return sess

    def _scene_with_profile(self):
        """一条场景 + 一条引它的画像——测"删 vs 归档"对依据链的差别用。"""
        s = self._scene()
        p = Profile(id="S3-0001", topic="用户·压力", statement="受挫后倾向离开",
                    status=PROFILE_ESTABLISHED, evidence=1, sources=[s.id])
        self.store.add_profile(p)
        return s, p

    def test_wrong_kind_id_is_guided_not_just_not_found(self):
        """归不进「记忆三层」的编号要给引导，别只说"找不到"（2026-09-24 补）。

        实测的坑：她拿 `S3-0001` 调 `forget_memory`，只收到"找不到"——
        于是她说"删除接口找不到 S3 记忆"，卡在那儿。**2026-09-24 三层合并后**，
        S1/S2/S3 走同一套、不再有"画像不走这条路"；只剩一种：**M**
        （它不是删，是"了结"）——那一条仍要指到 `close_memo`。
        """
        ctx = {"confirm_channel": True}
        out = execute(self.store, "forget_memory", {"id": "M-0001"}, ctx)
        self.assertFalse(out["ok"])
        self.assertIn("close_memo", out["detail"], "备忘走闭合那条路")

        # 混合：一条 M + 一条真不存在——**两句都要说**（2026-09-24 检查补）
        out = execute(self.store, "forget_memory", {"id": "M-0001,S1-9999"}, ctx)
        self.assertIn("close_memo", out["detail"], "M 那条要给引导")
        self.assertIn("S1-9999", out["detail"], "不存在的编号也要提，别被引导吞掉")

        # **真实挂着**的 M 也要走引导（2026-09-24 检查修）：原来 `_memory_title`
        # 给 M 也取标题，于是"存在"的备忘被当成可处置对象——确认条弹出来、点了
        # 却是"认不出编号"（`delete_by_layer` 不认 M）。上一段用的是**不存在**的
        # M-0001，所以一直没抓到。
        m = Memo(content="月底回老家", kind=MEMO_USER_TASK)
        self.store.add_memo(m)
        out = execute(self.store, "forget_memory", {"id": m.id}, ctx)
        self.assertFalse(out["ok"], "挂着的 M 不是「删」的对象")
        self.assertIn("close_memo", out["detail"])
        out = execute(self.store, "revise_memory", {"id": m.id, "text": "x"}, ctx)
        self.assertFalse(out["ok"], "挂着的 M 也不是「改」的对象")
        self.assertIn("close_memo", out["detail"])

    def test_proposal_touches_nothing(self):
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        self.assertIsNotNone(self.store.get_scene(s.id), "她只能提议，不能自己动手")
        self.assertIsNotNone(sess.pending_confirm, "提议要留到下一轮等他点头")

    def test_clicking_yes_deletes_without_guessing(self):
        """**主通道**：点一下就算数，不用去判定他一句话是什么意思。"""
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        self.assertEqual((sess.pending_view() or {}).get("scene_id"), s.id)
        self.assertTrue(sess.confirm(True)["ok"])
        self.assertIsNone(self.store.get_scene(s.id))

    def test_receipt_comes_back_to_her(self):
        """界面点过之后，**回执要回到她那里**（2026-09-24 修）。

        那条路不经过工具回合，结果原本只回到界面——她不知道自己提议的事后来
        怎么了：下一轮会再提议一次，或还引用已经删掉的那条。
        """
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        self.assertTrue(sess.confirm("delete")["ok"])
        note = sess._actions_note()
        self.assertIn("删掉", note, "下一轮她要看得见：他点了、结果是删掉了")
        self.assertIn(s.id, note)

        # 归档那档也一样——他点的哪一档，她都要知道
        s2 = Summary(topic="用户·X", text="叙述")
        self.store.add_summary(s2)
        sess2 = self._ask({"name": "forget_memory", "arguments": {"id": s2.id}})
        self.assertTrue(sess2.confirm("archive")["ok"])
        self.assertIn("归档", sess2._actions_note(), "归档的回执也要回来")

    def test_receipt_lives_a_few_turns_then_drops(self):
        """回执留 `_RECEIPT_TURNS` 轮就丢（2026-09-25 定）：够"隔一两轮再问起"，
        又不一直占着上下文——真要查证她自己 `memory_search` 就能查。"""
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        sess.confirm("delete")
        # 推进轮数（每轮结束都会走 `_remember_actions`；这里假设后面几轮没调工具）
        for _ in range(_RECEIPT_TURNS - 1):
            sess._remember_actions([])
            self.assertIn("删掉", sess._actions_note(), "这几轮之内要还在")
        sess._remember_actions([])          # 再多一轮，超出留存
        self.assertNotIn("删掉", sess._actions_note(), "过几轮就丢，不占地方")

    def test_receipt_covers_keep_and_refused(self):
        """「留着」和「没成」两档也要回给她——她不该靠猜。"""
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        self.assertTrue(sess.confirm("keep")["ok"])
        self.assertIsNotNone(self.store.get_scene(s.id), "留着 = 一条都不动")
        self.assertIn("不删除", sess._actions_note(),
                      "他选了留着她要知道——不然还会再提议一次")

        sess2 = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        out = sess2.confirm("随便")
        self.assertFalse(out["ok"])
        self.assertIn("没成", sess2._actions_note(), "失败要标出来——不然她以为做成了")

    def test_clicking_no_leaves_it_alone(self):
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        self.assertTrue(sess.confirm(False)["ok"])
        self.assertIsNotNone(self.store.get_scene(s.id))
        self.assertIsNone(sess.pending_confirm)

    def test_clicking_archive_cools_it_without_deleting(self):
        """确认条三选一的中间档（2026-09-23，工具箱稿 §3.4）：
        归档 = 数据全在、只是她不召回；**不摘引用、不留痕**。"""
        s, p = self._scene_with_profile()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})

        self.assertTrue(sess.confirm("archive")["ok"])

        got = self.store.get_scene(s.id)
        self.assertIsNotNone(got, "归档不是删——记录必须还在")
        self.assertTrue(got.archived, "archived=1 = 进冷层")
        self.assertNotIn(s.id, [x.id for x in self.store.query_scenes()],
                         "归档后不进召回（query_scenes 默认排除归档）")
        self.assertIn(s.id, self.store.get_profile(p.id).sources,
                      "归档**不摘引用**——画像的依据链不断（两档都不摘，见下条）")

    def test_delete_confirms_removes_the_row(self):
        """彻底删除那一档：**删一行**（2026-09-24 起不摘引用——读端过滤）。

        与归档那档的对照也在这里：归档**不摘**、删除**也不摘**——
        两档现在的差别只剩"行在不在"（记录是"当时凭什么"，谁都不动它）。
        """
        s, p = self._scene_with_profile()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})

        self.assertTrue(sess.confirm("delete")["ok"])

        self.assertIsNone(self.store.get_scene(s.id))
        self.assertIn(s.id, self.store.get_profile(p.id).sources,
                      "记录不动——引用读成 `sources ∩ 现存节点`（读端过滤）")
        self.assertFalse(self.store.exists(s.id))

    def test_talking_archive_word_works(self):
        """对话兜底认「归档」（工具箱稿 §六的词表）——他不点按钮也算数。"""
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        sess.reply("先归档吧")

        got = self.store.get_scene(s.id)
        self.assertIsNotNone(got, "归档不是删")
        self.assertTrue(got.archived)
        self.assertIsNone(sess.pending_confirm)

    def test_single_archive_does_not_touch_profile(self):
        """单条归档也不动画像引用（同 `test_clicking_archive_*` 的直连版：
        走 `archive_scene_confirmed`，确认"归档 ≠ 摘引用"在 store 层就成立）。"""
        from core.weave import archive_scene_confirmed
        s, p = self._scene_with_profile()
        out = archive_scene_confirmed(self.store, s.id)
        self.assertTrue(out["ok"] and out["changed"])
        self.assertIn(s.id, self.store.get_profile(p.id).sources)
        self.assertTrue(self.store.get_scene(s.id).archived)

    def test_archive_then_restore_roundtrip(self):
        """归档**可逆**："留后路"要真的留得住（回热层 → 重新进召回）。"""
        from core.weave import archive_scene_confirmed, unarchive_scene_confirmed
        s, p = self._scene_with_profile()

        archive_scene_confirmed(self.store, s.id)
        self.assertEqual(self.store.query_scenes(), [], "归档后不进召回")

        out = unarchive_scene_confirmed(self.store, s.id)
        self.assertTrue(out["ok"] and out["changed"])
        self.assertEqual([x.id for x in self.store.query_scenes()], [s.id],
                         "回热层后重新进召回")
        self.assertFalse(self.store.get_scene(s.id).archived)
        self.assertIn(s.id, self.store.get_profile(p.id).sources, "全程不摘引用")
        self.assertFalse(unarchive_scene_confirmed(self.store, s.id)["changed"],
                         "已经不在冷层：幂等（changed=False，不是失败）")

    def test_clicking_twice_does_not_run_twice(self):
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        sess.confirm(True)
        self.assertFalse(sess.confirm(True)["ok"], "第二次已经没有东西可执行了")

    def test_his_yes_deletes(self):
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        sess.reply("删了吧")
        self.assertIsNone(self.store.get_scene(s.id))
        self.assertIsNone(sess.pending_confirm, "执行完就不能再留着")

    def test_vague_does_not_count(self):
        """「嗯」不能触发删除——这是设计稿 3.6 里最硬的一条。"""
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        sess.reply("嗯")
        self.assertIsNotNone(self.store.get_scene(s.id))
        self.assertIsNotNone(sess.pending_confirm, "没说清楚就得再问一次")

    def test_distraction_drops_it(self):
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        sess.reply("今天天气不错")
        self.assertIsNone(sess.pending_confirm, "岔开就作废，不再追问")
        self.assertIsNotNone(self.store.get_scene(s.id))

    def test_his_wording_beats_her_suggestion(self):
        """他说了具体内容就用他说的，不是照抄她的提议。"""
        s = self._scene()
        sess = self._ask({"name": "revise_memory",
                          "arguments": {"scene_id": s.id, "text": "她的说法"}})
        sess.reply("不对，应该是他自己主动提的")
        self.assertIn("主动提", self.store.get_scene(s.id).text)

    def test_closing_drops_a_pending_delete(self):
        """会话结束就丢——**不该留一个隔夜的待执行删除**。"""
        s = self._scene()
        sess = self._ask({"name": "forget_memory", "arguments": {"scene_id": s.id}})
        sess.close()
        self.assertIsNone(sess.pending_confirm)
        sess.reply("删了吧")            # 隔了很久再说这句，不该算数
        self.assertIsNotNone(self.store.get_scene(s.id))


class ConfirmJudgementTest(unittest.TestCase):
    """判定的那张表（设计稿 3.6）。用规则不用模型，所以要能逐条钉住。"""

    def test_cases(self):
        self.assertEqual(_judge_confirmation("删了吧")[0], "delete")
        self.assertEqual(_judge_confirmation("不要了")[0], "delete")
        self.assertEqual(_judge_confirmation("嗯")[0], "vague")
        self.assertEqual(_judge_confirmation("好的")[0], "vague")
        self.assertEqual(_judge_confirmation("今天天气不错")[0], "drop")
        self.assertEqual(_judge_confirmation("")[0], "drop")

    def test_revise_picks_up_his_wording(self):
        verdict, said = _judge_confirmation("不对，应该是他自己主动提的")
        self.assertEqual(verdict, "revise")
        self.assertEqual(said, "他自己主动提的")

    def test_revise_without_detail_uses_her_suggestion(self):
        verdict, said = _judge_confirmation("改一下吧")
        self.assertEqual(verdict, "revise")
        self.assertEqual(said, "", "他没说具体内容 → 用她提议的那个")

    def test_discussion_does_not_trigger_change(self):
        """**讨论 ≠ 指令**（2026-09-23 定，改动不自动执行）：他评论一句
        "不太对"，不该把记忆改掉——那是覆盖他的表达。

        "不对 / 错了"这类评价词已从执行词表移除；疑问句一律不执行
        （"改一下吧？"是在问，再确认一次）。
        """
        for said in ("我觉得这说法不太对", "这个好像错了", "嗯……不太对"):
            self.assertNotEqual(_judge_confirmation(said)[0], "revise", said)
        for said in ("改一下吧？", "删了吧？", "归档吧？", "改改吗"):
            self.assertEqual(_judge_confirmation(said)[0], "vague", said)
        # 真让改的仍然照旧（他点名了要改 / 给了新说法）
        self.assertEqual(_judge_confirmation("改一下吧")[0], "revise")
        self.assertEqual(_judge_confirmation("不对，应该是他自己主动提的")[0], "revise")

    def test_negation_beats_the_keyword(self):
        """「别删，归档吧」不能判成删——删除不可恢复，最贵的误判（2026-09-23 检查发现）。

        否定只往左看两个字符：够压住「别删 / 不要删 / 不是删」；窗口再大就会
        把隔了标点的否定带过来（「别删，归档吧」的归档前两字是「删，」）。
        """
        self.assertEqual(_judge_confirmation("别删，归档吧")[0], "archive")
        self.assertEqual(_judge_confirmation("不是删，是归档")[0], "archive")
        self.assertEqual(_judge_confirmation("别删")[0], "vague",
                         "只否定了删、没有别的动作 → 再问一次")
        self.assertEqual(_judge_confirmation("不要了")[0], "delete",
                         "「不要了」没有「删」字，不受否定逻辑影响")
        self.assertEqual(_judge_confirmation("删了吧，不用归档")[0], "delete",
                         "「删」前面没有否定 → 该删")


if __name__ == "__main__":
    unittest.main()
