"""LLM 调用——**结构化输出 + 重试 + 失败降级**。

三条设计约定，都跟「不崩」有关：
  1. **所有字段抽取走 `structured()`**，禁止自由文本 + 正则解析。
     正则解析 JSON 是脆的：LLM 换个措辞就崩，而崩在写入路径上就是丢记忆。
  2. **失败返回带默认值的 dict，不抛异常**。一次 LLM 抽风不该让整条链路停摆；
     上层拿到保守默认值继续走（`sensitive=True` 这种「拿不准就设成安全侧」）。
  3. **JSON schema 优先，不支持才降 json_object**。能力位是运行时探测出来的，
     不硬编码——不同服务端（OpenAI / 自建 / 本地）支持程度不一样，
     写死一个档位等于把兼容性赌在某一家上。

只依赖标准库 `urllib`（与 embedding.py 同风格）：引入 openai SDK 会带来
httpx / pydantic 一串依赖，而本项目要的只是「POST 一个 JSON、拿回一个 JSON」。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L2 外部服务（模型）
#   上游    ：config（endpoint / key / model / 超时）
#   下游    ：scene / recall / distill / memo / trend / chat / dashboard —— 所有要调模型的地方
#   对外入口：`LLM`（`structured` / `chat` / `chat_with_tools` / `chat_stream` / `search`）
#   边界    ：**绝不抛异常**——失败一律收敛成"保守默认值 + 一行日志"，
#             含能力位探测（不支持 json_schema → 这一进程起走 json_object）
# ---------------------------------------------------------------------
# 本文件分段
#   段 1  模块函数 —— _parse_json / _fill_defaults / 错误归类（纯函数，好测）
#   段 2  class LLM —— available / structured / chat / chat_with_tools /
#         chat_stream / search（搜索是独立于主对话的一条路）
#   段 3  内部 —— _post / _post_json（HTTP + 能力位降级）、_warn_once
# 搜索的形态知识（各家的请求拼装 / 响应解析 / 专属失败文案）**不在这里**——
# 见 `core/search.py` 的通道表：`search()` 只跑通用流程，不认识服务商。
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request

from . import config as cfgmod
from . import search as searchmod


class _UnsupportedSchema(Exception):
    """服务端明确表示不支持 json_schema 档（要永久降到 json_object）。"""


# ---- 段 1：纯函数 ----

def _parse_json(text: str) -> dict | None:
    """尽力把模型输出解析成 dict。

    只做两件容错：剥掉 ```json 围栏（模型最常见的"礼貌"）、
    取最外层一对花括号。**不做正则修补**——修不好的就返回 None，
    让上层走保守默认，比修出一个半对的结果安全。
    """
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    try:
        data = json.loads(s.strip())
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        pass
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(s[start:end + 1])
            return data if isinstance(data, dict) else None
        except (ValueError, TypeError):
            return None
    return None


def _fill_defaults(data: dict | None, defaults: dict) -> dict:
    """把模型返回值对齐到 schema 的默认值。

    schema 里声明过的键一定有值（缺了就用默认），多出来的键丢弃——
    「多出来的键」意味着模型在自由发挥，放它进库会污染数据模型。
    类型完全不符时用默认值：宁可丢一个字段，也不要一个 list 字段塞进 str。
    """
    out = {}
    data = data or {}
    for key, default in (defaults or {}).items():
        val = data.get(key, default)
        if isinstance(default, list) and not isinstance(val, list):
            val = default
        elif isinstance(default, str) and not isinstance(val, str):
            val = default
        elif isinstance(default, bool) and not isinstance(val, bool):
            val = default
        out[key] = val
    return out


# ---- 搜索 -----
#
# **形态知识不在这里**（2026-09-26 重构）：请求怎么拼、响应怎么读、哪家有什么
# 私话（"账号没开通联网搜索"这类），全在 `core/search.py` 的通道表里——
# 这张表当初就是长在这个文件里的 if/else，换一家要重写一次。
# 这里只剩通用流程：`LLM.search()`（取通道 → 发一次 → 解析 → 判"真的搜了没有"）。


def _classify_error(err: Exception, body: str = "") -> str:
    """把一次失败归类成**人能照着做点什么**的那一种。

    为什么要分：界面上原来只有一句「模型没有返回内容——检查配置」，
    而真实原因常常是"这一轮太长 / 想太久"，那句提示会把人引到完全错的方向
    （去改配置，而配置好好的——实测就是这么发生的：拖进一篇设计稿，
    她回了一句"检查 LLM 配置"）。
    同 `_looks_unsupported` 那条经验：**分不清失败的种类，就只能给一句谁都不得罪的废话。**
    """
    if isinstance(err, urllib.error.HTTPError):
        if err.code in (401, 403):
            return "auth"
        if err.code == 429:
            return "rate"
        # `body` 由调用方传进来时**不要再读一次**——HTTPError 的体是一次性的，
        # 前一个分支读过就空了，重读会把 "too_long" 判成 "http_400"
        # （这是本轮最长的一类失败，判错就把人引去改配置）。
        low = (body or _http_error_body(err)).lower()
        if "context" in low or "too long" in low or "length" in low:
            return "too_long"
        return f"http_{err.code}"
    if isinstance(err, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(err, urllib.error.URLError):
        # urlopen 超时也是 URLError，真正的超时藏在 `reason` 里
        return ("timeout" if isinstance(err.reason, (socket.timeout, TimeoutError))
                else "network")
    return "other"


def _http_error_body(err: Exception) -> str:
    """取 HTTPError 的响应体（读不出来就当没有）。

    失败分支里要**先读体再判种类**：400 是"我这次请求有毛病"还是
    "你这个功能我没有"，答案只写在体里，不写在状态码里。读体自己也可能炸
    （连接已经半关了），所以一律吞掉——**取不到体只是少一条降级线索，
    不该让报错变成第二次崩**。
    """
    try:
        return (err.read() or b"").decode("utf-8", "ignore")
    except Exception:
        return ""


# ---- 段 2：LLM ----

class LLM:
    """OpenAI 兼容 /chat/completions 客户端（纯标准库）。

    没有配置 endpoint / api_key 时 `available()` 为 False，
    `structured()` 直接返回保守默认值并只提示一次——
    demo 和测试因此不需要真模型也能把整条链路跑通。
    """

    def __init__(self, timeout: float | None = None):
        llm = cfgmod.cfg("llm", default={}) or {}
        self.endpoint = (llm.get("endpoint") or "").rstrip("/")
        self.api_key = llm.get("api_key") or ""
        self.model = llm.get("model") or ""
        self.timeout = timeout or float(llm.get("timeout") or 30)
        # 默认温度固定 0：抽取类任务要的是稳定、可复现（见 config 的 chat 段注释）。
        # **刻意不做成旋钮**——CONFIG / 本地配置 / 环境变量三处都没有这个键，
        # 曾经写成 `llm.get("temperature", 0.0)` 是个"读点等不到旋钮"的幽灵入口；
        # 生成回复那条路的温度走 `config.chat.temperature`，由调用方显式传进来。
        self.temperature = 0.0
        # 能力位：unknown / json_schema / json_object。运行时探测，不硬编码。
        self._cap = "unknown"
        # 工具能力位：unknown / unsupported。**探测到不支持就整块关掉工具箱**，
        # 不退化成"让它输出 JSON 再解析"——那会污染每一句回复（设计稿第十节）。
        self._tools_cap = "unknown"
        # 搜索能力位：unknown / unsupported。**探测到没有就整块关掉**
        # （连工具一起从提示词里摘掉），不做半吊子（设计稿第十一节）。
        self._search_cap = "unknown"
        self._warned = False
        self._mock = None
        # 最近一次真实调用的结果（2026-09-25）：给仪表盘顶栏的「模型」状态用。
        # 为什么要有它：界面上的勾原来只看「配置填了没有」——今天向量服务死了两天，
        # 勾一直亮着，谁都不知道。**状态只记在 `_send` 这一个出口**：
        # 四个调用点各写一遍必然漏一处，而漏的那一处就是出问题的那一次。
        # 语义：`last_ok_at` / `last_err_at` 谁晚谁说了算；`last_error` 是
        # `_classify_error` 的机器码（auth / rate / timeout / network / http_xxx…），
        # 翻人话是界面的事（同 `_no_reply_hint` 的分工）。
        self.last_ok_at = ""
        self.last_err_at = ""
        self.last_error = ""

    def available(self) -> bool:
        return bool(self.endpoint and self.api_key and self.model)

    def set_mock(self, fn):
        """注入假模型（测试与 demo 用）：fn(prompt, schema) -> dict。

        放在生产类里而不是子类，是为了让「测试跑的是同一条代码路径」——
        换个子类的话，被绕过的那几行恰恰是最容易出问题的那几行。
        """
        self._mock = fn

    # ---- 对外入口 ----

    def structured(self, prompt: str, schema: dict, retries: int = 3) -> dict:
        """结构化抽取：返回**一定包含 schema.defaults 全部键**的 dict。

        schema 形状（自定义，比 OpenAI 原生多一个 defaults 键）：
            {"name": "scene_card", "strict": True,
             "schema": {...JSON Schema...},
             "defaults": {"title": "", "valence": None, ...}}

        `defaults` 是本地降级用的，不会发给服务端。
        **绝不抛异常**——所有失败路径都收敛到「返回默认值」。
        """
        defaults = dict(schema.get("defaults") or {})

        if self._mock is not None:
            try:
                return _fill_defaults(self._mock(prompt, schema), defaults)
            except Exception as e:
                print(f"[llm] mock 失败，走默认值: {e}")
                return defaults

        if not self.available():
            self._warn_once("未配置 LLM（endpoint / api_key / model），结构化抽取一律走保守默认值")
            return defaults

        delay = 0.5
        for attempt in range(max(1, retries)):
            for fmt in self._formats(schema):
                try:
                    text = self._post(prompt, fmt)
                except _UnsupportedSchema:
                    self._cap = "json_object"     # 永久降档，下次不再试
                    continue
                except Exception as e:
                    print(f"[llm] 调用失败（第 {attempt + 1} 次）: {e}")
                    break                          # 网络类错误：跳出 fmt 循环去退避重试
                data = _parse_json(text)
                if data is not None:
                    if fmt.get("type") == "json_schema":
                        self._cap = "json_schema"
                    return _fill_defaults(data, defaults)
                # 解析失败：换档再试一次（json_object 只保证合法 JSON，
                # 但真出现空返回时换档也没用，所以只用同一次机会）
                print(f"[llm] 输出无法解析为 JSON，长度={len(text or '')}")
            time.sleep(delay)
            delay = min(delay * 2, 4.0)
        self._warn_once("结构化抽取连续失败，本轮到保守默认值（不中断链路）")
        return defaults

    def chat(self, messages: list[dict], max_tokens: int | None = None,
             temperature: float | None = None) -> str:
        """纯文本调用（单轮），失败返回空串。

        `temperature` 单独开一个口：抽取任务要 0（稳定、可复现），
        生成回复要留余地。写死一个值就会让其中一边将就。

        ⚠️ **`max_tokens` 不传时取 `config.chat.max_tokens`（同一处数字，
        不再在签名里写死）——这个额度会被推理吃掉。**
        实测：DeepSeek 的 `deepseek-flash`（推理模型）回答前先生成
        `reasoning_content`，额度小（当年签名里写死过 3000 / 800）经常在
        思考阶段就用完，`content` 返回空串。症状是「模型没有返回内容」，
        看着像模型坏了或配置填错了，真正原因只是额度不够。
        上限调大**不会**让模型多写（它写到自然结束就停）。
        """
        if max_tokens is None:
            max_tokens = int(cfgmod.cfg("chat", "max_tokens", default=16000))
        if self._mock is not None:
            try:
                return self._mock(messages, {}) or ""
            except Exception:
                return ""
        if not self.available():
            self._warn_once("未配置 LLM，chat 返回空串")
            return ""
        payload = {"model": self.model, "messages": messages,
                   "temperature": (self.temperature if temperature is None else temperature),
                   "max_tokens": max_tokens}
        try:
            with self._send(payload) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[llm] chat 失败: {e}")
            return ""
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        text = msg.get("content") or ""
        if not text and msg.get("reasoning_content"):
            # 有思考、没正文 = 额度全被思考吃掉了。**必须说出来**：
            # 否则上层只看到「模型没有返回内容」，会往配置上查半天。
            print(f"[llm] 只回了思考没回正文（finish_reason={choice.get('finish_reason')}，"
                  f"max_tokens={max_tokens}）——调大 max_tokens 或换非推理模型")
        return text

    def chat_with_tools(self, messages: list[dict], tools: list[dict] | None = None,
                        max_tokens: int | None = None,
                        temperature: float | None = None,
                        timeout: float | None = None) -> dict:
        """带工具的一次调用：`{"content": str, "tool_calls": [...]}`。

        **不解析模型的自由文本**——`tool_calls` 只从服务端返回的结构里拿。
        检测不到工具调用就是一次普通回复（`tool_calls` 为空）。

        为什么不做"让它输出 JSON 再解析"的退路：那会污染每一句回复
        （她说不说人话都在赌解析对不对），而工具本来就是"锦上添花"——
        不稳就整块关掉（设计稿第十节）。这条守住了，她才始终是那个会聊天的 air。
        """
        if max_tokens is None:
            max_tokens = int(cfgmod.cfg("chat", "max_tokens", default=16000))
        if self._mock is not None:
            try:
                out = self._mock(messages, {"_tools": tools} if tools else {})
            except Exception as e:
                print(f"[llm] mock 失败，走默认值: {e}")
                return {"content": "", "tool_calls": []}
            if isinstance(out, dict):
                # mock 想模拟工具调用就给这两个键；给别的（比如按 schema 分支
                # 返回的 `{}`）一律当"这次没有话要说"，**不能把 dict 当正文**——
                # 那会在 `.strip()` 上炸，而且炸在对话路径上。
                if "content" in out or "tool_calls" in out:
                    return {"content": out.get("content") or "",
                            "tool_calls": out.get("tool_calls") or []}
                return {"content": "", "tool_calls": []}
            return {"content": out or "", "tool_calls": []}

        if not self.available():
            self._warn_once("未配置 LLM，chat 返回空串")
            return {"content": "", "tool_calls": []}

        payload = {"model": self.model, "messages": messages,
                   "temperature": (self.temperature if temperature is None else temperature),
                   "max_tokens": max_tokens}
        if tools and self._tools_cap != "unsupported":
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            with self._send(payload, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = _http_error_body(e)
            if (e.code == 400 and tools and self._tools_cap != "unsupported"
                    and self._looks_tools_unsupported(body)):
                self._tools_cap = "unsupported"
                print("[llm] 服务端不支持工具调用（400）——本会话起关掉工具箱，"
                      "退回人点按钮（按钮不会飘，见设计稿第十节）")
                return self.chat_with_tools(messages, tools=None,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            timeout=timeout)
            kind = _classify_error(e, body)
            print(f"[llm] chat_with_tools 失败（{kind}）: HTTP {e.code}: {body[:120]}")
            return {"content": "", "tool_calls": [], "error": kind}
        except Exception as e:
            kind = _classify_error(e)
            print(f"[llm] chat_with_tools 失败（{kind}）: {e}")
            return {"content": "", "tool_calls": [], "error": kind}

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                args = _parse_json(args) or {}     # 参数是 JSON 串（服务端约定）
            calls.append({"id": c.get("id") or "",
                          "name": fn.get("name") or "",
                          "arguments": args if isinstance(args, dict) else {}})
        text = msg.get("content") or ""
        if not text and not calls and msg.get("reasoning_content"):
            print(f"[llm] 只回了思考没回正文（finish_reason={choice.get('finish_reason')}，"
                  f"max_tokens={max_tokens}）——调大 max_tokens 或换非推理模型")
        return {"content": text, "tool_calls": calls}

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None,
                    max_tokens: int | None = None, temperature: float | None = None,
                    timeout: float | None = None):
        """流式调用：yield `{"type": "reasoning"|"content"|"tool_calls"|"error", …}`。

        **为什么要有它**：复杂问题就是要想很久，超时不该是砍思考的那把刀——
        干等的问题该靠**把过程显示出来**解决，不该靠掐断解决。
        它和 `chat_with_tools` 是同一件事的两种走法（形状一样，只是给得早晚不同）。

        **一次 read 一个超时**（同一个值用在每次 recv 上）：
        所以想十分钟也没关系（只要一直在出字），而真卡住（一个字都不来）会立刻报。
        这比"总时长上限"准得多——后者砍的是思考，不是卡死。
        """
        if max_tokens is None:
            max_tokens = int(cfgmod.cfg("chat", "max_tokens", default=16000))
        if not self.available():
            yield {"type": "error", "error": "no_llm"}
            return
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "temperature": (self.temperature if temperature is None else temperature),
                   "max_tokens": max_tokens}
        if tools and self._tools_cap != "unsupported":
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            resp = self._send(payload, timeout=timeout)
        except urllib.error.HTTPError as e:
            # **和非流式同一条降级**（400 +「不支持工具」→ 整块关掉工具箱）。
            # 这里必须自己写一遍：能力位写在 `chat_with_tools` 里，而流式走不进
            # 那条路，于是仪表盘（默认走 `/api/chat/stream`）上等于没有这条保护
            # ——服务端不支持时每一轮都重发 tools 再吃一次 400，那句"关掉"永不生效。
            body = _http_error_body(e)
            if (e.code == 400 and tools and self._tools_cap != "unsupported"
                    and self._looks_tools_unsupported(body)):
                self._tools_cap = "unsupported"
                print("[llm] 服务端不支持工具调用（400）——本会话起关掉工具箱，"
                      "退回人点按钮（按钮不会飘，见设计稿第十节）")
                yield from self.chat_stream(messages, tools=None,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            timeout=timeout)
                return
            yield {"type": "error", "error": _classify_error(e, body)}
            return
        except Exception as e:
            yield {"type": "error", "error": _classify_error(e)}
            return

        calls: dict[int, dict] = {}
        finish = ""
        try:
            for raw in resp:                      # 逐行读（每次 read 都受同一个超时管）
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except ValueError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                # **结束原因要读**：`length` 表示被额度掐断（她的回答不完整），
                # 而"想完了没正文"和"被掐断"长得一样——不读它就只能猜。
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                # 推理模型的思考：DeepSeek 放 `reasoning_content`，别人可能放 `reasoning`
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if thought:
                    yield {"type": "reasoning", "text": thought}
                if delta.get("content"):
                    yield {"type": "content", "text": delta["content"]}
                # 工具调用是**分片**来的：名字和参数都要拼
                for tc in delta.get("tool_calls") or []:
                    i = tc.get("index") or 0
                    cur = calls.setdefault(i, {"id": "", "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if tc.get("id"):
                        cur["id"] = tc["id"]
                    if fn.get("name"):
                        cur["name"] += fn["name"]
                    if fn.get("arguments"):
                        cur["arguments"] += fn["arguments"]
        except Exception as e:
            yield {"type": "error", "error": _classify_error(e)}

        if finish == "length":
            # 被额度掐断：正文可能只说了一半，也可能一个字都没有
            yield {"type": "truncated"}
        if calls:
            yield {"type": "tool_calls",
                   "calls": [{"id": calls[i]["id"] or f"call_{i}",
                              "name": calls[i]["name"],
                              "arguments": _parse_json(calls[i]["arguments"]) or {}}
                             for i in sorted(calls)]}

    # ---- 联网搜索（独立于主对话的一条路）----

    def search_capable(self) -> bool:
        """这个服务能不能搜（探测过按探测结果答，没探测过先当能）。

        `chat._generate` 用它决定这一轮要不要把 `web_search` 放进工具列表——
        **探测到不支持就连工具一起摘掉**（防错第 1 条：工具要少，选择越多越容易选错）。
        """
        return self._search_cap != "unsupported"

    def search(self, query: str, timeout: float | None = None) -> dict:
        """联网搜索（服务端工具）。**不抛异常**。

        **这里是通用流程，不认识任何具体服务商**（2026-09-26 重构）：
        取通道（`search.channel` → `core/search.py` 的通道表）→ 拼请求 →
        发一次 → 按通道自己的读法解析 → 判"真的搜了没有"。
        换一家 / 加一种新形态只动那张表（或配置），不动这个函数——
        在此之前，两条通道是写死在这里的 if/else，换一家得把这里重写一遍。

        通道自带三件事：端点从哪来（独立端点 / 与对话同端）、请求怎么拼、
        响应怎么读（**含这家专属的失败人话**，如"账号没开通联网搜索"）。

        所有通道共用一条纪律：**不退化成"让它写一段文本再解析"**——
        没有结构化结果就是没搜、判失败（各家响应形状不同，判据跟着通道走）。

        返回 `{"ok", "text", "queries", "urls", "sources", "error"}`。
        `sources` 是结构化结果（标题 + URL），`text` 是那次调用的汇报（素材）。
        """
        out = {"ok": False, "text": "", "queries": [], "urls": [],
               "sources": [], "error": ""}
        q = (query or "").strip()
        if not q:
            out["error"] = "没有说要查什么"
            return out
        if self._search_cap == "unsupported":
            out["error"] = "这个服务不支持联网搜索（已经关掉了）"
            return out
        if not self.available():
            out["error"] = "没有配置 LLM"
            return out

        ch, why = searchmod.selected()
        if ch is None:
            # 配置写错了要看得见，而且要**两边都看见**：她那边是这句 error（工具会
            # 翻成"没查到"），人那边靠日志——只留前者的话，人会以为"这几天网上
            # 查不到东西"是网络问题（不静默回退默认通道，也不静默吞掉原因）。
            self._warn_once(f"搜索通道配置有问题：{why}")
            out["error"] = why
            return out
        if ch.base_from == "search":
            base = (cfgmod.cfg("search", "endpoint") or "").rstrip("/")
            if not base:
                out["error"] = "没有配置搜索端点（search.endpoint）"
                return out
        else:
            # 与对话同端——**不读 `search.endpoint`**：那条默认值属于独立端点
            # 那条通道，误用它会把请求发去别人的端点（拿新家的 key 打旧家的
            # 端点，401）。这件事由通道自己声明（`base_from`），主流程不猜。
            base = self.endpoint
        model = cfgmod.cfg("search", "model") or self.model
        try:
            headers, payload = ch.build(q, model, self.api_key)
        except Exception as e:
            # 通道是**扩展点**，它自己炸了也得由这里兜住（本函数的契约是绝不抛）——
            # 拼装是新写的一条描述里最先出错的地方，报错要指名道姓：一句笼统的
            # "没查到"会让人去查网络，而真正的原因在表里。
            out["error"] = f"搜索通道「{ch.name}」拼不出请求（{type(e).__name__}: {e}）"
            return out
        try:
            data = self._post_json(f"{base}{ch.path}", headers, payload, timeout)
        except Exception as e:
            body = _http_error_body(e) if isinstance(e, urllib.error.HTTPError) else ""
            # 顺序有讲究：**先问通道的私话，再落通用分类**——响应体是一次性的
            # （`_note_err` 那条注释），`hint` 要先读。
            hint = ch.hint(body) if ch.hint else ""
            if hint:
                # 账号**没开通**联网搜索这类：不置 unsupported——
                # 开通后直接重试即可，不必重启（记成"服务没这个能力"会逼人重启）。
                out["error"] = hint
            elif searchmod.looks_unsupported(e):
                # 整块关掉，不做半吊子（设计稿第十节的精神，同样适用于搜索）
                self._search_cap = "unsupported"
                out["error"] = "这个服务不支持联网搜索（已关掉，之后不再试）"
            else:
                out["error"] = f"没查成（{type(e).__name__}: {e}）"
            return out

        try:
            text, queries, sources = ch.parse(data)
        except Exception as e:
            # 同上：读法也由通道给。出格的响应（顶层不是对象、键的类型变了）
            # 不该穿透出去——那会让工具侧只剩一句"没查到"，真因（表里那条
            # 描述的假想不对）看不见。
            out["error"] = (f"搜索通道「{ch.name}」读不懂这次的响应"
                            f"（{type(e).__name__}: {e}）")
            return out
        # **没有结构化结果 = 服务端没执行搜索**（模型没工具时照样会写文本：
        # 自白 / 审核标签 / 假 tool_call）。那段文本不能当结果——把这条
        # 判死，才不会重演「查到了：当前环境未提供可用的联网搜索工具」。
        # 判据是**结果成不成形**（`{url, title}`）而非"列表非空"：对所有通道
        # 一样，顺手也挡住新通道给回来的半成品（`urls` 由它推出，形状不齐
        # 就会在下几行炸）。
        if not sources or not all(isinstance(s, dict) and s.get("url")
                                  for s in sources):
            out["error"] = "这次没有真的联网查（响应里没有搜索结果）"
            return out
        if not queries:
            queries = [q]      # 拿不到"实际搜的词"的通道——用请求词兜底（给 trace 看）
        out.update(ok=True, text=text, queries=queries, sources=sources,
                   urls=[s["url"] for s in sources])
        return out

    # ---- 段 3：内部 ----

    def _formats(self, schema: dict) -> list[dict]:
        """候选 response_format，按能力位排序。

        json_schema 档是真约束解码（字段、枚举都锁死）；
        json_object 档只保证「是合法 JSON」，字段靠 defaults 兜底。
        """
        if self._cap == "json_object":
            return [{"type": "json_object"}]
        return [
            {"type": "json_schema",
             "json_schema": {"name": schema.get("name", "out"),
                             "strict": bool(schema.get("strict", True)),
                             "schema": schema.get("schema") or {}}},
            {"type": "json_object"},
        ]

    def _note_err(self, e: Exception) -> None:
        """把一次失败记进状态。**不读响应体**——体是一次性的（`_http_error_body`
        先读先得），这里读一口，调用点的降级判据就瞎了（工具箱 / 搜索的 400 降级
        全指望那个体）。状态显示按状态码分档就够。
        """
        if isinstance(e, urllib.error.HTTPError):
            code = e.code
            self.last_error = ("auth" if code in (401, 403)
                               else "rate" if code == 429 else f"http_{code}")
        else:
            self.last_error = _classify_error(e)
        self.last_err_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def _send(self, payload: dict, timeout: float | None = None):
        """发送一次请求——**所有真实调用的唯一出口**，顺带记「最近一次调用」。

        返回响应对象（读流是调用方的事）；失败原样抛出——降级 / HTTP 分支
        仍是各调用点自己的事，这里只记账。
        """
        try:
            resp = urllib.request.urlopen(self._request(payload),
                                          timeout=timeout or self.timeout)
        except Exception as e:
            self._note_err(e)
            raise
        self.last_ok_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return resp

    def _request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )

    def _post_json(self, url: str, headers: dict, payload: dict,
                   timeout: float | None = None) -> dict:
        """POST 一份 JSON、读回 JSON——**搜索这类"旁路调用"的唯一网络出口**。

        头由**发起方给**（搜索那边由通道拼，见 `core/search.py`）：各家形态的
        头不一样（`x-api-key` / `anthropic-version` 这些是 Anthropic 侧的事），
        出口不该认识它们——出口认识的钱包只有一个：这个 URL 是不是安全。

        为什么不复用主对话的 `_send` / `_request`：那条路记的是**模型服务**的
        状态（仪表盘顶栏那盏灯），而搜索允许走另一个端点、另一个模型
        （官方要求分开配）——把它的成败算进模型状态，那盏灯就会说谎。

        超时用 `search.timeout`：一次搜索要"搜 → 读 → 再生成"，比普通回复慢得多。
        """
        tmo = timeout or float(cfgmod.cfg("search", "timeout", default=60) or 60)
        req = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=tmo) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, prompt: str, fmt: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": fmt,
        }
        try:
            with self._send(payload) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = _http_error_body(e)
            if e.code == 400 and fmt.get("type") == "json_schema" and self._looks_unsupported(body):
                raise _UnsupportedSchema(body) from e
            raise
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    @staticmethod
    def _looks_unsupported(body: str) -> bool:
        """判断 400 是不是「不支持这个档」，而不是「我这次请求有别的毛病」。

        分不清的话就会把一次真正的参数错误记成「永久降档」，
        之后所有请求都走无约束档、枚举约束静默失效（求知版踩过这个坑：
        日志里出现「未知动作」，且不重启恢复不了）。
        """
        low = (body or "").lower()
        keys = ("json_schema", "response_format", "structured output", "unavailable", "not support")
        return any(k in low for k in keys)

    @staticmethod
    def _looks_tools_unsupported(body: str) -> bool:
        """这个 400 是不是「不支持工具调用」，而不是「我这次请求有别的毛病」。

        分不清就会把一次参数错误记成「永久关掉工具箱」——
        而那会让她的手动不动就没了一整块。
        """
        low = (body or "").lower()
        keys = ("tools", "tool_choice", "function calling", "functions",
                "not support", "unsupported", "unavailable")
        return any(k in low for k in keys)

    def _warn_once(self, msg: str) -> None:
        """降级提示只打一次：每轮都刷屏会淹没真正重要的日志。"""
        if not self._warned:
            print(f"[llm] {msg}")
            self._warned = True
