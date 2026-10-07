"""联网搜索的**通道表**——一条通道 = 一份自描述（端点从哪来 / 请求怎么拼 / 响应怎么读）。

为什么要有这张表（2026-09-26）：原来两条通道硬编码在 `llm.search()` 的 if/else 里，
而且**服务商专属的知识**散在三处（请求形状在 `llm.search`、HTTP 头在 `llm._post_*`、
端点与模型在 `config.search`）——换一家的正确姿势是"把 `search()` 重写一遍"，
而重写就意味着再踩一遍已经踩过的坑（沿用 `search()` 里那些注释写的教训）。

现在**换家 = 换配置**（已知形态）或**只加一条描述**（新形态）：
`llm.search()` 是通用流程（取通道 → 拼请求 → 发一次 → 按通道读法解析 →
判"真的搜了没有"），**它不认识任何具体服务商**。

一条通道要说清五件事（前三个必填）：
  - `base_from`：端点取哪段配置——`"search"`（独立端点）或 `"llm"`（与对话同端）。
    两种接法都是实测里存在的（DeepSeek 要求分开配、MiMo 就是同一个端点），
    所以它得是通道自己的属性，不能由主流程规定。
  - `path`：接在端点后面的路径。
  - `build`：请求怎么拼（headers + payload）——**形态知识只在这里**。
  - `parse`：响应怎么读（正文 / 搜索词 / 结构化来源）——**判「搜到了没有」也在这里**：
    只认结构化结果块，从回复文本里凑数的那条纪律对所有通道一样（`llm.search()` 兜底）。
  - `hint`（可选）：这家专属的失败人话（如"账号没开通联网搜索"）——通用错误分类
    认不出服务商的私话，只有这条通道知道。

加一条新通道的模板（三步，**不动 `llm.search()`**）：
  1. 写 `_build_xxx(q, model, key)` / `_parse_xxx(data)` 两个纯函数；
  2. 在 `CHANNELS` 里加一项（名字 + 标签 + 两个函数 + 端点来源 + 路径）；
  3. 俗名进 `ALIASES`（可选）。
  测试照 `tests/test_phase6.py` 的 `SearchChannelTest`——那里有一条"假通道"
  演示注册之后 `search()` 立刻能用（证明主流程确实不认识服务商）。

边界：**只管"一次搜索怎么发、怎么读"**——工具档位（`web` 的只读）、限额（每轮两次）、
"结果不进记忆"那些纪律在 `tools.py` / `prompts.py`，不在这张表里。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L2 外部服务（搜索通道）——`llm.search()` 的按图索骥处
#   上游    ：config（`search.channel` / `search.endpoint` / 模型）
#   下游    ：llm（`search()` 调 `selected()` 取通道）
#   对外入口：`selected()`（当前配置的通道 + 认不出的引导）· `resolve()`
#   边界    ：**不碰网络、不碰配置写入**——纯描述 + 纯函数（好测）
# ---------------------------------------------------------------------
from __future__ import annotations

import urllib.error
from dataclasses import dataclass
from typing import Callable

from . import config as cfgmod

# 搜索那一次调用的指令：**照官方 harness 的原句**（`web-search-deepseek`
# provider 逐字发这一句）。那次调用也是一个模型在生成——用"回答我"的口吻
# 就等于有第二个人在替她说话；要的是**素材**，不是答案。
SEARCH_INSTRUCTION = "Perform a web search for the query: "

# 搜索的生成额度（所有通道共用）：搜索要「搜 → 读 → 再生成」，比普通回复耗 token。
# **只写这一个地方**——每条通道各写一份的话，调的时候必漏一处。
SEARCH_MAX_TOKENS = 4096


@dataclass(frozen=True)
class Channel:
    """一条搜索通道的自描述（见模块头五件事）。

    `build(query, model, key) -> (headers, payload)`；`parse(data) -> (text, queries, sources)`；
    `hint(body) -> str`（非空 = 这条通道认得的失败人话；空串 = 交回通用分类）。
    `used_by` 只服务配置时的人眼（"我现在该填哪个"），代码不读它。
    """

    name: str
    label: str
    base_from: str                      # "search"（独立端点）| "llm"（与对话同端）
    path: str
    build: Callable[[str, str, str], tuple[dict, dict]]
    parse: Callable[[dict], tuple[str, list[str], list[dict]]]
    hint: Callable[[str], str] | None = None
    used_by: str = ""


# =====================================================================
# 段 1：两条实测通道的请求拼装
# =====================================================================

def _build_anthropic(q: str, model: str, key: str) -> tuple[dict, dict]:
    """Anthropic Messages 形态（DeepSeek 走这条）。

    头照官方 provider 发：`x-api-key` 和 `Authorization: Bearer` **都带上**，
    官方端点认前者、Anthropic 兼容代理认后者，两个都发谁都能解
    （`anthropic-version` 也是官方 provider 的默认值）。
    """
    headers = {
        "x-api-key": key,
        "Authorization": f"Bearer {key}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": SEARCH_MAX_TOKENS,
        "messages": [{"role": "user",
                      "content": [{"type": "text",
                                   "text": f"{SEARCH_INSTRUCTION}{q}"}]}],
        # max_uses 照官方默认（5）：一次搜索里模型可能追着查几轮
        "tools": [{"type": "web_search_20250305",
                   "name": "web_search", "max_uses": 5}],
    }
    return headers, payload


def _build_openai(q: str, model: str, key: str) -> tuple[dict, dict]:
    """OpenAI 形态（MiMo 走这条）：`tools` 里声明 `{"type": "web_search"}`。

    正文用字符串（不是 Anthropic 那种块数组）——服务端搜索的声明在 `tools`，
    开不开通是**账号级**的事（没开通时 400，见 `_hint_openai`）。
    """
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}
    payload = {"model": model, "max_tokens": SEARCH_MAX_TOKENS,
               "messages": [{"role": "user",
                             "content": f"{SEARCH_INSTRUCTION}{q}"}],
               "tools": [{"type": "web_search"}]}
    return headers, payload


# =====================================================================
# 段 2：两条实测通道的响应解析（纯函数，跟网络分开、好测）
# =====================================================================

def _parse_anthropic(data: dict) -> tuple[str, list[str], list[dict]]:
    """从 Messages 响应里取正文 / 搜索词 / 来源（标题 + URL）。

    响应是 `content` 块数组：
      - `server_tool_use`：服务端工具调用记录（搜了什么词）；
      - `web_search_tool_result`：**结构化结果**（`web_search_result` 条目）；
      - `text`：模型写的汇报（她用的素材）。
    「只取结构化块，绝不从回复文本里抓 URL」是官方的原话——同一条纪律。

    **没有 `web_search_tool_result` 块 = 这次没有真的搜**（调用方判失败）：
    模型在没有搜索工具时照样会生成文本（实测：一句「没有搜索工具」的自白、
    安全审核标签、或把工具调用演成正文），那些都不是结果。
    """
    text, queries, sources = "", [], []
    for block in (data or {}).get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = (text + "\n" + (block.get("text") or "")).strip()
        elif kind == "server_tool_use":
            inp = block.get("input") if isinstance(block.get("input"), dict) else {}
            raw = inp.get("query") or inp.get("queries") or []
            for q in ([raw] if isinstance(raw, str) else raw):
                q = str(q).strip()
                if q:
                    queries.append(q)
        elif kind == "web_search_tool_result":
            for item in block.get("content") or []:
                if not isinstance(item, dict) or item.get("type") != "web_search_result":
                    continue
                url = str(item.get("url") or "").strip()
                if url:
                    sources.append({"url": url,
                                    "title": str(item.get("title") or "").strip()})
    return text, queries, sources


def _parse_openai(data: dict) -> tuple[str, list[str], list[dict]]:
    """从 **OpenAI 形态**（`/chat/completions` + `tools:[{"type":"web_search"}]`）
    的响应里取素材与来源——MiMo 走这条（2026-09-22 实测）。

    与 Anthropic 版的差异（同一条纪律的另一半）：
      - `choices[0].message.content`：正文（素材）；
      - `choices[0].message.annotations[]`：**结构化引用**（`url_citation`，
        带 url / title；还会带 site_name / publish_time / summary——先只取
        前两样，多的留给将来）；
      - `usage.web_search_usage.tool_usage`：服务端**真实搜索计数**
        （判「真的搜了没有」的正式依据）。

    没有 `web_search_usage`、也没有 annotations = **这次没有真的联网查**
    （同 Anthropic 版：服务端没执行搜索时，模型照样会写一段文本，那不是结果）。
    `queries`（实际搜了什么词）这边拿不到——服务端不回报，由调用方用请求词兜底。
    """
    msg = (((data or {}).get("choices") or [{}])[0] or {}).get("message") or {}
    text = str(msg.get("content") or "").strip()
    sources = []
    for a in (msg.get("annotations") or []):
        if not isinstance(a, dict):
            continue
        url = str(a.get("url") or "").strip()
        if url:
            sources.append({"url": url, "title": str(a.get("title") or "").strip()})
    usage = ((data or {}).get("usage") or {}).get("web_search_usage") or {}
    try:
        searched = int(usage.get("tool_usage") or 0) > 0
    except (TypeError, ValueError):
        # 服务端给了非数字（实测里没见过）——当没搜，宁可判失败。
        # ⚠️ `llm.search()` 现在也兜住通道的解析异常，但**别依赖那条兜底**：
        # 兜底给的是一句"读不懂这次的响应"，而这里判"没搜"更准（同一条纪律的
        # 落地：拿不准就当没搜，绝不从文本里凑数）。
        searched = False
    if not (searched or sources):
        return text, [], []            # 没搜：sources 置空，调用方判失败
    return text, [], sources


# =====================================================================
# 段 3：服务商专属的失败知识（通用分类认不出的那些私话）
# =====================================================================

def _hint_openai(body: str) -> str:
    """MiMo：账号**没开通**联网搜索（400 + `webSearchEnabled is false`）。

    为什么值得单独认：这不是"服务没这个能力"，是**还没开**——开通后重试即可，
    不该像"不支持"那样被永久关掉（记错就要重启才恢复）。
    判据放宽到小写包含：这话是服务商私有的措辞，多认几种写法比漏认安全
    （漏认的代价是把一次"去开一下"的提示说成"没查到"）。
    """
    if "websearchenabled" in (body or "").lower():
        return ("这个账号还没开通联网搜索（去平台控制台开通后直接重试，"
                "不用重启）")
    return ""


# =====================================================================
# 段 4：通道表
# =====================================================================

CHANNELS: dict[str, Channel] = {
    "anthropic": Channel(
        name="anthropic",
        label="Anthropic Messages + 原生 web_search 服务端工具",
        # 官方明确要求搜索端点与对话端点**分开配**，不要拼 chat 那条
        # （`/v1` 前缀的坑在搜索这里不存在）
        base_from="search",
        path="/messages",
        build=_build_anthropic,
        parse=_parse_anthropic,
        used_by="DeepSeek（官方 harness 的 web-search-deepseek 同款）",
    ),
    "openai": Channel(
        name="openai",
        label="OpenAI 形态 /chat/completions + tools:[{type: web_search}]",
        # 搜索与对话**同端**：独立端点的默认值是 anthropic 通道的，
        # 误用它会把请求发去别人的端点（拿新家的 key 打旧家的端点，401）
        base_from="llm",
        path="/chat/completions",
        build=_build_openai,
        parse=_parse_openai,
        hint=_hint_openai,
        used_by="小米 MiMo（2026-09-22 实测；需先在平台开通联网搜索）",
    ),
}

# 俗名 → 正名。认俗名是为了让人按"我家叫什么"来填，而不是背这张表的内部命名；
# 但**表里的正名只有一个**（别名不产生第二条通道，免得同一件事两条维护线）。
ALIASES: dict[str, str] = {
    "messages": "anthropic",
    "anthropic-messages": "anthropic",
    "claude": "anthropic",
    "deepseek": "anthropic",
    "chat": "openai",
    "openai-chat": "openai",
    "chat-completions": "openai",
    "mimo": "openai",
}

DEFAULT = "anthropic"      # 没配时走这条（= 这些注释写就时项目主用的那家）


def resolve(name: str) -> tuple[Channel | None, str]:
    """按名字取通道。返回 `(通道, 错误说明)`；取不到时通道为 None。

    空名字 = 没配过 = 默认通道（这是"缺省"不是"错"）。
    认不出 **不回退默认**——静默回退的症状是"配置写着 A、实际走了 B"，
    这种谎比一次明确的失败贵得多（`settings.py` 里"配置在说谎"那类坑的亲戚）。
    """
    key = (name or "").strip().lower()
    if not key:
        return CHANNELS[DEFAULT], ""
    ch = CHANNELS.get(key) or CHANNELS.get(ALIASES.get(key, ""))
    if ch is None:
        available = " / ".join(sorted(CHANNELS))
        alias = " / ".join(sorted(ALIASES))
        return None, (f"认不出搜索通道「{name}」——可用的是 {available}"
                      f"（俗名也认：{alias}）")
    return ch, ""


def selected() -> tuple[Channel | None, str]:
    """**这次该走哪条通道**（从 `search.channel` 读）。

    只有一个键要看：旧名 `style`（2026-09-22 ~ 09-26 期间用的那个）在
    `settings._apply_file` 里被搬进 `channel`——**不是并排读两个**：
    并排读的话"正名优先"会被默认值挡住，老配置写 `style: openai` 而
    `channel` 还是默认的 anthropic，人看到的就是"配置没生效"。
    """
    return resolve(str(cfgmod.cfg("search", "channel") or ""))


def looks_unsupported(err: Exception) -> bool:
    """这次失败是「没有这个能力」还是「这次调用出错」？

    分不清就会把一次网络抖动记成「永久关掉搜索」——
    同 `llm._looks_unsupported` 那条经验：**误判的代价是永久的**（要重启才恢复）。
    只按状态码判，不读响应体（体是一次性的，读完通道的 `hint` 就没得读了——
    顺序在 `llm.search()` 里：先 `hint` 后这里）。
    """
    if isinstance(err, urllib.error.HTTPError):
        return err.code in (400, 404, 405, 501)
    return False
