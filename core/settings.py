"""外部配置的加载与保存：`config.local.json` + 环境变量 + 服务商预设。

**优先级：环境变量 > `config.local.json` > `config.py` 里的默认值。**

为什么单独一个模块，而不是塞进 `config.py`：
  - `config.py` 装的是「旋钮」（可调参数），这里装的是「凭据」（key / endpoint）——两回事
  - 凭据要能被仪表盘的「设置」页读写、要能脱敏展示、要能测试连通性，
    这些都不是常量定义该干的事

**key 绝不进版本库**：`config.local.json` 已加进 `.gitignore`。
开源项目里最丢人的事就是提交了一个带 key 的文件——哪怕后来删了，git 历史里还在。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L2 外部服务（配置）
#   上游    ：config（默认值）、store（原子写）
#   下游    ：dashboard（「设置」页）· 各启动入口（`apply()`）
#   对外入口：`load_local()` / `save_local()` / `apply()` / `apply_preset()` /
#             `describe()` / `local_config_path()` / `PRESETS` / `test_connection`
#   边界    ：读完就把结果交给 `config.CONFIG`，自己**不长期持有状态**
#             （和 `config` 的关系是"默认值在那边、敏感值在这边"）
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import config as cfgmod
from . import net
from .store import atomic_write_json

# 服务商预设：省得填的人去翻文档。
# 只预设 endpoint（那是查得到的事实）；model 给一个常见值，key 一律留空。
PRESETS = {
    "deepseek": {
        "label": "DeepSeek（国内直连）",
        # 实测可选：deepseek-flash / deepseek-v4-pro（`/v1/models` 查得到）。
        # endpoint 带 /v1：不带其实也通（服务端两种都收），带着更规范。
        "llm": {"endpoint": "https://api.deepseek.com/v1", "model": "deepseek-flash"},
        # ⚠️ DeepSeek **不提供 embedding 接口**，所以这里没有 embedding 预设。
        #    向量另配：本地 Ollama，或 OpenAI（见下面两家）。
        #    不配也能跑——检索会降级成字符重叠（`degraded_c1_threshold`），
        #    只是相关度变糙。
    },
    "openai": {
        "label": "OpenAI",
        "llm": {"endpoint": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
        "embedding": {"endpoint": "https://api.openai.com/v1",
                      "model": "text-embedding-3-small"},
    },
    "moonshot": {
        "label": "月之暗面 Kimi",
        "llm": {"endpoint": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    },
    "ollama": {
        "label": "本地 Ollama（离线）",
        # 本地服务不需要真 key，但 embedding 那套要一个非空字符串当占位
        "llm": {"endpoint": "http://127.0.0.1:11434/v1", "model": "qwen3.5:4b",
                "api_key": "ollama"},
        "embedding": {"endpoint": "http://127.0.0.1:11434/v1", "model": "bge-m3",
                      "api_key": "ollama"},
    },
}

# 环境变量名 → CONFIG 里的位置
_ENV_MAP = {
    "AIR_LINK_LLM_ENDPOINT": ("llm", "endpoint"),
    "AIR_LINK_LLM_API_KEY": ("llm", "api_key"),
    "AIR_LINK_LLM_MODEL": ("llm", "model"),
    "AIR_LINK_EMBEDDING_ENDPOINT": ("embedding", "endpoint"),
    "AIR_LINK_EMBEDDING_API_KEY": ("embedding", "api_key"),
    "AIR_LINK_EMBEDDING_MODEL": ("embedding", "model"),
}

# 也认求知版的环境变量名——两个版本能用同一套 shell 配置，少一次抄写。
# （两版只共享「服务」，不共享数据，见逻辑层 §1。）
_ENV_MAP_LEGACY = {
    "AIR2_LLM_ENDPOINT": ("llm", "endpoint"),
    "AIR2_LLM_API_KEY": ("llm", "api_key"),
    "AIR2_LLM_MODEL": ("llm", "model"),
    "AIR2_EMBEDDING_ENDPOINT": ("embedding", "endpoint"),
    "AIR2_EMBEDDING_API_KEY": ("embedding", "api_key"),
    "AIR2_EMBEDDING_MODEL": ("embedding", "model"),
}

# 打码串——**纯 ASCII 星号，不用圆点（`••••••••`）**：
# 圆点是 U+2022，GBK 控制台下一 print 就 UnicodeEncodeError
# （2026-09-16 实测：仪表盘启动横幅里打印打码 key，直接把启动搞崩）。
# 日志 / 终端 / 复制粘贴里都少一类编码坑。
_MASK = "********"
# 2026-09-16 之前用的圆点掩码：老配置里可能存着它（当时的防呆判据漏判过长 key），
# 仍要认得出来——认不出就会把打码值当成真 key 用。
_MASK_LEGACY = "••••••••"


def local_config_path():
    """`config.local.json` 的位置。**一个出口**——读、写、删除三件事共用它。"""
    return cfgmod.abspath(cfgmod.PATHS["local_config"])


def load_local() -> dict:
    """读 `config.local.json`；没有或坏了都返回空 dict（不抛）。

    坏了不抛是刻意的：一个手滑写坏的配置文件不该让整个程序打不开——
    顶多是「key 没读到」，那会在 `available()` 那里被明确地报出来。
    """
    path = local_config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[settings] {path.name} 读取失败（当作没配）: {e}")
        return {}


def _is_masked(value) -> bool:
    """这个值是不是「打码后的 key」而不是真 key。

    仪表盘的设置页会拿 `describe()` 的结果回填表单：用户只改 endpoint、
    点保存时，表单里那个打码串就被当作新 key 写回了文件——之后每一次请求
    都会拿着一个不是 key 的东西去鉴权，全部失败（真实踩过）。

    ⚠️ 判据必须认**夹在首尾明文之间**的形式：`describe()` 对长 key 给的是
    `key[:4] + _MASK + key[-4:]`，只比「整串等于掩码」会漏掉它——
    而漏判的代价正是把打码值当成真 key 存回去（2026-09-16 修）。
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return False
    if s == _MASK or _MASK in s:
        return True
    # 旧圆点掩码：整串、夹在明文之间、或任意长度的全圆点串都认
    return s == _MASK_LEGACY or _MASK_LEGACY in s or set(s) == {"•"}


def save_local(data: dict) -> None:
    """写回 `config.local.json`（仪表盘的「设置」页用）。**原子写 + 合并已有键**。

    两条都有具体理由：
      - **原子写**：这个文件里装着 key，`write_text` 写到一半崩掉
        （断电 / 被强杀）就是个半截 JSON，下次启动读不到任何配置。
        `store.atomic_write_json` 就是为这件事写的（临时文件 + fsync + 替换），
        它同样是这里该用的东西。
      - **合并已有键**：页面上没有的键（比如手写的 `insecure_ssl`）
        不该因为一次保存就被丢掉——表单没覆盖到的键，原样保留。

    防呆仍然在：打码值一律不落盘（打码串不是 key，存进去只会毁掉原来那把），
    遇到就打回磁盘上的原值。
    """
    existing = load_local()
    # **先保段，再并键**：表单只传 llm / embedding 两个段，磁盘上别的段
    # （手写的 `tts` / `search`）不该因为一次保存被抹掉——这是
    # 「表单没覆盖到的键不该被丢掉」的同一条道理，只是升到了段一级。
    data = {**existing, **data}
    for section in ("llm", "embedding"):
        sec = data.get(section)
        if not isinstance(sec, dict):
            continue
        # 先合并：磁盘上有、这次没传的键原样保留（保存动作不该顺手删配置）
        merged = {**(existing.get(section) or {}), **sec}
        data[section] = merged
        key = merged.get("api_key")
        # **空串和打码值都不是有效 key** ——都当作「这次没改」，保留磁盘上那把。
        # 少了这条，用户只想改 endpoint（key 框空着）就会把真 key 抹成空串，
        # 而症状是「保存之后就连不上了」，很难联想到是保存动作干的。
        if not key or _is_masked(key):
            old = ((existing.get(section) or {}).get("api_key") or "")
            if old:
                merged["api_key"] = old
            else:
                merged.pop("api_key", None)
    atomic_write_json(local_config_path(), data)


def apply_preset(name: str) -> dict:
    """把预设填进本地配置并保存，返回改动后的配置。"""
    preset = PRESETS.get(name)
    if not preset:
        raise ValueError(f"未知预设: {name}")
    data = load_local()
    for section in ("llm", "embedding"):
        if section in preset:
            data.setdefault(section, {})
            # 保留用户已经填过的 key：预设只补 endpoint / model
            key = data[section].get("api_key", "")
            data[section].update(preset[section])
            if key:
                data[section]["api_key"] = key
    save_local(data)
    apply()
    return data


def apply() -> None:
    """把外部配置合进 `CONFIG`。启动时调一次即可（幂等）。

    **优先级（低 → 高）**：
        config.py 默认值  <  `AIR2_*`（求知版遗留）  <  `config.local.json`  <  `AIR_LINK_*`

    legacy 环境变量排在**本版配置文件之下**，这条是踩出来的：
    `AIR2_*` 只是「顺便兼容求知版」才认的名字，可用户早年为求知版设过它，
    于是本版在设置页改成 DeepSeek 之后怎么都不生效——文件是对的，
    被一个他几乎不会想到的环境变量静默压住了。
    兼容来的名字只该做**兜底**，不该压过本版的显式配置。

    部署时想用环境变量覆盖仍然可以，但请用 `AIR_LINK_*`。
    """
    _apply_env(_ENV_MAP_LEGACY)      # 最低：兜底
    _apply_file()                    # 中间：本版显式配置
    _apply_env(_ENV_MAP)             # 最高：本版环境变量
    _warn_masked_keys()
    # 网络适配（core/net.py）：**loopback 永远直连**——挂 VPN（系统代理）时，
    # 本机的奥拉马 / 语音服务不能被代理走。装一次全局生效，幂等。
    net.install()


def _apply_file() -> None:
    """把 `config.local.json` 合进 `CONFIG`（只认下面这几个段与键）。

    `tts` 也在列：语音服务地址是**本机的事**（每台机器装没装都不一样），
    而"配置留在 Python 一侧"正是它不该由页面直连那个服务的理由之一。
    `search` 也在列：搜索走**独立的 Messages 端点**（工具箱第十一节），
    端点 / 模型 / 开关 / 通道（`channel`，旧名 `style`）都可能和主对话分开配
    （`model` 留空 = 沿用 `llm.model`）。
    `web` 同理：抓取（web_fetch）的开关 / 超时 / 上限也该能本地覆盖——
    代理不用配在这里（`urllib` 默认读 `HTTPS_PROXY` 等环境变量）。
    只认列出来的键：拼错的键不悄悄生效——那是「配置在说谎」那一类坑。
    """
    local = load_local()
    for section in ("llm", "embedding", "tts", "search", "web"):
        values = local.get(section) or {}
        if not isinstance(values, dict):
            continue
        for name in ("endpoint", "api_key", "model", "timeout"):
            if values.get(name):
                cfgmod.CONFIG[section][name] = values[name]
        # `insecure_ssl` / `search.enabled` / `web.fetch_enabled` 是布尔，
        # **存在即生效**（`false` 也要能覆盖默认值）——用真值判断会漏掉
        # 「显式关掉」这一步
        if section == "embedding" and "insecure_ssl" in values:
            cfgmod.CONFIG["embedding"]["insecure_ssl"] = bool(values["insecure_ssl"])
        if section == "search":
            if "enabled" in values:
                cfgmod.CONFIG["search"]["enabled"] = bool(values["enabled"])
            # 通道选择：`channel` 是正名（2026-09-26 起），`style` 是旧名
            # （2026-09-22 ~ 09-26）——**旧名搬进正名那个键**，不是并排留着：
            # 默认字典里 `channel` 有值（anthropic），并排留 `style` 的话
            # 读取端"channel 优先"就永远看不到旧配置写的 openai——老配置被
            # 默认值挡住而静默失效，「配置在说谎」那一类（这条踩过一次）。
            # 非空才覆盖（空串当"没给"，同 `_apply_env` 的道理）。
            # ⚠️ 这段整体曾经不在白名单里（写进文件被静默忽略，运行时还是
            # 默认的 anthropic）——症状：拿新家的 key 去打旧家的端点，401。
            # **以后加通道不用再动这里**：合法名单由 `search.py` 的通道表
            # 说了算，写错的名字会被指名报出来（不会静默回退默认通道）。
            if values.get("channel"):
                cfgmod.CONFIG["search"]["channel"] = str(values["channel"])
            elif values.get("style"):
                cfgmod.CONFIG["search"]["channel"] = str(values["style"])
        if section == "web":
            # 抓取段的键比别的段多（开关 / 上限 / UA）——**只认下面这几个名字**。
            # 加过又漏认的教训：`fetch_enabled` 曾不在名单里，文档承诺的开关
            # 写进文件被静默忽略——「配置在说谎」那一类，最不该在安全开关上发生。
            if "fetch_enabled" in values:
                cfgmod.CONFIG["web"]["fetch_enabled"] = bool(values["fetch_enabled"])
            for name in ("max_bytes", "max_chars", "max_url_chars"):
                if isinstance(values.get(name), int):
                    cfgmod.CONFIG["web"][name] = values[name]
            if values.get("user_agent"):
                cfgmod.CONFIG["web"]["user_agent"] = str(values["user_agent"])


def _apply_env(env_map: dict) -> None:
    """环境变量覆盖（优先级最高的一层）。**空串不算给了值**——
    `export X=` 是常见的"清掉它"写法，当成给了值就会把 key 覆盖成空串，
    于是请求 401，而配置页面上看着有内容。
    """
    for env_key, (section, name) in env_map.items():
        val = os.environ.get(env_key)
        if val:
            cfgmod.CONFIG[section][name] = val


def _warn_masked_keys() -> None:
    """生效的 key 是打码值 → 当场喊出来。

    这种错误不喊就会一直沉默：调用每次都失败，但失败信息是
    `UnicodeEncodeError`（谁也想不到是 key 的问题）。
    """
    for section in ("llm", "embedding"):
        key = (cfgmod.cfg(section, default={}) or {}).get("api_key") or ""
        if _is_masked(key):
            # 这行是「报警」——它自己绝不能崩在编码上（`⚠️` 这类 emoji 编不进 GBK），
            # 所以只用 ASCII / 中文。没有兜底的地方（demo / 脚本）也照常能喊出来。
            print(f"[settings] {section} 的 api_key 是打码值（{_MASK}），不是真 key；"
                  f"请重新填一次——否则每次请求都会失败")


def env_overrides(section: str) -> dict[str, str]:
    """哪些字段正被环境变量顶着（字段名 → 环境变量名）。

    这个信息的用途只有一个：**让「改了配置文件却不生效」当场可见**。
    用户上一次踩的坑就是它——文件里改成了 DeepSeek，env 里还留着
    另一家的 `AIR2_LLM_ENDPOINT`，生效的是 env，而他无从知道。
    """
    local = (load_local().get(section) or {})
    if not isinstance(local, dict):
        local = {}
    out: dict[str, str] = {}
    # 判定必须和 apply() 的优先级完全一致，否则界面会说谎：
    # 文件明明生效了，却提示「来自环境变量」。
    for env_map, beats_file in ((_ENV_MAP, True), (_ENV_MAP_LEGACY, False)):
        for env_key, (sec, name) in env_map.items():
            if sec != section or not os.environ.get(env_key):
                continue
            # 本版 env 总能赢；legacy 只在文件里没这一项时才兜底
            if beats_file or not local.get(name):
                out[name] = env_key
    return out


def describe(mask: bool = True) -> dict:
    """当前配置（给仪表盘显示用）。`mask=True` 时 key 打码。

    打码不是防谁——是防「截图发给别人看」和「日志里躺着一个真 key」。

    ⚠️ `mask=False`（真 key）**当前没有任何调用方**（2026-09-21 核过）：
    设置页的编辑靠「**留空 = 不改**」的约定（见 `save_local`），不需要把真 key
    回填进表单。别为了"让页面显示完整 key"去接它——回填过的值会被当成新值
    写回文件，而漏判打码值的代价是 2026-09-16 那次「每次请求都带着掩码符号
    去鉴权」的真事故。
    """
    out = {"presets": {k: v.get("label", k) for k, v in PRESETS.items()}}
    for section in ("llm", "embedding"):
        conf = cfgmod.cfg(section, default={}) or {}
        key = conf.get("api_key") or ""
        if mask:
            shown = (key[:4] + _MASK + key[-4:]) if len(key) > 12 else (_MASK if key else "")
        else:
            shown = key
        env = env_overrides(section)
        out[section] = {
            "endpoint": conf.get("endpoint") or "",
            "model": conf.get("model") or "",
            "api_key": shown,
            "configured": bool(conf.get("endpoint") and conf.get("model")),
            # 界面据此提示「此项来自环境变量，改文件不生效」
            "from_env": env,
            # 当前生效的 key 本身就不对（打码值 / 空）——比「通不通」更早暴露问题
            "key_looks_masked": _is_masked(key),
            # 证书校验是否被显式放宽（只有 embedding 段有这个开关）——
            # 它是「我不校验对端身份」这件事在界面上的可见性
            "insecure_ssl": bool(conf.get("insecure_ssl", False)),
        }
    return out


def test_connection(section: str = "llm") -> dict:
    """测一下配的服务通不通（仪表盘的「测试连接」按钮）。

    分两件不同的事，别混：
      - **通不通**（网络 / 鉴权）：这里答得了
      - **能不能用**（模型名对不对、有没有权限）：要看返回体，所以也看一眼
    """
    conf = cfgmod.cfg(section, default={}) or {}
    endpoint = (conf.get("endpoint") or "").rstrip("/")
    api_key = conf.get("api_key") or ""
    model = conf.get("model") or ""
    if not endpoint or not model:
        return {"ok": False, "detail": "endpoint 或 model 没填"}

    try:
        if section == "embedding":
            req = urllib.request.Request(
                f"{endpoint}/embeddings",
                data=json.dumps({"model": model, "input": ["连接测试"]}).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                method="POST")
            # 和 `EmbeddingService.embed()` 用**同一套**证书策略：
            # 测试比真实调用更严的话，会出现「测试说通、真跑却降级」这种怪事
            from .embedding import _unverified_ctx
            ctx = _unverified_ctx() if conf.get("insecure_ssl") else None
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            dim = len((data.get("data") or [{}])[0].get("embedding") or [])
            return {"ok": bool(dim), "detail": f"通了，向量维度 {dim}"}
        req = urllib.request.Request(
            f"{endpoint}/chat/completions",
            # 给 256 而不是 8：推理模型会先花掉一截在 `reasoning_content` 上，
            # 额度太小的话正文必空——测试连接会显示「通了，模型回了：(空)」，
            # 让人以为配置有问题（其实是额度不够）。
            data=json.dumps({"model": model,
                             "messages": [{"role": "user", "content": "说\"好\"一个字"}],
                             "max_tokens": 256}).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "detail": f"通了，模型回了：{content.strip()[:20] or '(空)'}"}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")[:200]
        except Exception:
            pass
        return {"ok": False, "detail": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
