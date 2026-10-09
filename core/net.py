"""出站网络适配：代理从环境（含系统设置）来，但 **loopback 永远直连**。

为什么需要这一层（2026-09-18，用户报的真事）：挂 VPN 时（尤其开了
"系统代理"的），`urllib` 会把对 `127.0.0.1` 的请求也交给代理——于是本机的
奥拉马（语义模型）、语音服务全"连不上"。它们明明就在本机。浏览器通常没事，
因为系统代理设置里带着"绕过本地地址"；而 Python 这边这条豁免要我们自己
保证（DeepSeek Harness 的 http-proxy 同款规则：loopback 始终被绕过——
`localhost`、整个 `127.0.0.0/8`、`::1`、`0.0.0.0`，含 IPv4 映射写法）。

开源语境（做这一层的另一个理由）：别人家的代理环境五花八门——这层
**不发明新配置**，就用 `urllib` 的标准读取（`getproxies()`：环境变量，
Windows / macOS 上还认系统设置），只加一条它没有的：loopback 豁免。
`NO_PROXY` 一类照常生效（走 `proxy_bypass`）。

四个出口：
  - `install()`：重建并装上**全局** opener——所有已有的 `urlopen` 调用点
    （llm / embedding / 语音客户端）不用逐个改就受益。装点有两处：
    `core/__init__.py`（**import 即生效**——demo / 实验回放等不走设置页的
    路径也覆盖）与 `settings.apply()`（保存设置时顺手重读一次代理环境）。
    **重装即重读**：`ProxyHandler` 的代理清单是构造时快照（urllib 固有；
    harness 同样是"启动时解析一份策略"）——运行中开关 VPN，重装 / 重启
    才跟随；loopback 豁免不受此影响（每次请求都判）。
  - `open(req, timeout=…)`：显式入口——`context` 分支（放宽证书那条）必须
    走它：`urlopen` 在给了 context 时会自己 build 一个 opener、**绕开全局**
    的那份，loopback 豁免就没了。
  - `proxied(url)`：这次请求会不会走代理——**与安装的 opener 同一判据**，
    `webfetch` 的连接固定靠它决定 pin 不 pin（两处判据一旦漂移就会错连）。
  - `is_loopback(host)`：判定本身（纯函数，好测）。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L2 外部服务（出站网络）——所有网络调用的公共底座
#   上游    ：标准库 urllib（无项目内依赖）
#   下游    ：core/__init__（import 即装）、settings（保存时重装）、embedding（走 `open()`）、
#             webfetch（`proxied`）；llm / 语音客户端经全局 opener 受益（不 import 它）
#   对外入口：`install()` / `open()` / `proxied()` / `is_loopback()`
#   边界    ：**只做"走哪条路"**——超时 / 重试 / 证书 / 上限各自在调用方，
#             这里不替它们做主
# ---------------------------------------------------------------------
from __future__ import annotations

import ipaddress
import urllib.parse
import urllib.request

_opener: urllib.request.OpenerDirector | None = None


def is_loopback(host: str) -> bool:
    """本机地址吗——loopback（`127.0.0.0/8`、`::1`）与 `0.0.0.0`，含 `localhost`。

    只做**字面**判定（`localhost` 这个名字也算）：不做 DNS 解析——
    解析会引入"校验时是本机、连接时不是"的另一类问题；而这里需要的
    只是"一眼看出来的本机地址不绕代理"，够用且没有副作用。
    """
    h = (host or "").strip().strip("[]").strip().lower()
    if not h:
        return False
    if h == "localhost" or h.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_loopback or ip.is_unspecified)


class _LoopbackDirectProxy(urllib.request.ProxyHandler):
    """标准代理规则 + 一条：loopback 直连。

    `proxy_open` 返回 None = "这个 handler 没处理"——`urllib` 会落回直连的
    HTTP / HTTPS handler。这是 `ProxyHandler` 的公开约定（它自己的
    `proxy_bypass` 分支就是这么返回的），不是这里发明的东西。

    ⚠️ 一个容易误会的地方（实测确认过）：**环境里没有代理时，这个实例不在
    opener 的 `handlers` 列表里**——`ProxyHandler` 是按代理清单动态挂
    `*_open` 钩子的，一个钩子都没有的 handler 会被 `add_handler` 静默跳过
    （默认的 `ProxyHandler()` 在无代理时同样如此）。这是它正常工作的一部分
    （没代理就没有"绕不绕"的问题），不是"没装上"。
    """

    def proxy_open(self, req, proxy, type):
        if is_loopback(urllib.parse.urlsplit(req.full_url).hostname or ""):
            return None
        return super().proxy_open(req, proxy, type)


def opener() -> urllib.request.OpenerDirector:
    """带规则的 opener（懒建单例——建一次装一次，重复装的是同一个对象）。"""
    global _opener
    if _opener is None:
        _opener = urllib.request.build_opener(_LoopbackDirectProxy)
    return _opener


def install() -> None:
    """重建并装上**全局** opener（重装 = **重读代理环境**）。

    为什么动全局：几个出点（llm / embedding / 语音客户端）用的是
    `urllib.request.urlopen`——它读的正是全局 opener。换一次，所有调用点
    （含以后新写的）都在这条规则下；逐个改调用点迟早会漏一个，而漏掉的
    那个恰好会在"挂了 VPN"的用户那里翻车。

    为什么是"重建"而不是复用单例：`ProxyHandler` 的代理清单是**构造时
    快照**（urllib 的固有设计）——运行中开了 / 关了代理（挂断 VPN）不会
    自动跟随。重装一次就重读一次，于是"保存设置"（`settings.apply()`）
    顺手带上了刷新语义；没有别的口子时，重启即恢复。loopback 豁免不受
    快照影响——它每次请求都判。
    """
    global _opener
    _opener = urllib.request.build_opener(_LoopbackDirectProxy)
    urllib.request.install_opener(_opener)


def open(req, timeout=None, context=None):
    """显式入口：带规则的 opener。

    `context`（放宽证书的 `insecure_ssl` 那条路）必须显式走这里：
    `urllib.request.urlopen` 在给了 `context` 时会**自己 build 一个 opener**、
    绕开全局那份——loopback 豁免在那条路上会悄悄失效（而自签 / 内网端点
    恰恰是最容易被 VPN 代理坑的一类）。
    """
    if context is None:
        return opener().open(req, timeout=timeout)
    op = urllib.request.build_opener(
        _LoopbackDirectProxy,
        urllib.request.HTTPSHandler(context=context))
    return op.open(req, timeout=timeout)


def proxied(url: str) -> bool:
    """这次请求会不会被代理接管——判据与**刚构造的** `ProxyHandler` 一致。

    "刚构造"是刻意的：`webfetch` 每次抓取都新建 opener（新快照 = 当下
    环境），所以这里按**实时**读（`getproxies`）才对得上它——对不上就会在
    "假定直连去 pin、实际走代理"处连错对象（`webfetch` 的注释里写过）。
    三条对齐全在这里：
      1. loopback 豁免同 `_LoopbackDirectProxy`；
      2. 用 `Request(url)` 造出与它手里一模一样的对象——`req.host` 是
         `host[:port]`（带端口），`urlsplit().hostname` 不带，`no_proxy`
         匹配上这点差别足以给出两个答案；
      3. 代理来源与绕过同 `getproxies()` / `proxy_bypass()`（标准读取，
         环境变量 + 系统设置）。
    """
    try:
        req = urllib.request.Request(url)
    except Exception:
        return False
    if is_loopback(urllib.parse.urlsplit(url).hostname or ""):
        return False
    try:
        if not urllib.request.getproxies().get(req.type):
            return False
        return not urllib.request.proxy_bypass(req.host)
    except Exception:
        return False
