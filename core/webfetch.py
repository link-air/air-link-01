"""取网页（`web` 的 url 分支）——把「她打不开的链接」变成能读的文本。

照 DeepSeek Harness 的 `dsh-web-fetch-http` 设计（2026-09-18），四条边界：

  1. **URL 校验**：只收 http / https、不带内嵌凭据（`user:pass@`）、长度有上限。
  2. **只许公共地址**：域名解析出的**每一个**地址都必须是公共单播，否则整体
     拒绝——她读不到 127.0.0.1 上的本地服务（奥拉马、语音服务、仪表盘都在那儿）。
     直连时把连接**固定**在已校验的地址上（防 DNS 重绑定：校验时解析到公网、
     连接时解析到内网的经典穿墙法）。
  3. **有界**：仅同源重定向（≤5 跳）、响应字节上限、解码字符上限、超时；
     只收文本类内容（HTML / text / JSON / XML），二进制与不声明类型的一律拒绝。
  4. **匿名**：不发送任何凭据——抓的是公共页面，不该带上 air 的 key。

代理：**不挂 VPN 时靠代理出网**。判据统一在 `core/net.py`（`net.proxied`——
和全局安装的 opener 同一套规则）：必须同判据，不一致就会出现「假定直连去
pin、实际走代理」的错连。loopback 由 `net` 保证**永远直连**，所以这里
"代理生效"时面对的都是"外面"的地址；而非公共地址（含 IP 字面量）在两种
情形下都**永远拒绝**——代理也救不了那种 SSRF。

与 harness 的一处差异：重定向**只跟同源**（跨源变成一次明确的失败，要求
重新调用）——这也是连接固定能成立的前提：换主机就得重新解析、重新校验。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L9 工具箱（她的手）——`web` 的 url 分支实现；动作表在 `tools.py`
#   上游    ：config（超时 / 上限 / UA）
#   下游    ：tools（`_web_fetch` 薄封装）
#   对外入口：`fetch()`（一个函数走完全程）+ `to_text()`（HTML → 文本，纯函数）
#   边界    ：**不改记忆、不进 trace**——抓来的只是这一轮说话的燃料
#             （同 `web_search`：来源不能是「网上说的」）
# ---------------------------------------------------------------------
from __future__ import annotations

import base64
import gzip
import http.client
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from functools import partial
from html.parser import HTMLParser

from . import config as cfgmod
from .net import proxied

# 重定向的编码：urllib 把它们抛成 HTTPError——要认出「这是跳转、不是结果」
_REDIRECT_CODES = (301, 302, 303, 307, 308)


# =====================================================================
# 段 1：纯函数——URL / 地址 / 文本（不碰网络，好测）
# =====================================================================

def _validate_url(url: str, max_url: int = 2048) -> str:
    """URL 校验。返回错误说明；空串 = 通过。

    为什么连长度都要管：超长 URL 本身就是攻击面（日志膨胀、解析歧义），
    harness 把上限写死在 2048，照抄。
    """
    if not url:
        return "没说抓哪个地址"
    if len(url) > max_url:
        return f"地址太长（上限 {max_url} 字符）"
    try:
        p = urllib.parse.urlsplit(url)
        _ = p.port                      # 端口非法会在这里抛
    except ValueError:
        return "这个地址读不出来（拼写有问题）"
    if p.scheme not in ("http", "https"):
        return "只认 http / https 开头的地址"
    if not p.hostname:
        return "地址里没有主机名"
    if p.username or p.password:
        return "地址里带了用户名密码——不抓这种（不发凭据）"
    return ""


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address((host or "").strip("[]"))
        return True
    except ValueError:
        return False


def _is_public_ip(ip: str) -> bool:
    """公共单播地址才放行——本机 / 内网 / 保留段全拒。

    用 `is_global`（标准库的「全球可达」判定）而不是手写网段表：网段表会漏
    （CGNAT 100.64/10、IPv6 的 fc00::/7……），`is_global` 是标准库维护的
    同一份口径。IPv4-mapped IPv6（`::ffff:127.0.0.1`）先还原成 IPv4 再判——
    不还原的话它会长得像公网 IPv6 地址。
    """
    try:
        a = ipaddress.ip_address((ip or "").strip("[]"))
    except ValueError:
        return False
    if a.version == 6 and a.ipv4_mapped is not None:
        a = a.ipv4_mapped
    return bool(a.is_global)


def _resolve_public(host: str, port: int) -> tuple[list[str], str]:
    """解析域名并校验**全部**地址。返回 `(地址表, 错误说明)`。

    为什么是"全部"：DNS 可以一次回多个 A 记录，混一个内网地址进来、连接时
    选中它就穿墙了。规则照 harness：**只要有一个非公共地址，拒绝整个结果**。
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        return [], f"这个域名解析不出来（{e}）"
    ips: list[str] = []
    for info in infos:
        ip = (info[4][0] or "").split("%")[0]      # 去掉 IPv6 的 scope 后缀
        if not ip or ip in ips:
            continue
        if not _is_public_ip(ip):
            return [], "它解析到了本机或内网地址——不抓（不让页面把手伸到本地服务）"
        ips.append(ip)
    if not ips:
        return [], "这个域名解析不出地址"
    return ips, ""


class _TextExtractor(HTMLParser):
    """HTML → 纯文本的收集器：块级断行、标题带 #、列表带 -，script 不要。"""

    _BLOCK = frozenset((
        "p", "div", "section", "article", "header", "footer", "main",
        "ul", "ol", "li", "table", "tr", "td", "th", "blockquote",
        "pre", "br", "hr", "form", "nav", "aside", "figure", "figcaption",
        "h1", "h2", "h3", "h4", "h5", "h6",
    ))
    # 跳过的是"不是内容"的标签：它们的文字（JS 源码 / CSS）进正文只是噪声
    _SKIP = frozenset(("script", "style", "noscript", "svg", "template", "iframe"))

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._depth = 0                    # >0 = 正在跳过的元素里面

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1
            return
        if self._depth:
            return
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n" + "#" * int(tag[1]) + " ")
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        # `<br />` 这类自闭合：HTMLParser 不给 endtag，要自己断行
        if not self._depth and tag in ("br", "hr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._depth = max(0, self._depth - 1)
            return
        if self._depth:
            return
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._depth:
            self.parts.append(data)


def to_text(html: str) -> str:
    """HTML → 可读文本，纯函数。

    不做完整 markdown 转换（harness 那边用 turndown，是 JS 生态的库；标准库
    没有对应的）：只求"读得懂"——标题带 #、列表带 -、块级断行、压缩空白。
    畸形 HTML 不抛：拿到多少算多少（`HTMLParser` 本身容错，这里再兜一层）。
    """
    ex = _TextExtractor()
    try:
        ex.feed(html or "")
        ex.close()
    except Exception:
        pass
    s = "".join(ex.parts)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _unwrap_base64_json(text: str) -> str:
    """整包 JSON、正文却在 `content` 里 base64 编码 → 拆出可读的正文。

    这是 GitHub 内容 API（`/repos/{o}/{r}/readme`、`/contents/…`）的固定
    结构，也算一类通用形态。**起因（2026-09-18 实证）**：她抓 README 时
    `raw.githubusercontent.com` 被间歇阻断（TLS 握手超时），能通的
    `api.github.com/.../readme` 返回的就是这种 JSON——不拆开，她拿到的是
    编码后的乱码（再被 max_chars 截一刀，等于没有一条路读得到）。

    判定收得很紧（三条全要）：顶层是对象、`encoding` 恰为 `"base64"`、
    `content` 是非空字符串且解得开、解出的是非空 UTF-8 文本。任何一条
    不符**原样放行**——只认识这一种结构，不做"凡 JSON 都翻"的猜测。
    """
    try:
        d = json.loads(text)
    except Exception:
        return text
    if not isinstance(d, dict) or d.get("encoding") != "base64":
        return text
    content = d.get("content")
    if not isinstance(content, str) or not content.strip():
        return text
    try:
        decoded = base64.b64decode(content).decode("utf-8")
    except Exception:
        return text
    return decoded if decoded.strip() else text


def _is_text_type(ct: str) -> bool:
    """文本类内容才收（同 harness 的白名单口径）。

    **不声明类型 = 不知道是不是二进制 → 拒绝**：宁可不抓，也不把一段乱码
    塞进她的上下文里。
    """
    if not ct:
        return False
    if ct.startswith("text/"):
        return True
    if ct in ("application/json", "application/xml", "application/xhtml+xml"):
        return True
    return ct.endswith("+json") or ct.endswith("+xml")


# =====================================================================
# 段 2：网络——连接固定、同源重定向、一次抓取
# =====================================================================

def _connect_any(addresses, timeout, source_address):
    """按已校验的地址表逐个尝试连接——**连接只连这些 IP，不再做 DNS**。"""
    err: OSError | None = None
    for ip, port in addresses:
        try:
            return socket.create_connection((ip, port), timeout, source_address)
        except OSError as e:
            err = e
    raise err or OSError("连不上")


class _PinnedConnection:
    """混入：把 `_create_connection` 换成"只连已校验地址"的那版。

    ⚠️ 必须**在实例上**覆盖：`http.client` 的 `__init__` 里把
    `self._create_connection` 设成了实例属性（`socket.create_connection`），
    子类里定义一个同名方法没用——会被实例属性盖掉。
    """

    def __init__(self, *args, pin=(), **kw):
        self._pin = list(pin)
        super().__init__(*args, **kw)
        self._create_connection = self._pinned_connect

    def _pinned_connect(self, address, timeout=None, source_address=None):
        port = address[1]                  # 端口来自 URL；主机名忽略，用 pin 的 IP
        return _connect_any([(ip, port) for ip in self._pin], timeout, source_address)


class _PinnedHTTPConnection(_PinnedConnection, http.client.HTTPConnection):
    pass


class _PinnedHTTPSConnection(_PinnedConnection, http.client.HTTPSConnection):
    pass


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, addresses):
        super().__init__()
        self._addresses = addresses

    def http_open(self, req):
        return self.do_open(partial(_PinnedHTTPConnection, pin=self._addresses), req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, addresses):
        super().__init__()
        self._addresses = addresses

    def https_open(self, req):
        # context 是父类的（默认 SSL context，证书校验照常）——
        # SNI / 证书针对原主机名（http.client 自己负责），连接连的是 pin 的 IP
        return self.do_open(partial(_PinnedHTTPSConnection, pin=self._addresses),
                            req, context=self._context)


def _same_origin(a: str, b: str) -> bool:
    """scheme + host + port 全同（端口按 scheme 补默认值）。"""
    def norm(u: str):
        p = urllib.parse.urlsplit(u)
        port = p.port or (443 if p.scheme == "https" else 80)
        return (p.scheme, (p.hostname or "").lower(), port)

    try:
        return norm(a) == norm(b)
    except ValueError:
        return False


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """只跟同源跳转；跨源**不跟**——让它变成一次明确的失败。

    换主机就意味着换一次抓取：目标要重新解析、重新校验（pin 也就跟着换），
    那属于「另一次调用」。harness 同款（"跨源重定向会失败并要求重新调用"）。
    `redirect_request` 返回 None → urllib 抛 HTTPError(30x)，由 `fetch` 翻成人话。
    """

    max_repeats = 3          # 同一条地址重复跳的上限
    max_redirections = 5     # 整条链的上限

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _same_origin(req.full_url, newurl):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# 代理判定不在这里——统一收在 `core/net.py` 的 `proxied()`（与全局安装的
# opener 同一套规则，含 loopback 永远直连）。此处曾有一份自己的实现，与
# `ProxyHandler` 的判据漂移过一次（hostname 不带端口 vs req.host 带端口）——
# 修的时候收回去：**同一件事只留一份实现**，它才没机会再漂。


def _build_opener(addresses: list[str] | None):
    """装这次请求要用的 opener。

    `addresses is None`（走代理）：默认连接 + 同源重定向；
    有地址（直连）：换成 pin 版的连接 handler——它们是 `HTTPHandler` /
    `HTTPSHandler` 的子类实例，`build_opener` 会自动跳过默认的那两个，
    不会出现两套并存。
    """
    if addresses is None:
        return urllib.request.build_opener(_SameOriginRedirect)
    return urllib.request.build_opener(
        _PinnedHTTPHandler(addresses), _PinnedHTTPSHandler(addresses),
        _SameOriginRedirect)


def _headers() -> dict:
    return {
        # 单一来源：值只在 `config.web.user_agent` 写一份（原来这里抄了
        # 同值兜底——config 一改就漂移，2026-10-10 收口）
        "User-Agent": str(cfgmod.cfg("web", "user_agent") or ""),
        # 不发 Accept-Encoding：让服务器别压缩（省掉解压这一环）；
        # 真收到 gzip 也会解（有的服务器不看声明）。
        "Accept": "text/html,application/xhtml+xml,text/*;q=0.9,"
                  "application/json;q=0.8",
    }


def _header(headers, name: str, default: str = "") -> str:
    try:
        v = headers.get(name)
    except Exception:
        v = None
    return v if v is not None else default


def _build_result(out: dict, status: int, headers, raw: bytes,
                  max_bytes: int, max_chars: int) -> dict:
    """把响应拼成结果（类型检查 / 解压 / 解码 / HTML 转文本 / 截断）。"""
    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    ct = _header(headers, "Content-Type").split(";")[0].strip().lower()
    if not _is_text_type(ct):
        out["error"] = f"这不是文本内容（{ct or '没声明类型'}）——不抓二进制或不明内容"
        return out
    enc = _header(headers, "Content-Encoding").lower()
    if "gzip" in enc:
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass                       # 解不开就按原样试读（下面 errors=replace）
    charset = ""
    try:
        charset = headers.get_content_charset() or ""
    except Exception:
        charset = ""
    text = raw.decode(charset or "utf-8", errors="replace")
    if "html" in ct:
        text = to_text(text)
    if "json" in ct:
        # 拆包要在截断**之前**：不然截的是 base64 串（见 `_unwrap_base64_json`）
        text = _unwrap_base64_json(text)
    if len(text) > max_chars:
        text = (text[:max_chars]
                + "\n\n…（内容太长被截断——要后面的部分，换更具体的地址再抓）")
        truncated = True
    out.update(ok=True, status=status, text=text, truncated=truncated)
    return out


def fetch(url: str, *, opener=None, timeout=None, max_bytes=None,
          max_chars=None, max_url_chars=None) -> dict:
    """抓一个页面。**不抛异常**——失败给 `error`。

    返回 `{ok, url, final_url, status, text, truncated, error}`。
    非 2xx **是结果不是错误**（404 也是一种回答，状态码是被抓资源的属性，
    同 harness）——只有"取不到 / 不能安全地取"才算失败。

    `opener` 可注入（测试用）；给了就不碰网络栈的构造，`fetch` 仍走
    完整流程（校验 → 结果拼装）。
    """
    url = (url or "").strip()
    out = {"ok": False, "url": url, "final_url": "", "status": 0,
           "text": "", "truncated": False, "error": ""}
    # 四个上限只写 `config.web` 一份（原来这里抄了同值兜底——config 改了会漂移）
    max_url_chars = int(max_url_chars or cfgmod.cfg("web", "max_url_chars"))
    err = _validate_url(url, max_url_chars)
    if err:
        out["error"] = err
        return out
    max_bytes = int(max_bytes or cfgmod.cfg("web", "max_bytes"))
    max_chars = int(max_chars or cfgmod.cfg("web", "max_chars"))
    tmo = float(timeout or cfgmod.cfg("web", "timeout"))

    p = urllib.parse.urlsplit(url)
    host = p.hostname or ""
    try:
        port = p.port or (443 if p.scheme == "https" else 80)
    except ValueError:
        out["error"] = "端口不对"
        return out

    addresses: list[str] | None = None
    if proxied(url):
        # 代理接管：代理解析目标（本地无法也不该解析）——但 IP 字面量的
        # 非公共地址照样拒绝（代理照发不误，那是最直白的一条 SSRF）
        if _is_ip_literal(host) and not _is_public_ip(host):
            out["error"] = "这个地址指向本机或内网——不抓。"
            return out
    elif _is_ip_literal(host):
        if not _is_public_ip(host):
            out["error"] = "这个地址指向本机或内网——不抓（不让页面把手伸到本地服务）"
            return out
        addresses = [host]
    else:
        addresses, rerr = _resolve_public(host, port)
        if rerr:
            out["error"] = rerr
            return out

    if opener is None:
        opener = _build_opener(addresses)
    req = urllib.request.Request(url, headers=_headers())
    try:
        with opener.open(req, timeout=tmo) as resp:
            out["final_url"] = resp.geturl() or url
            raw = resp.read(max_bytes + 1)     # +1：用来发现"其实还没读完"
            return _build_result(out, resp.status, resp.headers, raw,
                                 max_bytes, max_chars)
    except urllib.error.HTTPError as e:
        if e.code in _REDIRECT_CODES:
            # 两种情形共用 30x：跨源（不跟——那要重新解析、重新校验）与跳得太多。
            # **把跳转目标带给她**："换成最终那个地址"得有个地址可换，否则她只能
            # 干瞪眼（jsDelivr → fastly 节点这类跨源跳转很常见）。
            loc = ""
            try:
                loc = (e.headers.get("Location") or "").strip() if e.headers else ""
            except Exception:
                loc = ""
            out["error"] = ("它跳去了别的站点、或者一直跳个不停"
                            + (f"（跳转目标是 {loc}）" if loc else "")
                            + "——换成那个地址再抓一次")
            return out
        # 4xx / 5xx：**是结果**——把状态码和 body 带回去
        out["final_url"] = getattr(e, "url", "") or url
        try:
            raw = e.read(max_bytes + 1)
        except Exception:
            raw = b""
        return _build_result(out, e.code, getattr(e, "headers", None), raw,
                             max_bytes, max_chars)
    except Exception as e:
        out["error"] = f"没抓到（{type(e).__name__}: {e}）"
        return out
