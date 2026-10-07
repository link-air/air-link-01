"""出站网络适配的测试：**loopback 永远直连**（挂 VPN 时本地服务不能断）。

用户报的真事（2026-09-18）：开着 VPN（系统代理）时，本机的语音服务与
奥拉马全"连不上"——因为它们也被代理走了。浏览器没事（系统代理自带
"绕过本地地址"）；Python 这边这条豁免要自己保证（`core/net.py`）。
"""
# 用例分组：
#   LoopbackTest 本机地址判定 · ProxySkipTest 代理下的走向 ·
#   InstallTest 全局安装 · LoopbackEndToEndTest 真服务回归
import os
import ssl
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import net


class LoopbackTest(unittest.TestCase):
    def test_loopback_addresses(self):
        for good in ("127.0.0.1", "127.1.2.3", "::1", "[::1]", "localhost",
                     "LocalHost", "app.localhost", "0.0.0.0", "::ffff:127.0.0.1"):
            self.assertTrue(net.is_loopback(good), good)

    def test_public_hosts_are_not_loopback(self):
        for bad in ("example.com", "8.8.8.8", "192.168.1.1", "10.0.0.5",
                    "github.com", ""):
            self.assertFalse(net.is_loopback(bad), bad)


class ProxySkipTest(unittest.TestCase):
    """挂上代理（模拟 VPN 的系统代理）时：**外面走代理、本机直连**。"""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("http_proxy", "HTTP_PROXY")}
        os.environ["http_proxy"] = "http://127.0.0.1:9"      # 一个不会应答的代理
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_loopback_never_goes_through_proxy(self):
        """本地服务（奥拉马 / 语音 / 仪表盘）永远直连——这就是修的那件事。"""
        for url in ("http://127.0.0.1:11434/api/tags",
                    "http://localhost:8977/health",
                    "http://[::1]:8765/api/state"):
            self.assertFalse(net.proxied(url), url)

    def test_remote_goes_through_proxy_when_set(self):
        """有代理就该走代理——DeepSeek / 抓取这类"外面"的请求靠它出网。"""
        self.assertTrue(net.proxied("http://example.com/"))

    def test_handler_skips_proxy_for_loopback(self):
        """handler 层面钉死：loopback 的代理钩子必须"不处理"（落回直连）。

        非 loopback 那条路会走 `parent.open`（真发请求），这里不碰它——
        由上面的 `proxied()` 断言覆盖。
        """
        h = net._LoopbackDirectProxy({"http": "http://127.0.0.1:9"})
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        self.assertIsNone(h.proxy_open(req, "http://127.0.0.1:9", "http"))


class InstallTest(unittest.TestCase):
    """`install()` 之后，所有 `urlopen` 调用点（llm / embedding / 语音客户端）
    自动走带规则的 opener——这就是"一处安装、全部覆盖"的落点。"""

    def test_install_replaces_global_opener(self):
        net.install()
        self.assertIs(urllib.request._opener, net.opener())

    def test_reinstall_refreshes_proxy_snapshot(self):
        """重装 = 重建（重读代理环境）——`ProxyHandler` 是构造时快照，
        运行中开关 VPN 后靠重装跟随（loopback 豁免不受此影响，每次请求都判）。"""
        net.install()
        first = net.opener()
        net.install()
        self.assertIsNot(net.opener(), first, "重装要重建、不能复用旧快照")
        self.assertIs(urllib.request._opener, net.opener(),
                      "重建之后全局装的也是新的那份")


class LoopbackEndToEndTest(unittest.TestCase):
    """端到端回归：**假代理 + 真本机服务**。

    这就是用户场景的原样复现——代理环境变量指着一条死路，而本机服务
    （这里用一个真的 loopback HTTP 服务冒充）必须照常可达。
    """

    def test_local_service_reachable_even_with_proxy_env(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"LOCAL_OK")

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)      # 逆序执行：先 shutdown、再关 socket
        self.addCleanup(srv.shutdown)
        port = srv.server_address[1]

        saved = {k: os.environ.get(k) for k in ("http_proxy", "HTTP_PROXY")}
        os.environ["http_proxy"] = "http://127.0.0.1:9"      # 没人听的代理

        def _restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(_restore)
        net.install()
        url = f"http://127.0.0.1:{port}/health"
        with urllib.request.urlopen(url, timeout=5) as r:
            self.assertEqual(r.read(), b"LOCAL_OK",
                             "挂了代理也要能连上本机服务——loopback 直连豁免")
        # `net.open` 的 context 分支（embedding 的 insecure_ssl 那条路）：
        # `urlopen(context=…)` 会绕开全局 opener，这条显式出口必须同样直连
        with net.open(urllib.request.Request(url), timeout=5,
                      context=ssl.create_default_context()) as r:
            self.assertEqual(r.read(), b"LOCAL_OK",
                             "context 分支也走规则，不许绕开")
