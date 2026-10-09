"""语义向量服务。

**本文件是从求知版（`e:/air/core/embedding.py`）原样搬来的**——
「从求知版复用」清单里点名的三个复用件之一（`EmbeddingService` + `cosine` +
`embedding_novelty`）。**降级逻辑本身一字未改**（未配置 / 调用失败都返回 None）；
搬来后只做过三类本地化接线：统一出站出口 `net.open`（loopback 直连 / 证书）、
`insecure_ssl` 开关、服务状态记账（`_note_ok` / `_note_err`）。

为什么原样搬而不是重写：它的降级逻辑是**踩出来的**（未配置 / 调用失败都返回
None，调用方回退字符重叠，air 不会崩），符合本项目的「降级不崩」验收项；
重写一遍只会把那几个坑再踩一次。

接 OpenAI 兼容的 /embeddings 端点（Ollama 本地 /v1 端点也可），或本地
sentence-transformers 模型（endpoint 以 local:// 开头，如 local://BAAI/bge-small-zh-v1.5）。

设计原则：
  - 未配置、或调用失败时优雅降级 —— 返回 None，调用方回退字符重叠检索，不崩。
  - 存储时就在条目上算好向量（有服务则算，无则留缺省）；检索时只对新查询算一次向量。
  - novelty：embedding_novelty(query_vec, ref_vecs) = 1 − max 余弦。

远程模式用 urllib.request（纯标准库，零第三方依赖）；
local:// 模式才需要 sentence-transformers（可选，懒加载）。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L2 外部服务（向量）
#   上游    ：config（在 `chat.build_embedding` 里读）
#   下游    ：scene（算检索向量）、recall（算查询向量与相似度）、chat（建服务）、
#             memo / settings（测试连接）、distill / trend / weave / tools / salvage
#   对外入口：`EmbeddingService`（`embed` / `embed_one` / `degraded`）、
#             `cosine` / `embedding_novelty`
#   边界    ：**失败一律返回 None，不抛**——降级与否由调用方决定怎么兜
# ---------------------------------------------------------------------
# 本文件分段
#   段 1  EmbeddingService —— 语义向量服务封装（本文件主体）
#   段 2  cosine / embedding_novelty —— 两个纯函数
# ---------------------------------------------------------------------
import json
import ssl
import time
import urllib.request

from . import net

# 证书校验**默认开启**——api_key 是随这条连接发出去的，
# 「服务是只读的所以低危」那个说法不成立：被截的就是凭据本身。
# 需要放宽的只剩一种场景：自签 / 内网端点（证书链不完整）。
# 那必须由使用方**显式**打开：`embedding.insecure_ssl`（走 config / 设置页）。
# 这里做成惰性单例：不放宽的时候，这个上下文根本不会被创建。
_UNVERIFIED_CTX: ssl.SSLContext | None = None


def _unverified_ctx() -> ssl.SSLContext:
    """惰性建一个"不校验"的 SSL 上下文——**默认路径上根本创建不出它**。"""
    global _UNVERIFIED_CTX
    if _UNVERIFIED_CTX is None:
        _UNVERIFIED_CTX = ssl._create_unverified_context()
    return _UNVERIFIED_CTX


# ---- 段 1：EmbeddingService（向量服务封装）----

class EmbeddingService:
    def __init__(self, endpoint: str = "", api_key: str = "",
                 model: str = "text-embedding-3-small", timeout: float = 20.0,
                 insecure_ssl: bool = False):
        """**不做网络连接**——只判定"配齐了没有"。真正的失败要到第一次
        `embed` 才知道，所以 `available` 是"能不能试"，不是"一定能成"。
        """
        self.endpoint = (endpoint or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        # 证书校验默认开启；自签 / 内网端点由使用方显式放开（见文件头）。
        self.insecure_ssl = bool(insecure_ssl)
        # 本地模式：endpoint 以 local:// 开头，后缀 sentence-transformers 模型名
        self.is_local = self.endpoint.startswith("local://")
        self.local_model = self.endpoint[len("local://"):] if self.is_local else ""
        if self.is_local:
            self.available = bool(self.local_model)
        else:
            self.available = bool(self.endpoint and self.api_key)
        self._local_model = None     # 懒加载的 SentenceTransformer
        self._local_loaded = False   # 避免反复重试加载
        self.degraded = False        # 最近一次调用是否失败（当前处于字符重叠降级）
        # 最近一次调用的结果（2026-09-25）：给仪表盘顶栏的「语义」状态用。
        # 原来那里只有"配了没有"一个勾——今天这个服务死了两天，勾一直亮着，
        # 直到翻 trace 才发现。`last_error` 存异常原文（截断）：embedding 的
        # 失败多是"连不上 / 超时"，原文里就写着主机和原因，比再造一套错误码便宜。
        self.last_ok_at = ""
        self.last_err_at = ""
        self.last_error = ""

    def _note_ok(self) -> None:
        self.degraded = False
        self.last_ok_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def _note_err(self, why: str) -> None:
        self.degraded = True
        self.last_err_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.last_error = (why or "")[:160]

    def _get_local_model(self):
        """懒加载本地 sentence-transformers 模型；失败则降级返回 None。"""
        if self._local_loaded:
            return self._local_model
        self._local_loaded = True
        try:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self.local_model)
            print(f"[embedding] 本地模型已加载: {self.local_model}")
        except Exception as e:
            print(f"[embedding] 本地模型加载失败（降级字符重叠）: {e}")
            self._local_model = None
        return self._local_model

    def embed(self, texts: list) -> list | None:
        """批量取向量；不可用或失败返回 None（调用方降级）。

        成功置 degraded=False，失败置 degraded=True——让上层知道自己正处在
        字符重叠降级（相关度 / 新奇值失真），别拿降级结果当真归因。
        """
        if not self.available or not texts:
            return None
        try:
            if self.is_local:
                model = self._get_local_model()
                if model is None:
                    self._note_err("本地模型加载失败")
                    return None
                # normalize_embeddings=True → 余弦≈点积，与 cosine() 兼容
                vecs = model.encode(texts, normalize_embeddings=True)
                self._note_ok()
                return [v.tolist() for v in vecs]
            req = urllib.request.Request(
                f"{self.endpoint}/embeddings",
                data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            ctx = _unverified_ctx() if self.insecure_ssl else None
            # 走 `net.open` 而不是 `urlopen`：`urlopen` 在给了 context 时会
            # **自己 build 一个 opener、绕开全局那份**——loopback 豁免在那条
            # 路上会悄悄失效，而自签 / 内网端点恰恰最容易被 VPN 的代理坑
            # （见 `core/net.py` 模块头）。
            with net.open(req, timeout=self.timeout, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = [item["embedding"] for item in data.get("data", [])]
            if len(out) == len(texts):
                self._note_ok()
                return out
            self._note_err(f"返回 {len(out)} 条向量，期望 {len(texts)} 条")
            return None
        except Exception as e:
            self._note_err(str(e))
            print(f"[embedding] 调用失败（降级字符重叠）: {e}")
            return None

    def embed_one(self, text: str) -> list | None:
        """`embed` 的单条包装。**拿不到就是 None**——调用方据此走字符重叠兜底。"""
        r = self.embed([text])
        return r[0] if r else None


# ---- 段 2：纯函数（余弦相似度 / 内容新奇值）----

def cosine(a: list, b: list) -> float:
    """余弦相似度；维度不一致或任一为空返回 0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embedding_novelty(query_vec: list, ref_vecs: list) -> float:
    """内容新奇值 v_new = 1 − 与已有记忆的最大余弦相似度（离得越远越新奇）。

    ref_vecs 空 → 全新奇（没有任何参照）。返回值 0~1。

    ⚠️ 当前没有任何调用方（2026-10-06 核过）——它是文件头「从求知版复用」
    清单点名的三个复用件之一，先留着（同 `recall.theta` / `r0_floor` 的待遇）。
    """
    if not query_vec or not ref_vecs:
        return 1.0
    best = 0.0
    for rv in ref_vecs:
        c = cosine(query_vec, rv)
        if c > best:
            best = c
    return 1.0 - best
