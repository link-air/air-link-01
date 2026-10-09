"""本地仪表盘：把记忆系统的内部状态摆出来看。

为什么必须有这个东西：**「为什么召回了这个」不可见，就没法调试**，
也看不出 air 到底是真的记得、还是碰巧说对（设计稿 §5.5：可观测是逻辑实验的命脉）。
终端里画不下这些——线索、动作、召回、抑制、依据链，得有个版面摊开。

零依赖：`http.server`（标准库）+ 一个单页 HTML（`web/index.html`）。
绑 `127.0.0.1`——这是看记忆的东西，不该有别人能访问的口子。

两条约定：
  - **只读接口不碰写**：查库、看 trace 都不会改记忆
  - **内容的写只有一个入口**：`POST /api/chat/stream`，走 `ChatSession` 那条完整链路
    （唤醒 → 生成 → 写入）。绕过对话层直接塞场景 / 画像 / 备忘的口子一个都不开——
    开了就迟早有人绕过链路塞数据进来。

**另有一类写是允许的：人对系统的纠正**（`POST /api/topic-merge`）。
它不属于「内容写入」，而是人直接表达意志——和 `user_reject_profile` 同一性质。
两者别混：**内容写入走链路（那是 air 在记），纠正走人的按钮（那是人在改）**。
纠正同样要留痕（`weave._write_merge_trace`），因为"人什么时候纠正过什么"
本身就是最有价值的那批数据。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L11 界面层
#   上游    ：几乎全部（它是唯一把所有层拼起来的地方）
#   下游    ：无（`python -m core.dashboard` 的入口就在这里）
#   对外入口：`App`（HTTP 服务）/ `serve()`（`run.cmd` 与「全拉起」用它）
#             `Handler.do_GET` / `Handler.do_POST` 是全部 API 的两张路由表
#   边界    ：**不放业务逻辑**——处理函数只做"收参数 → 调 → 打包响应"，
#             要判断就在它该住的那个层里加方法
# ---------------------------------------------------------------------
from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import config as cfgmod
from . import persona as persona_file
from . import salvage
from . import settings
from .chat import ChatSession, build_embedding, persona_names
from .prompts import fixed_blocks_report
from .distill import maintenance_cycle, run_distill_cycle
from .llm import LLM
from .memo import close as memo_close_flow, due
from .model import FACT_KEYS, MEMO_CLOSED, lang_from_prefs
from .shortterm import ShortTerm
from .store import Store, atomic_write_json, now_str
from .weave import (archive_by_layer, delete_by_layer,
                    merge_entities_confirmed, merge_suggestions, merge_topics_confirmed,
                    render_mirror, save_facts_confirmed, save_lang_confirmed,
                    unarchive_by_layer, update_by_layer,
                    user_reject_profile)


# 备忘留痕（`备忘-*.jsonl`）2026-09-22 下移到 `memo.write_memo_trace`：
# 她判（今：命中判定）/ 她调工具 / 用户随手划三个发起方共用一份，都经过 `memo.close`；
# 退役（`memo.retire_due`）自己写一份——四个发起方都有痕（2026-10-05 补全）。


def _tts_base() -> str:
    """语音服务的根地址（末尾不带斜杠）。**读它只在这一个地方**。

    五个调用点都要它：两处播报代理（整段 / 流式）、查状态、起、关。
    抄五份的下场不是"多几行"，是改端点时漏改一处——那一处会去联一个不存在的地址，
    而且只在那条路上出错，看起来像"偶尔坏的"。配置的规矩本来就是「数字只写一个地方」。
    """
    return (cfgmod.cfg("tts", "endpoint", default="") or "").strip().rstrip("/")


# 语音配置文件（`tts/voice.json`）里**允许从界面改**的字段——白名单。
# 别的键一律不动：
#   - `speaker` / `instruct_extra`：界面走 `user_prefs`（每次请求带），
#     这里再写一份两边会打架；
#   - `port`：改了 `tts.endpoint` 就对不上，换端口是手工的事；
#   - `_note` / `_model_choices` 这类注释键：原样保留（表单不该顺手删注释）。
_VOICE_CONFIG_KEYS = ("model", "language", "device", "dtype", "idle_unload_minutes",
                      # 克隆的参考（Base 模型用）：音频路径 + 它的逐字文本。
                      # 也由界面保存（换参考 = 换嗓子）——语音服务重启时读。
                      "ref_audio", "ref_text")


def _dims_from_pref(raw: str) -> dict:
    """`user_prefs` 里那条 `tts_dims`（JSON）→ 刻度字典。**坏了当空的**（{}\）。

    为什么说它是"坏了当空的"而不是报错：这是**声音**的设置，最坏的结果也就是
    "这几个刻度回到默认"，她还能说话。因为一份写坏的偏好让整个仪表盘报错，
    是拿可用性换一个没人会看的异常。
    """
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return dict(d) if isinstance(d, dict) else {}


def _clean_dims(dims) -> dict:
    """把关 `dims` 的**形状**（必须是 `{键: 整数}`），不把关它有哪些键。

    范围的另一端在 `tts/server.py` 的 `voice_dim_bits`：只有它能答"这个维度有几档"，
    所以**它也顺手把越界的下标忽略了**。两边各守一段，主项目不用长模型的知识。
    值不是数字就丢掉——宁可少一条指令，也不给模型喂一串它读不懂的东西。
    """
    if not isinstance(dims, dict):
        return {}
    out = {}
    for k, v in dims.items():
        try:
            out[str(k).strip()] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _service_status(available: bool, svc) -> dict:
    """一个外部服务的真实状态（2026-09-25）：配了没有 + 最近一次调用成没成。

    为什么要有它：顶栏原来的勾只看"配置填了没有"（向量那条更粗，是
    `bool(self.emb)`）——向量服务死了两天，勾一直亮着，直到有人翻 trace
    才发现。**判据是真实调用的记账**（`LLM._send` / `EmbeddingService.embed`
    里写的），不是"配置看起来对不对"。

    `ok` 三态：`None` = 还没调用过（配置好了但没试过）；`True` / `False` =
    最近一次成功 / 失败——`last_ok_at` 与 `last_err_at` 谁晚谁说了算
    （同一格式的本地时刻串，字符串序即时间序）。
    """
    ok_at = getattr(svc, "last_ok_at", "") or ""
    err_at = getattr(svc, "last_err_at", "") or ""
    state = None if not ok_at and not err_at else ok_at >= err_at
    return {
        "available": bool(available),
        "ok": state,
        "last_ok_at": ok_at,
        "last_err_at": err_at,
        "last_error": getattr(svc, "last_error", "") or "",
    }


class App:
    """跨请求共享的状态。HTTP 是多线程的，所以会话拿锁保护。

    这里的方法基本是「给前端的一份视图」：`Handler` 收完参数调过来，
    这里的每个方法把 store / 会话里的东西摊平成能直接 JSON 的形状。
    所以它们的命名就是 API 的名字（`scenes()` ↔ `GET /api/scenes`）——
    两边对得上，找起来才不用翻两张表。
    """

    def __init__(self, port: int = 8765):
        """起库连接 / 模型客户端 / 向量服务；会话**懒建**——
        `GET /api/state` 每几秒被轮询一次，只读视图一律不碰会话。"""
        settings.apply()
        self.port = port
        self.store = Store()
        # 每日一份快照（幂等：当天已有就跳过）。放在**启动点**是刻意的：
        # 一天里第一次开仪表盘时，库还是"昨天的尾巴"——那正是值得留一份的状态。
        # 失败不拦启动（`backup_daily` 自己吞异常；备份是兜底，不是前置条件）。
        # ⚠️ 这行以前不存在：方法写好了、注释写着"每日一份"，却没有任何调用点。
        self.store.backup_daily()
        self.llm = LLM()
        self.emb = build_embedding()
        self._lock = threading.Lock()
        self._session: ChatSession | None = None
        self._distilling = False      # 后台提炼是否在跑（同一时刻只允许一个）
        self._distill_thread: threading.Thread | None = None
        self.last_distill: dict = {}  # 最近一次提炼的结果（给界面看）
        # （原来这里记「最近一次互动」给主动开口的防撞车判据用——主动开口
        #   2026-10-05 晚整块删了，这个字段跟着走。）
        # 体检（记忆整理稿 §三，2026-09-23）：累计素材量。**不落库**——
        # 它只服务"距上次整理攒了多少"，重启归零无害（时间兜底会接住）；
        # "上次整理时刻"落 `meta`（重启后仍要判"距上次多久"，见 `_mark_maint_done`）。
        # 体检结果不进新字段——`last_distill` 就是它（带 `trigger` 的那份，界面同一处显示）。
        self._chars_since_maint = 0

    # ---- 会话 ----

    @property
    def session(self) -> ChatSession:
        """懒建会话：第一次说话时才建，这样启动仪表盘不会先花一次 LLM 调用。"""
        if self._session is None:
            self._session = ChatSession(self.store, self.llm, self.emb)
        return self._session

    def chat(self, msg: str) -> dict:
        """走完整链路回一句话（**加锁**）——非流式，流式那版是 `chat_stream`。

        HTTP 那个 `/api/chat` 口 2026-10-09 删了（仓内无人调），这个方法仍被
        测试与进程内调用用着：两条路共用对话层同一条链路。

        为什么要锁：会话带状态（短期窗口），两个请求同时进来会让窗口错乱——
        比如两段对话被交叉写进同一个窗口段。
        只读接口不加锁：`Store` 是每线程一个连接，查询之间本来就不互相干扰。
        """
        with self._lock:
            out = self.session.reply(msg)
            # 体检的账（记忆整理稿 §三）：他说的 + 她说的都算素材量
            self._chars_since_maint += len(msg or "") + len((out or {}).get("reply") or "")
            return out

    def speak(self, text: str, voice: dict | None = None) -> dict:
        """把一段话交给**本地语音服务**，拿回音频（打包成 data URI 给页面）。

        为什么走仪表盘代理、不让页面直连那个服务：
          - 页面不用知道端口和模型（配置留在 Python 一侧，同「数字只写一个地方」）；
          - 少一层 CORS，语音服务那边只管本机。
        服务没起是**正常状态**（语音是可选外挂，见 `tts/README.md`）：
        如实回报、并说清怎么把它起起来——播报失败绝不能影响对话本身。
        """
        base = _tts_base()
        if not base:
            return {"ok": False, "detail": "没配语音服务（config 的 tts.endpoint 是空的）"}
        text = (text or "").strip()
        if not text:
            return {"ok": False, "detail": "没有要念的文本"}
        req = urllib.request.Request(
            base + "/speak",
            data=self._speak_payload(text, voice),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            timeout = float(cfgmod.cfg("tts", "timeout", default=120) or 120)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                wav = r.read()
                mime = (r.headers.get("Content-Type") or "audio/wav").split(";")[0]
        except urllib.error.HTTPError as e:
            # 服务起来了但生成失败：把它说清的原因**原样带出来**
            # （显存不够 / 权重没下全 / 缺依赖——处理方式完全不同，不能被吞成一句"失败了"）
            try:
                detail = (json.loads(e.read().decode("utf-8")) or {}).get("detail") or ""
            except Exception:
                detail = ""
            return {"ok": False, "detail": detail or f"语音服务报错 {e.code}"}
        except Exception as e:
            return {"ok": False, "detail": (
                f"连不上语音服务（{base}）：{type(e).__name__}。"
                f"顶栏那个「语音」按钮点一下就能起（没装过就按 tts/_dl/setup_tts.ps1 装一次）；"
                f"不想用就把 config.local.json 里的 tts.endpoint 设成空串")}
        if not wav:
            return {"ok": False, "detail": "语音服务返回了空音频"}
        return {"ok": True, "ms": int((time.time() - t0) * 1000),
                "audio": f"data:{mime};base64," + base64.b64encode(wav).decode("ascii")}

    def speak_stream(self, text: str, voice: dict | None = None):
        """流式版播报：**把语音服务的 PCM 帧原样转给页面**。

        返回值两种，调用方要分清：
          - `{"ok": False, ...}` —— 连不上 / 没有要念的文本 / 服务在开始前就报错
            （版本对不上时多一个 `"fallback": True`，页面据此退回"整段"那条路）；
          - `(采样率, 帧迭代器)` —— 成功了，每帧是 16bit 单声道 PCM 字节。

        "先摸到第一帧"跟 `tts/server.py` 那边同一个道理：**响应头一旦发出去就改不了口**，
        所以开始前能炸的错必须还能当 JSON 报出去，否则页面只会看到"没有声音"，
        分不清是没接上还是模型炸了。
        """
        base = _tts_base()
        if not base:
            return {"ok": False, "detail": "没配语音服务（config 的 tts.endpoint 是空的）"}
        text = (text or "").strip()
        if not text:
            return {"ok": False, "detail": "没有要念的文本"}
        req = urllib.request.Request(
            base + "/speak",
            data=self._speak_payload(text, voice, stream=True),
            headers={"Content-Type": "application/json"})
        timeout = float(cfgmod.cfg("tts", "timeout", default=120) or 120)
        try:
            # 不 with：**这条流要留给调用方读**，这里只负责把它拿到手
            r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            try:
                detail = (json.loads(e.read().decode("utf-8")) or {}).get("detail") or ""
            except Exception:
                detail = ""
            return {"ok": False, "detail": detail or f"语音服务报错 {e.code}"}
        except Exception as e:
            return {"ok": False, "detail": (
                f"连不上语音服务（{base}）：{type(e).__name__}。"
                f"顶栏那个「语音」按钮点一下就能起（没装过就按 tts/_dl/setup_tts.ps1 装一次）；"
                f"不想用就把 config.local.json 里的 tts.endpoint 设成空串")}
        head = r.read(12)
        if len(head) < 12 or head[:4] != b"ARAU":
            r.close()
            # 旧版语音服务会忽略 stream、直接回整段 wav：页面据此退回去，不影响出声
            return {"ok": False, "fallback": True,
                    "detail": "语音服务没按流式回（版本对不上？）"}
        sr, _ch = struct.unpack("<IH", head[4:10])

        def frames():
            try:
                while True:
                    b = r.read(4)
                    if len(b) < 4:
                        return
                    need = struct.unpack("<i", b)[0] * 2
                    buf = b""
                    while len(buf) < need:
                        part = r.read(need - len(buf))
                        if not part:
                            return
                        buf += part
                    yield buf
            finally:
                r.close()      # 页面中途停 / 连接断：把这条流关干净

        return int(sr), frames()

    def voice(self) -> dict:
        """她的声音：**基底音色 + 微调刻度 + 那句手写描述**。**他定的**，存 `user_prefs`。

        跟对话偏好分开存（`tts_` 前缀），**也不留痕**——档案留痕的理由是
        「改的是写入的原料」，而嗓子不影响她记住什么：同一句话，只是谁念的。
        空 = 跟随 `tts/voice.json`（他也可以直接在那边定，两边都不写死）。

        `dims` 只回**他调过的**那些刻度——没调过的不落库，否则今天加了第 6 个维度，
        库里那份旧字典会随着每次读扩散出去。补默认值是读的那一边的事。
        """
        p = self.store.all_prefs()
        return {"speaker": (p.get("tts_speaker") or "").strip(),
                "dims": _dims_from_pref(p.get("tts_dims") or ""),
                "instruct_extra": (p.get("tts_instruct_extra") or None)}

    def voice_save(self, speaker: str, extra, dims=None) -> dict:
        """保存声音设置。空值一律存成"跟随 voice.json"，不往库里写一份死值。

        `dims` 逐个键过一遍，只留「能当整数用」的那些——主项目**不认识**有哪些维度
        （那是语音服务的知识，同「主项目只当文件代理」那条：维度改名不该这边跟着动）。
        但格式得把关：库里躺一串乱七八糟的东西，下次读出来还得有人去擦。
        `dims` 是 `None`（这次没提它）就**不动**——别因为一次只改描述就把刻度清空。
        """
        self.store.set_pref("tts_speaker", (speaker or "").strip())
        self.store.set_pref("tts_instruct_extra", "" if extra is None else str(extra).strip())
        if dims is not None:
            self.store.set_pref("tts_dims", json.dumps(_clean_dims(dims)))
        return self.voice()

    def voice_status(self) -> dict:
        """语音服务现在什么样：在不在、加载哪个模型、用哪个音色、有哪些可选。

        **读它就是读服务**：没起的时候如实说"没接上"，但设置页照旧能改设置——
        设置存的是"他要什么"，服务起来才生效（不依赖它活着，也不需要重启）。
        """
        base = _tts_base()
        if not base:
            return {"ok": False, "detail": "没配语音服务（config 的 tts.endpoint 是空的）"}
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                d = json.loads(r.read().decode("utf-8")) or {}
            d["ok"] = True
            return d
        except Exception as e:
            return {"ok": False, "detail": f"连不上（{base}）：{type(e).__name__}"}

    def voice_start(self) -> dict:
        """把语音服务拉起来——**人点，系统不自动起**（同"只有人能切模式"那条：
        自动拉起等于系统替他开一个吃几个 G 显存的东西）。

        只起 HTTP 服务，**不碰模型**：模型仍然第一次播报时才加载（那条懒加载的规矩没变），
        所以这个按钮点下去几秒就回来，不会卡在"加载中"。
        拉起来的进程**脱离仪表盘**（仪表盘关了它还得活着——重启一次要重读几个 G 权重）。
        **起的是 venv 里的 `pythonw`（没有控制台黑框）**；日志照样写 `tts/_dl/server.log`。
        """
        st = self.voice_status()
        if st.get("ok"):
            return {"ok": True, "already": True, "status": st}
        base = _tts_base()
        host = (urlparse(base).hostname or "") if base else ""
        if host not in ("127.0.0.1", "localhost", "::1", ""):
            return {"ok": False, "detail": (
                f"endpoint 指在 {host}（不是本机）——这个按钮只能起本机那个服务")}
        tts_dir = cfgmod.abspath(cfgmod.PATHS["tts"])
        scripts = tts_dir / ".venv" / ("Scripts" if os.name == "nt" else "bin")
        # **优先 pythonw**：GUI 子系统程序，起服务时**不会弹控制台黑框**
        # （python.exe 是控制台程序，哪怕后台起也会闪一个窗——没用还碍眼）。
        # 找不到 pythonw 才退回 python.exe，那时用 CREATE_NO_WINDOW 把窗口藏起来
        # （它和 DETACHED_PROCESS 互斥，按解释器二选一）。
        py = scripts / ("pythonw.exe" if os.name == "nt" else "pythonw")
        if not py.exists():
            py = scripts / ("python.exe" if os.name == "nt" else "python")
        if not py.exists():
            return {"ok": False, "detail": (
                f"没找到 {py}——按 tts/_dl/setup_tts.ps1 装一次（要 torch，几个 G）")}
        log = tts_dir / "_dl" / "server.log"
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "ab") as f:
                flags = 0
                if os.name == "nt":
                    if py.name.lower().startswith("pythonw"):
                        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                    else:
                        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                subprocess.Popen([str(py), "server.py"], cwd=str(tts_dir),
                                 stdout=f, stderr=subprocess.STDOUT,
                                 creationflags=flags, close_fds=True,
                                 start_new_session=(os.name != "nt"))
        except Exception as e:
            return {"ok": False, "detail": f"起不来：{type(e).__name__}: {e}"}
        for _ in range(20):                     # 等它开口（起服务很快，慢的是加载模型）
            time.sleep(0.5)
            st = self.voice_status()
            if st.get("ok"):
                return {"ok": True, "already": False, "status": st, "log": str(log)}
        return {"ok": False, "detail": f"起来了但十秒没回应，看日志：{log}"}

    def voice_stop(self) -> dict:
        """关掉语音服务（**放掉显存**）——让它自己退（`/shutdown`），不是外面硬杀。

        为什么让它自己退：显存干净放掉、正在生成的那句收尾；更重要的是
        **不管它是谁起的都能关**（人工开的窗口、上一会话留下的）——外面没有 PID 可猜，
        猜错了 taskkill 就是杀错进程。

        本就没在跑也算成功（`already`）：这是一个"让它不在"的动作，不是"执行一次关机"。
        """
        if not self.voice_status().get("ok"):
            return {"ok": True, "already": True}
        base = _tts_base()
        req = urllib.request.Request(base + "/shutdown", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                r.read()
        except urllib.error.HTTPError as e:
            # 404 = 那个服务是**旧版**（还不认识 /shutdown）——怎么收拾写清楚，别让人猜
            hint = ("（它是旧版、还不认识 /shutdown：先在任务管理器里结束那个 python，"
                    "再从顶栏「语音」起一个换新版）") if e.code == 404 else ""
            return {"ok": False, "detail": f"关不掉：HTTP {e.code}{hint}"}
        except Exception as e:
            return {"ok": False, "detail": f"关不掉：{type(e).__name__}: {e}"}
        # 等它真的松口（端口不再应答）。**用截止时间，不用固定轮数**：探测自己带 5 秒超时，
        # 进程要是卡死，轮数一乘能把人晾一分多钟——"关不掉"这件事也得快点说出口。
        deadline = time.time() + 8
        while time.time() < deadline:
            time.sleep(0.25)
            if not self.voice_status().get("ok"):
                return {"ok": True, "already": False}
        return {"ok": False, "detail": "它回了话但没退出去（任务管理器里那个 python）"}

    # ---- 语音配置（模型 / 精度 / 语言 / 空闲卸载）----

    def voice_config(self) -> dict:
        """读语音服务的配置（`tts/voice.json`）+ 本地可选的模型清单。

        主项目在这里只当**文件代理**：不解释 model / dtype 这些字段的语义，
        原样读、白名单写（语义的主人还是语音服务自己，同 `tts.endpoint` 的定位）——
        这样「界面里能换模型」不必让主项目长出模型知识。
        """
        tts_dir = cfgmod.abspath(cfgmod.PATHS["tts"])
        path = tts_dir / "voice.json"
        try:
            conf = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "detail": f"读不到 tts/voice.json：{e}"}
        if not isinstance(conf, dict):
            return {"ok": False, "detail": "tts/voice.json 不是一个对象"}
        # 本地模型：`tts/models/` 下的子目录（相对名，服务端自己解析）
        models = []
        try:
            models = [f"models/{p.name}" for p in sorted((tts_dir / "models").iterdir())
                      if p.is_dir()]
        except OSError:
            pass
        cur = str(conf.get("model") or "")
        if cur and cur not in models:
            models.insert(0, cur)        # 当前是 HF 仓库名 / 手写路径：也让它出现在候选里
        # 参考音频候选：`tts/refs/` 下的文件（克隆模式的声音来源；换文件 = 换嗓子）
        refs = []
        try:
            refs = [f"refs/{p.name}" for p in sorted((tts_dir / "refs").iterdir())
                    if p.is_file()]
        except OSError:
            pass
        return {"ok": True, "path": str(path), "current_model": cur, "models": models,
                "refs": refs,
                "config": {k: conf.get(k, "") for k in _VOICE_CONFIG_KEYS}}

    def voice_config_save(self, patch: dict) -> dict:
        """写回 `tts/voice.json` 的**白名单字段**（合并 + 原子替换）。

        只认 `_VOICE_CONFIG_KEYS`：注释键（`_note` 那类）原样保留——
        「表单没覆盖到的键不该因为一次保存被丢掉」同 `settings.save_local` 的道理。
        空值 = 这次不改（模型名不能写成空串）。**改完要重启语音服务才生效**
        （它启动时读这份配置）。
        """
        tts_dir = cfgmod.abspath(cfgmod.PATHS["tts"])
        path = tts_dir / "voice.json"
        try:
            conf = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "detail": f"读不到 tts/voice.json：{e}"}
        if not isinstance(conf, dict):
            return {"ok": False, "detail": "tts/voice.json 不是一个对象"}
        changed: dict = {}
        for k in _VOICE_CONFIG_KEYS:
            if k not in (patch or {}):
                continue
            v = patch[k]
            if k == "idle_unload_minutes":
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
                v = max(0, min(24 * 60, v))
            else:
                v = str(v if v is not None else "").strip()
                if not v:
                    continue             # 空值 = 这次不改（别把模型名写成空串）
            if conf.get(k) != v:
                conf[k] = v
                changed[k] = v
        if changed:
            atomic_write_json(path, conf)     # 原子写：这是"他定的东西"，别写坏
        return {"ok": True, "changed": changed,
                "config": {k: conf.get(k, "") for k in _VOICE_CONFIG_KEYS},
                "restart_hint": "改完要重启语音服务才生效（「语音」页有按钮）。"}

    def _speak_payload(self, text: str, voice: dict | None = None,
                       **extra) -> bytes:
        """发给语音服务的请求体：**带上他在设置里定的嗓子与那句额外语气**。

        没定就不带这个字段——服务那边回落 `voice.json`（两条路给同一份默认，
        页面和文件不会各说各话）。`voice` 传了**逐项**盖过存的那份：只给一项就只盖
        那一项（**试听**走这条：他还没保存，想先听听这个嗓子什么动静）——
        整份替换会让"只传语气"把音色也一起重置掉。
        """
        v = dict(self.voice())
        for k in ("speaker", "instruct_extra", "dims"):
            if voice and k in voice:
                v[k] = voice[k]
        body: dict = {"text": text}
        if v.get("speaker"):
            body["speaker"] = v["speaker"]
        if v.get("instruct_extra") is not None:
            body["instruct_extra"] = v["instruct_extra"]
        # 刻度过 `_clean_dims` 再出门：只发整数，别让一份手写的偏好原样过网。
        # 空字典照样发——服务那边认得"一个都没给"和"给了但全是默认"的分别。
        if "dims" in v:
            body["dims"] = _clean_dims(v.get("dims"))
        body.update(extra)
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def confirm(self, action: str = "") -> dict:
        """他点了确认条上的动作（`delete` / `archive` / `keep` / `accept`）——
        改与删的主通道（三档见 `ChatSession.confirm`）。"""
        with self._lock:
            return self.session.confirm(action)

    def undo_turns(self, turns: int = 1, expect: str = "") -> dict:
        """撤掉窗口末尾的 N 轮（改与重新生成的前半步，见 `ShortTerm.undo_turns`）。

        两条路，必须走对：**有会话就撤内存里那一份**——否则下一次 append 落盘
        会把这次撤销整段覆盖回来（窗口是"整文件原子写"，不是增量改）。
        没有会话（刚重启、还没开口）就从窗口文件构造一个临时的：
        构造只做 `_load`，撤完 `_save`，不发 LLM。

        **不留痕**：撤的是还没进长期库的一段，没有记忆被改动——
        没有"谁改了什么"要回答（旧那套「改一条消息」动的是记录，才需要留痕）。
        """
        with self._lock:
            st = (self._session.st if self._session is not None
                  else ShortTerm(self.store, self.llm, emb_service=self.emb))
            return st.undo_turns(turns, expect)

    def chat_stream(self, msg: str):
        """流式：**和非流式用同一把锁**——窗口是串行的，对话本来就是一句一句来。

        锁会持到这一轮结束，所以她还在想的时候你发的下一句是**排在后面**，
        不是插进来把窗口搅乱。界面不会卡（服务器是多线程的），只是那条要等一等。
        """
        with self._lock:
            gen = self.session.reply_stream(msg)
            # **显式收尾**（`finally` 里 close，不交给 GC）：客户端断开的路上
            # "把说到一半的那半句落进窗口"发生在 `close()` 里（见
            # `chat._commit_partial_turn`），而它**必须排在放锁之前**——
            # 让 GC 去关的话，锁可能先放了，紧接着来的 `/api/turn-undo`
            # 撤到的就不是末尾那一轮（末尾马上要被那半截补上），窗口顺序会错。
            try:
                for ev in gen:
                    if (ev or {}).get("type") == "end":
                        # 体检的账（同 `chat`）：从 end 事件拿她的整段回复
                        #（流式是逐句 yield 的，只有 end 上带着全文）
                        self._chars_since_maint += (len(msg or "")
                                                    + len((ev or {}).get("reply") or ""))
                    yield ev
            finally:
                gen.close()

    def end(self) -> dict | None:
        """收尾：提取最后一段 → 后台整理。与 `new_session` 只差「摘要清不清」。

        界面上的按钮现在是「新对话」；**退出路径自己也在用它**（`serve()` 末尾的
        收尾就调 `App.end()`）。HTTP 那个口 `/api/end` 2026-10-09 删了：它和
        `/api/new-session` 是同一个动作（只差清不清摘要），而仓内没人调。
        """
        return self._close_session(clear_digest=False)

    def new_session(self) -> dict | None:
        """「新对话」：结束这一段 → 窗口（**连压缩摘要**）整个清空 → 后台整理。

        与 `end()` 只差一处：摘要也清。收尾是"同一段继续聊"的边界，摘要还当
        「更早的对话」的背景；新对话是"重新开始"，窗口里不留上一段的东西。
        内容没丢——提取照做，摘要代表的那几段早在长期库里，需要时靠回忆拉回来。
        """
        return self._close_session(clear_digest=True)

    def _close_session(self, clear_digest: bool) -> dict | None:
        """会话收尾的公共路径（`end` / `new_session` 都走它）。

        为什么整理挂在这里：**会话边界就是整理边界**——一段聊完了才有得整理。
        （原来这一步根本没接线：`run_distill_cycle()` 只在实验脚本里被调用过，
        配置里曾经有个谁也没读的 `distill.check_interval`（**那个旋钮已经删了**）。
        后果是：在仪表盘里聊多久，场景卡一直写、**S2 和画像却永远不会形成**。）
        """
        with self._lock:
            out = self.session.close(clear_digest=clear_digest)
        self.start_distill()
        return out

    def start_distill(self, maintenance: bool = False, trigger: str = "") -> bool:
        """起一个后台线程跑一次整理；已在跑则忽略（返回 False）。

        **为什么异步**：一次整理要跑好几个 LLM 调用（每个 topic 一次聚合 + 一次抽象，
        再加漂移检测、备忘维护、复核），同步做会让「新对话」这个按钮转十几秒。
        收尾该立刻结束，整理在后台慢慢做。

        两种整理走同一个口（只是内容不同，锁与线程收尾完全一致）：
          - **边界整理**（默认）：`run_distill_cycle` 全量——新对话 / 收尾 / 退出时；
          - **体检**（`maintenance=True`，记忆整理稿 §四）：`maintenance_cycle`
            ——主整理封顶 + S3 复核 + 留痕，由 `maintenance_tick` 按量按时触发。
        `trigger`：体检为什么跑（进留痕）；边界整理不用它。

        （2026-09-23 改注：原来这里写着"提炼不是常驻的——它只在会话结束时
        发生一次……没有'空闲时该做的事'"。体检上线后后半句不再成立，
        区别在**对内勤快、对外矜持**：她自己在做的只有维护（不说话、不改内容）。
        ⚠️ 2026-10-05 晚：主动开口（`proactive_tick`）整块删了——
        **"对外"那一半没有了**：她现在任何时候都不先开口（见待优化稿 K 条）。）
        """
        if self._distilling:
            return False
        self._distilling = True
        # 体检的账**起跑就结**：失败也认（下个满足条件再来，不在同一批上反复试）
        self._mark_maint_done()

        def work() -> None:
            try:
                if maintenance:
                    # 体检结果进 `last_distill`（镜像页"最近一次整理"同一处显示，
                    # 带 `trigger` 字段的就是体检）——不另开一个只给界面用的字段
                    self.last_distill = maintenance_cycle(self.store, self.llm,
                                                          emb=self.emb, trigger=trigger)
                else:
                    self.last_distill = run_distill_cycle(self.store, self.llm, emb=self.emb)
            except Exception as e:
                # 提炼失败不该影响已经写好的记忆——它们是两件事。
                print(f"[dashboard] 后台提炼失败（不影响已有记忆）: {e}")
            finally:
                self._distilling = False
                # **关掉这个线程自己的连接**：`Store` 是每线程一个连接，
                # 线程收工时关它——不关的话，每次收尾都留一条挂着的连接
                # （Windows 上还会一直占着库文件）。
                self.store.close()

        t = threading.Thread(target=work, daemon=True)
        self._distill_thread = t
        t.start()
        return True

    def wait_distill(self) -> bool:
        """等后台提炼收工（**只给退出用**）。返回是否跑完了。

        为什么退出要等：提炼是「会话边界就是整理边界」的那另一半——退出也是边界，
        只收尾不整理，这次聊的东西就永远停在 S1（S2 / 画像再也不会从它长出来）。
        而线程是 daemon，进程一退它就被掐断，光起线程等于没起。

        **给上限**：退出卡住比少整理一次更糟，超了就放它走并如实说——
        不静默（下次收尾会再整理一次，S1 已经落库，不会丢）。
        """
        t = self._distill_thread
        if t is None or not t.is_alive():
            return True
        t.join(float(cfgmod.cfg("distill", "exit_wait_seconds", default=30)))
        return not t.is_alive()

    def _mark_maint_done(self) -> None:
        """把「体检的账」结掉：计数清零 + 时刻落 `meta`。

        为什么时刻要落库：`maintenance_tick` 判"距上次多久"要用它，
        而重启后内存里没有——不落库的话每次重启都会立刻体检一次
        （幂等无害，但白花钱：明明十分钟前刚整过）。

        ⚠️ 记的是"跑过整理"而不是"整理成功"：失败也认——同一批素材不该
        因为一次失败被反复重试（下个满足条件再来就是了，幂等不着急）。
        """
        self._chars_since_maint = 0
        try:
            self.store.set_meta("last_maint_at", now_str())
        except Exception as e:
            print(f"[dashboard] 整理时刻落库失败（不影响整理本身）: {e}")

    def maintenance_tick(self) -> dict:
        """体检的触发判定（记忆整理稿 §三）：量到、或时间到，就跑一次。

        挂在 `_maintenance_loop`（每跳判一次，默认 10 分钟）——**检查很便宜**
        （比字数、比时间），真触发才花钱，钱在 `distill.maintenance_cycle`
        里封顶（topic 数 + 复核条数）。空闲时 **0 次** LLM 调用。

        为什么不放进"每轮说完话"：对话中的整理时机归四条提取触发
        （`shortterm.flush_if_needed`）；这里管的是**没有会话边界时**的兜底
        （长会话不点新对话 / 服务常驻不收尾），10 分钟的粒度足够
        （人不会十分钟变一次自我认知，见整理稿 §二）。
        """
        if self._distilling:
            return {"skipped": "整理在跑"}
        need = int(cfgmod.cfg("maintenance", "trigger_chars", default=10000) or 10000)
        hours = float(cfgmod.cfg("maintenance", "max_idle_hours", default=12) or 12)
        due_chars = self._chars_since_maint >= need
        last = self.store.get_meta("last_maint_at")
        since = _hours_since(last)
        due_time = since >= hours
        if not (due_chars or due_time):
            return {"skipped": "没到"}
        # 文案三种：到量 / 距上次够久 / **从没记过**——最后一种不能拿
        # `_hours_since` 的哨兵值去写"距上次 1000000000.0 小时"。
        reason = (f"累计 {self._chars_since_maint} 字" if due_chars
                  else (f"距上次 {since:.1f} 小时" if last else "首次整理（没有记录）"))
        started = self.start_distill(maintenance=True, trigger=reason)
        return {"started": started, "trigger": reason}

    # （`proactive_tick` 2026-10-05 晚删除——主动开口整块不要了：
    # 她的话只发生在他说话之后，见待优化稿 K 条。）

    # ---- 只读视图 ----

    def window_text(self) -> str:
        """窗口的渲染预览（「她这轮看到了什么」）——**不依赖会话**。

        重启后会话还没建（懒建），但窗口文件里有内容——界面不该因此空着。
        有会话就用内存里的（最新）；没有就从文件临时构造一个只读 `ShortTerm`
        （构造只做 `_load` 读文件：不发 LLM、不写盘）。
        """
        if self._session is not None:
            return self._session.window_preview()
        return ShortTerm(self.store, self.llm).build_window()

    def history(self, limit: int = 80) -> dict:
        """重启/刷新后对话区回填：窗口 + 最近几段已提取的原文——**只读，不建会话**。

        三个来源拼出完整的"刚才聊到哪了"（互不重叠，按时间从旧到新）：
          - `raws`：最近几段**已提取**的对话原文（`salvage.recent_dialogue`；
            更早的走「打捞」页）；
          - `digest`：窗口里已压缩的摘要批（前端折叠显示）；
          - `messages`：窗口里还没提取的逐字消息。
        提取时同一步"写 raws + 清窗口"，所以三层天然衔接——
        **窗口被收尾清空 ≠ 界面上记录没了**（它只是从逐字变成了原文那一段）。
        会话是懒建的，这里不碰 `self.session`——读文件即可
        （每次 append 后窗口都会原子落盘，永远是最新的）。
        """
        out = ShortTerm.read_state(cfgmod.abspath(cfgmod.PATHS["shortterm"]))
        return {"messages": out["messages"][-limit:],
                "digest": out["digest"],
                "raws": salvage.recent_dialogue(self.store)}

    def state(self) -> dict:
        """首屏那包：计数 + 开关 + 语言 + 窗口内容（不含列表）。

        ⚠️ **不能建会话**——它被高频轮询，在这里碰会话等于每次刷新都花钱。
        """
        s = self.store
        return {
            "input": {
                # 拖入文件的上限（工具箱第十二节）。**数字在 config 里**，
                # 前端照着截断，不在页面里抄一份——抄了就会和后端漂移。
                "file_max_chars": int(
                    cfgmod.cfg("input", "file_max_chars", default=20000)),
                # 「偏长」的线，和后端放宽超时的那条线是同一个数字
                # （`chat.long_input_chars`）——两边各写一个就会各说各话。
                "long_chars": int(cfgmod.cfg("chat", "long_input_chars", default=6000)),
            },
            "counts": {
                "scenes": s.count("scenes"),
                # 原文按天存文档——这里报的是**文档份数（天数）**，不是条数
                "raw_docs": s.raw_doc_days(),
                "summaries": s.count("summaries"),
                "profiles": s.count("profiles"),
                "memos": s.count("memos"),
                "entities": s.count("entities"),
                "archived": s.count("scenes", "archived = 1"),
            },
            "config": settings.describe(),
            "db": str(s.path),
            # 两个外部服务的真实状态：配了没有 + 最近一次调用成没成。
            # 原来是 `bool(self.emb)`（对象在不在）——顶栏的勾几乎恒亮，
            # 换 `_service_status` 的理由见它。
            "llm": _service_status(self.llm.available(), self.llm),
            "vector": _service_status(bool(self.emb and self.emb.available),
                                      self.emb),
            # 「air 这轮看到了什么」——**不依赖会话**（重启后会话还没建，
            # 但窗口文件里有内容；见 `window_text`。前端每几秒轮询一次状态，
            # 顺手的开销要小，也不能把会话建起来）
            "window": self.window_text(),
            "distilling": self._distilling,
            # `last_distill` 不在这里：它挪去了 `/api/mirror`（和「重新整理」
            # 同一个页面）——挂在状态里没人读，等于一个字段在说谎
            # 界面与她的输出语言（中 / 英）：前端拿它渲染对应文案，
            # 页面里不抄一份默认值——抄了就会和后端漂移。
            "lang": self.lang(),
        }

    def mirror(self) -> dict:
        """镜像视图（复用编织层那条路，不另写一份拼装逻辑）。

        顺带带上「最近一次整理的结果」——它和这一页是同一件事的因果：
        点了「重新整理」之后，S2/S3 新增了什么、谁收敛了、谁被老化，
        显示在这里才有人看（以前挂在 `/api/state` 里，前端从没读过它）。
        """
        out = render_mirror(self.store)
        out["last_distill"] = self.last_distill
        out["reviews"] = self.maint_reviews()
        return out

    def maint_reviews(self) -> list[dict]:
        """未处理的复核提议（`wrong`）+ 那条画像的陈述（界面直接显示用）。

        提议**不自己执行**：作废是人的裁决（`user_reject_profile`），
        这里只把"她为什么觉得不对"摆出来——理由要带上，不然没法判
        （记忆整理稿 §五：`wrong` 只出提议，人点头才作废）。

        画像已经作废 / 不在了的提议**顺手自动过期**：人从任何入口处理过了，
        镜像页就不该再挂一条点了也没用的东西。
        """
        out = []
        for r in self.store.unhandled_reviews():
            p = self.store.get_profile(r.profile_id)
            if p is None or p.invalidated_at:
                self.store.mark_review_handled(r.id)
                continue
            out.append({"id": r.id, "profile_id": r.profile_id,
                        "statement": p.statement, "topic": p.topic,
                        "reason": r.reason, "created_at": r.created_at})
        return out

    def persona(self) -> str:
        """当前人格名（air / mia / xina……）。**不建会话**——读个偏好不该花一次 LLM。"""
        return self.store.get_pref("persona") or "air"

    def set_persona(self, name: str) -> str:
        """切人格：**在名单里才落库**；对话每轮现读——下一句就换人。

        不改会话内存（人格是每轮注入的东西，不是会话状态），所以不需要 `_lock`。
        """
        name = (name or "").strip()
        if name and name in persona_names():
            self.store.set_pref("persona", name)
        return self.persona()

    def persona_list(self) -> dict:
        """设置页「人格」块要的一屏：每个人的显示名 / 字数 / 在不在用 / 常驻块体量。

        全部**现算**（`persona.list_info()` + 每个试算一次固定提示词）——不落第二份索引：
        人格的真源是那几个 md 文件，索引只是影子，而影子会过期（他手改文件、git 切分支，
        都不经过这儿）。`fixed` = "假设换成它，常驻块一共多大"（口径与预算测试同一处，
        `prompts.fixed_blocks_report`）；**停用的不试算**——`load` 读不到 `.off`，
        硬算出来是 air 的数，比空着更容易看错。
        """
        names = persona_names()
        items = persona_file.list_info()
        limit = 0
        for it in items:
            if not it["enabled"]:
                continue
            rep = fixed_blocks_report(persona_file.load(it["id"]), it["id"], names)
            it["fixed"] = rep["chars"]
            limit = rep["limit"]
        return {"persona": self.persona(), "names": names, "items": items,
                "limit": limit, "template": persona_file.TEMPLATE}

    def lang(self) -> str:
        """界面与她的输出语言（中 / 英）。同上——读一个偏好不该花一次 LLM。"""
        return lang_from_prefs(self.store.all_prefs())

    def reject(self, pid: str) -> dict:
        """用户否决一条画像（**呈现的另一半**）：**真删**（2026-09-24）。

        接线在这里的理由同「整理」：`user_reject_profile` 一直只有测试在调，
        界面上只写着「哪条不对你直接说」却没有可以说的入口——
        **一条不能否的画像，等于把 air 的判断强加给人**（编织层 §可否决）。
        删的是"一条不成立的画像"：素材都在，删了能重立；防的只有误操作
        （界面点「不对」时弹一次确认窗）。
        """
        outcome = user_reject_profile(self.store, pid)
        # 行、边、体检提议一起删——都在 `store.delete_profile` 里（同一条路径）
        return {"ok": outcome == "deleted", "outcome": outcome, "id": pid}

    def memory_action(self, ids: list[str], action: str, text: str = "",
                      field: str = "text") -> dict:
        """记忆动作的统一口：**改与删，三层同一套**（2026-09-24，工具箱稿 §3.4）。

        `action` ∈ delete / archive / unarchive / update（update 带 `text` 与 `field`，
        一次一条）；`ids` 可以多条——删除 / 归档 / 取消归档支持"要么全动、要么不动"。

        `field`（2026-09-24：**三层都能改标签**）：改哪个字段——中文名或键都认，
        每层的可改字段表见 `store.LAYER_EDIT`（场景最多；摘要 / 画像 = 主文本 + 主题）。

        分派在 `weave.*_by_layer`——**和对话确认条共用一份**：那边是她提议、他点
        确认；这边是他在台账页直接点（画像 / 摘要 / 场景三处页面同一个口）。
        """
        ids = [str(i).strip() for i in (ids or []) if str(i).strip()]
        if not ids:
            return {"ok": False, "detail": "要给编号（如 S1-0003 / S2-0001 / S3-0001）"}
        if action == "delete":
            return delete_by_layer(self.store, ids)
        if action == "archive":
            return archive_by_layer(self.store, ids)
        if action == "unarchive":
            return unarchive_by_layer(self.store, ids)
        if action == "update":
            if len(ids) != 1:
                return {"ok": False, "detail": "改一次只能一条"}
            return update_by_layer(self.store, ids[0], text, field)
        return {"ok": False, "detail": f"不认的动作：{action}"}

    def scenes(self, limit: int = 200) -> list[dict]:
        """场景卡列表（**含冷层**——这一页是台账，藏起冷层会让人以为记忆变少了）。

        每条带 `has_vec`：降级期写进去的卡没有向量，查「它为什么从没被唤醒」先看这个。
        """
        out = []
        for s in self.store.query_scenes(include_archived=True, limit=limit):
            out.append({
                "id": s.id, "title": s.title, "topic": s.topic, "subject": s.subject,
                "text": s.text, "time": (s.time_event or "")[:16],
                "valence": s.valence, "arousal": s.arousal,
                "intensity": round(s.intensity or 0, 3),
                "trigger": s.trigger, "trigger_class": s.trigger_class,
                "reaction": s.reaction, "outcome": s.outcome,
                "sensitive": bool(s.sensitive),
                "mention": s.mention_count, "cited": s.cited_by_profile,
                "archived": bool(s.archived),
                "entities": self.store.entities_of_scene(s.id),
                "has_vec": s.emb is not None,
            })
        return out

    def summaries(self) -> list[dict]:
        """S2 列表（聚合出来的主题叙述）。50 是台账页自己的量，
        与「注入带几条」（`recall.inject_summaries_n`）不是一件事。

        素材列给**现存的**（引用 = `sources ∩ 现存节点`，2026-09-24）：
        删掉的场景不该在台账里继续挂着。
        """
        return [{"id": x.id, "topic": x.topic,
                 "topics": x.topics or [x.topic],      # 多标签：1 主 + 最多 2 附
                 "text": x.text,
                 "sources": [i for i in (x.sources or []) if self.store.exists(i)],
                 "archived": bool(x.archived),
                 "time": (x.created_at or "")[:16]}
                for x in self.store.hot_summaries(50)]

    def memos(self) -> list[dict]:
        """备忘录（含到点标记 `due_now`）。

        2026-10-05 晚改口径：**`due_now` 就是"有资格进注入"**——时机表与
        `should_raise` 都删了，后面不再有别的闸。它还是会被"同组提过 / 被挤掉"
        挡住，所以界面照旧原样给出——「为什么这句没问出口」要有据可查。
        """
        now = None
        due_ids = {m.id for m in due(self.store, now)}
        out = []
        for m in self.store.open_memos():
            out.append({"id": m.id, "content": m.content, "kind": m.kind,
                        # 组名（2026-09-22）：界面靠它把"一件事的多步"聚在一个组头下
                        "group_name": getattr(m, "group_name", "") or "",
                        "kind_class": m.kind_class, "window_days": m.window_days,
                        "due_at": m.due_at, "status": m.status,
                        "sensitive": bool(m.sensitive),
                        "raise_mode": int(m.sensitive or 0),
                        "due_now": m.id in due_ids})
        return out

    # （`openings` / `seen_openings` 2026-10-05 晚删除——留言这条路整块不要了，
    #   `openings` 表在 `store` 里退役留底。）

    def memo_close(self, mid: str) -> dict:
        """人手划掉一件备忘录（2026-09-21，设计稿 D 条 4）——**和她说 `close_memo` 同一个动作**。

        「三方都能关」里的那一方：她判（命中判定 / close_memo 工具）、
        用户随手划（对话右侧面板走这里）、系统兜底（超期退役）。走 `memo.close` 而不是 `store.close_memo`——
        **闭合回流**（场景卡里的钩子一起标掉）和工具那条路完全一致。

        划错了没有撤销按钮：但代价只是"她不再主动提它"，钩子本身还在卡上
        （`closed_at` 标着）——真要翻案，改库比加一个撤销通道便宜。
        留痕（"用户随手划"那一份）由 `memo.close` 统一写，这里不再自己写——
        免得同一件事留两条。
        """
        mid = (mid or "").strip()
        if not mid:
            return {"ok": False, "detail": "没给编号"}
        m = self.store.get_memo(mid)
        if m is None:
            return {"ok": False, "detail": f"找不到 {mid}"}
        if m.status == MEMO_CLOSED:
            return {"ok": False, "detail": f"{mid} 已经关过了"}
        memo_close_flow(self.store, mid, by="用户随手划")
        return {"ok": True, "detail": f"划掉了 {mid}：「{m.content}」"}

    def entities(self) -> list[dict]:
        """实体索引 + 每条涉及多少场景 + 关系词 + 活跃标记。

        没有场景计数就分不出「他反复提起的是哪个」；`relations` 是逐场景
        关系词的去重汇总（「妈妈」「同事」）——2026-09-25 加，
        它回答的是「这个人对他意味着什么」的头半句。

        `last_at` / `active`（2026-09-25）：最近一次交互（关联场景里最新一条的
        事件时间）与「`entity.active_days` 内」的判断。**派生、不落库**——
        实体没有"状态"字段，活跃只是给人看的一眼（标记 + 排序）；
        它不参与准入、不触发删除（沉寂的名字正是长程记忆该留着的——
        半年没提的人被提起时，索引必须在）。
        """
        cutoff = (datetime.now() - timedelta(
            days=int(cfgmod.cfg("entity", "active_days", default=30)))
        ).strftime("%Y-%m-%d %H:%M:%S")
        out = []
        for e in self.store.all_entities():
            n = self.store.conn.execute(
                "SELECT count(*) AS n FROM scene_entities WHERE entity_id = ?",
                (e.id,)).fetchone()["n"]
            rels = [r["relation"] for r in self.store.conn.execute(
                "SELECT DISTINCT relation FROM scene_entities"
                " WHERE entity_id = ? AND relation IS NOT NULL AND relation <> ''"
                " LIMIT 5", (e.id,)).fetchall()]
            last_at = self.store.conn.execute(
                "SELECT max(s.time_record) AS t FROM scene_entities se"
                " JOIN scenes s ON s.id = se.scene_id WHERE se.entity_id = ?",
                (e.id,)).fetchone()["t"] or ""
            out.append({"id": e.id, "name": e.name, "kind": e.kind,
                        "aliases": e.aliases, "scenes": n, "relations": rels,
                        "last_at": last_at, "active": last_at >= cutoff})
        # 最近交互的在前面（活跃的自然在前）——实体页一眼看到「还在用的」。
        out.sort(key=lambda x: x["last_at"] or "", reverse=True)
        return out

    def trace(self, limit: int = 20) -> list[dict]:
        """留痕页要的那几行（按记录自己的时间倒序）。

        这里读的是 JSONL 文件而不是库——trace 本来就是「不占库的观测数据」，
        量再大也不影响记忆本身的容量。

        ⚠️ 取文件**按修改时间倒序**（2026-10-07 改；原来是"按文件名倒序取三份"）：
        文件名首字决定胜负，`漂移` / `档案` / `整理` 的码点都比 `唤醒` 大——
        那几类一多，三个名额就被占满，**唤醒记录一条也进不来**，而"她为什么没想起
        那件事"只有唤醒那类答得出来（2026-10-06 复核实测：取到三份「漂移」）。
        现在取"最近写过的若干份"，再按记录自己的 `ts` 倒序——页面上看到的是
        **最近发生了什么**，不再是"文件名叫什么"。

        每条记录补一个 `kind`（取自文件名前缀：唤醒 / 备忘 / 工具 / 漂移 / 整理 /
        档案 / 场景改动 / 否决）：**页面靠它分派渲染**。以前页面靠字段猜形状，
        「工具」「档案」那几类猜不中，就被硬当漂移检测渲染
        （"检查 undefined 个主题，漂移 0 个"就是这么冒出来的）。
        """
        trace_dir = cfgmod.abspath(cfgmod.PATHS["trace_dir"])
        if not trace_dir.exists():
            return []
        files = list(trace_dir.glob("*.jsonl"))

        def _mtime(p) -> float:
            """文件修改时间——**读不到就当 0**（排到最后）：一个刚被别处挪走 / 锁住的
            文件不该把整页打掉（`stat` 会抛 `OSError`，而这里是页面每次打开都要走的）。"""
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        # 按修改时间倒序（最近的观测先读）；份数封顶只是别在文件越攒越多时越读越慢
        files.sort(key=_mtime, reverse=True)
        out: list[dict] = []
        for f in files[:12]:
            try:
                lines = f.read_text(encoding="utf-8").strip().splitlines()
            except Exception:
                continue
            kind = f.stem.split("-")[0]          # `唤醒-20261007` → 唤醒
            # 每份**只看尾巴**：文件是追加写的（天然按时间排），而页面只要最近那些——
            # 一份攒到几万行时整读一遍纯属白花（读 12 份 × 全量就更明显了）
            for line in lines[-200:]:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    rec.setdefault("kind", kind)
                    out.append(rec)
        out.sort(key=lambda r: str(r.get("ts") or ""), reverse=True)
        return out[:limit]

    def topic_merge_suggestions(self) -> list[dict]:
        """疑似「同一个主题被写成两种说法」的候选对（**只建议，不合并**）。

        topic 是 LLM 自由生成的字符串，裂开是常态；裂缝不补，聚合与画像
        就会因为「每组都不够 3 条」而永远立不起来。这里把嫌疑列出来，
        合不合由人在界面上点——合并是不可逆的语义断言，不该由代码自己决定。
        """
        return merge_suggestions(self.store, self.emb)


# ---------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    app: App = None            # 由 serve() 注入
    server_version = "air-link-dashboard"

    # ---- 工具 ----

    def _json(self, data, code: int = 200) -> None:
        """唯一的响应出口。`default=str` 兜住偶然混进回包的 `datetime` / `Path`——
        没有它整条响应会变成 500，而那个 500 连"哪个字段"都不说。"""
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        """读 JSON 请求体。**坏 / 空都给 `{}` 而不抛**——让请求走到各 handler 里，
        那里本来就有「必须给 xx」的提示，比一个 400 更好改。"""
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            return {}

    def _html(self, name: str) -> None:
        """只服务 `web/` 下的白名单文件（不做通用静态服务：本地工具不需要，也不需要那种风险）。"""
        path = cfgmod.abspath(cfgmod.PATHS["web"]) / name
        if not path.exists() or path.suffix not in (".html", ".css", ".js"):
            self._json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        ctype = {"html": "text/html", "css": "text/css", "js": "application/javascript"}
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype.get(path.suffix[1:], 'text/plain')}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- 路由 ----

    def do_GET(self) -> None:                     # noqa: N802
        """**只读**路由表——都不改记忆（送达标记、改和删都在 POST 那边）。

            /api/state       首屏那包（计数 / 开关 / 语言 / 窗口）
            /api/mirror      镜像：画像 + 依据 + 上次整理
            /api/persona  /api/voice  /api/voice/config
                                                人格（列表；`?id=` 给单份正文）/ 声音
            /api/scenes  /api/summaries  /api/memos           记忆三件套
                                                 （`/api/openings` 2026-10-05 晚撤）
            /api/history /api/entities                            回填 / 实体
            /api/salvage /api/trace /api/settings                 打捞 / 唤醒留痕 / 配置
            /api/topic-merge /api/facts                           归并建议 / 档案

        返回形状看 `App` 上同名的方法。整体 try/except 包一层：
        一个视图炸了不该让整个服务死掉，500 里带上异常名（使用者唯一看得到的）。
        """
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        app = self.app
        try:
            if u.path in ("/", "/index.html"):
                return self._html("index.html")
            if u.path == "/api/state":
                return self._json(app.state())
            if u.path == "/api/mirror":
                return self._json(app.mirror())
            if u.path == "/api/persona":
                # 人格列表**从文件系统来**（扫 self/personas/）——加一个 md 即多一个选项。
                # 2026-10-05 起把设置页要的那一屏（显示名 / 字数 / 常驻块体量 / 新建模板）
                # 一并给出去：前端不自己拼这些数（同"不抄后端常量"那条）。
                # `?id=` 则单给一份正文（编辑框用，停用的也读得到）。
                pid = (q.get("id") or "").strip()
                if pid:
                    it = persona_file.info(pid)
                    if it["enabled"]:
                        it["fixed"] = fixed_blocks_report(
                            persona_file.load(pid), pid, persona_names())["chars"]
                    return self._json({"id": pid, "item": it,
                                       "text": persona_file.text(pid)})
                return self._json(app.persona_list())
            if u.path == "/api/voice":
                # 她的声音：**他的设置 + 服务当前状态一趟取全**（页面要同时显示两样：
                # 选什么，以及服务现在到底是什么样——后者今天出过两次"没接上"）
                return self._json({"voice": app.voice(), "status": app.voice_status()})
            if u.path == "/api/voice/config":
                # 语音服务的配置文件（模型 / 精度 / 语言 / 空闲卸载）——**只读白名单**
                return self._json(app.voice_config())
            if u.path == "/api/scenes":
                return self._json(app.scenes(int(q.get("limit", 200))))
            if u.path == "/api/summaries":
                return self._json(app.summaries())
            if u.path == "/api/memos":
                return self._json(app.memos())
            if u.path == "/api/history":
                # 重启/刷新后对话区回填：窗口里还留着的对话
                # （只读文件，**不建会话**——页面加载不该顺手把她唤醒）
                return self._json(app.history())
            if u.path == "/api/entities":
                return self._json(app.entities())
            if u.path == "/api/salvage":
                # 打捞：**先按时间缩范围，再在原文里找**。
                # 这是人主动翻原文的口子，**不进自动唤醒**（原文没有索引，翻文件慢）。
                return self._json({"rows": salvage.search_raws(
                    app.store, q.get("q") or "", q.get("from") or "", q.get("to") or "",
                    emb=app.emb, limit=int(q.get("limit", 20)))})
            if u.path == "/api/trace":
                return self._json(app.trace(int(q.get("limit", 20))))
            if u.path == "/api/settings":
                return self._json(settings.describe())
            if u.path == "/api/topic-merge":
                return self._json({"suggestions": app.topic_merge_suggestions()})
            if u.path == "/api/facts":
                # `base` 一并返回：常用键从 `model.FACT_KEYS` 来，页面照着渲染——
                # 不在前端再抄一份（抄了就会和后端漂移，同档位那份的道理）。
                return self._json({"facts": app.store.all_user_facts(),
                                   "base": list(FACT_KEYS)})
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    @staticmethod
    def _voice_override(body: dict):
        """页面直接指定的嗓子——**只有「试听」会给**（他还没保存，想先听这一口）。
        没给就 None：用他存在 `user_prefs` 里的那份（见 `App.voice()`）。
        """
        out: dict = {}
        if "speaker" in body:
            out["speaker"] = (body.get("speaker") or "").strip()
        if "instruct_extra" in body:
            out["instruct_extra"] = body.get("instruct_extra")
        if "dims" in body:
            out["dims"] = _clean_dims(body.get("dims"))
        return out or None      # **只给哪项就只盖哪项**——给一项不能顺手把另一项也重置了

    def _pcm_stream(self, sr: int, frames) -> None:
        """PCM 帧流：**来一帧写一帧**（同 `_sse`，只是换成二进制）。

        帧格式（和 `tts/server.py` 是同一份约定，两边要一起改）：
            12 字节头：`ARAU` + 采样率 u32 + 声道 u16 + 保留 u16（小端）
            之后重复：  样本数 i32 + Int16 PCM

        为什么不让页面直连语音服务：页面不用知道端口和模型（同「数字只写一个地方」），
        也少一层 CORS。
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self.wfile.write(struct.pack("<4sIHH", b"ARAU", int(sr), 1, 0))
            for pcm in frames:
                self.wfile.write(struct.pack("<i", len(pcm) // 2))
                self.wfile.write(pcm)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return          # 他按了停 / 关了页面：什么都没发生

    def _sse(self, events) -> None:
        """SSE 响应：来一块写一块，**写完立刻 flush**。

        不设 `Content-Length`（不知道会有多长）。`X-Accel-Buffering: no` 是写给
        中间层看的——有的反向代理会攒着不发，那这一整条流式就白做了。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for ev in events:
                # `_sse_safe`：end 事件里的 recall 装着 dataclass 对象，
                # 不过这一道 `json.dumps` 会抛 TypeError——**整条流断在结尾**，
                # 右侧面板和提取回执全都收不到（见 `_sse_safe` 的说明）。
                self.wfile.write(
                    f"data: {json.dumps(_sse_safe(ev), ensure_ascii=False)}\n\n"
                    .encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return          # 他关了页面 / 按了取消：什么都没发生，不用管

    def do_POST(self) -> None:                    # noqa: N802
        """**写**路由表。四类，边界很清楚：

        ① **内容写入——只有一个口子**：/api/chat/stream
           走 `ChatSession` 的完整链路。绕过对话层直接塞场景 / 画像 / 备忘的
           口子一个都不开。（非流式的 `/api/chat` 2026-10-09 删——仓内无人调。）

        ② **人的纠正**：/api/confirm（提议的改删三选一、他点一下才算数）·
           /api/memory-action（**改 / 删 / 归档 / 取消归档——三层同一个口**，
           2026-09-24 由原来的 `/api/scene-*` 五个口并成）· /api/reject ·
           /api/reviews（复核提议「留着」）·
           /api/topic-merge · /api/entity-merge · /api/facts ·
           /api/lang · /api/salvage-rebuild · /api/memo-close（划掉一件备忘录）

        ③ **人格文件**（不改记忆，改的是"她是谁"）：/api/persona（切换）·
           /api/persona-create · /api/persona-save · /api/persona-enable（停用 / 恢复）
           ——**文件是真源**（`self/personas/*.md`），界面只是编辑器；每次真改动
           写 `data/trace/人格-*.jsonl`（含改前全文）。2026-10-05 加

        ④ **环境与外挂**（不改记忆）：/api/settings(+preset/test) ·
           /api/voice(+config/start/stop) · /api/speak ·
           /api/new-session · /api/distill · /api/turn-undo · /api/quit
           （`/api/openings-seen` 2026-10-05 晚撤——留言整块不要了；
           `/api/end` 2026-10-09 撤——与 `/api/new-session` 同一个动作、仓内无人调）

        ⚠️ `/api/quit` 先回话、再在后台线程里收尾（收语音要等十几秒，见 `_quit_all`）。
        """
        u = urlparse(self.path)
        body = self._body()
        app = self.app
        try:
            if u.path == "/api/chat/stream":
                # 内容写入的**唯一** HTTP 口（流式，思维链可展开）。2026-10-09 删掉了
                # 非流式的 `/api/chat`：仓内没人调（demo 与实验回放直接拿 `ChatSession`），
                # 留着就是一条没人走过的路。`App.chat` 那个非流式方法仍在（测试与
                # 进程内调用用），两条路共用对话层同一条链路。
                msg = (body.get("msg") or "").strip()
                if not msg:
                    return self._json({"error": "空消息"}, 400)
                return self._sse(app.chat_stream(msg))
            if u.path == "/api/salvage-rebuild":
                # 照原文重新记一遍（**不是撤销**：备份才给回原来那张卡）
                out = salvage.rebuild_confirmed(
                    app.store, app.llm, (body.get("scene_id") or "").strip(),
                    on_date=(body.get("day") or "").strip(), emb=app.emb)
                return self._json(out)
            if u.path == "/api/confirm":
                # 改与删的**主通道**：她问出口 → 他在这里点一下。
                # 点了就算数，不用去猜他一句话是什么意思。
                # 删除三档 `action`：delete / archive / keep（2026-09-23，工具箱稿 §3.4）；
                # 老页面缓存的 `accept: bool` 仍认（旧值就是"执行/算了"两档）。
                return self._json(app.confirm(
                    body.get("action") if "action" in body else body.get("accept")))
            if u.path == "/api/memory-action":
                # **记忆动作的统一口**（2026-09-24，工具箱稿 §3.4）：改 / 删 / 归档 /
                # 取消归档——三层同一个口，按编号前缀分派（`weave.*_by_layer`，
                # 与上面的确认条共用一份）。原来散着的 `/api/scene-*` 五个口已并进来。
                return self._json(app.memory_action(
                    body.get("ids") or ([body.get("id")] if body.get("id") else []),
                    (body.get("action") or "").strip(),
                    body.get("text") or "",
                    (body.get("field") or "text").strip()))
            if u.path == "/api/memo-close":
                # 人随手划掉一件备忘录（2026-09-21，设计稿 D 条 4）：
                # 对话右侧那个折叠面板点一下就走这里——**和她说 `close_memo`
                # 同一个动作**（含闭合回流）。返回刷新后的全列，面板直接重渲染。
                out = app.memo_close((body.get("id") or "").strip())
                out["memos"] = app.memos()
                return self._json(out)
            if u.path == "/api/new-session":
                # 「新对话」：同样的提取 + 整理，多一步连压缩摘要一起清（见 App.new_session）
                return self._json({"written": app.new_session()})
            if u.path == "/api/distill":
                # 手动补跑：没点「新对话」、或者想立刻看结果时用
                return self._json({"started": app.start_distill()})
            if u.path == "/api/settings":
                settings.save_local(body.get("config") or {})
                settings.apply()
                # 配置变了要重建客户端（endpoint / key 换了，旧的连接没有意义）
                app.llm = LLM()
                app.emb = build_embedding()
                app._session = None
                return self._json(settings.describe())
            if u.path == "/api/settings/preset":
                settings.apply_preset(body.get("name") or "")
                app.llm = LLM()
                app.emb = build_embedding()
                app._session = None
                return self._json(settings.describe())
            if u.path == "/api/settings/test":
                return self._json(settings.test_connection(body.get("section") or "llm"))
            if u.path == "/api/topic-merge":
                # 人确认后的纠正（不是内容写入，见文件头「两条约定」的补充）
                out = merge_topics_confirmed(app.store, body.get("from") or "",
                                             body.get("to") or "")
                out["suggestions"] = app.topic_merge_suggestions()
                return self._json(out)
            if u.path == "/api/entity-merge":
                # 两个名字其实是同一个东西（两个「小明」）——人点一下，把其中一个并进另一个。
                # 同 topic-merge：这是「人的纠正」，不是内容写入；留痕在 weave。
                return self._json(merge_entities_confirmed(
                    app.store, body.get("from") or "", body.get("to") or ""))
            if u.path == "/api/persona":
                # 切人格：谁陪他聊。**他挑的**——切了下一句就换人（每轮现读）。
                return self._json({"persona": app.set_persona((body.get("persona") or "").strip())})
            if u.path == "/api/persona-create":
                # 新建人格：**文件是真源**——这里只负责校验 + 落文件 + 留痕
                # （校验在 `persona.create`：id 字符集 / 保留名 / 撞名 / 撞停用副本）。
                return self._json(persona_file.create(
                    (body.get("id") or "").strip(), body.get("text") or ""))
            if u.path == "/api/persona-save":
                out = persona_file.save((body.get("id") or "").strip(),
                                        body.get("text") or "")
                if out.get("ok") and out.get("changed") and \
                        out["item"]["id"] == app.persona():
                    # 正用着它：说清楚"什么时候生效"——每轮现读，下一句就是新的
                    out["note"] = "下一句起，说话就用这份新的（人格每轮现读）"
                return self._json(out)
            if u.path == "/api/persona-enable":
                # 停用 / 恢复（改名 `.md.off` ↔ `.md`，**不删文件**）。
                # 停掉的若是**当前正用着的**，顺手切回 air——否则顶栏指着一个不在
                # 名单里的名字（下拉里没有它，他自己都选不回来）。
                pid = (body.get("id") or "").strip()
                on = bool(body.get("on", True))
                out = persona_file.set_enabled(pid, on)
                if out.get("ok") and not on and app.persona() == pid:
                    app.set_persona("air")
                    out["switched_to"] = "air"
                return self._json(out)
            if u.path == "/api/speak":
                # 语音播报：**可选外挂**（本地语音服务，见 tts/）。它没起，这里如实回报，
                # 页面照常说话——播报失败绝不能影响对话本身。
                # `stream: true` 走帧流（她开口只等第一帧）；其余仍走整段 + data URI。
                ov = self._voice_override(body)     # 只有试听会给（他还没保存）
                if body.get("stream"):
                    got = app.speak_stream(body.get("text") or "", ov)
                    if isinstance(got, dict):
                        return self._json(got)
                    return self._pcm_stream(got[0], got[1])
                return self._json(app.speak(body.get("text") or "", ov))
            if u.path == "/api/turn-undo":
                # 改 / 重新生成的前半步：撤掉窗口末尾的 **N 轮**（默认 1）。
                # 铅笔挂在窗口里**每一条**用户消息上，改第 2 轮就撤 2 轮
                # （见 `ShortTerm.undo_turns`）；更早的那些已经进长期库，撤不了。
                # `expect` 是页面那边那条消息的正文：带上它，窗口在两次点击之间
                # 被提取过（话题切换 / 新对话 / 超预算）时会**一个字都不撤**地拒掉，
                # 免得按 N 撤到别的轮次上。
                # 撤完由页面照常走一次 `/api/chat/stream`，把新的一句重说一遍。
                return self._json(app.undo_turns(
                    int(body.get("turns") or 1), (body.get("expect") or "").strip()))
            if u.path == "/api/lang":
                # 语言（中 / 英）：**他定**。改动留痕——语言也是她说话的一部分
                # （切了它，对话原文与提取都跟着变）。
                return self._json({"lang": save_lang_confirmed(
                    app.store, (body.get("lang") or "").strip())})
            if u.path == "/api/voice":
                # 声音设置：**他定，存下就生效、不用重启**——音色是每次请求的参数，
                # 不是加载时定死的东西（换模型才要重启，那件事在「语音」页改 voice.json）。
                # 不留痕，理由见 App.voice()：嗓子不改变她记住什么。
                return self._json({"voice": app.voice_save(
                    (body.get("speaker") or ""), body.get("instruct_extra"),
                    body.get("dims"))})
            if u.path == "/api/voice/config":
                # 写语音配置（白名单字段）：**主项目只当文件代理**——不解释这些字段的
                # 语义，原样读、原样写（语义的主人在语音服务那边，同 tts.endpoint 的定位）
                return self._json(app.voice_config_save(body.get("config") or {}))
            if u.path == "/api/voice/start":
                # 起语音服务：**人点的一下**（系统不自动拉起——它会吃几个 G 显存）
                return self._json(app.voice_start())
            if u.path == "/api/voice/stop":
                # 关语音服务：**放掉显存**（它自己退，见 App.voice_stop）
                return self._json(app.voice_stop())
            if u.path == "/api/quit":
                # **一键全退**：先把外挂收干净（语音服务 + 显存），再关自己——
                # 顺序不能反（仪表盘一没，就没人去收那个进程了），但"回话"要**提前**：
                # 收语音可能要等它十几秒（探测 + 等端口不再应答），让调用方干等这么久，
                # 它一不耐烦断开连接，写响应失败还可能把后面的 `shutdown()` 一起带走
                # （`_json` 在断开的 socket 上会抛）。所以先回一句"正在退出"，
                # 真正的收尾在后台线程里按原顺序跑（见 `_quit_all`）。
                threading.Thread(target=self._quit_all, daemon=True).start()
                return self._json({"ok": True,
                                   "detail": "正在退出：先收语音服务（放显存），再关仪表盘"})
            if u.path == "/api/reject":
                # 否决一条画像 = **真删**（2026-09-24）：他在说「我不是这样」，
                # 删的是"一条不成立的推断"——素材都在，判断真成立会重立。
                # 不再收 reason：那个理由以前入库没人读（界面也不问了）。
                return self._json(app.reject((body.get("id") or "").strip()))
            if u.path == "/api/reviews":
                # 「留着」：人看过复核提议、决定不动这条画像（提议过期，画像不改）。
                # 作废那条路走 `/api/reject`（人的否决，同一个出口）——
                # 两个按钮两个动作：只给"作废"会逼人二选一，而"没什么问题"也该能点。
                n = app.store.mark_review_handled((body.get("id") or "").strip())
                return self._json({"ok": bool(n), "handled": n,
                                   "reviews": app.maint_reviews()})
            if u.path == "/api/facts":
                # 档案也是「人的纠正」这一类：这是他自己的信息，当然由他直接改。
                # 空值 = 删除；改动会留痕（见 weave.save_facts_confirmed）。
                n = save_facts_confirmed(app.store, body.get("facts") or {})
                return self._json({"saved": n, "facts": app.store.all_user_facts()})
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _quit_all(self) -> None:
        """`/api/quit` 的后台收尾：**先语音、再自己**（顺序理由见那里的处理）。"""
        try:
            self.app.voice_stop()
        finally:
            self.server.shutdown()

    def log_message(self, fmt, *args) -> None:
        """默认的每请求一行日志太吵（前端轮询会刷屏），只留错误。"""
        if args and str(args[1]).startswith(("4", "5")):
            print(f"[dashboard] {self.address_string()} {fmt % args}")


def _sse_safe(ev: dict) -> dict:
    """SSE 序列化前的最后一道转换。

    **为什么必须有它**：流式 end 事件里的 `recall` 是唤醒引擎的原始结果——
    里面装着 `Scene` / `Profile` / `Summary` / `Raw` 这些 dataclass 对象，
    `json.dumps` 直接抛 `TypeError`（对象不是 JSON 可序列化的）。
    后果不是少一个字段，是**整条流在结尾处断掉**：右侧线索面板、
    「提取了一段对话」的回执、确认条全都收不到（真断过）。
    `_json()` 有 `default=str` 兜着，SSE 这边没有，也不该有——
    `default=str` 只会把对象变成一串 `Scene(id=...)` 的字符串塞给前端，
    前端拿到的是 `undefined`，比报错更难查。所以在这里**显式压成视图**。
    """
    if isinstance(ev, dict) and ev.get("type") == "end":
        ev = dict(ev)
        ev["recall"] = _recall_view(ev.get("recall") or {})
    return ev


def _recall_view(recall: dict) -> dict:
    """把唤醒结果压成前端好用的形状（不带向量、不带 ORM 对象）。"""
    return {
        "cues": recall.get("cues_view") or {},
        "actions": recall.get("actions") or {},
        "scenes": [{"id": s.id, "title": s.title, "topic": s.topic,
                    "why": (recall.get("why") or {}).get(s.id, "")}
                   for s in recall.get("scenes") or []],
        "profiles": [{"id": p.id, "statement": p.statement, "status": p.status,
                      "evidence": p.evidence}
                     for p in recall.get("profiles") or []],
        # 抑制名单两份：id 串（拼着看）+ 带原因（回答「为什么没有它」）
        "suppressed": recall.get("suppressed") or [],
        "suppressed_detail": recall.get("suppressed_detail") or [],
        # 整轮卡在哪一步（`""` / `"R0"` / `"weak_gate"`）——
        # 「什么都没想起」的时候，界面得能说出是**为什么没想起**。
        "stage": recall.get("stage") or "",
        # `Raw` 没有 id——它的编号就是 scene_id（原文没有独立索引，见 model.Raw）
        "raws": [{"id": r.scene_id, "content": (r.content or "")[:120]}
                 for r in recall.get("raws") or []],
        # 常备备忘录（设计稿 D2b；2026-10-05 晚并栏）：这轮她手上有哪几件
        # （含为什么给——到点/最近/相关/追问；到点那件带 `due` 标记）
        "standing_memos": recall.get("standing_memos") or [],
        "flags": recall.get("flags") or {},
        # 上下文总预算裁掉了什么（`chat.fit_context` 把报告挂在 recall 上）——
        # 「应该出现的东西没出现」这一类要靠它回答
        "budget": recall.get("budget") or {},
    }


def _hours_since(stamp: str) -> float:
    """距某个时间戳多少小时；没记过 / 解析不了 → 一个大数（= 该跑）。

    为什么解析不了也当"该跑"：体检幂等且空转便宜——宁可多跑一次，
    也不要因为一个坏时间戳**从此永不体检**（那正是这次要修的毛病）。
    """
    try:
        t = datetime.strptime(stamp or "", "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return 1e9
    return max(0.0, (datetime.now() - t).total_seconds() / 3600.0)


def _maintenance_loop(app: App, interval: float) -> None:
    """服务跑着时的低频循环（原 `_proactive_loop`，2026-10-05 晚改名）：

    一跳只做一件事——**体检判定**（`maintenance_tick`，2026-09-23 加）：
    对内、勤快，量到或时间到就跑一次整理（记忆整理稿 §三），不开口、不改内容。

    原来它还管"主动开口检查"（她先开一句 + 留言）——那条 2026-10-05 晚整块删了
    （见待优化稿 K 条），循环因此只剩体检一条职责。
    放在**服务端**：体检是定时的事，不该等页面开着才做；先跑第一次再睡。
    """
    while True:
        try:
            app.maintenance_tick()
        except Exception as e:
            print(f"[dashboard] 体检判定失败（不影响其它功能）: {e}")
        time.sleep(interval)


def serve(port: int = 8765, open_browser: bool = True, with_voice: bool = False) -> None:
    """起服务（**阻塞**）：`run.cmd` / `启动.vbs` 最终走到这里。

    `with_voice=True` 时才一并起语音服务（它要吃几个 G 显存，
    不该由一个默认参数悄悄占着）。收尾顺序见 `App.wait_distill` / `_quit_all`：
    提炼要跑完、语音要先收、仪表盘最后关。
    """
    # 控制台可能是 GBK（chcp 936），而横幅 / 提示里可能有编不进 GBK 的字符
    # （`⚠️` 这类 emoji 等）——print 一次就 UnicodeEncodeError：**启动横幅处直接崩**，
    # 表现为双击 run.cmd 后黑框里一段报错、浏览器根本不打开。
    # （pythonw 下没有控制台，print 被静默丢弃，所以启动.vbs 反而看不到这个毛病。）
    # 兜底成「?」而不是崩：横幅是给人看的，不值得为它让整个仪表盘起不来。
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:                       # pythonw 下是 None
                stream.reconfigure(errors="replace")
        except Exception:
            pass
    settings.apply()
    app = App(port)
    Handler.app = app

    # 体检的定时判定（原"主动开口"那条 2026-10-05 晚删了）：低频看一眼，10 分钟一跳。
    # 和 HTTP 同进程的 daemon 线程：Ctrl+C 就一起结束了。
    interval = max(30, int(cfgmod.cfg("memo", "check_interval", default=600) or 600))
    threading.Thread(target=_maintenance_loop, args=(app, interval), daemon=True).start()

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print("=" * 62)
    print(" air-link-01 · 记忆实验台")
    print("=" * 62)
    print(f" 地址     : {url}")
    print(f" 记忆库   : {app.store.path}")
    print(f" 已有场景 : {app.store.count('scenes')} 条")
    conf = settings.describe()
    print(f" LLM      : {conf['llm']['endpoint']}  {conf['llm']['model']}"
          f"  key={conf['llm']['api_key'] or '(未配)'}")
    print(f" 语义     : {conf['embedding']['endpoint']}  {conf['embedding']['model']}"
          if app.emb else " 语义     : 未配置（降级为字符重叠）")
    print("\n Ctrl+C 退出")
    print("=" * 62)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    if with_voice:
        # 启动器带了 `--with-voice`：**等于他事先点了一次顶栏「语音」**。
        # 不写这个开关就不自动起——自动拉起是替他做决定，得由他显式写下来
        # （同「只有人能切模式」那条）。放线程里起，别让页面等它。
        threading.Thread(target=app.voice_start, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    # **先把端口松开再收尾**：收尾要调一次 LLM 提取（十几秒），期间页面已经写"已退出"了，
    # 人要是在这十几秒里再双击启动器，会撞上一个"还在听但没人应答"的端口——
    # 那种失败是无声的（无窗口）。松开端口，收尾慢慢做。
    httpd.server_close()
    # 收尾放在**两条退出路径都要经过的地方**：Ctrl+C 走到这，「全退出」也走到这。
    # 原来它写在 `except KeyboardInterrupt` 里，而 `/api/quit` 走的是 `server.shutdown()`
    # ——那条路不抛 KeyboardInterrupt，于是"从界面退出"会**少收一次尾**（最后一段对话丢了
    # 就真丢了）。同一件收尾，不该因为怎么退出的而有差别。
    print("\n[退出] 正在收尾（提取最后一段对话）…")
    try:
        # **走 `app.end()`，不要在这里另写一遍 `session.close()`**：
        # 「收尾」和「整理」是一件事的两半，分开写就一定会漏一半
        # （原来这行是 `session.close()`，于是从界面 / Ctrl+C 退出都只收尾、不整理）。
        app.end()
        if not app.wait_distill():
            print("[退出] 整理还没跑完，先退了（S1 已落库，下次收尾会再整理）")
    except Exception:
        pass


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="air-link-01 记忆实验台")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument("--with-voice", action="store_true",
                   help="启动时顺手把本地语音服务也拉起来（启动.vbs 用的就是它）")
    a = p.parse_args()
    serve(a.port, open_browser=not a.no_browser, with_voice=a.with_voice)
