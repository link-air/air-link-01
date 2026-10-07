"""短期记忆：上下文窗口管理。

一句话定位：**短期记忆是长期记忆谱系的最浅层，不是另一套东西。**
所以这里的「压缩」不是另写一套摘要逻辑，而是直接调 `distill_step1`——
一次 LLM 调用同时完成「腾窗口」和「存记忆」。

四条提取触发，任一满足即提取：
  1. **话题切换**（`should_cut`）——主路径。切换的那一刻就是上一段该提取的时刻，
     切场景和提取触发**是同一件事**，不引入新机制。
  2. **超 M 轮未切换**——防一段太长一直不提取。
  3. **窗口 token 超预算**——设计稿要求「按预算不按条数」：
     内容长度差异极大，按条数会失控。它只压**最老的一批**，保留尾部逐字。
  4. **会话结束**——`end_session()`（界面上「新对话」/ 收尾走它，`chat.close()` 调）。
     ⚠️ 空闲那条（`session_idle_min`，默认 30 分钟没消息）**当前判不出来**：判定挂在
     `flush_if_needed` 里，而 `chat` 是**先 `append`（刷新 `_last_active`）再 flush**
     ——那一刻的空闲时长恒为 0，只有"读文件后直接 flush"（测试里）才为真。
     本轮只记不改（要改得把它挪到 `append` 之前，动的是 `chat` 的调用顺序）。

第 3 条和另外三条的语义差别，是这份实现里唯一需要留神的地方：
预算触发时对话**还在继续**（只压最老一批），其余三种是**这段结束了**（整段提取）。

提取时**有一条例外**（2026-10-07 定）：被叫停的半句（`interrupted`，「停止」
那一路）**只活在窗口里，不进长期库**——滤在 `compress_and_extract`，
理由写在那一步（半句没有信息量，留着还会让她以为那句说完了）。

窗口渲染分两档（`build_window`）：
  1. **最近 `verbatim_messages` 条**：逐字（LLM 需要确切知道用户刚说了什么）
  2. **更早的**：一行一条（「用户: …」「air: …」），无论是已压缩的还是还没轮到压缩的

中间不再分第三档（「一档 100 字的摘要」）——理由写在 `build_window` 的注释里。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L6 短期记忆
#   上游    ：config、distill（提取就是 `distill_step1`）、scene（判寒暄与切分）、store
#   下游    ：chat / demo（每轮结束后调 `flush_if_needed`）
#   对外入口：`ShortTerm`（写 `append` / 判 `flush_if_needed` / 渲染 `build_window` ·
#             `build_window_blocks` / 收尾 `end_session` · `clear_digest` /
#             撤销 `undo_turns` / 回执 `take_closed_memo` / 回填 `read_state`）
#             + `estimate_tokens`
#   边界    ：**不自己写长期记忆**——它只决定"该提取了"，写是 `distill` 的事
#             （例外：`append` 里那次 `memo.judge_hits`，所以那处用了延迟 import）
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import re

from . import config as cfgmod
from .distill import distill_step1, write_skip_trace
from .prompts import rel_day, rel_stamp
from .scene import is_trivial, should_cut, should_cut_texts
from .store import atomic_write_json, now_str

Message = dict

_DIGEST_HEAD_RE = re.compile(r"^〔(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)")


def _relative_digest_head(chunk: str) -> str:
    """渲染时给 digest 批头补相对日（「（今天）」之类）；非批头原样返回。

    为什么不在存的时候写死（`_dated_digest`）：digest 是**持久化文本**
    （落进 `shortterm.json`，重启也不变）——存进去的「今天」明天就错了，
    同 `rel_day` 的注释（必须渲染时算）。批头只存绝对日期，每次注入现补。
    """
    m = _DIGEST_HEAD_RE.match(chunk or "")
    if not m:
        return chunk
    rel = rel_day(m.group(1))
    if not rel:
        return chunk
    i = m.start(1)
    return chunk[:i] + f"（{rel}）" + chunk[i:]


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（不引 tokenizer）。

    中文按 1 字 1 token、非中文按 4 字符 1 token 估。
    这个数只用来判「超没超预算」，不需要精确——预算本身就是个量级判断
    （4000 是"大约"，不是"正好"）。引一个 tokenizer 依赖来换小数点后的精度，
    不划算。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(0, other // 4)


class ShortTerm:
    """当前会话的上下文窗口（逐字尾部 + 压缩摘要）。

    窗口**落盘**（`data/shortterm.json`）：规格说压缩摘要「存在对象里、非 DB」，
    但没说可以不持久化。进程重启丢掉整段未提取的对话，等于丢记忆——
    「素材不丢」是项目第一原则，所以这里补一层落盘（原子写，见 store.atomic_write_json）。
    """

    def __init__(self, store, llm, emb_service=None, session_id: str = "",
                 state_path=None):
        """构造时就 `_load()`——窗口文件里还有内容，就说明"这一段没聊完"。

        `emb_service` 可选：没有它窗口照样管，切分退回字符重叠（`append`）。
        """
        self.store = store
        self.llm = llm
        self.emb = emb_service
        self.session_id = session_id or now_str()[:10]
        self.state_path = state_path or cfgmod.abspath(cfgmod.PATHS["shortterm"])

        self.messages: list[Message] = []     # 尚未提取的当前话题段
        # 已压缩的部分：**一批一段**（第一段是这批的日期范围 `〔…〕`，其后是若干行一句话）。
        # 存成「批的列表」而不是一整坨字符串，是为了能**整批整批地裁**——
        # 一行一句话没法单独丢（丢掉了就没日期了），按批丢才丢得干净。
        self.digest: list[str] = []
        self._emb_window: list[list[float]] = []   # 消息向量（判 should_cut）
        self._pending_cut = False             # append 时判出的话题切换
        self._last_active = ""                # 最后活动时刻（判会话空闲）
        self._budget_hit = False              # 本次触发是否来自预算
        self.last_closed_memo = None          # 上一条消息了结的 memo 编号（`take_closed_memo` 取走）
        self._load()

    # ---- 写入 ----

    def append(self, speaker: str, text: str, ts: str = "", persona: str = "",
               interrupted: bool = False) -> None:
        """追加一条消息（user / air **都要进**——记忆的对象是「这段互动」）。

        air 自己的话也进缓冲，因为 air 是这段关系的一半；
        但抽取时只留它的**实质性表态**（判据在 prompts 里，用「三个月后还有用吗」卡）。

        `interrupted`：这句**说到一半就被叫停**（2026-10-07，按「停止」那一路）。
        正文一个字都不动（那是她的原话），它管两件事：

          ① 界面回填时标一句「（已停止）」——落成字段而不是只标在屏幕上，
             是因为**刷新前后要一致**（屏幕上有标记、回填后没有，读的人就
             分不清"说完了"和"被叫停了"）；
          ② **提取时把它滤掉**（见 `compress_and_extract`）：窗口里留着，
             长期库里不留——理由写在那一步。

        `persona`：这句是**哪个人格**说的（air / mia / xina，2026-09-21）。
        记忆共享、人格是外壳——但"谁说的"不能抹平：切人格后窗口里全写 "air"，
        新人格会把上一个人格的话当成自己说过的。空 = 旧数据 / user 消息，
        渲染侧回退 "air"（`scene.render_conversation` 与 `gist_line` 都认这个字段）。
        """
        text = (text or "").strip()
        if not text:
            return
        msg = {"speaker": speaker or "user", "text": text, "ts": ts or now_str()}
        if persona:
            msg["persona"] = persona
        if interrupted:
            msg["interrupted"] = True
        self.messages.append(msg)
        self._last_active = msg["ts"]

        # 本条**命中**了哪几件未了结的事、各是什么动作（2026-10-05：命中判定，
        # 取代词表预筛 + `scan_closure`）——「面试过了」这类结果在这里被接住，
        # 「改到下周三」这类变更在这里写回（判不准就 `none`）。
        # 只对 **user** 判：air 说的话不该替用户了结他自己的事。
        # 没有未关闭的 memo 时这一步直接短路，不会产生 LLM 调用。
        if (speaker or "user") == "user":
            from .memo import judge_hits
            hits = judge_hits(self.store, text, self.llm, emb=self.emb)
            self.last_closed_memo = hits.get("closed") or None

        # 向量可用 → 用语义距离判切分
        vec = self.emb.embed_one(text) if self.emb is not None else None
        if vec:
            if self._emb_window:
                self._pending_cut = should_cut(vec, self._emb_window)
            self._emb_window.append(vec)
            # 滑动窗口要封顶：`cut.window` 是"看最近几条估分布"，不是"攒一整段"。
            # 不封顶它就随会话一直长——统计越来越钝（长窗口的 std 被老话题撑大，
            # 真正的转向反而突不出来），还白占内存。
            w = int(cfgmod.cfg("cut", "window", default=5))
            if len(self._emb_window) > w:
                del self._emb_window[:len(self._emb_window) - w]
        else:
            # 降级：**也要判**——不判的话自动提取只剩「攒够 60 条消息」这类兜底，
            # 实际就是「从来不自动提取」。字符重叠比余弦糙，但方向是对的。
            older = [m.get("text", "") for m in self.messages[:-1]][-8:]
            if len(older) >= 2:
                self._pending_cut = should_cut_texts(text, older)
        self._save()

    def take_closed_memo(self) -> str:
        """取走「上一条消息了结的那件 memo」的编号——**读后就清**（消费语义）。

        这个值是 `append` 的产出，给调用方（`chat` 把它塞进这一轮回执）用的，
        不是窗口自己的状态。读后必须清：不清的话下一轮没有新结果时它会被
        当成"这一轮又结了一件"再报一次（同 `pending_confirm` 那条——
        隔了几天回来的第一句话，不该被当成对它的点头）。
        """
        mid, self.last_closed_memo = self.last_closed_memo or "", None
        return mid

    # ---- 撤销（**只有人能发起**）----

    def undo_turns(self, turns: int = 1, expect: str = "") -> dict:
        """撤掉窗口**末尾的 N 轮**（N=1 就是"重说最后一句"）——改与重新生成都用它。

        一轮 = 他的一句 + 到下一句之前她说的（通常一条）。语义是「**回到那一句
        的地方重说**」，不是「改一条记录」：被撤掉的这些从没进过长期库
        （窗口就是还没提取的那一段），撤掉之后不会被提取、不会留下原文
        ——等于没说过。（不留痕也由此而来：没有记忆被改动，就没有要审计的东西。）

        `turns > 1` 是 2026-10-07 加的：**窗口里的任意一轮都能改**（不止最后一条）。
        界面上铅笔挂到窗口里每一条用户消息上——改第 2 轮，就等于把第 2 轮
        和它后面那些一起撤掉重说。**更早的（已经不在窗口里的）撤不了**：
        它们已经进长期库，撤销它不是改错字、是改历史；那时候能改的只有
        "理解"（「场景」页），原文动不了。

        `expect`：调用方**预期的第一句原文**（界面上那条消息的正文）。给了就核对，
        对不上**一个字都不撤**并如实回报——窗口在两次点击之间可能被提取过
        （话题切换 / 新对话 / 超预算都会触发），那时按 N 撤就会撤到**别的**一轮上。

        返回被撤掉的内容：调用方要拿 `text` 去界面就地换掉那句
        （**成功响应里带着旧文本**，好回答"到底撤了什么"）。
        """
        turns = max(1, int(turns or 1))
        if not self.messages:
            return {"ok": False, "detail": "窗口是空的——没有可重说的"}
        # 从后往前数 N 个「他说的」：每个 user 消息起一轮，撤到第 N 个为止
        i = None
        seen = 0
        for j in range(len(self.messages) - 1, -1, -1):
            if (self.messages[j].get("speaker") or "user") == "user":
                seen += 1
                if seen >= turns:
                    i = j
                    break
        if i is None:
            return {"ok": False,
                    "detail": f"窗口里只剩 {seen} 轮，撤不了 {turns} 轮"}
        user_msg = self.messages[i]
        if expect and (user_msg.get("text") or "") != expect:
            return {"ok": False, "changed": False,
                    "detail": "窗口和你那边看到的不一样了——刷新一下再改"}
        # 她对此的回话（第一条 air；被叫停的那条也算——它就是这段的产物）
        air_msg = next((m for m in self.messages[i + 1:]
                        if (m.get("speaker") or "user") != "user"), None)
        cut = self.messages[i:]
        del self.messages[i:]
        # 向量窗口跟着删（`_emb_window` 是**可选**的：没向量服务时它本来就是空的，
        # 所以按"删了几条"裁剪，而不是假装它们一一对应）
        if self._emb_window:
            del self._emb_window[-len(cut):]
        # 上一轮判出来的切换点跟着作废——这段要重说，不是要另起一段
        self._pending_cut = False
        self._save()
        return {"ok": True, "turns": turns, "ts": user_msg.get("ts") or "",
                "text": (user_msg.get("text") or ""),
                "air": (air_msg.get("text") or "") if air_msg else ""}

    # ---- 判定 ----

    def should_extract(self) -> bool:
        """四条触发，任一成立就该提取。"""
        if not self.messages:
            return False
        if self._pending_cut:
            return True
        if self.turns_no_cut() >= cfgmod.cfg("shortterm", "max_turns_no_cut", default=30):
            return True
        if self.window_tokens() > cfgmod.cfg("shortterm", "token_budget", default=4000):
            return True
        return self.session_idle()

    def turns_no_cut(self) -> int:
        """距上一次话题切换的轮数（一轮 ≈ 一条 user + 一条 air）。"""
        return len(self.messages) // 2

    def window_tokens(self) -> int:
        """窗口当前占多少 token（预算触发的输入）。

        **压缩区也算在内**——不算的话它可以无限长，而预算永远不触发。
        """
        return (estimate_tokens("\n".join(self.digest))
                + sum(estimate_tokens(m.get("text", "")) for m in self.messages))

    def session_idle(self) -> bool:
        """按空闲时长判「会话已结束」（默认 30 分钟无消息）。"""
        if not self._last_active:
            return False
        from datetime import datetime, timedelta
        try:
            last = datetime.strptime(self._last_active, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return False
        return datetime.now() - last > timedelta(
            minutes=cfgmod.cfg("shortterm", "session_idle_min", default=30))

    def end_session(self) -> None:
        """显式结束会话（触发提取最后一段，不然尾巴丢了）。"""
        self._pending_cut = True

    def clear_digest(self) -> None:
        """把压缩摘要也清掉（「新对话」用）。

        摘要代表的内容在提取时**已经进长期库**（原文 + 摘要一起写），清它不丢数据——
        需要时靠回忆（recall）把相关场景拉回来。**只有「新对话」清**
        （`close(clear_digest=True)`；收尾 `end()` 不传它）：收尾是同一段对话的延续，
        摘要还要当「更早的对话」的背景用。
        """
        if not self.digest:
            return
        self.digest = []
        self._save()

    # ---- 渲染 ----

    def build_window(self) -> str:
        """渲染要注入的短期记忆：**最近 N 条逐字 + 更早的一行一条**。

        分层保真的理由：LLM 需要确切知道用户刚说了什么（逐字），
        但更早的内容只提供背景即可（压缩）——全部逐字会撑爆预算，
        全部压缩又会丢掉最近对话的语气和细节。

        **只有两档，中间不再插"100 字的摘要"那一档**（想过，放弃了）：

        1. **落差在「逐字 ↔ 摘要」这一步，不在「100 字 ↔ 一句话」之间。**
           一旦进了摘要，模型对它的用法是同一套（"我们聊过 X"这个背景），
           多给 80 个字不会改变她的行为，却要为此多写一份格式约束、
           多一份输出、多一处可能写歪的地方。
        2. **均匀的材料才能被再压一次。** 一行一条的东西，攒多了可以一次 LLM
           把 N 行并成一节（就像 S1→S2）；三档结构做不到——粗细混在一块儿，
           没有统一的挤压规则，只会越攒越参差。
        3. **顺带补了一个真漏洞**：以前窗口以外（`messages[-keep:]` 之外）的消息
           是**既不逐字也不摘要，直接从渲染里消失**的，要等下一次提取才回来。
           现在它们一律降成一行——窗口里不再有暗洞。**降级变糙可以，变哑不行**
           （这条在切场景那儿就用过一次，这里是同一个道理的第二次出现）。
        """
        return "\n\n".join(text for _, text in self.build_window_blocks())

    def build_window_blocks(self) -> list[tuple[str, str]]:
        """窗口拆成**可分别丢弃的两块**：`[("更早", text), ("最近", text)]`。

        顺序 = **先裁的排前面**：总量不够时（见 `chat.fit_context`）
        先丢「更早」那一块（压缩成一行一条的背景），「最近」那几句逐字最后丢
        ——丢背景可以，丢"他刚说的原话"不行，那是保真的下限。

        为什么要拆开而不是给一整串：**总量不够时只能丢一部分**。
        整串要么全留要么全丢，等于把「刚才发生了什么」也一起丢了，
        而它恰恰是最不该丢的那一块。
        """
        keep = int(cfgmod.cfg("shortterm", "verbatim_messages", default=4))
        recent = self.messages[-keep:] if keep > 0 else []
        older = self.messages[:-keep] if keep > 0 else list(self.messages)

        # digest 是存储文本（相对日不能在存的时候写死）——注入前现补批头标签
        lines = ([_relative_digest_head(c) for c in self.digest]
                 + [self.gist_line(m) for m in older])
        # 硬上限：**不管攒了多少，注入的就这么多行**，超了从最早的开始丢。
        # 被丢的那部分不是消失——它们还在 `messages` 里，下一次提取照样逐字进长期库；
        # 这里只是不占用这一轮的注意力预算。
        cap = int(cfgmod.cfg("shortterm", "older_line_cap", default=40))
        if cap > 0 and len(lines) > cap:
            lines = lines[-cap:]
        blocks: list[tuple[str, str]] = []
        if lines:
            # ⚠️ 说清"这是转述"（2026-09-21，设计稿 A 条）：她曾把摘要的措辞当成
            # 自己的原话复述（"手里还压着个更凉的推论" vs 原文"我有预感…"）——
            # 要引用原话，工具是 `memory_search`（带 `raw=true` 下钻；2026-09-24：
            # `memory_read` 已并入它，见工具箱稿 §3.1）。
            blocks.append(("更早", "[更早的对话（一行一条，**是转述不是原话**；"
                                   "要引用原话先用 memory_search 带 raw 读）]\n"
                          + "\n".join(lines)))
        if recent:
            from .scene import render_conversation
            # `with_date`：窗口里可能攒着昨天的消息（提取没触发时会），
            # 不标日期的话她会把整段当成刚说的、或反过来把刚说的当成昨天的。
            blocks.append(("最近", "[最近对话]\n"
                           + render_conversation(recent, with_date=True)))
        return blocks

    def gist_line(self, msg: Message) -> str:
        """一条消息降成一行：**由代码写，不叫模型写**。

        它服务对象不是拿去长期记忆，只是窗口里"这段还没轮到压缩"的那些消息
        ——还没有 gists、但也不该从渲染里凭空消失。所以这里用**截断**：
        确定的、可预测的、零成本。真要精炼的那一步在压缩时由 LLM 做
        （见 prompts 里 `window_digest` 的格式要求），两件事分开。

        截断必须留标记（`…`）：不带的话，她会把截了一半的句子当成完整的。
        """
        # 署名用**当时的真名**（persona；旧的/缺的回退 "air"）——同
        # `scene.render_conversation`：切人格后"谁说的"不能抹平（2026-09-21）。
        who = ("用户" if (msg.get("speaker") or "user") == "user"
               else (msg.get("persona") or "air"))
        text = " ".join((msg.get("text") or "").split())
        limit = int(cfgmod.cfg("shortterm", "gist_line_chars", default=60))
        clipped = text if len(text) <= limit else text[:limit].rstrip() + "…"
        # 「（今天）」也标：降进「更早」栏之后，位置不再说明"这是刚才说的"，
        # 不给标签它就自己推（推错的样子见 `_dated_digest` 的注释）。
        tag = rel_stamp((msg.get("ts") or "")[:16])
        return f"{who}{tag}: {clipped}".strip()

    # ---- 压缩 = 提取 ----

    def compress_and_extract(self) -> dict | None:
        """窗口超限 / 话题切换 / 会话结束时调用：一次 LLM 调用，产出两侧。

        返回值是给调用方记录用的摘要 dict（trace / demo 会打印），
        没有可提取的内容时返回 None（**空窗口不该白调一次 LLM**）。
        """
        if not self.messages:
            return None

        budget_only = (self._budget_hit
                       and not self._pending_cut
                       and self.turns_no_cut() < cfgmod.cfg("shortterm", "max_turns_no_cut", default=30)
                       and not self.session_idle())
        keep = int(cfgmod.cfg("shortterm", "verbatim_messages", default=4))
        if budget_only and len(self.messages) > keep:
            # 预算触发：对话还在继续，只压最老的一批，尾部逐字留着
            oldest, rest = self.messages[:-keep], self.messages[-keep:]
        else:
            oldest, rest = self.messages, []

        # 纯寒暄 / 纯应答**代码直接丢**，连 LLM 都不调：
        # 省一次调用，也省得它把「在吗」存成一张场景卡。摘要也不留——
        # 这几句本来就不值得记（"在吗"不需要被记住）。
        if is_trivial(oldest):
            print("[shortterm] 纯寒暄 / 纯应答，直接丢弃（不调 LLM）")
            # **代码判的跳过也要留痕**（与模型判共用 `write_skip_trace`）：
            # 这条路径连 LLM 都没调，trace 是它唯一的痕迹——不留的话，
            # 「我那句怎么没记住」在这里永远答不出来。
            write_skip_trace(oldest, "纯寒暄 / 纯应答（代码判定，没花 LLM）")
            self.messages = rest
            self._emb_window = self._emb_window[-len(rest):] if rest else []
            self._pending_cut = False
            self._budget_hit = False
            self._save()
            return {"scene_id": "", "skipped": True,
                    "reason": "纯寒暄 / 纯应答（代码判定，没花 LLM）",
                    "title": "（这段没进长期库：只是寒暄）", "digest_chars": 0}

        # 「说到一半被叫停」的那半句（`interrupted`）：**窗口里留着，长期库里不留**
        # （2026-10-07 定）。短期留着有用——刷新不丢、能重新生成、下一轮她知道
        # 自己说到一半；但长期记忆里留一句被掐断的话没有信息量，还会让她以后
        # 以为那句是说完了的。所以**提取这一步把它滤掉**：不进原文、不进场景卡，
        # 摘要也一起用滤过的那份（口径只有这一处，别在别处再滤一遍）。
        keepable = [m for m in oldest if not m.get("interrupted")]

        if not keepable:
            # 一段里只有被叫停的半句（理论上到不了：一段总有他那句）。
            # 真到了就别白调一次模型——清窗口、不落库、留一行痕（同寒暄那条路）。
            write_skip_trace(oldest, "只有被叫停的半句（代码判定，没花 LLM）")
            self.messages = rest
            self._emb_window = self._emb_window[-len(rest):] if rest else []
            self._pending_cut = False
            self._budget_hit = False
            self._save()
            return {"scene_id": "", "skipped": True,
                    "reason": "只有被叫停的半句（代码判定，没花 LLM）",
                    "title": "（这段没进长期库：只有被叫停的半句）", "digest_chars": 0}

        scene, digest, entities = distill_step1(
            self.store, keepable, self.llm, emb_service=self.emb, source=self.session_id)

        # 压缩摘要接在前面：更早的摘要 + 这次的摘要，构成"被压掉的全部内容"。
        # ⚠️ **即使这段被判成"不值得存"（scene is None），摘要也照留、窗口也照清**——
        # 跳过只该意味着「不进长期库」，不该意味着「当没发生」。
        # 不留摘要的话，那几句会在她的短期记忆里凭空消失，下一轮像是没聊过。
        # （**唯一的例外**就是上面滤掉的那半句：它连摘要一起不进——那是"半句
        #   本来就没有信息量"这一条单独的取舍，不是这条规则松了口。）
        chunk = self._dated_digest(digest, keepable)
        if chunk:
            self.digest.append(chunk)
            self._trim_digest()
        self.messages = rest
        self._emb_window = self._emb_window[-len(rest):] if rest else []
        self._pending_cut = False
        self._budget_hit = False
        self._save()

        if scene is None:
            return {"scene_id": "", "skipped": True,
                    "title": "（这段没进长期库：没有实质内容）",
                    "digest_chars": len(digest)}
        return {"scene_id": scene.id, "title": scene.title, "topic": scene.topic,
                "intensity": round(scene.intensity, 3), "valence": scene.valence,
                "arousal": scene.arousal, "trigger_class": scene.trigger_class,
                "open_loops": len(scene.open_loops), "entities": [e["name"] for e in entities],
                "digest_chars": len(digest)}

    def _dated_digest(self, digest: str, messages: list[Message]) -> str:
        """把模型给的摘要整理成**一批一行条**的文本块——**格式由代码定**。

        三件事都由代码做，各有原因：

        1. **加日期**（原本就在做的事；「（一天前）」不在这里写——它是渲染时
           现补的，见 `_relative_digest_head`）：模型爱把时间写成
           「凌晨 00:12」「凌晨三点半」——**只有时刻、没有日期**（prompt 里那句
           「提到时间必须带日期」管不住每一行，见 `SCENE_SCHEMA.window_digest`）。
           这段摘要随后被塞进 `[更早的对话]` 那一栏，**位置本身就在暗示"这是更早的事"**，
           于是模型手里只有一个没日期的「凌晨三点半」+ 提示词末尾的当前时间，
           它算不出那是多久以前，只能往后推——
           把 17 分钟前说成「昨晚三点半」、3 小时前说成「再前一晚刚过午夜」。

           **批头这一行的日期由代码给**（取 `messages` 首末 `ts`）：日期是这段消息里的
           现成数据，同一类坑（topic 措辞、open_loops 键名）这项目踩过三次，
           结论都是最后得由代码兜底。

        2. **按行切开**（这次新加）：模型会把几行写成一整段（也可能恰好写成一行），
           这里统一按行整理——进压缩区的就是「一行一条」，不多不少。

        3. **行长由代码截**（这次新加）：`gist_line_chars` 是硬截，**不求模型守**。
           模型不会数字数（这个坑本项目踩过：`max_tokens`、说话方式的档位取值，
           都是最后改成"描述给模型、上限由代码管"才稳）。
        """
        lines = [ln.strip() for ln in (digest or "").splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            return ""
        limit = int(cfgmod.cfg("shortterm", "gist_line_chars", default=60))
        lines = [ln if len(ln) <= limit else ln[:limit].rstrip() + "…" for ln in lines]

        stamps = sorted(m.get("ts") for m in messages or [] if m.get("ts"))
        if not stamps:
            return "\n".join(lines)
        lo, hi = stamps[0], stamps[-1]
        # 这里**只存绝对日期**：相对日是渲染时现补的（见 `_relative_digest_head`）
        tag = (f"{lo[:10]} {lo[11:16]}–{hi[11:16]}" if lo[:10] == hi[:10]
               else f"{lo[:10]} {lo[11:16]} → {hi[:10]} {hi[11:16]}")
        # 日期单独占一行：**一批一个头**，而不是每行都挂一个（40 行的头能顶半屏）。
        return f"〔{tag}〕\n" + "\n".join(lines)

    def _trim_digest(self) -> None:
        """压缩区超 `older_line_cap` 行就丢最早的**整批**。

        渲染时已经裁过一次了，为什么还要裁存储里的这一份：
        渲染裁的是「这一轮喂进去多少」，存储这一份还在参与 `window_tokens`
        （它是预算触发的输入）。不裁的话它会一直长。
        """
        cap = int(cfgmod.cfg("shortterm", "older_line_cap", default=40))
        if cap <= 0:
            return
        total = sum(len(c.splitlines()) for c in self.digest)
        while self.digest and total > cap:
            total -= len(self.digest.pop(0).splitlines())

    def flush_if_needed(self) -> dict | None:
        """先判预算（设置 `_budget_hit` 供 compress 决定压多少），再决定提不提取。"""
        if not self.messages:
            return None
        if (self.window_tokens() > cfgmod.cfg("shortterm", "token_budget", default=4000)
                and not self._pending_cut):
            self._budget_hit = True
        if not self.should_extract():
            return None
        return self.compress_and_extract()

    # ---- 落盘 ----

    def _save(self) -> None:
        try:
            atomic_write_json(self.state_path, {
                "session_id": self.session_id,
                "digest": self.digest,
                "messages": self.messages,
                "last_active": self._last_active,
            })
        except Exception as e:
            # 落盘失败不阻塞对话（内存里还在，本轮照常）：这是补强，不是主链路
            print(f"[shortterm] 落盘失败（不影响本轮）: {e}")

    @staticmethod
    def read_state(path) -> dict:
        """只读窗口文件，规整成 `{"session_id", "messages", "digest", "last_active"}`。

        **一个出口，两个用途**：启动恢复（`_load`）和界面回填历史
        （`dashboard.App.history`）——分头解析迟早漂移（老格式的转换就是个坑）。
        文件不存在 / 读失败 / 坏格式一律给空结构，**不抛**：
        一个坏文件不该让界面或启动卡住（数据没丢，只是这次没读到；
        `_save` 走的是原子写，正常路径不会写出坏文件）。
        """
        out = {"session_id": "", "messages": [], "digest": [], "last_active": ""}
        try:
            if not path.exists():
                return out
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return out
        out["messages"] = raw.get("messages") or []
        # 老文件里 digest 是一整坨字符串（那时一行 = 一批），转过来接着用。
        # 空字符串（而不是空列表）要还原成列表，否则后面 append 会炸。
        old = raw.get("digest") or ""
        if isinstance(old, str):
            out["digest"] = [old] if old.strip() else []
        else:
            out["digest"] = [d for d in old if str(d).strip()]
        out["session_id"] = raw.get("session_id") or ""
        out["last_active"] = raw.get("last_active") or ""
        return out

    def _load(self) -> None:
        raw = self.read_state(self.state_path)
        if not (raw["messages"] or raw["digest"] or raw["last_active"]):
            return          # 没有可恢复的东西（文件不存在 / 读不出来）
        self.messages = raw["messages"]
        self.digest = raw["digest"]
        self._last_active = raw["last_active"]
        if raw["session_id"]:
            self.session_id = raw["session_id"]
        # 向量不落盘（可重算），重启后第一轮不判切分——
        # 保守方向：不切（这段会在下次触发时整体提取，不会丢）
