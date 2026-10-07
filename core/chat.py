"""对话层：把记忆层接进一次真实的对话。

这是**唯一**一个同时碰「读」和「写」的模块，所以顺序很讲究：

    用户说了一句话
      → ① 唤醒（用这句话去翻记忆）        ← 读
      → ② 拼上下文（尊重 + 人格 + 记忆 + 窗口 + 当前消息）
      → ③ 生成回复
      → ④ 写入（用户这句 + air 的回话）    ← 写
      → ⑤ 记账（提过的备忘转成「已提」；被说到的场景记一次提及）

**为什么唤醒在写入之前**：唤醒要用「当前消息」去查，
如果先把消息写进窗口，窗口里就有了它——既会重复注入，
也会让 `compute_cues` 拿到的历史里混进"未来"。所以顺序不能换。

**为什么 air 的回话也要写进窗口**：记忆的对象是「这段互动」，不是「用户」。
air 是这段关系的一半，它说过的话（尤其承诺和立场）是下一轮理解的前提
（设计稿：双主体）。

这个模块**不做**判断——它只编排：唤醒是 `recall` 的事，写入是 `shortterm` 的事。
对话层该有的分寸写在**人格文件**里（`self/personas/*.md`——2026-09-19 起纪律退役，见 prompts.py），
不是写在这里的 if-else 里。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L10 对话层
#   上游    ：recall（读）、shortterm（写）、tools（她的手）、weave（确认后的动作）、
#             prompts / model / memo / llm / embedding / persona / store / config
#   下游    ：dashboard（应用侧唯一调用方）、run_experiment（实验回放）
#   对外入口：`ChatSession`（`reply` / `reply_stream` / `confirm` + `pending_view` /
#             `close` / `window_preview`）+ `build_embedding` / `load_persona` / `persona_names`
#   边界    ：只编排，不判断——该不该翻是 recall 的事，该不该存是 distill 的事
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import re

from . import config as cfgmod
from . import persona as persona_file
from .embedding import EmbeddingService
from .llm import LLM
from .memo import mark_raised
from .model import lang_from_prefs
from .prompts import build_system_prompt
from .recall import mark_mentioned, recall_for_message
from .shortterm import ShortTerm, estimate_tokens
from .store import Store, layer_edit
from .tools import execute as run_tool, openai_tools, tool_names
from .weave import (archive_by_layer, delete_by_layer, update_by_layer)


def build_embedding() -> EmbeddingService | None:
    """按配置建向量服务；没配就返回 None（上层据此走字符重叠降级）。

    `insecure_ssl` 一路带上：自签 / 内网端点要靠它才连得上，
    而它**只在 embedding 这条路上生效**（llm 走的一直是默认校验）。
    """
    conf = cfgmod.cfg("embedding", default={}) or {}
    svc = EmbeddingService(endpoint=conf.get("endpoint", ""),
                           api_key=conf.get("api_key", ""),
                           model=conf.get("model", "text-embedding-3-small"),
                           insecure_ssl=bool(conf.get("insecure_ssl", False)))
    return svc if svc.available else None


def load_persona(name: str = "") -> str:
    """当前人格的提示词——**每轮现读**：切了 / 改完，下一句就换。

    2026-10-05 起只是 `core.persona.load` 的**转发**（读 / 写 / 停用全在那边——
    设置页能新建人格之后，"人格文件长什么样"不该有两处知道）。
    留着这个壳是因为调用点还在：`_prepare` 调 `load_persona`，
    dashboard 调同组的 `persona_names`；且"对话层从哪拿人格"是它该回答的问题。
    """
    return persona_file.load(name)


def persona_names() -> list[str]:
    """`self/personas/` 里有哪些人格——界面的选项列表用（扫目录，加文件即生效）。"""
    return persona_file.names()


# 他这句话算不算点头——**按工具箱设计稿 §六 那张表判**。
# 用规则不用模型：那张表是设计定死的，判据就该可预测、可测；
# 交给模型反而会漂——它会替他做主，把一句"嗯"当成同意。
_DELETE_WORDS = ("删", "去掉", "不要了", "清掉", "删除", "remove", "delete")
# 归档 = 确认条三选一的中间档（2026-09-23，工具箱稿 §3.4 / §六）：
# "别删，但也别再让它影响你"这类话认它；**不说这层意思就不认**（宁可再问一次）。
_ARCHIVE_WORDS = ("归档", "冷存", "冷层", "收起来", "先收着", "别让它影响")
# 改动：**只认"让改"的表达**（2026-09-23 收紧）——"不对""错了"这类是评价词，
# 在"跟她讨论"里太常见（"我觉得这说法不太对"），而改动的执行会直接动数据：
# 讨论 ≠ 指令。含糊的评价 → vague（再问一次），他要改会说"改/不是这样/应该是…"。
_REVISE_WORDS = ("改", "不是这样", "应该是", "其实是", "记错了", "纠正")
_VAGUE_WORDS = ("嗯", "对", "好", "行", "ok", "可以", "是的", "行吧", "随便", "你定")


def _hit(low: str, words: tuple) -> bool:
    """词表命中且**未被否定**——"别删"不是"删"，"不是删，是归档"该判归档。

    为什么要有这道（2026-09-23 检查发现）：删除不可恢复，而"别删，归档吧"
    里含"删"字——按裸关键词判会**真删**，那是这套判定里最贵的一类误判
    （归档判错可逆、删除判错不可逆——否定必须能压住删除）。

    否定只往左看**两个字符**（"别删"/"不要删"/"不是删"都够）：窗口再大就会
    把隔了标点的否定带过来——"别删，归档吧"里的"归档"前两字是"删，"
    （不是否定），**该判归档**。
    """
    for w in words:
        i = low.find(w)
        while i >= 0:
            if not any(n in low[max(0, i - 2):i] for n in ("别", "不")):
                return True
            i = low.find(w, i + 1)
    return False


def _judge_confirmation(text: str) -> tuple[str, str]:
    """返回 `(verdict, 他说的新内容)`。

    `delete` 明确让删 / `archive` 明确要归档（中间档）/ `revise` 明确让改 /
    `vague` 含糊（要再问）/ `drop` 岔开（作废）。

    **含糊不算数是刻意的**——删除和改动都不该由一句语气词触发；
    **疑问句也不算数**（2026-09-23 加）："删了吧？""这个不对吧？"是他在问、
    不是他在说——动数据（改）和不可逆（删）的动作，宁可再问一次。
    """
    t = (text or "").strip()
    if not t:
        return "drop", ""
    low = t.lower()
    if t.endswith(("？", "?")) or low.endswith(("吗", "呢")):
        return "vague", ""
    if _hit(low, _DELETE_WORDS):
        return "delete", ""
    if _hit(low, _ARCHIVE_WORDS):
        return "archive", ""
    if _hit(low, _REVISE_WORDS):
        # 他自己说了"是……"就用他说的；没说具体的，用她提议的那个
        m = re.search(r"(?:改成|改为|应该是|其实是|不是.{0,8}是)\s*(.+)$", t)
        return "revise", (m.group(1).strip() if m else "")
    if low in _VAGUE_WORDS or len(t) <= 3:
        return "vague", ""
    return "drop", ""


def _no_reply_hint(kind: str) -> str:
    """模型没回内容时说什么——**按失败的种类说**。

    原来统一是「检查设置里的 LLM 配置」，而真实原因常常是"这一轮太长、想太久"，
    那句话会把人引到完全错的方向（配置好好的，去检查它纯属浪费时间）。
    """
    return {
        "timeout": "（这一轮 air 想得太久没赶上——内容大概太长了。删掉一部分再发一次？）",
        "too_long": "（这一轮超出模型能接的长度了——把文件截短一点，或者分几次发。）",
        "rate": "（模型那边现在有点挤——等一下再发。）",
        "auth": "（模型的 key 或地址不对，去「设置」里测一下连通性。）",
        "network": "（连不上模型服务——看看网络，或者去「设置」里测一下。）",
        # 想得比额度还长：这不是失败，是**她还没说完就被掐了**——说清楚，
        # 否则界面上只看到"没有内容"，会往配置上查半天
        "truncated": "（air 想得比额度还长，正文被截断了——"
                     "把 `chat.max_tokens` 再调大，或者把问题拆小一点。）",
    }.get(kind, "（模型没有返回内容——看看设置里的 LLM 配置，或者稍后再试）")


# （三层分派 `delete_by_layer` / `archive_by_layer` / `unarchive_by_layer` /
#   `update_by_layer`——原在这里——2026-09-24 搬去 `weave`：那是"人的动作"的出口，
#   对话确认条与台账页按钮**两条入口共用**。本文件只 import 使用。）


# ---------------------------------------------------------------------
# 上下文总预算的仲裁
#
# 分块上限（`recall.inject_n` / `inject_summaries_n` / `shortterm.older_line_cap`）
# 各管各的，**加起来没有上限**：库一大，注入总量就随它一起长，
# 长到某个程度就不再是「记得多」，是**稀释注意力**——真正要紧的那条被淹没。
# `context.total_budget` 因此必须由这里读（以前它写在配置里、没有任何代码读它，
# 摆在那儿像是生效的：配置在说谎）。
#
# 裁剪顺序来自设计稿（短期记忆稿第六节）：
#   当前消息 > 短期记忆（窗口） > 召回的记忆
#   召回内部：原文 > 摘要 > 画像 > 场景（**越靠后越先被裁**）
# 两个设计稿没点名的，理由写在 `_TRIM_ORDER` 下面。
# ---------------------------------------------------------------------

_TRIM_ORDER = (
    ("scenes", "场景"),
    ("profiles", "画像"),
    ("summaries", "摘要"),
    # 「备忘录」插在摘要与原文之间：它一行一条（代价最小）。
    # 2026-10-05 晚：键从 `memo_candidates`（候选，已删）改成 `standing_memos`——
    # 那一栏里**到点的那件排在第一个**，而这里从尾部摘，所以它最后才被裁（最该留）。
    ("standing_memos", "备忘录"),
    ("raws", "原话"),
)


def _item_tag(item) -> str:
    """给被裁掉的那一条一个**认得出来的名字**（报告里要看得见是谁被丢了）。"""
    sid = getattr(item, "id", "") or getattr(item, "scene_id", "") or ""
    if sid:
        return sid
    text = getattr(item, "statement", "") or getattr(item, "text", "") or ""
    if not text and isinstance(item, dict):
        text = item.get("content", "") or ""
    return (text or "")[:12]


def _drop_one(recall: dict, window_blocks: list) -> str:
    """裁掉当前「最不重要的一格」，返回一句说明；**没得裁返回空串**。

    列表内部已经按重要度排过序（`recall` 那边排的），所以从**尾部**摘。
    """
    for key, label in _TRIM_ORDER:
        items = recall.get(key) or []
        if items:
            return f"{label} {_item_tag(items.pop())}"
    if window_blocks:
        return f"窗口「{window_blocks.pop(0)[0]}」"
    return ""


def fit_context(recall: dict, window_blocks: list, measure,
                budget: int | None = None) -> dict:
    """把这一轮的注入量压进 `context.total_budget`（**就地**改前两个参数）。

    `measure()` 由调用方给（只有它知道系统提示词怎么拼），返回当前的 token 估算。
    为什么要回调而不是传一串文本：裁掉一格之后要**重新量一次**——
    各块的长度不是简单相加（分栏标题、是否整块消失都会变），
    拿"裁之前的总量"去减"裁掉的那一格"是算不准的。

    `budget <= 0` = **不仲裁**。留这个口子是刻意的：这个数还没标定过
    （见短期记忆设计稿「待确认」第 5 条），写死又不能关，
    等于把一个没标定的阈值变成了硬约束。
    """
    budget = int(budget if budget is not None
                 else cfgmod.cfg("context", "total_budget", default=0) or 0)
    if budget <= 0:
        return {"budget": 0, "before": 0, "after": 0, "dropped": [], "alone": False}

    before = measure()
    total = before
    dropped: list[str] = []
    while total > budget:
        what = _drop_one(recall, window_blocks)
        if not what:
            # 全裁光了还超 = **当前消息自己就超了**（拖进来一篇两万字的设计稿）。
            # 它不可裁——那就如实报上去，让 `too_long` 那条失败分类去说。
            print(f"[chat] 上下文 {total} > {budget}：裁到只剩当前消息仍然超，"
                  f"不动它（当前消息不可裁）")
            return {"budget": budget, "before": before, "after": total,
                    "dropped": dropped, "alone": True}
        dropped.append(what)
        total = measure()
    if dropped:
        print(f"[chat] 上下文 {before} > {budget}，按优先级裁掉："
              + "、".join(dropped))
    return {"budget": budget, "before": before, "after": total,
            "dropped": dropped, "alone": False}


# 回执（他点过确认条之后的结果）在她上下文里留几轮（2026-09-25 定）。
#
# **两三轮就够，不该一直挂着**：它的用处只是"隔一两轮再问起时她答得上来"，
# 再久就是白占上下文——而真相在库里，她要查证自己 `memory_search` 就行。
# 不做成配置项：这种"留几轮"的旋钮没人会调，写死反而不会被误改。
_RECEIPT_TURNS = 3


class ChatSession:
    """一次会话。持有 store / llm / embedding / 短期窗口。"""

    def __init__(self, store: Store | None = None, llm: LLM | None = None,
                 emb: EmbeddingService | None = None, session_id: str = "",
                 state_path=None):
        """一次会话的全部状态（三个必填件没给就自己造）。

        `state_path` 可换：实验回放要给每段对话独立窗口，否则串起来就测不出
        「跨会话还记得吗」。
        """
        self.store = store or Store()
        self.llm = llm or LLM()
        self.emb = emb if emb is not None else build_embedding()
        # 人格**不存成实例属性**：`_prepare` 每轮按偏好现读（他在界面上切了
        # 版本，下一句就换人）——存下来反而让人以为它跟着会话走。
        self.st = ShortTerm(self.store, self.llm, emb_service=self.emb,
                            session_id=session_id, state_path=state_path)
        # 上一轮她动过什么（**给她自己看的一行**，不进记忆）。
        # 见 `_actions_note()`：工具结果是那一轮的临时消息，下一轮就没了。
        self.last_actions: list[str] = []
        # 他点过确认条之后的**回执**（`_note_confirm`），按轮衰减（留 `_RECEIPT_TURNS` 轮）。
        # 与 `last_actions` 分开存：那是每轮覆盖的，这个要跨轮留几轮才有用。
        self.receipts: list[tuple[str, int]] = []
        # 跨轮的「看过的场景」（2026-09-21，设计稿 E 条）：同一批结果再被查出来
        # 就是"没有新增"→ 停手（`tools._scene_search` 里判）。
        # 挂在会话上（跨轮有效），重启即清——无损。
        self._seen_scenes: set[str] = set()
        # 待确认（工具箱 3.6：**确认通道就是对话本身**）。
        # 它只活在会话里——会话一结束就丢，**不留一个隔夜的待执行删除**。
        self.pending_confirm: dict | None = None

    def lang(self) -> str:
        """语言（中 / 英，他自己定的）。每轮都读：切完下一句就生效。"""
        return lang_from_prefs(self.store.all_prefs())

    # ---- 主流程 ----

    def reply(self, user_msg: str) -> dict:
        """收一句话，回一段，并把这一轮的记忆动作一并返回（给仪表盘用）。

        返回的 `recall` / `written` 是**给人和实验看的**，不参与生成——
        但它们是这个项目的重点：没有它们，就看不出 air 到底是「记得」还是「碰巧说对」。
        """
        user_msg = (user_msg or "").strip()
        if not user_msg:
            return {"reply": "", "recall": {}, "written": None, "system": ""}

        # ① 唤醒 + ② 拼上下文（**与流式共用同一个 `_prepare`**：
        # 两套上下文迟早会长得不一样，而那是最难查的一类 bug）
        # 上一句问出口的事：**他这句是不是在回答它**（必须在唤醒之前判）
        note = self._resolve_pending(user_msg)
        acts = self._actions_note()          # 上一轮她动过什么（给她自己看）
        if acts:
            note = f"{note} {acts}".strip()
        recall, system, messages, max_tokens = self._prepare(user_msg, note)

        # ③ 生成
        reply, tool_notes, err = self._generate(messages, max_tokens)
        # 失败提示走 `hint` 字段（界面当**系统消息**显示），**不冒充她的正文**：
        # 它是给用户看的告示，不是她说过的话。塞进 reply 的后果很具体——
        # 下一轮她会"记得自己说过「检查 LLM 配置」"，提取时这句还会被当成对话内容。
        # （`ShortTerm.append` 会跳过空串，所以失败时窗口里不会留下一条空的 air。）
        hint = _no_reply_hint(err) if not reply.strip() else ""

        # ④ 写入（用户这句 + air 的回话）
        # `persona`：**记下这句是谁说的**（2026-09-21）——记忆共享、人格是外壳，
        # 但署名不能抹平（切到 mia 后窗口里全 "air"，她会把别人的话当成自己的）。
        # 与 `_prepare` 读人格同一来源：同一轮内不会变。
        self.st.append("user", user_msg)
        self.st.append("air", reply, persona=self.store.get_pref("persona") or "air")
        written = self.st.flush_if_needed()

        # ⑤ 说完之后的记账（**放在生成之后**：它不该影响本轮说了什么）
        memo_raised = self._mark_memos_raised(recall)
        memo_closed = self._closed_memo_view()
        mentioned = mark_mentioned(self.store, recall.get("scenes") or [],
                                   [user_msg, reply])
        self._remember_actions(tool_notes)

        return {"reply": reply, "recall": recall, "written": written, "system": system,
                "hint": hint,
                "memo_raised": memo_raised, "memo_closed": memo_closed,
                "mentioned": mentioned,
                "tools": tool_notes,
                # 她在等什么（界面拿它弹那条确认）
                "confirm": self.pending_view()}

    def _generate(self, messages: list[dict], max_tokens: int) -> tuple[str, list[dict]]:
        """生成回复；她调了工具就执行、把结果回给她、再问一次（**最多两轮**）。

        轮数上限：参考的 Harness 不设上限，是因为它是代码 agent、要跑长循环；
        air 是聊天，一轮调一两个动作就够了——多一轮就是多一次等待。
        工具本来就该是"顺手做件事"，不是主线。

        **工具结果只回给她看**，不进用户的对话：用户看到的是她的话，
        不是"她调用了哪个工具"（那个在仪表盘这一轮里显示）。
        """
        tools = None
        if cfgmod.cfg("chat", "tools_enabled", default=True):
            tools = openai_tools(self._tool_names())
        temperature = float(cfgmod.cfg("chat", "temperature", default=0.7))
        # 超时给得宽（10 分钟）：**复杂问题和字数无关**，一句很短的提问她也可能要想很久。
        # 这里的超时只用来兜底"卡死了"，不是限制她想多久。
        timeout = float(cfgmod.cfg("chat", "timeout_reply", default=600))
        out = self.llm.chat_with_tools(messages, tools=tools,
                                       max_tokens=max_tokens, temperature=temperature,
                                       timeout=timeout)
        notes: list[dict] = []
        # ⚠️ **一份 ctx 用整轮**：跨调用状态住在它里面（只读计数 `read_used` /
        # `drill_used`、查过的词 `recall_queries`、看过的场景 `seen_scenes`）——
        # 每次调用都新建的话，判据闸与提醒等于没有（测试抓出来的：她一轮连查三次）。
        # 搜索要调 LLM（走 `/responses`），所以 llm 也在 ctx 里。
        # `confirm_channel`：**确认通道就是对话本身**（工具箱 3.6）——
        # 需确认档的工具因此能放行到"提议"，等他下一句话，而不是直接拒绝。
        ctx = {"emb": self.emb, "llm": self.llm,
               "confirm_channel": True, "seen_scenes": self._seen_scenes}
        rounds = 0
        while tools and out.get("tool_calls") and rounds < 2:
            rounds += 1
            calls = out["tool_calls"]
            # 把"她调用了什么"拼回消息（OpenAI 的工具回合格式）：
            # assistant 说它要调什么，tool 逐个给结果，然后才有下一轮
            messages = messages + [{
                "role": "assistant",
                "content": out.get("content") or "",
                "tool_calls": [
                    {"id": c.get("id") or f"call_{i}", "type": "function",
                     "function": {"name": c.get("name") or "",
                                  "arguments": json.dumps(c.get("arguments") or {},
                                                          ensure_ascii=False)}}
                    for i, c in enumerate(calls)],
            }]
            for i, c in enumerate(calls):
                res = run_tool(self.store, c.get("name") or "", c.get("arguments") or {},
                               ctx=ctx)
                notes.append({"name": c.get("name") or "", "ok": bool(res.get("ok")),
                              "detail": res.get("detail", ""),
                              "mode": res.get("mode", "")})
                messages.append({"role": "tool",
                                 "tool_call_id": c.get("id") or f"call_{i}",
                                 "content": res.get("detail", "") or "（没做成）"})
            out = self.llm.chat_with_tools(messages, tools=tools,
                                           max_tokens=max_tokens, temperature=temperature,
                                           timeout=timeout)
        self._pick_up_proposals(ctx)
        # 失败种类**原样带出去**（不在这里翻成人话）——它是"她说了什么"之外的事，
        # 由调用方放进 `hint`。塞进 `text` 的话，那句告示会被当正文写进窗口和记忆，
        # 而它明明是给用户看的系统消息（`reply()` / `reply_stream()` 都这么处理）。
        return (out.get("content") or ""), notes, (out.get("error") or "")

    def _pick_up_proposals(self, ctx: dict) -> None:
        """把她问出口的提议存成「待确认」（**一次只留一个**——多了她自己也说不清）。"""
        props = ctx.get("proposals") or []
        if props:
            self.pending_confirm = props[0]

    def pending_view(self) -> dict | None:
        """给界面看的「她在等什么」——用来弹那条确认。"""
        p = self.pending_confirm
        if not p:
            return None
        sid = p.get("scene_id") or ""
        rule = layer_edit(sid)
        field = p.get("field") or rule["main"]
        return {"scene_id": sid, "title": p.get("title") or "",
                "tool": p.get("tool") or "", "text": p.get("text") or "",
                # 改哪个字段（展示名**按层取**；2026-09-24 起三层都能改标签）
                "field": field,
                "field_label": (rule["labels"].get(field)
                                or rule["labels"].get(rule["main"], "文本")),
                "scene_ids": p.get("scene_ids") or []}

    def confirm(self, action: str) -> dict:
        """他点了确认条上的一个动作。**点了就算数**，不用再猜他的话是什么意思。

        `action`：处置提议三档——`"delete"` 彻底删除 / `"archive"` 归档冷存 /
        `"keep"` 不删除（提议作废）；改的提议两档——`"accept"` / `"keep"`。
        **三层同一套**（2026-09-24，工具箱稿 §3.4）：按编号前缀分派——
        S1 场景 / S2 摘要 / S3 画像，各调各的方法，流程与后果一致。

        还认旧的 `accept: True/False`（老页面缓存 / 实验脚本）：`True` → `"accept"`、
        `False` → `"keep"`——**只有这一层兼容，别再加第二层**。

        这是改与删的**主通道**：她负责问出口，界面负责把选择收清楚。
        对话里回答仍然有效（那是兜底，见 `_resolve_pending`）——
        但界面上点一下更直接，也用不着判定。

        **回执也回到她那里**（2026-09-24 修）：界面这条路不经过工具回合，
        结果原本只回到界面（`detail` 给他看）——她不知道自己提议的那件事
        后来怎么了。每档都记一笔，见 `_note_confirm`。
        """
        if action is True:
            action = "accept"
        elif action is False:
            action = "keep"
        action = (action or "").strip().lower()
        p = self.pending_confirm
        if not p:
            return {"ok": False, "detail": "现在没有等着确认的事"}
        self.pending_confirm = None
        if action in ("", "keep", "no", "cancel"):
            out = {"ok": True, "detail": "算了，不动它。"}
            self._note_confirm(action or "keep", out)
            return out
        raw = p.get("ids") or p.get("scene_ids") or []
        sid = p.get("scene_id") or (raw[0] if raw else "")
        if p.get("tool") == "forget_memory":
            ids = [str(i).strip() for i in (raw or ([sid] if sid else []))
                   if str(i).strip()]
            if not ids:
                out = {"ok": False, "detail": "这条提议里没有编号"}
                self._note_confirm(action, out)
                return out
            if action == "archive":
                out = archive_by_layer(self.store, ids)
                self._note_confirm(action, out, "、".join(ids))
                return out
            if action in ("delete", "accept", "yes"):
                out = delete_by_layer(self.store, ids)
                self._note_confirm(action, out, "、".join(ids))
                return out
            out = {"ok": False, "detail": f"认不出的动作：{action}"}
            self._note_confirm(action, out)
            return out
        if action not in ("accept", "yes"):
            out = {"ok": False, "detail": "这条提议只有「执行 / 算了」两档"}
            self._note_confirm(action, out, sid)
            return out
        new = (p.get("text") or "").strip()
        if not new:
            # 空值会把内容**清空**——那是脏数据，不是"改成没有"
            out = {"ok": False, "detail": "没说改成什么，不动它。"}
            self._note_confirm(action, out, sid)
            return out
        out = update_by_layer(self.store, sid, new, p.get("field") or "text")
        self._note_confirm(action, out, sid)
        return out

    def _note_confirm(self, action: str, out: dict, what: str = "") -> None:
        """**回执回到她那里**（2026-09-24 修）：把她提议、他点过的结果记进
        `receipts`——由 `_actions_note` 带上，与工具回合（`last_actions`）同一行。

        为什么要专门补：界面点确认条**不经过工具回合**，结果只回到界面
        （`detail` 是给他看的），`last_actions` 里没有它——于是她不知道自己
        提议的那件事后来怎么了：下一轮会**再提议一次**，或者**还引用已经删掉
        的那条**（你问"刚才那条呢"她答不上来）。

        留 `_RECEIPT_TURNS` 轮就够（不一直挂着）：隔一两轮再问起时她答得上来；
        再久是白占上下文——而真相在库里，真要查证她自己 `memory_search` 就行。
        """
        verb = {"delete": "用户点了彻底删除", "archive": "用户点了归档冷存",
                "keep": "用户点了不删除", "accept": "用户点了执行",
                "yes": "用户点了执行"}.get(action, f"用户点了「{action}」")
        line = verb + (f"（{what}）" if what else "")
        line += f"：{(out.get('detail') or '')[:40]}"
        if not out.get("ok"):
            line += "（没成）"
        self.receipts = ((self.receipts or []) + [(line, _RECEIPT_TURNS)])[-3:]

    def _actions_note(self) -> str:
        """她动过什么——**给她自己看的一行**，不进记忆。

        工具结果只是**那一轮**的临时消息，下一轮就没了：
        于是她会忘掉自己刚删过 / 改过什么，你问"刚才那条呢"她答不上来。

        两份拼在这一行里：
          - `last_actions`：本轮（上一轮）的工具动作，**每轮覆盖**；
          - `receipts`：他点过确认条之后的结果，**按轮衰减**（`_RECEIPT_TURNS`）。

        措辞是"你动过"而不是"上一轮你动过"——回执可能来自更早那一两轮。
        """
        parts = list(self.last_actions or []) + [t for t, _ in (self.receipts or [])]
        if not parts:
            return ""
        return "（你动过：" + "；".join(parts) + "）"

    def _remember_actions(self, notes: list[dict]) -> None:
        """把这一轮的工具结果压成**最多三行**（给下一轮当上下文）。

        详情在 `data/trace/工具-*.jsonl`，这里只是"她瞄一眼自己刚做了什么"。
        """
        self.last_actions = [
            f"{t.get('name') or ''}：{(t.get('detail') or '')[:40]}"
            + ("" if t.get('ok') else "（没成）")
            for t in (notes or [])][:3]
        # 回执**过一轮少一轮**，归零就丢——不一直挂着（见 `_RECEIPT_TURNS`）
        self.receipts = [(t, n - 1) for t, n in (self.receipts or []) if n > 1]

    def _resolve_pending(self, user_msg: str) -> str:
        """他这句是不是在回答上一句问出口的提议。返回一句给她的说明（空串 = 没事发生）。

        **必须在唤醒之前判**：答案决定这一轮该怎么接话，也可能改变该想起什么。
        真执行走的是 `weave` 那两个 confirmed 函数——**和界面按钮同一个动作**，
        按钮和工具只是发起方不同。
        """
        pend = self.pending_confirm
        if not pend:
            return ""
        verdict, said = _judge_confirmation(user_msg)
        sid = pend.get("scene_id") or ""
        if verdict == "vague":
            # 不动，再问一次：**含糊的不能算数**
            return (f"（用户没说清楚。再问一次：{sid} 是删、归档冷存，还是改？"
                    f"「嗯」这类不算数——删除和改动都不该由一句语气词触发。）")
        if verdict == "drop":
            self.pending_confirm = None
            return "（用户岔开了——上一句问的那件事作废，别再追问。）"

        self.pending_confirm = None
        if verdict == "delete":
            ids = [str(i).strip() for i in
                   (pend.get("ids") or pend.get("scene_ids")
                    or ([sid] if sid else [])) if str(i).strip()]
            out = delete_by_layer(self.store, ids)
            done = out.get("ids") or []
            how = f"{len(done)} 条（{'、'.join(done)}）" if len(ids) > 1 else sid
            return (f"（用户说删，已删掉 {how}——真删，没有备份。）"
                    if out.get("ok") else
                    f"（没删成：{out.get('detail') or '未知原因'}）")
        if verdict == "archive":
            ids = [str(i).strip() for i in
                   (pend.get("ids") or pend.get("scene_ids")
                    or ([sid] if sid else [])) if str(i).strip()]
            out = archive_by_layer(self.store, ids)
            done = out.get("ids") or []
            how = f"{len(done)} 条（{'、'.join(done)}）" if len(done) > 1 else sid
            return (f"（用户说归档，{how} 已进冷层——她不会再想起它，数据都在、还能捞回来。）"
                    if done else "（没能归档。）")
        new = said or pend.get("text") or ""
        rule = layer_edit(sid)
        field = pend.get("field") or rule["main"]
        out = update_by_layer(self.store, sid, new, field)
        label = rule["labels"].get(field, "")
        what = f"的{label}" if label and field != rule["main"] else ""
        if out.get("ok"):
            return f"（用户说改，已把 {sid}{what} 改成「{new}」。原来的说法留了痕。）"
        return f"（没改成：{out.get('detail') or '内容没变'}）"

    def _tool_names(self) -> list[str]:
        """这一轮实际可用的动作名。`web` 是搜 + 抓两个能力（2026-09-24 合并）：
        **两个都关**才摘掉它（搜索被人关 / 服务不支持 **且** 抓取也关）——
        留一个用不了的工具只会让她选错（防错第 1 条：工具要少）。
        提示词里的分寸块和 `tools` 参数共用这一份，两边不会漂。

        抓取**独立于搜索**：搜索挂了不挡她抓链接——那是本机出网的事，
        和搜索端点没关系（所以只有一边关时，工具留着）。"""
        names = tool_names()
        can_search = (cfgmod.cfg("search", "enabled", default=True)
                      and self.llm.search_capable())
        can_fetch = cfgmod.cfg("web", "fetch_enabled", default=True)
        if not (can_search or can_fetch):
            names = [n for n in names if n != "web"]
        return names

    def _prepare(self, user_msg: str, note: str = "") -> tuple[dict, str, list[dict], int]:
        """① 唤醒 + ② 拼上下文。**流式和非流式共用这一份**。

        顺序、档案、人格、语言……全在这里定好——
        分成两套的话，迟早出现"流式下她看到的记忆不一样"这种无从查起的 bug。
        """
        recall = recall_for_message(user_msg, self.store, llm=self.llm, emb=self.emb)
        # 基础档案（称呼/年龄…）每轮都取一次：它随时可能被改，
        # 而且它排在所有推测之前——「该叫他什么」不该等到下一次重启才生效。
        lang = self.lang()     # 每轮读：切了语言下一句就生效
        # 工具的分寸进系统提示词（同 DeepSeek Harness：描述只说是什么，
        # 该不该用在提示词里讲）。没开的动作不进块——说了她也没法用，
        # 还会诱导她假装做了。
        avail = (self._tool_names()
                 if cfgmod.cfg("chat", "tools_enabled", default=True) else None)
        # 档案每轮只算一次：下面的 `measure` 会跑好几遍（裁一格量一次），
        # 把它留在闭包里等于每裁一格查一次库。
        facts = self.store.all_user_facts()
        window_blocks = self.st.build_window_blocks()
        # 当前消息（以及「刚刚发生了什么」那句说明）**不可裁**——
        # 它是这一轮存在的原因，先把它占掉的算出来，剩下的才是可裁的预算。
        tail_tokens = estimate_tokens((f"{note}\n\n" if note else "")
                                      + f"[当前消息]\n用户: {user_msg}")

        # 每轮现读人格：切了版本，下一句就换人（同 lang 的理由）。
        persona = self.store.get_pref("persona") or "air"
        charter = load_persona(persona)

        def _render() -> str:
            return build_system_prompt(
                charter, recall, recall.get("summaries"),
                facts=facts, tools=avail, lang=lang,
                persona=persona, personas=persona_names())

        def _measure() -> int:
            window = "\n\n".join(t for _, t in window_blocks)
            return (estimate_tokens(_render()) + estimate_tokens(window)
                    + tail_tokens)

        # 总量仲裁：**超了才裁**（`total_budget <= 0` = 不仲裁，见 `fit_context`）。
        # 报告挂回 `recall` 里——裁掉了什么必须看得见（同「抑制名单」那条理由：
        # 「应该出现的东西没出现」是这类系统最难查的故障）。
        recall["budget"] = fit_context(recall, window_blocks, _measure)

        system = _render()
        window = "\n\n".join(text for _, text in window_blocks)
        content = (f"{window}\n\n[当前消息]\n用户: {user_msg}" if window else user_msg)
        if note:
            # 「刚刚发生了什么」排在最前：它是这一轮的前提，不是他消息的一部分
            content = f"{note}\n\n{content}"
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
        # 上限只有一个，**不按任何"简短"偏好往下压**：
        # 推理模型（`deepseek-flash` 这类）把**思维链也算进这笔额度**，曾经给「简洁」档
        # 压到 600，结果是她想都没想完就没额度说话了——正文 100% 为空，
        # 症状看着像模型坏了，其实是额度不够。
        # 「简洁」要的是**话短**，不是**想法短**——那是人格文件的事
        # （说话方式五档 2026-09-19 已退役），不是掐额度的事：
        # **额度这一个旋钮管不了这两件事**。
        # 兜底值须与 `config` 同值（只在配置键缺失时生效；不一致即漂移）
        max_tokens = int(cfgmod.cfg("chat", "max_tokens", default=16000))
        return recall, system, messages, max_tokens

    def reply_stream(self, user_msg: str):
        """流式版：yield 事件，最后一个是 `end`（带 recall / written / 工具笔记）。

        **它和 `reply()` 是同一条链路**——同一个 `_prepare`、同一套唤醒 → 写入 → 记账，
        只是生成那一步换成边想边吐。**流式不是另一套系统**，否则两边的记忆会不一样。

        事件：`reasoning`（思考增量）/ `content`（正文增量）/ `tool`（她调了什么）/
        `end`（收尾：hint / written / recall / truncated）。失败**不单独发事件**——
        提示随 end 的 `hint` 一起给，前端把它当系统消息显示。
        """
        user_msg = (user_msg or "").strip()
        if not user_msg:
            yield {"type": "end", "reply": "", "recall": {}, "written": None,
                   "system": "", "tools": [], "memo_raised": [], "mentioned": [],
                   "hint": "", "truncated": False}
            return

        # 上一句问出口的事：**他这句是不是在回答它**（必须在唤醒之前判）
        note = self._resolve_pending(user_msg)
        acts = self._actions_note()          # 上一轮她动过什么（给她自己看）
        if acts:
            note = f"{note} {acts}".strip()
        recall, system, messages, max_tokens = self._prepare(user_msg, note)
        # 她**说到一半被打断**时也要算数：`GeneratorExit` 是"这一头断开了"的唯一信号
        # （按停止、关页面、刷新都走它）。已经吐出去的那半句得落进**窗口**——
        # 抄的是 AI SDK（`chatbot-resume-streams`：停止时保存 assistant 快照），
        # 只是这一侧写更省事：她说了什么服务端本来就知道，不必让客户端回传，
        # 也就没有那份"客户端快照比服务端旧、覆盖掉新内容"的风险。
        # ⚠️ 只抄到窗口这一步：**长期库不留它**（提取时滤掉，见
        # `_commit_partial_turn` 与 `ShortTerm.compress_and_extract`）。
        partial: dict = {"reply": ""}
        try:
            reply, notes, err = yield from self._generate_stream(messages, max_tokens,
                                                                partial)
        except GeneratorExit:
            self._commit_partial_turn(user_msg, partial["reply"])
            raise
        # 失败提示走 `hint` 字段（前端当**系统消息**显示），不冒充她的正文——
        # 同 `reply()`：写进记忆的那句必须是她真说的话。
        hint = _no_reply_hint(err) if not reply.strip() else ""

        # ④ 写入 ⑤ 记账——一个都不少，和 `reply()` 完全一致
        # （含 `persona` 署名，见 `reply()` 里的说明）
        self.st.append("user", user_msg)
        self.st.append("air", reply, persona=self.store.get_pref("persona") or "air")
        written = self.st.flush_if_needed()
        memo_raised = self._mark_memos_raised(recall)
        memo_closed = self._closed_memo_view()
        mentioned = mark_mentioned(self.store, recall.get("scenes") or [],
                                   [user_msg, reply])
        self._remember_actions(notes)
        yield {"type": "end", "reply": reply, "recall": recall, "written": written,
               "system": system, "tools": notes, "memo_raised": memo_raised,
               "memo_closed": memo_closed, "mentioned": mentioned,
               # 这一轮她一个字都没回时的**系统提示**（前端显示为「系统」气泡）。
               # 它不属于 `reply`——那句是"她说过的话"，会被写进窗口和记忆。
               "hint": hint,
               # 被额度掐断（`finish_reason=length`）**且正文非空**时也要说一声：
               # 原来的失败提示只在"一个字都没回"时出现，于是"说了一半被截断"
               # 看起来像正常结束（真出过）。
               "truncated": err == "truncated",
               "confirm": self.pending_view()}

    def _commit_partial_turn(self, user_msg: str, partial: str) -> None:
        """把**说到一半就被叫停**的这一轮落进窗口（`reply_stream` 收到 `GeneratorExit` 时调）。

        与正常那一轮的差别只有一处：**不跑 `flush_if_needed`**。
        提取是一次 LLM 调用，不该出现在"正在关闭连接"这条路线上；
        而且少这一次提取不会丢东西——窗口没清、原文照样在，
        触发条件（话题切换 / 预算 / 空闲）下一次 append 会照常判。

        她那半句是**逐字**落的（前端只加一个「已停止」的标记，不动正文），
        但**只活在窗口里**：`interrupted` 那条在提取时会被滤掉，不进原文、
        不进场景卡（见 `ShortTerm.append` 与 `compress_and_extract`）——
        「短期留着、长期不留」是 2026-10-07 定的口径。

        一个字都没吐出来时不落 air 那条——她没答话，窗口里就只有他问的那句
        （下一轮她仍看得见它）。
        """
        text = (partial or "").strip()
        self.st.append("user", user_msg)
        if text:
            # `interrupted` 管两件事：界面回填时标「（已停止）」；**提取时被滤掉**
            # （所以长期库里没有它——见 `ShortTerm.append` / `compress_and_extract`）
            self.st.append("air", text, interrupted=True,
                           persona=self.store.get_pref("persona") or "air")

    def _generate_stream(self, messages: list[dict], max_tokens: int,
                         partial: dict | None = None):
        """生成（流式，最多两轮工具）。返回 `(正文, 工具笔记, 失败种类)`。

        `partial`：一个可变的 `{"reply": ...}` 口袋，边生成边更新。存在的理由只有一个
        ——**被打断时**（`GeneratorExit`）上层要拿得到"她已经说了多少"，
        而中断点在 `yield` 上，`reply` 是这一帧的局部变量，外面看不见。
        """
        tools = None
        if cfgmod.cfg("chat", "tools_enabled", default=True):
            tools = openai_tools(self._tool_names())
        temperature = float(cfgmod.cfg("chat", "temperature", default=0.7))
        # **空闲超时**：这个值是每一次 read 的上限，不是整轮的上限——
        # 她可以想十分钟（只要一直在出字真有字出来），真卡住（60 秒一个字都没有）
        # 才报错。用 `timeout_reply` 那个 600 就错了：那会让"卡住"等满十分钟。
        timeout = float(cfgmod.cfg("chat", "timeout_stream", default=60))
        # `confirm_channel`：**确认通道就是对话本身**（工具箱 3.6）——
        # 需确认档的工具因此能放行到"提议"，等他下一句话，而不是直接拒绝。
        ctx = {"emb": self.emb, "llm": self.llm,
               "confirm_channel": True, "seen_scenes": self._seen_scenes}
        reply, notes, err = "", [], ""
        truncated = False
        rounds = 0
        while True:
            truncated = False      # 每轮重置：只反映「最后一次生成」是否被掐断
            calls: list[dict] = []
            for ev in self.llm.chat_stream(messages, tools=tools, max_tokens=max_tokens,
                                           temperature=temperature, timeout=timeout):
                kind = ev.get("type")
                if kind == "reasoning":
                    yield ev
                elif kind == "content":
                    reply += ev.get("text") or ""
                    if partial is not None:
                        partial["reply"] = reply      # 被打断时外面要拿得到
                    yield ev
                elif kind == "tool_calls":
                    calls = ev.get("calls") or []
                elif kind == "truncated":
                    truncated = True          # 不往下传：界面上只影响"没内容时怎么说"
                elif kind == "error":
                    # 失败信号**不转发给前端**：它带的是机器码（timeout / network），
                    # 要由 `_no_reply_hint` 翻成人话——而翻译的判据都在后端，
                    # 扔给前端只会得到一句"出错了"。翻好的提示随 end 一起给（hint）。
                    err = ev.get("error") or "other"
            if not calls or not tools or rounds >= 2:
                break
            rounds += 1
            messages = messages + [{
                "role": "assistant", "content": reply,
                "tool_calls": [
                    {"id": c.get("id") or f"call_{i}", "type": "function",
                     "function": {"name": c.get("name") or "",
                                  "arguments": json.dumps(c.get("arguments") or {},
                                                          ensure_ascii=False)}}
                    for i, c in enumerate(calls)]}]
            for i, c in enumerate(calls):
                res = run_tool(self.store, c.get("name") or "", c.get("arguments") or {},
                               ctx=ctx)
                notes.append({"name": c.get("name") or "", "ok": bool(res.get("ok")),
                              "detail": res.get("detail", ""), "mode": res.get("mode", "")})
                # 她调了什么，界面这一轮就看得见（不用等结束）
                yield {"type": "tool", "name": c.get("name") or "",
                       "ok": bool(res.get("ok")), "detail": res.get("detail", "")}
                messages.append({"role": "tool",
                                 "tool_call_id": c.get("id") or f"call_{i}",
                                 "content": res.get("detail", "") or "（没做成）"})
        self._pick_up_proposals(ctx)
        return reply, notes, (err or ("truncated" if truncated else ""))

    def _mark_memos_raised(self, recall: dict) -> list[str]:
        """把本轮**到点进过注入**的那件备忘录标成「已提」——**提过一次就不再提**。

        2026-10-05 晚改口径：候选机制（`memo_candidates`）删了，记账点搬到
        `standing_memos` 那一栏的 **`due` 标记**上（每轮最多一件，由 `memo.standing_memos`
        打标）。判据不变：**它进了这一轮的注入**，不是"air 真的说出口了"——
        进注入 = 已经给过它机会；用不用那个话头是她的分寸（那一栏的分寸写在
        `render_memory_block` 里），不该反过来让 memo 再来一次。
        """
        out = []
        for m in recall.get("standing_memos") or []:
            if not (m or {}).get("due"):
                continue
            mid = (m or {}).get("id") or ""
            if not mid:
                continue
            # 同一轮里被**命中判定改成「变更」**的那条：库里的内容已经不是注入时那份了
            # （`judge_hits` 在 `st.append("user", …)` 里跑，早于这里）——别记成"提过"：
            # 记账打上的话，**新内容就永远没机会再进注入**（它已经 raised）。
            # 下轮它自己会以新内容进注入——那才是「内容变了 = 有新的事实要提醒」。
            cur = self.store.get_memo(mid)
            if cur is not None and (cur.content or "") != (m.get("content") or ""):
                continue
            # `mark_raised` 返回"真的转了吗"：同一轮里被判定"已了结"的那条
            # 已经不是 pending，不该被记成"提过"——回执只报真进账的
            if mark_raised(self.store, mid):
                out.append(mid)
        return out

    def _closed_memo_view(self) -> dict | None:
        """这一轮他把哪件备忘录说成了结果（**命中判定**判的）——进回执。

        为什么补这一步（2026-09-22）：`ShortTerm.last_closed_memo` 原来是
        **只写不读**的（注释写着"给调用方留痕"，而全项目没有那个调用方）——
        闭合照样发生、钩子照样回流，只是"这一句结掉了哪件事"谁都看不见。
        接法对齐 `memo_raised`：进本轮回执；留痕在 `memo.close` 那边落
        （`备忘-*.jsonl`，含 id / 内容 / 谁关的）。

        带 `content` 是有意的：光有编号，调用方要说"结掉了哪件"还得再查一次库。
        """
        mid = self.st.take_closed_memo()
        if not mid:
            return None
        m = self.store.get_memo(mid)
        return {"id": mid, "content": (m.content if m is not None else "")}

    # ---- 收尾 ----

    def close(self, clear_digest: bool = False) -> dict | None:
        """结束会话：把最后一段提取掉（不然尾巴丢了）。

        `clear_digest=True` 是「新对话」（见 `App.new_session`）：连压缩摘要一起清，
        窗口真的从零开始——摘要里的内容早进长期库了，需要时会靠回忆拉回来。
        （收尾 / 空闲结束不传它：那是同一段对话的延续，摘要还要当背景。）
        """
        # 会话都结束了，就**不该留一个待执行的删除**——
        # 隔了几天回来第一句话，很可能不是在对它点头。
        self.pending_confirm = None
        self.receipts = []            # 回执也随会话走（同上：隔几天回来不该还挂着）
        self.st.end_session()
        out = self.st.flush_if_needed()
        if clear_digest:
            self.st.clear_digest()
        return out

    def window_preview(self) -> str:
        """当前窗口的渲染结果（仪表盘上显示「air 这轮看到了什么」）。"""
        return self.st.build_window()
