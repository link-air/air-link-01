# 模块接口总表 · Module index

> **生成物，别手改**——素材是各模块头部的「模块速查」块与公开符号。
> 重新生成：`python tools/module_index.py`；对账（不改文件）：`python tools/module_index.py --check`。
> 中文是正文，英文只在标题与字段名上（这个项目的中文即文档原文，翻译见 README.en.md 的说明）。

字段：`Layer` 层级 · `Upstream` 上游 · `Downstream` 下游 · `Entry points` 对外入口 · `Boundary` 边界。

## 目录 · Contents

- [`core/chat.py`](#corechatpy)
- [`core/config.py`](#coreconfigpy)
- [`core/dashboard.py`](#coredashboardpy)
- [`core/distill.py`](#coredistillpy)
- [`core/embedding.py`](#coreembeddingpy)
- [`core/entity.py`](#coreentitypy)
- [`core/llm.py`](#corellmpy)
- [`core/memo.py`](#corememopy)
- [`core/model.py`](#coremodelpy)
- [`core/net.py`](#corenetpy)
- [`core/persona.py`](#corepersonapy)
- [`core/prompts.py`](#corepromptspy)
- [`core/recall.py`](#corerecallpy)
- [`core/salvage.py`](#coresalvagepy)
- [`core/scene.py`](#corescenepy)
- [`core/search.py`](#coresearchpy)
- [`core/settings.py`](#coresettingspy)
- [`core/shortterm.py`](#coreshorttermpy)
- [`core/store.py`](#corestorepy)
- [`core/tools.py`](#coretoolspy)
- [`core/trend.py`](#coretrendpy)
- [`core/weave.py`](#coreweavepy)
- [`core/webfetch.py`](#corewebfetchpy)

---

## `core/chat.py`

对话层：把记忆层接进一次真实的对话。

| | |
|---|---|
| 层级 · Layer | L10 对话层 |
| 上游 · Upstream | recall（读）、shortterm（写）、tools（她的手）、weave（确认后的动作）、 prompts / model / memo / llm / embedding / persona / store / config |
| 下游 · Downstream | dashboard（应用侧唯一调用方）、run_experiment（实验回放） |
| 对外入口 · Entry points | `ChatSession`（`reply` / `reply_stream` / `confirm` + `pending_view` / `close` / `window_preview`）+ `build_embedding` / `load_persona` / `persona_names` |
| 边界 · Boundary | 只编排，不判断——该不该翻是 recall 的事，该不该存是 distill 的事 |

**公开符号 · public API**

- `build_embedding() -> EmbeddingService | None` — 按配置建向量服务；没配就返回 None（上层据此走字符重叠降级）。
- `load_persona(name) -> str` — 当前人格的提示词——**每轮现读**：切了 / 改完，下一句就换。
- `persona_names() -> list[str]` — `self/personas/` 里有哪些人格——界面的选项列表用（扫目录，加文件即生效）。
- `fit_context(recall, window_blocks, measure, budget) -> dict` — 把这一轮的注入量压进 `context.total_budget`（**就地**改前两个参数）。
- **class `ChatSession`** — 一次会话。持有 store / llm / embedding / 短期窗口。
- 　· `lang(self) -> str` — 语言（中 / 英，他自己定的）。每轮都读：切完下一句就生效。
- 　· `reply(self, user_msg) -> dict` — 收一句话，回一段，并把这一轮的记忆动作一并返回（给仪表盘用）。
- 　· `pending_view(self) -> dict | None` — 给界面看的「她在等什么」——用来弹那条确认。
- 　· `confirm(self, action) -> dict` — 他点了确认条上的一个动作。**点了就算数**，不用再猜他的话是什么意思。
- 　· `reply_stream(self, user_msg)` — 流式版：yield 事件，最后一个是 `end`（带 recall / written / 工具笔记）。
- 　· `close(self, clear_digest) -> dict | None` — 结束会话：把最后一段提取掉（不然尾巴丢了）。
- 　· `window_preview(self) -> str` — 当前窗口的渲染结果（仪表盘上显示「air 这轮看到了什么」）。

## `core/config.py`

全局配置与常量。

| | |
|---|---|
| 层级 · Layer | L0 基础层 |
| 上游 · Upstream | 无（python 标准库之外不依赖任何东西） |
| 下游 · Downstream | 几乎全部（每个需要读数字的模块都 import 它） |
| 对外入口 · Entry points | `CONFIG` / `PATHS` / `cfg()` / `abspath()` / `db_path()` / `ensure_dirs()` |
| 边界 · Boundary | 只放常量与路径，**不放任何需要推理的逻辑**—— 一旦这里开始做判断，它就成了第二个 store |

**公开符号 · public API**

- `cfg(*keys, default)` — 按路径取配置：`cfg("recall", "inject_n")`。
- `abspath(rel) -> Path` — 把配置里的相对路径解析成基于项目根目录的绝对路径。
- `db_path() -> Path` — 库文件路径。`CONFIG.db_path` 优先、`PATHS.db` 兜底——
- `ensure_dirs() -> None` — 确保运行期目录存在（data / trace / backups / self）。

## `core/dashboard.py`

本地仪表盘：把记忆系统的内部状态摆出来看。

| | |
|---|---|
| 层级 · Layer | L11 界面层 |
| 上游 · Upstream | 几乎全部（它是唯一把所有层拼起来的地方） |
| 下游 · Downstream | 无（`python -m core.dashboard` 的入口就在这里） |
| 对外入口 · Entry points | `App`（HTTP 服务）/ `serve()`（`run.cmd` 与「全拉起」用它） `Handler.do_GET` / `Handler.do_POST` 是全部 API 的两张路由表 |
| 边界 · Boundary | **不放业务逻辑**——处理函数只做"收参数 → 调 → 打包响应"， 要判断就在它该住的那个层里加方法 |

**公开符号 · public API**

- **class `App`** — 跨请求共享的状态。HTTP 是多线程的，所以会话拿锁保护。
- 　· `session(self) -> ChatSession` — 懒建会话：第一次说话时才建，这样启动仪表盘不会先花一次 LLM 调用。
- 　· `chat(self, msg) -> dict` — 走完整链路回一句话（**加锁**）——非流式，流式那版是 `chat_stream`。
- 　· `speak(self, text, voice) -> dict` — 把一段话交给**本地语音服务**，拿回音频（打包成 data URI 给页面）。
- 　· `speak_stream(self, text, voice)` — 流式版播报：**把语音服务的 PCM 帧原样转给页面**。
- 　· `voice(self) -> dict` — 她的声音：**基底音色 + 微调刻度 + 那句手写描述**。**他定的**，存 `user_prefs`。
- 　· `voice_save(self, speaker, extra, dims) -> dict` — 保存声音设置。空值一律存成"跟随 voice.json"，不往库里写一份死值。
- 　· `voice_status(self) -> dict` — 语音服务现在什么样：在不在、加载哪个模型、用哪个音色、有哪些可选。
- 　· `voice_start(self) -> dict` — 把语音服务拉起来——**人点，系统不自动起**（同"只有人能切模式"那条：
- 　· `voice_stop(self) -> dict` — 关掉语音服务（**放掉显存**）——让它自己退（`/shutdown`），不是外面硬杀。
- 　· `voice_config(self) -> dict` — 读语音服务的配置（`tts/voice.json`）+ 本地可选的模型清单。
- 　· `voice_config_save(self, patch) -> dict` — 写回 `tts/voice.json` 的**白名单字段**（合并 + 原子替换）。
- 　· `confirm(self, action) -> dict` — 他点了确认条上的动作（`delete` / `archive` / `keep` / `accept`）——
- 　· `undo_turns(self, turns, expect) -> dict` — 撤掉窗口末尾的 N 轮（改与重新生成的前半步，见 `ShortTerm.undo_turns`）。
- 　· `chat_stream(self, msg)` — 流式：**和非流式用同一把锁**——窗口是串行的，对话本来就是一句一句来。
- 　· `end(self) -> dict | None` — 收尾：提取最后一段 → 后台整理。与 `new_session` 只差「摘要清不清」。
- 　· `new_session(self) -> dict | None` — 「新对话」：结束这一段 → 窗口（**连压缩摘要**）整个清空 → 后台整理。
- 　· `reload_clients(self) -> None` — 配置变了：重建 LLM / 向量客户端、清掉会话（**加锁**）。
- 　· `start_distill(self, maintenance, trigger) -> bool` — 起一个后台线程跑一次整理；已在跑则忽略（返回 False）。
- 　· `wait_distill(self) -> bool` — 等后台提炼收工（**只给退出用**）。返回是否跑完了。
- 　· `maintenance_tick(self) -> dict` — 体检的触发判定（记忆整理稿 §三）：量到、或时间到，就跑一次。
- 　· `window_text(self) -> str` — 窗口的渲染预览（「她这轮看到了什么」）——**不依赖会话**。
- 　· `history(self, limit) -> dict` — 重启/刷新后对话区回填：窗口 + 最近几段已提取的原文——**只读，不建会话**。
- 　· `state(self) -> dict` — 首屏那包：计数 + 开关 + 语言 + 窗口内容（不含列表）。
- 　· `mirror(self) -> dict` — 镜像视图（复用编织层那条路，不另写一份拼装逻辑）。
- 　· `maint_reviews(self) -> list[dict]` — 未处理的复核提议（`wrong`）+ 那条画像的陈述（界面直接显示用）。
- 　· `persona(self) -> str` — 当前人格名（air / mia / xina……）。**不建会话**——读个偏好不该花一次 LLM。
- 　· `set_persona(self, name) -> str` — 切人格：**在名单里才落库**；对话每轮现读——下一句就换人。
- 　· `persona_list(self) -> dict` — 设置页「人格」块要的一屏：每个人的显示名 / 字数 / 在不在用 / 常驻块体量。
- 　· `lang(self) -> str` — 界面与她的输出语言（中 / 英）。同上——读一个偏好不该花一次 LLM。
- 　· `reject(self, pid) -> dict` — 用户否决一条画像（**呈现的另一半**）：**真删**（2026-09-24）。
- 　· `memory_action(self, ids, action, text, field) -> dict` — 记忆动作的统一口：**改与删，三层同一套**（2026-09-24，工具箱稿 §3.4）。
- 　· `scenes(self, limit) -> list[dict]` — 场景卡列表（**含冷层**——这一页是台账，藏起冷层会让人以为记忆变少了）。
- 　· `summaries(self) -> list[dict]` — S2 列表（聚合出来的主题叙述）。50 是台账页自己的量，
- 　· `memos(self) -> list[dict]` — 备忘录（含到点标记 `due_now`）。
- 　· `memo_close(self, mid) -> dict` — 人手划掉一件备忘录（2026-09-21，设计稿 D 条 4）——**和她说 `close_memo` 同一个动作**。
- 　· `entities(self) -> list[dict]` — 实体索引 + 每条涉及多少场景 + 关系词 + 活跃标记。
- 　· `trace(self, limit) -> list[dict]` — 留痕页要的那几行（按记录自己的时间倒序）。
- 　· `topic_merge_suggestions(self) -> list[dict]` — 疑似「同一个主题被写成两种说法」的候选对（**只建议，不合并**）。
- **class `Handler`** — （无 docstring）
- 　· `do_GET(self) -> None` — **只读**路由表——都不改记忆（送达标记、改和删都在 POST 那边）。
- 　· `do_POST(self) -> None` — **写**路由表。四类，边界很清楚：
- 　· `log_message(self, fmt, *args) -> None` — 默认的每请求一行日志太吵（前端轮询会刷屏），只留错误。
- `serve(port, open_browser, with_voice) -> None` — 起服务（**阻塞**）：`run.cmd` / `启动.vbs` 最终走到这里。

## `core/distill.py`

提炼流水线。

| | |
|---|---|
| 层级 · Layer | L5 提炼层 |
| 上游 · Upstream | config、embedding、model、prompts、scene、entity、store |
| 下游 · Downstream | shortterm（`distill_step1` = 它的压缩）、trend（渐变检测由这里带动）、 memo（`memo_cycle` 挂在提炼周期末尾）、dashboard（手动/周期跑/体检） |
| 对外入口 · Entry points | `distill_step1` / `distill_step2` / `distill_step3` / `run_distill_cycle` / `converge_detail` / `revise_profile` / `age_out_profiles` / `cap_profiles` / `maintenance_cycle`（体检：封顶主整理 + S3 复核 + 留痕）/ `review_profiles` |
| 边界 · Boundary | **改画像前一律从库里重读**（调用方手里那个对象可能是旧的， 拿它判「是否已升级」会重复计数、重复写库） |

**公开符号 · public API**

- `distill_step1(store, messages, llm, emb_service, source) -> tuple[Scene | None, str, list[dict]]` — S0→S1：把一段对话落成场景卡，并处理它的副产品。
- `write_skip_trace(messages, reason) -> None` — 跳过落库也要留痕——**「没存什么、为什么」和「存了什么」一样重要**。
- `resolve_topic(store, statement, subject, llm) -> str` — 为画像决定 topic（存储层 §3「新建 vs 复用」）。
- `distill_step2(store, topic, llm, n) -> Summary | None` — 提炼 2：S1 → S2 **聚合**（攒够 n 条同主题才做）。
- `converge(store, profile, emb) -> bool` — 印证收敛：判定一条画像能否从 `pending` 升为 `established`（提炼 3 的四道关）。
- `converge_detail(store, profile, emb) -> tuple[bool, str]` — 带原因的收敛判定：四道关**卡在哪一关、差多少**。
- `revise_profile(store, pid, new_statement, evidence_ids, by, reason) -> str` — **修正**：旧记录填 `invalidated_at`，新记录 `valid_at`，同 topic 串联。
- `distill_step3(store, topic, llm, emb) -> Profile | None` — 提炼 3：S2 → S3 **抽象**（唯一在「下判断」的一步）。
- `age_out_profiles(store, now) -> list[str]` — **老化降级**：长期未印证**且**未被提及 → 降回 `pending`。
- `cap_profiles(store, cap) -> list[str]` — 画像总数有上限：超了**印证最少的降回待验证**。
- `run_distill_cycle(store, llm, emb, topic_cap) -> dict` — 后台提炼周期：一次跑完该跑的事。
- `review_profiles(store, llm, cap, min_days, downgrade_cap, emb) -> dict` — S3 复核（记忆整理稿 §五）：定期回头看「已立」的画像还站不站得住。
- `write_maint_trace(stats) -> None` — 体检留痕（`data/trace/整理-YYYYMMDD.jsonl`）。
- `maintenance_cycle(store, llm, emb, trigger, topic_cap) -> dict` — 体检（记忆整理稿 §四）：不靠会话边界的那次整理。

## `core/embedding.py`

语义向量服务。

| | |
|---|---|
| 层级 · Layer | L2 外部服务（向量） |
| 上游 · Upstream | config（在 `chat.build_embedding` 里读） |
| 下游 · Downstream | scene（算检索向量）、recall（算查询向量与相似度）、chat（建服务）、 memo / settings（测试连接）、distill / trend / weave / tools / salvage |
| 对外入口 · Entry points | `EmbeddingService`（`embed` / `embed_one` / `degraded`）、 `cosine` / `embedding_novelty` |
| 边界 · Boundary | **失败一律返回 None，不抛**——降级与否由调用方决定怎么兜 |

**公开符号 · public API**

- **class `EmbeddingService`** — （无 docstring）
- 　· `embed(self, texts) -> list | None` — 批量取向量；不可用或失败返回 None（调用方降级）。
- 　· `embed_one(self, text) -> list | None` — `embed` 的单条包装。**拿不到就是 None**——调用方据此走字符重叠兜底。
- `cosine(a, b) -> float` — 余弦相似度；维度不一致或任一为空返回 0。
- `embedding_novelty(query_vec, ref_vecs) -> float` — 内容新奇值 v_new = 1 − 与已有记忆的最大余弦相似度（离得越远越新奇）。

## `core/entity.py`

实体索引与消歧。

| | |
|---|---|
| 层级 · Layer | L4 写入侧（实体） |
| 上游 · Upstream | model（Scene） |
| 下游 · Downstream | distill（写入时挂索引）、recall（读取时的实体旁路）、salvage（重记时重新挂） |
| 对外入口 · Entry points | `link_entities`、`match_known_entities`、`recall_by_entities` |
| 边界 · Boundary | 不管"什么名字才算同一个"的语义判断（精确匹配在 `store.find_entity`） |

**公开符号 · public API**

- `link_entities(scene_id, entities, store) -> list[str]` — 把场景里的实体挂进索引，返回命中的 entity_id 列表。
- `match_known_entities(text, store, max_hits) -> list[str]` — 从一句话里找出**已知实体**（唤醒层 §5 的实体旁路入口）。
- `recall_by_entities(names, store, limit) -> list[Scene]` — 实体旁路检索：命中的场景直接进候选，**不经语义检索**。

## `core/llm.py`

LLM 调用——**结构化输出 + 重试 + 失败降级**。

| | |
|---|---|
| 层级 · Layer | L2 外部服务（模型） |
| 上游 · Upstream | config（endpoint / key / model / 超时） |
| 下游 · Downstream | scene / recall / distill / memo / trend / chat / dashboard —— 所有要调模型的地方 |
| 对外入口 · Entry points | `LLM`（`structured` / `chat` / `chat_with_tools` / `chat_stream` / `search`） |
| 边界 · Boundary | **绝不抛异常**——失败一律收敛成"保守默认值 + 一行日志"， 含能力位探测（不支持 json_schema → 这一进程起走 json_object） |

**公开符号 · public API**

- **class `LLM`** — OpenAI 兼容 /chat/completions 客户端（纯标准库）。
- 　· `available(self) -> bool` — —
- 　· `set_mock(self, fn)` — 注入假模型（测试与 demo 用）：fn(prompt, schema) -> dict。
- 　· `structured(self, prompt, schema, retries) -> dict` — 结构化抽取：返回**一定包含 schema.defaults 全部键**的 dict。
- 　· `chat(self, messages, max_tokens, temperature) -> str` — 纯文本调用（单轮），失败返回空串。
- 　· `chat_with_tools(self, messages, tools, max_tokens, temperature, timeout) -> dict` — 带工具的一次调用：`{"content": str, "tool_calls": [...]}`。
- 　· `chat_stream(self, messages, tools, max_tokens, temperature, timeout)` — 流式调用：yield `{"type": "reasoning"|"content"|"tool_calls"|"error", …}`。
- 　· `search_capable(self) -> bool` — 这个服务能不能搜（探测过按探测结果答，没探测过先当能）。
- 　· `search(self, query, timeout) -> dict` — 联网搜索（服务端工具）。**不抛异常**。

## `core/memo.py`

备忘录：`open_loops` 的可提醒子集。

| | |
|---|---|
| 层级 · Layer | L8 备忘录 |
| 上游 · Upstream | config、model、prompts、store（CRUD 在那边） |
| 下游 · Downstream | distill（`_write_memos` 与 `memo_cycle`）、recall（`standing_memos`）、 shortterm（`judge_hits`）、chat（`mark_raised`、到点那件的记账）、 tools（`close`，延迟 import）、dashboard（`close` / `due`） |
| 对外入口 · Entry points | `due` / `standing_memos` / `judge_hits` / `hit_candidates` / `retire_due` / `memo_cycle` |
| 边界 · Boundary | **不认识对话层**——它只回答"哪些到点了、哪些该退役了"； 注不注入、怎么说是 recall / chat 的事 |

**公开符号 · public API**

- `classify_window_kinds(store, memos, llm) -> int` — 给备忘录补「类别窗口」与「时机类别」，返回处理的条数。
- `classify_window_kind(content, llm) -> str` — 单条分类。
- `classify_group_names(store, memos, llm) -> int` — 给**还没有组名**的备忘录补「事项组」短名，返回处理的条数（2026-09-22）。
- `due(store, now) -> list` — **到点了**的备忘录（有 `due_at` 已过 / 无 `due_at` 超窗口）。
- `standing_memos(store, msg, emb, full, limit, now) -> list[dict]` — 常备备忘录：**她手上得有的那几件**（2026-09-21，设计稿 D2b / F）。
- `mark_raised(store, mid) -> bool` — 待提 → 已提。**之后不再主动提**（提过一次就够了）。
- `close(store, mid, by) -> None` — 关闭（用户给了结果）——**顺带把场景卡里的钩子标掉**（闭合回流）。
- `write_memo_trace(act, changes) -> None` — 备忘录状态改动的留痕（`data/trace/备忘-YYYYMMDD.jsonl`）。
- `hit_candidates(store, msg, emb, now) -> list` — 命中粗筛：这条消息**可能碰到了哪几件**挂着的事（2026-10-05）。
- `judge_hits(store, msg, llm, now, emb) -> dict` — 命中判定：这一句碰到了哪几件挂着的事、各是什么动作（2026-10-05）。
- `retire_due(store, now) -> list[dict]` — **超期退役**（2026-10-05，取代「窗口 × 3 / 90 天硬顶」）：提的窗口过完仍未提成
- `memo_cycle(store, llm, now) -> dict` — 后台备忘录维护：补分类 + 补组名 + **超期退役**（并入提炼周期跑）。

## `core/model.py`

数据模型。

| | |
|---|---|
| 层级 · Layer | L0 基础层 |
| 上游 · Upstream | 无 |
| 下游 · Downstream | store（读写行）、以及一切需要造 `Scene` / `Profile` 的层 |
| 对外入口 · Entry points | Entity / Scene / Raw / Summary / Profile / ProfileReview / Memo + 各枚举常量 + `lang_from_prefs`（`Opening` 随主动开口退役，2026-10-05） |
| 边界 · Boundary | **纯数据结构**——不写「为空就取那个」的兜底，那是逻辑层的判断 |

**公开符号 · public API**

- `lang_from_prefs(prefs) -> str` — 从 `user_prefs` 读语言；没设过、或值非法，回中文（默认不打扰）。
- **class `Entity`** — 实体索引里的一条（人物 / 宠物 / 地点 / 作品——与 `ENTITY_KINDS` 同）。
- 　· `to_row(self) -> dict` — —
- 　· `from_row(row) -> 'Entity'` — —
- **class `Scene`** — S1 场景卡——**所有字段都由七条唤醒线索反推**（存储层 §3）。
- 　· `to_row(self) -> dict` — —
- 　· `from_row(row, emb) -> 'Scene'` — —
- **class `Raw`** — S0 原文：对话原文 + 反指向它的 S1。
- **class `Summary`** — S2：同主题多个 S1 **聚合**成的叙述（可逆，可从 S1 重建）。
- 　· `to_row(self) -> dict` — —
- 　· `from_row(row) -> 'Summary'` — —
- **class `Profile`** — S3 画像：跨主题的模式陈述（**唯一在「下判断」的一层**）。
- 　· `to_row(self) -> dict` — —
- 　· `from_row(row) -> 'Profile'` — —
- **class `ProfileReview`** — 一次复核的打分**记录**（每次复核一行，不覆盖历史）。
- 　· `to_row(self) -> dict` — —
- 　· `from_row(row) -> 'ProfileReview'` — —
- **class `Memo`** — 备忘录：`open_loops` 的可提醒子集（短期记忆 §5）。
- 　· `to_row(self) -> dict` — —
- 　· `from_row(row) -> 'Memo'` — —

## `core/net.py`

出站网络适配：代理从环境（含系统设置）来，但 **loopback 永远直连**。

| | |
|---|---|
| 层级 · Layer | L2 外部服务（出站网络）——所有网络调用的公共底座 |
| 上游 · Upstream | 标准库 urllib（无项目内依赖） |
| 下游 · Downstream | core/__init__（import 即装）、settings（保存时重装）、embedding（走 `open()`）、 webfetch（`proxied`）；llm / 语音客户端经全局 opener 受益（不 import 它） |
| 对外入口 · Entry points | `install()` / `open()` / `proxied()` / `is_loopback()` |
| 边界 · Boundary | **只做"走哪条路"**——超时 / 重试 / 证书 / 上限各自在调用方， 这里不替它们做主 |

**公开符号 · public API**

- `is_loopback(host) -> bool` — 本机地址吗——loopback（`127.0.0.0/8`、`::1`）与 `0.0.0.0`，含 `localhost`。
- `opener() -> urllib.request.OpenerDirector` — 带规则的 opener（懒建单例——建一次装一次，重复装的是同一个对象）。
- `install() -> None` — 重建并装上**全局** opener（重装 = **重读代理环境**）。
- `open(req, timeout, context)` — 显式入口：带规则的 opener。
- `proxied(url) -> bool` — 这次请求会不会被代理接管——判据与**刚构造的** `ProxyHandler` 一致。

## `core/persona.py`

人格文件（`self/personas/*.md`）：读 / 列 / 写 / 停用。

| | |
|---|---|
| 层级 · Layer | L3 提示词（与 prompts 同层；prompts 管"拼"，这里管"存"） |
| 上游 · Upstream | config / store（只要一个 now_str） |
| 下游 · Downstream | chat（转发）、salvage（转发）、dashboard（写侧入口） |
| 对外入口 · Entry points | load / names / info / list_info / create / save / set_enabled |
| 边界 · Boundary | 不碰声音、不碰记忆、不做"人格 = 新记忆库"那套隔离 |

**公开符号 · public API**

- `check_id(name) -> str` — id 合不合格。返回错误说明；空串 = 通过。
- `write_persona_trace(act, name, before, note) -> None` — 人格改动的留痕（`data/trace/人格-YYYYMMDD.jsonl`）。
- `names(include_disabled) -> list[str]` — 人格目录里有哪些——界面的选项列表用（扫目录，加文件即生效）。
- `load(name) -> str` — 读 `<id>.md`——当前人格的提示词。**每轮现读**：切了 / 改完，下一句就换。
- `info(name) -> dict` — 一个人格的档案：显示名 / 字数 / 在不在用 / 是不是出厂的。
- `list_info() -> list[dict]` — 在用的排前面、停用的排后面（各按 id 排序）——设置页一屏看完有哪些人。
- `text(name) -> str` — 原样读一份人格正文（设置页的编辑框用）。
- `create(name, text) -> dict` — 新建一个人格。`text` 空 → 给 `TEMPLATE`（新增即可用，改由他）。
- `save(name, text) -> dict` — 整份写回（设置页的"保存"）。**只在真改了时留痕**——点开看一眼再保存，
- `set_enabled(name, on) -> dict` — 停用 / 恢复（`<id>.md` ↔ `<id>.md.off`）。

## `core/prompts.py`

所有 LLM 提示词与输出 schema——**集中管理，不散落业务逻辑**。

| | |
|---|---|
| 层级 · Layer | L3 提示词层 |
| 上游 · Upstream | model（枚举常量） |
| 下游 · Downstream | scene / recall / distill / memo / trend / chat（所有要调模型的地方） |
| 对外入口 · Entry points | `<任务>_prompt()` 一组 + 同名的 `*_SCHEMA` 一组 + `build_system_prompt` |
| 边界 · Boundary | **不调 LLM、不碰 store**——只生产"要说的话"和"要回来的形状" |

**公开符号 · public API**

- `time_context(now) -> str` — 当前时间的完整表述——**注入在提示词末尾**。
- `rel_day(when, now) -> str` — 「刚刚 / 17 分钟前 / 3 小时前 / 一天前 / …」——给绝对时间补一个**算好的**相对时间。
- `rel_stamp(when, now) -> str` — 时间戳带上相对日：`（今天）2026-09-20 06:24`——**所有出口共用这一种拼法**。
- `required_keys(schema) -> str` — 从 schema 生成「必填键」清单——**不手写第二遍**。
- `extract_scene_prompt(conversation, topic_candidates, now) -> str` — 场景卡抽取的 prompt。
- `cue_prompt(msg) -> str` — 线索判定（C2/C3/C5/C7）→ `CUE_SCHEMA`。
- `topic_prompt(statement, subject, candidates) -> str` — topic「新建 vs 复用」的判断 prompt（存储层 §3）。
- `summary_prompt(topic, scenes, candidates) -> str` — S1→S2 **聚合**（`distill_step2` 用）→ `SUMMARY_SCHEMA`。
- `profile_prompt(topic, summaries, scenes, candidates) -> str` — S2→S3 **抽象**（`distill_step3` 用）→ `PROFILE_SCHEMA`。
- `revision_prompt(topic, statement, new_scenes) -> str` — 新证据 vs 已有画像（突变分支）→ `REVISION_SCHEMA`。
- `drift_prompt(topic, statement, early_scenes, recent_scenes) -> str` — 渐变判定（`trend.detect_drift` 用）→ `REVISION_SCHEMA`。
- `review_prompt(items) -> str` — 给一批「已立」画像做定期复核 → `REVIEW_SCHEMA`。
- `memo_class_prompt(contents) -> str` — 给一批未了结的事分类（类别 + 时机）→ `MEMO_CLASS_SCHEMA`。
- `memo_group_prompt(contents, existing) -> str` — 给一批备忘录补事项组名（`memo.classify_group_names` 用）→ `MEMO_GROUP_SCHEMA`。
- `memo_hit_prompt(msg, memos) -> str` — 这句话碰到了哪几件未了结的事、各是什么动作（`memo.judge_hits` 用）。
- `scene_line(s, now) -> str` — 一条场景渲染成一行：`- S1-0008（21 小时前）2026-09-20 09:50：标题 —— 摘要`。
- `render_memory_block(recall, summaries, now) -> str` — 把唤醒结果渲染成「记忆内容」块（注入 system prompt）。
- `render_facts_block(facts) -> str` — 基础档案块：**用户明说的事实**——排在画像前面，因为它是理解一切的前提。
- `render_persona_note(persona, personas) -> str` — 署名说明（2026-09-25）：**多个人格共存时**才注入——人格可中途切换，
- `render_think_block(lang) -> str` — 思维语言：**中文模式一句，英文模式零注入**（理由见 `THINK_IN_CHINESE`）。
- `render_lang_block(lang) -> str` — 语言段（2026-09-17）：**只在英语模式注入**——中文是默认，零注入。
- `render_tools_block(names) -> str` — 「该不该用工具」的分寸块。`names` 是这一轮实际可用的动作名。
- `build_system_prompt(charter, recall, summaries, now, facts, tools, lang, persona, personas) -> str` — 拼最终的系统提示词：**安全底线 + 尊重 + 诚实（固定）** + 人格 + 署名 + **关于用户（事实）** + 工具分寸 + 记忆内容 + 当前时间 +〔思维语言〕。
- `fixed_blocks_report(charter, persona, personas) -> dict` — 常驻块的**体量体检**：测出来的字符数 + 预算上限。

## `core/recall.py`

唤醒引擎——「该不该翻记忆、翻什么、翻多深」。

| | |
|---|---|
| 层级 · Layer | L7 读取侧（唤醒） |
| 上游 · Upstream | config、embedding、entity（旁路）、model、prompts（线索判定）、store |
| 下游 · Downstream | chat（每轮唯一入口）、dashboard（把线索和抑制名单画出来）、 demo / run_experiment（直接调）、scene / distill / memo / salvage / tools （延迟 import `char_overlap` / `core_score` / `bump_counters`） |
| 对外入口 · Entry points | `recall_for_message`（一个函数走完全程）/ `compute_cues` / `recall` / `core_score` / `mark_mentioned` / `cue_hits`（给前端算高亮） / `cue_votes`（票制：强 2 / 弱 1，2026-10-05） |
| 边界 · Boundary | **读取侧**——唯一会写的是两个计数器，且由调用方判断该不该记 |

**公开符号 · public API**

- `char_overlap(a, b) -> float` — 字符 bigram Jaccard 相似度——embedding 不可用时的兜底。
- `core_score(scene) -> float` — 核心度（存储层 §4）——决定**同一动作内**谁优先、谁先进冷层。
- `compute_cues(msg, store, emb, llm) -> dict` — 算查询侧七条线索（C1–C7）+ **两条旁路**（实体 / 字面）。
- `r0_should_recall(msg, entities) -> bool` — R0 轻量感知：要不要翻记忆（所有输入都跑的地板）。
- `cue_hits(cues) -> dict[str, bool]` — 逐维的「明确命中」判定——**阈值的唯一落点**。
- `cue_votes(cues) -> dict` — 这一轮的**票数**与"有没有强维"（唤醒层 §八，2026-10-05）。
- `multi_hit(cues) -> bool` — 强信号判定（唤醒层 §8）：**票数 ≥ `recall.signal_votes` 且含至少一个强维**。
- `profile_decay(p, now) -> float` — 画像的时效衰减（0–1）：半衰期 `rank.profile_recency_halflife_days`（默认 45 天）。
- `recall(cues, store, emb) -> dict` — 按线索执行唤醒动作（规则版，逐条对应唤醒层 §7 的耦合表）。
- `bump_counters(store, scene_id, by, at) -> None` — 两个指标的递增入口（**递增时机不同，别合并**）。
- `scene_mentioned(scene, texts) -> bool` — 这条场景在对话里被**真的提到**了吗（用户复述 / air 说出口都算）。
- `mark_mentioned(store, scenes, texts, at) -> list[str]` — 给「这一轮真的被提到」的场景记一次 `mention_count`（返回记了哪些）。
- `write_trace(record) -> None` — 每次唤醒落一行 JSONL 到 `data/trace/唤醒-YYYYMMDD.jsonl`。
- `asked_for_referent(msg) -> bool` — 他在追问「你刚才说的那个指什么」吗——D2b 常备栏的 `full` 开关（F 条）。
- `recall_for_message(msg, store, llm, emb) -> dict` — 高层入口：一句话进 → 唤醒结果出（含 trace）。

## `core/salvage.py`

打捞：**人主动从原文里找回一件东西**。

| | |
|---|---|
| 层级 · Layer | L9 打捞 |
| 上游 · Upstream | config、embedding（语义找）、store（读原文目录与 scene） |
| 下游 · Downstream | dashboard（「打捞」页） |
| 对外入口 · Entry points | `search_raws` / `rebuild_confirmed` / `recent_dialogue` |
| 边界 · Boundary | **不进自动唤醒**。它是翻文件，慢——这正是「S0 最难召回」的物理形状 |

**公开符号 · public API**

- `recent_dialogue(store, limit) -> list[dict]` — 最近几段对话原文（**已提取的那部分**）——重启后对话区回填用。
- `search_raws(store, query, date_from, date_to, emb, limit) -> list[dict]` — 按时间缩小范围，再在原文里找（语义优先，没有向量就用字符重叠）。
- `rebuild_confirmed(store, llm, scene_id, on_date, emb) -> dict` — 照着原文**重新记一遍**（删掉的是理解，原文还在，所以记得回来）。

## `core/scene.py`

场景切分与字段抽取。

| | |
|---|---|
| 层级 · Layer | L4 写入侧（切分与抽取） |
| 上游 · Upstream | config、embedding（距离）、model、prompts（抽字段的 prompt 与 schema） |
| 下游 · Downstream | shortterm（什么时候该切，并从这取 `Message` 契约）、 distill（它来调这一层的抽取）、salvage（照原文重记） |
| 对外入口 · Entry points | `should_cut` / `should_cut_texts`（降级版）/ `extract_scene` / `render_conversation` / `behavior_intensity` / `is_trivial` |
| 边界 · Boundary | **不落库**。它给出"这一段是一张什么卡"，写是 distill 的事 |

**公开符号 · public API**

- `render_conversation(messages, with_date) -> str` — 把消息列表渲染成 prompt 里的对话文本。
- `should_cut(new_msg_emb, window_embs, gamma, min_distance, min_window) -> bool` — 这条新消息是否开启了一个新场景。
- `should_cut_texts(new_text, window_texts, gamma, min_distance, min_window) -> bool` — **没有向量服务时**的切分判据：用字符重叠代替余弦。
- `behavior_intensity(messages, prev_mentions) -> float` — 强度 = 行为信号计数，归一化到 0–1。**第一版只用三个信号**。
- `is_pleasantry(content) -> bool` — 这条 `air_promise` 是不是其实只是客套话。
- `is_trivial(messages) -> bool` — 整段是不是「纯寒暄 / 纯应答」——**代码判，不问模型**。
- `extract_scene(messages, llm, emb_service, topic_candidates, source) -> tuple[Scene, str, list[dict], dict]` — 把一段对话抽成场景卡。

## `core/search.py`

联网搜索的**通道表**——一条通道 = 一份自描述（端点从哪来 / 请求怎么拼 / 响应怎么读）。

| | |
|---|---|
| 层级 · Layer | L2 外部服务（搜索通道）——`llm.search()` 的按图索骥处 |
| 上游 · Upstream | config（`search.channel` / `search.endpoint` / 模型） |
| 下游 · Downstream | llm（`search()` 调 `selected()` 取通道） |
| 对外入口 · Entry points | `selected()`（当前配置的通道 + 认不出的引导）· `resolve()` |
| 边界 · Boundary | **不碰网络、不碰配置写入**——纯描述 + 纯函数（好测） |

**公开符号 · public API**

- **class `Channel`** — 一条搜索通道的自描述（见模块头五件事）。
- `resolve(name) -> tuple[Channel | None, str]` — 按名字取通道。返回 `(通道, 错误说明)`；取不到时通道为 None。
- `selected() -> tuple[Channel | None, str]` — **这次该走哪条通道**（从 `search.channel` 读）。
- `looks_unsupported(err) -> bool` — 这次失败是「没有这个能力」还是「这次调用出错」？

## `core/settings.py`

外部配置的加载与保存：`config.local.json` + 环境变量 + 服务商预设。

| | |
|---|---|
| 层级 · Layer | L2 外部服务（配置） |
| 上游 · Upstream | config（默认值）、store（原子写） |
| 下游 · Downstream | dashboard（「设置」页）· 各启动入口（`apply()`） |
| 对外入口 · Entry points | `load_local()` / `save_local()` / `apply()` / `apply_preset()` / `describe()` / `local_config_path()` / `PRESETS` / `test_connection` |
| 边界 · Boundary | 读完就把结果交给 `config.CONFIG`，自己**不长期持有状态** （和 `config` 的关系是"默认值在那边、敏感值在这边"） |

**公开符号 · public API**

- `local_config_path()` — `config.local.json` 的位置。**一个出口**——读、写、删除三件事共用它。
- `load_local() -> dict` — 读 `config.local.json`；没有或坏了都返回空 dict（不抛）。
- `save_local(data) -> None` — 写回 `config.local.json`（仪表盘的「设置」页用）。**原子写 + 合并已有键**。
- `apply_preset(name) -> dict` — 把预设填进本地配置并保存，返回改动后的配置。
- `apply() -> None` — 把外部配置合进 `CONFIG`。启动时调一次即可（幂等）。
- `env_overrides(section) -> dict[str, str]` — 哪些字段正被环境变量顶着（字段名 → 环境变量名）。
- `describe(mask) -> dict` — 当前配置（给仪表盘显示用）。`mask=True` 时 key 打码。
- `test_connection(section) -> dict` — 测一下配的服务通不通（仪表盘的「测试连接」按钮）。

## `core/shortterm.py`

短期记忆：上下文窗口管理。

| | |
|---|---|
| 层级 · Layer | L6 短期记忆 |
| 上游 · Upstream | config、distill（提取就是 `distill_step1`）、scene（判寒暄与切分）、store |
| 下游 · Downstream | chat / demo（每轮结束后调 `flush_if_needed`）、dashboard（`ShortTerm`） |
| 对外入口 · Entry points | `ShortTerm`（写 `append` / 判 `flush_if_needed` / 渲染 `build_window` · `build_window_blocks` / 收尾 `end_session` · `clear_digest` / 撤销 `undo_turns` / 回执 `take_closed_memo` / 回填 `read_state`） + `estimate_tokens` |
| 边界 · Boundary | **不自己写长期记忆**——它只决定"该提取了"，写是 `distill` 的事 （例外：`append` 里那次 `memo.judge_hits`，所以那处用了延迟 import） |

**公开符号 · public API**

- `estimate_tokens(text) -> int` — 粗略估算 token 数（不引 tokenizer）。
- **class `ShortTerm`** — 当前会话的上下文窗口（逐字尾部 + 压缩摘要）。
- 　· `append(self, speaker, text, ts, persona, interrupted) -> None` — 追加一条消息（user / air **都要进**——记忆的对象是「这段互动」）。
- 　· `take_closed_memo(self) -> str` — 取走「上一条消息了结的那件 memo」的编号——**读后就清**（消费语义）。
- 　· `undo_turns(self, turns, expect) -> dict` — 撤掉窗口**末尾的 N 轮**（N=1 就是"重说最后一句"）——改与重新生成都用它。
- 　· `should_extract(self) -> bool` — 四条触发，任一成立就该提取。
- 　· `turns_no_cut(self) -> int` — 距上一次话题切换的轮数（一轮 ≈ 一条 user + 一条 air）。
- 　· `window_tokens(self) -> int` — 窗口当前占多少 token（预算触发的输入）。
- 　· `session_idle(self) -> bool` — 他隔了很久才回来说话 → 上一段算「会话已结束」（默认 30 分钟，`session_idle_min`）。
- 　· `end_session(self) -> None` — 显式结束会话（触发提取**最后一段**——整窗收干净，不然尾巴丢了）。
- 　· `clear_digest(self) -> None` — 把压缩摘要也清掉（「新对话」用）。
- 　· `build_window(self) -> str` — 渲染要注入的短期记忆：**最近 N 条逐字 + 更早的一行一条**。
- 　· `build_window_blocks(self) -> list[tuple[str, str]]` — 窗口拆成**可分别丢弃的两块**：`[("更早", text), ("最近", text)]`。
- 　· `gist_line(self, msg) -> str` — 一条消息降成一行：**由代码写，不叫模型写**。
- 　· `compress_and_extract(self) -> dict | None` — 窗口超限 / 话题切换 / 会话结束（含空闲）时调用：一次 LLM 调用，产出两侧。
- 　· `flush_if_needed(self) -> dict | None` — 先判预算（设置 `_budget_hit` 供 compress 决定压多少），再决定提不提取。
- 　· `read_state(path) -> dict` — 只读窗口文件，规整成 `{"session_id", "messages", "digest", "last_active"}`。

## `core/store.py`

SQLite 存储封装——**唯一的落库出口**。

| | |
|---|---|
| 层级 · Layer | L1 存储层 |
| 上游 · Upstream | config（路径与容量参数）、model（数据模型与序列化） |
| 下游 · Downstream | 除 L0 外几乎全部——它是**唯一落库出口** |
| 对外入口 · Entry points | `Store` 类；`now_str`（全项目的时间源）、`atomic_write_json`、 `append_trace`（留痕的唯一写法）、`scene_ids`（sources 里的 S1）、 `RAW_HEAD`（原文文档的小节标题判据——写与读共用，salvage 也认它） |
| 边界 · Boundary | 只管"怎么存"，不判断"该不该存"；不认识 scene/topic/profile 的业务含义 |

**公开符号 · public API**

- `atomic_write_json(path, data) -> None` — 原子替换地写一个 JSON 文件。
- `now_str() -> str` — 当前时间戳（本地时间，秒精度）。全项目统一从这里取时间。
- `append_trace(kind, record) -> None` — 往 `data/trace/<kind>-YYYYMMDD.jsonl` 追加留痕——**全项目 trace 的唯一写法**。
- `scene_ids(ids) -> list[str]` — 从一组编号里挑出 S1 的——`sources` 的下钻口径（S2 是聚合、S3 是判断，
- `layer_edit(sid) -> dict` — 某层的「改」规则（认不出前缀当 S1——**默认层兜底**而已；
- `split_topics(raw) -> list[str]` — 把「她 / 他给的主题串」切成主题列表（逗号 / 顿号 / 分号 / 空格都认）。
- `load_str_list(raw) -> list[str]` — 把一列 JSON 数组文本读成 `list[str]`（NULL / 坏值 → `[]`）。
- **class `Store`** — SQLite 封装。构造即建表；库文件损坏时**报错而不是清空**（见 `_verify_not_corrupt`）。
- 　· `conn(self)` — 当前线程的连接（**thread-local**）。
- 　· `close(self) -> None` — 关掉**当前线程**的连接（其他线程的由它们自己关，或进程退出时释放）。
- 　· `backup_daily(self, keep_days) -> Path | None` — 每天一份快照，留 keep_days 天。当天已有则跳过（幂等）。
- 　· `count(self, table, where, params) -> int` — 表行数（table 走白名单校验，防止拼 SQL 被注入）。
- 　· `add_scene(self, scene) -> str` — 写入一条 S1。若 scene.id 为空则分配新 id 并回填。
- 　· `get_scene(self, scene_id) -> Scene | None` — 取一条 S1（含向量）；没有则返回 None。
- 　· `query_scenes(self, subject, topic, since, until, valence, arousal, min_intensity, include_archived, limit) -> list[Scene]` — 按维度查场景卡（C1–C6 的存储侧比对都走这里）。
- 　· `scenes_by_entities(self, names, limit) -> list[Scene]` — 实体**旁路**：按实体名 / 别名直接定位场景（唤醒层 §5）。
- 　· `all_embeddings(self, include_archived) -> list[tuple[str, list[float]]]` — 全量加载向量——**进程内缓存**（2026-09-24）。
- 　· `bump_mention(self, scene_id, at) -> None` — `mention_count` +1（**这条场景在对话中被提及**），并更新 last_mention_at。
- 　· `bump_cited(self, scene_id, at) -> None` — `cited_by_profile` +1（**画像引用这条场景**）——重要性，**不设上限**。
- 　· `raws_dir(self) -> Path` — 原文文档目录（`data/raws/`），确保存在后返回。
- 　· `add_raw(self, raw, on_date) -> str` — 把一段原文追加进**那天**的文档，返回日期（`2026-09-13`）。
- 　· `get_raws_by_scene(self, scene_id, on_date) -> list[Raw]` — 经 scene_id 下钻原文（R5 的唯一入口，S0 没有别的检索方式）。
- 　· `raw_doc_days(self) -> int` — 有原文的文档份数（仪表盘显示用——它按天，不是按条）。
- 　· `add_summary(self, summary) -> str` — 写一条 S2；空 id / 空时间戳就地补。返回 id。
- 　· `get_summary(self, sid) -> Summary | None` — 取一条 S2（**含归档的**）。没有则返回 None——同 `get_scene` / `get_profile`。
- 　· `summaries_by_topic(self, topic, include_archived) -> list[Summary]` — 某个 topic 下的 S2（提炼 3 做抽象时的原料）。
- 　· `hot_summaries(self, n) -> list[Summary]` — 热层「活跃 S2」：最近创建的前 n 条；不传 n 才用 `capacity.hot_s2_n` 兜底。
- 　· `archive_s2(self, cap) -> int` — S2 归档：超上限时最旧的先走（同 S1，只标记不删行）。
- 　· `set_summary_fields(self, summary_id, fields) -> dict` — 改一条 S2 的**可改字段**（白名单：叙述 / 主题，2026-09-24）——
- 　· `set_summary_text(self, summary_id, text) -> bool` — 改一条 S2 的叙述——`set_summary_fields` 的单字段简写（旧调用点与测试沿用）。
- 　· `archive_summary(self, summary_id) -> bool` — 把一条 S2 放进冷层（**人的动作**，2026-09-24）——她不召回它，数据全在、可取消。
- 　· `unarchive_summary(self, summary_id) -> bool` — 取消归档（**人的动作**）——"后悔"的出口。
- 　· `delete_summary(self, summary_id) -> dict` — 真删一条 S2（**人的处置**，2026-09-24）——**只删一行**（边表已退役）。
- 　· `add_profile(self, p) -> str` — 写一条 S3；空 id 与三个时间戳就地补（都退到 `created_at`）。
- 　· `current_profiles(self, status) -> list[Profile]` — 当前有效的画像。
- 　· `get_profile(self, pid) -> Profile | None` — 取一条画像（**含已失效的**）。没有则返回 None。
- 　· `invalidate_profile(self, pid, at, by, reason) -> None` — **修正**：填 `invalidated_at`（旧记录进历史，不删）。
- 　· `downgrade_profile(self, pid, at) -> None` — **老化降级**：established → pending，**不填 `invalidated_at`**。
- 　· `touch_profile(self, pid, at) -> None` — 更新 `last_support_at`（被印证 / 被提及都算）——老化判据的另一半。
- 　· `set_profile_reviewed(self, pid, at) -> None` — 记一次「复核过了」——**只服务于复核间隔防抖**。
- 　· `add_profile_review(self, pid, verdict, reason) -> str` — 写一条复核记录（每次复核一行，不覆盖历史）。返回 id。
- 　· `unhandled_reviews(self) -> list[ProfileReview]` — 还没被人处理的提议（`wrong` 且 `handled = 0`），新的在前。
- 　· `mark_review_handled(self, rid) -> int` — 标记一条提议已处理（作废 / 留着）——返回改动条数（重复标记为 0）。
- 　· `profile_history(self, topic) -> list[Profile]` — 同一 topic 的版本序列（按 valid_at 排序）——双时间戳的「轨迹」用法。
- 　· `current_profile_by_topic(self, topic) -> Profile | None` — 该 topic 当前有效的那条画像（无效的进历史，不算「当前」）。
- 　· `set_profile_status(self, pid, status, evidence) -> None` — 改画像状态（pending ↔ established）。**不动 `invalidated_at`**——
- 　· `set_profile_sources(self, pid, sources, evidence, pack) -> None` — 更新画像的可追溯来源（追加印证证据时用）。
- 　· `profiles_citing(self, scene_id, only_current) -> list[Profile]` — 反查：哪些画像的 `sources` 引用了这条场景。
- 　· `list_topics(self, subject) -> list[str]` — 已有 topic 清单（候选来源：场景写入与 `resolve_topic` 都要用）。
- 　· `all_topics(self) -> list[str]` — 三层**全部**主题（含附加主题、含 S2 独有的）——排序后返回。
- 　· `summaries_with_topic(self, topic, include_archived) -> list[Summary]` — 按主题找 S2（子串匹配，认主主题 + 附加主题）。
- 　· `profiles_with_topic(self, topic, only_current) -> list[Profile]` — 按主题找画像（子串匹配，认主主题 + 附加主题；默认只找当前有效的）。
- 　· `scenes_with_topic(self, topic, include_archived, limit) -> list[Scene]` — 按主题找场景（S1 是**单主题**——只有主主题，没有多标签）——子串匹配。
- 　· `all_user_facts(self) -> list[dict]` — 全部档案（按 key 排序），带 `source` 与出处。
- 　· `set_user_fact(self, key, value, source, note) -> None` — 写一条档案。**空值 = 删除这条**——档案里没有"空"这个状态。
- 　· `save_user_facts(self, facts, note) -> int` — 批量写（界面保存用）。返回写入条数。
- 　· `get_pref(self, key, default) -> str` — 读一条偏好。没存过给 `default`（**默认给最少的加工**，保守侧）。
- 　· `set_pref(self, key, value) -> None` — 写一条偏好。**只有人能改它**——系统不自动重置（见建表注释）。
- 　· `all_prefs(self) -> dict` — 全部用户偏好（`key → value`，**都是字符串**）。
- 　· `get_meta(self, key, default) -> str` — 读一条系统状态（体检时刻这类）。没存过给 `default`。
- 　· `set_meta(self, key, value) -> None` — 写一条系统状态。与 `set_pref` 分开：那边装「他想要什么」（只有人能改），
- 　· `top_entities(self, limit) -> list[dict]` — 出现次数最多的实体（「用户反复提到的那些人和地方」）。
- 　· `merge_topics(self, from_topic, to_topic) -> int` — topic 合并（「拿不准就新建」的补救口）：把裂开的两个序列接回去。
- 　· `find_entity(self, name) -> Entity | None` — 精确匹配 name / aliases（大小写不敏感）。
- 　· `add_entity(self, name, kind, aliases) -> str` — 建一个新实体并返回 id。**不做去重**——找旧的是 `find_entity` 的事。
- 　· `all_entities(self) -> list[Entity]` — 全部实体（**全量**）——实体旁路每轮都拿它做匹配，几百条的量级，不需要索引。
- 　· `entities_of_scene(self, scene_id) -> list[str]` — 一条场景涉及哪些实体（名字列表）。
- 　· `link_scene_entity(self, scene_id, entity_id, relation) -> None` — 挂一条「场景 ↔ 实体」关系（同一对重复挂时**更新关系**、不重复插）。
- 　· `get_entity(self, eid) -> Entity | None` — 按编号取一个实体；没有则返回 None。
- 　· `merge_entities(self, from_id, to_id) -> dict` — 把两个实体合成一个（「两个小明其实是同一个人」的人工补救口）。
- 　· `add_memo(self, memo) -> str` — 写一条备忘录；空 id / 空时间戳就地补，返回 id。
- 　· `get_memo(self, mid) -> Memo | None` — 取一条备忘录（**任意状态**）；没有则返回 None。
- 　· `memos_by_scene(self, scene_id) -> list[Memo]` — 这条场景带出来的备忘录（任意状态）。
- 　· `open_memos(self) -> list[Memo]` — 所有没关闭的备忘录（pending = 待提，raised = 已提过一次）。
- 　· `due_memos(self, now) -> list[Memo]` — **到期待提**的备忘录。
- 　· `close_open_loop(self, scene_id, content, at, loop_id) -> bool` — 把某张卡里对应的那条 `open_loops` 标成**已闭合**——标，不删。
- 　· `retire_open_loop(self, scene_id, content, loop_id, at) -> bool` — 给钩子标 `retired_at`——**退役 ≠ 完成，所以另起一个字段**（2026-10-05）。
- 　· `mark_memo_raised(self, mid, at) -> bool` — 待提 → 已提。**提过一次就不再主动提**（用户没接话也不追）。
- 　· `close_memo(self, mid) -> None` — 已提 / 待提 → 关闭（用户给了结果，或超期放弃）。
- 　· `set_memo_class(self, mid, kind_class, window_days) -> None` — 补写无具体时间备忘条的分类与窗口（写入时算好，供到期判定用）。
- 　· `set_memo_timing(self, mid, timing) -> None` — 补写时机类别（soon / later）——**只认 `soon`**：身体状况不等常规窗口
- 　· `set_memo_group(self, mid, group_name) -> None` — 补写事项组名（同一件事的多步共用一个短名，2026-09-22）。
- 　· `update_memo_content(self, mid, content, due_at) -> bool` — 内容变更（`memo.judge_hits` 的 `update` 动作，2026-10-05）——**回待提**。
- 　· `set_memo_scene(self, mid, scene_id) -> None` — 把一条 memo 的 `scene_id` 指到另一张卡（2026-10-05）。
- 　· `set_profile_evidence_at(self, pid, mapping) -> bool` — 写「引用起点时间」（id → 时刻，2026-09-24）——同 `set_profile_sources` 的写法。
- 　· `protected_scene_ids(self, now) -> set[str]` — 当前**不能归档**的场景 id 集合。
- 　· `archive_s1(self, cap, score_fn, now) -> int` — S1 归档：超上限时把**核心度最低的**先送进冷层。
- 　· `set_scene_fields(self, scene_id, fields) -> dict` — 改场景的**可改字段**（白名单，2026-09-24）——**人的纠正**的一条口。
- 　· `set_scene_text(self, scene_id, text) -> bool` — 改一条场景的摘要——`set_scene_fields` 的单字段简写（旧调用点与测试沿用）。
- 　· `set_summary_topics(self, summary_id, topics) -> dict` — 改一条 S2 的**主题标签**（1-3 个，第一个 = 主主题）——原地改
- 　· `set_profile_topics(self, pid, topics) -> dict` — 改一条画像的**主题标签**（1-3 个，第一个 = 主主题）——原地改，**不走修正**。
- 　· `set_profile_topic(self, pid, topic) -> dict` — 改一条画像的主题——`set_profile_topics` 的**单值简写**（改成一个主题）。
- 　· `archive_scene(self, scene_id) -> bool` — 把一条场景放进冷层（`archived=1`）——**人的动作**（工具箱稿 §3.4 的中间档）。
- 　· `unarchive_scene(self, scene_id) -> bool` — 取消归档（**人的动作**）——"后悔没删干净"和"后悔删了"都该有出口。
- 　· `exists(self, sid) -> bool` — 这个编号在不在（跨三层 + 备忘）——**读端过滤"引用"用**（2026-09-24）。
- 　· `delete_scene(self, scene_id) -> dict` — 真删一条场景：**删行 + 实体链接**（不留快照、不留痕，2026-09-23）。
- 　· `delete_profile(self, pid) -> dict` — 真删一条画像：**删行 + 清它的体检提议**（**人的否决**，2026-09-24）。
- 　· `topic_neighbors(self, scene_id) -> list[str]` — 同 topic 里**时间上紧邻**的两条（前一条 / 后一条）——R1 扩散的派生依据。
- 　· `archive_profile(self, pid) -> bool` — 把一条画像放进冷层（**人的动作**，2026-09-24）——"别再让它影响你，但留着"。
- 　· `unarchive_profile(self, pid) -> bool` — 取消归档（**人的动作**）——只清"归档"那一类失效，修正的历史不动。
- 　· `archive_all(self, score_fn) -> dict` — 按配置跑一遍三个归档（后台维护任务的入口）。

## `core/tools.py`

工具箱：她的手。

| | |
|---|---|
| 层级 · Layer | L9 工具箱（她的手） |
| 上游 · Upstream | config、embedding（语义排序）、store（读） |
| 下游 · Downstream | chat（唯一的执行方：一个工具循环里逐次调 `execute`） |
| 对外入口 · Entry points | `TOOLS`（工具表）/ `openai_tools` / `tool_names` / `execute` |
| 边界 · Boundary | 提议类动作（改 / 删）**一个字都不改**，只登记进 `ctx["proposals"]`； 真执行走 `weave` 那几个 `*_confirmed`（和界面按钮同一个动作） |

**公开符号 · public API**

- `openai_tools(names) -> list[dict]` — 转成 OpenAI 的 `tools` 参数格式。
- `tool_names() -> list[str]` — 所有动作的名字（`chat._tool_names` 会按这一轮的情况再筛一遍）。
- `execute(store, name, args, ctx) -> dict` — 执行一次工具调用。**永远不抛异常**（失败返回 `{"ok": False, detail}`）。

## `core/trend.py`

渐变检测——「不滞留」里最难的那一半。

| | |
|---|---|
| 层级 · Layer | L9 渐变检测 |
| 上游 · Upstream | config、embedding（余弦）、prompts（`drift_prompt`）、store |
| 下游 · Downstream | distill（`run_distill_cycle` 在老化之后带它跑一轮） |
| 对外入口 · Entry points | `topic_drift`（只给数）/ `detect_drift`（带着 LLM 判断）/ `split_by_time` |
| 边界 · Boundary | **发现 + 判；修正调用 `distill.revise_profile` 完成**——两条路共用 同一个修订函数，别在这里自己再写一遍改画像的逻辑 |

**公开符号 · public API**

- `split_by_time(scenes, ratio) -> tuple[list, list]` — 按时间把场景切成前 / 后两段。
- `vector_drift(early_vecs, recent_vecs) -> float | None` — 语义中心的漂移度 = 1 − 余弦（0 = 没变，越大越远）。
- `emotion_shift(early_scenes, recent_scenes) -> dict` — 情绪的移动量：早期均值 → 近期均值。
- `topic_drift(store, topic) -> dict` — 算一个 topic 的漂移指标（**只给数，不做判断**）。
- `detect_drift(store, llm) -> dict` — 扫所有 topic，对漂移的做一次判定；判到「确实变了」就修正画像。

## `core/weave.py`

编织层：把散落的线索连起来呈现给他。

| | |
|---|---|
| 层级 · Layer | L9 编织层（呈现与可否决；原 R6） |
| 上游 · Upstream | config、embedding（主题归并要凑向量中心）、model、store |
| 下游 · Downstream | chat（工具提议后的真执行）、dashboard（镜像页 / 主题页 / 改删按钮） |
| 对外入口 · Entry points | `render_mirror` / `pattern_stats` / `merge_suggestions` / `user_reject_profile` （画像的「删」走它——三层里只有它没挂 `_confirmed` 名字） + 一组 `*_confirmed`（档案 / 语言 / 三层改·归档·取消归档 / 删〔场景 / 摘要〕 / 合并主题 / 合并实体） + 三层分派 `*_by_layer`（确认条与统一口 `/api/memory-action` 共用一份） |
| 边界 · Boundary | **只做呈现与人确认后的动作**——它不替人自动开口、也不自动合并主题； 带 `_confirmed` 的都是「界面那个按钮背后的同一个动作」 |

**公开符号 · public API**

- `user_reject_profile(store, pid) -> str` — 用户否决一条画像：**真删**（2026-09-24 定），返回处理结果。
- `render_mirror(store, topic, include_pending) -> dict` — 把「air 眼中的他」拉出来给人看（R6 的呈现动作）。
- `pattern_stats(store, min_count, limit) -> list[dict]` — 「遇到 X 情境 → 典型反应 → 结果」的统计（唤醒层 §3② 整体分析）。
- `merge_suggestions(store, emb, threshold) -> list[dict]` — 找「可能是同一个主题、却被写成两种说法」的 topic 对——**只建议，不合并**。
- `merge_topics_confirmed(store, from_topic, to_topic) -> dict` — 执行主题合并——**只由人确认后调用**（仪表盘的按钮）。
- `merge_entities_confirmed(store, from_id, to_id) -> dict` — 执行实体合并——**只由人确认后调用**（仪表盘实体页的按钮）。
- `save_facts_confirmed(store, facts, note) -> int` — 人确认后写档案 + **留痕**；返回写入条数。
- `save_lang_confirmed(store, lang) -> str` — 写语言（中 / 英）+ 留痕，返回生效后的值。
- `update_scene_confirmed(store, scene_id, text, **fields) -> dict` — 改一条场景的**可改字段**（2026-09-24 起不只摘要）+ 留痕。
- `archive_scene_confirmed(store, scene_id) -> dict` — 归档一条场景——**人的动作**（确认条的"归档冷存"那一档，工具箱稿 §3.4）。
- `unarchive_scene_confirmed(store, scene_id) -> dict` — 取消归档（**人的动作**）——"后悔"的出口（同归档，不留痕）。
- `delete_scene_confirmed(store, scene_id) -> dict` — 真删一条场景（**人的动作**：删行；**不留快照、不留痕**）。
- `update_summary_confirmed(store, summary_id, text, field) -> dict` — 改一条 S2 的**可改字段**（叙述 / 主题，2026-09-24）+ 留痕——
- `delete_summary_confirmed(store, summary_id) -> dict` — 真删一条 S2（**人的处置**，2026-09-24）：**只删一行**——**不留痕、不摘引用**。
- `archive_summary_confirmed(store, summary_id) -> dict` — 归档一条 S2（**人的动作**）——她不召回它，数据全在、可取消。不留痕（可逆、查库即真相）。
- `unarchive_summary_confirmed(store, summary_id) -> dict` — 取消归档（**人的动作**）——"后悔"的出口。
- `update_profile_confirmed(store, pid, text, field) -> dict` — 改一条画像的**可改字段**（**人的动作**，2026-09-24）：
- `archive_profile_confirmed(store, pid) -> dict` — 归档一条画像（**人的动作**）——不召回它，数据全在、可取消；**旧版历史不动**。
- `unarchive_profile_confirmed(store, pid) -> dict` — 取消归档（**人的动作**）——只清"归档"那一类失效，修正的历史不动。
- `delete_many_confirmed(store, scene_ids) -> dict` — 一次删多条（**人的动作**）：逐条删行（**不留快照、不留痕、不摘引用**）。
- `ids_by_layer(ids) -> dict[str, list[str]]` — 编号按层分组（认不出前缀的落在分组之外，由调用方报错）。
- `delete_by_layer(store, ids) -> dict` — 三层分派删除（**先全查再动手**：有一个找不到就一条都不删）。
- `archive_by_layer(store, ids) -> dict` — 三层分派归档——不召回、数据在、可取消（与删除的差别只剩"行在不在"）。
- `unarchive_by_layer(store, ids) -> dict` — 三层分派取消归档（"后悔了"的出口）——与归档对称，三层同一套。
- `update_by_layer(store, sid, text, field) -> dict` — 三层分派改：S1 / S2 原地改留旧值；S3 的**陈述**走修正（主题是原地改）。

## `core/webfetch.py`

取网页（`web` 的 url 分支）——把「她打不开的链接」变成能读的文本。

| | |
|---|---|
| 层级 · Layer | L9 工具箱（她的手）——`web` 的 url 分支实现；动作表在 `tools.py` |
| 上游 · Upstream | config（超时 / 上限 / UA） |
| 下游 · Downstream | tools（`_web_fetch` 薄封装） |
| 对外入口 · Entry points | `fetch()`（一个函数走完全程）+ `to_text()`（HTML → 文本，纯函数） |
| 边界 · Boundary | **不改记忆、不进 trace**——抓来的只是这一轮说话的燃料 （同 `web_search`：来源不能是「网上说的」） |

**公开符号 · public API**

- `to_text(html) -> str` — HTML → 可读文本，纯函数。
- `fetch(url, *, opener, timeout, max_bytes, max_chars, max_url_chars) -> dict` — 抓一个页面。**不抛异常**——失败给 `error`。

