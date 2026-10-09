# AGENTS.md
This file provides guidance to AI coding agents (CodeBuddy and others) working with code in this repository.
（文件名用 `AGENTS.md` 是刻意的：它是跨工具的通用约定，换哪个助手来读都认。）

聊天版**关系记忆系统**：记住的是「这段互动」，并在该想起的时候想起它。

## 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 运行时 | Python 3.11+（开发 3.14） | 不打包、不构建、无 linter，直接跑脚本；**别用 3.12+ 专属语法**（`prepublish_check.py` 按 3.11 查） |
| 依赖 | **零第三方依赖** | `requirements.txt` 通篇是注释：存储 `sqlite3`、HTTP `http.server`、LLM / 向量 `urllib`（只需"POST 一个 JSON"）。唯一可选 `sentence-transformers`，仅 embedding 配成 `local://<模型名>` 时懒加载 |
| 存储 | SQLite 单文件 + 按天 Markdown 原文 + JSONL 留痕 | 全在 `data/` 下，运行期数据，不进版本库 |
| 服务端 | 标准库 `ThreadingHTTPServer`（`core/dashboard.py`） | 只读请求不排队；凡碰会话状态就加锁串行 |
| 外部服务 | OpenAI 兼容 `/chat/completions` + embedding；搜索走 `core/search.py` 通道表 | 端点 / key / 模型走配置优先级链（见「常用命令」末段） |
| 前端 | 原生 HTML / CSS / JS，**单文件** `web/index.html` | 零依赖、零构建，中 / 英词典内嵌（`I18N` + `t()`） |
| 语音 | 可选外挂，只留 HTTP 契约（`tts/README.md`） | 实现 / 权重 / 音色不进库；没接上不影响其它功能 |
| 测试 / CI | 标准库 `unittest`，**离线**；GitHub Actions | Windows py3.11 / 3.13 必过，ubuntu py3.13 标 experimental |

## 常用命令

| 命令 | 说明 |
|---|---|
| `python -m core.dashboard` | 主入口：本地记忆实验台（`127.0.0.1`）。Windows 双击 `run.cmd` 等价；`启动.vbs` = 仪表盘 + 语音、不开黑框 |
| `python -m unittest discover -s tests` | 全量测试（629 个，约 40 秒，离线：mock LLM + mock 向量）。`run.cmd test` 是别名，CI 跑同一条 |
| `python -m unittest discover -s tests -p "test_phase5.py"` | 只跑一个测试文件 |
| `python -m unittest discover -s tests -k FrameConventionTest` | 只跑名字匹配的用例（`-k` 按类名 / 方法名过滤） |
| `python demo.py` / `demo_distill.py` / `demo_memo.py` / `demo_trend.py` | 阶段 1–4 演示：假模型走同一条代码路径，不需要 key；`--fresh` 清演示库，`--real` 用真实服务 |
| `python run_experiment.py --script <脚本.json> --fresh` | 对话回放 + 报告。**脚本素材自备**（在 `private/`，仓库不带，格式见该文件头部） |
| `python tools/module_index.py` | 重新生成 `docs/modules.md`；`--check` 只对账、不改文件（CI 用） |
| `python tools/prepublish_check.py --history` | 发布前自检：密钥（含全部历史）/ 个人路径 / PII / i18n 覆盖 / 3.11 语法 / README 测试数。有 error 退 1；`--all` 连未跟踪文件一起扫 |

用真实服务：设 `AIR_LINK_LLM_*` / `AIR_LINK_EMBEDDING_*`，或填 `config.local.json`（实验台「设置」页写的是同一份文件）。优先级：默认 < `AIR2_*` < `config.local.json` < `AIR_LINK_*`；密钥输入框留空 = 不改。**DeepSeek 不提供 embedding 接口**，向量得另配 Ollama / OpenAI。

## 目录结构

```
air-link-01/
├── core/               后端全部（24 个模块，按 L0–L11 分层——见「架构」；逐模块接口见 docs/modules.md）
├── web/index.html      全部前端（单文件）
├── self/personas/      人格提示词（air / mia / xina）——**文件是真源**，界面只是编辑器
├── tests/              按阶段 test_phase1–7.py + 专题（net / persona / salvage / report / prompt_settings / tts_split）
├── tools/              prepublish_check.py · module_index.py · i18n_allow.txt · README.md
├── docs/modules.md     接口总表——**生成物**（由 tools/module_index.py 生成，勿手改）
├── tts/                语音接口契约（README.md）
├── 设计与实现对照-结构.md · 设计与实现对照-核对/（按层一份）     改代码前读的对照文档
├── demo*.py · run_experiment.py                                  演示与实验回放
└── run.cmd · 启动.vbs · 开关.hta（+ 别名 start.vbs / switch.vbs）   Windows 入口
```

不进版本库：`data/` 运行期数据 · `private/` 设计记录与实验素材 · `config.local.json`（含 key）· `tts/` 的本机实现与权重 · `ref/` 第三方克隆。
运行期落点（`config.PATHS`）：`data/air_link.db` · `raws/YYYY-MM-DD.md`（S0 原文，按场景发生日）· `trace/*.jsonl` · `backups/` 每日快照 · `shortterm.json` 窗口。

## 架构

### 分层与依赖方向

L11 `dashboard` → L10 `chat` → L9 `tools` / `weave` / `salvage` / `trend` → L8 `memo` → L7 `recall` → L6 `shortterm` → L5 `distill` → L4 `scene` / `entity` → L3 `prompts` / `persona` → L2 `llm` / `embedding` / `settings` / `net` / `search` → L1 `store` → L0 `config` / `model`。

规则：**只能向上依赖**。两处刻意的例外别"修"掉：函数内延迟 import（`scene`→`recall.char_overlap`、`distill`→`recall.bump_counters`、`shortterm`→`memo.judge_hits`，放顶层会成环）；`dashboard` 谁都能 import。读代码顺序：`model → store → scene → shortterm → recall`——这五个读完，整条链路就有了。

### 一次回应 = 九步，她只在第 4 步出现

① 算线索（1 次轻量 LLM + 向量）→ ② 定动作（规则）→ ③ 组织注入 → ④ **说话**（1 次 LLM，流式）→ ⑤ 写窗口 → ⑥ 判断提取 → ⑦ 记账 → ⑧⑨ 收尾提取 + 后台整理。

**顺序是死的：唤醒（读）必须在写入前**——否则当前这句先混进历史，既重复注入、又让线索判断拿到"未来"。边界画在「有没有确定答案」：算相似度 / 排序 / 比阈值归代码，语义判断（是不是同一件事、这句什么情绪）归她。全链路展开见 `设计与实现对照-结构.md` §一。

### 记忆四级与读写两条路径

S0 原话（按天文件）→ S1 场景卡 → S2 主题摘要 → S3 画像，抽象度越高越不可逆，所以分级存；`profiles`（她的推断，要印证、会老化）与 `user_facts`（用户明说的事实，不印证、不老化）**分开存**。

- **读**（`recall.py`）：七条线索 C1–C7（每轮只一次轻量 LLM 判 C2/C3/C5/C7；C1/C4 用向量）+ 两条旁路（实体、字面罕见词）→ 动作 R0–R5 → 四键排序（票数 → 证据层级 → 新鲜度 → 核心度）取前 N，其余进抑制名单。「不相关」必须挡在门口，否则 R2 就成了"把最近 N 条全捞出来"。
- **写**（`distill.py`）：四条触发（话题切换为主、超轮数、超 token 预算、会话结束）+ 会话之间体检兜底；**压缩 = 提取**（一次产出回窗口的摘要与进库的场景卡）。S1→S2 是聚合（可逆），S2→S3 是抽象（唯一下判断的一层，四道关：≥3 次印证且同情境 / `sources` 必填 / 跨来源计数 / 长期不印证降回 `pending`）；修正 = 旧记录填 `invalidated_at` + 新记录另起，**人的否决 = 真删**。
- **未了结**：到点 → 进注入（一次一件，进了就算提过）→ 命中判定（变更 / 完结）→ 超期退役。

### 存储、留痕与观测

`store.py` 是**唯一落库出口**（SQLite；无 `raws` / `edges` / `openings` 表——原文按天存文件，关系全派生：引用 = `sources`、相邻 = 主题 + 时间）。

`data/trace/*.jsonl` 按类留痕。**"应该发生但没发生"是这类系统最难查的故障**，所以跳过 / 抑制 / 未收敛都要能答出**卡在哪一步、差多少**（`recall` 的 `suppressed_detail` / `stage`、`run_distill_cycle` 的 `blocked`）。仪表盘就是把留痕画出来。

### 三条硬约束（所有路径都要认）

1. **数据不丢**：任何路径（崩溃 / 大模型失败 / 字段缺失）都不静默删除已写入的场景；库损坏 → 备份 + 报错、拒绝重建；归档只置标记（可逆）、老化只降状态；**真删只有人能发起**。
2. **降级不崩**：LLM 失败给保守默认值、向量不可用走字符重叠——降级会变糙，但不该变哑（**阈值跟着降级走**，否则降级就是失忆）。
3. **S0 不进自动唤醒**：原文没有索引，自动唤醒要原文只能经场景编号下钻（`salvage` 打捞是唯一例外，且只由人发起）。

## 编码规范

**注释只写「为什么」**（最硬的一条）：做了什么一行代码就摆在那儿；被排除掉的更简单写法、以及它为什么不行，才是半年后唯一想知道而代码答不出来的东西。

- **模块头**：先说这个模块为什么存在 + 关键取舍 + 边界（什么**不该**放这里），紧跟固定格式的「模块速查」块（`层级 / 上游 / 下游 / 对外入口 / 边界`，格式照抄邻近模块）。
- **函数**：一句话契约，再讲非显然处（返回 `None` / 空意味着什么、为什么这里保守）。
- **分支**：凡「本可以发生但不发生」的（跳过 / 拒绝 / 没提 / 放弃）都要出声——`print` 或落 trace。
- **不写**：签名里已有的类型、复述函数体的注释、日期 + 人名的落款（那是 `git blame` 的事）。

**数字与配置**：可调参数、阈值、预算、条数**只写 `core/config.py`**（标 `*` = 还没实测标定的初值；改它要写理由）。业务代码里不写数字，前端也不抄一份（经 `/api/state` 下发）。路径基于 `config.ROOT` / `abspath()` 解析；时间取 `store.now_str()`，时间戳渲染共用 `prompts.rel_stamp()`。

**分层与边界**：`dashboard.py` 的处理函数只做"收参数 → 调 → 打包响应"，**不放业务逻辑**；`store.py` 只管"怎么存"、不判断"该不该存"；`prompts.py` 不调 LLM、不碰 store。

**用词各归各位**（同一件事只有一种叫法）：面向人一律说**「备忘录」**；`Scene.open_loops` 只指卡上字段；常备注入那一路叫 `standing_memos`；提炼的未收敛名单叫 `blocked`，唤醒的"卡在哪一步"叫 `stage` / `suppressed_detail`。

**语言与文案**：中文是正文（英文只在标题与字段名上）。`web/index.html` 里界面文案**一律走 `t()`**，否则英文模式会露中文；确认不是缺口的豁免写进 `tools/i18n_allow.txt`，一行一条并写明理由。入口脚本（`.cmd` / `.vbs` / `.hta`）保持纯 ASCII；给 Windows 控制台写的脚本显式 `sys.stdout.reconfigure(encoding="utf-8")`。

**失败与降级**：底层服务契约是**绝不抛异常**——`llm` / `embedding` / `tools.execute` / `webfetch.fetch` 失败一律收敛成 `None` 或 `{"ok": False, ...}` + 一行日志，由调用方决定怎么兜。**跳过也要留痕**：「没存什么、为什么」和「存了什么」一样重要。

**测试与改动流程**：测试一律离线（mock LLM + 直接构造向量），用例对着**验收标准**写、按阶段归位。动手前读 `设计与实现对照-结构.md`；改完跑 `python -m unittest discover -s tests`、`python tools/module_index.py --check`、`python tools/prepublish_check.py --history`（CI 就这三件事）。**实现与描述出现分歧时：先改描述、再改代码**；文档里的「设计稿 §x」是**文字坐标**（指 `private/` 里的本地设计记录），不是能点的链接。

## Never 规则

三条硬约束上面已列，不重复；以下是同样硬的其它红线。

**数据**

1. **不改 S0 原文、不回写已落库的内容**：记忆是**快照不是视图**——要改改的是「理解」（场景 / 摘要 / 画像，改动留痕），要作废走归档或真删。
2. **不用 `0` 顶替拿不准的语义值**（给 `null`）；**判定失败一律当「值得存」**——漏存 = 永久丢记忆，方向不能反。

**行为**

3. **不主动开口**：她的话只发生在用户说话之后。
4. **不静默不作为**：跳过 / 抑制 / 未收敛都要能答出"卡在哪一步、差多少"。
5. **不替人做不可逆的语义断言**：不自动合并主题 / 实体、不替用户决定「你需要被分析」——这类动作**只建议，执行要人点确认**。
6. **不让模型估天数、不让模型当硬判据**：可算的（印证数 / `sources` / 同情境 / 跨来源）先由代码复查，模型只判语义。
7. **不把敏感做成「不召回」**：敏感只调节说出口的**力度**，召回是理解的前提。
8. **不用代码硬保证「不迎合 / 不诊断」**：做不到，只会写出一堆无效逻辑——交给固定层（安全 / 尊重 / 诚实）与人格文件；人格文件也**不重复**这四条。

**工程**

9. **不引入第三方依赖**：主项目、测试与 `tools/` 都只用标准库。
10. **不复活已退役的设计**：`edges` 边表、主动开口 / `openings`、两套姿态（`/api/mode`）、`remember_fact`（她只读档案、不写）、`give_up_memos`；老库里的 `*_retired` 表是留底，不是待迁移。
11. **不把 loopback 走代理**：本机服务永远直连（`net.py` 是唯一代理判据），挂 VPN 时本机服务不能断。
12. **不让外挂拖累主流程**：播报失败不影响对话；退出必须**先收语音（放显存）再关仪表盘**，顺序反了就没人收那个进程。

## 文档地图

| 文件 | 什么时候读 |
|---|---|
| `设计与实现对照-结构.md` | **改代码前先读**：分层图 + 九步分解 + 模块 → 关键函数 + 运行方式 |
| `设计与实现对照-核对/`（按层一份） | 核对"某一层有没有做、有没有做歪" |
| `docs/modules.md` | 要碰某个模块：层级 / 上下游 / 对外入口 / 边界 + 公开签名（生成物，别手改） |
| `README.md` / `README.en.md` | 使用面：怎么跑 / 怎么配 / 机制 / 许可 |
