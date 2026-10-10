#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开源前自检 / 脱敏扫描（标准库，零依赖）。

    python tools/prepublish_check.py              # 扫工作区（git 跟踪的文件）
    python tools/prepublish_check.py --all        # 连未跟踪文件一起扫
    python tools/prepublish_check.py --history    # 另扫**全部提交历史里的 blob**
    python tools/prepublish_check.py --json out.json
    python tools/prepublish_check.py --write-example   # 顺手写 config.example.json

扫什么（每条都有理由，不是泛泛的"安全扫描"）：

  1 密钥与凭据        —— sk-xxx / AKIA / ghp_ / xoxb / "api_key": "非空" / 私钥块 /
                        Bearer 长串。命中即 **error**（一旦进了历史就麻烦，所以 --history 也扫）
  2 个人绝对路径      —— C:\\Users\\<真名>\\ / /Users/<真名>/ / /home/<真名>/ / 盘符工作目录。
                        命中即 **error**：源码里只该有相对路径与运行时算出来的根
  3 联系方式          —— 邮箱 / 手机号（作者痕迹）。**warn**（有的项目故意留联系方式）
  4 不该入库的实体    —— data/ private/ ref/ tts/models/ tts/refs/ config.local.json
                        *.db tts/_dl tts/_run.log 被 git 跟踪 → **error**（.gitignore 漏了）
  5 中文文件名        —— 提示用（不影响功能，但英文用户敲不出来；给 ASCII 别名/映射）
  6 大文件            —— > 512 KB 提示（别把模型权重或数据误提交进来）
  7 i18n 覆盖（双语） —— web/index.html：EN 词条 vs 读点——**死键 / 缺译 / 没走 t() 的
                        中文字面量**（后者是"英文模式下仍露中文"的真缺口）
  8 最低版本语法      —— 所有 .py 用 `ast.parse(feature_version=(3,11))` 过一遍：
                        声明"Python 3.11+"就得真能在 3.11 语法下解析
  9 README 数字       —— README 里写的测试条数 vs 实际收集到的测试条数（漂了就是文档在说谎）

历史扫描（--history）只跑第 1 条规则（密钥）：遍历 `git rev-list --objects --all` 的
每个 blob，命中就报出**它出现在哪些提交**——因为"删掉文件"不解决问题，历史里还在。

退出码：有 error → 1；只有 warn → 0（便于放进 CI 或 git hook）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_EXT = {".py", ".md", ".json", ".cmd", ".bat", ".vbs", ".hta", ".html", ".txt", ".yml", ".yaml", ".js", ".toml", ".cfg", ".ini"}

OK, WARN, ERR = "ok", "warn", "error"


@dataclass
class Finding:
    level: str
    rule: str
    where: str
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, level: str, rule: str, where: str, detail: str) -> None:
        self.findings.append(Finding(level, rule, where, detail))

    def count(self, level: str) -> int:
        return sum(1 for f in self.findings if f.level == level)


# ---------------------------------------------------------------- git helpers

def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    return out.stdout.decode("utf-8", "replace")


def tracked_files() -> list[str]:
    return [p for p in git("ls-files", "-z").split("\0") if p]


def untracked_files() -> list[str]:
    out = git("ls-files", "-z", "--others", "--exclude-standard")
    return [p for p in out.split("\0") if p]


def tracked_dirs() -> set[str]:
    return {p.split("/")[0] for p in tracked_files() if "/" in p}


# -------------------------------------------------------------------- rules

SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai/通用 sk- 前缀", re.compile(r"sk-[A-Za-z0-9_\-]{16,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[abpor]-[A-Za-z0-9-]{10,}\b")),
    ("HuggingFace token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("Bearer 长串", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{24,}")),
    ("私钥块", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("非空 api_key", re.compile(r"""["']?(?:api[_-]?key|secret|token|password)["']?\s*[:=]\s*["'][^"'\s]{8,}["']""", re.I)),
]

# 这些"非空 api_key"是合法占位/示例，不算密钥。
# 注意：豁免只看命中处 ±20 字符——`test` / `fake` / `mask` 这类字样紧挨着的
# 值按"测试夹具"放行（真 key 误粘进来时靠人眼复核，正则判不了"像不像真的"）。
PLACEHOLDER_OK = re.compile(
    r"sk-xxx|your[_-]?key|real-key|test|fake|dummy|mask|[•*…]|<[^>]+>|\.\.\.|placeholder|example|"
    r"env|os\.environ|\{[a-z_]+\}", re.I)

PATH_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Windows 用户目录带真名", re.compile(r"[A-Za-z]:\\\\?Users\\\\?(?!<|%|\*|\.\.\.)[A-Za-z0-9._\u4e00-\u9fff-]+")),
    ("macOS 用户目录带真名", re.compile(r"/Users/(?!<|%|\*|\.\.\.|shared\b)[A-Za-z0-9._-]+/")),
    ("Linux 用户目录带真名", re.compile(r"/home/(?!<|%|\*|\.\.\.)[A-Za-z0-9._-]+/")),
    ("个人盘符路径", re.compile(r"\b[DE]:\\\\?(?:air-link|air-|码头|work|dev|code)[^\"'\s]*")),
    ("中文路径（可能含真名）", re.compile(r"[A-Z]:(?:\\\\|\\)[\u4e00-\u9fff][^\"'\s]{0,40}")),
]

CONTACT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("邮箱", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
]
# 示例域名与 URL 里的 user:pass@ 不是联系方式
CONTACT_OK = re.compile(r"@(example|test|invalid|localhost)[.\w]*|\w+:\w+@|@[\w-]+\.(local|test)\b", re.I)

FORBIDDEN_TRACKED = [
    "data/", "private/", "ref/", "tts/models/", "tts/refs/", "tts/_dl/",
    "config.local.json", "tts/_run.log", "tts/_run.err", "config.local.json.",
]
FORBIDDEN_SUFFIX = (".db", ".db.corrupt", ".pyc", ".pyo", ".log", ".wav", ".pth", ".safetensors", ".gguf", ".zip", ".bundle")


def scan_text_files(rep: Report, files: list[str], *, skip_rules: set[str] = frozenset()) -> None:
    """对文本文件跑规则 1 / 2 / 3。"""
    for rel in files:
        p = ROOT / rel
        if p.suffix.lower() not in SCRIPT_EXT or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            where = f"{rel}:{lineno}"
            if "secrets" not in skip_rules:
                for name, pat in SECRET_PATTERNS:
                    for m in pat.finditer(line):
                        # 占位符豁免只看命中处前后一小段，不看整行（整行放行太宽）
                        window = line[max(0, m.start() - 20):m.end() + 20]
                        if PLACEHOLDER_OK.search(window):
                            continue
                        rep.add(ERR, "1 密钥", where, f"{name}: {m.group(0)[:60]}")
            if "paths" not in skip_rules:
                for name, pat in PATH_PATTERNS:
                    for m in pat.finditer(line):
                        rep.add(ERR, "2 个人路径", where, f"{name}: {m.group(0)[:70]}")
            if "contact" not in skip_rules:
                for name, pat in CONTACT_PATTERNS:
                    for m in pat.finditer(line):
                        window = line[max(0, m.start() - 15):m.end() + 15]
                        if CONTACT_OK.search(window):
                            continue
                        rep.add(WARN, "3 联系方式", where, f"{name}: {m.group(0)[:40]}")


def check_tracked_hygiene(rep: Report, files: list[str]) -> None:
    for rel in files:
        low = rel.lower()
        for bad in FORBIDDEN_TRACKED:
            if low.startswith(bad.lower()) or low == bad.lower():
                rep.add(ERR, "4 不该入库的实体", rel, f"命中忽略清单 `{bad}` 却被跟踪")
        if low.endswith(FORBIDDEN_SUFFIX):
            rep.add(ERR, "4 不该入库的实体", rel, "二进制/日志/权重类后缀")
        try:
            size = (ROOT / rel).stat().st_size
        except OSError:
            continue
        if size > 512 * 1024:
            rep.add(WARN, "6 大文件", rel, f"{size / 1024:.0f} KB（确认它该在库里）")

    # 反向检查：这些"本机有、但不该进库"的东西，得真被 .gitignore 挡住
    for sub in ("data", "private", "ref", "tts/models", "tts/refs", "tts/_dl", "config.local.json"):
        if not (ROOT / sub).exists():
            continue
        r = subprocess.run(["git", "check-ignore", "-q", sub], cwd=ROOT)
        if r.returncode != 0:
            rep.add(ERR, "4 不该入库的实体", sub, "本机存在、但 `git check-ignore` 说不被忽略——补 .gitignore")


def check_filenames(rep: Report, files: list[str]) -> None:
    cn = [f for f in files if any("\u4e00" <= ch <= "\u9fff" for ch in f)]
    rep.stats["中文文件名"] = len(cn)
    if cn:
        rep.add(WARN, "5 中文文件名", f"{len(cn)} 个文件",
                "英文用户敲不出来——建议给 ASCII 别名或映射表：" + "、".join(cn[:8])
                + ("…" if len(cn) > 8 else ""))


def check_min_version(rep: Report, files: list[str], version: tuple[int, int] = (3, 11)) -> None:
    import ast
    for rel in files:
        if not rel.endswith(".py"):
            continue
        src = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(src, filename=rel, feature_version=version)
        except SyntaxError as e:
            rep.add(ERR, "8 最低版本语法", f"{rel}:{e.lineno}", f"{e.msg}（超出 {version[0]}.{version[1]} 语法）")
    rep.stats["最低版本语法"] = f"{version[0]}.{version[1]}"


def check_readme_numbers(rep: Report, files: list[str]) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    claimed = {int(m) for m in re.findall(r"(\d{2,4})\s*个测试", text)}
    if not claimed:
        return
    actual = 0
    for rel in files:
        if not rel.startswith("tests/") or not rel.endswith(".py"):
            continue
        actual += len(re.findall(r"^\s*def test_", (ROOT / rel).read_text(encoding="utf-8", errors="replace"), re.M))
    rep.stats["README 声称测试数"] = sorted(claimed)
    rep.stats["tests/ 里 def test_ 计数"] = actual
    if claimed and actual not in claimed:
        rep.add(WARN, "9 README 数字", "README.md",
                f"写着 {sorted(claimed)}，tests/ 里 def test_ 是 {actual} 个——对不上就改成实数")


# ------------------------------------------------------------------- i18n

def _find_block(text: str, start_marker: str) -> tuple[int, int]:
    """找到 `name = {` 的整块（字符串 / 注释感知的括号配对）。"""
    i = text.index(start_marker) + len(start_marker)
    depth, j, instr = 1, i, None
    while j < len(text):
        c = text[j]
        if instr:
            if c == "\\":
                j += 2
                continue
            if c == instr:
                instr = None
        elif c in "\"'`":
            instr = c
        elif c == "/" and j + 1 < len(text) and text[j + 1] == "/":
            j = text.index("\n", j)
        elif c == "/" and j + 1 < len(text) and text[j + 1] == "*":
            j = text.index("*/", j) + 2
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i - len(start_marker), j + 1
        j += 1
    raise ValueError(start_marker)


def _strip_html_noise(src: str) -> str:
    """HTML 注释与 `<style>` 段换成空格（只留结构与脚本，位置不变）。"""
    src = re.sub(r"<!--.*?-->", lambda m: re.sub(r"[^\n]", " ", m.group(0)), src, flags=re.S)
    return re.sub(r"<style[^>]*>.*?</style>", lambda m: re.sub(r"[^\n]", " ", m.group(0)), src, flags=re.S)


def _walk_js(js: str, mask_literals: bool) -> str:
    """JS 词法走一遍：注释一律清成空格；`mask_literals` 为真时连字符串 / 模板串一起清。

    为什么不是"看见 / 就当注释"：这文件里有正则字面量（`/`([^`]+)`/g`、`/[*_]/` 这些），
    正则里带着反引号和引号——朴素状态机会从那里起就一直以为自己"在字符串里"，
    于是注释漏剥、误报一片。所以照 JS 的规矩判：只有出现在"该出现正则"的位置
    （上一个有意义字符是 `(,=:[!&|?{};+-` 或行首）才按正则字面量整条跳过。

    ⚠️ `mask_literals=True` 这一档当前**没有调用方**（③ 修复后只剩
    `_strip_js_comments`）：留着是刻意的——两档共用同一套词法走法，删参数要动
    七处分支而这里没有测试兜底；真不再需要时整段一起简化。
    """
    out = list(js)
    i, n, instr, prev = 0, len(js), None, ""
    while i < n:
        c = js[i]
        if instr:
            if c == "\\":
                if mask_literals:
                    out[i] = " "
                    if i + 1 < n:
                        out[i + 1] = " "
                i += 2
                continue
            if c == instr:
                instr = None
                if mask_literals:
                    out[i] = " "
            elif mask_literals and c != "\n":
                out[i] = " "
            i += 1
            continue
        if c in "\"'`":
            instr = c
            if mask_literals:
                out[i] = " "
            i += 1
            continue
        if c == "/" and i + 1 < n and js[i + 1] == "/":
            while i < n and js[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if c == "/" and i + 1 < n and js[i + 1] == "*":
            while i < n and not (js[i] == "*" and i + 1 < n and js[i + 1] == "/"):
                if js[i] != "\n":
                    out[i] = " "
                i += 1
            out[i:i + 2] = [" ", " "]
            i += 2
            continue
        if c == "/" and (prev == "" or prev in "(,=:[!&|?{};+-"):
            j, in_class, closed = i + 1, False, False
            while j < n:
                ch = js[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "\n":
                    break
                if ch == "[":
                    in_class = True
                elif ch == "]":
                    in_class = False
                elif ch == "/" and not in_class:
                    closed = True
                    break
                j += 1
            if closed:
                i = j + 1
                prev = "/"
                continue
        if not c.isspace():
            prev = c
        i += 1
    return "".join(out)


def _strip_js_comments(js: str) -> str:
    """只去注释（保留字符串字面量）——用作"读点"扫描的底稿。"""
    return _walk_js(js, mask_literals=False)


CN_RUN = re.compile(r"[\u4e00-\u9fff][\u4e00-\u9fff\s，。：；！？（）「」·—…0-9A-Za-z/%+-]{1,}")
# ↑ 首字必是汉字，后面至少跟一个可跟随字符（`{1,}`）。原来写 `{2,}` 要 3 个
#   字符起匹配，**两字词（"语义""模型"）永远漏检**——下限收到 2 才符合"汉字片段"的直觉。

# 取词点：`t("中文")` 与 `t('中文')` 两种引号都算（index.html 里两种都有用）
T_CALL = [
    re.compile(r'(?<![\w$.])t\(\s*"((?:[^"\\]|\\.)*)"'),
    re.compile(r"(?<![\w$.])t\(\s*'((?:[^'\\]|\\.)*)'"),
]


def _split_script(src: str) -> tuple[str, str, int]:
    """把 index.html 切成（HTML 段, script 段, script 段**首行的文件行号**）。
    只认**行首**的 <script> / </script>——注释与文案里提到 `<script>` 的地方
    （本文件里正好有两处）不能被当成真标签。
    行号是给报警定位用的：script 段内的行号要么换算成文件行号，要么读的人
    在文件里根本找不到地方。"""
    opens = [m for m in re.finditer(r"(?m)^<script>\s*$", src)]
    if not opens:
        return src, "", 1
    closes = [m for m in re.finditer(r"(?m)^</script>\s*$", src)]
    start = opens[-1].end()
    end = closes[-1].start() if closes else len(src)
    return src[:start] + src[end:], src[start:end], src[:start].count("\n") + 1


def _i18n_allow() -> list[str]:
    """`tools/i18n_allow.txt`：确认"JS 会覆盖 / 故意不翻"的初始文案，一行一条。"""
    p = ROOT / "tools" / "i18n_allow.txt"
    if not p.exists():
        return []
    lines = [l.split("#")[0].strip() for l in p.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l]


def check_i18n(rep: Report) -> None:
    html_path = ROOT / "web" / "index.html"
    if not html_path.exists():
        rep.add(WARN, "7 i18n", "web/index.html", "文件不在，跳过")
        return
    src = html_path.read_text(encoding="utf-8")
    allow = _i18n_allow()
    html_part, js_part, script_first = _split_script(src)

    # ① EN 词条（中文原文即 key）
    # 缺 `<script>` / `const I18N = {` / `en: {` 时 `_find_block` / `index` 会抛
    # `ValueError`——那是"文件被大改过"的信号：报 warn 走人，**别让整个自检崩**
    # （CI 跑这个脚本，崩了会把后面的检查项一起挡住）。
    try:
        start, end = _find_block(js_part, "const I18N = {")
        en_block = js_part[start:end]
        en_block = en_block[en_block.index("en: {"):]
    except ValueError:
        rep.add(WARN, "7 i18n", "web/index.html",
                "找不到 `<script>` / `const I18N = {` / `en: {`——词条表结构变了？")
        return
    keys = re.findall(r'(?<![\w"])"((?:[^"\\]|\\.)*)"\s*:(?=\s*")', en_block)
    keyset = set(keys)
    rep.stats["EN 词条"] = len(keys)

    # ② 读点：script 里字面量 t("…") + HTML 里 data-t/tt/tp="…"；注释不算读点
    js_scan = _strip_js_comments(js_part[:start] + js_part[end:])   # 去掉词条表本身
    html_scan = _strip_html_noise(html_part)
    calls: set[str] = set()
    for pat in T_CALL:
        calls |= set(pat.findall(js_scan))
    for v in re.findall(r'data-(?:t|tt|tp)="((?:[^"\\]|\\.)*)"', html_scan):
        if "${" in v or "<" in v:          # 值是 JS 现算的（如 data-t="${esc(x.to)}"）——不是字面量 key
            continue
        calls.add(v)
    missing = sorted(c for c in calls if c not in keyset and c not in allow)
    for c in missing:
        rep.add(ERR, "7 i18n 缺译", "web/index.html", repr(c))
    rep.stats["EN 词条覆盖"] = "字面量读点全部命中" if not missing else f"{len(missing)} 条缺译"

    # ③ script 里"没走 t() 的中文"——**保留字符串 / 模板串**（只抠注释），把
    #    `t("…")` / `t('…')` 的实参替换成空格，再跳过词条表本身。两个判据：
    #      - 词条表里已有的中文算**受管文案**（tab 名的数组、状态码表、署名这类
    #        "数据 key"都长这样——显示路径上仍过 `t()`，只是静态看不见），跳过；
    #      - 剩下"连词条都没有"的中文才是缺口候选。
    #    （旧版先把字符串整个抠掉再找中文——中文文案恰恰住在字符串里，
    #     抠完恒为 0，是个死检查，2026-10-10 修。）
    js = _strip_js_comments(js_part)
    for pat in T_CALL:
        js = pat.sub(lambda m: " " * len(m.group(0)), js)
    en_from = js_part[:start].count("\n") + 1     # 词条表占的行区间（整行跳过）
    en_to = js_part[:end].count("\n") + 1
    cn_js = []
    for lineno, line in enumerate(js.splitlines(), 1):
        if en_from <= lineno <= en_to:
            continue
        for m in CN_RUN.finditer(line):
            frag = m.group(0).strip()
            if any(frag in key for key in keyset):
                continue
            if any(a and a in frag for a in allow):
                continue
            cn_js.append((lineno + script_first - 1, frag[:40]))
    for lineno, text in cn_js:
        rep.add(WARN, "7 i18n 未走 t()（script）", f"web/index.html:{lineno}", text)

    # ④ HTML 段里"没挂 data-t"的中文：排除注释 / <style> / data-* 自带的值 /
    #    同一元素里 data-t 与正文同字符串的情况（`data-t="发送">发送<` 这种是标准写法）
    cn_html = []
    for lineno, line in enumerate(html_scan.splitlines(), 1):
        if not CN_RUN.search(line):
            continue
        values = re.findall(r'data-(?:t|tt|tp)="([^"]*)"', line)
        for m in CN_RUN.finditer(line):
            frag = m.group(0).strip()
            if any(frag and frag in v for v in values):     # 正文与 data-t 同串：JS 会换掉
                continue
            if any(a and a in frag for a in allow):
                continue
            cn_html.append((lineno, frag[:40]))
    for lineno, text in cn_html:
        rep.add(WARN, "7 i18n 未挂 data-t（HTML）", f"web/index.html:{lineno}", text)
    rep.stats["未走 t() 的中文（script / HTML）"] = f"{len(cn_js)} / {len(cn_html)}"


# --------------------------------------------------------------- history scan

def check_history(rep: Report) -> None:
    revs = [l for l in git("rev-list", "--objects", "--all").splitlines() if l.strip()]
    seen: set[str] = set()
    blobs = []
    for line in revs:
        sha, _, path = line.partition(" ")
        if sha in seen or not path:
            continue
        seen.add(sha)
        if path.lower().endswith(tuple(SCRIPT_EXT)):
            blobs.append((sha, path))
    rep.stats["历史 blob（文本类）"] = len(blobs)
    hits: dict[str, list[str]] = {}
    for sha, path in blobs:
        content = subprocess.run(["git", "cat-file", "-p", sha], cwd=ROOT,
                                 capture_output=True).stdout.decode("utf-8", "replace")
        for name, pat in SECRET_PATTERNS:
            m = pat.search(content)
            if m and not PLACEHOLDER_OK.search(m.group(0)):
                hits.setdefault(f"{path}（{name}）", []).append(sha[:8])
    if not hits:
        rep.add(OK, "H 历史密钥", "git history", f"扫过 {len(blobs)} 个文本 blob，未发现密钥类串")
    for where, shas in hits.items():
        rep.add(ERR, "H 历史密钥", where, f"命中于 {sorted(set(shas))[:5]}——历史里的东西删文件不算清")

    # ② 历史里**新增过哪些路径**：私密素材 / 运行期数据一旦进过历史，后来删了也还在
    added = subprocess.run(["git", "log", "--all", "--name-only", "--pretty=format:", "--diff-filter=A"],
                           cwd=ROOT, capture_output=True).stdout.decode("utf-8", "replace")
    priv = sorted({p for p in added.splitlines()
                   if p.strip().startswith(("private/", "data/", "ref/", "tts/refs/",
                                            "tts/models/", "tts/_dl/", "config.local"))})
    if priv:
        rep.add(ERR, "H 历史路径", "git history",
                "这些私密路径在历史里新增过（删掉文件也不算清）：" + "、".join(priv[:10]))
    else:
        rep.stats["历史路径"] = "无私密素材 / 数据类路径进过历史"


# ------------------------------------------------------------------ example

EXAMPLE_KEYS = ["llm", "embedding", "search", "voice", "tts"]
SECRETISH = re.compile(r"(key|token|secret|password)", re.I)


def write_example(rep: Report) -> None:
    """从 config.local.json 生成脱敏模板 config.example.json（值全空）。"""
    src = ROOT / "config.local.json"
    if not src.exists():
        rep.add(WARN, "E 示例配置", "config.local.json", "本机没有这份文件，跳过")
        return
    data = json.loads(src.read_text(encoding="utf-8"))

    def scrub(node):
        if isinstance(node, dict):
            return {k: ("" if isinstance(v, str) and SECRETISH.search(k) else scrub(v))
                    for k, v in node.items()}
        if isinstance(node, list):
            return [scrub(v) for v in node]
        return node

    out = scrub(data)
    out["_note"] = ("配置模板：复制成 config.local.json 再填。留空 = 用默认值 / 环境变量"
                    "（AIR_LINK_* 优先）。api_key 留空 = 不改动已有 key。")
    dst = ROOT / "config.example.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rep.add(OK, "E 示例配置", "config.example.json", "已从本机配置生成（值全部清空）")


# -------------------------------------------------------------------- print

def main() -> int:
    ap = argparse.ArgumentParser(description="开源前自检 / 脱敏扫描")
    ap.add_argument("--all", action="store_true", help="连未跟踪文件一起扫")
    ap.add_argument("--history", action="store_true", help="另扫全部提交历史里的文本 blob")
    ap.add_argument("--json", metavar="PATH", help="把报告写成 JSON")
    ap.add_argument("--write-example", action="store_true", help="从 config.local.json 生成 config.example.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    rep = Report()
    files = tracked_files()
    if args.all:
        files += untracked_files()
    rep.stats["扫描文件数"] = len(files)

    scan_text_files(rep, files)
    check_tracked_hygiene(rep, files)
    check_filenames(rep, files)
    check_min_version(rep, files)
    check_readme_numbers(rep, files)
    check_i18n(rep)
    if args.history:
        check_history(rep)
    if args.write_example:
        write_example(rep)

    order = {ERR: 0, WARN: 1, OK: 2}
    print("=" * 78)
    print(f"开源前自检 · {ROOT}")
    print("=" * 78)
    for f in sorted(rep.findings, key=lambda f: (order[f.level], f.rule, f.where)):
        tag = {"error": "ERROR", "warn": "warn ", "ok": "ok   "}[f.level]
        print(f"[{tag}] {f.rule:<16} {f.where}\n         {f.detail}")
    print("-" * 78)
    for k, v in rep.stats.items():
        print(f"  · {k}: {v}")
    print("-" * 78)
    print(f"结论：error {rep.count(ERR)} · warn {rep.count(WARN)}"
          + ("　→ 有 error，先清干净再开源" if rep.count(ERR) else "　→ 可以开源（warn 逐条判断）"))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"stats": rep.stats,
             "findings": [f.__dict__ for f in rep.findings]},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if rep.count(ERR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
