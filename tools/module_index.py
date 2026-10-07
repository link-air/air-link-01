#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""各模块接口总表（生成器）。

    python tools/module_index.py            # 生成 docs/modules.md
    python tools/module_index.py --check     # 只对账：和磁盘上的 docs/modules.md 不一样就退出 1

素材全在代码里，不另写一份（另一份必然漂移）：

  · 模块头的「模块速查」块——层级 / 上游 / 下游 / 对外入口 / 边界
    （那是这个项目自己的规矩：每个模块头部都有一块，`设计与实现对照-结构.md` §四 是它的总表）
  · 模块 docstring 的第一段（这个模块是干嘛的）
  · AST 扫出来的公开符号：顶层 `def` / `class`（不含 `_` 开头），
    类再列一个公开方法；每个符号带签名与 docstring 首行

输出 docs/modules.md，**生成物不要手改**（改代码里的速查块，再跑一遍这个脚本）。
CI 里用 --check 兜住"改了代码忘了跑生成器"。
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "modules.md"

# 模块清单：core 全量。**tts 实现不进库**（仓库里只留 tts/README.md 那份接口契约）——
# 所以这里不收它：收了的话，本机（有 server.py）和公开副本会各生成一份总表，
# `--check` 在两边必有一边对不上。
TARGETS = [f"core/{p.name}" for p in sorted((ROOT / "core").glob("*.py")) if p.name != "__init__.py"]

FIELDS = ["层级", "上游", "下游", "对外入口", "边界"]


def module_summary(doc: str | None) -> str:
    """docstring 第一段（到空行为止），压成一行。"""
    if not doc:
        return ""
    para = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    return " ".join(para)


def quick_ref(lines: list[str]) -> dict[str, str]:
    """抠「模块速查」块：`#   层级    ：…` 起头，续行也是 `#`。"""
    out: dict[str, str] = {}
    inside = False
    for raw in lines:
        s = raw.strip()
        if "模块速查" in s:
            inside = True
            continue
        if not inside:
            continue
        if not s.startswith("#"):
            if out:
                break
            continue
        body = s.lstrip("#").strip()
        if set(body) <= {"-"} or not body:      # `# -----` 收尾
            if out:
                break
            continue
        for f in FIELDS:
            if body.startswith(f):
                out[f] = body.split("：", 1)[1].strip() if "：" in body else body[len(f):].strip()
                break
        else:
            if out:
                last = list(out)[-1]
                out[last] = (out[last] + " " + body).strip()
    return out


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    a = node.args
    parts: list[str] = []
    for x in a.posonlyargs:
        parts.append(x.arg)
    if a.posonlyargs:
        parts.append("/")
    for x in a.args:
        parts.append(x.arg)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for x in a.kwonlyargs:
        parts.append(x.arg)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    ret = ""
    if node.returns is not None:
        try:
            ret = " -> " + ast.unparse(node.returns)
        except Exception:
            ret = ""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}{node.name}({', '.join(parts)}){ret}"


def public_api(tree: ast.Module) -> list[str]:
    rows: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            rows.append(f"`{_sig(node)}` — {first_line(node) or '（无 docstring）'}")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            rows.append(f"**class `{node.name}`** — {first_line(node) or '（无 docstring）'}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith("_"):
                    rows.append(f"　· `{sub.name}{_sig(sub)[len(sub.name):]}` — {first_line(sub) or '—'}")
    return rows


def first_line(node) -> str:
    doc = ast.get_docstring(node) or ""
    for line in doc.strip().splitlines():
        if line.strip():
            return line.strip()[:200]
    return ""


def build() -> str:
    w = io.StringIO()
    w.write("# 模块接口总表 · Module index\n\n")
    w.write("> **生成物，别手改**——素材是各模块头部的「模块速查」块与公开符号。\n")
    w.write("> 重新生成：`python tools/module_index.py`；对账（不改文件）：`python tools/module_index.py --check`。\n")
    w.write("> 中文是正文，英文只在标题与字段名上（这个项目的中文即文档原文，翻译见 README.en.md 的说明）。\n\n")
    w.write("字段：`Layer` 层级 · `Upstream` 上游 · `Downstream` 下游 · `Entry points` 对外入口 · `Boundary` 边界。\n\n")
    w.write("## 目录 · Contents\n\n")
    for rel in TARGETS:
        w.write(f"- [`{rel}`](#{rel.replace('/', '').replace('.', '').replace('_', '')})\n")
    w.write("\n---\n\n")

    for rel in TARGETS:
        p = ROOT / rel
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=rel)
        ref = quick_ref(src.splitlines())
        w.write(f"## `{rel}`\n\n")
        w.write(module_summary(ast.get_docstring(tree)) + "\n\n")
        w.write("| | |\n|---|---|\n")
        for f, en in (("层级", "Layer"), ("上游", "Upstream"), ("下游", "Downstream"),
                      ("对外入口", "Entry points"), ("边界", "Boundary")):
            w.write(f"| {f} · {en} | {ref.get(f, '—')} |\n")
        rows = public_api(tree)
        w.write("\n**公开符号 · public API**\n\n")
        for r in rows:
            w.write(f"- {r}\n")
        w.write("\n")
    return w.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 / 对账 模块接口总表")
    ap.add_argument("--check", action="store_true", help="不改文件，只对账")
    args = ap.parse_args()

    text = build()
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != text:
            print(f"docs/modules.md 与代码不一致——跑 `python tools/module_index.py` 重新生成", file=sys.stderr)
            return 1
        print(f"docs/modules.md 与代码一致（{len(TARGETS)} 个模块）")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"已写 {OUT.relative_to(ROOT)}（{len(TARGETS)} 个模块，{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
