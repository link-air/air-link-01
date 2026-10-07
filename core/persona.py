"""人格文件（`self/personas/*.md`）：读 / 列 / 写 / 停用。

2026-10-05 新增——在他那句「我觉得应该加上一个新建人格的设置功能，建好加入人格库，
顺便也能查看/修改已有人格」之前，这个模块只有**读**的那半边（`chat.load_persona` /
`persona_names`，两处各扫一遍目录）；写侧进来之后，读也一并挪到这儿，
`chat` / `salvage` 只留薄转发——**"人格文件长什么样"只该有一处知道**。

三条定位（照 2026-10-05 定的口径）：

1. **文件是真源，界面只是编辑器**（同 `tts/voice.json` 的文件代理定位）：
   不落库、不建注册表——"加一个 md 即多一个选项"是 2026-09-19 起就有的性质，
   写进库里等于把真源挪走，文件、git、他手改的路子会三处打架。
2. **停用 ≠ 删除**：改名 `<id>.md.off`（`glob("*.md")` 自然扫不到，文件还在，
   改回来即恢复）。同"归档只标记不删"的纪律；`air` 不许停用（`load()` 的兜底）。
3. **改前值必须留**：每次真改动写一行 `data/trace/人格-*.jsonl`（含改前全文）——
   人格是"他定的东西"里最重的一类（它决定她是谁），这份留痕最不该省。

边界：**这里不碰声音**（全局一份，在仪表盘「语音」页，且不留痕——嗓子不影响她记住什么）；
也不碰记忆（人格是外壳，换谁说话看到的是同一份记忆）。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L3 提示词（与 prompts 同层；prompts 管"拼"，这里管"存"）
#   上游    ：config / store（只要一个 now_str）
#   下游    ：chat（转发）、salvage（转发）、dashboard（写侧入口）
#   对外入口：load / names / info / list_info / create / save / set_enabled
#   边界    ：不碰声音、不碰记忆、不做"人格 = 新记忆库"那套隔离
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from . import config as cfgmod
from .store import now_str

# id（= 文件名）的字符集。**不是审美**：它会写进原文署名（`air: …`），
# `salvage._split_dialogue` 认行首 `名字:`——中文 / 冒号 / 空格进去就把原文解析弄坏。
ID_RE = re.compile(r"^[a-z0-9_-]{2,16}$")

# 说话人保留名：`用户` 在原文解析里是 user 的署名，人格不能占。
_RESERVED = ("用户",)

# 不许停用的：`air` 是 `load()` 的兜底——停掉它，人格文件缺失时就只剩一句占位。
_NO_DISABLE = ("air",)

# 出厂的三个人格（**只用于界面上的"内置"标记**，不参与任何逻辑判断）。
BUILTIN = ("air", "mia", "xina")

# 单份人格的字符上限。**不是预算**（预算由设置页的常驻块余额管），是防手滑：
# 一份人格是每轮都要付的固定成本，粘错一整个文档进来它会一路顶穿。
MAX_CHARS = 8000

# 新建人格的起步骨架：新增即可用，改由他。
# 章节名与出厂三份对齐——想从 mia / xina 抄一段过来，结构是现成的。
TEMPLATE = """# 新人格

## 我的身份

我是一个陪他说话的人。认真地听，跟着他的节奏，不评判，不诊断。

## 我怎么陪用户

先分清他这次要什么：想倾诉、只是分享，还是要解决。分不清就问，别猜。
他没问就不给——分享不是请人点评。

## 我怎么思考

一次观察不下结论；矛盾证据能推翻我的判断。
摆出来的是「我看到的样子」，不是对他的判决。

## 我怎么说话

长度跟着他来：想倾诉的多留，要解决的短说；说完把话头交回去，不拔高。
"""


def _base() -> Path:
    """人格目录（配置 `PATHS.personas`——测试会临时改它）。"""
    return cfgmod.abspath(cfgmod.PATHS.get("personas") or "self/personas")


def _path(name: str) -> Path:
    return _base() / f"{name}.md"


def _off_path(name: str) -> Path:
    """停用后的名字。`.md.off` 不被 `glob("*.md")` 匹配——停用即从名单消失，
    而文件还在原地（改回来就是恢复，不经过任何"导入"）。"""
    return _base() / f"{name}.md.off"


def _display(text: str, name: str) -> str:
    """显示名 = 文件首行 `# xxx`（中文 / 空格随他写）；没有就用 id。

    为什么让文件自己带显示名、而不是在库里存一个 label：id 有字符集限制（见 `ID_RE`），
    而"她叫什么"是给人看的东西，不该被那条限制绑住；写进库里又多了第二处真源。
    """
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            title = s.lstrip("#").strip()
            if title:
                return title
        break          # 只认首行——写在正文中间的标题不是"她叫什么"
    return name


def check_id(name: str) -> str:
    """id 合不合格。返回错误说明；空串 = 通过。

    为什么不自动小写 / 自动去空格：**拒绝不静默**——把他输入的东西悄悄改掉，
    下次他找 `Mia.md` 会找不到文件。
    """
    name = (name or "").strip()
    if not name:
        return "名字不能是空的"
    if name in _RESERVED:
        return f"「{name}」是原文里的保留说话人，换一个"
    if not ID_RE.match(name):
        return ("名字只用小写英文 / 数字 / `_` / `-`，2–16 位"
                "（它会写进对话原文的署名，中文和符号进去会让原文解析认不出人）")
    return ""


def _write_text(path: Path, text: str) -> None:
    """原子替换地写一份文本（同 `store.atomic_write_json` 的思路，简化版）。

    退避重试那套没照搬（人格几 KB、只在人点保存时写一次）；但**先写临时文件再
    replace** 要留着：直接覆盖写、写一半崩了，那份人格就剩半截——而它下一轮
    就要进提示词。临时文件用 `.md.tmp` 结尾，`glob("*.md")` 扫不到它。
    """
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_persona_trace(act: str, name: str, before: str = "",
                        note: str = "") -> None:
    """人格改动的留痕（`data/trace/人格-YYYYMMDD.jsonl`）。

    `before` 存**改前全文**——"改前值必须留"这条规矩在这里最不该省：
    单条记忆丢了还能从原文找回来，人格被改坏了没有第二处副本（git 那份要他会用）。
    写不成不拦改动本身（同各处 trace 的兜底）。
    """
    try:
        trace_dir = cfgmod.abspath(cfgmod.PATHS["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"人格-{datetime.now().strftime('%Y%m%d')}.jsonl"
        rec = {"ts": now_str(), "act": act, "id": name, "note": note}
        if before:
            rec["before"] = before
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[persona] 人格留痕失败（不影响改动本身）: {e}")


def names(include_disabled: bool = False) -> list[str]:
    """人格目录里有哪些——界面的选项列表用（扫目录，加文件即生效）。

    `include_disabled=True` 时把停用的（`.md.off`）也带上（设置页要显示"有但停着"）；
    默认只要在用的。**永远保底一个 `air`**：目录空了（新克隆、误删）也不能让
    顶栏变成一个空下拉。
    """
    base = _base()
    try:
        out = [p.stem for p in base.glob("*.md") if p.stem]
        if include_disabled:
            out += [p.name[:-7] for p in base.glob("*.md.off") if p.name[:-7]]
    except Exception:
        out = []
    out = sorted(set(out))
    return out or ["air"]


def load(name: str = "") -> str:
    """读 `<id>.md`——当前人格的提示词。**每轮现读**：切了 / 改完，下一句就换。

    读不到一级级回退：`<name>` → `air` → 一句占位。**不因为文件缺失就罢工**：
    一句提示 + 继续跑，比什么都做不了强（同旧宪章的兜底理由）。
    """
    name = (name or "").strip() or "air"
    try:
        for cand in (_path(name), _path("air")):
            if cand.exists():
                return cand.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[persona] 人格读取失败（{name}）: {e}")
    return "（人格文件缺失：我是 air，记住的是一个人和一段关系。）"


def info(name: str) -> dict:
    """一个人格的档案：显示名 / 字数 / 在不在用 / 是不是出厂的。

    读不出正文也给得出一份（`chars=0`）——设置页拿它渲染列表，
    一个读坏的文件不该让整块空掉（它恰恰是最该看见"这里坏了"的时候）。
    """
    text = ""
    enabled = _path(name).exists()
    p = _path(name) if enabled else _off_path(name)
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        pass
    return {"id": name,
            "display": _display(text, name),
            "chars": len(text),
            "enabled": enabled,
            "builtin": name in BUILTIN}


def list_info() -> list[dict]:
    """在用的排前面、停用的排后面（各按 id 排序）——设置页一屏看完有哪些人。"""
    items = [info(n) for n in names(include_disabled=True)]
    return sorted(items, key=lambda d: (not d["enabled"], d["id"]))


def text(name: str) -> str:
    """原样读一份人格正文（设置页的编辑框用）。

    **停用的也读得到**（`.md.off`）——他可能在恢复之前想先看看它当初写了什么。
    读不到返回空串，由调用方显示"这个文件读不出来"：一个读坏的文件不该让
    整块编辑区报错——那恰恰是最该看见"这里坏了"的时候。
    """
    name = (name or "").strip()
    for p in (_path(name), _off_path(name)):
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


def create(name: str, text: str = "") -> dict:
    """新建一个人格。`text` 空 → 给 `TEMPLATE`（新增即可用，改由他）。"""
    name = (name or "").strip()
    err = check_id(name)
    if err:
        return {"ok": False, "detail": err}
    if _path(name).exists():
        return {"ok": False, "detail": f"「{name}」已经有了——改名，或直接编辑它"}
    if _off_path(name).exists():
        # 撞上停用副本：直接新建会让目录里躺着同 id 的两份，谁在生效从此说不清
        return {"ok": False, "detail": f"「{name}」停用过（文件还在）——先恢复它，或换个名字"}
    text = (text or "").strip()
    if not text:
        # 模板标题跟着 id 走（不叫"新人格"）：建完不改也能用，
        # 列表与顶栏里一眼看得清是谁
        text = TEMPLATE.replace("新人格", name, 1).strip()
    if len(text) > MAX_CHARS:
        return {"ok": False, "detail": f"太长了（{len(text)} 字，上限 {MAX_CHARS}）——人格是每轮都要付的成本"}
    try:
        _base().mkdir(parents=True, exist_ok=True)
        _write_text(_path(name), text.rstrip() + "\n")
    except Exception as e:
        return {"ok": False, "detail": f"写文件失败：{type(e).__name__}: {e}"}
    write_persona_trace("新建", name, note=f"{len(text)} 字")
    return {"ok": True, "item": info(name)}


def save(name: str, text: str) -> dict:
    """整份写回（设置页的"保存"）。**只在真改了时留痕**——点开看一眼再保存，
    不该在 `人格-*.jsonl` 里留一行空记录。"""
    name = (name or "").strip()
    if not _path(name).exists():
        return {"ok": False, "detail": f"没有「{name}」这个人格文件"}
    text = (text or "").strip()
    if not text:
        return {"ok": False,
                "detail": "不能存成空的——人格文件空着，她就没有「怎么说话」那一层了"}
    if len(text) > MAX_CHARS:
        return {"ok": False, "detail": f"太长了（{len(text)} 字，上限 {MAX_CHARS}）——人格是每轮都要付的成本"}
    try:
        before = _path(name).read_text(encoding="utf-8")
    except Exception:
        before = ""
    if before.strip() == text:
        return {"ok": True, "changed": False, "item": info(name)}
    try:
        _write_text(_path(name), text.rstrip() + "\n")
    except Exception as e:
        return {"ok": False, "detail": f"写文件失败：{type(e).__name__}: {e}"}
    write_persona_trace("编辑", name, before=before, note=f"{len(before)} → {len(text)} 字")
    return {"ok": True, "changed": True, "item": info(name)}


def set_enabled(name: str, on: bool) -> dict:
    """停用 / 恢复（`<id>.md` ↔ `<id>.md.off`）。

    **不做删除**：人格是"她是谁"，真删该走文件系统（他手上有 git 和回收站）。
    停用是这里的最强动作；`air` 不许停（`load()` 的兜底）。
    """
    name = (name or "").strip()
    if not on and name in _NO_DISABLE:
        return {"ok": False, "detail": f"「{name}」是兜底人格，不能停用——它要在别人格缺失时顶上"}
    if on:
        if not _off_path(name).exists():
            return {"ok": False, "detail": f"没有停用中的「{name}」"}
        try:
            os.replace(_off_path(name), _path(name))
        except Exception as e:
            return {"ok": False, "detail": f"恢复失败：{type(e).__name__}: {e}"}
        write_persona_trace("启用", name)
    else:
        if not _path(name).exists():
            return {"ok": False, "detail": f"没有「{name}」这个人格"}
        try:
            os.replace(_path(name), _off_path(name))
        except Exception as e:
            return {"ok": False, "detail": f"停用失败：{type(e).__name__}: {e}"}
        write_persona_trace("停用", name)
    return {"ok": True, "item": info(name)}
