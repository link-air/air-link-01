"""人格文件（`core/persona.py`）的测试。

对应 2026-10-05 的设置页「人格」块（新建 / 编辑 / 停用）。这一层的测试盯的是
**"文件是真源"那条纪律的边角**：

  - id 会写进原文署名（`salvage._split_dialogue` 认行首 `名字:`）——非法字符与
    保留名一律**拒**，不自动改名（拒绝不静默：悄悄改成别的，他下次找文件找不到）
  - 停用是**改名**不是删除：名单里消失、文件还在、恢复即回来；`air` 不许停（兜底）
  - 改前值必须留：**真改了才写** `人格-*.jsonl`，且留的是改前全文
  - 读侧回退链不变：`<id>` → `air` → 一句占位（文件缺失也不罢工）
"""
# 用例分组：
#   脚手架  Base（临时目录 = 人格库 + 留痕目录）
#   建      TestCreate 校验 / 模板 / 撞名
#   改      TestSave 写回与留痕（改前全文）
#   停      TestEnable 停用 / 恢复 / air 不许停
#   读      TestLoad 回退链与名单保底
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core import persona

AIR = "# air\n\n我是 air，认真地听。\n"


class Base(unittest.TestCase):
    """临时人格库：**别碰真的 `self/personas/`**。

    `trace_dir` 也一起搬——不然跑一次测试就往真的 `data/trace/` 里塞几行
    人格留痕，那些行会混进他以后翻的痕迹里（"谁在什么时候改了 air"——
    其实没人改过）。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._old = (cfgmod.PATHS["personas"], cfgmod.PATHS["trace_dir"])
        cfgmod.PATHS["personas"] = str(self.dir / "personas")
        cfgmod.PATHS["trace_dir"] = str(self.dir / "trace")
        (self.dir / "personas").mkdir(parents=True)
        (self.dir / "personas" / "air.md").write_text(AIR, encoding="utf-8")

    def tearDown(self):
        cfgmod.PATHS["personas"], cfgmod.PATHS["trace_dir"] = self._old
        self._tmp.cleanup()

    def p(self, name: str) -> Path:
        return self.dir / "personas" / name

    def traces(self) -> list[dict]:
        """今天那份人格留痕（没有 → 空列表）。"""
        f = self.dir / "trace" / f"人格-{datetime.now().strftime('%Y%m%d')}.jsonl"
        if not f.exists():
            return []
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]

    def clear_traces(self):
        """清掉 fixture 自己留下的那几条（`create` 会写「新建」）——
        本用例只关心它之后发生的事。"""
        d = self.dir / "trace"
        if d.exists():
            for f in d.glob("*.jsonl"):
                f.unlink()


class TestCreate(Base):
    def test_creates_with_template_and_joins_list(self):
        out = persona.create("mo")
        self.assertTrue(out["ok"], out)
        self.assertTrue(self.p("mo.md").exists())
        # 模板标题跟着 id 走（不叫"新人格"）：建完不改也能用、看得清是谁
        self.assertEqual(out["item"]["display"], "mo")
        self.assertIn("mo", persona.names())

    def test_creates_with_given_text(self):
        out = persona.create("mo", "# 小满\n\n我是小满。\n")
        self.assertTrue(out["ok"], out)
        self.assertEqual(persona.load("mo").strip(), "# 小满\n\n我是小满。".strip())
        self.assertEqual(out["item"]["display"], "小满")
        # id 与显示名分开：id 仍是文件名（会进署名），中文只活在正文首行
        self.assertIn("mo", persona.names())

    def test_bad_ids_rejected(self):
        for bad in ("", "  ", "Mia", "小满", "a", "a" * 17, "a b", "a:b", "a.b", "用户"):
            out = persona.create(bad)
            self.assertFalse(out["ok"], f"{bad!r} 该被拒")
        self.assertFalse(self.p("Mia.md").exists())
        # 拒了就不该留下任何文件（半个也不留）
        self.assertEqual(list((self.dir / "personas").glob("*.md")), [self.p("air.md")])

    def test_duplicate_rejected(self):
        self.assertTrue(persona.create("mo")["ok"])
        out = persona.create("mo")
        self.assertFalse(out["ok"])
        self.assertIn("有了", out["detail"])

    def test_disabled_collision_rejected(self):
        """撞上停用副本也要拒——否则目录里会躺着同 id 的两份，谁生效说不清。"""
        self.assertTrue(persona.create("mo")["ok"])
        self.assertTrue(persona.set_enabled("mo", False)["ok"])
        out = persona.create("mo")
        self.assertFalse(out["ok"])
        self.assertIn("停用", out["detail"])

    def test_too_long_rejected(self):
        out = persona.create("mo", "x" * (persona.MAX_CHARS + 1))
        self.assertFalse(out["ok"])
        self.assertFalse(self.p("mo.md").exists())


class TestSave(Base):
    def setUp(self):
        super().setUp()
        persona.create("mo", "# mo\n\n第一版。\n")
        self.clear_traces()

    def test_save_writes_and_traces_before(self):
        out = persona.save("mo", "# mo\n\n第二版。\n")
        self.assertTrue(out["ok"] and out["changed"], out)
        self.assertIn("第二版", persona.load("mo"))
        rec = self.traces()
        self.assertEqual(len(rec), 1)
        self.assertEqual(rec[0]["act"], "编辑")
        self.assertEqual(rec[0]["id"], "mo")
        self.assertIn("第一版", rec[0]["before"])       # 改前全文（"改前值必须留"）

    def test_same_text_is_not_a_change_and_writes_no_trace(self):
        txt = persona.text("mo")
        out = persona.save("mo", txt)
        self.assertTrue(out["ok"])
        self.assertFalse(out["changed"])
        self.assertEqual(self.traces(), [])             # 点开看一眼再保存，不该留痕

    def test_empty_rejected(self):
        out = persona.save("mo", "   \n ")
        self.assertFalse(out["ok"])
        self.assertIn("第一版", persona.load("mo"))      # 原文没被动

    def test_unknown_id_rejected(self):
        self.assertFalse(persona.save("nobody", "x")["ok"])

    def test_too_long_rejected(self):
        self.assertFalse(persona.save("mo", "x" * (persona.MAX_CHARS + 1))["ok"])
        self.assertIn("第一版", persona.load("mo"))


class TestEnable(Base):
    def setUp(self):
        super().setUp()
        persona.create("mo", "# 小满\n\n我是小满。\n")
        self.clear_traces()

    def test_disable_is_rename_not_delete(self):
        out = persona.set_enabled("mo", False)
        self.assertTrue(out["ok"], out)
        self.assertFalse(self.p("mo.md").exists())
        self.assertTrue(self.p("mo.md.off").exists())
        self.assertNotIn("mo", persona.names())          # 名单里没了
        self.assertIn("mo", persona.names(include_disabled=True))
        self.assertFalse(persona.info("mo")["enabled"])
        # 正文还在（恢复前想先看看它写了什么：`text()` 读得到）
        self.assertIn("我是小满", persona.text("mo"))

    def test_restore_brings_it_back(self):
        persona.set_enabled("mo", False)
        out = persona.set_enabled("mo", True)
        self.assertTrue(out["ok"], out)
        self.assertTrue(self.p("mo.md").exists())
        self.assertIn("mo", persona.names())
        self.assertIn("我是小满", persona.load("mo"))

    def test_traces_tell_who_moved_it(self):
        persona.set_enabled("mo", False)
        persona.set_enabled("mo", True)
        self.assertEqual([r["act"] for r in self.traces()], ["停用", "启用"])

    def test_air_cannot_be_disabled(self):
        out = persona.set_enabled("air", False)
        self.assertFalse(out["ok"])
        self.assertTrue(self.p("air.md").exists())
        self.assertEqual(persona.load("air"), AIR)

    def test_restore_unknown_rejected(self):
        self.assertFalse(persona.set_enabled("nobody", True)["ok"])


class TestLoad(Base):
    def test_falls_back_to_air(self):
        self.assertEqual(persona.load("nobody"), AIR)
        self.assertEqual(persona.load(""), AIR)

    def test_placeholder_when_all_missing(self):
        self.p("air.md").unlink()
        self.assertIn("人格文件缺失", persona.load("nobody"))

    def test_names_keep_one_option_when_dir_empty(self):
        """目录空了（新克隆 / 误删）也不能让顶栏变成一个空下拉。"""
        self.p("air.md").unlink()
        self.assertEqual(persona.names(), ["air"])

    def test_list_info_enabled_first(self):
        persona.create("mo")
        persona.set_enabled("mo", False)
        ids = [i["id"] for i in persona.list_info()]
        self.assertEqual(ids, ["air", "mo"])             # 在用的排前面
        self.assertTrue(persona.list_info()[0]["enabled"])
        self.assertFalse(persona.list_info()[1]["enabled"])


if __name__ == "__main__":
    unittest.main()
