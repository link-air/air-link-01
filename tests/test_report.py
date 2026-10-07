"""实验报告渲染（`run_experiment.render_markdown`）。

报告是三个"给人看"的入口之一（另两个：仪表盘、CLI 打印）。
它最容易出的一类问题是**静默漏掉一整段**——写了逻辑但条件判断写反、
或者键名对不上，于是那段永远不出现，而没人会注意到"少了什么"。
所以这里直接断言段落存在。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_experiment import render_markdown


def _minimal_rep(**over) -> dict:
    """一份最小的、能渲染通过的实验报告数据。"""
    rep = {
        "name": "测试", "note": "", "script": "x.json", "db": "x.db",
        "model": "m", "vector": True, "started": "2026-09-12 10:00:00",
        "sessions": [],
        "cycle": {"s2_new": [], "s3_new": [], "established": [],
                  "blocked": [], "aged": []},
        "drift": {},
        "topic_merge": [],
        "final": {
            "counts": {"scenes": 0, "summaries": 0, "profiles": 0,
                       "memos": 0, "entities": 0},
            "scenes": [], "profiles": [], "summaries": [], "memos": [],
            "entities": [],
        },
    }
    rep.update(over)
    return rep


class TestReportSections(unittest.TestCase):
    def test_blocked_is_listed_with_reason(self):
        """未收敛的画像要连原因一起列出来——那是"调 prompt 的靶子"。"""
        rep = _minimal_rep()
        rep["cycle"]["blocked"] = [
            {"id": "S3-0001", "topic": "用户·X", "reason": "印证不足（2/3）"}]
        md = render_markdown(rep)
        self.assertIn("未收敛", md)
        self.assertIn("印证不足（2/3）", md)

    def test_no_blocked_says_so_explicitly(self):
        """没有未收敛的要**明说**「（无）」——「这段被查过了」本身是信息。"""
        self.assertIn("未收敛：（无）", render_markdown(_minimal_rep()))

    def test_merge_suggestions_are_rendered(self):
        """主题归并建议要出现在报告里（它不是只有仪表盘才看得到的东西）。"""
        rep = _minimal_rep(topic_merge=[{
            "from": "用户·被批评的反应", "to": "用户·被当众批评的反应",
            "basis": "措辞相近", "similarity": 0.951,
            "from_n": 1, "to_n": 3, "from_sample": "a", "to_sample": "b"}])
        md = render_markdown(rep)
        self.assertIn("## 主题归并建议", md)
        self.assertIn("用户·被批评的反应", md)
        self.assertIn("0.951", md)
        self.assertIn("系统不会自动合并", md, "要说清「合不合由人定」")

    def test_no_suggestions_no_section(self):
        """没有建议就不该冒出空段落（空段落会让人以为功能坏了）。"""
        self.assertNotIn("主题归并建议", render_markdown(_minimal_rep()))


if __name__ == "__main__":
    unittest.main()
