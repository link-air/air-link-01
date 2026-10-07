"""tts/server.py 的**切分纯函数**——长文分段（单次生成有 2048 帧硬上限）。

为什么单独测它：服务本体要 torch（独立 venv，主项目跑不了），但切分是纯字符串
逻辑，而且它有一条硬性质——**一个字都不能丢**（丢了就是静默漏读，正是这次要
修的那类故障）。用 importlib 按路径加载 server.py：模块级只读 voice.json，
不碰 torch。
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_server():
    path = ROOT / "tts" / "server.py"
    if not path.exists():
        # 本机实现不进库（仓库里只留 tts/README.md 那份契约）：
        # 公开副本上没有这个文件——这一组按"本机才跑"跳过，不是失败。
        raise unittest.SkipTest("tts/server.py 不在本机（本机实现的测试）")
    spec = importlib.util.spec_from_file_location("tts_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SplitForTTSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_server()

    def test_short_text_stays_whole(self):
        self.assertEqual(self.m._split_for_tts("你好，我在。"), ["你好，我在。"])

    def test_empty_stays_empty(self):
        self.assertEqual(self.m._split_for_tts(""), [])
        self.assertEqual(self.m._split_for_tts("   "), [])

    def test_long_text_within_limit_and_lossless(self):
        t = "这是一句完整的话，用来凑长度。" * 30 + "最后一句没有句号"
        segs = self.m._split_for_tts(t)
        self.assertGreater(len(segs), 1, "长文必须被切")
        self.assertTrue(all(len(s) <= self.m._TTS_SEG_CHARS for s in segs),
                        "每段都得在闸内（超了就是静默截断）")
        self.assertEqual("".join(segs), t, "切分只挪位置，一个字都不许丢")

    def test_no_punctuation_is_hard_split_lossless(self):
        t = "很长" * 300          # 600 字，一个标点都没有
        segs = self.m._split_for_tts(t)
        self.assertTrue(all(len(s) <= self.m._TTS_SEG_CHARS for s in segs))
        self.assertEqual("".join(segs), t)

    def test_consecutive_punctuation_survives(self):
        t = "好！！那……就这样吧。" * 40
        segs = self.m._split_for_tts(t)
        self.assertEqual("".join(segs), t)
        self.assertTrue(all(len(s) <= self.m._TTS_SEG_CHARS for s in segs))

    def test_newlines_are_kept(self):
        t = ("第一段。\n第二段，也不长。\n" * 20)
        segs = self.m._split_for_tts(t)
        self.assertEqual("".join(segs), t.strip(),
                         "换行原样保留；只有**首尾**空白会被去掉（那是排版噪声，不是内容）")


if __name__ == "__main__":
    unittest.main()
