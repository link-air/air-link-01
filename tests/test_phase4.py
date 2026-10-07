"""阶段 4（渐变检测）的测试。

对应阶段 4：「向量持久化 + 趋势比对」（完整的「不滞留」）。

这一层测的是那条**最难的自洽要求**：人慢慢变了，没有任何一条单独的场景
算得上反例。所以测试要能证明：
  - 「单看每条都还好，合起来是变化」能被发现
  - 「最近聊的事不一样」不会被误判成「人变了」（那是模型判的，这里只证明它被问了）
  - 判到「确实变了」才会动画像；判到 holds 一根手指都不碰
  - **没有向量服务时仍然工作**（降级只失去语义那一半，情绪那一半照常）
"""
# 用例分组：
#   脚手架  Base
#   检测    TestSplit 切两段 · TestDriftMath 算法（不碰库）· TestTopicDrift 只给数 ·
#           TestDetectDrift 判到变了才改 · TestCycleIntegration 周期集成
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod
from core.distill import run_distill_cycle
from core.llm import LLM
from core.model import Profile, Scene
from core.store import Store
from core.trend import detect_drift, emotion_shift, split_by_time, topic_drift, vector_drift

TOPIC = "用户·面对被评价的反应"


def unit(deg: float) -> list[float]:
    r = math.radians(deg)
    return [math.cos(r), math.sin(r)]


def trend_llm(verdict="holds", statement="") -> LLM:
    def fn(prompt, schema):
        if schema.get("name") == "profile_revision":
            return {"verdict": verdict, "statement": statement}
        return {}

    llm = LLM()
    llm.set_mock(fn)
    return llm


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = Store(self.root / "t.db")
        # 三个目录都要重定向：留痕、原文文档、备份——少一个就会被测试污染真实目录
        self._old_paths = {k: cfgmod.PATHS[k]
                           # shortterm 也要：有些测试直接建 ChatSession（没传
                           # state_path），默认会落到真实的窗口文件——漏了它，
                           # 跑一次测试就往真窗口里塞几条消息（实锤过）
                           for k in ("trace_dir", "raws_dir", "backup_dir",
                                     "shortterm")}
        cfgmod.PATHS["trace_dir"] = str(self.root / "trace")
        cfgmod.PATHS["raws_dir"] = str(self.root / "raws")
        cfgmod.PATHS["backup_dir"] = str(self.root / "backups")
        # 窗口也要重定向：直接建 `ChatSession`（没传 state_path）的测试
        # 默认会落到真实窗口文件——漏了它，跑一次测试就往真窗口里塞几条消息
        cfgmod.PATHS["shortterm"] = str(self.root / "shortterm.json")

    def tearDown(self):
        for k, v in self._old_paths.items():
            cfgmod.PATHS[k] = v
        self.store.close()
        self._tmp.cleanup()

    def scene(self, n: int, valence=None, arousal=None, emb=None, topic=TOPIC) -> Scene:
        s = Scene(topic=topic, subject="user", title=f"场景{n}", text="摘要",
                  trigger_class="被评价", valence=valence, arousal=arousal, emb=emb,
                  time_event=f"2026-0{1 + n // 28}-{1 + n % 28:02d} 09:00:00")
        self.store.add_scene(s)
        return s

    def profile(self, statement="遇到被评价会先退出", status="established") -> Profile:
        p = Profile(topic=TOPIC, subject="user", statement=statement, status=status,
                    evidence=3, sources=[s.id for s in self.store.query_scenes(topic=TOPIC)])
        self.store.add_profile(p)
        return p


# ---------------------------------------------------------------------
# 一、切两段
# ---------------------------------------------------------------------

class TestSplit(Base):
    def test_splits_by_time(self):
        scenes = [self.scene(i) for i in range(6)]
        early, recent = split_by_time(scenes, 0.5)
        self.assertEqual(len(early), 3)
        self.assertEqual(len(recent), 3)
        self.assertLess(early[0].id, recent[0].id, "早的那段该排在前面")

    def test_too_few_scenes(self):
        early, recent = split_by_time([self.scene(0)])
        self.assertEqual(len(early), 1)
        self.assertEqual(recent, [])

    def test_empty(self):
        self.assertEqual(split_by_time([]), ([], []))


# ---------------------------------------------------------------------
# 二、漂移计算（纯函数）
# ---------------------------------------------------------------------

class TestDriftMath(Base):
    def test_same_direction_is_no_drift(self):
        early = [unit(0), unit(2), unit(4)]
        recent = [unit(1), unit(3), unit(5)]
        drift = vector_drift(early, recent)
        self.assertIsNotNone(drift)
        self.assertLess(drift, 0.01, "同一个方向的样本，中心几乎没动")

    def test_opposite_direction_is_big_drift(self):
        drift = vector_drift([unit(0)] * 3, [unit(90)] * 3)
        self.assertAlmostEqual(drift, 1.0, places=3)

    def test_center_absorbs_noise(self):
        """平均能吸收个别噪声——渐变恰恰是一堆「单看都正常」的样本累积出来的偏移，
        逐条看会被每一条自己骗过去。"""
        early = [unit(0), unit(0), unit(0)]
        recent = [unit(2), unit(2), unit(178)]      # 中间那条是噪声
        drift = vector_drift(early, recent)
        self.assertLess(drift, 0.9, "一条反向噪声不该把中心整个拽过去")

    def test_no_vectors_returns_none(self):
        self.assertIsNone(vector_drift([], []))
        self.assertIsNone(vector_drift([None], [None]))

    def test_emotion_shift(self):
        early = [Scene(valence=-1, arousal=1), Scene(valence=-1, arousal=1)]
        recent = [Scene(valence=0, arousal=0), Scene(valence=0, arousal=0)]
        shift = emotion_shift(early, recent)
        self.assertEqual(shift["valence_shift"], 1.0)
        self.assertEqual(shift["arousal_shift"], 1.0)

    def test_null_emotion_is_skipped_not_zero(self):
        """拿不准的（NULL）跳过，**不当作 0**——0 是「确定的中性」，含义不同。"""
        early = [Scene(valence=None), Scene(valence=-1)]
        recent = [Scene(valence=None), Scene(valence=-1)]
        shift = emotion_shift(early, recent)
        self.assertEqual(shift["valence_shift"], 0.0)
        self.assertEqual(shift["valence_early"], -1.0)


# ---------------------------------------------------------------------
# 三、某个 topic 漂没漂
# ---------------------------------------------------------------------

class TestTopicDrift(Base):
    def test_not_enough_scenes(self):
        for i in range(3):
            self.scene(i)
        d = topic_drift(self.store, TOPIC)
        self.assertFalse(d["drifting"])
        self.assertIn("场景不足", d["reason"])

    def test_emotion_drift_detected_without_vectors(self):
        """**没有向量服务也能工作**：情绪那一半存在库里，不依赖 embedding。"""
        for i in range(3):
            self.scene(i, valence=-1, arousal=1)
        for i in range(3, 6):
            self.scene(i, valence=0, arousal=0)
        d = topic_drift(self.store, TOPIC)
        self.assertTrue(d["drifting"])
        self.assertIsNone(d["vector_drift"])
        self.assertIn("效价移动", d["reason"])

    def test_vector_drift_detected(self):
        for i in range(3):
            self.scene(i, emb=unit(0))
        for i in range(3, 6):
            self.scene(i, emb=unit(90))
        d = topic_drift(self.store, TOPIC)
        self.assertTrue(d["drifting"])
        self.assertIn("语义中心漂移", d["reason"])

    def test_stable_topic_is_not_drifting(self):
        for i in range(6):
            self.scene(i, valence=-1, arousal=1, emb=unit(i))
        d = topic_drift(self.store, TOPIC)
        self.assertFalse(d["drifting"])
        self.assertEqual(d["reason"], "")


# ---------------------------------------------------------------------
# 四、扫一遍并判定（唯一会动画像的地方）
# ---------------------------------------------------------------------

class TestDetectDrift(Base):
    def _make_drifting(self):
        for i in range(3):
            self.scene(i, valence=-1, arousal=1)
        for i in range(3, 6):
            self.scene(i, valence=0, arousal=0)
        return self.profile()

    def test_revise_creates_new_version(self):
        p = self._make_drifting()
        stats = detect_drift(self.store, trend_llm("revise", "遇到被评价时会先扛一下再决定"))

        self.assertEqual(len(stats["drifting"]), 1)
        self.assertEqual(len(stats["revised"]), 1)
        old = self.store.get_profile(p.id)
        self.assertNotEqual(old.invalidated_at, "")
        self.assertEqual(old.invalidated_by, "revision")
        self.assertEqual([x.statement for x in self.store.current_profiles(status=None)],
                         ["遇到被评价时会先扛一下再决定"])

    def test_holds_touches_nothing(self):
        """**漂移 ≠ 变了**：模型说「只是最近聊的事不一样」→ 一根手指都不碰。"""
        p = self._make_drifting()
        stats = detect_drift(self.store, trend_llm("holds", ""))

        self.assertEqual(len(stats["drifting"]), 1)
        self.assertEqual(stats["revised"], [])
        got = self.store.get_profile(p.id)
        self.assertEqual(got.invalidated_at, "")
        self.assertEqual(got.statement, p.statement)
        self.assertEqual(self.store.count("profiles"), 1)

    def test_stable_topic_not_judged(self):
        """没漂的 topic 不调模型（省调用，也避免无事生非）。"""
        for i in range(6):
            self.scene(i, valence=-1, arousal=1)
        self.profile()
        calls = []

        def fn(prompt, schema):
            calls.append(schema.get("name"))
            return {}

        llm = LLM()
        llm.set_mock(fn)
        stats = detect_drift(self.store, llm)
        self.assertEqual(stats["drifting"], [])
        self.assertEqual(calls, [], "不漂就不该调模型")

    def test_topic_without_profile_is_skipped(self):
        for i in range(6):
            self.scene(i, valence=-1, arousal=1)
        for i in range(6, 7):
            self.scene(i, valence=1)
        stats = detect_drift(self.store, trend_llm("revise", "x"))
        self.assertEqual(stats["checked"], 0, "没有画像就没有「变没变」的问题")

    def test_writes_trace(self):
        self._make_drifting()
        detect_drift(self.store, trend_llm("holds", ""))
        files = list((self.root / "trace").glob("漂移-*.jsonl"))
        self.assertEqual(len(files), 1)

    def test_revise_keeps_evidence_traceable(self):
        """修正后的新版本仍要能追溯到具体场景（可追溯是可否决的前提）。"""
        self._make_drifting()
        detect_drift(self.store, trend_llm("revise", "新陈述"))
        new = self.store.current_profiles(status=None)[0]
        self.assertTrue(new.sources)
        for sid in new.sources:
            if str(sid).startswith("S1"):
                self.assertIsNotNone(self.store.get_scene(sid))


# ---------------------------------------------------------------------
# 五、并入后台周期
# ---------------------------------------------------------------------

class TestCycleIntegration(Base):
    def test_cycle_reports_drift(self):
        for i in range(3):
            self.scene(i, valence=-1, arousal=1)
        for i in range(3, 6):
            self.scene(i, valence=0, arousal=0)
        self.profile()

        stats = run_distill_cycle(self.store, trend_llm("holds", ""))
        self.assertIn("drift", stats)
        self.assertEqual(stats["drift"]["checked"], 1)
        self.assertEqual(len(stats["drift"]["drifting"]), 1)
        self.assertEqual(stats["drift"]["revised"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
