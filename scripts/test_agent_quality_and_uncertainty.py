#!/usr/bin/env python3
from __future__ import annotations

import unittest

from main import AudioEvent, FoleyOrchestrator


class FakeExecutionNode:
    def generate_audio(self, prompt: str, timestamp: float, duration: float, attempt: int) -> str:
        return f"/tmp/fake_event_{timestamp}_{attempt}.wav"


class FakeExecutionNodeRecording(FakeExecutionNode):
    def __init__(self):
        self.prompts = []

    def generate_audio(self, prompt: str, timestamp: float, duration: float, attempt: int) -> str:
        self.prompts.append(prompt)
        return super().generate_audio(prompt, timestamp, duration, attempt)


class FakeVerificationNode:
    def __init__(self, final_score: float, agreement_ok: bool = True):
        self.final_score = float(final_score)
        self.agreement_ok = bool(agreement_ok)

    def evaluate(self, prompt: str, audio_path: str) -> dict:
        score_primary = self.final_score + 1.0
        score_secondary = self.final_score
        return {
            "score_primary": score_primary,
            "score_secondary": score_secondary,
            "final_score": self.final_score,
            "score_gap": abs(score_primary - score_secondary),
            "agreement_ok": self.agreement_ok,
            "verifier_gap_delta": 0.25,
        }


class FakePlannerNode:
    def decide_iteration(
        self,
        event: AudioEvent,
        current_prompt: str,
        score: float,
        threshold: float,
        attempt: int,
        max_retries: int,
        state,
    ) -> dict:
        return {
            "action": "ACCEPT",
            "reasoning": "test planner requests acceptance",
            "confidence": 0.9,
            "next_prompt": "",
            "source": "test",
        }

    def refine_prompt(self, failed_prompt: str, score: float) -> str:
        return failed_prompt + " refined"


class FakePlannerRewriteWithNullPrompt:
    def decide_iteration(
        self,
        event: AudioEvent,
        current_prompt: str,
        score: float,
        threshold: float,
        attempt: int,
        max_retries: int,
        state,
    ) -> dict:
        if attempt == 1:
            return {
                "action": "RETRY_REWRITE",
                "reasoning": "force rewrite in test",
                "confidence": 0.8,
                "next_prompt": None,
                "source": "test",
            }
        return {
            "action": "STOP_BEST",
            "reasoning": "stop after rewrite path exercised",
            "confidence": 0.8,
            "next_prompt": "",
            "source": "test",
        }

    def refine_prompt(self, failed_prompt: str, score: float) -> str:
        return failed_prompt + " refined"


class OrchestratorQualityTests(unittest.TestCase):
    def _build_orchestrator(self) -> FoleyOrchestrator:
        orch = FoleyOrchestrator(event_callback=None)
        orch.clap_score_min = 0.0
        orch.clap_score_max = 10.0
        orch.quality_threshold = 0.60
        orch.execution = FakeExecutionNode()
        orch.planner = FakePlannerNode()
        return orch

    def test_normalize_quality_score_clamps_and_is_monotonic(self):
        orch = self._build_orchestrator()
        values = [-5.0, 0.0, 2.0, 5.0, 10.0, 20.0]
        normalized = [orch.normalize_quality_score(v) for v in values]

        self.assertEqual(normalized[0], 0.0)
        self.assertEqual(normalized[1], 0.0)
        self.assertEqual(normalized[2], 0.2)
        self.assertEqual(normalized[3], 0.5)
        self.assertEqual(normalized[4], 1.0)
        self.assertEqual(normalized[5], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(normalized, normalized[1:])))

    def test_acceptance_uses_normalized_score_threshold(self):
        orch = self._build_orchestrator()
        orch.max_retries = 1
        orch.verification = FakeVerificationNode(final_score=8.0, agreement_ok=True)

        event = AudioEvent(
            timestamp_sec=0.0,
            duration_sec=2.0,
            original_prompt="wind rustle in trees",
            refined_prompt="",
        )
        out = orch.run_event_agent(event, expected_keywords={"wind"})

        self.assertAlmostEqual(out.similarity_score, 0.8, places=6)
        self.assertEqual(out.agent_trace[-1]["action"], "ACCEPT")
        self.assertFalse(out.agent_trace[-1]["acceptance_blocked_by_uncertainty"])

    def test_uncertain_candidate_never_accepted_on_final_attempt(self):
        orch = self._build_orchestrator()
        orch.max_retries = 1
        orch.verification = FakeVerificationNode(final_score=9.0, agreement_ok=False)

        event = AudioEvent(
            timestamp_sec=0.0,
            duration_sec=2.0,
            original_prompt="gentle wind ambience",
            refined_prompt="",
        )
        out = orch.run_event_agent(event, expected_keywords={"thunder"})

        trace = out.agent_trace[-1]
        self.assertEqual(trace["action"], "STOP_BEST")
        self.assertTrue(trace["acceptance_blocked_by_uncertainty"])
        self.assertIn("verifier_disagreement", trace["uncertainty_reasons"])
        self.assertIn("cross_modal_mismatch", trace["uncertainty_reasons"])

    def test_retry_rewrite_with_null_next_prompt_uses_refiner(self):
        orch = self._build_orchestrator()
        orch.max_retries = 2
        orch.execution = FakeExecutionNodeRecording()
        orch.verification = FakeVerificationNode(final_score=2.0, agreement_ok=True)
        orch.planner = FakePlannerRewriteWithNullPrompt()

        event = AudioEvent(
            timestamp_sec=0.0,
            duration_sec=2.0,
            original_prompt="racing car passing by",
            refined_prompt="",
        )
        orch.run_event_agent(event, expected_keywords={"car"})

        self.assertEqual(orch.execution.prompts[0], "racing car passing by")
        self.assertEqual(orch.execution.prompts[1], "racing car passing by refined")
        self.assertNotEqual(orch.execution.prompts[1], "None")

    def test_prompt_only_relaxes_verifier_disagreement_gate(self):
        orch = self._build_orchestrator()
        orch.max_retries = 1
        orch.verification = FakeVerificationNode(final_score=9.0, agreement_ok=False)

        event = AudioEvent(
            timestamp_sec=0.0,
            duration_sec=2.0,
            original_prompt="formula one car accelerating on track",
            refined_prompt="",
        )
        out = orch.run_event_agent(event, prompt_only_mode=True)
        trace = out.agent_trace[-1]

        self.assertEqual(trace["action"], "ACCEPT")
        self.assertFalse(trace["acceptance_blocked_by_uncertainty"])


if __name__ == "__main__":
    unittest.main()
