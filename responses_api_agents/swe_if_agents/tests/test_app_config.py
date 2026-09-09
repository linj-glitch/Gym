# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWEIFWrapperConfig validates the reward knob at server start, and run() folds / masks as documented.

Needs the gym's dependencies (app.py imports the SWE wrapper: ray, tomlkit, ...), so it is SKIPPED under the plain
interpreter the other tests use. Run it from an environment with those packages, e.g.

    PYTHONPATH=<Gym> uv run --no-project --python 3.12 --with pydantic --with fastapi ... python -m unittest \
        responses_api_agents/swe_if_agents/tests/test_app_config.py
"""
import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

try:
    from pydantic import ValidationError

    from responses_api_agents.swe_agents import app as swe
    from responses_api_agents.swe_if_agents import app as if_app
    from responses_api_agents.swe_if_agents.if_constraints.reward import REWARD_MODES
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

    DEPS = True
except ImportError as exc:  # pragma: no cover - plain interpreter
    DEPS = False
    SKIP_REASON = f"gym dependencies not importable here: {exc}"


BASE = {"host": "127.0.0.1", "port": 1, "entrypoint": "", "name": "swe_if", "model_server": {"type": "responses_api_models", "name": "m"}}


def _config(**over):
    return if_app.SWEIFWrapperConfig.model_validate(BASE | over)


@unittest.skipUnless(DEPS, "gym dependencies" if DEPS else SKIP_REASON)
class TestConfigValidation(unittest.TestCase):
    def test_literal_is_pinned_to_reward_modes(self):
        self.assertEqual(tuple(if_app.RewardMode.__args__), REWARD_MODES)

    def test_default_is_outcome_and_every_mode_validates_with_grading(self):
        self.assertEqual(_config().reward_mode, "outcome")
        for mode in REWARD_MODES:
            self.assertEqual(_config(reward_mode=mode, if_grading=True).reward_mode, mode)

    def test_unknown_or_miscased_mode_is_rejected_at_startup(self):
        for bad in ("Strict", "stric", "shape", ""):
            with self.assertRaises(ValidationError, msg=bad):
                _config(reward_mode=bad)

    def test_training_mode_without_grading_is_rejected_at_startup(self):
        for mode in ("shaped", "strict", "tiered"):
            with self.assertRaises(ValidationError, msg=mode) as ctx:
                _config(reward_mode=mode, if_grading=False)
            self.assertIn("if_grading", str(ctx.exception))
        # the benchmark configuration (records off, outcome reward) is fine
        self.assertFalse(_config(reward_mode="outcome", if_grading=False).if_grading)

    def test_partial_reward_range_is_checked_at_startup(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValidationError, msg=bad):
                _config(success_partial_reward=bad)
        self.assertEqual(_config(success_partial_reward=1.0).success_partial_reward, 1.0)

    def test_mask_contract_exists_on_the_base_wrapper(self):
        """NeMo-RL reads full_result.instance_config.mask_sample; SWEIFVerifyResponse mirrors it at top level."""
        self.assertIn("mask_sample", swe.SWEBenchWrapperInstanceConfig.model_fields)
        self.assertIs(swe.SWEBenchVerifyResponse.model_fields["instance_config"].annotation, swe.SWEBenchWrapperInstanceConfig)
        self.assertIn("mask_sample", if_app.SWEIFVerifyResponse.model_fields)
        self.assertIn("reward_settings_error", if_app.SWEIFVerifyResponse.model_fields)


# ------------------------------------------------------------------ run(): fold + mask on a stubbed episode
def _instance_config(config: dict, body: dict, mask_sample: bool = False) -> dict:
    """A fully populated SWEBenchWrapperInstanceConfig (its many path fields are dummies)."""
    paths = (
        "metrics_fpath", "persistent_dir", "instance_dataset_path", "agent_instance_dataset_path", "trajectories_root",
        "prediction_path", "output_for_eval_mounted_path", "output_for_eval_path", "model_patch_path", "agent_script_path",
        "final_eval_apptainer_spinup_timestamp_fpath", "final_eval_apptainer_spinup_timestamp_mounted_fpath",
        "generation_apptainer_spinup_timestamp_fpath", "generation_apptainer_spinup_timestamp_mounted_fpath",
        "base_mounted_dir", "profiling_dir", "profiling_mounted_dir", "swebench_setup_dir", "r2e_gym_setup_dir",
        "swe_rebench_setup_dir", "swebench_multilingual_setup_dir", "base_results_dir",
    )
    return config | {p: f"/tmp/{p}" for p in paths} | {
        "ng_global_config_dict_str": "", "model_server_name": "m", "run_session_id": "s",
        "problem_info": {"instance_id": "x__y-1"}, "body": body, "ray_queue_timestamp": 0.0, "inference_params": {},
        "agent_run_id": "r", "container": "c", "eval_dir_in_openhands": "e", "openhands_config_file_path": "o",
        "mask_sample": mask_sample,
    }


def _output(text: str, call_name: str):
    return [
        {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": text, "annotations": []}]},
        {"type": "function_call", "id": "fc1", "call_id": "c1", "name": call_name, "arguments": "{}", "status": "completed"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "message", "id": "m2", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "done", "annotations": []}]},
    ]


@unittest.skipUnless(DEPS, "gym dependencies" if DEPS else SKIP_REASON)
class TestRunFoldAndMask(unittest.TestCase):
    """run() with the base episode stubbed: the fold lands in `reward`, a grading error masks through instance_config."""

    TAG_BASH = {"id": "t#c1", "verifier_parameter": {"template": "turn_output", "trigger": {"tool": "BASH_TOOL_NAME"},
                                                     "obligation": {"match": "prefix", "value": "[RUN]"}}}

    def _run(self, config: dict, metadata: dict, output: list, resolved: bool = True, base_mask: bool = False):
        import json

        cfg = _config(**config)
        body_dict = {"input": [{"type": "message", "role": "user", "content": "fix it"}], "metadata": metadata}
        body = swe.BaseRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(**body_dict))

        async def fake_base_run(self_, body_):
            response = NeMoGymResponse(id="r", created_at=1, model="m", object="response", output=output,
                                       parallel_tool_calls=True, tool_choice="auto", tools=[])
            return swe.SWEBenchVerifyResponse(
                responses_create_params=body_dict, response=response, reward=1.0 if resolved else 0.0, resolved=resolved,
                instance_config=_instance_config(json.loads(cfg.model_dump_json()), body_dict, mask_sample=base_mask),
            )

        # model_post_init would set up the agent harness (clones, datasets): skipped, the episode itself is stubbed
        with mock.patch.object(swe.SWEBenchWrapper, "run", fake_base_run), mock.patch.object(
            if_app.SWEIFWrapper, "model_post_init", lambda self_, ctx: None
        ):
            wrapper = if_app.SWEIFWrapper.model_construct(config=cfg, server_client=mock.MagicMock())
            return asyncio.run(wrapper.run(body))

    def _md(self, constraints, **extra):
        import json

        return {"tool_name_overrides": json.dumps({"BASH_TOOL_NAME": "shell"}),
                "sdg_item": json.dumps({"type": "fresh", "constraints": constraints})} | extra

    def test_strict_fold_and_outcome_identity(self):
        violating = _output("running it", "shell")  # the bash-calling turn is not tagged [RUN]
        strict = self._run({"reward_mode": "strict"}, self._md([self.TAG_BASH]), violating)
        self.assertEqual((strict.reward, strict.task_reward, strict.reward_mode), (0.0, 1.0, "strict"))
        self.assertEqual((strict.n_constraints, strict.n_applicable, strict.n_grading_errors), (1, 1, 0))
        self.assertEqual([(v["turn"], v["passed"]) for v in strict.turn_verdicts], [(1, False)])
        self.assertFalse(strict.mask_sample)
        self.assertFalse(strict.instance_config.mask_sample)
        self.assertIsNone(strict.reward_settings_error)
        outcome = self._run({}, self._md([self.TAG_BASH]), violating)
        self.assertEqual((outcome.reward, outcome.reward_mode), (1.0, "outcome"))
        self.assertEqual(outcome.if_constraints[0]["id"], "t#c1")

    def test_grading_error_passes_the_task_through_and_masks_the_sample(self):
        md = self._md([self.TAG_BASH])
        md["sdg_item"] = "{not json"  # the grader's catch-all record
        out = self._run({"reward_mode": "strict"}, md, _output("[RUN] ok", "shell"))
        self.assertEqual(out.reward, 1.0)
        self.assertEqual((out.n_constraints, out.n_grading_errors), (0, 1))
        self.assertTrue(out.mask_sample)
        self.assertTrue(out.instance_config.mask_sample, "NeMo-RL reads this one")
        self.assertEqual(out.if_constraints[0]["id"], "<grading_error>")
        # outcome mode: same pass-through, no mask
        out = self._run({}, md, _output("[RUN] ok", "shell"))
        self.assertEqual((out.reward, out.mask_sample, out.instance_config.mask_sample), (1.0, False, False))

    def test_base_wrapper_mask_is_preserved(self):
        out = self._run({"reward_mode": "strict"}, self._md([self.TAG_BASH]), _output("[RUN] ok", "shell"), base_mask=True)
        self.assertEqual(out.reward, 1.0)
        self.assertTrue(out.instance_config.mask_sample and out.mask_sample)
        self.assertEqual(out.n_grading_errors, 0)

    def test_bad_row_override_falls_back_and_is_reported(self):
        md = self._md([self.TAG_BASH], reward_mode="stric", constraint_alpha="abc")
        out = self._run({"reward_mode": "strict"}, md, _output("running it", "shell"))
        self.assertEqual((out.reward, out.reward_mode, out.constraint_alpha), (0.0, "strict", 1.0))
        self.assertIn("reward_mode='stric'", out.reward_settings_error)
        self.assertIn("constraint_alpha='abc'", out.reward_settings_error)
        # a valid override is applied
        out = self._run({"reward_mode": "strict"}, self._md([self.TAG_BASH], reward_mode="tiered", success_partial_reward="0.25"),
                        _output("running it", "shell"))
        self.assertEqual((out.reward, out.reward_mode, out.success_partial_reward, out.reward_settings_error), (0.25, "tiered", 0.25, None))

    def test_row_without_constraints_is_an_ordinary_task_row(self):
        out = self._run({"reward_mode": "strict"}, {}, _output("running it", "shell"))
        self.assertEqual((out.reward, out.n_constraints, out.mask_sample), (1.0, 0, False))
        self.assertIsNone(out.if_constraints)


if __name__ == "__main__":
    unittest.main()
