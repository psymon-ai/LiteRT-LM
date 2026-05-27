# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for audio LoRA graph contract checking."""

from __future__ import annotations

import unittest

from litert_lm_builder import audio_lora_converter
from litert_lm_builder import audio_lora_graph_checker


def _spec(name: str, is_consumed: bool = True):
  return audio_lora_converter.TargetTensorSpec(
      name=name,
      shape=(2, 4),
      tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT16,
      byte_size=16,
      is_consumed=is_consumed,
  )


def _expected_specs(layers: int = 1):
  return {
      name: _spec(name)
      for name in audio_lora_graph_checker.expected_audio_lora_input_names(
          layers
      )
  }


class AudioLoraGraphCheckerTest(unittest.TestCase):

  def test_expected_audio_lora_input_names(self):
    self.assertEqual(
        audio_lora_graph_checker.expected_audio_lora_input_names(1)[:4],
        (
            "lora_audio_attn_q_a_weight_0",
            "lora_audio_attn_q_b_weight_0",
            "lora_audio_attn_k_a_weight_0",
            "lora_audio_attn_k_b_weight_0",
        ),
    )
    self.assertIn(
        "lora_audio_output_proj_b_weight_0",
        audio_lora_graph_checker.expected_audio_lora_input_names(
            1, require_output_proj=True
        ),
    )

  def test_report_passes_when_all_expected_inputs_are_consumed(self):
    report = audio_lora_graph_checker.inspect_audio_lora_target_specs(
        _expected_specs(layers=2),
        expected_audio_layers=2,
    )

    self.assertTrue(report.passed)
    self.assertEqual(report.unconsumed_target_inputs, ())
    self.assertEqual(report.missing_expected_inputs, ())
    self.assertEqual(report.module_counts["attn_q"]["inputs"], 4)
    self.assertEqual(report.module_counts["lconv_start"]["consumed"], 4)

  def test_report_fails_on_unconsumed_inputs(self):
    specs = _expected_specs(layers=1)
    specs["lora_audio_attn_k_a_weight_0"] = _spec(
        "lora_audio_attn_k_a_weight_0", is_consumed=False
    )
    specs["lora_audio_lconv_start_b_weight_0"] = _spec(
        "lora_audio_lconv_start_b_weight_0", is_consumed=False
    )

    report = audio_lora_graph_checker.inspect_audio_lora_target_specs(
        specs,
        expected_audio_layers=1,
    )

    self.assertFalse(report.passed)
    self.assertEqual(
        report.unconsumed_target_inputs,
        (
            "lora_audio_attn_k_a_weight_0",
            "lora_audio_lconv_start_b_weight_0",
        ),
    )
    with self.assertRaisesRegex(ValueError, "not consumed"):
      audio_lora_graph_checker.validate_audio_lora_graph_report(report)

  def test_report_fails_on_missing_expected_inputs(self):
    specs = _expected_specs(layers=1)
    del specs["lora_audio_attn_v_b_weight_0"]

    report = audio_lora_graph_checker.inspect_audio_lora_target_specs(
        specs,
        expected_audio_layers=1,
    )

    self.assertFalse(report.passed)
    self.assertEqual(
        report.missing_expected_inputs,
        ("lora_audio_attn_v_b_weight_0",),
    )
    with self.assertRaisesRegex(ValueError, "missing expected"):
      audio_lora_graph_checker.validate_audio_lora_graph_report(report)

  def test_no_audio_lora_inputs_is_invalid(self):
    report = audio_lora_graph_checker.inspect_audio_lora_target_specs({})

    self.assertFalse(report.passed)
    with self.assertRaisesRegex(ValueError, "no LiteRT-LM audio LoRA"):
      audio_lora_graph_checker.validate_audio_lora_graph_report(report)

  def test_allow_flags_are_for_bringup_only(self):
    specs = {
        "lora_audio_attn_q_a_weight_0": _spec(
            "lora_audio_attn_q_a_weight_0", is_consumed=False
        )
    }
    report = audio_lora_graph_checker.inspect_audio_lora_target_specs(
        specs,
        expected_audio_layers=1,
    )

    audio_lora_graph_checker.validate_audio_lora_graph_report(
        report,
        allow_missing_expected_inputs=True,
        allow_unconsumed_target_inputs=True,
    )


if __name__ == "__main__":
  unittest.main()
