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

"""Command-line entry point for checking audio LoRA target graphs."""

from __future__ import annotations

import argparse
import json
import pathlib

from litert_lm_builder import audio_lora_graph_checker


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Validate that a target audio TFLite graph exposes and consumes the "
          "LiteRT-LM audio LoRA ABI inputs."
      )
  )
  parser.add_argument("--target_tflite", required=True)
  parser.add_argument(
      "--expected_audio_layers",
      type=int,
      help=(
          "Optional number of audio tower layers expected by the ABI. For "
          "Gemma audio tower graphs this is typically 12. If omitted, the "
          "checker only validates the LoRA inputs that are present."
      ),
  )
  parser.add_argument(
      "--require_output_proj",
      action="store_true",
      help="Require lora_audio_output_proj_{a,b}_weight_0 in the graph.",
  )
  parser.add_argument(
      "--signature_subgraph_index",
      type=int,
      default=0,
      help="TFLite subgraph index to inspect.",
  )
  parser.add_argument(
      "--allow_missing_expected_inputs",
      action="store_true",
      help="Allow missing expected ABI inputs. Intended only for graph bring-up.",
  )
  parser.add_argument(
      "--allow_unconsumed_target_inputs",
      action="store_true",
      help="Allow LoRA inputs that are not consumed by any op. Intended only for debugging.",
  )
  parser.add_argument("--report_json")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  report = audio_lora_graph_checker.inspect_audio_lora_target_graph(
      target_tflite_path=args.target_tflite,
      expected_audio_layers=args.expected_audio_layers,
      require_output_proj=args.require_output_proj,
      signature_subgraph_index=args.signature_subgraph_index,
  )
  audio_lora_graph_checker.validate_audio_lora_graph_report(
      report,
      allow_missing_expected_inputs=args.allow_missing_expected_inputs,
      allow_unconsumed_target_inputs=args.allow_unconsumed_target_inputs,
  )
  report_json = json.dumps(report.to_json_dict(), indent=2, sort_keys=True)
  if args.report_json:
    pathlib.Path(args.report_json).write_text(report_json + "\n", encoding="utf-8")
  print(report_json)


if __name__ == "__main__":
  main()

