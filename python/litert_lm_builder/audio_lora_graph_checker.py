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

"""Checks the LiteRT-LM audio LoRA input contract for a target graph.

This module validates the graph side of the audio LoRA ABI.  A graph that
lists LoRA inputs but does not feed those inputs into any operator is not a
valid deployment target, even if a PEFT sidecar can be compiled for it.
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from litert_lm_builder import audio_lora_converter


_AUDIO_ATTENTION_RE = re.compile(
    r"^lora_audio_attn_(?P<proj>[qkvo])_(?P<side>[ab])_weight_(?P<layer>\d+)$"
)
_AUDIO_MODULE_RE = re.compile(
    r"^lora_audio_"
    r"(?P<module>ff1_l1|ff1_l2|ff2_l1|ff2_l2|lconv_start|lconv_end|"
    r"output_proj)"
    r"_(?P<side>[ab])_weight_(?P<layer>\d+)$"
)

_AUDIO_TOWER_MODULES = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "ff1_l1",
    "ff1_l2",
    "lconv_start",
    "lconv_end",
    "ff2_l1",
    "ff2_l2",
)
_SIDES = ("a", "b")


@dataclasses.dataclass(frozen=True)
class AudioLoraInput:
  """A parsed audio LoRA target-graph input."""

  name: str
  module: str
  side: str
  layer: int
  shape: tuple[int, ...]
  tensor_type: int
  byte_size: int
  is_consumed: bool

  def to_json_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AudioLoraGraphReport:
  """Summary of the audio LoRA graph contract."""

  target_tflite: str | None
  expected_audio_layers: int | None
  require_output_proj: bool
  passed: bool
  target_inputs: tuple[str, ...]
  unconsumed_target_inputs: tuple[str, ...]
  missing_expected_inputs: tuple[str, ...]
  module_counts: Mapping[str, Mapping[str, int]]

  def to_json_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _sort_key(name: str) -> tuple[int, int, int, str]:
  parsed = parse_audio_lora_input_name(name)
  if parsed is None:
    return (999, 999, 999, name)
  module_order = {
      module: index for index, module in enumerate(_AUDIO_TOWER_MODULES)
  }
  module_order["output_proj"] = len(module_order)
  side_order = {"a": 0, "b": 1}
  return (
      parsed["layer"],
      module_order.get(parsed["module"], 999),
      side_order.get(parsed["side"], 999),
      name,
  )


def parse_audio_lora_input_name(name: str) -> dict[str, Any] | None:
  """Parses a LiteRT-LM audio LoRA input name."""

  match = _AUDIO_ATTENTION_RE.fullmatch(name)
  if match:
    return {
        "module": f"attn_{match.group('proj')}",
        "side": match.group("side"),
        "layer": int(match.group("layer")),
    }
  match = _AUDIO_MODULE_RE.fullmatch(name)
  if match:
    return {
        "module": match.group("module"),
        "side": match.group("side"),
        "layer": int(match.group("layer")),
    }
  return None


def expected_audio_lora_input_names(
    expected_audio_layers: int,
    require_output_proj: bool = False,
) -> tuple[str, ...]:
  """Returns the expected audio LoRA ABI names for a Gemma-style audio tower."""

  names = []
  for layer in range(expected_audio_layers):
    for module in _AUDIO_TOWER_MODULES:
      for side in _SIDES:
        if module.startswith("attn_"):
          proj = module.removeprefix("attn_")
          names.append(f"lora_audio_attn_{proj}_{side}_weight_{layer}")
        else:
          names.append(f"lora_audio_{module}_{side}_weight_{layer}")
  if require_output_proj:
    names.extend(
        f"lora_audio_output_proj_{side}_weight_0" for side in _SIDES
    )
  return tuple(sorted(names, key=_sort_key))


def inspect_audio_lora_target_specs(
    target_specs: Mapping[str, audio_lora_converter.TargetTensorSpec],
    expected_audio_layers: int | None = None,
    require_output_proj: bool = False,
    target_tflite: str | None = None,
) -> AudioLoraGraphReport:
  """Checks pre-extracted audio LoRA target specs."""

  parsed_inputs = []
  for name, spec in target_specs.items():
    parsed = parse_audio_lora_input_name(name)
    if parsed is None:
      continue
    parsed_inputs.append(
        AudioLoraInput(
            name=name,
            module=parsed["module"],
            side=parsed["side"],
            layer=parsed["layer"],
            shape=tuple(spec.shape),
            tensor_type=spec.tensor_type,
            byte_size=spec.byte_size,
            is_consumed=spec.is_consumed,
        )
    )
  parsed_inputs = sorted(parsed_inputs, key=lambda item: _sort_key(item.name))
  target_names = tuple(item.name for item in parsed_inputs)
  unconsumed = tuple(item.name for item in parsed_inputs if not item.is_consumed)

  if expected_audio_layers is None:
    missing = ()
  else:
    expected_names = expected_audio_lora_input_names(
        expected_audio_layers=expected_audio_layers,
        require_output_proj=require_output_proj,
    )
    present = set(target_names)
    missing = tuple(name for name in expected_names if name not in present)

  module_counts: dict[str, dict[str, int]] = {}
  for item in parsed_inputs:
    counts = module_counts.setdefault(
        item.module, {"inputs": 0, "consumed": 0, "unconsumed": 0}
    )
    counts["inputs"] += 1
    if item.is_consumed:
      counts["consumed"] += 1
    else:
      counts["unconsumed"] += 1

  passed = bool(target_names) and not unconsumed and not missing
  return AudioLoraGraphReport(
      target_tflite=target_tflite,
      expected_audio_layers=expected_audio_layers,
      require_output_proj=require_output_proj,
      passed=passed,
      target_inputs=target_names,
      unconsumed_target_inputs=unconsumed,
      missing_expected_inputs=missing,
      module_counts=module_counts,
  )


def inspect_audio_lora_target_graph(
    target_tflite_path: os.PathLike[str] | str,
    expected_audio_layers: int | None = None,
    require_output_proj: bool = False,
    signature_subgraph_index: int = 0,
) -> AudioLoraGraphReport:
  """Checks a TFLite target graph for the LiteRT-LM audio LoRA ABI."""

  specs = audio_lora_converter.extract_lora_input_specs(
      target_tflite_path,
      signature_subgraph_index=signature_subgraph_index,
  )
  return inspect_audio_lora_target_specs(
      specs,
      expected_audio_layers=expected_audio_layers,
      require_output_proj=require_output_proj,
      target_tflite=str(target_tflite_path),
  )


def validate_audio_lora_graph_report(
    report: AudioLoraGraphReport,
    allow_missing_expected_inputs: bool = False,
    allow_unconsumed_target_inputs: bool = False,
) -> None:
  """Raises ValueError if the report is not release-valid."""

  if not report.target_inputs:
    raise ValueError("target graph has no LiteRT-LM audio LoRA input tensors")
  if report.missing_expected_inputs and not allow_missing_expected_inputs:
    preview = ", ".join(report.missing_expected_inputs[:8])
    if len(report.missing_expected_inputs) > 8:
      preview += ", ..."
    raise ValueError(f"target graph is missing expected audio LoRA inputs: {preview}")
  if report.unconsumed_target_inputs and not allow_unconsumed_target_inputs:
    preview = ", ".join(report.unconsumed_target_inputs[:8])
    if len(report.unconsumed_target_inputs) > 8:
      preview += ", ..."
    raise ValueError(
        "target graph has audio LoRA inputs that are not consumed by any op: "
        f"{preview}"
    )

