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

"""Rewrites an audio encoder TFLite graph to expose runtime audio LoRA inputs.

The rewriter consumes a mapping manifest that identifies, for each PEFT audio
LoRA module, the source activation, projection output, and float landing tensor
inside the target TFLite graph.  It then inserts a runtime-bound float branch:

  source -> BATCH_MATMUL(lora_A) -> BATCH_MATMUL(lora_B)
         -> optional cloned view chain -> ADD at float landing

It does not bake LoRA deltas into quantized weights.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import json
import os
import pathlib
import struct
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import flatbuffers


_ATTENTION_MODULE_NAMES = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "post": "attn_o",
    "o_proj": "attn_o",
}

_AUDIO_MODULE_NAMES = {
    "ff1_l1": "ff1_l1",
    "ff1_l2": "ff1_l2",
    "lconv_start": "lconv_start",
    "lconv_end": "lconv_end",
    "ff2_l1": "ff2_l1",
    "ff2_l2": "ff2_l2",
}


@dataclasses.dataclass(frozen=True)
class TfliteObjectSchema:
  """TFLite object API classes required for graph rewriting."""

  AddOptionsT: Any
  BatchMatMulOptionsT: Any
  BufferT: Any
  BuiltinOperator: Any
  BuiltinOptions: Any
  Model: Any
  ModelT: Any
  OperatorCodeT: Any
  OperatorT: Any
  ReshapeOptionsT: Any
  SignatureDefT: Any
  SubGraphT: Any
  TensorMapT: Any
  TensorT: Any
  TensorType: Any


@dataclasses.dataclass(frozen=True)
class ManifestTarget:
  hf_module: str
  layer: int
  module_key: str
  source_idx: int
  consumer_output_idx: int
  consumer_op_idx: int
  landing_indexes: tuple[int, ...]
  view_op_indexes: tuple[int, ...]
  view_output_indexes: tuple[tuple[int, ...], ...]


@dataclasses.dataclass(frozen=True)
class RewriteReport:
  input: str
  output: str
  targets: int
  lora_inputs_added: int
  lora_inputs_in_signature: bool
  view_ops_cloned: int
  adds: int
  bytes: int

  def to_json_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _add_schema_search_paths(schema_path: os.PathLike[str] | str | None) -> None:
  if schema_path:
    import sys

    sys.path.insert(0, str(schema_path))
  for raw in os.environ.get("TFLITE_SCHEMA_OBJECT_PATH", "").split(os.pathsep):
    if raw:
      import sys

      sys.path.insert(0, raw)


def import_tflite_object_schema(
    schema_path: os.PathLike[str] | str | None = None,
) -> TfliteObjectSchema:
  """Imports a generated TFLite schema with the Python object API.

  The schema may come from LiteRT-LM's Bazel-generated monolithic
  `tflite_schema_py_generated.py`, or from a directory containing split
  `tflite.*` modules produced by `flatc --python --gen-object-api`.
  """

  _add_schema_search_paths(schema_path)
  try:
    from litert_lm_builder import tflite_schema_py_generated as schema  # pylint: disable=g-import-not-at-top

    return TfliteObjectSchema(
        AddOptionsT=schema.AddOptionsT,
        BatchMatMulOptionsT=schema.BatchMatMulOptionsT,
        BufferT=schema.BufferT,
        BuiltinOperator=schema.BuiltinOperator,
        BuiltinOptions=schema.BuiltinOptions,
        Model=schema.Model,
        ModelT=schema.ModelT,
        OperatorCodeT=schema.OperatorCodeT,
        OperatorT=schema.OperatorT,
        ReshapeOptionsT=schema.ReshapeOptionsT,
        SignatureDefT=schema.SignatureDefT,
        SubGraphT=schema.SubGraphT,
        TensorMapT=schema.TensorMapT,
        TensorT=schema.TensorT,
        TensorType=schema.TensorType,
    )
  except (ImportError, AttributeError):
    pass

  modules = {
      "AddOptions": importlib.import_module("tflite.AddOptions"),
      "BatchMatMulOptions": importlib.import_module("tflite.BatchMatMulOptions"),
      "Buffer": importlib.import_module("tflite.Buffer"),
      "BuiltinOperator": importlib.import_module("tflite.BuiltinOperator"),
      "BuiltinOptions": importlib.import_module("tflite.BuiltinOptions"),
      "Model": importlib.import_module("tflite.Model"),
      "Operator": importlib.import_module("tflite.Operator"),
      "OperatorCode": importlib.import_module("tflite.OperatorCode"),
      "ReshapeOptions": importlib.import_module("tflite.ReshapeOptions"),
      "SignatureDef": importlib.import_module("tflite.SignatureDef"),
      "SubGraph": importlib.import_module("tflite.SubGraph"),
      "Tensor": importlib.import_module("tflite.Tensor"),
      "TensorMap": importlib.import_module("tflite.TensorMap"),
      "TensorType": importlib.import_module("tflite.TensorType"),
  }
  required = (
      ("AddOptions", "AddOptionsT"),
      ("BatchMatMulOptions", "BatchMatMulOptionsT"),
      ("Buffer", "BufferT"),
      ("Model", "ModelT"),
      ("Operator", "OperatorT"),
      ("OperatorCode", "OperatorCodeT"),
      ("ReshapeOptions", "ReshapeOptionsT"),
      ("SignatureDef", "SignatureDefT"),
      ("SubGraph", "SubGraphT"),
      ("Tensor", "TensorT"),
      ("TensorMap", "TensorMapT"),
  )
  missing = [
      f"tflite.{module}.{attr}"
      for module, attr in required
      if not hasattr(modules[module], attr)
  ]
  if missing:
    raise ImportError(
        "TFLite graph rewriting requires a schema generated with "
        f"`flatc --python --gen-object-api`; missing {', '.join(missing)}"
    )
  return TfliteObjectSchema(
      AddOptionsT=modules["AddOptions"].AddOptionsT,
      BatchMatMulOptionsT=modules["BatchMatMulOptions"].BatchMatMulOptionsT,
      BufferT=modules["Buffer"].BufferT,
      BuiltinOperator=modules["BuiltinOperator"].BuiltinOperator,
      BuiltinOptions=modules["BuiltinOptions"].BuiltinOptions,
      Model=modules["Model"].Model,
      ModelT=modules["Model"].ModelT,
      OperatorCodeT=modules["OperatorCode"].OperatorCodeT,
      OperatorT=modules["Operator"].OperatorT,
      ReshapeOptionsT=modules["ReshapeOptions"].ReshapeOptionsT,
      SignatureDefT=modules["SignatureDef"].SignatureDefT,
      SubGraphT=modules["SubGraph"].SubGraphT,
      TensorMapT=modules["TensorMap"].TensorMapT,
      TensorT=modules["Tensor"].TensorT,
      TensorType=modules["TensorType"].TensorType,
  )


def audio_lora_input_names(module_key: str, layer: int) -> tuple[str, str]:
  if module_key in _ATTENTION_MODULE_NAMES:
    stem = f"lora_audio_{_ATTENTION_MODULE_NAMES[module_key]}"
  elif module_key in _AUDIO_MODULE_NAMES:
    stem = f"lora_audio_{_AUDIO_MODULE_NAMES[module_key]}"
  else:
    raise ValueError(f"unsupported audio LoRA module key: {module_key}")
  return f"{stem}_a_weight_{layer}", f"{stem}_b_weight_{layer}"


def _shape(tensor: Any) -> tuple[int, ...]:
  return tuple(int(dim) for dim in (tensor.shape or ()))


def _as_i32(values: Sequence[int]) -> list[int]:
  return [int(value) for value in values]


def _make_buffer(schema: TfliteObjectSchema, payload: bytes | None = None) -> Any:
  buffer = schema.BufferT()
  if payload:
    buffer.data = list(payload)
  return buffer


def _make_shape_buffer(
    schema: TfliteObjectSchema, target_shape: tuple[int, ...]
) -> Any:
  return _make_buffer(
      schema, struct.pack("<" + "i" * len(target_shape), *target_shape)
  )


def _make_tensor(
    schema: TfliteObjectSchema,
    tensor_name: str,
    tensor_shape: tuple[int, ...],
    tensor_type: int,
    buffer_index: int = 0,
) -> Any:
  tensor = schema.TensorT()
  tensor.name = tensor_name
  tensor.shape = _as_i32(tensor_shape)
  tensor.type = int(tensor_type)
  tensor.buffer = int(buffer_index)
  return tensor


def _make_op(
    schema: TfliteObjectSchema,
    opcode_index: int,
    inputs: Sequence[int],
    outputs: Sequence[int],
    options_type: int,
    options: Any,
) -> Any:
  op = schema.OperatorT()
  op.opcodeIndex = int(opcode_index)
  op.inputs = _as_i32(inputs)
  op.outputs = _as_i32(outputs)
  op.builtinOptionsType = int(options_type)
  op.builtinOptions = options
  return op


def _make_bmm(
    schema: TfliteObjectSchema, opcode_index: int, lhs: int, rhs: int, out: int
) -> Any:
  return _make_op(
      schema,
      opcode_index,
      [lhs, rhs],
      [out],
      schema.BuiltinOptions.BatchMatMulOptions,
      schema.BatchMatMulOptionsT(),
  )


def _make_unary(
    schema: TfliteObjectSchema, opcode_index: int, src: int, out: int
) -> Any:
  return _make_op(schema, opcode_index, [src], [out], schema.BuiltinOptions.NONE, None)


def _make_reshape(
    schema: TfliteObjectSchema,
    opcode_index: int,
    tensor: int,
    shape_tensor: int,
    out: int,
    new_shape: tuple[int, ...],
) -> Any:
  opts = schema.ReshapeOptionsT()
  opts.newShape = _as_i32(new_shape)
  return _make_op(
      schema,
      opcode_index,
      [tensor, shape_tensor],
      [out],
      schema.BuiltinOptions.ReshapeOptions,
      opts,
  )


def _make_add(
    schema: TfliteObjectSchema, opcode_index: int, lhs: int, rhs: int, out: int
) -> Any:
  return _make_op(
      schema,
      opcode_index,
      [lhs, rhs],
      [out],
      schema.BuiltinOptions.AddOptions,
      schema.AddOptionsT(),
  )


def _builtin_code(opcode: Any) -> int:
  code = int(getattr(opcode, "builtinCode", 0) or 0)
  deprecated = int(getattr(opcode, "deprecatedBuiltinCode", 0) or 0)
  return code or deprecated


def _opcode_index(model: Any, schema: TfliteObjectSchema, builtin: int) -> int:
  for index, code in enumerate(model.operatorCodes):
    if _builtin_code(code) == int(builtin):
      return index
  op = schema.OperatorCodeT()
  op.builtinCode = int(builtin)
  op.deprecatedBuiltinCode = int(builtin)
  op.version = 1
  model.operatorCodes.append(op)
  return len(model.operatorCodes) - 1


def _producer_map(subgraph: Any) -> dict[int, int]:
  producers = {}
  for op_index, op in enumerate(subgraph.operators):
    for output in [] if op.outputs is None else op.outputs:
      producers[int(output)] = op_index
  return producers


def _replace_inputs(
    op: Any, replacement: Mapping[int, int], skip: set[int] | None = None
) -> None:
  if op.inputs is None:
    return
  skip = skip or set()
  op.inputs = _as_i32(
      [
          int(idx) if int(idx) in skip else replacement.get(int(idx), int(idx))
          for idx in op.inputs
      ]
  )


def _clone_view_op_with_delta_outputs(
    schema: TfliteObjectSchema,
    subgraph: Any,
    original_op: Any,
    mapped: Mapping[int, int],
    stem: str,
) -> tuple[Any, dict[int, int]]:
  original_inputs = [
      int(idx) for idx in ([] if original_op.inputs is None else original_op.inputs)
  ]
  original_outputs = [
      int(idx) for idx in ([] if original_op.outputs is None else original_op.outputs)
  ]
  cloned = copy.deepcopy(original_op)
  cloned.inputs = _as_i32([mapped.get(idx, idx) for idx in original_inputs])
  new_outputs = {}
  for output in original_outputs:
    new_index = len(subgraph.tensors)
    subgraph.tensors.append(
        _make_tensor(
            schema,
            f"{stem}/view_of_{output}",
            _shape(subgraph.tensors[output]),
            schema.TensorType.FLOAT32,
        )
    )
    new_outputs[output] = new_index
  cloned.outputs = _as_i32([new_outputs[idx] for idx in original_outputs])
  return cloned, new_outputs


def _add_graph_input(
    model: Any,
    subgraph: Any,
    schema: TfliteObjectSchema,
    name: str,
    shape: tuple[int, ...],
    dtype: int,
    expose_in_signature: bool,
) -> int:
  tensor_index = len(subgraph.tensors)
  subgraph.tensors.append(_make_tensor(schema, name, shape, dtype, 0))
  subgraph.inputs = _as_i32(
      ([] if subgraph.inputs is None else list(subgraph.inputs)) + [tensor_index]
  )
  if expose_in_signature and model.signatureDefs:
    signature = model.signatureDefs[0]
    if signature.inputs is None:
      signature.inputs = []
    tensor_map = schema.TensorMapT()
    tensor_map.name = name.encode("utf-8")
    tensor_map.tensorIndex = tensor_index
    signature.inputs.append(tensor_map)
  return tensor_index


def parse_manifest_targets(manifest: Mapping[str, Any]) -> tuple[ManifestTarget, ...]:
  targets = []
  for entry in manifest.get("entries", []):
    if entry.get("stage") != "audio_after_dequant_graft":
      continue
    if entry.get("status") != "ready":
      raise ValueError(
          f"manifest contains non-ready audio target: {entry.get('hf_module')}"
      )
    path = entry["path"]
    landing_tensors = path.get("landing_tensors") or [path["landing_tensor"]]
    view_ops = path.get("view_ops", [])
    targets.append(
        ManifestTarget(
            hf_module=str(entry["hf_module"]),
            layer=int(entry["layer"]),
            module_key=str(entry["module_key"]),
            source_idx=int(entry["source_tensor"]["index"]),
            consumer_output_idx=int(entry["consumer_output_tensor"]["index"]),
            consumer_op_idx=int(entry["consumer_op"]["index"]),
            landing_indexes=tuple(int(tensor["index"]) for tensor in landing_tensors),
            view_op_indexes=tuple(int(op["index"]) for op in view_ops),
            view_output_indexes=tuple(
                tuple(int(output) for output in op.get("outputs", ()))
                for op in view_ops
            ),
        )
    )
  if not targets:
    raise ValueError("manifest contains no ready audio_after_dequant_graft entries")
  return tuple(sorted(targets, key=lambda item: (item.consumer_op_idx, item.hf_module)))


def _lora_shapes_by_module(
    manifest: Mapping[str, Any]
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], int]]:
  shapes = {}
  for entry in manifest.get("entries", []):
    if entry.get("stage") != "audio_after_dequant_graft":
      continue
    if entry.get("status") != "ready":
      continue
    lora = entry.get("lora", {})
    a_shape = tuple(int(dim) for dim in lora.get("a_prime_shape", ()))
    b_shape = tuple(int(dim) for dim in lora.get("b_prime_shape", ()))
    rank = int(lora.get("rank") or (a_shape[-1] if a_shape else 0))
    if not a_shape or not b_shape or rank <= 0:
      raise ValueError(f"missing LoRA shape in manifest entry: {entry.get('hf_module')}")
    shapes[str(entry["hf_module"])] = (a_shape, b_shape, rank)
  return shapes


def _validate_manifest_target(subgraph: Any, target: ManifestTarget) -> None:
  for tensor_index, label in (
      (target.source_idx, "source"),
      (target.consumer_output_idx, "consumer output"),
      *[(idx, "landing") for idx in target.landing_indexes],
  ):
    if tensor_index < 0 or tensor_index >= len(subgraph.tensors):
      raise ValueError(
          f"{target.hf_module}: {label} tensor index out of range: {tensor_index}"
      )
  if target.consumer_op_idx < 0 or target.consumer_op_idx >= len(subgraph.operators):
    raise ValueError(
        f"{target.hf_module}: consumer op index out of range: {target.consumer_op_idx}"
    )
  op = subgraph.operators[target.consumer_op_idx]
  outputs = [] if op.outputs is None else [int(idx) for idx in op.outputs]
  if target.consumer_output_idx not in outputs:
    raise ValueError(
        f"{target.hf_module}: manifest consumer output "
        f"{target.consumer_output_idx} is not produced by op {target.consumer_op_idx}"
    )


def _remap_target_ops(
    subgraph: Any, target: ManifestTarget, producers: Mapping[int, int]
) -> ManifestTarget:
  consumer_op_idx = producers.get(target.consumer_output_idx, target.consumer_op_idx)
  view_op_indexes = []
  for fallback_idx, outputs in zip(target.view_op_indexes, target.view_output_indexes):
    resolved_idx = fallback_idx
    for output in outputs:
      if output in producers:
        resolved_idx = producers[output]
        break
    view_op_indexes.append(int(resolved_idx))
  return dataclasses.replace(
      target,
      consumer_op_idx=int(consumer_op_idx),
      view_op_indexes=tuple(view_op_indexes),
  )


def rewrite_audio_lora_graph(
    input_path: os.PathLike[str] | str,
    manifest_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    *,
    schema_path: os.PathLike[str] | str | None = None,
    alignment: int = 16,
    expose_lora_in_signature: bool = True,
) -> RewriteReport:
  """Rewrites a target audio encoder TFLite graph for runtime LoRA binding."""

  schema = import_tflite_object_schema(schema_path)
  manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
  targets = parse_manifest_targets(manifest)
  shape_map = _lora_shapes_by_module(manifest)

  raw = pathlib.Path(input_path).read_bytes()
  model = schema.ModelT.InitFromObj(schema.Model.GetRootAs(raw, 0))
  if not model.subgraphs:
    raise ValueError("TFLite model contains no subgraphs")
  if model.operatorCodes is None:
    model.operatorCodes = []
  if model.buffers is None:
    model.buffers = [_make_buffer(schema)]
  subgraph = model.subgraphs[0]

  bmm_opcode = _opcode_index(model, schema, schema.BuiltinOperator.BATCH_MATMUL)
  dequantize_opcode = _opcode_index(model, schema, schema.BuiltinOperator.DEQUANTIZE)
  reshape_opcode = _opcode_index(model, schema, schema.BuiltinOperator.RESHAPE)
  add_opcode = _opcode_index(model, schema, schema.BuiltinOperator.ADD)
  producers = _producer_map(subgraph)

  branch_by_insert: dict[int, list[Any]] = defaultdict(list)
  replacement: dict[int, int] = {}
  skip_replacement_by_op: dict[int, set[int]] = defaultdict(set)
  total_view_ops = 0
  total_adds = 0
  total_inputs = 0

  for manifest_target in targets:
    target = _remap_target_ops(subgraph, manifest_target, producers)
    _validate_manifest_target(subgraph, target)

    a_shape, b_shape, rank = shape_map[target.hf_module]
    live_source_idx = replacement.get(target.source_idx, target.source_idx)
    source_shape = _shape(subgraph.tensors[live_source_idx])
    direct_shape = _shape(subgraph.tensors[target.consumer_output_idx])
    if not source_shape or source_shape[-1] != a_shape[0]:
      raise ValueError(
          f"{target.hf_module}: A shape {a_shape} does not match source "
          f"shape {source_shape}"
      )
    if not direct_shape or direct_shape[-1] != b_shape[-1]:
      raise ValueError(
          f"{target.hf_module}: B shape {b_shape} does not match direct "
          f"output shape {direct_shape}"
      )

    name_a, name_b = audio_lora_input_names(target.module_key, target.layer)
    a_idx = _add_graph_input(
        model,
        subgraph,
        schema,
        name_a,
        a_shape,
        schema.TensorType.FLOAT32,
        expose_lora_in_signature,
    )
    b_idx = _add_graph_input(
        model,
        subgraph,
        schema,
        name_b,
        b_shape,
        schema.TensorType.FLOAT32,
        expose_lora_in_signature,
    )
    total_inputs += 2

    stem = f"litert_lora_audio/L{target.layer:02d}_{target.module_key}"
    branch: list[Any] = []
    float_source_idx = live_source_idx
    if int(subgraph.tensors[live_source_idx].type) != int(schema.TensorType.FLOAT32):
      float_source_idx = len(subgraph.tensors)
      subgraph.tensors.append(
          _make_tensor(
              schema,
              f"{stem}/source_f32",
              source_shape,
              schema.TensorType.FLOAT32,
          )
      )
      branch.append(_make_unary(schema, dequantize_opcode, live_source_idx, float_source_idx))

    rank_shape = source_shape[:-1] + (rank,)
    bout_shape = source_shape[:-1] + (b_shape[-1],)
    rank_idx = len(subgraph.tensors)
    subgraph.tensors.append(
        _make_tensor(schema, f"{stem}/rank", rank_shape, schema.TensorType.FLOAT32)
    )
    bout_idx = len(subgraph.tensors)
    subgraph.tensors.append(
        _make_tensor(schema, f"{stem}/bout", bout_shape, schema.TensorType.FLOAT32)
    )
    branch.append(_make_bmm(schema, bmm_opcode, float_source_idx, a_idx, rank_idx))
    branch.append(_make_bmm(schema, bmm_opcode, rank_idx, b_idx, bout_idx))

    delta_idx = bout_idx
    if bout_shape != direct_shape:
      shape_buffer_idx = len(model.buffers)
      model.buffers.append(_make_shape_buffer(schema, direct_shape))
      shape_tensor_idx = len(subgraph.tensors)
      subgraph.tensors.append(
          _make_tensor(
              schema,
              f"{stem}/reshape_shape",
              (len(direct_shape),),
              schema.TensorType.INT32,
              shape_buffer_idx,
          )
      )
      reshaped_idx = len(subgraph.tensors)
      subgraph.tensors.append(
          _make_tensor(
              schema, f"{stem}/reshaped", direct_shape, schema.TensorType.FLOAT32
          )
      )
      branch.append(
          _make_reshape(
              schema, reshape_opcode, bout_idx, shape_tensor_idx, reshaped_idx, direct_shape
          )
      )
      delta_idx = reshaped_idx

    branch_by_insert[target.consumer_op_idx].extend(branch)
    direct_tensor = subgraph.tensors[target.consumer_output_idx]
    if int(direct_tensor.type) == int(schema.TensorType.FLOAT32):
      lhs = replacement.get(target.consumer_output_idx, target.consumer_output_idx)
      added_idx = len(subgraph.tensors)
      subgraph.tensors.append(
          _make_tensor(
              schema,
              f"{stem}/add_to_float_output",
              direct_shape,
              schema.TensorType.FLOAT32,
          )
      )
      branch_by_insert[target.consumer_op_idx].append(
          _make_add(schema, add_opcode, lhs, delta_idx, added_idx)
      )
      replacement[target.consumer_output_idx] = added_idx
      total_adds += 1
      continue

    mapped: dict[int, int] = {target.consumer_output_idx: delta_idx}
    for op_idx in target.view_op_indexes:
      skip_replacement_by_op[op_idx].add(target.consumer_output_idx)
      original_op = subgraph.operators[op_idx]
      cloned, new_outputs = _clone_view_op_with_delta_outputs(
          schema, subgraph, original_op, mapped, f"{stem}/op_{op_idx}"
      )
      branch_by_insert[op_idx].append(cloned)
      mapped.update(new_outputs)
      total_view_ops += 1

    for landing_idx in target.landing_indexes:
      dequant_op_idx = producers.get(landing_idx)
      if dequant_op_idx is None:
        raise ValueError(f"{target.hf_module}: no producer for landing {landing_idx}")
      dequant_op = subgraph.operators[dequant_op_idx]
      dequant_inputs = [] if dequant_op.inputs is None else [int(idx) for idx in dequant_op.inputs]
      if len(dequant_inputs) != 1 or dequant_inputs[0] not in mapped:
        raise ValueError(
            f"{target.hf_module}: dequant input not mapped at op "
            f"{dequant_op_idx}: {dequant_inputs}"
        )
      skip_replacement_by_op[dequant_op_idx].add(dequant_inputs[0])
      lhs = replacement.get(landing_idx, landing_idx)
      rhs = mapped[dequant_inputs[0]]
      landing_shape = _shape(subgraph.tensors[landing_idx])
      added_idx = len(subgraph.tensors)
      subgraph.tensors.append(
          _make_tensor(
              schema,
              f"{stem}/add_to_dequant_{landing_idx}",
              landing_shape,
              schema.TensorType.FLOAT32,
          )
      )
      branch_by_insert[dequant_op_idx].append(
          _make_add(schema, add_opcode, lhs, rhs, added_idx)
      )
      replacement[landing_idx] = added_idx
      total_adds += 1

  new_ops = []
  for op_idx, op in enumerate(subgraph.operators):
    cloned = copy.deepcopy(op)
    _replace_inputs(cloned, replacement, skip_replacement_by_op.get(op_idx))
    new_ops.append(cloned)
    new_ops.extend(branch_by_insert.get(op_idx, ()))
  subgraph.operators = new_ops
  if subgraph.outputs is not None:
    subgraph.outputs = _as_i32([replacement.get(int(idx), int(idx)) for idx in subgraph.outputs])

  builder = flatbuffers.Builder(256 * 1024 * 1024)
  root = model.Pack(builder)
  builder.Finish(root, file_identifier=b"TFL3")
  output_bytes = bytes(builder.Output())
  if alignment > 1:
    output_bytes += b"\0" * (
        ((len(output_bytes) + alignment - 1) // alignment) * alignment
        - len(output_bytes)
    )
  output_path = pathlib.Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_bytes(output_bytes)

  return RewriteReport(
      input=str(input_path),
      output=str(output_path),
      targets=len(targets),
      lora_inputs_added=total_inputs,
      lora_inputs_in_signature=expose_lora_in_signature,
      view_ops_cloned=total_view_ops,
      adds=total_adds,
      bytes=len(output_bytes),
  )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input_tflite", required=True)
  parser.add_argument("--mapping_manifest", required=True)
  parser.add_argument("--output_tflite", required=True)
  parser.add_argument("--schema_path")
  parser.add_argument("--report_json")
  parser.add_argument("--alignment", type=int, default=16)
  parser.add_argument(
      "--hide_lora_signature_inputs",
      action="store_true",
      help="Append LoRA tensors to subgraph inputs without adding signature maps.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  report = rewrite_audio_lora_graph(
      input_path=args.input_tflite,
      manifest_path=args.mapping_manifest,
      output_path=args.output_tflite,
      schema_path=args.schema_path,
      alignment=args.alignment,
      expose_lora_in_signature=not args.hide_lora_signature_inputs,
  )
  report_json = json.dumps(report.to_json_dict(), indent=2, sort_keys=True)
  if args.report_json:
    pathlib.Path(args.report_json).write_text(report_json + "\n", encoding="utf-8")
  print(report_json)


if __name__ == "__main__":
  main()
