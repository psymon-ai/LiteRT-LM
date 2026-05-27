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

"""Tests for audio LoRA graph rewriter policy helpers."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import flatbuffers

from litert_lm_builder import audio_lora_graph_checker
from litert_lm_builder import audio_lora_graph_rewriter


def _manifest_entry(layer: int = 0, module_key: str = "q_proj"):
  return {
      "hf_module": f"audio_tower.layers.{layer}.self_attn.{module_key}.linear",
      "stage": "audio_after_dequant_graft",
      "status": "ready",
      "layer": layer,
      "module_key": module_key,
      "source_tensor": {"index": 10},
      "consumer_output_tensor": {"index": 20},
      "consumer_op": {"index": 30},
      "path": {
          "landing_tensor": {"index": 40},
          "view_ops": [
              {"index": 31, "outputs": [21]},
              {"index": 32, "outputs": [22, 23]},
          ],
      },
      "lora": {
          "a_prime_shape": [4, 2],
          "b_prime_shape": [2, 8],
          "rank": 2,
      },
  }


class AudioLoraGraphRewriterTest(unittest.TestCase):

  def test_audio_lora_input_names(self):
    self.assertEqual(
        audio_lora_graph_rewriter.audio_lora_input_names("q_proj", 3),
        ("lora_audio_attn_q_a_weight_3", "lora_audio_attn_q_b_weight_3"),
    )
    self.assertEqual(
        audio_lora_graph_rewriter.audio_lora_input_names("lconv_start", 11),
        (
            "lora_audio_lconv_start_a_weight_11",
            "lora_audio_lconv_start_b_weight_11",
        ),
    )
    with self.assertRaisesRegex(ValueError, "unsupported"):
      audio_lora_graph_rewriter.audio_lora_input_names("unknown", 0)

  def test_parse_manifest_targets(self):
    target = audio_lora_graph_rewriter.parse_manifest_targets(
        {"entries": [_manifest_entry()]}
    )[0]

    self.assertEqual(target.layer, 0)
    self.assertEqual(target.module_key, "q_proj")
    self.assertEqual(target.source_idx, 10)
    self.assertEqual(target.consumer_output_idx, 20)
    self.assertEqual(target.consumer_op_idx, 30)
    self.assertEqual(target.landing_indexes, (40,))
    self.assertEqual(target.view_op_indexes, (31, 32))
    self.assertEqual(target.view_output_indexes, ((21,), (22, 23)))

  def test_parse_manifest_rejects_non_ready_audio_target(self):
    entry = _manifest_entry()
    entry["status"] = "blocked"

    with self.assertRaisesRegex(ValueError, "non-ready"):
      audio_lora_graph_rewriter.parse_manifest_targets({"entries": [entry]})

  def test_lora_shapes_by_module(self):
    shapes = audio_lora_graph_rewriter._lora_shapes_by_module(
        {"entries": [_manifest_entry()]}
    )

    self.assertEqual(
        shapes["audio_tower.layers.0.self_attn.q_proj.linear"],
        ((4, 2), (2, 8), 2),
    )

  def test_lora_shapes_require_rank_and_shapes(self):
    entry = _manifest_entry()
    entry["lora"] = {}

    with self.assertRaisesRegex(ValueError, "missing LoRA shape"):
      audio_lora_graph_rewriter._lora_shapes_by_module({"entries": [entry]})

  def test_rewrite_tiny_float_graph_adds_consumed_lora_inputs(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      input_path = temp_path / "input.tflite"
      manifest_path = temp_path / "manifest.json"
      output_path = temp_path / "output.tflite"

      _write_tiny_float_projection_graph(input_path)
      entry = _manifest_entry()
      entry["source_tensor"]["index"] = 0
      entry["consumer_output_tensor"]["index"] = 2
      entry["consumer_op"]["index"] = 0
      entry["path"]["landing_tensor"]["index"] = 2
      entry["path"]["view_ops"] = []
      manifest_path.write_text(
          json.dumps({"entries": [entry]}),
          encoding="utf-8",
      )

      report = audio_lora_graph_rewriter.rewrite_audio_lora_graph(
          input_path=input_path,
          manifest_path=manifest_path,
          output_path=output_path,
          alignment=1,
      )

      self.assertEqual(report.targets, 1)
      self.assertEqual(report.lora_inputs_added, 2)
      self.assertEqual(report.view_ops_cloned, 0)
      self.assertEqual(report.adds, 1)

      graph_report = audio_lora_graph_checker.inspect_audio_lora_target_graph(
          output_path
      )
      self.assertTrue(graph_report.passed)
      self.assertEqual(
          graph_report.target_inputs,
          (
              "lora_audio_attn_q_a_weight_0",
              "lora_audio_attn_q_b_weight_0",
          ),
      )


def _write_tiny_float_projection_graph(path: pathlib.Path) -> None:
  schema = audio_lora_graph_rewriter.import_tflite_object_schema()

  model = schema.ModelT()
  model.version = 3
  model.description = b"tiny audio LoRA rewrite test"
  model.buffers = [schema.BufferT(), schema.BufferT()]

  opcode = schema.OperatorCodeT()
  opcode.builtinCode = schema.BuiltinOperator.BATCH_MATMUL
  opcode.deprecatedBuiltinCode = schema.BuiltinOperator.BATCH_MATMUL
  opcode.version = 1
  model.operatorCodes = [opcode]

  subgraph = schema.SubGraphT()
  subgraph.name = b"main"
  subgraph.tensors = [
      _make_tensor(schema, "source", (1, 4), schema.TensorType.FLOAT32),
      _make_tensor(schema, "base_weight", (4, 8), schema.TensorType.FLOAT32),
      _make_tensor(schema, "projection_out", (1, 8), schema.TensorType.FLOAT32),
  ]
  subgraph.inputs = [0]
  subgraph.outputs = [2]

  op = schema.OperatorT()
  op.opcodeIndex = 0
  op.inputs = [0, 1]
  op.outputs = [2]
  op.builtinOptionsType = schema.BuiltinOptions.BatchMatMulOptions
  op.builtinOptions = schema.BatchMatMulOptionsT()
  subgraph.operators = [op]
  model.subgraphs = [subgraph]

  signature = schema.SignatureDefT()
  signature.signatureKey = b"serving_default"
  signature.inputs = [_make_tensor_map(schema, "source", 0)]
  signature.outputs = [_make_tensor_map(schema, "output", 2)]
  model.signatureDefs = [signature]

  builder = flatbuffers.Builder(4096)
  root = model.Pack(builder)
  builder.Finish(root, file_identifier=b"TFL3")
  path.write_bytes(bytes(builder.Output()))


def _make_tensor(schema, name: str, shape: tuple[int, ...], tensor_type: int):
  tensor = schema.TensorT()
  tensor.name = name
  tensor.shape = list(shape)
  tensor.type = tensor_type
  tensor.buffer = 0
  return tensor


def _make_tensor_map(schema, name: str, tensor_index: int):
  tensor_map = schema.TensorMapT()
  tensor_map.name = name.encode("utf-8")
  tensor_map.tensorIndex = tensor_index
  return tensor_map


if __name__ == "__main__":
  unittest.main()
