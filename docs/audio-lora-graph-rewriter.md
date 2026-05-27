# Audio LoRA Graph Rewriter

This document describes the target-graph rewrite step for LiteRT-LM audio LoRA.

The PEFT audio LoRA converter creates a sidecar. The graph rewriter creates the
matching audio encoder graph contract: runtime LoRA inputs with official names,
float LoRA branches, and ADD nodes at the float landing points identified by a
mapping manifest.

## Formula

For each manifest entry, the rewriter inserts:

```text
source -> BATCH_MATMUL(lora_A) -> BATCH_MATMUL(lora_B)
       -> optional cloned view chain -> ADD at float landing
```

The base quantized path remains unchanged. LoRA is not baked into low-bit
weights.

## Command

When run through Bazel, the rewriter uses LiteRT-LM's generated TFLite Python
schema. For non-Bazel use, point `--schema_path` or `TFLITE_SCHEMA_OBJECT_PATH`
to a directory generated with `flatc --python --gen-object-api schema.fbs`.

Then run:

```bash
bazel run //python/litert_lm_builder:audio_lora_graph_rewriter_cli -- \
  --input_tflite=/path/to/audio_encoder_hw.tflite \
  --mapping_manifest=/path/to/audio_mapping_manifest.json \
  --output_tflite=/path/to/audio_encoder_lora_ready.tflite \
  --report_json=/path/to/audio_lora_graph_rewrite_report.json
```

The mapping manifest must record, for each audio LoRA target:

- source tensor,
- projection consumer op and output tensor,
- float landing tensor or tensors,
- any view ops between the projection output and landing tensor,
- LoRA A/B runtime shapes.

## Release Gate

After rewriting, validate the target graph:

```bash
bazel run //python/litert_lm_builder:audio_lora_graph_checker_cli -- \
  --target_tflite=/path/to/audio_encoder_lora_ready.tflite \
  --expected_audio_layers=12
```

For a Gemma audio tower with 12 layers and 10 module groups, the expected result
is:

```text
target_inputs: 240
unconsumed_target_inputs: 0
missing_expected_inputs: 0
```

Only after this gate should a PEFT audio LoRA sidecar be compiled for the graph
or evaluated on a downstream metric.

## Scope

This tool rewrites an extracted audio encoder TFLite section. It does not:

- build a full `.litertlm` container,
- bind the sidecar in the runtime,
- claim task quality or CER improvement,
- discover the mapping manifest automatically.

Those steps should remain separate so graph ABI failures do not get confused
with model-quality or runtime-binding issues.
