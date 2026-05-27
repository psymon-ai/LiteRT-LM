# Audio LoRA Graph Checker

This document describes the target-graph contract checker for LiteRT-LM audio
LoRA.

The PEFT audio LoRA converter can compile adapter tensors into a LiteRT-LM LoRA
sidecar, but that sidecar is useful only when the target audio graph exposes
matching LoRA inputs and actually consumes those inputs. A graph that lists
LoRA tensors as signature inputs while leaving some of them unused can silently
drop part of the adapter.

## Command

```bash
bazel run //python/litert_lm_builder:audio_lora_graph_checker_cli -- \
  --target_tflite=/path/to/audio_encoder_lora_ready.tflite \
  --expected_audio_layers=12 \
  --report_json=/path/to/audio_lora_graph_report.json
```

For graph bring-up, `--expected_audio_layers` may be omitted. In that mode the
checker validates only the LoRA inputs that are present. For Gemma audio tower
release checks, pass the expected layer count so missing LoRA slots fail before
any sidecar compilation or CER run.

Use `--require_output_proj` when the target ABI includes
`lora_audio_output_proj_{a,b}_weight_0`.

## Strict Gates

The CLI fails by default when:

- the graph exposes no LiteRT-LM audio LoRA inputs,
- an expected audio LoRA input is missing,
- an exposed audio LoRA input is not consumed by any operator.

The `--allow_missing_expected_inputs` and `--allow_unconsumed_target_inputs`
flags are intended only for graph bring-up. Release conversion should keep the
strict defaults.

## Why This Exists

Audio LoRA target graphs can fail in a misleading way: the graph signature can
look LoRA-ready, and a sidecar can be compiled, while some inputs are not wired
into the computation. In that state, downstream task metrics such as CER are
not valid evidence about adapter quality.

The graph checker is the first target-side gate:

```text
LoRA-ready audio TFLite
  -> graph checker: exposed + consumed ABI inputs
  -> PEFT sidecar converter: tensor names, shapes, dtype, bytes
  -> zero-sidecar no-op gate
  -> non-zero sidecar binding gate
  -> task metric
```

This keeps graph ABI failures separate from LoRA quality or quantization
experiments.

