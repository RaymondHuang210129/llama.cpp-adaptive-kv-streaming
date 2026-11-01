# Finding image capacity in a shared arena

`benchmarks/find-max-arena.py` reports processed image resolutions for Qwen and Gemma when `--mmproj` is configured and `--image` is omitted. Supply the actual target, matching projector, optional draft/assistant and decoder settings. No sample image is required. The finder starts and cleans up its own temporary servers; make GPU memory available and choose an unused port.

## What the report means

The report covers source ratios **1:1, 4:3, 3:4, 16:9, 9:16, 2:1 and 1:2**. Width comes first. Ratio labels describe the source image; native patch alignment can slightly change the processed ratio. Larger uploads may be resized. The measured Qwen projector aligns to 32 pixels and Gemma to 48 pixels. These are processed resolutions, not maximum original-file dimensions.

The finder first searches for the maximum arena that completes its text workload and minimum images for all seven ratios. At the selected arena it reports each ratio's largest validated resolution, including the required `--image-max-tokens` setting. It then validates one shared setting for the generated launch command. If that setting produces smaller resolutions than the individual row settings, the console prints a second table. It does not publish a command if any ratio or the common setting fails.

The native planner considers every image budget up to the configured ceiling. Runtime capacity uses a boundary search ordered by processed token count, including the selected boundary and rejected neighbor where one exists. These are executed fits, not a mathematical proof that every larger shape fails: native graph choices and CUDA scratch requirements can vary with shape. Unsupported operations, device faults and incomplete responses stop the search rather than being counted as capacity failures.

One image is tested at a time, with real preprocessing, encoder execution, text decoding and actual drafts when speculation is enabled. The shared setting is also checked with the seven images issued serially in one server after text. Arbitrary other ratios, multiple-image requests, other batch sizes, changed weights and concurrent GPU users are outside these receipts. Use `--image` for a specific image or multi-image workload.

## Run without a sample image

For example, use Qwen with the smaller matching DFlash2 sidecar:

```sh
python3 benchmarks/find-max-arena.py \
  --server build-v2/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --model-draft /path/to/Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-27B-F16.gguf \
  --context-size 262144 --batch-size 256 --ubatch-size 256 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-mode dflash --draft-length 3 --decode-tokens 8 \
  --probe-mode preflight --no-webui --port 1249 \
  --output benchmarks/results/my-qwen-resolution-fit
```

For Gemma, substitute the Q3_K_S target, matching external Q8 assistant and Q8 projector, and select `--spec-mode mtp --draft-length 3`. The [README](../../README.md#find-a-usable-arena-size) gives both embedded-Qwen-MTP and external-Gemma-MTP examples. DFlash2+K4V image history remains unsupported.

Omitting image-budget flags uses projector metadata. `--image-max-tokens` can set a lower search ceiling; it is not an input resolution. The finder keeps your context, batch sizes, KV types and draft width fixed. Gemma's whole-image microbatch limit can be lower than its native 1120-token image ceiling. Increasing `--ubatch-size` also increases protected SWA storage, so refit rather than changing it after sizing.

## Measured examples

The following runs used one RTX 5070 Ti, UVM off, Q8_0 K / Q4_0 V, 256/256 batching, native context capacity 262144, a 2048-token preflight and eight generated tokens per request. They did **not** prefill 262144 tokens. Qwen used IQ4_XS + Q2_K_S-MIX DFlash3 + F16 projector; Gemma used Q3_K_S + external Q8 MTP3 + local Q8 projector. These are different weight profiles, not a model-performance comparison.

| Source ratio | Qwen processed maximum | Row budget | Gemma processed maximum | Row budget |
| --- | ---: | ---: | ---: | ---: |
| 1:1 | 192x192 | 48 | 528x528 | 143 |
| 4:3 | 256x192 | 60 | 720x528 | 191 |
| 3:4 | 192x256 | 60 | 528x720 | 191 |
| 16:9 | 288x160 | 56 | 720x384 | 143 |
| 9:16 | 160x288 | 56 | 384x720 | 143 |
| 2:1 | 288x128 | 49 | 912x432 | 199 |
| 1:2 | 128x288 | 49 | 432x912 | 199 |

Qwen selected **2066 MiB**, with 2067 MiB rejected; its common setting is `--image-max-tokens 48`. Gemma selected **2410 MiB**, with 2411 MiB rejected; its common setting is `--image-max-tokens 143`. Qwen's common setting achieves its independent maxima. Gemma's shared setting produces the following resolutions; use an individual row setting above when running only that ratio from a fresh server.

| Source ratio | Gemma at shared budget 143 |
| --- | ---: |
| 1:1 | 528x528 |
| 4:3 | 624x480 |
| 3:4 | 480x624 |
| 16:9 | 720x384 |
| 9:16 | 384x720 |
| 2:1 | 768x384 |
| 1:2 | 384x768 |

The independent row settings and the common setting are stored in `result.json`; use the generated `command.txt` or `command.ps1`. Image dimensions and token counts are checked against the native encoder's completed `vision_phase` diagnostic, together with actual weight/workspace bytes and successful return of its arena grants.

A maximum arena need not allow the largest images. Main/draft weights and some CUDA/cuBLAS scratch live outside the arena; taking almost all available VRAM for the arena can leave little room for encoder scratch. The live checks observed this limit after valid text execution. A smaller arena can therefore allow larger images. To require a larger minimum image profile without a sample file, raise `--image-min-tokens` and keep a compatible maximum/batch configuration; the arena search will then execute those larger native shapes. Token budgets still round to native processed dimensions. There is no arbitrary safety reserve and no claim that allocation success bounds all driver allocations.

For contrast, separate fixed-arena checks at context 8192 completed all seven ratios with a 1024 MiB Qwen parent (square 928x928) and a 2048 MiB Gemma parent (square 768x768). Those are fitting fixed-budget controls, not maximum-arena results or full-context text checks.

## Explicit images and saved evidence

Adding `--image /path/to/photo.png` retains the existing complete-image workload mode and preprocessing defaults (64/4096 unless overridden). Repeat it for multiple images; a `--prompt-file` needs one `<__media__>` marker per image. The automatic ratio table does not replace or change that supplied workload. See [Qwen DFlash2 image requests](dflash2-vision.md#launch-and-size-the-actual-image-workload) and [Gemma image fitting](gemma4-vision.md#find-a-fit-for-your-images).

Without an image, `--probe-mode preflight` uses bounded text history plus actual decoding and vision execution. `--validate-selected` additionally runs the complete requested text workload once at the selected arena and fails without publishing a fit if that request fails. The default `--probe-mode full` runs full text history during each arena probe; resolution boundary checks still use bounded text followed by the image. A configured context capacity alone is not full-history validation.

The output directory contains `result.json`, shell launch commands, `probes.jsonl`, native `arena-*-vision-plan.json` reports, logs and `vision-capacity/` image/response records. Receipts include artifact hashes. Keep them when comparing configurations. Existing [historical native-context finder examples](../../benchmarks/data/native-context-finder-examples.json) used configured projectors but tested text only; their older arena numbers do not certify the new image-aware profile.

For developer inspection, the matching server's `--vision-memory-report` loads the real shared-arena configuration, measures shape-only native vision graphs, prints JSON and exits. This opt-in planning mode does not encode an image and is not itself a runtime-fit guarantee. Normal server startup does not run the budget scan. No new attention kernel, resize policy or device allocator is involved.
