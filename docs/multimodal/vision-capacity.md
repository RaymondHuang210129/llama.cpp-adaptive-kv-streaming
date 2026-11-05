# Finding image capacity in a shared arena

`benchmarks/find-max-arena.py` reports processed image resolutions for Qwen and Gemma when `--mmproj` is configured; no sample image files are required. Supply the actual target, matching projector, optional draft/assistant and decoder settings. The finder starts and cleans up its own temporary servers; make GPU memory available and choose an unused port.

## What the report means

The report covers source ratios **1:1, 4:3, 3:4, 16:9, 9:16, 2:1 and 1:2**. Width comes first. Ratio labels describe the source image; native patch alignment can slightly change the processed ratio. Larger uploads may be resized. The measured Qwen projector aligns to 32 pixels and Gemma to 48 pixels. These are processed resolutions, not maximum original-file dimensions.

The finder first searches for the maximum arena that completes its text workload and minimum images for all seven ratios. At the selected arena it reports each ratio's largest validated resolution, including the required `--image-max-tokens` setting. It then validates one shared setting for the generated launch command. If that setting produces smaller resolutions than the individual row settings, the console prints a second table. It does not publish a command if any ratio or the common setting fails.

The matching server emits `vision-memory-v2`: graph, operator, explicit library and alignment bytes sum to its complete compute grant, with projector weights reported separately. The finder rejects older graph-only reports and inconsistent or executed-mismatched totals. Opaque runtime allocations are explicitly excluded. The native planner considers every image budget up to the configured ceiling. Runtime capacity uses a boundary search ordered by processed token count, including the selected boundary and rejected neighbor where one exists. These are executed fits, not a mathematical proof that every larger shape fails: native graph choices and CUDA scratch requirements can vary with shape. Unsupported operations, device faults and incomplete responses stop the search rather than being counted as capacity failures.

One image is tested at a time, with real preprocessing, encoder execution, text decoding and actual drafts when speculation is enabled. The shared setting is also checked with the seven images issued serially in one server after text. Arbitrary other ratios, multiple-image requests, other batch sizes, changed weights and concurrent GPU users are outside these receipts. Validate other image or multi-image workloads separately.

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

For Gemma, substitute the Q3_K_S target, matching external Q8 assistant and Q8 projector, and select `--spec-mode mtp --draft-length 3`. The [README](../../README.md#find-a-usable-arena-size) gives both embedded-Qwen-MTP and external-Gemma-MTP examples. Qwen DFlash2+K4V uses `--spec-mode dflash-k4v` with explicit `--ngram-n`, `--ngram-m` and `--ngram-min-hits`; lookup is restricted to post-last-image text. The finder proves real image-conditioned copy and neural fallback in separate requests, accounts for their additional encoder phases and preserves all text/image response totals. A text-only hit is not image resource qualification. The [measured bounded example](dflash2-vision.md#size-and-launch-mixed-vision) retains its own 4-Ki/256-image-budget scope.

Omitting image-budget flags uses projector metadata. `--image-max-tokens` can set a lower search ceiling; it is not an input resolution. The finder keeps your context, batch sizes, KV types and draft width fixed. Gemma's whole-image microbatch limit can be lower than its native 1120-token image ceiling. Increasing `--ubatch-size` also increases protected SWA storage, so refit rather than changing it after sizing.

## Measured examples

The current qualified examples use one RTX 5070 Ti, UVM off, native context capacity 262144, b/ub256, MTP3 and Q8_0 K / Q4_0 V. Qwen uses IQ4_XS with embedded MTP and its matching F16 projector; Gemma uses Q3_K_S, the matching external Q8 assistant and local Q8 projector. The search starts at the default 2240 MiB guess, uses a 2048-token bounded prefill plus actual decoding/images, and independently detects each parent. No arena input, sample image, arbitrary reserve or running inference service is required.

| Model profile | Selected parent MiB | Rejected neighbor MiB | Common image-max-tokens |
| --- | ---: | ---: | ---: |
| Qwen embedded MTP3 | 2314 | 2315 | 2186 |
| Gemma shared-KV MTP3 | 2462 | 2463 | 256 |

Both selected commands complete one full 261873-token text request and 256 outputs, then all seven image requests with 256 outputs each in the same process. This validates the exact published configuration, rather than a smaller bootstrap image setting. The image requests are separate prompts after text, not images appended to a full 262K prefix. Long lifecycle and fitting native/component numerical controls have their own scopes.

### Qwen processed resolutions

The shared setting is `--image-max-tokens 2186`. Independent row settings and one common serial setting are separate executed controls.

| Source ratio | Independent processed maximum | Row budget | At the common setting |
| --- | ---: | ---: | ---: |
| 1:1 | 2048x2048 | 4096 | 1472x1472 |
| 4:3 | 2336x1760 | 4096 | 1696x1280 |
| 3:4 | 1760x2336 | 4096 | 1280x1696 |
| 16:9 | 2720x1536 | 4096 | 1984x1120 |
| 9:16 | 1536x2720 | 4096 | 1120x1984 |
| 2:1 | 2880x1440 | 4096 | 2112x1056 |
| 1:2 | 1440x2880 | 4096 | 1056x2112 |

### Gemma processed resolutions

The shared setting is `--image-max-tokens 256`. Independent row settings and one common serial setting are separate executed controls.

| Source ratio | Independent processed maximum | Row budget | At the common setting |
| --- | ---: | ---: | ---: |
| 1:1 | 768x768 | 256 | 768x768 |
| 4:3 | 864x624 | 256 | 864x624 |
| 3:4 | 624x864 | 256 | 624x864 |
| 16:9 | 1008x576 | 256 | 1008x576 |
| 9:16 | 576x1008 | 256 | 576x1008 |
| 2:1 | 1056x528 | 256 | 1056x528 |
| 1:2 | 528x1056 | 256 | 528x1056 |

Budgets are ceilings, so processed token counts may round down. Qwen's common square is 2116 tokens. Gemma's current ceiling is limited by the tested 256-row microbatch; its native 1120-budget profile requires a wider microbatch, larger protected SWA and a separate fit. These tables describe processed dimensions for the seven generated source ratios, not visual quality, maximum upload-file dimensions or arbitrary image batches.

The fitter records each independent boundary, the common sequence, the actual rejected +1 MiB arena neighbor and the final full-workload receipt. CUDA code/module initialization, graph executables and ordinary decoder allocations can still limit a common sequence after the encoder has returned its loans. The current Qwen 4096 setting succeeds independently but its serial sequence fails decoder capacity; do not substitute independent maxima into the shared launch command.

Earlier fixed 2048 MiB controls complete all seven ratios at their configured 4096/256 Qwen/Gemma ceilings. Those are separate equal-parent capacity controls from before the writer metadata fix, not fresh maximum-arena measurements or a full-history native-equivalence claim. The historical 48/143 figures are explained below.

### Gemma 1120-budget profile

A separate follow-up uses the same Q3_K_S target, external Q8 assistant and Q8 projector, MTP3, Q8_0/Q4_0 and native context capacity 262144, with batch/microbatch raised to 1280. The finder selects **2444 MiB**, rejects **2445 MiB**, and validates the common native image budget **1120** across all seven ratios. Both the printed rows and completed encoder phases match the expected native geometry.

This run uses bounded 2560-token text probes and 256 generated tokens per request, including the seven images issued serially in one process. Its `full_workload_validated` field is **false**. It is a larger-image capacity check, separate from the fully validated b/ub256 examples above; it does not certify a full 261873-token prefill with the wider batch.

| Source ratio | Processed resolution at budget 1120 | Actual image tokens |
| --- | ---: | ---: |
| 1:1 | 1584x1584 | 1089 |
| 4:3 | 1824x1344 | 1064 |
| 3:4 | 1344x1824 | 1064 |
| 16:9 | 2112x1200 | 1100 |
| 9:16 | 1200x2112 | 1100 |
| 2:1 | 2256x1104 | 1081 |
| 1:2 | 1104x2256 | 1081 |

The token setting is a ceiling; native geometry rounds down to whole patches. Increasing the microbatch also increases protected SWA, so the earlier 2462 MiB b/ub256 result cannot be reused unchanged: that parent fails the wider profile, while the refitted 2444 MiB parent completes it.

To fit this wider configuration with automatic ratio probes:

```sh
python3 benchmarks/find-max-arena.py \
  --server build-v2/bin/llama-server --model /path/to/gemma-4-31B-it-Q3_K_S.gguf \
  --model-draft /path/to/mtp-gemma-4-31B-it-Q8_0.gguf \
  --mmproj /path/to/mmproj-gemma4-31B-Q8_0-local.gguf \
  --context-size 262144 --batch-size 1280 --ubatch-size 1280 \
  --cache-type-k q8_0 --cache-type-v q4_0 --spec-mode mtp --draft-length 3 \
  --decode-tokens 256 --probe-mode preflight --no-webui --port 1249 \
  --output benchmarks/results/fit-gemma-native-image-budget
```

`--initial-arena-mib` remains an optional search guess; the follow-up started from the earlier 2462 MiB result. Add `--validate-selected` for a full requested-text validation after fitting; that additional check was not part of this ratio sweep. All arena numbers are specific to the measured card, weights and configuration.

## What consumes the vision loan

The active vision phase may reclaim text graph workspace and reusable device KV storage. It preserves registered target recurrent state, bounded draft state and Gemma SWA. Its temporary loans hold projector weights and complete controlled encoder compute: graph tensors, backend operator conversions/outputs/pointer arrays, native-size explicit cuBLAS workspace and placement padding. Main/assistant/DFlash weights remain outside the parent. Host KV, decoded pixels and completed embeddings are host allocations.

The compute allowance is bounded by `parent - protected persistent state - projector weights - placement padding`; an image borrows its actual measured requirement, not a fixed percentage. For the 2048 MiB Gemma b/ub256 reference, 406.25 MiB SWA and 770.41 MiB projector weights leave about 871.34 MiB for compute and placement. A Qwen square image with 4096 tokens uses 884.62 MiB weights plus 682.75 MiB complete compute: 480.25 MiB graph, 170.50 MiB operator scratch, 32 MiB library and 128 padding bytes. These are different workloads/weight profiles, not a model-performance comparison.

The earlier 48-token Qwen DFlash3 and 143-token Gemma results predate this migration. Their larger image requests could allocate graph/weights inside the parent but then run out of free VRAM for backend-private encoder scratch outside it. Those historical budgets are not fixed arena partitions, MiB values, intrinsic image ceilings or visual-quality scores. The matched earlier embedded-MTP Qwen control used common budget 47 rather than the separate DFlash budget 48 profile.

A maximum parent still need not allow every larger image sequence. CUDA code/module initialization, graph executables, opaque library state and ordinary decoder allocations remain outside the controlled encoder contract. The earlier complete-workspace Qwen capacity run reached a decoder `cudaFuncSetAttribute` OOM after six encoder phases had returned their loans. Preserve such failures as physical-capacity evidence, not an encoder quota pass or a guarantee for arbitrary future requests. To require a larger minimum generated image profile, raise `--image-min-tokens` with compatible batch settings and refit. No arbitrary safety reserve is added.

## Explicit images and saved evidence

Adding `--image /path/to/photo.png` retains the existing complete-image workload mode and preprocessing defaults (64/4096 unless overridden). Repeat it for multiple images; a `--prompt-file` needs one `<__media__>` marker per image. The automatic ratio table does not replace or change that supplied workload. See [Qwen DFlash2 image requests](dflash2-vision.md#launch-and-size-the-actual-image-workload) and [Gemma image fitting](gemma4-vision.md#find-a-fit-for-your-images).

Without an image, `--probe-mode preflight` uses bounded text history plus actual decoding and vision execution. `--validate-selected` additionally runs the complete requested text workload once at the selected arena and fails without publishing a fit if that request fails. The default `--probe-mode full` runs full text history during each arena probe; resolution boundary checks still use bounded text followed by the image. A configured context capacity alone is not full-history validation.

The output directory contains `result.json`, shell launch commands, `probes.jsonl`, native `arena-*-vision-plan.json` reports, logs and `vision-capacity/` image/response records. Receipts include artifact hashes. Keep them when comparing configurations. Existing [historical native-context finder examples](../../benchmarks/data/native-context-finder-examples.json) used configured projectors but tested text only; their older arena numbers do not certify the new image-aware profile.

For developer inspection, the matching server's `--vision-memory-report` loads the real shared-arena configuration, measures shape-only native vision graphs, prints JSON and exits. This opt-in planning mode does not encode an image and is not itself a runtime-fit guarantee. Normal server startup does not run the budget scan. No new attention kernel, resize policy or device allocator is involved.
