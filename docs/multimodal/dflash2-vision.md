# DFlash2 image support with the shared phase arena

DFlash2 can draft for still-image/text requests using the matching Qwen target and projector. It uses the existing image-request API and the same device parent as text prefill, decode and verification. Optional CPU K4V can copy continuations from contiguous text after the latest complete image, falling through to DFlash on a miss. Gemma uses its existing target-only/shared-KV MTP vision path and cannot attach this Qwen DFlash sidecar.

## Supported and measured scope

The reference configuration is one serial request on one CUDA RTX 5070 Ti (SM120), Flash Attention, GPU KV offload, fit disabled and context shifting disabled. Target weights are the existing Qwen3.8-27B IQ4_XS artifact; the projector remains its matching F16 artifact. The smaller draft checkpoint below makes more room for the image reference. Architecture, tokenizer, target-layer mapping and positional metadata are validated; a filename or quantization label alone does not establish compatibility.

| Artifact | Measured identity |
| --- | --- |
| Qwen target | IQ4_XS; SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| F16 projector | SHA256 `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` |
| Smaller matching sidecar | HermiHg `Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf`, revision `2b4527438b07460dcf27713107aedf0c6853e850`, 561241824 file bytes; SHA256 `0c1a01efa952a752b8bf139e8beb934d07278422eefc2e0ba968a75ee8bed528` |

The [sidecar publisher](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF) provides this mixed weight encoding. Its validated DFlash metadata retains block size 8, the 2048-position window, five target input-layer features and RoPE sections `[64,0,0,0]`. Weight encoding is independent of the decoder KV format. The original Q4 sidecar remains a text regression control; changing sidecar precision can change acceptance and memory fit.

Neural proposal maxima are F=1-7; optional copied maxima K=1-7 are independent. Target verification and rollback cover max(F,K), through TG1-TG8; neural prediction still uses F+1 input rows. Q8_0/Q4_0, Q8_0/Q8_0 and Q5_1/Q5_1 have exact decoder controls; F3 full native/streamed API controls also pass for all three pairs at the fitting short profile; that is not a long-context sweep of every pair. Adaptive execution inherits target cache types. Do not pass draft-cache overrides, even identical ones. LoRA, parallel slots, multi-GPU, audio/video, simultaneous MTP+K4V and K15 remain outside this profile. This work does not qualify DFlash vision on Windows or other physical GPUs.

## One device parent and separate lifetimes

| Resource | During image encoding |
| --- | --- |
| Target weights, measured 12726.35 MiB | Remain outside the parent on GPU |
| Q2 mixed sidecar weights, measured 524.80 MiB | Remain outside the parent on GPU; no drafting runs concurrently |
| Full target KV history | Authoritative host history survives; resident/ring bindings retire and rebuild |
| Bounded DFlash KV and rollback publication staging | Protected persistent regions inside the same parent |
| Projector weights, 927588992 allocated bytes | Temporary vision grant inside the parent |
| Encoder graph, operator scratch and explicit cuBLAS workspace | One complete measured compute loan, including alignment; no decoder history |
| Completed image embeddings | Owned host data survives release of the vision grants |

Managed CUDA encoder temporaries and explicit native-size cuBLAS workspace now use the same parent; successful bound execution neither grows a separate native pool nor retains an old native owned library reservation. The arena bounds these participating allocations, not total VRAM, decoded host images, saved states, driver/library state, ordinary decoder allocations or every future request. Image dimensions, token budget, batching, context and proposal width need their own fit. The existing fixed 2K-row prefill preparation stage and pipeline remain unchanged.

The attached draft registers its actual bounded-storage lease with the parent. Suspension admits that validated writer rather than pretending it is a shared-cache reader. Existing phase fences complete outstanding work before temporary grants can overwrite scratch. Persistent target/draft state stays protected; failed reload, queued encoder failure and held grants cannot expose stale graphs or half-returned storage.

The private image-aware initialization overload in [llama-ext.h](../../src/llama-ext.h) requests these startup measurements without changing the existing two/three-argument contracts. The child calls `register_bounded_write_consumer` with its actual persistent lease; `can_suspend_for_vision` checks the parent/resource relationship. `mtmd_batch_encode_arena` owns the measure/suspend/lend/encode/release/resume transaction, and `llama_serial_features_match` validates the completed forward. This reuses the existing leases, graph revisions and completion fences rather than adding an allocator or second device parent. Gemma's existing shared-cache reader registration remains a distinct role.

Before the child attaches, the target measures embedding-input graph shapes with all required feature outputs. The completed image source owns exact F32 inputs and row/position metadata until the next forward or state restoration. The callback rejects another image at the same positions, changed spatial coordinates and stale sources. Text-only execution avoids those image-vector copies.

DFlash consumes selected target-layer features, not raw projector embeddings. Native M-RoPE policy skips insertion of multirow image groups pinned to one temporal position, avoiding bounded-window exhaustion. The target still processes the image, and subsequent text features are image-conditioned. A single-row image tail follows native injection. Target verification remains authoritative, and positive tests require real draft proposals rather than silent target-only execution.

## Launch and size the actual image workload

Build the [vision-enabled server](../../README.md#build), provide matching artifacts and make GPU memory available. No existing service or container is required. From the repository root, run the finder with an unused port and a fresh output directory:

```bash
python3 benchmarks/find-max-arena.py \
  --server build-v2/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --model-draft /path/to/Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-27B-F16.gguf \
  --context-size 8192 --batch-size 256 --ubatch-size 256 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-mode dflash --draft-length 3 \
  --image-min-tokens 64 --image-max-tokens 4096 \
  --decode-tokens 256 --probe-mode preflight --no-webui --port 1249 \
  --output benchmarks/results/my-dflash-image-fit
```

The finder starts and cleans up its own servers. With `--mmproj`, it measures complete controlled workspace, runs the [automatic landscape/portrait capacity checks](vision-capacity.md), and prints independent resolutions plus a common image setting. No sample images or existing inference service are required. `--initial-arena-mib` is an optional search guess. The development 1024 MiB image probe is a fixed-budget successful request, not a measured maximum for arbitrary images.

Use the generated `command.txt`, or read both selected arena and common image setting before launching a fresh server:

```bash
DF_ARENA=$(python3 -c 'import json; print(json.load(open("benchmarks/results/my-dflash-image-fit/result.json"))["arena_mib"])')
DF_IMAGE_MAX=$(python3 -c 'import json; print(json.load(open("benchmarks/results/my-dflash-image-fit/result.json"))["vision_capacity"]["common_image_max_tokens"])')
build-v2/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --model-draft /path/to/Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-27B-F16.gguf \
  --shared-device-memory-mib "$DF_ARENA" --ctx-size 8192 \
  --parallel 1 --batch-size 256 --ubatch-size 256 --fit off \
  --n-gpu-layers 999 --spec-draft-ngl 999 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-type draft-dflash --spec-draft-n-max 3 \
  --spec-draft-n-min 0 --spec-draft-p-min 0 \
  --image-min-tokens 64 --image-max-tokens "$DF_IMAGE_MAX" \
  --offline --no-context-shift --host 127.0.0.1 --port 8080
```

Open that server's Web UI and attach an image, or use the existing [HTTP image request](gemma4-vision.md#running-an-image-request-locally). There is no DFlash-specific image wire format. Refit when changing images, cache types, context, batch sizes, proposal limits or weights; a previous text-only budget is not an image fit guarantee.

## Pure-DFlash foundation qualification

The following records describe the separately qualified DV0-DV17 foundation. The mixed-image extension is documented below; the older metrics are not relabelled as new measurements.

Fresh fitting full native F1-F4 servers and streamed servers at context 2048, b/ub256 and image budget 256 compare corresponding image scenarios with exact tokens and draft/prompt/cache counts. F5-F7 at the recorded complete-workspace control profile reach preserved native physical capacity failures; unchanged smaller Q2 weights are tried first. Native encoder/native-decoder component references and the archived streamed end-to-end reference are explicitly distinguished from full native-server equivalence. F7 full native startup cannot allocate the 927588992-byte projector alongside its decoder state, even with the smaller draft. That capacity failure is preserved. F7 instead has separately qualified native encoder export and native-decoder import of identical embeddings, positions and weights, plus end-to-end streamed API/lifecycle tests. This is native-arithmetic component coverage, not full native-server equivalence.

The internal framed oracle checks complete consumed target features, logits/hidden rows, selector outputs, target/draft KV and recurrent state through every accepted prefix, correction and TG1-TG8. F1/F3/F7 compare 504/907/1749 exact records; F7 covers 2834713708 bytes. The additional KV pairs retain exact native comparisons. The complete public image path also passes fresh strict device memcheck with API-all reporting and real DFlash predictions, plus actual host ASAN/UBSAN image checks. Normal warmup has a separate fitting native/streamed API control.

Earlier F1/F3 lifecycle tests each compare 23 native and 23 streamed scenarios. Their corresponding 321935388-byte state files have identical SHA256 digests. Tests cover changed/adjacent/interleaved/later images, portrait/narrow and first/last/only-image prompts, malformed media, image/context limits, cancellation, truncated state, erase/restore and process restart. Rewinds into an image or unavailable old history remain rejected. Each cached/restored scenario uses its corresponding native scenario; fresh-versus-cached native identity is not assumed.

The long F7 profile uses capacity 65536, a 48000-token text prefix, 512 outputs, a 1024 MiB device parent and a 4096 MiB host prompt cache. All 23 streamed scenarios pass; three initial/state/restart logs prove 188-191 active nonresident pages with 66 ring slots. A complete 1799655104-byte state with 48584 tokens survives restart. These are long lifecycle/self-consistency checks, not long native equivalence or a full native-context image-prefill qualification. The earlier 2048 MiB host-cache attempt correctly failed its cache-restoration proof because intervening entries evicted the large prompt; that failure remains recorded.

Attached-writer reload cancellation and a queued encoder exception preserve target/child state byte-for-byte, publish no failed output, retry exact encoder values and resume native-matching prediction. Fourteen host ASAN/UBSAN suites pass with leak detection. GPU ASAN uses NVIDIA's documented [CUDA compatibility setting](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/reference/troubleshooting-analysis-tools.html) `protect_shadow_gap=0`, retaining leak detection and halt-on-error. Strict device checks use default PDL, API-all reporting, error exit 99 and no filters; eager execution is limited to checker runs. No instrumentation OOM or initialization failure is counted as a pass.

## Ordinary performance

The first measurements below belong to the initial DFlash image feature, before complete CUDA scratch joined the parent. Current migration regression checks follow them; these cohorts must not be combined into a speedup figure.

The ordinary image workload uses a 256x256 pattern, Q8_0/Q4_0 KV, a 4096-position context, 256/256 batching, a fixed 1024 MiB parent and 256 generated tokens. Each process has one warmup and three uncached measured requests. Median results for the implementation are:

| Mode | Decode tok/s | Total request seconds | Accepted/generated proposals |
| --- | ---: | ---: | ---: |
| Target-only | 48.33 | 5.64 | 0/0 |
| Embedded MTP3 | 72.77 | 3.87 | 152/308 |
| Q2 DFlash3 | 71.95 | 3.91 | 141/342 |
| Q2 DFlash7 | 63.62 | 4.39 | 153/698 |

DFlash7 proposes more tokens but accepts a smaller fraction on this fixture; it does not beat MTP3 here. These are different drafting algorithms and weight encodings, not a code-only speedup comparison or a visual-quality certificate. All three DFlash samples have identical output IDs and work counts within their profile. Projector reload is about 150-152 ms and the warmed encoder scope about 6.4-6.6 ms; those are existing host elapsed-time diagnostics, not GPU utilization/bandwidth counters, and must not be added again to total request latency.

The pre-feature reference is a coherent build of merged revision `7425d14c3`; the measured implementation is `e8007865f`. Same-mode code comparisons require exact generated IDs, prompt/draft/source counts and physical decode layouts. Three target-only image process pairs, including alternating order and three requests per process, give median changes of -0.35% prefill, -0.08% decode and +0.18% total latency. Individual total changes stay within 0.41%. An initial small-prefill difference did not reproduce consistently. The MTP image comparison changes +0.01% decode and +0.04% total latency across its three paired request samples. No material existing-image regression is observed; these measurements do not establish zero overhead for every workload.

Separate text regression checks use natural NVIDIA Open-SWE request prefixes and a fixed 1 GiB parent, ordinary graphs, UVM off and 256 outputs. The 44 requests form 22 exact same-mode pairs, with 11264 generated IDs. Gemma uses its own chat template/tokenizer rather than reusing Qwen token IDs.

| Existing mode | Actual prefix tokens | Process pairs | Median paired prefill change | Median paired decode change |
| --- | ---: | ---: | ---: | ---: |
| Qwen target-only | 8186 | 3 | -0.04% | -0.02% |
| Qwen MTP3 | 8186 | 3 | +0.08% | +0.29% |
| Qwen original Q4 DFlash7 | 8186 | 3 | -0.09% | +0.04% |
| Qwen original Q4 DFlash7+K4V | 8186 | 3 | -0.05% | +0.09% |
| Gemma target-only | 8191 | 3 | -0.04% | +0.17% |
| Gemma shared-KV MTP3 | 8191 | 3 | +0.06% | +1.24% |

Single 64524-token Qwen controls for all four modes vary from -0.54% to +0.31% decode and within 0.05% prefill. Those long controls are single paired observations, not repeated statistical results. Positive variation is not claimed as a new optimization. Text, image and independently maximum-fit sweeps remain separate cohorts.

## Complete-workspace regression controls

The shared extension audit passes 128 runtime gates, sixteen host ASAN/UBSAN suites and thirteen zero-error strict GPU gates, with actual graph-enabled ownership and MTP2 replay where required. Fresh framed decoder controls cover all F1-F7; full native/API versus component-reference scopes remain separate. A cached writer destination lifetime failure was reproduced with one valid behavioral red, fixed by retaining its existing buffer metadata through executor retirement, and requalified without a new allocation, logical lease, fence or arithmetic change. The long F7 state/cancel/restart profile passes again with actual ring traffic.

Three ordinary Open-SWE process pairs per existing text mode retain exact IDs, work/source counts and decode layouts. Median paired prefill/decode changes are -0.059%/-0.152% target-only, -0.043%/-0.001% MTP3, -0.054%/+0.215% original Q4 DFlash7 and -0.057%/-0.615% Q4 DFlash7+K4V. No material existing text regression is observed on these scoped checks; this is not statistical proof of zero overhead.

Current image comparisons use three warmed uncached samples at each of 256x256, 1024x1024 and 2048x2048 source sizes, with 128 outputs and exact same-mode responses. Target-only/MTP3 retain 2048 MiB parents. The archived Q2 DFlash3 baseline at 2048 MiB runs out of memory in decoder CUDA kernel preparation after the medium encoder phase returns. The complete workspace quote for the current 4096-token square needs 1682285312 bytes; 1605 MiB admits it and 1604 rejects. The fitting DFlash comparison uses that same 1605 MiB parent on both implementations, unchanged images/models/outputs/ordinary graphs. It is a separate equal-parent cohort, not a maximum-arena or old-failure pass.

Image request-latency median changes span -0.292%..+0.423% for Qwen target/MTP and -0.138%..+0.240% for Q2 DFlash3. Small roughly 6.4 ms encodes add about 0.3-0.4 ms of admission/ownership overhead; largest roughly 2648 ms encodes remain close to baseline. Sampled 50 ms device peaks decrease by about 202-204 MiB at matched parents. These samples may miss shorter transients and are not a hardware utilization measurement. Current processed-capacity examples are in the [common capacity report](vision-capacity.md).

## Reproduction and release boundary

The [implementation roadmap](../../DFLASH2_VISION_PHASE_ROADMAP.md) records stage completion and scoped failures. Existing reusable tests are `tests/test-mtmd-projector-storage.cpp`, the image oracle in `tests/test-kv-stream-context.cpp`, `tests/test-server-vision.cpp`, and [the public vision harness](../../tools/server/tests/test_adaptive_vision.py). Local raw results and binary/source manifests remain outside the committed report.

The optional K4V image extension below has separate media-boundary, numerical, lifecycle and performance qualification. Existing Qwen/Gemma target-only/MTP vision remains supported; no attention arithmetic, second arena or projector-weight quantization changes are introduced.

### Finder admission tests

The existing finder negative case now retains unsupported K8 rather than rejecting the qualified mixed-image pairing. No new server-AB harness or raw data is added. The current host suites pass 53 vision, 46 Gemma and 182 benchmark tests, with one expected platform skip.

## Optional K4V image-conditioned text

K4V receives a checked suffix of the existing full text snapshot. It never joins keys or continuations across an image and excludes all pre-image text. DFlash keeps its full existing snapshot and the actual model frontier. Image rows, filtered-text offsets and M-RoPE positions are separate quantities: the real Qwen reference contains 64 image rows across eight model positions.

The map and acceptance feedback rebuild on every real generation, including equal-length image replacement, cache reuse, restore and restart. Restored image placeholders use validated whole-chunk metadata. Successful verification refreshes the next snapshot from committed, trimmed prompt state; tentative tokens do not become lookup history. K4V is disposable host state and adds no serialized model history, GPU map or new arena. A short/empty suffix legitimately uses DFlash.

Copied proposals still require target verification and DFlash feature injection. Source feedback charges only original accepted proposals. `common_speculative_accept(spec, seq, boundary_row, n_accepted)` keeps the verified neural hidden-state row separate from that logical count when a correction is replayed; the three-argument wrapper retains its equal-count behavior. MTP/EAGLE keep their original neural boundary. The private lookup epoch is not a GPU cache generation or lease identity.

### Size and launch mixed vision

Use the same image API and matching artifacts as pure DFlash. Fit the exact mixed configuration; an MTP or pure-DFlash arena is not a mixed fit. This executed bounded example uses the default optional 2240 MiB starting guess:

```bash
python3 benchmarks/find-max-arena.py \
  --server build-v2/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --model-draft /path/to/Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-27B-F16.gguf \
  --spec-mode dflash-k4v --draft-length 3 --ngram-n 12 --ngram-m 7 --ngram-min-hits 2 \
  --context-size 4096 --batch-size 256 --ubatch-size 256 \
  --cache-type-k q8_0 --cache-type-v q4_0 --decode-tokens 8 \
  --probe-mode preflight --image-max-tokens 256 --no-webui \
  --output benchmarks/results/my-mixed-image-fit
```

The recorded local sizing cohort selects 2110 MiB and rejects 2111 MiB for both F3/K7 and F7/K3. It executes bounded 2048-token text history, eight ordinary outputs, real image-conditioned copy/miss proofs and all seven ratios. The common image budget is 256 under that explicit ceiling. Processed resolutions are 512x512 (1:1), 576x416 (4:3), 416x576 (3:4), 672x384 (16:9), 384x672 (9:16), 704x352 (2:1) and 352x704 (1:2). Source aspect and processed aspect can differ through native alignment. These are bounded, workload-specific measurements, not full-context validation or maximum original-file dimensions.

Launch the generated `command.txt`. It includes the selected arena, projector, independent widths and common image setting. No existing server is required. The finder uses actual images for separate post-image-copy and genuine-miss resource probes; every text prelude, ordinary request, extra encoder phase and source total is accounted once. A text-only copy hit cannot qualify image resources. Insufficient proof history is reported separately from physical OOM. N/K/hit policy is fixed at server configuration; requests cannot replace sources or exceed admitted widths.

### Mixed safety and recovery

F3/K7, F7/K3 and F7/K7 match complete native-decoder component transcripts: image inputs/positions, all target logits/hidden/layer features, selector values, visible target/draft KV, recurrent state and source counters, including every accepted prefix and TG1-TG8. Full native F7 projector-capacity failures remain preserved; component values are not described as full native-server equivalence.

F3/K3 compares 30 corresponding full-native and streamed completion scenarios per mode, with exact output/work/cache counts and identical complete 321082524-byte state files. F3/K7 and an F7/K3 8-Ki window profile add separately labelled streamed recovery controls; the latter preserves a complete 571911928-byte state. New copied and neural cancellation controls wait for real output rather than progress placeholders. Unavailable old prefixes and mid-image rewinds remain rejected.

The final runtime passes 34 accepted GPU regression/numerical/memory gates and nine strict zero-error jobs, plus 13 native and 16 host-ASAN suites. Strict checks retain default PDL, API-all reporting, exit 99 and no filters; eager mode is checker-only. Capture-required ownership tests use actual graphs. GPU ASAN retains its already-qualified `protect_shadow_gap=0` CUDA compatibility setting, leak detection and UBSAN. A baseline/current no-model diagnostic distinguishes the default shadow-gap initialization failure from model/arena OOM; the harness rejects CPU fallback as a GPU qualification. Raw logs, source/model/library hashes and preserved failures remain local.

The corrected replay contract preserves all 1497376 captured bytes of MTP embedding inputs against the established baseline; an unshipped logical-count-only prototype changes those bytes and is rejected. This is a functional input comparison, not instrumented performance evidence.

### Mixed-image performance scope

Ordinary requests use the IQ4_XS/Q2/F16 artifacts, fixed 1-GiB parent, context capacity 8192, Q8/Q4, b/ub256, UVM off, fresh images and 128 outputs. Three alternating pairs per case include one warmup per process. Caption and constructed post-image code-copy prompts contain 213 and 1792 actual input rows respectively. All 6144 timed output IDs match within the comparisons; paired physical decode-layout sequences also match.

| Same-mode code control | Caption paired-median decode change | Constructed code paired-median decode change |
| --- | ---: | ---: |
| Existing MTP3 | +0.02% | -0.27% |
| Existing Q2 DFlash7 | +0.07% | -0.20% |

No material existing-mode regression is observed in these scoped samples. Small-prefill timings vary and are not claimed as a new optimization.

N12/M7/H2 makes no eligible copies on these ordinary fixtures and shows no meaningful gain. A separate N12/M7/H1 cohort leaves captioning effectively unchanged. On the deliberately constructed code-copy task, median decode rises from 181.17 to 199.14 tok/s, with a +9.98% paired-median change (all three pairs +9.43% to +10.99%) and roughly 2.7% lower paired request latency. K4V accepts 98 copied tokens; DFlash prediction forwards fall from 17 to 3, while target verification remains 17 passes and total generated/accepted proposals remain 119/110. Output IDs match exactly. Separate medians and medians of paired ratios need not give identical percentages.

This demonstrates useful copying when the supplied post-image text is reusable. It is not a general image-captioning gain, native multimodal agent trace, quality result or long-context speed guarantee. Do not combine it with the older Q4 text/scheduling measurements. One measured code sample spends about 0.102 ms on all K4V lookup/statistics work plus 0.055 ms on map begin; those are whole-call host scopes, not single-hash latency guarantees. The hash index reserves 1 MiB per map, plus dynamic key statistics (88 bytes per entry on this build), history capacity and allocator overhead; the private view descriptor is 16 bytes. All remain in RAM.
