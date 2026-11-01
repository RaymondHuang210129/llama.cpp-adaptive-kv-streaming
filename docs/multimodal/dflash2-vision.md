# DFlash2 image support with the shared phase arena

Pure DFlash2 can draft for still-image/text requests using the matching Qwen target and projector. It uses the existing image-request API and the same device parent as text prefill, decode and verification. DFlash2+K4V remains text-only; mixed image history needs a separate boundary-aware lookup policy. Gemma uses its existing target-only/shared-KV MTP vision path and cannot attach this Qwen DFlash sidecar.

## Supported and measured scope

The reference configuration is one serial request on one CUDA RTX 5070 Ti (SM120), Flash Attention, GPU KV offload, fit disabled and context shifting disabled. Target weights are the existing Qwen3.8-27B IQ4_XS artifact; the projector remains its matching F16 artifact. The smaller draft checkpoint below makes more room for the image reference. Architecture, tokenizer, target-layer mapping and positional metadata are validated; a filename or quantization label alone does not establish compatibility.

| Artifact | Measured identity |
| --- | --- |
| Qwen target | IQ4_XS; SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| F16 projector | SHA256 `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` |
| Smaller matching sidecar | HermiHg `Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf`, revision `2b4527438b07460dcf27713107aedf0c6853e850`, 561241824 file bytes; SHA256 `0c1a01efa952a752b8bf139e8beb934d07278422eefc2e0ba968a75ee8bed528` |

The [sidecar publisher](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF) provides this mixed weight encoding. Its validated DFlash metadata retains block size 8, the 2048-position window, five target input-layer features and RoPE sections `[64,0,0,0]`. Weight encoding is independent of the decoder KV format. The original Q4 sidecar remains a text regression control; changing sidecar precision can change acceptance and memory fit.

Proposal maxima are F=1-7 and verification widths TG1-TG8. Q8_0/Q4_0, Q8_0/Q8_0 and Q5_1/Q5_1 have exact decoder controls; F3 full native/streamed API controls also pass for all three pairs at the fitting short profile; that is not a long-context sweep of every pair. Adaptive execution inherits target cache types. Do not pass draft-cache overrides, even identical ones. LoRA, parallel slots, multi-GPU, audio/video, simultaneous MTP, K15 and K4V image requests remain outside this profile. This work does not qualify DFlash vision on Windows or other physical GPUs.

## One device parent and separate lifetimes

| Resource | During image encoding |
| --- | --- |
| Target weights, measured 12726.35 MiB | Remain outside the parent on GPU |
| Q2 mixed sidecar weights, measured 524.80 MiB | Remain outside the parent on GPU; no drafting runs concurrently |
| Full target KV history | Authoritative host history survives; resident/ring bindings retire and rebuild |
| Bounded DFlash KV and rollback publication staging | Protected persistent regions inside the same parent |
| Projector weights, 927588992 allocated bytes | Temporary vision grant inside the parent |
| Encoder attention and activations | Measured temporary vision workspace; no decoder history |
| Completed image embeddings | Owned host data survives release of the vision grants |

The arena bounds its participating allocations, not total VRAM, decoded host images, saved states, driver allocations or every future request. Image dimensions, token budget, batching, context and proposal width need their own fit. The existing fixed 2K-row prefill preparation stage and pipeline remain unchanged.

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
  --spec-mode dflash --draft-length 3 --image /path/to/photo.png \
  --image-min-tokens 64 --image-max-tokens 4096 \
  --decode-tokens 256 --probe-mode preflight --no-webui --port 1249 \
  --output benchmarks/results/my-dflash-image-fit
```

The finder starts and cleans up its own servers. Image preflight executes the complete supplied image request and actual decoding; it does not replace images with a short causal text surrogate. A configured projector without an image remains a text check. `--initial-arena-mib` is an optional search guess. The development 1024 MiB image probe is a fixed-budget successful request, not a measured maximum for arbitrary images.

Read the selected `arena_mib`, then launch a fresh server:

```bash
DF_ARENA=$(python3 -c 'import json; print(json.load(open("benchmarks/results/my-dflash-image-fit/result.json"))["arena_mib"])')
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
  --image-min-tokens 64 --image-max-tokens 4096 \
  --offline --no-context-shift --host 127.0.0.1 --port 8080
```

Open that server's Web UI and attach an image, or use the existing [HTTP image request](gemma4-vision.md#running-an-image-request-locally). There is no DFlash-specific image wire format. Refit when changing images, cache types, context, batch sizes, proposal limits or weights; a previous text-only budget is not an image fit guarantee.

## Numerical and lifecycle qualification

Fitting full native F1/F3 servers and streamed servers compare corresponding image scenarios with exact tokens and draft/prompt/cache counts. F7 full native startup cannot allocate the 927588992-byte projector alongside its decoder state, even with the smaller draft. That capacity failure is preserved. F7 instead has separately qualified native encoder export and native-decoder import of identical embeddings, positions and weights, plus end-to-end streamed API/lifecycle tests. This is native-arithmetic component coverage, not full native-server equivalence.

The internal framed oracle checks complete consumed target features, logits/hidden rows, selector outputs, target/draft KV and recurrent state through every accepted prefix, correction and TG1-TG8. F1/F3/F7 compare 504/907/1749 exact records; F7 covers 2834713708 bytes. The additional KV pairs retain exact native comparisons. The complete public image path also passes fresh strict device memcheck with API-all reporting and real DFlash predictions, plus actual host ASAN/UBSAN image checks. Normal warmup has a separate fitting native/streamed API control.

Expanded F1/F3 lifecycle tests each compare 23 native and 23 streamed scenarios. Their corresponding 321935388-byte state files have identical SHA256 digests. Tests cover changed/adjacent/interleaved/later images, portrait/narrow and first/last/only-image prompts, malformed media, image/context limits, cancellation, truncated state, erase/restore and process restart. Rewinds into an image or unavailable old history remain rejected. Each cached/restored scenario uses its corresponding native scenario; fresh-versus-cached native identity is not assumed.

The long F7 profile uses capacity 65536, a 48000-token text prefix, 512 outputs, a 1024 MiB device parent and a 4096 MiB host prompt cache. All 23 streamed scenarios pass; three initial/state/restart logs prove 188-191 active nonresident pages with 66 ring slots. A complete 1799655104-byte state with 48584 tokens survives restart. These are long lifecycle/self-consistency checks, not long native equivalence or a full native-context image-prefill qualification. The earlier 2048 MiB host-cache attempt correctly failed its cache-restoration proof because intervening entries evicted the large prompt; that failure remains recorded.

Attached-writer reload cancellation and a queued encoder exception preserve target/child state byte-for-byte, publish no failed output, retry exact encoder values and resume native-matching prediction. Fourteen host ASAN/UBSAN suites pass with leak detection. GPU ASAN uses NVIDIA's documented [CUDA compatibility setting](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/reference/troubleshooting-analysis-tools.html) `protect_shadow_gap=0`, retaining leak detection and halt-on-error. Strict device checks use default PDL, API-all reporting, error exit 99 and no filters; eager execution is limited to checker runs. No instrumentation OOM or initialization failure is counted as a pass.

## Ordinary performance

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

## Reproduction and release boundary

The [implementation roadmap](../../DFLASH2_VISION_PHASE_ROADMAP.md) records stage completion and scoped failures. Existing reusable tests are `tests/test-mtmd-projector-storage.cpp`, the image oracle in `tests/test-kv-stream-context.cpp`, `tests/test-server-vision.cpp`, and [the public vision harness](../../tools/server/tests/test_adaptive_vision.py). Local raw results and binary/source manifests remain outside the committed report.

K4V vision needs a separate media-boundary-aware text lookup and its own numerical/lifecycle qualification. It is explicitly rejected today. Text DFlash+K4V and existing Qwen/Gemma target-only/MTP vision retain their existing contracts; this work introduces no attention kernel, second arena or projector-weight quantization change.

### Finder admission tests

The existing finder rejection test retains the unsupported mixed DFlash+K4V image case; pure DFlash image admission is covered by the server vision host harness. All 31 finder tests complete successfully with one platform skip. The 17 vision-harness tests and nine actual startup rejections plus a native-matching positive request cover the current pure/mixed policy; the remaining tracked benchmark host suites also pass.
