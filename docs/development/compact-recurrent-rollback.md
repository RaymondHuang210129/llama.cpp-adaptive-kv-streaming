# Compact rollback: P0 frozen references

Source: `3ca9c5d4c520971568a9856d674d4744a69278a0`. Date: 2026-11-25.

The signed-in user authorized isolated GPU tests after production requests finished. The existing production container was restored after each GPU block. No candidate build was deployed.

## Provenance

Device: `NVIDIA GeForce RTX 5070 Ti, GPU-94db7ceb-31fc-464c-1ba7-7a0a2ca672a6, 610.57.04, 16303 MiB`.

Published Linux CUDA 13 / sm120a reference archive SHA256:
`ed1acc3bd0005b25d61afa8c8c403e3f77e9755abbf482254ef3b055199d3a70`.

Immutable local evidence: `/home/raymond/LLM/compact-rollback-20261125/`. Each reference JSON contains exact commands, model/draft/corpus/executable hashes, loaded-library hashes, prompt token hashes, response hashes, timing counters and logs. The source/build manifest is also in the extracted release. These are same-build native/RAM comparisons, not independent stock numerical equivalence.

## Matched executions

All profiles use 8,192 combined context, q8_0 K/V, 256 batch/ubatch, greedy 64 outputs, no startup warmup and no retries inside a measurement. Single-slot history is the finder's bounded preflight history. Two-slot probes include serial selections and genuine joint decode selections with equal and asymmetric histories. Each placement receives identical histories. These short probes freeze references; they are not a statistically powered speed sweep.

| Profile | Arena MiB | Matching output tokens | Reference JSON SHA256 |
| --- | ---: | ---: | --- |
| dflash-1 | 2096 | 64 | `52f885c0302d5cc8ceccad7856dfd77612d412b67718a42c94d510508acea020` |
| dflash-2 | 3072 | 256 | `4c9a364d6b1a2a7b1556f935f11fdedf61912b990bb6a3018203e488df5cd13b` |
| dflash-k4v-2 | 3072 | 256 | `a08194ccdaa2dff2d7f1cbc590cad91bc1fd1fc28c4ec7f4bf146fc620c37f87` |
| gemma-mtp-2 | 1536 | 256 | `499a4e3217167db530caf968fd576dc9331cd4d0d83b567919f00113b1b43f88` |
| gemma-none-2 | 1536 | 256 | `270b4ec07b645f9b3435cb911e84164b9c30229c3ea1417b4521d05a68655db8` |
| mtp-1-retry | 2096 | 64 | `b993aa7ae2650e7e77492714f2009f3365aae2ed46efa94efb8abccc7f215cac` |
| mtp-2 | 2096 | 256 | `bc132494cfc2e7c7e1104b0fee0b82c8ea69454d5a913338703e550b41773f6f` |
| none-2 | 2096 | 256 | `280d46d9a49d1ab86631e9273210959e89d6e1060f95a170be255055d01edbf8` |

Native and RAM tokens, proposal counts and accepted-proposal counts match in all eight profiles. The target-only and Gemma controls have no Qwen-style rollback matrix bank. Gemma MTP has an external assistant and its independent KV recovery. DFlash+K4V probes validate both neural and copy routes.

## Allocation and transfer audit

| Qwen, per selected sequence | Dense VRAM | Dense RAM |
| --- | ---: | ---: |
| Current matrix | 144 MiB | 144 MiB |
| Current convolution history | 5.625 MiB | 5.625 MiB |
| MTP3 additional rollback matrices/history | 448.875 MiB device | 448.875 MiB host |
| DFlash7 additional rollback matrices/history | 1047.375 MiB device | 1047.375 MiB host |
| MTP3 two-generation publication stage | none | 18.703125 MiB device |
| DFlash7 two-generation publication stage | none | 43.640625 MiB device |

The stage is the maximum per-layer snapshot payload times depth times two generations, not the sum of all layers. Selected groups multiply these per-sequence quantities by the actually activated sequence count. Sparse logical IDs do not widen the physical bank. Logs show MTP3 native banks of 598.50/1197.00 MiB for one/two selected sequences and DFlash7 banks of 1197.00/2394.00 MiB; RAM paths show current-only banks of 149.625/299.25 MiB. Deferred metadata buffers are zero-byte and are not physical payload.

Serial recurrent banks are outside the shared arena. Selected-group banks and publication staging are inside the phase arena. Model weights, draft weights and some scheduler reservations remain outside it. Consequently the same arena size does not imply the same total device footprint. Record arena-owned and external allocations independently.

The native GDN output contains attention plus K full matrix images, K=depth+1. Cache fusion writes these images directly to the destination bank when legal; the unfused graph retains the full output and copy nodes. RAM rollback retains dense intermediate graph outputs before publishing host snapshots, so host placement alone does not remove that graph peak. Existing graph quotes, rather than the payload table, are authoritative for admission.

Convolution snapshots are overlapping windows of `[previous history | verification inputs]`. Matrix rollback slot zero is the final state; slot j is j inputs back. A short block writes only its own trailing snapshots; older slots are not implicitly initialized. The batch splitter keeps the required tail in one microbatch.

RAM publication transfers the written noncurrent snapshots, bounded by depth times 149.625 MiB per selected sequence per verification; a partial restore uploads one 149.625 MiB image. Full acceptance avoids that restore. These are audited transfer payload bounds, not instrumented PCIe byte counters. Native rollback performs no equivalent host snapshot transfer. P7 must measure actual traffic and phase/KV effects rather than treating these bounds as observations.

Zero accepted proposals normally still commits the anchor verification input. A transaction abort instead keeps the pre-verification checkpoint. Prefix replay tests must cover both.

## TDD and limits

The new reference-comparison tests first failed because the harness did not exist, then passed four tests covering changed tokens, incomplete runs, changed prompts and missing requests. Existing CPU spill tests pass: 11,932 assertions, zero failures, exceptions or skips.

No compact storage, CUDA replay, raw-state equivalence, image qualification or four-slot candidate admission is claimed by P0. These are explicit later gates. Reference artifact commands are replayable with `benchmarks/compact-rollback-reference.py`; GPU memory must first be available.

## P1 capability and requirements contract

`llama-memory-recurrent-provider.{h,cpp}` describes dense device, dense host and capability-gated scalar-decay FP32 GDN payloads. It does not select a runtime provider or alter defaults. Callers supply the existing owner's measured graph workspace quote, actual selected count and remaining hard budget. Configured capacity never multiplies physical storage. Capabilities must come from qualified operators/backends; merely constructing a compact description does not qualify it. Other recurrent and auxiliary state retains its existing independent requirements.

The regression first failed on the missing provider contract. Five tests then pass 31 assertions for expected Qwen payloads, selected/configured separation, graph/stage overlap, one-byte budget shortfall, missing replay capabilities, incompatible grouped heads, invalid geometry/alignment, overflow, unchanged output on failure and no compact acquisition by target-only/empty controls. Existing sparse-ID and selected-storage tests remain in `test-memory-recurrent-spill`; that suite passes unchanged.
