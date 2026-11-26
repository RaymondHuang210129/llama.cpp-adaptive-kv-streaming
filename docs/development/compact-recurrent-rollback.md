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

At the frozen P0 baseline, serial recurrent banks were outside the shared arena. Selected-group banks and publication staging were inside the phase arena. Model weights, draft weights and some scheduler reservations remain outside it. Consequently the same arena size does not imply the same total device footprint. Record arena-owned and external allocations independently.

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

## P2 shared-history operator qualification

`llama-memory-recurrent-history.{h,cpp}` retains a single bounded timeline using ordinary graph-owned views/copies and reconstructs independently selected windows with the existing gather operation. It has no allocator or persistent raw workspace pointer. Host-validated descriptors distinguish checkpoint abort (prefix zero) from an accepted anchor (prefix one). All descriptor bounds are checked before modifying the output. A retained tail rebases a larger input block without copying K overlapping histories.

The missing history API first produced a compilation failure. CPU and the frozen sm120a CUDA backend then each passed 98,777 assertions, with zero failures, exceptions or skips. Cases cover one to four selected sequences, one to eight input rows, unequal per-sequence prefixes, abort/anchor/full acceptance, tail rebasing, repeated materialization and arbitrary FP32 bit patterns. The retained tensor has exactly `(history + retained rows) * channels * selected * sizeof(float)` bytes. Gather/transpose temporaries remain part of the graph quote and can be reused after their consumers complete.

This qualifies the bounded history operations. The P4 evidence below additionally closes their transaction, parking/reselection and production-provider integration gates. Production was restored after the isolated CUDA fixture.

## P3 exact FP32 GDN capture and replay

The missing capture/replay API first produced a compilation failure. CPU and CUDA fixtures subsequently each pass 4,128 assertions, with raw attention and matrix bytes compared against the native operator for every prefix. Cases cover widths 64/128, grouped keys, one to four sequences, unequal acceptance, abort, anchor-only acceptance, short blocks and retained tails of larger prefill blocks. CPU retains its scale-then-update vector routines. CUDA replay preserves the native CUDA update expression; CPU and CUDA records are not interchangeable under this exact contract.

A separate bounded fixture calls the unchanged GDN implementation from the frozen V3 CUDA library through its exported entry point and calls the newly compiled capture/replay kernels on the same inputs. All 4,128 assertions pass against that frozen implementation as well. Its bridge source, compile/link commands and logs are retained in the P0 evidence directory. This qualifies the operator; it is not a claim of full stock-server equivalence.

Capture stores final state, a checkpoint and FP32 evaluated decay, grouped keys and corrections. Its output has two matrix images, independently of the native rollback depth. It retains at most eight update rows, taking its checkpoint at the start of that tail for longer prefill blocks. Replay fuses the ordered prefix into one materialization pass. The scalar-prefix form needs no GPU descriptor allocation.

The initial native-exact CUDA contract is enabled only on the qualified Linux CUDA 13.0.88 / sm120a build and a compute-12.0 device. Other builds retain native state in the model adapter. An executable-retirement capability is also required; a graph mode that cannot retire temporary executables cannot acquire this provider. Registered inference operators remain separately testable on other backends.

The qualification build uses the repository's existing CI object-manifest mechanism to retain 221 unchanged CUDA objects from the full build. Modified GDN/dispatcher objects and host sources were rebuilt. Existing GGML declarations/layouts and op identifiers were checked unchanged; new identifiers were appended. The object manifest records source/object hashes and compiler commands. All FlashAttention quant pairs remain compiled.

CUDA memory checking reports zero errors, race checking zero hazards/errors/warnings, and synchronization checking zero errors. Graphs and PDL retain their normal settings. The combined host ASan/UBSan suites pass without errors or skips.

### Isolated operator timings

Measured with `test-backend-ops perf`, using Qwen geometry (128-wide states, 48 value heads, 16 grouped keys), in microseconds per layer operation. Replay restores half the captured inputs. The harness amortizes graph submission and reuses its data; these cached microbenchmarks are not end-to-end decode rates or physical PCIe/DRAM bandwidth measurements.

| Rows | Sequences | Native K images | Capture | Prefix replay |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 10.59 | 7.61 | 13.94 |
| 4 | 2 | 18.35 | 12.53 | 26.82 |
| 4 | 3 | 35.70 | 17.60 | 39.63 |
| 4 | 4 | 88.92 | 22.88 | 52.85 |
| 8 | 1 | 18.46 | 11.75 | 18.53 |
| 8 | 2 | 76.80 | 19.17 | 36.01 |
| 8 | 3 | 119.50 | 28.25 | 53.43 |
| 8 | 4 | 160.31 | 37.13 | 71.02 |

Replay cost is material even after reducing snapshot writes. P7 must measure its full inference cost, including integration copies, before retaining an optimization.

## P4 compact transactions and measured integration

`--recurrent-rollback-compact` selects the qualified GDN adapter with dense fallback for unsupported operators/backends. `--recurrent-rollback-ram` remains an independent explicit override; they are mutually exclusive. Defaults remain native. The finder and fixed-span sweeps preserve the requested policy in commands/results and record whether the compact provider was actually observed. Legacy host-spill environment switches cannot override an explicit compact request.

Each selected logical sequence has independent committed/working matrix and convolution images, FP32 update records, and a bounded convolution timeline. Physical capacity follows the selection, including sparse logical IDs. No dense intermediate rollback planes are allocated for this plan. Partial acceptance reconstructs directly into the working images, waits for completion, then publishes their roles together. Abort preserves the committed images and refuses uncommitted export until explicit cleanup. A restore failure retains its checkpoint/log source and can retry without publishing partially written destinations. Active verification rejects parking, rebinding and state import.

Long prefill retains its tail checkpoint as a graph-owned output while the original committed image survives verification. After graph computation and KV publication succeed, that output replaces the retired checkpoint image. Its overlapping lifetime is included in the normal graph quote. Prefix restoration has no additional device result allocation or prefix-descriptor buffer. Parking uses the existing owner/lease mechanism and preserves all internal resources; exported state still uses the native current-state format.

The full-model regression initially failed because hybrid input wrappers bypassed the recurrent input setter, leaving the new working-row indices uninitialized. The wrappers now use the shared setter and reuse checks. The exact comparison was retained. The same test now passes across depths 1/3/7 and histories 249, 505, 2,041 and 16,381: 1,873 assertions over full-vocabulary logits, recurrent bytes, retained KV bytes, forced rejection/correction and context-boundary failure.

The two-slot GPU test passes all 64 accepted-prefix pairs at depth seven, including correction, reconstruction and phase handoff: 6,844 total assertions including its ordinary control gates. Three-/four-slot MTP comparisons pass native features, acceptance and rollback (116/146 profile assertions). The injected owner-failure test passes its 65 profile assertions and closes admission before reconstruction. These are same-fork native comparisons, complementing the frozen GDN operator comparison rather than replacing an independent stock-server qualification.

The integrated CPU tests pass 330,205 assertions, including all depths 1-7, sparse IDs through 255, unequal prefixes, working-role swaps, retained prefill tails, native-format export/import, parking/rebinding, different reselection/import order, copying, source removal, slot reuse, abort, stale completion, active-lease rejection and retry after a real destination write fails. The combined ASan/UBSan suites pass with leak detection enabled. Existing dense spill, requirement and budget tests remain unchanged and pass.

### Actual representation payload

The conservative integration additionally retains two small convolution images. Its persistent tensor payload is therefore 11.25 MiB per selected Qwen sequence above the earlier two-matrix-plus-timeline illustration. The allocator quotes all four tensor families and their alignment; prefill checkpoint workspace and other consumers remain separate. CUDA buffer/selected-bank logs agree with these derived payloads.

| Per selected Qwen sequence | Native bank | Compact bank | Reduction |
| --- | ---: | ---: | ---: |
| MTP3 | 598.50 MiB | 318.41015625 MiB | 280.08984375 MiB |
| DFlash7 | 1,197.00 MiB | 331.9453125 MiB | 865.0546875 MiB |

Four selected MTP3 sequences have 1,273.640625 MiB of compact persistent payload, versus 2,394 MiB native. Target-only and Gemma do not acquire these resources. The convolution component alone has no standalone MTP3 memory saving under this conservative ownership policy. These are tensor/arena-region payloads, not a claim that driver allocation rounding or whole-process peak equals the table. At this P4 checkpoint, selected banks lived inside the existing parent arena while serial banks remained external. The P5 correction below moves serial banks into that arena too.

### Untuned end-to-end reference

One no-startup-warmup pass, Qwen Q3 target, embedded MTP3, q8_0/q4_0 KV, 8,192 combined budget, 2,096 MiB arena and 512 output tokens per request. The finder uses bounded article histories and its single/joint profiles. Native, RAM and compact outputs match exactly: 2,048 output tokens compared per placement. This is not the SWE sweep. The one-request rows are selections within two configured slots, not a separate `--parallel 1` implementation. Joint columns are the individual request-reported rates; no exact common-window aggregate is inferred by summing them.

| Placement | Single selection A t/s | Single selection B t/s | Joint slot 0 t/s | Joint slot 1 t/s |
| --- | ---: | ---: | ---: | ---: |
| Native VRAM | 102.02 | 102.00 | 69.17 | 73.41 |
| Dense RAM | 96.24 | 95.45 | 61.36 | 66.19 |
| Compact prototype | 100.38 | 100.12 | 68.86 | 73.10 |

The memory reduction alone does not establish an end-to-end speedup. P7 must reduce capture/integration and replay costs, then remeasure matched workloads. An immutable pre-optimization runtime and hashes are retained in `P4-runtime/` and `P4-runtime-manifest.json` under the evidence directory. GPU pauses were bounded to the tests; the original production container/image was restored afterward. No candidate has been deployed or pushed.

## Qualified capture/replay optimization checkpoint

P7 kernel work was qualified while the P5 policy and P6 matrix remained in progress. Direct capture writes the working matrix and update records into the owner's existing persistent banks. Its graph output contains attention and, only for a long prefill, the retained tail checkpoint. This removes the intermediate final-matrix and record copies. Replay handles multiple state elements per thread and preserves the frozen native multiplication/FMA order with explicit rounding intrinsics. A first vector trial failed 80 bitwise checks and was rejected before the corrected implementation passed.

The expanded operator fixture passes 2,008,756 assertions on CPU, current CUDA, and the unchanged frozen native CUDA bridge. It checks captured factors, every prefix, reversed physical mappings, untouched log padding, signed zero, subnormals and decay underflow. The retained CUDA contract is restricted to the qualified Linux Release/NVCC 13.0.88/sm120a build and normal graph/PDL settings. The integrated CPU transaction suite still passes 330,205 assertions. CUDA model checks pass the 6,844-assertion two-slot prefix suite and the three-/four-slot MTP profiles. Final CUDA memcheck and synccheck report zero errors; racecheck reports zero hazards, errors or warnings. ASan/UBSan host suites pass with leak detection enabled.

A matched, single-pass SWE A/B comparison used 8,192 history tokens and 512 outputs per request, two genuinely selected requests, q8_0/q4_0 KV, a 262,144 combined context budget and a fixed 2,096 MiB arena. Every output token matches across placements. The common decode interval, rather than a sum of per-request rates, measured:

| Placement | Aggregate tokens/s |
| --- | ---: |
| Native device snapshots | 140.350 |
| Dense RAM snapshots | 126.441 |
| Frozen compact prototype | 138.588 |
| Optimized compact | 141.715 |

These are single observations without startup warmup, retries, repeated timing trials or statistical confidence intervals. The optimized compact result is 2.26% above the prototype in this short-context case; it is not a general throughput promise. All four sampled whole-device peaks were 15,537,405,952 bytes. The fixed arena reuses the reduced bank's capacity internally; it does not reduce the arena's physical allocation.

Raw records live under `/home/raymond/LLM/compact-rollback-20261125/P7-swe-qwen-mtp-n2-h8192-a2096-*`, with commands, runtime-library hashes, pinned SWE A-D token-bank hashes, actual prompt hashes, token timelines and NVML samples. The mathematical and memory improvements in this checkpoint do not close P5-P7: placement qualification, broader product profiles and long-context/refitted measurements remain open.

## P5 budget-aware placement

P5 adds `--recurrent-rollback-auto` for selected-group activation. The common selector accepts complete owner quotes for native device, dense host and qualified FP32 GDN storage. It has no model-name rules. Device peaks include current/bounded state, graph workspace, publication staging and the minimum KV pool. Advisory latency includes capture, acceptance-dependent restoration, KV streaming and the maximum of concurrent H2D/D2H payload times. The bandwidth/compute seeds are estimates, not measured PCIe utilization. The initial full-overlap assumption is superseded by the P7 calibration below. Independently restorable partial regions are not advertised; the qualified choices are whole-bank plans.

Placement changes occur only after the old group has drained and exported its current native roles. Candidate descriptions allocate no device state; a fresh normal admission check precedes each actual arena binding. Cached descriptions advise cost only. A cached choice that fails fresh admission triggers one bounded refresh. The cache retains one selected-identity profile, no GPU leases or executable tensor addresses. It is invalidated on an explicit provider change. All publication metadata is prepared before acquiring GPU grants.

The policy keeps eight real-work correction samples per representation and a workload key, with 5% hysteresis and a four-boundary cooldown for an unchanged workload. It shares a common correction until competing feasible representations have comparable samples, avoiding a penalty on the only measured provider. Explicit native/default, RAM and compact launch choices remain reproducible. The internal parked-owner setter supports qualified explicit provider changes and rejects an active group or a lent vision arena.

The initial P5 commit still left serial recurrent banks outside the shared arena, with placement fixed at launch. This behavior is superseded by the serial arena correction below. `benchmarks/select-recurrent-placement.py` fits native/RAM/compact separately, subtracts an explicit reserve, checks one full workload per feasible choice, compares exact output tokens and ranks measured common decode intervals. It emits a fixed command and retains the previous choice within the 5% margin. It does not start a production server or migrate a live serial bank. Use a free GPU, a fresh output directory, and `--no-warmup` when startup warmup is unwanted. For example:

```bash
./.venv/bin/python benchmarks/select-recurrent-placement.py \
  --server ./build/bin/llama-server --model /path/to/Qwen-with-MTP.gguf \
  --spec-mode mtp --draft-length 3 --parallel 1 --context-size 8192 \
  --cache-type-k q8_0 --cache-type-v q4_0 --prompt-tokens 2048 \
  --decode-tokens 512 --no-webui --no-warmup --output arena-placement
```

The result prints the selected provider and `arena_mib`; `selection.json` retains every fit and measurement, and `command.txt` contains the complete selected command. Default reserve is 16 MiB. Windows users must configure the executable's NVIDIA profile to Prefer No Sysmem Fallback before fitting; the tool cannot certify that driver setting. A full workload that fails validation is not a timed valid candidate.

TDD evidence includes missing-API/CLI regressions, tight budgets, transfer overlap, malformed costs, hysteresis across workload changes, bounded feedback and an attempted active-owner switch. The deterministic provider suite passes 85 assertions. Real GPU migration exercises native -> RAM -> compact and compact -> RAM -> native on one retained parent, with 176 profile assertions. The final cached-policy three-/four-slot MTP and all 64 two-slot depth-seven acceptance pairs pass, including 6,844 prefix assertions; the failed-owner gate passes as well. The four targeted host suites pass with both ASan and UBSan, including leak detection.

The first runtime policy trial chose the correct compact bank but repeatedly quoted losing plans between the first output token and the decode group. It measured 130.863 tokens/s versus 141.715 for explicit compact. Cached advisory quotes reduced that transition cost: the repaired trial measured 139.891 tokens/s, passed the declared 98%-of-explicit-compact gate and matched all 1,024 output tokens. The failed trial, gate and repaired result are preserved in `P5-auto-planning-{red,green}.json`; the failed trial is not discarded or averaged into the retained result.

At two SWE histories of 16,384 tokens, a 2,096 MiB arena, a 262,144-position combined budget and 512 outputs per request, the measured common-window rates were native 128.250, RAM 118.747, prototype compact 129.262, optimized compact 132.299 and automatic 131.713 tokens/s. Outputs match across all five runs. These are single observations. At short context, the serial launch selector fitted native/RAM/compact maxima of 2,980/3,394/3,260 MiB; after its 16 MiB reserve, measured rates were 113.181/108.997/113.987 tokens/s. It retained native because the compact difference was below its switching threshold. No timing repetitions, reverse-order pass or discarded startup requests were used.

The kernel checkpoint commit `c2b4ea696` also builds independently in `kernel-check-source/`; its four CPU component suites pass without the in-progress placement changes. Raw P5 records, commands, library/model identities, capacity boundaries, output timelines and sampled GPU peaks are retained under `/home/raymond/LLM/compact-rollback-20261125/`. Larger-context performance and the complete P6 matrix remain separate gates below.


### P5 serial arena ownership and live placement

The serial path now places its native, dense-host, or compact device bank in an owner-issued persistent prefix of the existing phase arena. RAM rollback keeps its device current image and publication staging inside that budget; its rollback snapshots remain in host RAM. The diagnostic `recurrent_state` field reports the protected device bytes. Target/draft workspace and elastic KV residency use the remaining region, with staging and bounded KV retained as protected suffixes. There is no second GPU parent allocation.

Automatic serial placement starts with the smallest device bank and describes all available providers at safe request/text-phase boundaries. Complete quotes use the actual serial graph topology and include target/draft graph envelopes, recurrent storage, publication, attention/writer scratch and minimum KV capacity. Infeasible plans remain excluded. The shared bounded cost selector includes acceptance-dependent replay, host transfers and displaced KV traffic, with its existing hysteresis and cooldown. Explicit native/RAM/compact launch overrides remain reproducible. The launch-time measurement helper is optional tooling.

A provider change drains affected work, materializes a native current-state checkpoint, retires executable graphs, suspends target and child scratch, and resizes the recurrent prefix on the same physical parent. It rebuilds workspace bindings, imports the checkpoint and reopens execution. Host scratch can grow when a different graph needs it; device admission remains within the original parent. An over-budget candidate is rejected before retiring the original state. A failed scratch-rebuild callback keeps submission gated until successful recovery. Serial draft scratch excludes the prefix after every resize. RAM restoration checks the actual draft grants for disjointness, allowing target and draft to share a parent safely. A separate MTP GPU workspace is no longer a fallback when the target has an arena-owned recurrent bank. Vision borrowers preserve the fixed regions too.

The new arena tests began with missing reservation/replan operations and the serial migration test exposed stale phase activation and incompatible serial graph measurement. Those failures are retained in `P5-serial-*.log`. One diagnostic test also read an asynchronous logits buffer through an internal accessor; it was corrected to use the synchronizing public API before comparing bytes. No arithmetic tolerance was introduced.

The full-model migration test passes 69 assertions: native -> RAM -> compact -> native, a partial acceptance rollback, every vocabulary logit and every serialized current-state byte, automatic placement in 512/3072 MiB arenas, and rejection of a native bank that cannot fit the smaller budget. The ownership test passes 399 assertions proving all serial recurrent tensors are within the parent and their allocation matches diagnostics. Sixteen separate MTP3/DFlash7 text/image probes produce identical 64-token outputs across native, RAM, compact and automatic placement. Three-/four-slot MTP and the 6,844-assertion two-slot acceptance-prefix suite remain exact. Targeted ASan/UBSan suites pass with leak detection; final migration CUDA memory checking is recorded separately.

Raw evidence and frozen runtimes are under `/home/raymond/LLM/compact-rollback-20261125/`: `P5-serial-final-lifecycle.log`, `P5-serial-final-probes.log`, `P5-arena-final-gates.log`, `P5-serial-asan-tests.log`, and `P5-serial-arena-final-runtime-manifest.json`. Earlier serial performance records above describe the pre-correction allocation policy and must not be relabeled as measurements of this implementation. P6 still needs the remaining product matrix and corrected serial profiles; P7 still needs its final matched performance/resource comparisons.

### Serial cost-query reuse checkpoint

A serial cost query now reuses at most three host-only graph templates. Each query still obtains a fresh complete arena quote and recomputes displaced KV traffic and acceptance-dependent cost. Queries for the active provider use its actual current workspace plan. A different winner always receives new graph measurement and admission before replacing storage. A graph reconfiguration invalidates templates; an unsuccessful cached choice permits one bounded refresh. Unsupported providers are cached as unavailable. No extra GPU storage or timing requests are introduced.

The replacement quote now releases the previous target's scratch envelope instead of charging the maximum of old and new target workspace. It retains every live child's phase requirement, using the same calculation for quoting and actual workspace rebuilding. A regression first rejected a valid 12 KiB state + 4 KiB workspace replacement in a 16 KiB parent; the corrected quote admits it and rejects it again when a live child needs 8 KiB. The complete owner suite passes 8,879 assertions.

The GPU cache regression first observed executable revision 4 becoming 5 after a repeated cost query. It now preserves the executable and also observes an increased quota when a new protected suffix is reserved, proving cached templates do not bypass current admission. Native/RAM/compact migration, all vocabulary logits and serialized current state remain exact. Actual MTP3/DFlash7 text/image automatic handoffs match the native 64-token references and report cached decode-phase quotes. The four host ASan/UBSan suites remain green with leak detection.

A single serial MTP3 measurement used pinned SWE suffixes A/B, 8,192 history tokens each, 512 outputs per request, q8_0/q4_0 KV, a 262,144-position context limit and a common 2,304 MiB arena. All 1,024 output tokens match across these runs:

| Placement | Mean A/B tokens/s |
| --- | ---: |
| Native device snapshots | 115.255 |
| Dense RAM snapshots | 111.901 |
| Explicit compact | 116.330 |
| Automatic before query reuse | 115.740 |
| Automatic with query reuse | 116.382 |

The query-reuse observation is 0.55% faster than the previous automatic run and has the same sampled device peak, 15,770,189,824 bytes. These are single observations without timing repetitions, discarded startup requests or statistical confidence. Compact storage recovers capacity inside the fixed parent; it does not reduce the parent's allocation. A 2,096 MiB serial native MTP3 launch at this full context limit is not admitted: its complete quote requires 2,326,112,384 bytes, rounded up to 2,219 MiB. This is a lower-bound quota result, not a physical OOM or a timed sample. The fixed 2,096 MiB parallel control remains a separate supported measurement.

Evidence is retained in `P7-serial-cache-{red,green}.log`, `P7-serial-complete-admission-green.log`, `P7-serial-cache-asan-tests.log`, `P7-cache-trial-queue.log`, `P7-capacity-limits.json` and the `P7-swe-qwen-mtp-n1-h8192-a2304-*` directories under the evidence root. Runtime and loaded-library identities distinguish these records from historical serial allocation results. This checkpoint does not close the remaining P6 matrix or the P7 long-context/resource comparisons.

## P6 product configuration results

The bounded configuration gate passes all 528 main pairs: 432 Qwen target-only/MTP1-3/DFlash1-7/DFlash+K4V profiles and 96 Gemma target-only/MTP1-3 profiles, each across Q8V8, Q8V4, Q4V4, one through four configured slots, and text/still-image inputs. Multi-slot cases must prove the requested selected count actually decoded; they do not qualify a serialized fallback as parallel execution. Qwen speculative controls use the dense RAM provider so the largest banks can be qualified on this card; target-only and Gemma retain their applicable native path.

The main matrix was frozen at the qualified direct-capture/vector-replay kernel checkpoint. Following the serial ownership correction, every one of its 132 serial combinations was run again against the corrected implementation. All 132 pass, including all 108 Qwen cases and all 24 Gemma controls. A further 48 pairs cover neural/copy widths 1/1 and 7/1 across the same KV formats, slot counts and modalities. They all pass on the corrected runtime. These 708 pairs compare 43,584 unconstrained output tokens; the mixed-mode probes additionally validate target-biased copy and genuine-miss routes. They are bounded product probes, not throughput samples or an independent full-model stock-server equivalence claim. The raw operator bridge and forced-prefix/state/logit fixtures supply the compositional numerical contract.

[The qualification CSV](compact-recurrent-rollback-qualification.csv) records every pair's context/arena/output limits, selected-count proof, sampled whole-device peaks, allocation-policy lineage, runtime fingerprint and relative evidence path. The evidence root remains `/home/raymond/LLM/compact-rollback-20261125/`. Historical serial records are labeled separately; the corrected serial matrix is the applicable serial result. The Gemma Q8V8/MTP3/four-image pilot is reused with its original receipts rather than rerun: it proves four selected slots and a 16,908,943,360-byte sampled device peak at a 2,304 MiB arena.

Two probe issues were corrected without changing inference policy. The grouped image copy check had required zero neural forwards despite `min_hits=2`, which necessarily invokes one cold-lookup fallback. It now requires the configured number of cold fallbacks and rejects additional ones. An attempted output-tail alignment did not remove that call and was rejected; both failed attempts are retained. The arena finder also recognizes recurrent-state and workspace admission quotas as lower bounds, separately from physical allocation failures. Red/green unit records and the complete 40-test Python run are retained; its two optional integration tests are covered by the explicit GPU runs rather than environment-driven discovery.

The final cached-migration fixture passes 75 model assertions (1,606 including its argument/contract suites) under CUDA memcheck with zero errors. Raw-kernel racecheck/synccheck/memcheck remain zero on the unchanged qualified kernels, and the four affected host component suites pass ASan/UBSan with leak detection. The current-state tensor containment, poisoned-export, forced-prefix, sparse-ID, host-publication, child-workspace and bidirectional migration gates remain in force.

Single-slot non-regression controls use 8,192 history tokens, 512 outputs per measured request, independent A/B prompts, greedy sampling, no startup warmup, and one pass per runtime. Qwen uses the pinned SWE suffixes; Gemma uses independently tokenized article ranges, not Qwen vocabulary IDs. Every output token matches before/after:

| Control | Before tokens/s | After tokens/s | Sampled GPU peak before/after |
| --- | ---: | ---: | ---: |
| Qwen target-only, 2,304 MiB arena | 48.204 | 47.992 | 15,564,668,928 / 15,407,382,528 bytes |
| Gemma target-only, 2,048 MiB arena | 41.727 | 41.679 | 16,147,677,184 / 16,147,677,184 bytes |
| Gemma MTP3, 2,048 MiB arena | 89.490 | 89.483 | 16,648,896,512 / 16,648,896,512 bytes |

The observed rate differences are -0.44%, -0.11% and -0.01%, respectively, without statistical confidence from repeated timings. The Qwen target-only device peak drops by exactly 150 MiB when its normal current recurrent image moves into the fixed parent; Gemma allocates no new recurrent resources. Long-context and instrumented resource comparisons are recorded in the P7 closeout separately.

## P7 final optimization and placement results

P7 retains direct capture into the owner-issued working bank/logs, vectorized exact prefix replay, a full-acceptance path without replay graph allocation, and host-only advisory graph-template reuse. These changes preserve the qualified FP32 arithmetic and the committed/working ownership protocol. Replay already batches affected layers/sequences into one backend graph and writes directly to the prebound working image. There is no extra device result allocator. Long-prefill tail checkpoints reuse measured graph workspace until computation and KV publication complete.

Returning the working image as scratch at the draft boundary would discard accepted current state; copying it over the checkpoint would destroy the original state required by late prefix rollback. Retaining a borrowed lease through that lifetime has the same two-image lower bound. A sealed-commit/deferred-merge or offloaded-checkpoint provider would need a different qualified contract. Those candidates were not substituted for the proven two-image provider. The measured replay share below does not justify adding a new cross-layer fusion or asynchronous state-publication mechanism in this delivery.

### Corrections found by the final performance gates

The first 2x64K automatic run selected RAM and measured 64.923 aggregate tokens/s, while explicit compact measured 77.792. Its cost seed had assumed perfect overlap of all H2D/KV and D2H work without qualification. Transfer cost now starts conservatively with both directions retained. An internal, finite `[0,1]` overlap fraction permits explicitly qualified overlap; it defaults to zero. The previous full-duplex unit cases now request that assumption explicitly. A new regression first selected RAM incorrectly, then passes with the conservative default and still proves a qualified full-duplex case can choose RAM. The provider suite passes 91 assertions, including invalid-overlap rejection and unchanged output on failure. Existing bounded real-work feedback, workload keys, hysteresis and cooldown remain intact.

A second gate found that preliminary server admission quoted only its default native bank. Four-slot DFlash7 therefore reported a 5,214 MiB minimum and repeatedly selected pairs even though compact could admit all four slots in 3,072 MiB. Admission now checks complete available representation plans before pruning the cohort. It tries compact first to avoid redundant larger-bank measurement; actual activation still makes its own cost choice and fresh hard-budget admission. No state bank is allocated by this preliminary quote. The real-server regression first failed, then proved four selected decode slots within the same arena.

After the capacity fix alone, automatic four-slot throughput was 136.130 tokens/s. Logs isolated the remaining gap to admission setup before decode: steady four-slot execution took approximately 7.24 seconds versus explicit compact's 7.27 seconds. Avoiding redundant native/RAM preliminary quotes brought the final automatic observation to 157.529 tokens/s. Both failed trials and their raw logs are retained; they are not treated as retained performance results. Serial migration, the three-/four-slot MTP oracle and all 64 two-slot seven-token acceptance combinations pass again after the costing/admission corrections.

### Matched throughput

All normal runs use greedy sampling, no discarded startup requests, no startup warmup, and 512 outputs per request. Serial rows average independent SWE A/B requests; parallel rows measure emitted tokens in the actual common decode interval. Every retained comparison checks prompt identities and all output tokens. These are single observations without repetitions, reverse-order passes or statistical confidence. Instrumented rates are excluded.

| Qwen workload | Arena policy | Native | RAM | Compact prototype | Optimized compact | Final automatic |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Serial MTP3, 8K A/B | Common 2,304 MiB | 115.255 | 111.901 | - | 116.330 | - |
| Two-slot MTP3, 8K each | Fixed 2,096 MiB | 140.350 | 126.441 | 138.588 | 141.715 | - |
| Two-slot MTP3, 16K each | Fixed 2,096 MiB | 128.250 | 118.747 | 129.262 | 132.299 | - |
| Two-slot MTP3, 64K each | Fixed 2,096 MiB | 68.534 | 64.776 | 75.051 | 77.792 | 77.858 |
| Four-slot DFlash7, 8K each | Fixed 3,072 MiB | Not admitted by state size | 92.493 | - | 157.300 | 157.529 |
| Serial DFlash7, 96K A/B | Each fitted maximum minus 16 MiB | 108.182 | 105.576 | - | 110.386 | - |

The 2x64K optimized observation is 13.5% above native and 3.7% above the compact prototype. Final automatic placement matches the explicit compact path closely. Four-slot optimized compact is 70.1% above the applicable RAM reference. The refitted 96K serial observation is 2.0% above native. The 96K fitted/used arenas are native 3,330/3,314 MiB, RAM 3,316/3,300 MiB and compact 3,332/3,316 MiB. The full 96K requests validated those selected sizes once; all six request outputs match. The 2,096 MiB serial native MTP3 capacity limit remains a separately recorded non-admission, not a speed sample.

[Normal performance CSV](compact-recurrent-rollback-performance.csv) records rates, acceptance, real arena sizes, sampled whole-device peaks, process RSS, runtime fingerprints and relative raw-evidence locations. Prototype and historical serial allocation records are kept distinct from the corrected serial arena measurements. Larger logical histories do not require relocating authoritative host KV.

### Arena and transfer accounting

The fixed parent's total allocation remains unchanged; compact storage increases the usable KV region. Payloads follow the actual selected sequence count. For MTP3, native/compact persistent recurrent storage is 598.500/318.410 MiB per selected sequence; for DFlash7 it is 1,197.000/331.945 MiB. The conservative convolution component alone is not smaller for MTP3, and a depth-one compact bank is not advertised as a universal memory reduction.

The instrumented 8K workloads report the following actual verify-phase grants:

| Workload/provider | Parent MiB | Recurrent MiB | Publication MiB | Graph MiB | KV pool MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP3 selected 2, native | 2,096 | 1,197.000 | 0 | 162.273 | 704.031 |
| MTP3 selected 2, RAM | 2,096 | 299.250 | 37.406 | 162.273 | 1,564.469 |
| MTP3 selected 2, compact | 2,096 | 636.820 | 0 | 162.273 | 1,264.250 |
| DFlash7 selected 4, RAM | 3,072 | 598.500 | 174.562 | 132.830 | 2,083.250 |
| DFlash7 selected 4, compact | 3,072 | 1,327.781 | 0 | 50.330 | 1,611.188 |

Bounded draft KV, writer/attention scratch and alignment are also charged in these parents; the table does not omit them from admission. Compact prefill graph envelopes are larger because they retain tail checkpoints until publication: MTP3 selected 2 uses 487.276 MiB, versus 208.510 native/RAM; DFlash7 selected 4 uses 795.276 MiB, versus 298.213 RAM. Decode releases that phase workspace to KV. The four-slot native DFlash7 bank alone needs 4,788 MiB and cannot be a 3,072 MiB control.

Across the complete instrumented MTP3 requests, RAM records 172.587 GiB of rollback D2H and 3.215 GiB of rollback H2D; compact records zero of either and 14.925 ms in 20 replay graph executions. DFlash7 selected 4 records 1,009.339/53.625 GiB of RAM rollback D2H/H2D; compact records zero and 255.452 ms in 141 replay graph executions. These counters include prefill, decode and applicable retirement, not just the timed common interval. Replay timers include backend graph execution and synchronization rather than only device kernel time. Prefix logs record actual materialized rows/prefixes; maximum-width rollback counters are not substituted for that distribution. [Instrumented resource CSV](compact-recurrent-rollback-resources.csv) provides the complete phase grants and request totals.

A separate Nsight capture uses a client NVTX range beginning after every cohort's first token and ending at the first completed stream. Its 1,024 output tokens match the normal compact reference. Within the 7,436.375 ms common interval, GPU kernel work sums to 6,031.062 ms; its union is 5,710.439 ms, and the union of kernels/memory operations is 5,826.782 ms. Quantized matrix multiplication accounts for 4,470.325 ms (74.1% of cumulative kernel work), attention 592.080 ms, GDN capture 72.790 ms and replay 7.791 ms. Capture plus replay is approximately 1.1% of client-observed decode wall time. Work sums are not exclusive critical-path time because operations can overlap. Other D2H traffic remains for logits/features even when rollback D2H is zero. [Decode-window kernel CSV](compact-recurrent-rollback-nsys-kernels.csv) and `P7-nsys-decode-summary.json` retain the detailed classification. Profile rates are not included in the throughput table.

### Qualification scope and completion

The retained exact CUDA contract remains the qualified Linux Release/NVCC 13.0.88/sm120a build with normal graphs/PDL and the existing compiler guards. CPU and CUDA each match their retained native arithmetic; this is not a cross-backend bitwise claim. Other builds/operators retain their applicable native fallback. This delivery does not add new model pairings, speculation depths or KV formats. Higher configured capacities/sparse IDs retain the existing generic bound; actual GPU product qualification on this card covers one through four selected requests. Admission may select a smaller cohort when its complete histories/resources do not fit, and such a run is not scored as full parallelism.

All P0-P7 delivery stages are complete. Final CPU/provider/owner/transaction/history checks, ASan/UBSan suites, kernel CUDA order/memory checks, full-model prefix/state/logit gates, product configuration pairs and matched non-regression controls pass under their stated scope. No numerical tolerance was relaxed for a speed result. The original production container/image is restored and healthy with three slots; the feature has not been pushed or deployed. Source commits and immutable runtime/library/model/workload identities are retained with the evidence.

## Native workspace accounting after deployment

The 2026-11-25 production crash occurred while binding three decode selections with histories 30,311, 17,539 and 40,303 tokens. Native MMQ scratch grew a CUDA VMM pool outside the fitted phase arena. A fresh short replay initially passed, so startup and one successful request were insufficient qualification.

Shared-arena target and draft schedulers now charge native operator scratch, explicit cuBLAS workspace and alignment to their existing compute loans. The adapter covers native weight/KV quantizations, attention dispatch, top-k, argsort and softmax without changing arithmetic. Logical KV/recurrent owners explicitly declare bounded execution; direct native attention also quotes its native scratch. Controlled work requiring an absent or insufficient bank fails before submission. Cleanup copies do not recreate an owned library workspace. See [the workspace contract](../../DEVICE_MEMORY_INFRASTRUCTURE.md#private-phase-and-operator-workspace).

Metadata preparation does not create unused cuBLAS handles for transient contexts. Synchronizing an idle backend does not create a CUDA stream. Actual work still initializes its required handles/streams and synchronizes normally. Covered bindings retain their root; failed detach invalidates reuse and retains cleanup ownership. Storage is validated on every submission, including captured graph replay. Tensor checks are reused only within one validation pass; matmul caching retains tile selection while recalculating scratch from current dimensions.

Performance neutrality is the target. Matched Qwen IQ4_XS / DFlash7 / compact / Q8V4 runs use the same 2,096 MiB arena, 256K context budget, SWE histories of 8,192 tokens per request and 512 outputs per request. There are no startup warmups or timing retries. The reference is frozen previous production commit `45bf32819`. The three-slot speed covers the common active decode window and excludes later reduced-parallelism tails.

| Selected slots | Previous production tokens/s | Candidate tokens/s | Change | Exact output tokens |
| --- | ---: | ---: | ---: | ---: |
| 1 | 157.30 | 157.08 | -0.14% | 512 |
| 3 | 157.91 | 156.40 | -0.95% | 1,536 |

These are individual matched measurements, not a confidence interval or a full performance sweep. Earlier candidates regressed about 4.5%; CPU profiling located repeated workspace planning. The retained implementation avoids per-node temporary allocations, repeated tile selection and duplicate tensor validation. Paired traces of the earlier slow candidate had identical kernel counts and transfer bytes, with effectively unchanged kernel times. The final sampled three-slot whole-device peak is 42 MiB below the previous build at the same arena size.

TDD receipts cover missing native quotes, late 32 MiB owned-workspace restoration, excessive rebinding, failed-detach reuse, changed storage ownership at an unchanged data pointer, managed-only completion ownership, and unnecessary handle/stream initialization. The final suites pass 1,446 generic CUDA assertions, 1,479 scheduler assertions, 2,471 native proposal/attention assertions, 841 text-quote assertions, 1,614 scheduler-failure assertions, 814 library-failure assertions, 712 library-ownership assertions and 628 idle-synchronization assertions. CPU ownership tests pass 621 ASan/UBSan/LSan assertions. Native attention/proposals and scheduler matmul tests pass CUDA memcheck with zero reported errors.

All 18 supported model/mode/vision/provider comparison cases match 4,992 output tokens from the previous build. After the near-limit workload, the 13 asymmetric cohorts additionally match 864 original-build tokens, for 5,856 exact compared tokens. This is previous-fork equivalence and native-operator coverage, not independent stock-server equivalence or MS17 closure. Native DFlash IQ4 placement remains outside the qualified card profile; its native control uses the documented Q3 weight profile.

The qualified production configuration is Qwen IQ4_XS with vision, DFlash7, compact rollback in VRAM, Q8V4 KV, three configured slots and a 262,144-token selected-group budget. Full-lifecycle calibration admits a **2,314 MiB arena with zero reserve** and rejects 2,315 MiB. A fresh process then completes three 86,784-token SWE histories with 512 outputs each, all 13 asymmetric cohorts and three large image requests at the 4,096-token image limit: 33 requests and 2,496 generated tokens. Native pool growth and external owned-library workspace events are both zero. Long-prefix replay was first checked against known failing sizes before being used to accelerate calibration; the final empty-slot execution independently validates the recommendation.

Opaque CUDA/library initialization and graph-executable metadata remain outside the arena and must be included by physical workload validation. A minimum-size image and an aligned short prompt can miss the real peak. The finder adds a wide-tail prefill profile, supports `--fixed-image-budget` to preserve the requested image limit for all reported ratios, and refits the full workload if `--validate-selected` observes a physical OOM. It does not convert non-capacity failures into OOM bounds. Placement selection defaults to the validated maximum with no automatic subtraction; an explicit user reserve remains optional.

Immutable local evidence is under `/home/raymond/LLM/production-crash-45bf32819-20261125/`: frozen runtime/source manifests, red/green logs, response/timing records, traces, `lifecycle-context-fit.json`, the fresh full-lifecycle receipt and `final-qualification.json`. The qualification is specific to the RTX 5070 Ti, CUDA 13.0.88 / sm120a, recorded models and tested workload envelope. Other workloads, driver state or processes can change the physical fit.
