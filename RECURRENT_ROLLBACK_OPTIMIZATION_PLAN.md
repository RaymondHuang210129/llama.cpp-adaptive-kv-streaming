# Compact recurrent rollback and budget-aware placement

Status: P0-P5 complete; P6-P7 remain in progress. Implementation branch `feature/compact-recurrent-rollback`. Exact operator and integrated transaction gates passed on the qualified backend; budget-aware placement is qualified; broader product qualification and performance closeout remain open.

Evidence: [P0 reference and allocation ledger](docs/development/compact-recurrent-rollback.md).

Date: 2026-11-25.

Tracking issue: [#23 - Plan rollback placement by memory capabilities and end-to-end decode cost](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming/issues/23).

Execution contract: implement P0 through P7 with recorded red/green evidence and a commit at each completed stage. Preserve unrelated local files. Never claim a stage complete while its required correctness, lifetime, admission or qualification gates remain open.

Inspected baseline: `3ca9c5d4c520971568a9856d674d4744a69278a0`, on `feature/adaptive-kv-stream-v3`. Freeze the actual source, binaries, model files and configuration again before implementation or measurement.

## Objective

Reduce speculative rollback storage and transfer traffic while preserving the committed model state, supported features and the fixed device-memory budget. Use the recovered capacity to improve KV residency or admitted concurrency when that reduces total decode latency.

The first compact provider targets the Gated DeltaNet operation used by the supported Qwen target. Common memory planning must remain independent of model names. Gemma and every other supported combination must continue to work through their applicable providers or the existing conservative fallback.

This is a representation and placement project. It does not extend supported MTP depth, introduce new model pairings, or relax numerical acceptance to obtain a performance result.

## Research basis

Current recurrent memory stores convolution history and matrix state in separate tensors, but native rollback widens both into `1 + n_rs_seq` groups. The graph copies overlapping convolution windows into those groups and asks the GDN operation to produce dense intermediate matrix states.

Relevant implementation locations:

- [State dimensions](src/llama-hparams.cpp): `n_embd_r()` and `n_embd_s()`.
- [State allocation and selected-slot storage](src/llama-memory-recurrent.cpp).
- [Convolution and matrix snapshot construction](src/models/delta-net-base.cpp): `build_conv_state()` and `build_recurrent_attn()`.
- [CUDA GDN update and snapshot writes](ggml/src/ggml-cuda/gated_delta_net.cu).
- [Host snapshots and publication staging](src/llama-memory-recurrent-spill.cpp).
- [Workload quoting, arena binding and group lifecycle](src/llama-context.cpp).

Two kinds of sharing are useful:

1. Convolution states are overlapping windows of one input timeline. Store the timeline once and identify each window by its position.
2. GDN matrix states change densely, but their transitions are described by compact factors. Store a checkpoint and ordered update records instead of a complete matrix for each prefix.

The CUDA GDN update has the form `S_next = alpha * S_previous + outer_product(k, u)`, in the storage orientation used by the implementation. The correction `u` is computed from the previous state during verification. Capture that computed correction, the normalized key and the evaluated decay; restoration need not rerun the model projections or recompute the correction.

[ReplaySSM](https://arxiv.org/html/2610.13778v1) and [DeltaLog](https://arxiv.org/html/2608.15533v1) investigate checkpoint/log representations. Their numerical and performance results are supporting research, not qualification of this branch. Their reordered or mixed-precision algorithms do not establish bitwise equivalence to the retained kernels.

## Illustrative storage budget

For the inspected Qwen3.8-27B geometry, across all 48 recurrent layers:

| Item, per selected sequence | FP32 payload |
| --- | ---: |
| One dense recurrent matrix image | 144 MiB |
| One convolution input frame | 1.875 MiB |
| Current three-frame convolution history | 5.625 MiB |
| One compact GDN update record | Approximately 1.509 MiB |

The log estimate retains the original grouped keys rather than expanding them across value heads: 16 key heads, 48 value heads, width 128, and one decay scalar per value head. These values are examples for accounting; implementation must derive requirements from the operator's actual layout and precision.

A conservative candidate retains two dense matrix images, logs for every verification input row, and a convolution timeline containing the pre-verification history plus those rows:

| Configuration, per selected sequence | Current native recurrent bank | Candidate representation payload |
| --- | ---: | ---: |
| MTP3, four verification rows including the anchor | 598.50 MiB | Approximately 307.16 MiB |
| DFlash7, eight verification rows including the anchor | 1,197.00 MiB | Approximately 320.70 MiB |

Four selected MTP3 sequences would therefore have approximately 1,228.64 MiB of representation payload instead of 2,394 MiB. This does not establish admission of four sequences in a 2,096 MiB arena: KV, graph workspace, descriptors, alignment and other live resources still need to fit.

These are design estimates, not measured GPU peaks or promised throughput. The estimates exclude additional graph temporaries and any staging required by the final implementation. A design that retains old full-snapshot operator outputs can keep the old peak even after reducing the persistent bank.

In multi-slot mode, capacity follows the activated selection, including sparse logical IDs; it must not reserve a full bank for every configured slot. Saved states of unselected requests remain in host memory. Reducing an internal bank frees capacity inside the existing arena; the arena's total device allocation remains fixed unless it is explicitly refitted. Account separately for allocations outside the arena, including the current serial execution path.

## Representation contract

Each state component advertises supported representations and their requirements. The common allocator selects among those plans without implementing model algebra.

| Component | Candidate representation | Required fallback |
| --- | --- | --- |
| Convolution or token-history window | Shared input timeline and prefix/window descriptors | Existing copied history |
| GDN matrix state | Dense checkpoint, working image and FP32 update log | Existing native or host snapshots |
| Other recurrent operators | Their own supported transition/input log | Existing dense representation |
| Append-only KV | Retain common history and discard provisional suffix | Existing KV recovery |
| Bounded or sliding-window KV | Version or preserve overwritten data | Existing qualified recovery |
| Opaque or unsupported state | Existing snapshot representation | Clear capacity error if it cannot be admitted |

KDA, Mamba-family and RWKV operators may support compact logs, but require separate proofs for their decay structure, update/readout order and auxiliary state. Gemma 4 has no Qwen-style recurrent matrix bank; its KV recovery remains independently required.

For every candidate plan, report:

- Persistent device and host bytes, with resource identities and generations.
- Peak workspace, staging, alignment and overlapping lifetimes.
- Supported verification widths and accepted prefixes, including the anchor convention.
- Data types, grouped-head layout, backend capabilities and arithmetic contract.
- Capture, replay, commit, transfer and synchronization costs.
- Export, parking, cancellation and failed-operation recovery requirements.

Reuse the existing requirements, resource owner, arena, lease, completion and publication mechanisms. A compact state is logically one state version even when it references several physical resources. Different requests keep independent checkpoints and logs.

## Conservative transaction design

Use two dense device images initially: committed checkpoint A and uncommitted working image B.

1. Begin verification with A unchanged. Retain the required convolution history and all affected KV/draft recovery resources.
2. Run verification using the retained arithmetic. Capture FP32 evaluated decay, grouped keys and computed corrections. Write only the final working matrix to B rather than every intermediate matrix.
3. Once acceptance is known, use B directly for full acceptance. For partial acceptance, reconstruct the required prefix from A and its ordered log into B. A transaction abort keeps A.
4. Publish the accepted prefix coherently across target matrix state, convolution history, target KV and draft state only after the required GPU work and transfers complete.
5. Swap committed/working roles and retire rejected records after dependent readers finish. Reuse resources only after their completion conditions are satisfied.

Define accepted-prefix indexing explicitly. Zero accepted draft proposals can still commit a verification anchor; aborting the transaction restores the pre-verification state. Tests must distinguish these cases.

Capture logs before graph workspace can be reused. References into transient tensors are not sufficient unless the plan retains their leases for the entire required lifetime. Preserve grouped-key sharing in storage without allowing duplicate or racing GPU writers.

### Reuse of restoration workspace

Treat prefix restoration as an explicit phase operation with an existing owner-issued workspace lease. Reuse the same bounded replay tile, descriptors and convolution concatenation space across layers when their lifetimes do not overlap. Release them only after replay and its consumers complete; a different phase must not overwrite pending restoration.

The first correct implementation may retain both dense images persistently. Evaluate borrowing a working/materialization image from verification or draft workspace only when the plan proves that the committed checkpoint survives failure and that the accepted state can remain valid after the workspace is returned. Promotion, relocation or a required commit copy must be explicit and charged. A borrowed address cannot become long-lived state merely by retaining a raw pointer.

Compare persistent and borrowed candidates using actual high-water marks and total latency. Workspace reuse must not create an unquoted simultaneous allocation, an extra arena, a cross-slot alias, or silent driver fallback.

Keep existing external state serialization usable through materialization of the requested committed state. Any new internal parked representation needs explicit format/version, identity and lifetime handling. Do not silently change public checkpoint interpretation.

## Numerical contract

The initial provider keeps FP32 checkpoints, factors and accumulation. Preserve the retained update order and verify FMA/contraction behavior; an algebraically equivalent reassociated sum is not automatically numerically equivalent.

Avoid inverse-update restoration: decay, near-singular updates and floating-point rounding can destroy information needed to recover a prior state. Retain a checkpoint and reconstruct forward.

Test raw state bytes where the retained backend supports an exact arithmetic contract, together with full-vocabulary logits and accepted-prefix behavior. Record any existing reference tolerance precisely. A tolerance-qualified provider must not be advertised as native-exact or enabled under a stricter contract without an explicit decision.

CPU replay is a useful first reference, but does not qualify CUDA replay, asynchronous ownership, or model-level numerical behavior. Keep independent dense native and dense host paths as references.

## TDD delivery stages

Each stage starts with a failing regression or capability/accounting test, records the failure, implements the smallest change, and passes its applicable correctness and non-regression gates. No stage is complete solely because a benchmark finishes.

### P0. Freeze references and audit memory

- [x] Record source/build IDs, model and draft checksums, device/driver/toolkit, flags and workload manifests.
- [x] Measure dense native and dense RAM rollback with identical prompts, outputs, context budgets and explicit placement.
- [x] Separate matrix, convolution, any additional state, graph outputs, staging, external allocations and arena-owned allocations.
- [x] Inspect actual verification widths, accepted-prefix mapping, direct-cache fusion and short-batch behavior.
- [x] Establish target-only and Gemma controls that must acquire no compact recurrent resources.

Exit: a reproducible allocation/transfer ledger and references for every currently supported execution path affected by the change.

### P1. Add capability and requirement descriptions

- [x] Describe existing representations through the provider contract without changing defaults.
- [x] Test unsupported capabilities, invalid dimensions, integer bounds, alignment, tight budgets and peak overlap.
- [x] Test selected-count sizing separately from configured capacity, including sparse/high logical IDs.
- [x] Preserve independent requirements for recurrent, append-only, bounded and auxiliary state.

Exit: existing paths pass unchanged, and common admission can compare complete candidate requirements without model-name branches.

### P2. Qualify shared convolution history

- [x] Reconstruct every supported history window from a retained timeline.
- [x] Test short blocks, repeated windows, timeline wrap/rebase, zero/full acceptance and transaction abort.
- [x] Cover slot parking/reselection/reuse and any operator-specific auxiliary history.
- [x] Measure retained timeline and temporary concatenation bytes; retain bitwise reference comparisons.

Exit: an exact, independently selectable history provider with measured requirements. Its modest standalone savings must not be reported as matrix-state savings.

### P3. Capture and replay GDN updates in shadow mode

- [x] Keep dense snapshots as the reference while capturing evaluated decay, grouped keys and corrections.
- [x] Build bounded CPU and CUDA replay fixtures before integrating replay into production commits.
- [x] Compare every prefix, including transaction abort, zero accepted proposals, full acceptance and short verification blocks.
- [x] Check state tensors, convolution state, logits and sampling/acceptance contracts.
- [x] Cover one through four selected sequences, configured capacity greater than selection, sparse IDs and nonuniform acceptance.
- [x] Verify writer ownership, event ordering, stale-generation rejection and failure cleanup.

Exit: replay is qualified under a stated arithmetic contract; capture overhead and log layouts are measured. Shadow capture is not a production performance result.

### P4. Integrate compact storage and transaction commit

- [x] Change the operator/fusion output contract so full intermediate matrices are no longer allocated or written for the compact plan.
- [x] Implement checkpoint/working-image ownership and coherent commit across every participating component.
- [x] Exercise partial-copy failure, failed compute, cancellation, failed admission, phase handoff and slot reuse.
- [x] Update workload quotes and test predicted versus observed high-water marks, including graph temporaries.
- [x] Preserve dense native/RAM fallback and deterministic explicit policy selection.
- [x] Propagate and record the selected representation/placement through arena fitting, all probes, generated commands and sweeps.

Exit: compact execution passes correctness and lifecycle gates, and actual allocation peaks demonstrate that the dense snapshot planes were removed.

### P5. Add budget-aware placement and bounded adaptation

- [x] Compare the supported dense and compact plans using total predicted decode cost within the hard device budget and reserve.
- [x] Account for extra KV streaming caused by state residency, reconstruction cost, transfers and acceptance-dependent behavior.
- [x] Support partial residency only for independently restorable regions with explicit dependencies.
- [x] Change plans only at qualified safe request/phase boundaries after draining affected work.
- [x] Add hysteresis and bounded measurements so the policy does not oscillate or profile indefinitely.
- [x] Retain reproducible user overrides and clear unsupported/admission errors.

Exit: deterministic budget tests pass and measured plan choices improve total execution cost over the applicable references.

### P6. Product qualification

- [ ] Qualify every supported model/mode/KV-format/slot/modality combination affected by the integration; list unsupported combinations separately.
- [ ] Preserve single-slot behavior and target-only/Gemma controls without extra recurrent allocations or material regressions.
- [ ] Run applicable host sanitizers and CUDA memory/order checks with the established graphs, PDL and error-reporting settings.
- [ ] Run matched performance/resource comparisons below and publish the raw configuration and evidence.
- [ ] Document qualified arithmetic contracts, fallbacks, policy controls, measured peaks and hardware limits.

Use the compositional numerical contracts in [MS17](MULTI_SLOT_PHASE_ARENA_ROADMAP.md) where available. Do not relabel existing same-fork comparisons as independent stock equivalence or make an unfinished tooling stage a substitute for bounded provider tests.

### P7. Optimize performance and close out

- [ ] Profile the qualified P6 implementation to separate capture, replay, commit, transfer, graph/launch, and KV-streaming costs.
- [ ] Reuse owner-issued workspace for bounded restoration scratch; compare safe borrowed-image plans with the persistent-image reference.
- [ ] Fuse prefix steps within a replay pass so intermediate dense matrices are not written between steps.
- [ ] Optimize full-acceptance and abort paths, grouped-key reuse, descriptor preparation and launch count without weakening transaction semantics.
- [ ] Evaluate cross-layer or selected-sequence batching and safe overlap only where profiling identifies a material cost.
- [ ] Calibrate placement using measured costs, preserve bounded adaptation and hysteresis, and retain explicit reproducible overrides.
- [ ] Rerun applicable P6 correctness, failure/lifetime and non-regression gates after every retained optimization.
- [ ] Publish matched before/after timings, real memory peaks, accepted-prefix distributions, KV-transfer changes, and the final qualification ledger.

Exit: retain only measured end-to-end improvements with unchanged numerical and lifecycle contracts. Target-only and unaffected models remain non-regression controls. Record rejected optimization trials and remaining hardware limits; an isolated replay-kernel speedup cannot close P7 if total decode regresses.

## Qualification matrix and measurements

Cover supported Qwen target-only, MTP1-3, DFlash1-7 and DFlash+K4V profiles, together with supported Gemma target-only/MTP controls. Exercise supported Q8V8, Q8V4 and Q4V4 product configurations and preserve the broader registered-format coverage wherever a touched operator requires it. Text and still-image paths remain mandatory where supported.

Use one through four genuinely selected sequences where the candidate fits, plus larger configured capacities and admitted subsets. Include roughly equal histories, asymmetric histories, prefix reuse, selected-group changes and different accepted lengths per sequence. A configured slot count does not prove that count executed together.

Compare dense native, dense RAM and each qualified compact placement. Keep model/draft files, precision, token histories, output lengths, context semantics, sampling and reserve identical for matched controls. Use the pinned SWE cohorts A-D where applicable and retain actual request/phase grouping evidence.

Include all-resident, near-streaming-threshold and heavy-streaming contexts, a matched fixed arena including the 2,096 MiB control, and separately refitted realistic-card configurations. A larger GPU constrained to that arena is not a simulation of a 5070 Ti's bandwidth or total footprint. Coordinate any production interruption separately before GPU experiments.

Report:

- Total decode latency and aggregate throughput, individual request latency, prefill and phase/commit overhead.
- Actual selected counts, context, draft generation/acceptance and verification widths.
- Persistent and peak device/host bytes, including external allocations and protected reserve.
- Dense-state reads/writes, compact-log bytes, publication/restoration traffic and synchronization cost.
- KV residency, KV transfer bytes, streaming activity and reconstruction cost.
- Numerical results, failed/cancelled cases, fallback selections and qualification limits.

Treat fixed-arena and independently refitted comparisons as different questions. A rollback-only speedup or smaller logical payload is insufficient if total decode regresses or the measured peak exceeds admission. Do not promise a specific token speed or transfer a large-GPU paper result to the 5070 Ti.

## Completion conditions

Close this work only when the selected representations are correct under their declared contracts, measured peaks match admission, lifecycle/failure tests pass, and the supported configuration matrix has explicit results or justified hardware qualification limits. Retain conservative fallback for unsupported operators and no-state paths.

The implementation must demonstrate recovered usable capacity or reduced transfer cost, preserve existing supported behavior, and show its end-to-end tradeoff against both native and RAM references. More aggressive output-only evaluation, reassociated materialization, lower-precision logs and longer deferred-merge windows require their own subsequent numerical and performance qualification.
