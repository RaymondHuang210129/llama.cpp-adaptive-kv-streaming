# Multi-slot adaptive KV streaming and phase-arena roadmap

Status: implementation in progress, 2026-11-09. MS1 (CPU planner), MS2 (demand-backed host storage) and MS3 (independent histories/native metadata) are complete as independent components; runtime multi-slot admission remains disabled. MS4-MS14 are pending. No new server configuration is qualified.

Inspected baseline: `f6ca3e8a172473578b909a4691e616bb2234fb54`, on `feature/ci-workflow-migration`. Recheck the implementation base and local changes before starting work on a dedicated feature branch. Historical measurements retain their original source, binaries, models and configuration; they are not new multi-slot evidence.

Roadmap placement base after pulling the main KV-stream V2 branch: `2306b0c1d9252791149714ea9820218ef2599a8a`, on `feature/adaptive-kv-stream-v2`. MS0 must freeze regression controls from the reviewed implementation base, rather than assuming the earlier inspected commit or an old production binary is equivalent.

Implementation branch: `feature/adaptive-kv-multi-slot`, starting at the placement base above. The user authorized implementation and one title-only commit per completed stage on 2026-11-09. Existing unrelated local files and production configuration remain untouched.

## Objective

Support two concurrent request slots using one shared phase arena on the existing CUDA Qwen and Gemma streaming paths. Batch compatible selected slots in the same attention launch, with separate histories and up to three spans per sequence per submitted wave. Design identities, descriptors and scheduling for future N-slot support without advertising unqualified slot counts.

The completed feature includes all currently supported combinations: target-only execution, supported Qwen/Gemma MTP, Qwen DFlash, Qwen DFlash plus K4V, and their existing supported still-image paths. This does not introduce Gemma DFlash, K15, new model pairings, audio/video, multi-GPU or previously unsupported LoRA combinations.

Preserve native arithmetic, conversion, masks, accumulator precision, checkpoint boundaries and reduction order. Preserve single-slot functionality and performance. Reuse the existing memory owner, leases, phase transitions, speculative dispatcher, MTMD and rollback interfaces. There is one GPU arena, not one arena per slot.

**Non-regression is a mandatory stage and release condition.** Every stage exit below also requires the applicable [non-regression gate](#mandatory-non-regression-gate). A successful two-slot test or an aggregate throughput gain cannot compensate for breaking an existing mode, reducing single-slot performance or weakening a safety/admission check.

## 1. Settled user decisions

| Topic | Contract |
| --- | --- |
| Initial concurrency | Two request slots; internal structures support N rather than hard-coding two. |
| Context limit | In the new streaming multi-slot mode, `--ctx-size C` is the per-slot limit and the aggregate logical-context budget of a selected execution group. |
| Large combined histories | If admitted slots cannot fit together under that budget, execute smaller selections sequentially. Preserve every paused history in RAM. |
| Full attention KV | Authoritative histories are in host memory. VRAM holds selected slots' resident pages and streamed ring waves. |
| Other live state | Only selected slots' current recurrent/SWA and required speculative state occupy VRAM. Park outgoing state in RAM. |
| Physical allocation | Assign exclusive whole-page ranges to selected slots. No mixed-slot physical pages or borrowing another slot's allocation during a selection. |
| Proportions | History allocation is proportional to context length, with equal logical append allowance for selected slots. |
| Growth allowance | An allowance is logical generation headroom, not dedicated VRAM for that many tokens. |
| Host addresses | Reserve maximum host virtual ranges during startup/preflight in the actual server process. Grow backing without moving existing history addresses. |
| Phases | Vision, prefill and decode execute separately. Do not retain a mixed prefill/decode workspace profile to serve incoming requests. |
| Request groups | Admit currently queued requests up to the slot limit, freeze membership, finish their required vision, then their prefill, then their decode. |
| Reselection | Trigger on allowance exhaustion, lack of space for the next operation, completion, cancellation or an individual context boundary. |
| Waiting policy | Return paused requests to eligibility at the next selection. No periodic rotation, waiting-time override or forced turn. Prefer parallelism; a long context may wait. |
| Generation | No new finite output cap. Existing EOS, stop strings, requested output limits, cancellation and individual context limits still apply. Context shifting remains outside the supported streaming profile. |
| Streaming output | Emit accepted tokens incrementally. Pausing is not completion; keep the same response open and resume without duplicate output. |
| Scope | All existing supported acceleration and still-image combinations must pass before the complete feature is declared supported. |

These decisions replace earlier suggestions for mixed-phase scheduling, equal resident quotas, borrowing pages within a selection, or periodic fairness rotation.

## 2. Baseline mechanisms and limitations

| Area | Inspected source and implication |
| --- | --- |
| Admission | [llama-context.cpp](src/llama-context.cpp) requires one sequence for streamed targets and attached MTP/DFlash contexts. Removing these checks alone is unsafe. |
| Stream entry | [llama-kv-cache.cpp](src/llama-kv-cache.cpp), `kv_stream_begin`, requires one sequence, ID zero, one contiguous append and one frontier. |
| Recurrent spill | [llama-memory-recurrent.cpp](src/llama-memory-recurrent.cpp) rejects multi-sequence spill and has `rs_idx[0]` integration. The [spill bank](src/llama-memory-recurrent-spill.h) already carries cache, sequence and generation identities. |
| Host backing | [llama-kv-stream-host.cpp](src/llama-kv-stream-host.cpp) allocates maximum storage and clears the whole allocation. K and V are separate planes with fixed per-layer strides. |
| Pinning | [ggml-cuda.cu](ggml/src/ggml-cuda/ggml-cuda.cu) implements strict mapped/pinned host buffers and whole-range registration. Sparse commitment is not currently represented by that contract. |
| GPU placement | [llama-kv-stream-policy.h](src/llama-kv-stream-policy.h) describes resident layers and a ring inside a bounded pool. Existing multi-cache identities are useful, but are not multi-slot qualification. |
| Transfer | [kv-stream-copy.cu](ggml/src/ggml-cuda/kv-stream-copy.cu) copies from a retained host source into separately specified GPU offsets. Host mapping is not a permanent placement in VRAM. |
| Spans | [kv-stream-span.h](ggml/src/ggml-cuda/kv-stream-span.h) permits at most three ordered, nonempty spans for one sequence interval. |
| Attention | [kv-stream-partial.cu](ggml/src/ggml-cuda/kv-stream-partial.cu) and its dispatch/plan helpers currently validate serial geometry. Native attention has sequence-aware layouts; the streaming adapters must gain equivalent routing. |
| Ownership | [llama-context-memory.h](src/llama-context-memory.h) and its implementation own one serial phase lifecycle and borrowed target/drafter workspace. Retain their transactional and completion contracts. |
| Server scheduling | [server-context.cpp](tools/server/server-context.cpp) currently infers streaming decode from a single generating slot. Replace this with explicit scheduled-batch metadata. |
| Multimodal accounting | [server-common.h](tools/server/server-common.h), `server_tokens`, distinguishes token/KV cells from M-RoPE positions. These counts cannot be substituted for one another. |
| Speculation | [common/speculative.h](common/speculative.h) already has per-sequence parameters, state hooks and an `n_max` override. Shared-arena sources, hidden features and rollback need complete slot isolation. |

Related references: [memory infrastructure](DEVICE_MEMORY_INFRASTRUCTURE.md), [memory consumers](DEVICE_MEMORY_CONSUMERS_ROADMAP.md), [Gemma streaming](GEMMA4_ADAPTIVE_KV_ROADMAP.md), [MTP](MTP_ADAPTIVE_KV_ROADMAP.md), [DFlash/K4V](DFLASH2_K4V_ADAPTIVE_KV_ROADMAP.md), [vision integration](DFLASH2_VISION_PHASE_ROADMAP.md), [vision capacity](docs/multimodal/vision-capacity.md).

## 3. Quantities that must remain independent

Use distinct fields and diagnostics for these quantities. Do not overload one integer or pointer to stand for several of them.

| Symbol | Meaning |
| --- | --- |
| `C` | Effective per-slot context limit and selected-group logical-context budget. Derive unset values through existing native model limits; preserve model/position bounds. |
| `N` | Configured request-slot limit, initially at most two in the new mode. |
| `A` | Frozen admitted request group, containing at most N live requests. |
| `S` | Currently selected subset of A for one phase's execution. |
| `L_i` | Slot i's occupied logical target-context capacity. Include accepted output positions that are pending their next target invocation; keep committed KV and pending-token counters separately. |
| `H_i` | Remaining logical growth allowance assigned to slot i for this selection. |
| `W_i` | Projected selected-group extent `L_i + H_i`. |
| `B_i` | Optional packed logical offset of slot i inside this selection. It is neither the slot's host base nor its RoPE position. |
| `P` | Actual allocatable GPU KV pages after persistent selected state, workspace, conversion, publication staging and alignment are quoted. This is not C. |
| `Q_i` | Slot i's exclusive physical page quota for the selection, including its resident/ring allocation. |
| Host identity | Slot/storage ID plus reuse generation, cache/layer identity, logical frontier and content generation. |
| Device binding | Current physical page IDs, leases, layout revision and arena generation. |

`L_i` uses native target context/token-cell accounting, including image embedding cells. Track native decoder positions and M-RoPE coordinates independently. Never add `B_i` to positional encodings. Audit the pending sampled anchor, correction token and speculative provisional rows before defining exact accounting helpers.

Two histories may jointly occupy up to `2*C` logical capacity in RAM while no selected execution group exceeds C. The GPU arena remains bounded by bytes, and includes selected recurrent/SWA state in addition to KV staging.

Record requested/effective limits and native allocation alignment explicitly. Physical padding does not create extra logical growth allowance beyond the accepted per-slot/group limit.

### CLI and API compatibility

- Enable the new behavior explicitly with the existing shared-arena streaming configuration, unified KV and `--parallel 2`. Preserve the serial default and the existing `--parallel 1` path.
- Keep stock/non-streaming context semantics unchanged. Document the streaming-specific meaning of C clearly.
- Separate per-slot capacity, total logical metadata capacity and execution-group budget internally. Do not simply double `n_ctx` and hope all consumers acquire the intended semantics.
- Audit `llama_n_ctx`, `llama_n_ctx_seq`, `/props`, `/slots`, truncation and model-limit checks together. Report all three capacities explicitly where needed rather than changing an existing public result silently.
- Reject conflicting explicit `--kv-unified-per-slot` and per-slot C settings with an actionable error; an equal value is consistent. Preserve native defaults outside this mode.
- Reject unsupported slot counts and combinations before loading/mutating request state. Do not silently disable speculation, change KV formats or execute a claimed joint batch as a per-slot fallback.

## 4. Selection and logical append allowance

### Selection objective

At a selection event, consider eligible requests from the admitted group, not new requests waiting for another group. Prefer the greatest number of compatible requests that can make progress under C and the measured arena contract. Prefer smaller combined history among equal-cardinality choices, with a deterministic slot-order tie break. This implements the agreed preference to park a large context rather than reduce the number of other active slots.

There is no age-based override or periodic rotation. Reconsider parked requests at later selection events. A request rejected by one selection remains live and must not receive a final response merely because it is parked.

### Equal headroom

For a candidate selection containing k slots:

```text
R = C - sum(L_i)
h = floor(R / k)
r = R mod k
H_i = h, with one extra token for r slots in deterministic order
W_i = L_i + H_i
B_0 = 0
B_(i+1) = B_i + W_i
```

The candidate is executable only if every chosen slot has sufficient allowance for a legal next operation. Shorten speculative proposals through existing per-sequence limits near a boundary. If even minimum progress cannot fit, select fewer slots. Never repeatedly reselect the same zero-headroom group.

Logical allowance is measured in context units; physical allocation is in complete pages. Page rounding must not lend a physical page to another slot, change RoPE positions or silently redefine C. Padding consumes physical capacity and is included in byte/page admission.

Example, with C = 128Ki and histories 90Ki and 30Ki:

| Slot | Existing logical history | Logical append allowance | Packed selected interval |
| --- | --- | --- | --- |
| 0 | 90Ki | 4Ki | `[0, 94Ki)` |
| 1 | 30Ki | 4Ki | `[94Ki, 128Ki)` |

If slot 0 consumes 4Ki while slot 1 consumes only 1Ki, reselection occurs before slot 0 exceeds its allowance even though the combined history is still below C. The new selection can choose both again and divide the remaining allowance. No tokens are retracted.

If both consume 4Ki, the group budget is full. Select a smaller subset for further generation; neither slot has yet reached its individual C. The physical ring may have been full and recycling safely throughout this interval.

### Operation-level bounds

- Check the maximum capacity required by the next native target operation, including pending accepted output and provisional verification rows, before launching it.
- Do not infer total response length or wait until the entire response finishes to enforce allowance.
- Rejected speculative rows do not become accepted history. Their temporary writes remain within the submitted reservation and are rolled back through native rules.
- Do not stream unverified draft tokens. Preserve the next sampled/pending token and output cursor across a pause.
- Use native context-end handling when no further legal target step fits. Preserve existing stop/finish reasons and avoid off-by-one loss or duplication.

## 5. Physical pages and the three-span contract

### Ownership and quotas

Allocate physical quotas only after all state and workspace requirements are known. For each homogeneous KV geometry group, reserve the existing required working minimum for each chosen slot. Distribute the elastic remainder proportionally to `W_i`; this represents history-proportional weight plus equal logical growth weight. Use checked integer arithmetic and deterministic remainder assignment. Cap allocations at useful demand and distribute any still-unallocated pages before publishing ownership.

This is a derived implementation rule to validate against the user's intent; it does not reserve dedicated GPU storage for all H_i appended tokens. Native ring minima, K/V row sizes, alignment and mandatory per-layer floors still apply.

Once published:

- Each physical page belongs to exactly one slot/cache generation and one declared role.
- No page contains payload from two slots. K and V planes retain their native layouts.
- No slot borrows another chosen slot's unused quota during that selection.
- No unchosen slot retains required GPU KV or current-state allocations after parking completes.
- Existing adaptive feedback may inform the next selection. It cannot move the boundary between slots behind the scheduler's back.
- Any internal resident/ring adjustment within a slot's own allocation still requires the existing checked layout and completion protocol.

Host virtual addresses are independent of `B_i` and physical GPU offsets. Changing selection must not relocate authoritative history. A fixed registered host source can copy into a different GPU destination in a later selection.

### Spans and waves

Each sequence has its own ordered description of a resident segment, a ring segment before wrap and a ring segment after wrap. Empty spans are omitted. One metadata allocation may contain every selected sequence's descriptors.

Prove that every submitted sequence interval is representable with at most three spans. A separate persistent appended-tail pocket, arbitrary fragmentation or a misplaced partition boundary can otherwise create a fourth segment. Avoid that by preserving the intended layout, coalescing genuinely adjacent ranges or using existing slot-owned staging. Split waves only at boundaries legal for the native resume/checkpoint plan. Do not increase the span count, change accumulation order, skip a geometry or serialize the selected slots silently to make a test pass.

Maintain independent canonical source cursors, reset/finish state and query-output slices. A source cursor ending is not permission to drop masked work from a still-active native unified operator: preserve its complete native checkpoint/reduction plan before finalizing those queries. Separate native microbatches can finish independently; later work must not overwrite their completed outputs.

In-flight copies and all readers retain the source and destination leases. Reassignment requires their completion, not merely CPU submission or one slot finishing its own head. Preserve the qualified two-wave transfer/conversion behavior within each exclusive allocation.

## 6. Stable, demand-backed host history

### Required behavior

Reserve a maximum virtual range per slot/cache covering all of its layer K and V planes at fixed strides during actual server initialization, before consumers capture addresses. Commit, initialize and register growth chunks on demand. Growing a history extends the live prefix of each relevant plane; committed coverage is therefore a set of ranges, not necessarily one contiguous prefix of the entire allocation. Existing addresses and per-layer strides remain stable until that storage generation is destroyed.

This is stable contiguous virtual addressing, not a requirement for physically contiguous RAM. A separate finder process validates sizes and behavior; it cannot choose pointer values that a later server process must reuse. Restarts and file restores must rebuild bindings at newly assigned bases.

Reuse the existing backend ownership interfaces. Extend their host-range contract where necessary; do not mislabel a partially committed reservation as a wholly readable, pinned buffer. The current model-file `llama_mmap` is read-only file mapping and is not a drop-in mutable KV allocator.

### Investigation requirements

- Linux: reserve an owned range, protect uncommitted regions, and grow only within it. Do not use `MAP_FIXED` to replace mappings not owned by this allocation.
- Windows: investigate corresponding reserve/commit behavior and allocation granularity. Preserve the current WDDM mapped-memory flags and existing Windows build behavior.
- Account separately for reserved virtual bytes, committed/RSS bytes, CUDA-registered bytes, live KV bytes and checkpoint/parked-state bytes.
- Register committed chunks rather than the complete maximum range. Validate device aliases without assuming CPU and CUDA pointer equality or device-alias contiguity across registrations.
- Initialize only newly committed bytes and required native padding. Do not `memset` the entire reserved capacity on create/reset, or touch holes through generic clear/view/copy/state APIs.
- Amortize commitment/registration in measured aligned chunks. Do not register/unregister history whenever GPU selection changes.
- Before any writer, DMA or kernel accesses a range, establish and retain its committed/registered coverage through completion.
- Publish growth metadata only after commitment/registration succeeds. Partial failures must leave existing history and registration ownership intact.
- A virtual reservation is not a guarantee that future physical RAM will be available. Scope admission guarantees to successfully acquired resources; do not claim protection from arbitrary system-wide OOM.

The native mapped/pinned backing is the comparison control. A pageable-history plus pinned-staging design would add copies and is not an automatic fallback; consider it only with an explicit design/performance review if chunk registration proves unsuitable.

### Host-memory accounting

Compute exact bytes from each model's real layer geometry and quantized row sizes. Include target KV, Qwen auxiliary MTP KV when enabled, parked recurrent/SWA state, rollback snapshots, draft features/caches, image embeddings and retained checkpoints.

For illustration, the inspected Qwen target's 16 full-attention layers with D256, four KV heads and Q8_0/Q4_0 consume 26,624 bytes per history token: about 3.25 GiB at 128Ki or 6.5 GiB at 256Ki per slot, excluding other state. The observed serial checkpoint size was about 166 MiB; copying its configured count to every slot can add several GiB. These are estimates from the existing profile, not new multi-slot measurements.

Measure before choosing a new hard host-memory cap. The existing `--cache-ram` controls retained prompt-cache entries, not all active streaming KV. Active/parked admitted requests cannot be evicted as expendable completed conversations. Do not introduce a fixed 32 GiB limit from an old test setting.

## 7. Slot state and transactional selection changes

Maintain distinct logical slot IDs, reusable physical state-bank indices and generations. Do not translate every selected request into semantic sequence zero. Reuse existing sequence-aware native recurrent metadata and spill identities where possible.

A parked-state record must preserve everything needed to resume the same invocation:

- Target full-history identity/frontier and local/SWA cells, positions, validity and necessary bytes.
- Current recurrent state and outstanding rollback/publication bookkeeping.
- Required per-sequence rollback snapshots and their committed/provisional identities.
- Auxiliary/shared-source KV metadata, pins and source generation relationships.
- MTP/DFlash hidden features, feature-layer mapping, convolution/selector/cache state and pending inputs.
- K4V's applicable history namespace, generation and continuation/acceptance bookkeeping.
- Sampler/RNG state, grammar/reasoning state, remaining output budget, accepted decisions and stop-string buffers.
- The pending sampled anchor/correction token, committed output cursor and public response identity.

Keep one GPU submission/transition owner for the arena. HTTP workers enqueue requests and cancellation; multiple slots do not authorize concurrent graph mutation by those workers. Validate every selected slot's sources, reservations and output regions before submitting any mutating operation for the batch.

At reselection:

1. Stop submitting work using the old bindings at a legal completed speculative/target round.
2. Complete pending publication, copies and reads required for a resumable state.
3. Validate incoming host state and all byte requirements without overwriting live GPU state.
4. Save outgoing resumable state and retire its physical leases/graph references. Completed/cancelled requests follow existing cache/cleanup policy; do not force an otherwise unnecessary state copy just to release them.
5. Restore incoming slots into the shared selected-state regions, allocate their exclusive pages, and prepare graph/reader bindings.
6. Publish the new selection only after successful activation. Retained selected members can keep valid state instead of needlessly round-tripping through RAM.

Do not require two complete selected-state banks simultaneously in VRAM. A failed activation must either restore the old selection from its retained host state or leave the owner safely quiescent with an explicit error. No output is replayed as newly generated during recovery.

Parking changes physical binding generations; it does not by itself change semantic history. Rewriting/truncating/replacing content does change the relevant semantic generation. A failure or cancellation in one slot must not invalidate another slot's history or clear the shared owner indiscriminately.

## 8. Phase and server scheduling policy

### Admitted groups and phases

Form an admitted group from requests already queued, up to N, without an artificial wait to fill it. Freeze membership. Requests arriving later wait for the next group, even if one member finishes early.

| Transition | Required condition |
| --- | --- |
| Idle to vision | The admitted group needs image encoding. |
| Idle to prefill | No encoding is required. |
| Vision to prefill | All required embeddings for the group are ready; encoder loans and work have completed. |
| Prefill to decode | Every remaining member has completed prefill. A completed prefill can wait with state parked while other prefills finish. |
| Decode to next group | All admitted members, including temporarily parked decoders, completed, cancelled or reached their native context boundary. |
| Any phase to idle | No admitted work remains and no next group is queued. |

Only one global phase executes at a time. Within prefill, select chunks/subsets with the same logical and physical bounds; combined histories need not fit all at once merely because they are admitted. Within decode, change selected subsets without admitting new prefills or vision work.

Reselection is not a global phase transition. Retain the decode workspace layout when possible; state/page rebinding may still require a checked transition. Do not retain the earlier proposed mixed-serving workspace profile, introduce periodic prefill service or impose a new generation quantum.

### Progress and stream lifecycle

- Stream accepted output immediately after native acceptance/stop handling makes it final.
- Allowance exhaustion parks the request, not its already emitted output. Resume the same logical HTTP response without duplicate tokens or a second final response.
- Send appropriate existing SSE heartbeat/ping traffic while paused. Client disconnect/cancellation remains observable while no GPU work is scheduled for that slot.
- Heartbeats address supported streaming idle-timeout behavior, not arbitrary client total deadlines or non-streaming response deadlines. Preserve configured timeouts and test actual client behavior; do not silently change production/client timeout settings.
- A paused slot is processing, not idle and available for reassignment. Administrative save/restore/erase must retain existing busy-slot semantics unless a separately qualified atomic pause operation is introduced.
- Do not replay RNG draws, accept decisions, usage counters or tool-call fragments on resume. Non-streaming requests complete once with the accumulated result.
- Model failure and recoverable slot validation errors are different from device faults. Isolate recoverable slot failures; do not promise other requests can continue on a poisoned CUDA context.
- Test existing response backpressure and cleanup. A response writer must not retain GPU leases after its computation no longer needs them; group computational completion is distinct from the last network byte being consumed.
- This intentionally permits high prefill waiting time and long-context waiting. There is no hidden fairness rotation or decode/prefill overlap.

## 9. Batched native attention and speculation

The shared-launch objective applies per compatible native attention microbatch/wave. A sequence-indexed descriptor carries its sources, live extent, query range, positions/mask, checkpoint plan, accumulator and output destination. Sequence count is runtime metadata; avoid a template Cartesian product over slot counts.

Avoid accidentally sizing/scanning GPU masks and working storage for all parked slots' maximum host reservations. Establish selected-group native working extents in the reference contract; quote any extent required by that contract rather than silently trimming it and changing the numerical plan.

| Configuration, both slots at maximum proposal length | Per-sequence target verification | Aggregate target rows |
| --- | --- | --- |
| Target-only | 1 | 2 |
| MTP3 | 4 | 8 |
| DFlash7 / K4V7 | 8 | 16 |

The last row is not K15 or a single sequence with 16 verification rows. Rollback depth remains per sequence. Draft prediction width, target verification width and image-feature injection width remain separate.

Native geometry clarification from the MS3 source audit and real-model probes: unified KV uses one K/V stream and sequence-isolating masks. Two four-query sequences produced an aggregate eight-query attention operation; two eight-query sequences produced sixteen query rows. Separate KV produced two four/eight-query streams instead. Actual Gemma Q3_K_M and Qwen IQ4_XS targets completed both layouts with all logits finite, Q8_0/Q4_0, a 1,024-cell ordinary context and microbatch capacity 16. With unequal two/four-token inputs, Gemma unified attention used one six-query operation; Qwen's recurrent path used native two- and four-query microbatches. Separate KV used a balanced two-sequence microbatch plus a single-sequence tail. Preserve these native partitions rather than forcing ragged work into an invented representation.

Preserve the user's unified-KV policy. Do not force a two-stream/TG4 tensor representation or confuse an aggregate TG8/TG16 operator width with a single sequence's proposal/rollback limit. Quote and qualify aggregate query widths separately from per-sequence maxima. The local `native-sequence-geometry-expanded-*` logs record ordinary native graph geometry, not multi-slot streaming qualification, numerical equivalence between different layouts, or throughput measurements. MS0's remaining baseline/accounting work is still required.

Actual widths can differ because of confidence, K4V hits/misses, remaining headroom and output limits. Audit the native `split_equal`/`split_simple` behavior and kernel-family selection before choosing a representation. Prefer native-compatible balanced microbatches and explicit ragged metadata where supported. Do not insert fake recurrent tokens or change native plans simply to advertise one launch. Incompatible native work may require separate microbatches; any extra per-slot serialization must be visible and justified, not hidden behind a successful multi-slot flag.

Retain the existing producer/reader roles and sequence-specific prefetch intent. A K4V hit must still maintain the neural drafter's required target features and genuine fallback path. Attribute generated/accepted counts to the actual supplying implementation and slot. Never use one slot's counters to satisfy another slot's proof.

## 10. Vision, caching and sizing

- Complete the admitted group's required image encoding in the vision phase, then retain embeddings and metadata in RAM for later prefill. No target decode overlaps the encoder.
- Preserve exact embeddings, projector/reference weights, image budgets, image order and native multimodal positions. Image token budgets are not pixel dimensions or M-RoPE positions.
- Park all required text/auxiliary state before borrowing arena regions for vision. Preserve the controlled operator workspace and cuBLAS ownership introduced by the existing vision work; no second arena or private GPU fallback.
- Retain current public image rejection rules and model/sidecar pairing. All currently supported still-image modes are release requirements, not permission to admit unsupported model/mode combinations.
- Existing completed-conversation caching and explicit slot files remain separate from live parked requests. Automatic SSD spill/prefix lookup is not part of this feature.
- Slot save files must store logical state and offsets, not process-specific raw addresses. Rebind checkpoint appendices, speculative state, images and source identities on restore, including after restart at different virtual bases.
- Preserve native file-format compatibility where supported. Version any necessary new metadata and reject incompatible/truncated state explicitly rather than interpreting old bytes under new layout assumptions.
- Extend the finder/preflight to the configured slot count, mode, C and all legal selected-state/graph shapes. A serial maximum arena is not multi-slot qualification.
- Report per-slot context, aggregate group budget, host reservation/commit/pin bytes, selected-state minima, arena phases, actual rejected upper neighbor, and the workload validated.
- Keep the user-facing mmproj ratio/resolution probe approach. Validate the largest admitted vision shape and relevant state transitions without requiring users to provide an example image.
- Prewarm required native variants/library resources before arena sizing where possible. Driver/executable allocations outside the application arena remain explicitly scoped; a short probe is not a guarantee for every long request.
- Include driver memory caused by host registration in multi-slot sizing. A CUDA-runtime probe observed 4 MiB less free VRAM after registering 2 GiB of 1 MiB host chunks; unregistering restored it. DMA-only default registration showed the same charge, so changing mapping flags is not a demonstrated remedy. The sizing probe must exercise the configured maximum canonical-history registration pressure as well as selected device-state/graph shapes. This is deliberate probe work, not permission to materialize maximum histories during normal lazy startup or to invent an unmeasured fixed reserve. Report the probe's peak host commitment and scope separately.

## 11. Implementation stages and exit gates

Every stage starts with a demonstrated failing contract, build or behavior, followed by the smallest appropriate change. Keep failures and accepted evidence. Existing tests should be extended before introducing a new test framework. No numeric/API relaxation, silent profile changes or unsupported-case omission.

Before editing a shared component, map its existing consumers to the regression controls below. Run the mapped gates before marking that stage complete; do not defer all regression discovery to MS13. Changes to shared headers/ABI require the relevant server and test targets to be rebuilt together. Record test inventories and expected skip reasons rather than treating a bare process exit as sufficient evidence.

### MS0 - Baseline and executable contracts

- [ ] Recheck source/branch, capture clean/dirty state and pin model, sidecar, projector, corpus, native controls and supported profiles.
- [ ] Freeze independent pre-multi-slot V2 binaries/libraries and establish fresh single-slot functional, numerical, resource and performance controls on the same profiles intended for candidate comparisons.
- [ ] Create the stage regression ledger, including applicable fork CI/build checks, exact test inventories, baseline results and any pre-existing failures. Classify existing failures explicitly; they do not authorize ignoring new ones.
- [ ] Record every serial guard and its dependent invariant, including sequence-zero recurrent spill, stream entry, graph reservation and server phase inference.
- [ ] Define exact context accounting for image cells, native positions, pending anchors, correction output and provisional rows. Add red CPU tests for ambiguity and off-by-one cases.
- [ ] Resolve public CLI/library reporting semantics without changing stock or single-slot behavior.
- [ ] Demonstrate native splitting/planning for equal and unequal two-sequence widths; settle the shared-launch scope before implementing a padded/ragged representation.

Exit: reviewed contracts, independent baseline artifacts and explicit supported/reference scope. No guard is removed merely to obtain a runnable configuration.

### MS1 - Pure selection and allowance planner

- [x] Add deterministic CPU planning for admitted groups, selected subsets, maximum-cardinality choice, equal logical headroom and packed offsets.
- [x] Cover the 90Ki/30Ki example, uneven consumption, no remaining headroom, one-token remainder, page tails, zero-length history, individual completion and future N-slot fixtures.
- [x] Prove progress or a terminal/rejected state; no zero-progress reselection loop, integer overflow or invisible age/rotation policy.

Exit: exact CPU invariants independent of GPU timing, including no sharing/lending and no output-count changes.

Accepted implementation: `llama-kv-stream-selection` plans only an already admitted group, with a pure optional resource-admission predicate. It exhausts candidate subsets to preserve the exact selection objective, including rejection of an otherwise preferred pair; it must not be applied to the unbounded waiting queue. Runtime admission will initially limit the group to two. CPU fixtures also cover groups of six without hard-coding two into the planner.

Evidence: missing-API compile red; eight tests with 17,254 assertions, including 2,048 independently enumerated subset-oracle fixtures, passed normally and with the new planner/test translation units under ASAN/UBSAN and leak detection. The complete selected host regression inventory passed: 54 executed tests plus two expected CUDA-only skips. The fatal-warning CPU build included the server. Baseline source, independent host binaries, inventories, logs and file hashes are local under `benchmarks/results/multi-slot-20261109/`; generated artifacts are not committed. Existing runtime paths do not call the planner yet, so no single-slot execution, kernel, allocation or CLI behavior changes in this stage. This does not substitute for MS0's outstanding fresh model/performance controls or later GPU qualification.

### MS2 - Stable host virtual reservation and incremental backing

- [x] Reproduce upfront clear/registration behavior and establish native pinned controls.
- [x] Implement owned stable reservations behind appropriate host-backend interfaces, with committed and registered coverage tracked separately.
- [x] Test growth across chunk/layer/K/V boundaries, registration-boundary segmentation, commit/register failure, rollback of partial growth, clear/truncate/reuse, guards and destruction with queued users. Device aliases remain per-registration; no contiguous alias assumption is introduced.
- [x] Verify existing pointers stay unchanged; untouched maximum capacity does not become resident merely through creation/reset.
- [x] Compare actual transfer/publication behavior and memory counters against the native pinned implementation before promotion; validate platform capabilities honestly.

Exit: stable addresses, correct strict range contracts, bounded demand growth and no implicit pageable fallback.

Accepted implementation: the backend memory interface reserves guarded host address space and exposes owned views only within individual committed ranges. Linux uses `mmap`/`mprotect`; the Windows implementation uses `VirtualAlloc`/`VirtualFree`, with Windows execution still awaiting platform qualification. Views block decommit and retain the reservation. `llama_kv_stream_host_reservation` prepares only newly needed chunks of fixed-stride K/V planes, registers them through the existing strict CUDA host adapter, splits borrowed ranges at registration boundaries and rolls back new chunks on failure. Growth does not rescan the old prefix. No second device arena, device allocation or pageable fallback is added. The eager single-slot allocator and server admission guards are unchanged; server integration follows in later stages.

Evidence: missing-API compile reds; 4,699 host assertions covering all64 formats at D256/D512, every registration-failure position, exception rollback, exact retained bytes and reuse. A Linux `mincore` check observed zero resident pages after reserving 64 MiB and exactly the three explicitly committed pages afterward. Deterministic syscall fault injection covered commit/decommit/discard failure and fail-closed protection recovery. The complete 56-test host inventory passed normally and with full host/server ASAN/UBSAN builds: 54 executions plus two expected CUDA-only skips each. Native registered-range bidirectional async transfers, including dropping the main owner before copy completion, passed 4,898 assertions normally and under strict default-PDL memcheck with API reporting `all`, error exit 99 and zero errors.

Native transport qualification explicitly combines the current host libraries with the archived `a0672f5ea` CUDA component; its strict host-adapter source is byte-identical. Before/after source/library/test hashes, actual loader paths and owned-job cleanup are recorded in local `ms2-native-qualified.json`. This is a host-registration/transport component qualification, not a new full-server or attention-kernel qualification. Linux CUDA SM120 was exercised; missing Windows/hardware execution is not called a pass.

Measured tradeoff for later integration: at the default 1 MiB registration size, two synthetic 8.5/17 MiB H2D payloads used 9/17 DMA calls and took about 0.197/0.394 ms versus 0.181/0.360 ms for the eager native pinned control. Larger 4/8 MiB chunks reduced calls but consumed more initial pinned RAM and retained measurable transfer overhead. Keep the bounded 1 MiB default for now and carry these new-mode costs into MS13 request-level measurements; there is no claimed transfer speedup or existing single-slot slowdown. A proposed single DMA across registrations was rejected by CUDA with `invalid argument`; that failed experiment is preserved and must not be used as a supported coalescing path. Writer publication and reader transport integration must split at the proven registration boundaries, including when a native writer tile crosses one. First-use/setup observations are not repeated startup-performance claims. All generated logs, probes and benchmark sources remain local under the stage evidence directory.

### MS3 - Independent slot histories and logical metadata

- [x] Bind separate slot/cache generations and frontiers to existing host logical-cache/content and writer paths.
- [x] Separate per-slot C, total logical metadata capacity and selected-group capacity in private context parameters and conditional server reporting; retain the runtime admission gate.
- [x] Use explicit slot/request epochs and independent native cell metadata in the new storage path. Keep existing sequence-zero runtime guards until state, kernel and scheduler replacements are qualified.
- [x] Poison neighboring slot ranges and verify byte-exact isolation through append, padding, truncate, replacement, slot-ID reuse and stale-source rejection.

Exit: both slots can retain histories up to their individual limits independently of the bounded physical GPU pool. This storage milestone does not enable two-slot server execution; selected-state activation and joint execution follow below.

Accepted implementation: `llama_kv_stream_slot_table` retains stable canonical full/auxiliary histories, request reuse epochs and the existing native `llama_kv_cells` metadata per slot/role. Original positions, M-RoPE coordinates and token labels remain independent of selected-group offsets. Checked execution-buffer byte I/O exposes partially committed reservations without an ordinary readable buffer or pageable fallback. The optional registered-copy interface retains submitted registrations and reuses the existing copy queue/events; writer encoding keeps native tiles and splits only transport at registration boundaries. Duplicate writer destinations are rejected before submission. Cache-ID reservations include the bounded-cache identity and reject exhaustion without wrapping. The common logical generation budget counts the pending sampled anchor once.

Context limits distinguish logical per-slot/group C, padded selected storage and total host-history metadata. Two-slot limits require a shared unified arena; legacy one-slot padding and ordinary-native context handling remain unchanged. Conditional `/props` fields describe all capacities once later runtime admission is enabled. No server `--parallel 2` configuration is qualified yet.

Evidence: API reds, a real eager-buffer admission abort, cache-ID collision and invalid-growth/metadata admission reds are preserved locally. Current normal and full host ASAN/UBSAN inventories each passed 54 executed tests plus two expected CUDA-only skips. Component counts include selection17,640, logical-cache945, context-memory1,848, host4,917 native and writer4,378 native assertions. A fresh full CUDA build and 16 component gates passed, including six strict default-PDL memchecks at API reporting `all`, error exit99 and zero errors. D256 producer304, ten-root D512 producer2,424 and reader1,179 assertions preserve native bits, finite outputs, streaming, publication, rewrite and reader lifetimes. Early diagnostics using archived CUDA math remain explicitly labeled and are superseded by the fresh-backend gates.

Native reference fixtures now read only active committed rows; the initial scalar oracle accessed uncommitted future rows, proved by the preserved debugger trace. Native DMA controls use separate ordinary input storage when a source crosses host registrations. Neither correction changes arithmetic, history lengths, numerical comparisons or runtime budgets. A runner initially expected the CPU host inventory rather than its two additional CUDA tests; all15 native tests had passed, and unchanged successful gates were revalidated by exact identities before continuing.

Regression controls freeze independent placement-base C++/original CUDA adapters and candidate runtimes with the same 223 unchanged CUDA math objects. Qwen IQ4_XS and Gemma Q3_K_M target-only/MTP3, Q8_0/Q4_0, UVM off, 4,096 context/1,333 prompt/64 outputs and a 1GiB arena passed 32 fresh requests with 2,048 exact output IDs, paired draft/accept counters and zero cached prompt tokens. One warmup plus three measured requests per process gave final median prefill changes -0.065% to+0.346% and decode -0.183% to+0.875%; these are short-screen observations, not speedup or long-context claims.

An earlier roughly1% Gemma MTP3 slowdown triggered library isolation and P/E-core/clock diagnostics. The legacy eager execution-storage size and callback bodies are now unchanged from baseline; virtual I/O has separate storage/callbacks. E-core placement alone reduced decode by about4% and prefill by about10% in the diagnostic. Three alternating pairs on four distinct P cores gave +0.040% prefill/+0.051% decode medians, with roughly1% variation present in both versions. No GPU clock settings or production configuration changed. The broader existing-mode/long-context regression ledger remains mandatory at MS13. Source/library/model/corpus/command/loader identities, failures, measurements and immutable runtimes are local under `benchmarks/results/multi-slot-20261109/`; raw artifacts are not committed.

### MS4 - Selected-state parking and activation

- [ ] Extend recurrent spill/rollback and Gemma bounded local/SWA ownership to physical banks for selected slots.
- [ ] Preserve pending anchors, rollback rows, current state and semantic generations through park/resume; distinguish physical rebinding from content mutation.
- [ ] Inject failures before/after queued copies, publication, lease retirement, graph rebinding and activation. Recover the previous state or fail quiescently without duplicate output.
- [ ] Prove outgoing slots retain no required GPU allocations after completion and that incoming slots do not require a second full arena/state bank.

Exit: compact real-model pause/resume exactness plus CPU/ASAN ownership/recovery coverage.

### MS5 - Exclusive page partition and span planning

- [ ] Quote per-slot native working minima; implement proportional projected-extent page allocation with deterministic rounding and stable ownership.
- [ ] Exhaustively exercise resident-only, ring-only, split resident/ring and both wrap positions, including empty segments, unequal histories, append boundaries and poisoned neighboring pages.
- [ ] Prove three-span coverage for every submitted wave or diagnose the precise layout conflict. Preserve legal native resume intervals and checkpoint order.
- [ ] Keep per-slot feedback and ring progress independent; no unchosen-slot pages or cross-slot quota borrowing.

Exit: CPU geometry/property tests and CUDA transport/native-bit tests with stable page ownership and complete reader fencing.

### MS6 - Shared-launch target-only attention

- [ ] Extend validated native dispatch/resume metadata for two sequences and independent accumulator/output regions.
- [ ] Cover D256 and D512, all 64 existing K/V reader pairs where currently supported, one/two selected sequences, causal/compact masks, unequal histories, wrap/tails and legal microbatch splits.
- [ ] Compare exact outputs and native plans against equivalent ordinary-native batched controls, not an assumed serial identity.
- [ ] Count actual launches to rule out a hidden per-slot attention loop for compatible joint work. Prove finished sequences do not keep modifying their output during another sequence's later waves.

Exit: exact native attention evidence, launch/resource accounting and strict sanitizer success for the changed paths.

### MS7 - Arena catalog and multi-slot admission

- [ ] Measure target prefill/decode, selected counts one/two, output counts, descriptor storage, conversion, transfer, current-state and rollback-publication requirements before arena allocation.
- [ ] Reuse the phase owner for transactional binding; do not infer phase from token count or number of slots.
- [ ] Test exact minima and one-byte-short rejection without mutation, illegal overlap, stale generations, graph replay and warmup error propagation.
- [ ] Reject an arena that cannot support the declared legal two-slot configurations rather than silently replacing joint batches with serial execution.

Exit: reproducible phase minima, no private GPU allocation fallback, and correctly scoped handling of external driver resources.

### MS8 - Server groups, reselection and continuous responses

- [ ] Wire explicit admitted/selected/parked states into existing queues, phase selection and output handling.
- [ ] Test all phase transitions, new arrivals while decoding, different prefill completion times, allowance exhaustion and independently finishing slots.
- [ ] Preserve the same stream and accepted-output cursor through repeated pauses; test SSE heartbeats, cancellation/disconnect, non-streaming results, stop strings, usage and tool-call fragments.
- [ ] Keep live parked slots busy to administrative APIs and keep newly queued prefills out of the current decode group.

Exit: two-slot target-only public API behavior, context-boundary progress and no duplicate/lost/falsely finalized output.

### MS9 - Qwen and Gemma MTP

- [ ] Batch supported MTP drafting and target verification without confusing aggregate rows with per-sequence widths.
- [ ] Bind Qwen auxiliary caches and Gemma full/local shared readers to the correct slot, frontier and physical bank.
- [ ] Qualify widths one through four, confidence cutoffs, unequal widths, each accepted prefix, parked/resumed source pins and target continuation.
- [ ] Retain existing inherited cache-format, sidecar pairing and serial-workspace safety contracts while making their ownership sequence-aware.

Exit: native-matched MTP internals/public output/counters and unchanged single-slot controls.

### MS10 - DFlash and DFlash plus K4V

- [ ] Isolate feature rows, selected-layer inputs, convolution/selector state, draft KV and rollback per slot.
- [ ] Preserve independently configured DFlash/K4V maxima one through seven; test mixed widths, source switches, every accepted prefix and boundary-shortened rounds.
- [ ] Exercise genuine copy hits and genuine neural fallback per slot, including cancellation, park/resume and restored history namespaces.
- [ ] Verify strict per-slot and aggregate generated/accepted/forward counters. A neighboring slot cannot supply missing proof.

Exit: exact ordinary-native internal/public controls, both routes exercised, and no K15 admission hidden in aggregate two-sequence row counts.

### MS11 - Vision groups and image-aware speculation

- [ ] Encode every admitted request's required still images before group prefill, retaining exact embeddings/positions in host storage.
- [ ] Qualify Qwen/Gemma target-only/MTP and the existing Qwen DFlash/K4V image combinations, multiple/interleaved images, ratio extremes and budget rejections.
- [ ] Preserve and restore cached text/image histories across exclusive vision loans without keeping unchosen current states in VRAM.
- [ ] Prove image-aware K4V boundaries, DFlash source-feature validity and independent state restore for each slot.

Exit: all scoped vision modes pass exact fitting native/component controls and actual end-to-end streamed server tests, with bounded resources and correct reference labels.

### MS12 - Cache, persistence and finder integration

- [ ] Qualify prefix reuse, changed requests, low/high/low histories, checkpoint selection, independent save/restore/erase, restart relocation and slot-ID reuse.
- [ ] Extend finder/preflight commands, host tests and reports for explicit two-slot context semantics, all legal selection shapes and automatic mmproj ratio probes.
- [ ] Validate final chosen arenas with representative real joint and serialized workloads, including long-history/shape transitions. Keep upper-neighbor rejection evidence.
- [ ] Measure committed/pinned/parked/checkpoint RAM and establish whether a hard host-memory policy requires a user decision.

Exit: faithful sizing claims, real output examples and lifecycle coverage; no claim of global OOM prevention from a bounded probe.

### MS13 - Full safety and performance qualification

- [ ] Rebuild affected server/tests and run appropriate host, ASAN, CUDA and strict default-PDL memcheck coverage with unchanged API reporting and numerical requirements.
- [ ] Keep strict memory gates at API reporting `all`, error exit code 99 and no kernel filters; eager loading is a sanitizer profile detail, not a production change. Preserve CUDA graphs in gates that explicitly test graph capture/replay.
- [ ] Requalify existing single-slot Qwen/Gemma, all64 reader guards, MTP, DFlash/K4V and vision paths affected by shared changes.
- [ ] Close every applicable stage regression entry against the frozen V2 controls, including ordinary non-streaming behavior and the established prefill conversion/transfer optimizations. Compare each supported mode separately; do not average a regression away across models or modes.
- [ ] Run matched uninstrumented comparisons for two short fitting histories, unequal fitting histories, over-budget serialized selections, growth-triggered reselection and image groups.
- [ ] Include short/medium/heavy-streaming contexts, cold/warm histories and one finished/cancelled member. Use identical model/sidecar/projector/input/sampling identities within each comparison.
- [ ] Report aggregate accepted tokens/s, per-request active and paused latency, group TTFT, phase/selection counts, park/restore time, H2D/D2H bytes, resident/ring ownership and memory peaks. Separate intentional blocking policy from unintended compute regressions.
- [ ] Confirm shared launches and transfer overlap with profiling where available; instrumented throughput is not speedup evidence. Do not assume batching wins when PCIe/state transport dominates.

Exit: no confirmed functional, numerical, safety, compatibility or single-slot performance regression; correctly accounted resources and capacity; correct scope for native fitting versus long streamed-only evidence; current-source safety passes and transparent multi-slot performance results. An explained regression still requires a fix/revert or an explicit user decision to change scope; it is not a passing non-regression result. A new full-context sweep needs its own explicit scope; it is not implied by writing this roadmap.

### MS14 - Documentation and release review

- [ ] Document opt-in configuration, context semantics, phases, fixed group membership, pauses/heartbeats, proportional pages, host address/commit/pin accounting and actual mode/platform limits.
- [ ] Publish measured finder examples and reproducible performance summaries without committing raw benchmark artifacts or local server-AB harness additions.
- [ ] Consolidate status only after reviewing stage evidence; no complete checkbox based solely on a process exit.
- [ ] Review the final regression ledger and ensure no new failure, confirmed slowdown, capacity reduction or missing applicable gate is concealed by a scope change or marked resolved solely by explanation.
- [ ] Keep stage changes reviewable and use the user's one-sentence/title-only commit preference when implementation commits are authorized. Do not commit or push this planning document merely because it was requested.

Exit: all scoped combinations supported for two slots, single-slot compatibility preserved, no unqualified N-slot/physical-GPU/platform claims, and no production configuration changes without authorization.

## 12. Acceptance matrix and evidence rules

The following dimensions are required across appropriate stages; they need not form one blind expensive Cartesian server matrix.

| Dimension | Required cases |
| --- | --- |
| Selected slots | One and two on GPU; future N in CPU metadata/planner tests. |
| Histories | Equal, highly unequal, zero/short initial prefixes, group total below/at/over C, each slot near its own cap. |
| Growth | Equal/unequal acceptance, allowance exhaustion before the other slot, sub-page headroom, rejected provisional rows and pending sampled anchors. |
| Placement | Resident-only, streamed-only where admitted, mixed, odd/small rings, both wraps, append boundary and registration-chunk boundary. |
| State | Qwen recurrent/rollback and auxiliary MTP; Gemma full/local KV and assistant source views; supported DFlash feature/selector/cache state. |
| Formats | All existing reader-pair operator coverage; at least existing Q8_0/Q4_0, Q8_0/Q8_0 and Q5_1/Q5_1 end-to-end profiles for both supported targets. Preserve native F16 and existing optimized-conversion admission. |
| API | Both public completion styles, stream/non-stream, cached/changed prompts, stop/EOS/context end, cancellation/disconnect and queued arrivals. |
| Vision | Existing still-image modes, multiple/interleaved images, independent M-RoPE positions, largest admitted shapes, encoder rejection and return to text. |
| Faults | Commit/register/copy/compute/retirement/restore failure, stale IDs/generations, short grants, poisoned neighbor pages and cleanup retry. |
| References | Same native batching, microbatch, masks, positions, quantization and proposal/acceptance semantics; explicit self-consistency labels where full native capacity is unavailable. |

### Mandatory non-regression gate

There are two distinct controls: ordinary-native execution establishes numerical correctness at equivalent execution shapes; the frozen pre-multi-slot V2 implementation establishes preservation of existing functionality, resources and performance. Keep their roles explicit.

| Protected behavior | Required existing-mode control |
| --- | --- |
| Single-slot target-only | Qwen and Gemma, short through heavy-streaming histories, prefill and decode, native plans/output checks and public API accounting. |
| Single-slot speculation | Qwen/Gemma MTP, Qwen DFlash and DFlash+K4V across their supported depths, confidence/source transitions, acceptance, rollback, hidden features and genuine neural/copy routes. |
| Single-slot vision | Existing Qwen/Gemma target-only/MTP and Qwen DFlash/K4V image paths; exact encoder/decoder references, native positions, rejection, return to text and state lifecycles. |
| KV formats and attention | Preserve all64 operator-reader coverage and the existing end-to-end three-pair profiles. Preserve F16/native handling, precision, masks, tails, wraps and alias/short-grant canaries. |
| Established prefill optimizations | Bounded conversion reuse, native F16/noneligible paths, half-ring transfer behavior, source publication and role-specific feedback/accounting. |
| Memory and admission | Same-profile one-slot phase minima, required-byte diagnostics, bounded workspace, image capacity and maximum-arena behavior; no new private GPU allocations, leaks or unexplained pinned/RSS growth. |
| Server and persistence | Prefix reuse, recurrent/SWA checkpoints, slot save/restore, cache controls, restart, cancellation, stream/non-stream output, tool/usage accounting and preserved default/CLI semantics. |
| Ordinary non-streaming paths | Existing native single/multi-slot context semantics and relevant server tests remain unchanged when adaptive streaming is disabled. |
| Builds and platforms | Relevant current fork CI checks, fatal-warning/shared-library builds, Windows interfaces and affected CUDA architecture/backend compile paths. Shared code must not break HIP/MUSA or other affected backend builds. Missing hardware is reported, not called a runtime pass. |

Apply controls according to changed dependencies:

| Milestones / shared area | Minimum regression focus before stage exit |
| --- | --- |
| MS1 / MS8 scheduler | Serial CLI/context behavior, group-independent legacy scheduling, output/cancellation/cache tests and ordinary server API tests. |
| MS2 host storage | Existing pinned backing, copy/writer behavior, clear/truncate/state I/O, ASAN ownership and selected single-slot growth/performance controls. |
| MS3 / MS4 histories/state | Qwen recurrent and speculative rollback, Gemma local/source readers, exact state/frontier/generation preservation and lifecycle recovery. |
| MS5 / MS6 layout/kernel adapters | D256/D512 all64 exact operator controls, native plans, established conversion/pipeline paths, memory canaries and applicable strict CUDA checks. |
| MS7 arena/preflight | Existing one-slot phase minima, exact/short-quota admission, graph lifecycle, driver-resource setup, memory footprint and finder contracts. |
| MS9 / MS10 speculation | Every existing supported single-slot acceleration combination affected by changed feature/rollback/source integration, with public tokens and exact source counters. |
| MS11 / MS12 vision/cache/finder | Existing image budgets/ratio reports, checkpoints, save/restore/restart, cached/changed inputs and controlled operator workspace. |
| MS13 / MS14 release | Fresh integrated coverage for the final source and all required existing-mode comparisons; earlier passes are reused only under the explicit identity rules. |

The table is a minimum, not permission to skip additional consumers revealed by dependency review. A common owner/header change can require several rows. Preserve a working legacy one-slot path during development; adopt new common host/kernel paths for it only after those paths pass the same applicable controls. No permanent hidden fallback is a substitute for qualification.

### Matched performance and resource comparisons

- Compare candidate `--parallel 1` against frozen V2 `--parallel 1` with the same models/sidecars/projector, KV types, sampling, context/prompt/output sizes, arena, image budgets, graph/UVM settings and runtime libraries. Use each variant's own pinned libraries and verify actual loading.
- Establish baseline run-to-run variation and acceptance criteria before interpreting candidate results. Use repeated matched, interleaved uninstrumented measurements where needed. Performance noise handling never changes numerical tolerances or API/memory checks.
- Measure prefill, accepted-token decode, image encoding/end-to-end latency, startup, first commitment/registration and subsequent growth stalls separately. Faster startup from lazy allocation cannot hide slower request execution; report where the cost moved.
- Include cold and warm paths, short and long histories, cached prompts and image transitions. Improvements in one mode do not excuse a confirmed slowdown in another. A faster two-slot total does not excuse a slower legacy one-slot path.
- Report new-mode overhead explicitly, including a two-slot server with only one selected request, parking/restoration, reselection and long pauses. The user-approved blocking policy and the intrinsic state cost of additional selected slots are scoped multi-slot tradeoffs, not permission for single-slot regressions or unquoted allocations.
- Keep target/draft/projector weights shared rather than duplicating them per slot. Preserve one parent arena and the approved existing one-slot preparation minima. Quote any additional two-slot requirement explicitly.
- Compare same-profile admitted capacities under controlled conditions. Investigate reductions in available arena/image budget or unexpected workspace/pinned-memory growth; do not enlarge fixtures, change quants or add a hidden reserve to mask them.

### Failure handling and regression ledger

For each stage, record baseline/candidate source and binary identities, affected modes, commands/profiles, functional/numerical results, resource/performance results, expected skips and unresolved findings. Keep generated measurements local rather than checking raw artifacts into Git.

If a regression appears, preserve the failing artifacts, reproduce its cause, add a focused regression test where appropriate, and fix or revert the responsible change. Rerun the affected gate and its dependent existing-mode controls after the fix. Do not mark the stage complete, promote its result or declare the feature ready while a confirmed regression remains. If an unavoidable tradeoff requires changing the agreed scope, present the measured evidence for a user decision; do not label it non-regressing.

Preserve source and binary hashes, actual commands/environments, hardware/runtime identity and exact input/model artifacts. Reuse a gate only with explicit relevant-source/binary/profile identity checks. Do not relabel old single-slot, legacy-compile or removed raw artifacts as new passes. Retain red fixtures and classify harness mistakes separately from product failures.

Fitting native control capacity is a measured prerequisite. Do not silently change target/draft/projector quantization, UVM, history length, output count, tolerance or API reporting to obtain a pass. Use previously approved reference procedures only within their stated scope; a material new reference concession requires discussion.

The later implementation authorization permits stage implementation, qualification and commits; it does not authorize production configuration changes, agent delegation, pushes or unrelated artifact cleanup. Implementation/testing must use owned jobs, preserve unrelated local work and account explicitly for production-service/GPU availability before disruptive qualification.

## 13. Review points and discussion gates

There are no additional blocking user questions before writing this plan. The following points must not be resolved silently during implementation:

1. **Unequal-width launch feasibility.** A shared launch is the target for compatible native microbatches/waves, not a promise of one launch for an entire layer regardless of native splitting. If preserving native arithmetic requires extra serialization beyond native behavior, measure and present the alternatives before accepting a weaker batched feature.
2. **Three-span feasibility.** Prove the selected physical layout and wave plan retain the bound. A demonstrated need for more spans, a different reduction plan or additional GPU staging needs a concrete design review, not a constant increase or enlarged fixture.
3. **Host backing and RAM budget.** Confirm incremental registration/device alias behavior and measured costs. A pageable/staging redesign, eager maximum commitment or a new hard RAM cap is a material decision if the preferred approach fails.
4. **Capacity/compatibility gaps.** Preserve real OOM/unsupported-device evidence. Report any unavailable native or platform qualification rather than lowering the scope or changing models without agreement.
5. **Client deadlines.** Long pauses are the agreed scheduling behavior, but some clients can terminate the connection regardless of SSE pings. Document tested behavior and discuss any required timeout/protocol change before claiming transparent indefinite pauses for those clients.

Routine choices such as exact internal class placement, checked arithmetic helpers, deterministic tie breaking, registration chunk size within measured limits and reuse of existing ownership interfaces can be resolved through implementation and TDD. They do not require another approval cycle merely because they appear in a roadmap.

Avoid two further accidental scope changes: equal append allowance is logical and does not authorize new permanent GPU append banks; and a restarted process may restore a saved conversation but cannot resume the old TCP/SSE connection. In-process parking must resume the same connection, while restart tests use the existing explicit restoration/new-request contract.

Primary API references for MS2: [Linux mmap](https://man7.org/linux/man-pages/man2/mmap.2.html), [CUDA host registration and memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html), and [Windows VirtualAlloc](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc). Verify behavior against the deployed SDK, OS and device capabilities; API documentation alone is not a runtime qualification.
