# Multi-slot adaptive KV streaming and phase-arena roadmap

Status: MS1-MS14 have accepted implementations. The latest MS14 follow-up retains the original serial adaptive policy and applies it to combined per-layer KV demand through one shared grouped ring. At two 64K histories, all four Qwen modes improve aggregate throughput by 1.45%-9.19% over validated double buffering; eight single-slot controls remain within -0.10% to +0.34% of the original serial runtime. Earlier measurements below retain their original configurations and limitations. MS15 implements admission during prefill at safe microbatch boundaries, shared native prefill chunks and handoff to grouped decode; its TDD, qualification and focused SWE measurements are recorded below. Membership remains frozen after decode starts. The former 4,096-token growth-reserve proposal is withdrawn from MS15. The original MS15 qualification missed a fresh-state initialization defect; the 2026-11-17 investigation records its reproduction and repair, committed and deployed as 19099a105. MS15 performance acceptance remains open: speculative staggered pairs have higher end-to-end latency despite faster aggregate decode. MS16 is closed with accepted qualification limits, committed as af64b74dec91b0f811eefc12742115c30ef7b5bf and subsequently deployed with three configured slots. Its 236-run extended sweep is complete. MS17 is planned to productize compositional numerical-equivalence tooling and qualification across every supported model/mode/format/slot/modality combination, with no preferred configuration. Documentation/release closeout remains deferred.

Inspected baseline: `f6ca3e8a172473578b909a4691e616bb2234fb54`, on `feature/ci-workflow-migration`. Recheck the implementation base and local changes before starting work on a dedicated feature branch. Historical measurements retain their original source, binaries, models and configuration; they are not new multi-slot evidence.

Roadmap placement base after pulling the main KV-stream V2 branch: `2306b0c1d9252791149714ea9820218ef2599a8a`, on `feature/adaptive-kv-stream-v2`. MS0 must freeze regression controls from the reviewed implementation base, rather than assuming the earlier inspected commit or an old production binary is equivalent.

Implementation branch: `feature/adaptive-kv-multi-slot`, starting at the placement base above. The user authorized implementation and one title-only commit per completed stage on 2026-11-09. Existing unrelated local files and production configuration remain untouched.

## Objective

Support two concurrent request slots using one shared phase arena on the existing CUDA Qwen and Gemma streaming paths. Batch compatible selected slots in the same attention launch, with separate histories and up to three spans per sequence per submitted wave. Design identities, descriptors and scheduling for future N-slot support without advertising unqualified slot counts.

MS16 extends this established feature to configurable N request slots under the contract below. The two-slot descriptions in completed stages retain their historical scope; higher counts become supported only after their applicable qualification gates pass.

MS17 addresses the remaining numerical-reference gaps through bounded compositional qualification. Existing same-fork native comparisons, public smoke tests and throughput sweeps retain their recorded scope; an independent stock-equivalence claim requires the reference provenance and composition gates defined in MS17.

The completed feature includes all currently supported combinations: target-only execution, supported Qwen/Gemma MTP, Qwen DFlash, Qwen DFlash plus K4V, and their existing supported still-image paths. This does not introduce Gemma DFlash, K15, new model pairings, audio/video, multi-GPU or previously unsupported LoRA combinations.

Preserve native arithmetic, conversion, masks, accumulator precision, checkpoint boundaries and reduction order. Preserve single-slot functionality and performance. Reuse the existing memory owner, leases, phase transitions, speculative dispatcher, MTMD and rollback interfaces. There is one GPU arena, not one arena per slot.

**Non-regression is a mandatory stage and release condition.** Every stage exit below also requires the applicable [non-regression gate](#mandatory-non-regression-gate). A successful two-slot test or an aggregate throughput gain cannot compensate for breaking an existing mode, reducing single-slot performance or weakening a safety/admission check.

## 1. Settled user decisions

| Topic | Contract |
| --- | --- |
| Initial concurrency | The original target was two request slots. MS16 supports configurable N and resource-admitted subsets, with actual higher-count qualification limits recorded below. |
| Context limit | In the new streaming multi-slot mode, `--ctx-size C` is the per-slot limit and the aggregate logical-context budget of a selected execution group. |
| Large combined histories | If admitted slots cannot fit together under that budget, execute smaller selections sequentially. Preserve every paused history in RAM. |
| Full attention KV | Authoritative histories are in host memory. VRAM holds selected slots' resident pages and streamed ring waves. |
| Other live state | Only selected slots' current recurrent/SWA and required speculative state occupy VRAM. Park outgoing state in RAM. |
| Physical allocation | Assign exclusive whole-page ranges to selected slots. No mixed-slot physical pages or borrowing another slot's allocation during a selection. |
| Proportions | History allocation is proportional to context length, with equal logical append allowance for selected slots. |
| Growth allowance | An allowance is logical generation headroom, not dedicated VRAM for that many tokens. |
| Host addresses | Reserve maximum host virtual ranges during startup/preflight in the actual server process. Grow backing without moving existing history addresses. |
| Phases | Vision, prefill and decode execute separately. Do not retain a mixed prefill/decode workspace profile to serve incoming requests. |
| Request groups | Retained MS14 freezes membership at initial admission. MS15 additionally admits prefill-ready arrivals during prefill at safe microbatch boundaries, then freezes membership when decode starts. |
| Reselection | Trigger on allowance exhaustion, lack of space for the next operation, completion, cancellation or an individual context boundary. |
| Waiting policy | Return paused requests to eligibility at the next selection. No periodic rotation, waiting-time override or forced turn. Prefer parallelism; a long context may wait. |
| Generation | No new finite output cap. Existing EOS, stop strings, requested output limits, cancellation and individual context limits still apply. Context shifting remains outside the supported streaming profile. |
| Streaming output | Emit accepted tokens incrementally. Pausing is not completion; keep the same response open and resume without duplicate output. |
| Scope | All existing supported acceleration and still-image combinations must pass before the complete feature is declared supported. |

These decisions replace earlier suggestions for mixed-phase scheduling, equal resident quotas, borrowing pages within a selection, or periodic fairness rotation.

The allocation entries above describe the original per-slot implementation. The retained MS14 aggregate-ring follow-up supersedes per-slot KV pool/ring partitioning: resident slices are source-specific, one fenced ring serves combined live demand, and the controller does not reserve a ring per request. Logical append allowances and host histories retain their existing semantics. MS15 extends admission during prefill and prefill execution against this retained baseline; it keeps the accepted decode allocation/controller policy. Mixed prefill/decode execution and request rotation remain outside that scope.

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
| `N` | Configured request-slot limit, initially at most two; MS16 removes that adaptive-runtime restriction within native and checked resource bounds. Qualified GPU counts are recorded separately. |
| `A` | Admitted request group, containing at most N live requests. Retained MS14 freezes membership at admission; MS15 permits joins during prefill and freezes it for decode. |
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

At a selection event, consider eligible requests from the admitted group. Under MS15, a validated prefill-time join can extend that group before selection; decode-time arrivals remain queued for another group. Prefer the greatest number of compatible requests that can make progress under C and the measured arena contract. Prefer smaller combined history among equal-cardinality choices, with a deterministic slot-order tie break. This implements the agreed preference to park a large context rather than reduce the number of other active slots.

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

Retained MS14 forms an admitted group from requests already queued, up to N, without an artificial wait to fill it, then freezes membership. MS15 amends this rule only during prefill: admit eligible prefill-ready arrivals at completed microbatch boundaries, preserving existing progress and bounded resources. Check eligible arrivals before committing prefill to decode, then freeze membership. Requests arriving after decode starts wait for the next group, even if one member finishes early. The retained implementation and qualification are recorded under MS15 below.

| Transition | Required condition |
| --- | --- |
| Idle to vision | The admitted group needs image encoding. |
| Idle to prefill | No encoding is required. |
| Vision to prefill | All required embeddings for the group are ready; encoder loans and work have completed. |
| Prefill to expanded prefill (MS15) | A queued prefill-ready request fits a free slot and passes transactional admission at a safe microbatch boundary. |
| Prefill to decode | Every remaining admitted member has completed prefill. A completed prefill can wait with state parked while other prefills finish. MS15 checks visible eligible arrivals before committing this transition. |
| Decode to next group | All admitted members, including temporarily parked decoders, completed, cancelled or reached their native context boundary. |
| Any phase to idle | No admitted work remains and no next group is queued. |

Only one global phase executes at a time. Within prefill, select chunks/subsets with the same logical and physical bounds; MS15 packs compatible unfinished prefills into shared native microbatches. Combined histories need not fit all at once merely because they are admitted. Within decode, change selected subsets without admitting new prefills or vision work.

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

- [x] Extend recurrent spill/rollback and Gemma bounded local/SWA ownership to physical banks for selected slots.
- [x] Preserve pending anchors, rollback rows, current state and semantic generations through park/resume; distinguish physical rebinding from content mutation.
- [x] Inject failures before/after queued copies, publication, lease retirement, graph rebinding and activation. Recover the previous state or fail quiescently without duplicate output.
- [x] Prove outgoing slots retain no required GPU allocations after completion and that incoming slots do not require a second full arena/state bank.

Exit: compact real-model pause/resume exactness plus CPU/ASAN ownership/recovery coverage.

Completed as storage/ownership components. Selection projection, complete workload admission, server request/sampler ownership and speculative group integration remain MS6-MS10 work; these passes do not enable `--parallel 2`.

Recurrent current-state roots can start without device payload, quote native allocation sizes and bind checked persistent leases. Parking saves all raw current/provisional/rollback rows in RAM and retains native cells, source/tail indices, rollback indices, spill namespaces/generations and pending bookkeeping. It does not serialize only the accepted checkpoint. Partial capture keeps the original bank; partial restore keeps the saved image and closes execution until retry. State export/import, restore, materialization and publication cannot bypass the parked gate. Clear cannot resurrect old saved bytes.

Gemma local/SWA roots use the existing native constructor/binding path. Their parked image preserves physical encoded row order and native cell metadata, including spatial/token fields; native checkpoint repacking is not substituted for physical suspension. Bounded-owner pins and writer tickets prevent parking unfinished or escaped readers. Callers must drain work and retire every graph/reader before replacing native roots. Failed restore stays unreadable; semantic source IDs, generations and frontiers remain unchanged by a successful physical rebind.

Publication setup can prepare its native queues/events and host destinations without allocating a private device stage. Explicit borrowed current/publication regions must both be present before admission. Actual Qwen geometry quoted 156,893,184 bytes for one current state, 313,786,368 for two, and 19,611,648 for depth-three publication staging, inside the fixed 1 GiB component parent. Retained queue/event resources are not a claim of zero driver footprint; the complete admission catalog remains MS7 work. Native CUDA arithmetic, kernels and reader formats did not change.

TDD preserved missing-API and behavior reds for unbound I/O, invalid geometry, parked operations, incomplete publication binding and stale rows after clear. Real-copy failures exercise retry after actual native downloads/uploads. A strict-run host crash was localized with GDB to an existing short/padded fixture retaining CUDA captures across freed dependencies. The fixture now drains and explicitly retires native captures before releasing its buffers and also proves three live capture/replay epochs with exact bytes/generations. Graphs, API reporting and comparisons were not relaxed.

Final fresh qualification: 11 native gates, including three default-PDL strict memchecks with API reporting `all`, error exit 99, no filters and eager loading only for the tools; all three report zero errors. Native publication executes 42 tests/4,571 assertions; real-geometry Qwen current-state caches execute 1,872 assertions and Gemma physical SWA caches 3,127, normally and strictly. Normal and full ASAN/UBSAN host inventories each execute 54 tests plus two expected CUDA-only tile skips. Ordinary GPU-weight Qwen/Gemma runs compare full logits, hidden vectors and state against uninterrupted matching controls through repeated parking; Gemma uses a wrapped 1,333-token prefix. Existing Qwen MTP depths 1-3 at 16,381 prefill tokens/eight rounds compare 7,736 framed native/streamed records per variant; DFlash shared-workspace and Gemma assistant widths/quotas also pass.

Sanitizer scope is explicit: a full-model Qwen strict attempt could not allocate 12,726.35 MiB of weights before context creation and reported two API errors/exit 99. That failure is preserved, not counted as a memory pass. The strict cache fixtures map the unchanged model weights on CPU, perform no neural computation, and exercise actual native device state/cache geometry, transfers, bindings and failure recovery. The separate ordinary full-model controls retain the original GPU weights and arena. This is not a full-model sanitizer or global OOM-prevention claim.

The final-runtime short single-slot screen and alternating pairs cover Qwen/Gemma target-only/MTP3 and Qwen DFlash7/DFlash7+K4V7: 36 processes, 144 requests and 9,216 generated IDs, with exact prompts, output IDs and speculative counters. Paired decode median changes were +0.123% Qwen target-only, +0.392% Gemma target-only, -0.132% DFlash7 and -0.123% DFlash7+K4V7; residual sub-percent timing differences overlap observed control variation and are not advertised as gains or exact timing parity. The full MS13 long-context/per-mode regression ledger is still required. Both variants use their own pinned libraries, the same 1 GiB arena, 4 Ki context, 1,333-token natural SWE prompt, 64 outputs, one warmup/three fresh measured requests and four distinct P cores. No clock or production setting changed.

Evidence, failed attempts, source patches and immutable own-library cohorts remain local under `benchmarks/results/multi-slot-20261109/` (`ms4-qualification.json`, `ms4-final/`, `ms4-final-runtime-bin/` and the final performance directories). Final source/binary/input identities were checked before/after qualification. Binaries were compiled from the preceding commit plus the saved qualified patch and are not relabelled as rebuilt by the metadata-only milestone commit. Raw artifacts are not committed.

### MS5 - Exclusive page partition and span planning

- [x] Quote per-slot native working minima; implement proportional projected-extent page allocation with deterministic rounding and stable ownership.
- [x] Exhaustively exercise resident-only, ring-only, split resident/ring and both wrap positions, including empty segments, unequal histories, append boundaries and poisoned neighboring pages.
- [x] Prove three-span coverage for every submitted wave or diagnose the precise layout conflict. Preserve legal native resume intervals and checkpoint order.
- [x] Keep per-slot feedback and ring progress independent; no unchosen-slot pages or cross-slot quota borrowing.

Exit: CPU geometry/property tests and CUDA transport/native-bit tests with stable page ownership and complete reader fencing.

Completed as page-planning and reader-ownership components. The CPU planner consumes a validated, immutable MS1 selection and caller-measured native page minima, weights pages by projected history plus allowance, clamps minima and uses deterministic largest-remainder rounding. All physical pages belong to one selected slot until reselection. The planner does not infer backend admission or enable public multi-slot execution. Equal-growth snapshot validation was added after an exact one-failure behavioral red; missing-API evidence and all failures remain preserved.

Normal and full ASAN/UBSAN inventories each passed 54 executable tests with two expected CUDA-only tile skips. Selection checks passed 16 tests/123,622 assertions, including an independent active-bound/brute-integer allocation oracle, extreme products, future 256-slot metadata and transactional malformed-input guards. Span geometry passed 8 tests/20,112 assertions, including every resident/ring split, ring rotation, live extent and partial tail in the bounded geometry grid, with independent physical-row byte oracles and poisoned peer pages.

Fresh CUDA ownership controls passed all64 reader pairs at D256/D512, TG1/4/8, four physical layouts and dense/compact masks: 3,072 profiles/42,115 assertions in both normal and strict runs. Queued controls passed 42 profiles/2,803 assertions in both modes, covering all eight encodings, exact-minimum ring-only through resident-heavy quotas, one shared KV/scratch parent, independent sources/feedback, cancellation and errors returned only after successful real enqueue/acquire/release operations. Actual submitted feedback tickets were collected before freezing the isolation reference; ring-only histories legally have no complete-layer feedback samples. Canonical host bytes/generations, peer outputs and page canaries remain unchanged. These use separate per-slot attention calls and do not claim the MS6 shared-launch objective.

The mapped existing-reader/prefill controls also passed: D512 spans 9,295 assertions normal/strict, D512 tile 27,521 normal, prepared conversion 19,969 normal/strict, transfer pipeline 8,368 normal/strict, GQA6 89 normal and all64 D256 pairs 3,081 normal. The final qualification contains 13 native gates and five strict default-PDL memchecks with zero errors, API reporting all, exit99, no kernel filters and eager loading only for the sanitizer. Native graphs remain enabled; transient reference helpers explicitly retire their captures before freeing dependencies. No CUDA kernel or production arithmetic changed, and the new pure CPU planner has no existing runtime call sites. Full request-level performance qualification remains MS13 work.

The final runner expected one test for the legacy D256 command, which actually runs an earlier CPU geometry test plus the complete all64 native matrix. Both passed (2 tests/3,081 assertions, 512 zero-error comparisons, 64 pairs, exit0). The failed runner and primary error are preserved; a separate audit verified every completed log, inactive owned units, current source/library identities and CPU inventories without rerunning or changing a numerical check. Evidence is local under `benchmarks/results/multi-slot-20261109/ms5-final/` and `ms5-final-audit/`; artifacts are not committed.

A CPU-only ordinary-native placement probe for MS6 also confirmed that successive unified two-sequence appends interleave native KV columns: one 64-row tile already has eight owner runs after a short decode. Its exact masks contain holes and cannot be represented by one compact interval. This is a native logical-projection issue, distinct from the proved per-slot physical three-span bound. MS6 must review bounded staging/projection against native checkpoint boundaries before adopting a shared-launch representation; it cannot reorder columns, approximate masks or serialize compatible work to avoid the conflict.

### MS6 - Shared-launch target-only attention

- [x] Extend validated native dispatch/resume metadata for two sequences and independent accumulator/output regions.
- [x] Cover D256 and D512, all 64 existing K/V reader pairs where currently supported, one/two selected sequences, causal/compact masks, unequal histories, wrap/tails and legal microbatch splits.
- [x] Compare exact outputs and native plans against equivalent ordinary-native batched controls, not an assumed serial identity.
- [x] Count actual launches to rule out a hidden per-slot attention loop for compatible joint work. Prove finished sequences do not keep modifying their output during another sequence's later waves.

Exit: exact native attention evidence, launch/resource accounting and strict sanitizer success for the changed paths.

Completed as native attention/transport components. Unified KV keeps the ordinary single K/V stream, aggregate query rows and exact sequence-isolating masks. Repeated native appends can create more than three logical source runs inside one indivisible native attention tile. The reviewed solution preserves native column order and each slot's private three-span bound: gather exact encoded rows into one fixed 256-row attention tile, then execute the existing native resume plan over all query rows. This does not reorder arithmetic, expand the span limit, insert fake queries or run attention separately per slot.

The private copy registry's version12 operation validates all bounded strided rectangles and physical aliases before enqueue, clears the tile only for its first packet, and appends later private-page packets on the existing producer stream. Encoded readers release only after their D2D completion marker; attention then reads the independent stage. There is no new KV allocation, allocator, attention kernel or strong GPU cache. Native stream/driver resources must still be warmed before final arena admission. The stage is inside the same parent and must be included in MS7's quotes: Q8_0/Q4_0 costs 425,984 bytes at D256/4KVheads or 851,968 bytes at D512/4KVheads; all-F32 maxima are 2/4 MiB. This is distinct from the existing 2K-row prefill conversion stage and does not change the legacy single-slot minimum.

The full operator matrix covers 1,192 profiles: all64 pairs at Qwen's D25624Q/4KV and Gemma's D51232Q/4KV, one/two selected target-only sequences, with wider public-pair shapes including 1+3, 2+4, 4+4 and 8+8 rows. Packed, native append, repeated interleaving, a whole empty native tile and nonmonotonic native placement preserve all ordinary-native output bits. Compact controls are used only for exactly representable native masks; interleaved requests retain their dense masks with holes. Two layers and three actual graph replays give 7,152 full native-vector comparisons. Private three-page rings, three simultaneously ready spans, wrap, tail padding, canonical bytes/generations and parent canaries are checked. Null prototype K/V payloads prove that the candidate cannot read the reference's complete GPU cache. Native span plans use the existing validated span-resume interface, while vector plans retain resident resume.

The initial full normal run completed every numerical case but failed its final expected-grid assertion: 1,070 was an arithmetic mistake; the actual complete grid is 1,192. The original failed process/log/executable remain preserved. Independent grid/source/runtime audits retained its numerical records explicitly; the corrected client's only changed instruction byte is the expected-count comparison immediate. No numerical, API, tensor, shape or profile check changed. The corrected full strict run passed all 856,159 assertions and reported zero errors. Normal/full-ASAN-UBSAN CPU inventories each passed 54 executable tests plus two expected CUDA-only tile skips.

Fifteen fresh GPU gates, including seven strict default-PDL memchecks, passed for the final runtime. They include gather replay/blocked-producer ordering and rejection guards, existing copy queues and registered host transport, D512 spans/tile, the existing prepared conversion and transfer-pipeline paths, and all64 D256 controls. Strict gates retain API reporting all, exit99, no kernel filters and eager loading only for the tool; native graphs remain enabled. Independent test-only Runtime observation verified exactly one attention kernel in each of 15,084 native waves, including aggregate queries 1/2/4/6/8/16, with stock fixup kernels counted separately. An earlier observer omitted the existing D512 runtime-kernel name; that instrumentation failure was preserved and corrected without changing execution. This is launch accounting, not occupancy or throughput evidence.

All220 other CUDA objects retained their hashes; only the existing host adapters rebuilt. Source, test clients, runtime libraries, complete grids, owned-unit cleanup and CPU inventories were audited before the ledger edit, with GPU memory returned to its initial idle level. Local evidence is `benchmarks/results/multi-slot-20261109/ms6-qualification.json` and its linked logs/manifests; raw artifacts are not committed. Client Q/mask/output and ordinary reference allocations outside the component parent are explicitly not a full phase-budget claim. Native local-SWA/state projection, complete arena admission, public server scheduling and full existing-mode performance qualification remain MS7 onward; no public `--parallel 2` streaming configuration is enabled by this component stage.

### MS7 - Arena catalog and multi-slot admission

- [x] Measure target prefill/decode, selected counts one/two, output counts, descriptor storage, conversion, transfer, current-state and rollback-publication requirements before arena allocation.
- [x] Reuse the phase owner for transactional binding; do not infer phase from token count or number of slots.
- [x] Test exact minima and one-byte-short rejection without mutation, illegal overlap, stale generations, graph replay and warmup error propagation.
- [x] Reject an arena that cannot support the declared legal two-slot configurations rather than silently replacing joint batches with serial execution.

Exit: reproducible phase minima, no private GPU allocation fallback, and correctly scoped handling of external driver resources.

Completed as the target workload catalog and transactional ownership components; public two-slot execution remains disabled until server integration. The private measurement context builds real native graph topology without changing shared model flags, allocating output/cache/state payloads or admitting compute. Exact native input order, microbatch partitions and output flags are retained. Managed full-attention scratch is quoted separately while native local attention, graph copies and all other scratch remain in the measured graph peak.

The catalog combines per-source canonical history identities, selected-count native recurrent/SWA storage, bounded writer and conversion/gather storage, per-source minimum pools and staged rollback publication. Each live region has its own alignment. A successful quote does not activate an execution context, prove all startup allocations fit globally, or qualify auxiliary speculation graphs scheduled for MS9/MS10. Missing joint availability rejects the declared shape rather than silently serializing it. Qwen K3/K7 dry publication uses real native tensor geometry and zero committed host/device payload; no K15 configuration is enabled.

The existing phase owner now accepts measured profile IDs, an independent selected-set revision and all source identities. Same-phase reselection follows the existing transactional retire/bind/recover path. CPU tests cover stale/missing/duplicate identities, exact native member-range coverage, overflow, one-byte-short rejection without output mutation, failed activation and retry. Legacy serial validators and default owner calls preserve their original behavior.

Native component qualification covers all64 reader pairs at D256/D512, geometry-aware query quotes through aggregate16 (4,096 profiles/13,315 assertions/strict zero errors), and TG16 joint native vectors (128 profiles/245,891 assertions/768 complete native vector comparisons in both normal and strict runs). Legacy D512 spans and all64 D256 controls also pass. Only the existing partial-attention host adapter object changed; all222 other CUDA objects retain their hashes. Native kernels, arithmetic and the serial width limits are unchanged.

Actual Qwen/Gemma GPU model placement and CPU-weight/GPU-storage strict controls quote and bind one physical parent at the exact target-profile minimum. Native current/SWA storage and disjoint graph/writer/attention/gather/private-pool regions fit; one byte less is rejected before binding and byte canaries survive metadata remeasurement. Full host and ASAN/UBSAN inventories each execute54 tests with two expected CUDA-only skips. ASAN/CUDA metadata tests use the NVIDIA documented shadow-gap compatibility setting with leak checks retained; a deliberate overflow self-test is still detected. Strict memchecks retain default PDL, API reporting all, exit99, no filters and zero errors; eager loading is tool-only. These cache/binding controls do no neural arithmetic and are explicitly distinct from ordinary GPU-weight neural controls.

Fresh neural controls retain full native vectors/state through Qwen/Gemma parking, Qwen MTP, DFlash shared workspace and Gemma assistant source/verification widths/quotas. Performance evidence is a scoped existing-mode short-context screen against the immutable placement-base V2 runtime, using each variant's own libraries, exact model/prompt identities and output/draft counters. The six-mode screen covers 36 processes/144 requests/9,216 exact IDs. Small consistent decreases prompted a separate 256-output follow-up for Qwen DFlash7 and Gemma target-only/MTP3: 18 processes/72 requests/18,432 exact IDs. Their paired decode median changes were +0.185%, -0.084% and -0.295%, with observed variation and mixed-sign pairs; these are not speedup or exact timing-parity claims. Full long-context/vision/per-mode release qualification remains required at MS13.

A final review found that the new Qwen measurement adapter requested sequential IDs, whereas native unified batching accepts other query orders. A reversed-order fixture produced exactly one failure after the existing controls passed. The one-line measurement-only correction now uses the native unified split policy. Both real models passed the corrected native-order, exact-binding, strict memcheck and ASAN/UBSAN controls, followed by the complete host inventories. Current-source common warmup fault injection/retry passed 25 tests/1,569 assertions. Earlier neural/performance measurements retain their immutable runtime cohort; every other runtime source is byte-identical and the sole corrected method rejects execution contexts and has no legacy one-slot execution caller. This scope and source mapping are explicit rather than calling the earlier binaries rebuilt.

Failures, red fixtures, source patches, runtime/library/input hashes, loader evidence, all logs and resource cleanup remain local under `benchmarks/results/multi-slot-20261109/` (notably `ms7-qualified-source-review.json`, `ms7-native-order-green/` and `ms7-performance-audit.json`). No raw artifacts are committed and old binaries are not relabelled as rebuilt by a later stage commit.

### MS8 - Server groups, reselection and continuous responses

- [x] Wire explicit admitted/selected/parked states into existing queues, phase selection and output handling.
- [x] Test all phase transitions, new arrivals while decoding, different prefill completion times, allowance exhaustion and independently finishing slots.
- [x] Preserve the same stream and accepted-output cursor through repeated pauses; test SSE heartbeats, cancellation/disconnect, non-streaming results, stop strings, usage and tool-call fragments.
- [x] Keep live parked slots busy to administrative APIs and keep newly queued prefills out of the current decode group.

Exit: two-slot target-only public API behavior, context-boundary progress and no duplicate/lost/falsely finalized output.

Accepted implementation: an owned group manager reserves both canonical host address ranges before requests, keeps only selected native recurrent/SWA state on the common GPU parent, and parks native checkpoints in RAM between selections. Existing native cell metadata supplies compact shared attention columns; per-source resident prefixes, private page rings and bounded copy waves retain native masks, arithmetic and checkpoint order. Native workspace measurements admit a complete selection before the CPU controller commits it. A genuine joint quota shortage can select one slot; unsupported native geometry does not silently serialize. Logical allowances include the pending sampled anchor. Frontend samplers, response cursors, grammar, tool-parser state and request identities survive pauses.

Qualification covers both targets, all three existing full-server cache pairs, equal/unequal histories, allowance exhaustion, per-slot context ends, busy parked slots, late arrivals, independent stops, cancellation, seeded retry, usage, grammar and tool fragments. Actual native full-vocabulary controls cover selected-state parking/restoration and failed-peer isolation. CPU/GPU sampler chains initialize once and remain frontend-owned. A real compute-failure red on both models exposed an unfinished admission during retirement; explicit abort/discard now drains failed work before releasing frontend chains, and fresh paired requests recover exactly. All three native failure statuses are exercised before and after queued work. Retained KV bytes/frontiers stay correct, while cancellation advances its existing semantic generation exactly once.

A second real red exposed stock response-queue timeout starvation: other-request notifications repeatedly restarted a relative wait, preventing a paused stream's heartbeat/disconnect poll. A fixed steady-clock deadline resolves it. The new CPU queue test failed exactly one of 22 assertions before the fix and passes normally and under ASAN/UBSAN afterward. Actual production SSE generators emit heartbeat comments while a peer is paused, with both streams' 128 token IDs, text and usage matching their non-streaming controls.

The current host and ASAN inventories each pass 55 executed tests plus two expected CUDA-only skips. Thirty native/transport/neural component gates include seven strict default-PDL memchecks with zero errors, API reporting `all`, exit 99, no kernel filters and eager loading only for the checker. All64 D256/D512 reader cases, cached pending appends, one-page rings, wide transfers, queued-copy faults and the existing conversion/pipeline controls remain exact. Additional normal/ASAN neural cases prove an unselected peer's checkpoint survives selected-peer failure. Component reuse after the queue-only fix verifies every actually loaded project library, executable, relevant core source, input and profile; those clients do not link the changed server implementation. Current full-server tests separately qualify the updated queue. This is not a full-model CUDA memcheck or a non-CUDA hardware qualification.

Existing single-slot vision controls match the frozen V2 server in all six scoped modes, including changed/cached/multiple images. Ordinary-native two-slot JSON/SSE controls preserve stock unified-context reporting and outputs. The single-slot performance screen contains 192 requests and 21,504 exact generated IDs across a six-mode 64-output cohort and a separate focused Gemma 256-output cohort. A roughly 0.22% initial Gemma target-prefill decrease did not persist in the opposite-order follow-up; remaining small timing changes have mixed directions and overlap observed variation. No confirmed regression is established in these short-context profiles. No speedup, exact timing parity, heavy-context qualification or new multi-slot performance win is claimed; MS13 remains mandatory.

Local source patches, immutable runtimes, raw failures, classified fixture mistakes, commands, loader identities, models, inventories and audits remain under `benchmarks/results/multi-slot-20261109/`, particularly `ms8-qualification.json`, `ms8-core-reused-evidence.json` and `ms8-performance-audit.json`. They are not committed. CUDA arithmetic/kernel code is unchanged. The declaration remains target-only, text-only and cache-disabled for two slots until MS9-MS12 extend their own contracts.

### MS9 - Qwen and Gemma MTP

- [x] Batch supported MTP drafting and target verification without confusing aggregate rows with per-sequence widths.
- [x] Bind Qwen auxiliary caches and Gemma full/local shared readers to the correct slot, frontier and physical bank.
- [x] Qualify widths one through four, confidence cutoffs, unequal widths, each accepted prefix, parked/resumed source pins and target continuation.
- [x] Retain existing inherited cache-format, sidecar pairing and serial-workspace safety contracts while making their ownership sequence-aware.

Exit: native-matched MTP internals/public output/counters and unchanged single-slot controls.

MS9 qualification, 2026-11-11: the common initializer declares auxiliary roles from loaded-model metadata before arena admission. Qwen retains independent auxiliary KV and recurrent rollback banks per selected sequence; Gemma borrows the target's actual full/local source layers without independent KV storage. Both children borrow the existing parent, with hidden-output and sampler workloads measured before binding. Gemma verification widths remain independent of its zero rollback-snapshot count. Rebinding retains each logical slot's sampler, carry and counters, and occurs before the previous execution contexts retire. New-request reset clears carry; pause/resume does not. A failed queued MTP operation closes admission, clears proposals and produces an error rather than publishing a partial successful draft.

Current normal and ASAN/UBSAN controls cover Qwen's 37 accepted-prefix/depth combinations, GPU sampling, unequal widths, confidence/minimum/override behavior, real streaming, correction and target continuation. Gemma's 37 short accepted-prefix combinations pass 12,372 assertions in each build; the wrapped asymmetric-prefix case passes 1,827 in each. The additional Gemma long-reader fixture uses context 6,400, histories 3,073/3,075 and microbatch 32, with a measured 853,771,136-byte joint minimum. Normal and ASAN each compare 1,188 exact native records (149,394,640 bytes), with 105 actual assistant source acquisitions and protected full/local source state. Earlier context-8,192/microbatch-256 native allocation failures and context-6,400/microbatch-256 native attention-pool OOM remain preserved capacity findings. The default context-4,096/microbatch-256/1-GiB fixture is unchanged and passes separately. These are explicit additional profiles, not a claim that the failed native profiles fit.

Public ordinary-native versus streamed JSON/SSE controls pass both models at depths one through three: 48 completion records, 768 generated IDs, and exact draft/accept counters. Separate normal/ASAN headroom controls retain an open SSE response through reselection and enforce the individual context boundary. Cancellation of either member, independent grammars, tool-call fragments, fresh-request carry reset and queued backend failure/retry also pass. After cancellation the remaining native batch layout changes; only the matching pre-cancel prefix and matching retry are claimed identical, not arithmetic across different batch layouts. Two-slot vision, DFlash/K4V and persistence remain gated for their later stages.

CPU and ASAN inventories each contain 57 tests: 55 execute and two CUDA-only CLI tests skip as expected. The late Gemma additions changed only the context-test source and its two executables; both were freshly rerun, while exact source/header/library/executable hashes retain the other inventory results. Strict EAGER canonical-read and selected recurrent-publication components pass with zero errors, default PDL, graphs enabled where exercised, API reporting all, exit code 99 and no kernel filters. The publication fixture uses three real Qwen layer geometries, both selected banks and all three snapshots, including cancellation, proof isolation and exact retry bytes. Its CPU-mapped weights are a no-neural-computation component reference, not a full GPU-weight model test.

The full-model Q3_K_M sanitizer failed GPU weight allocation before any context or arena. The user-approved sanitizer-only Q3_K_S retry also failed its 12,583.79-MiB GPU weight allocation under EAGER loading, with two API OOM errors and exit 99; neither EAGER attempt is a memory pass. A no-model probe measured 8.77 GiB of total device usage with the strict EAGER backend, leaving 6.74 GiB free; the ordinary EAGER backend-only probe used 1.32 GiB. These totals exclude model weights and the arena and are not production-server footprint measurements.

The user subsequently approved LAZY loading for this sanitizer-only fixture. The frozen, matching MS9 client and libraries then passed 1,803 assertions, 36 full native-exact records (17,423,056 bytes), protected target-source and manager park/resume checks, and strict memcheck with zero errors in 26.494 seconds. All 38 before/after provenance pins match. This additional fixture retains GPU Q3_K_S weights, the Q8 assistant, Q8/Q4 KV, a 1-GiB arena, context 4,096, microbatch 256, default PDL, graphs, API reporting all, exit99 and no filters. It qualifies the preserved MS9 source/binary cohort, not later MS10 changes or Q3_K_M under memcheck. Ordinary and ASAN neural controls retain Q3_K_M and their original profiles. Production configuration was not changed. Original EAGER failures remain recorded; no numerical/API requirement or global OOM-prevention claim changed.

Existing single-slot vision, MTP, DFlash and DFlash/K4V controls pass, including genuine K4V hit/miss and feature maintenance, all verification widths, and additional Q8/Q8 and Q5_1/Q5_1 MTP checks. The six-mode matched short performance screen contains 36 processes, 144 requests and 9,216 exact output IDs. Prefill paired medians range from -0.14% to +0.12%. An initial DFlash7 decode result of -1.124% triggered investigation: an identical-baseline self comparison varied by +0.902%, and the reverse candidate comparison was +0.064%, without code edits. The initial negative result remains recorded; a stable slowdown did not reproduce. This is a stage regression screen, not the MS13 long-context/release performance qualification.

Evidence and original failures remain untracked under `benchmarks/results/multi-slot-20261109/`. `ms9-stage-audit/acceptance.json` maps 17 acceptance groups, checks the production patch against the immutable 31-file runtime, and retains the original MS8-plus-MS9-patch build identity. Test-only changes have explicit delta proofs. No CUDA kernel, conversion arithmetic or native reduction order changed; production configuration is untouched. MS10-MS14 remain required before the overall multi-slot feature is complete.

### MS10 - DFlash and DFlash plus K4V

- [x] Isolate feature rows, selected-layer inputs, convolution/selector state, draft KV and rollback per slot.
- [x] Preserve independently configured DFlash/K4V maxima one through seven; test mixed widths, source switches, every accepted prefix and boundary-shortened rounds.
- [x] Exercise genuine copy hits and genuine neural fallback per slot, including cancellation, park/resume and restored history namespaces.
- [x] Verify strict per-slot and aggregate generated/accepted/forward counters. A neighboring slot cannot supply missing proof.

Exit: exact ordinary-native internal/public controls, both routes exercised, and no K15 admission hidden in aggregate two-sequence row counts.

Completed: two independently bounded DFlash/K4V text slots share the existing phase owner and physical parent. The drafter consumes the selected target's five hidden-feature layers and writes its separately identified bounded cache. Admission validates complete native blocks, output regions, source epochs/generations and inherited cache formats before mutation. Known append/suffix changes preserve only proven source lineage; unknown replacements remain rejected. Driver rebinding retains per-slot maps and sampler state through parking and selection changes. Target rollback and K4V continuation remain limited to seven proposals; no K15 admission or private GPU-cache fallback is introduced.

TDD and numerical qualification include 49 independent width pairs (115 gates, 183,299 assertions), every accepted-prefix pair zero through seven (148 teacher/correction/next-draft gates, 252,264 assertions), and all 64 target rollback/correction/parking pairs (6,844 assertions each normal and ASAN). Genuine K4V copies and neural fallback execute independently for both slots, retain maps through parking, and compare full visible draft KV and canonical selector scores without float tolerances. Joint copied-token verification matches all 16 native logit vectors and five hidden-feature layers (21 records, 17,530,880 bytes), then maintains both neural histories without an extra prediction. Source/token/position mismatches, stale generations, empty/reset histories and populated peers remain guarded.

The user approved staged ordinary-native target/draft controls for seven-proposal modes: the simultaneous native recurrent allocation alone is 2,394 MiB and does not fit with the chosen GPU weights. Shared embedding/output-head placement, weights, features, positions, checkpoints and exact-vector comparisons remain unchanged. A fitting simultaneous native DFlash3 pair independently reproduces 15 target invocations, ten draft dispatches, both complete public responses (32 IDs) and per-slot counters under matching serial-prefill/selection/checkpoint protocol. Stock native HTTP mixed-prefill and serial phase-arena prefill have different native layouts; the original unmatched-protocol failure remains preserved and is not relabeled full native-server equivalence.

Public qualification covers DFlash1/3/7 and DFlash3/7 plus K4V7 with exact JSON/SSE tokens and counters, cancellation of either slot, independent grammars, fresh retries and same-key/different-value request namespaces. Changed K4V histories match fresh-process controls; original/changed/original requests retain exact original tokens and counters. A real queued-failure red exposed partial DFlash proposals followed by an abort during normal cleanup. Failed grouped prediction now clears partial neural proposals and reports an exception; only same-source metadata rebinding permits retirement, while failed live contexts remain unusable. Actual queued prediction and teacher failures return errors to both affected requests and recover with exact tokens/counters; the mixed-partition recovery passes normal/ASAN and strict default-PDL memcheck with zero errors, API reporting all, exit99, graphs and no filters.

The regression gates also found a genuine single-slot block-layout change: per-request clipping shortened the established seven-token neural block used by a three-token continuity probe. Clipping now applies only to multi-slot DFlash2, preserving the original single-slot selector geometry and numerical order. The restored P1 native/reference/stream control passes 4,480 assertions, with 2,634 ASAN import assertions; fresh K4V route and copied-target controls pass normal/ASAN and strict memcheck. The mapped Qwen/Gemma MTP controls pass, both portable inventories pass 55 executed tests plus two expected CUDA-only CLI skips, and all six existing single-slot vision modes retain exact tokens and prompt/cache/speculation counters.

The existing-mode performance screen contains 36 processes, 144 requests and 9,216 exact IDs against the frozen MS9 own-library runtime. Initial small consistent Qwen MTP3 and DFlash/K4V decode decreases prompted identical-baseline and opposite-order controls, followed by a separate 256-output cohort (12 processes, 48 requests, 12,288 exact IDs). Longer MTP3 decode changes were -0.60%, +0.83% and +0.19%; DFlash/K4V changes had mixed signs and a +0.54% paired median. A stable slowdown did not reproduce. These are scoped short-context regression observations, not speedup, exact timing-parity or full MS13 qualification claims; contexts, arenas, models, KV types and sampling remain matched within each cohort.

Local evidence, failed attempts, staged-reference labels, immutable runtimes and source mappings remain under `benchmarks/results/multi-slot-20261109/`; `ms10-stage-audit/acceptance.json` records the 18 audited cohorts and 2,977 checked files. An accidental reboot interrupted only the final protocol-controller hash closure after all ten cases had terminated successfully; a separate audit checked all 2,174 original pins, outputs, commands and logs without overwriting the original artifacts. The final immutable runtime retains its MS9-plus-MS10-patch build identity. No CUDA kernel, arithmetic, reduction order or production configuration changed. MS11-MS14 remain required.

### MS11 - Vision groups and image-aware speculation

- [x] Encode every admitted request's required still images before group prefill, retaining exact embeddings/positions in host storage.
- [x] Qualify Qwen/Gemma target-only/MTP and the existing Qwen DFlash/K4V image combinations, multiple/interleaved images, ratio extremes and budget rejections.
- [x] Preserve and restore cached text/image histories across exclusive vision loans without keeping unchosen current states in VRAM.
- [x] Prove image-aware K4V boundaries, DFlash source-feature validity and independent state restore for each slot.

Exit: all scoped vision modes pass exact fitting native/component controls and actual end-to-end streamed server tests, with bounded resources and correct reference labels.

MS11 acceptance, 2026-11-12: the server freezes the admitted request group, encodes all required still images before any member begins prefill, retains immutable host embedding views, performs serial image/text prefills and then selects joint decode work. Vision borrows the same physical phase parent while all current target/draft states are parked in RAM. Encoder weights, graph, operator and library grants retire before text admission; retained device references still close admission. Missing prepared outputs cannot trigger a new encoder during prefill. Existing single-slot paths retain their own encoder and owner checks.

The Qwen MTP driver now validates image rows and shifts hidden inputs independently per sequence. Grouped catch-up removes target-only output flags; the single-slot path preserves its original behavior. DFlash retains the existing exact image source proof, including original embedding bits, spatial positions, request epoch and cache generation. Its native policy still omits multirow pinned images from draft KV. Target physical coverage and the committed native DFlash position are separate, survive parking, and govern exact teacher injection, query positions and provisional trimming. One-row image injection remains native-compatible. Gemma assistants retain the existing shared read role, independent verification width and no private KV fallback.

Fitting numerical controls compare complete native target/assistant vectors, encoded KV and recurrent/SWA state, proposal IDs and canonical selector score bits. Native checkpoint reconstruction matches the selected layout; a preserved native direct-versus-restored difference demonstrates why unmatched physical layouts are not interchangeable reference protocols. Tests cover full-parent overwrite between parks, independent hidden carries, image-tail injection, direct image prediction, a populated peer, provisional draft restoration, changed/multiple images, landscape/portrait ratios, cancellation and completed-GPU encoder failure with peer continuation and exact retries. Genuine image K4V copies and neural fallback both execute; image boundary/key tests and separate request generations remain enforced. The additional Q8/Q8 and Q5_1/Q5_1 image profiles pass alongside Q8/Q4. These public protocol checks are not relabeled full ordinary-native server equivalence.

The audited evidence contains 29 cohorts and 2,741 checked artifacts, including 14 strict CUDA logs with zero errors and unchanged default PDL, graphs, API reporting all, exit99 and no kernel filters. EAGER component and approved LAZY full-model profiles remain distinct. Normal vision uses the existing Gemma Q3_K_S target and 2 GiB arena; its approved sanitizer-only arena is 1,536 MiB. Full normal and ASAN/UBSAN host inventories each contain 57 tests: 55 executed and two expected CUDA-only skips. Controlled paired HTTP arrival proves both images precede first prefill and joint decode in each scoped model/mode. The test-only arrival/failure adapters do not change tensor arguments, arithmetic or backend synchronization.

Existing single-slot vision matches committed MS10 in all six modes: 84 responses, 1,344 exact IDs and exact prompt/cache/draft counters. The matched short text screen contains 36 processes, 144 requests and 9,216 exact IDs using separate immutable own-library runtimes, the same natural SWE input, 4 Ki context, 1,333 prompt tokens, 64 outputs, 1 GiB arena, Q8/Q4 and four distinct P cores. Decode paired medians range from -0.161% to +0.072%; every mode has mixed-sign pairs. No consistent decrease reproduced in this scoped cohort. Small differences are retained, not advertised as speedup or exact timing parity; the MS13 broader matrix remains mandatory.

Original failures are retained, including missing causal/catch-up/image-source contracts, native versus physical position assumptions, a missing fixture provisional trim, a raw native cache observation lacking producer completion under instrumentation, ASAN preload ordering, and server recovery incorrectly treating an already-retired parent as a retained loan. Numerical comparisons and runtime profiles were not relaxed to pass them. The original immutable MS11 runtime retains its MS10-HEAD-plus-qualified-patch build identity. Local evidence, commands, failure records, model/library hashes and the stage audit remain under `benchmarks/results/multi-slot-20261109/` and are not committed. Source/library/resource priming is not a guarantee against external driver or graph-executable OOM. No CUDA kernel, arithmetic, reduction order, production model or service configuration changed.

### MS12 - Cache, persistence and finder integration

- [x] Qualify prefix reuse, changed requests, low/high/low histories, checkpoint selection, independent save/restore/erase, restart relocation and slot-ID reuse.
- [x] Extend finder/preflight commands, host tests and reports for explicit two-slot context semantics, all legal selection shapes and automatic mmproj ratio probes.
- [x] Validate final chosen arenas with representative real joint and serialized workloads, including long-history/shape transitions. Keep upper-neighbor rejection evidence.
- [x] Measure committed/pinned/parked/checkpoint RAM and establish whether a hard host-memory policy requires a user decision.

Exit: faithful sizing claims, real output examples and lifecycle coverage; no claim of global OOM prevention from a bounded probe.

MS12 acceptance, 2026-11-13: parked checkpoints retain native recurrent/SWA/draft state and canonical cell metadata with checked prefix lifetimes, without copying the complete target history at each checkpoint. Versioned slot files and RAM-cache entries carry self-contained host KV planes; compact checkpoint appendices attach to validated restored prefixes. Model signatures check structural compatibility, not cryptographic weight identity. Malformed native and MTP payloads are rejected before a successful file-restore response. Same-layout cached continuations preserve exact tokens and prompt/cache/draft counters across save, erase, restore and process restart. Independent native placement controls explain why different physical slot layouts are not interchangeable exact-output references. Cached images are encoded only when outside the reused prefix.

The two-slot finder checks both individual members, joint decode and over-budget serialized selections, including actual server context limits and seven paired image ratios. The measured 4 Ki fixtures accepted/rejected 2,516/2,517 MiB for Qwen MTP3, 2,480/2,481 MiB for the approved Gemma vision MTP profile and 2,290/2,291 MiB for Qwen DFlash7 plus K4V. Each retained the one-MiB upper-neighbor rejection and final selected-arena validation with a common 256-token image setting. These are workload-specific fits, not maximum-context or global OOM guarantees. A 16 Ki/40 Ki Qwen MTP fixture proved actual ring activity and measured about 1,598 MiB committed/registered backing, 299.69 MiB parked native payload and 602.94 MiB checkpoint payload; steady RSS was about 4.54 GiB. Payload counters exclude allocator/container overhead, and no new hard host-memory cap is imposed.

The local stage audit checked 47 cohorts and 3,285 artifacts. Current-source normal and ASAN/UBSAN inventories each passed 55 tests with two expected CUDA-only skips; twelve GPU checkpoint/vision-loan gates, including six strict zero-error memchecks, passed. Final mapped MTP/DFlash/K4 normal and ASAN gates and six public cache/image/file/restart sanitizer cases passed. Ten new finder host tests and the 31-test existing finder suite passed, with one existing platform skip. Exact six-mode single-slot image and text comparisons remain separate from performance claims: small short decode decreases were investigated with baseline-self, reversed ordering and longer-output MTP cohorts, which had mixed signs and did not reproduce a stable decrease. Historical cohorts retain their original source/runtime identities; final current-source controls cover later shared changes. Original failures remain preserved in the local evidence. MS13 broader release qualification remains mandatory.

### MS13 - Full safety and performance qualification

- [x] Rebuild affected server/tests and run appropriate host, ASAN, CUDA and strict default-PDL memcheck coverage with unchanged API reporting and numerical requirements.
- [x] Keep strict memory gates at API reporting `all`, error exit code 99 and no kernel filters; eager loading is a sanitizer profile detail, not a production change. Preserve CUDA graphs in gates that explicitly test graph capture/replay.
- [x] Requalify existing single-slot Qwen/Gemma, all64 reader guards, MTP, DFlash/K4V and vision paths affected by shared changes.
- [x] Close every applicable stage regression entry against the frozen V2 controls, including ordinary non-streaming behavior and the established prefill conversion/transfer optimizations. Compare each supported mode separately; do not average a regression away across models or modes.
- [x] Run matched uninstrumented comparisons for two short fitting histories, unequal fitting histories, over-budget serialized selections, growth-triggered reselection and image groups.
- [x] Include short/medium/heavy-streaming contexts, cold/warm histories and one finished/cancelled member. Use identical model/sidecar/projector/input/sampling identities within each comparison.
- [x] Report aggregate accepted tokens/s, per-request active and paused latency, group TTFT, phase/selection counts, park/restore time, H2D/D2H bytes, resident/ring ownership and memory peaks. Separate intentional blocking policy from unintended compute regressions.
- [x] Confirm shared launches and transfer overlap with profiling where available; instrumented throughput is not speedup evidence. Do not assume batching wins when PCIe/state transport dominates.

Exit: no confirmed functional, numerical, safety, compatibility or single-slot performance regression; correctly accounted resources and capacity; correct scope for native fitting versus long streamed-only evidence; current-source safety passes and transparent multi-slot performance results. An explained regression still requires a fix/revert or an explicit user decision to change scope; it is not a passing non-regression result. A new full-context sweep needs its own explicit scope; it is not implied by writing this roadmap.

MS13 acceptance, 2026-11-13: fifteen fresh CUDA operator/reader/prefill/pipeline gates passed 111,621 assertions, including seven strict zero-error memchecks with API reporting all, exit99, no filters, default PDL and graphs retained. The full D256/D512 all64 matrices, pending appends, ring wraps, shared waves, fault paths and established prefill conversion/transfer controls passed. The MS12 normal/ASAN inventories, checkpoint/vision-loan memory gates, mapped MTP/DFlash/K4 controls and six-mode single-slot vision checks retain identical relevant C++/CUDA source and libraries; their timestamps are not relabeled. No C++ or CUDA production content changed after MS12. The finder response collector's Python type narrowing was corrected and its new host tests normalized to four-space indentation with an identical syntax tree. Focused type checks, ten new finder tests and the 31-test existing finder suite pass with the existing one platform skip. The tracked-wide Python check still has dependency/type diagnostics in untouched files and none in a feature-changed file; it is not called a clean full-repository type-check pass.

Direct original-V2 versus final-runtime comparisons covered all six supported text modes at C=4,096/16,384/65,536 with 1,333/8,192/40,960 natural prompt tokens, 64 outputs, the same 1 GiB arena, Q8/Q4, four distinct P cores and own-library runtimes. Sixty processes produced 216 requests and 13,824 output IDs with exact paired tokens and draft counters. Nine flagged timing distributions received identical-baseline, reverse-order and forward-order controls. Original small decreases remain in the ledger. Most changed sign under ordering; the tiny Gemma MTP prefill differences were not distinguishable from the measured identical-baseline change. The negative Qwen MTP 64-output heavy medians received a separate same-model/history/context/arena 256-output cohort: three alternating-order pairs, eighteen requests and 4,608 exact IDs, with mixed-sign changes and a small positive decode paired median. These data did not establish a sustained code-induced decrease; they do not establish exact timing parity for the original short decode. Workload identities, timing variation and qualification limits remain explicit. Ordinary non-streaming Qwen/Gemma unified-slot context reporting, fixed-slot prefix reuse and save/erase/restore continuations match the original V2 tokens and counts. An initial fixture incorrectly assumed divided context under `--kv-unified`; its baseline failure is retained and the assertion was corrected to the original full-C report.

The uninstrumented six-mode group matrix contains 66 cases: one-slot sequential and two-slot equal/unequal fitting histories, over-budget serialization and paired images, plus two-slot growth, long histories and cached prefixes. Each case has a warmup and two measured trials, with per-request first-token/completion timestamps, output and cache/draft counts, and host/GPU memory observations. Growth includes a surviving member after its peer finishes; existing current-source cancellation/failure controls remain separately scoped. The measured equal-short two-slot aggregate rates were about 0.28-0.86 times serial rates at the declared 1 GiB profiles. This is a measured new-mode performance tradeoff; no single-slot acceleration or speedup claim is made. Different native batching/placement and speculative work counts remain visible rather than being relabeled identical-work numerical references.

Separate test-only lifecycle/NVTX hooks and CUDA graph-node traces measured activation/admission/restore, parking, selected and unselected wall intervals, transfer bytes, copy/kernel overlap and actual resident/ring ownership. The heavy trace has two-member decode and actual streaming; growth changes between two and one selected members. Active/transition/unselected values are instrumented wall accounting, not pure GPU execution or uninstrumented throughput. Inference-range traces exclude model upload and show copy/kernel overlap. No product math, GPU synchronization, graphs or memory checks were weakened to collect them. An initial trace omitted DEBUG ownership lines at verbosity4; the independently retained verbosity5 trace supplies them. An initial SSE collector rejected a valid null begin marker; only the collector was corrected and the failed run is retained. The final local audit checked 21 cohorts and 2,773 artifacts. Runtime qualification is Linux RTX 5070 Ti only; unavailable Windows/HIP/MUSA/other-card builds and runtime checks are not called passes. No new full-context sweep, production/service/model configuration change, delegation or push was performed.

### MS14 - Grouped speculative restore overlap and adaptive KV throughput

#### Current findings and chosen implementation

Accepted follow-up, 2026-11-15: preserve original single-slot decode throughput while making two-slot decode at two 64K histories comparable to or faster than validated double buffering. The retained implementation is the aggregate-layer adaptive ring. The historical checkpoints below describe the earlier investigations and their own accepted scopes.

The investigations found three additional costs after restore overlap was implemented:

- The 32K-to-64K drop crosses actual resident capacity. Both 64K histories stream KV; committed columns are already dense, so logical headroom offsets do not explain the transfer volume. Small per-slot rings and less cross-layer pipelining add overhead beyond the combined live KV footprint.
- A two-tail minimum improves grouped prefetch but consumes serial residency: separate near-128K controls measured regressions of 2.43% for MTP3 and 4.43% for DFlash7. Aggregate-policy experiments also exposed physical ring fragmentation and repeated streaming of the one-layer owned MTP prediction cache after resident capacity was spread across layers. Repeated prediction calls have no next owned layer with which to hide those transfers.
- A later aggregate candidate transfers fewer KV bytes but averages 6.75 native attention waves per 16-query layer versus 5.0 for double buffering. Its 11.95% DFlash7 regression is removed by combining each resident prefix with its following bounded streamed chunk. The 96-page release cadence is preserved so next-layer prefetch can start promptly.

The chosen implementation keeps the original serial scalar controller and feeds grouped execution the sum of page-padded request extents. Attention-layer count reflects owning cache roles, not the number of requests. One shared ring serves source-aware fragments, and resident planes are sliced by live page demand. The combined current tail can pipeline with a partial next layer; two complete tails are not an unconditional minimum. Deadline feedback can still enlarge the ring when prefetch falls behind. The double-buffer policy remains an internal opt-in reference.

The executor keeps cyclic FIFO placement, separate upload-event fences, pending-row producer fences and one joint readiness sample before attention. When feasible within the existing resident quota and current-tail ring bound, the repeatedly used owned prediction-cache role stays fully resident. Layout changes drain affected readers before mirror invalidation and view replacement. Stale layouts and unprepared page growth fail before publication or reads. Registered copies authorize only the declared disjoint host source ranges. Authoritative host histories, one fixed GPU arena, native arithmetic, masks, column order, rollback and logical scheduling remain unchanged. MS14 leaves group admission and single-source prefill scheduling unchanged; MS15 extends them as recorded below.

Final throughput on the RTX 5070 Ti is in tokens/s. Serial columns compare original serial -> retained implementation; parallel compares validated double buffer -> retained implementation and counts both slots over one wall interval.

| Qwen mode | Serial 8K | Serial near 128K | Two 64K slots, total | Parallel change |
| --- | ---: | ---: | ---: | ---: |
| Target-only | 46.61 -> 46.76 | 26.56 -> 26.53 | 43.91 -> 44.55 | +1.45% |
| MTP3 | 113.28 -> 113.64 | 63.96 -> 63.98 | 64.38 -> 70.29 | +9.19% |
| DFlash7 | 149.08 -> 149.31 | 72.06 -> 72.20 | 68.22 -> 69.90 | +2.47% |
| DFlash7+K4V | 167.68 -> 167.91 | 76.56 -> 76.73 | 71.69 -> 72.99 | +1.81% |

Each runtime has two uncached trials of 512 output tokens per slot on one native-tokenized Open-SWE trace, with separate warmup and the common 32..480 output-token wall interval. History lengths are 8192, 130560 and two times 65536; the combined context budget is 262144. Both variants in each comparison use the same fitted arena: serial target/MTP/DFlash/K4V use 2690/2308/2106/2092 MiB, and parallel uses 2856/2510/2282/2282 MiB. References precede candidates; no reverse-order trials were added. All output token IDs and draft counts match within each comparison, every parallel trial proves actual two-member decode, and all source/runtime/input/model pins close. These are one-trace results on one GPU; the broader context sweep and stopped Gemma investigation retain their separate evidence and limits.

Qualification passes 12 GPU/model gates, strict zero-error CUDA memcheck including native bit comparisons and the full-model MTP path, and normal plus ASAN/UBSAN host inventories with 56 executed tests and two expected GPU-only skips each. Preserved test-first failures cover scalar serial-policy compatibility, aggregate layout/planner routing, disjoint registered-copy scopes, FIFO ring placement, prediction-role residency, stale layouts, unprepared growth and the final prefix/stream attention wave. The last native red fails on the earlier candidate; its strict green passes 264 assertions with bit-identical controls. Earlier rejected experiments keep their original artifacts and are not relabeled as accepted runs.

Local evidence under `benchmarks/results/multi-slot-20261109/`: `aggregate-kv-policy-tdd-v1/` contains the red/green controls, `aggregate-kv-candidate-v9/` freezes source/runtime identities, `aggregate-kv-benchmark-v9/comparison.csv` records all twelve comparisons, and `aggregate-kv-final-qualification-v2/acceptance.json` maps final qualification and provenance. Documentation updates occur after the measured implementation is frozen; frozen artifacts remain immutable. The current implementation, tests and this description are included in the user-requested MS14 follow-up commit.

#### Earlier scope and investigation history

User direction, 2026-11-13: determine why enabling MTP/DFlash2 loses the aggregate decode gain seen with target-only execution, and determine the solution. The previous documentation/release task remains deferred.

Final user direction: close MS14 with the implemented restore-overlap optimization and its TDD qualification; further decode-speed optimization is outside this stage. The remaining aggregate MTP deficit is retained as a known limitation. The user will propose MS15 separately; no allocation-policy or other optimization scope is inferred for that stage.

- [x] Establish a longer uninstrumented shared decode-window baseline for target-only, Qwen/Gemma MTP3, Qwen DFlash2 and DFlash2 plus K4V. Count both slots over the same wall interval; exclude prefill, initial decode setup, and unequal completion tails explicitly rather than summing per-slot timers with different denominators.
- [x] Use identical model, input, sampling, cache format, context, output count and arena within each one-slot/two-slot comparison. Keep the two histories eligible for joint execution throughout the measured window. Retain separate end-to-end/TTFT results and actual speculative acceptance/work counts.
- [x] Attribute draft generation, target verification, acceptance/rollback, source and checkpoint publication, H2D/D2H/D2D transfers, phase/selection activation, graph capture/replay and synchronization. Distinguish intrinsic extra work/changed acceptance from implementation overhead using matched one-slot and two-slot traces.
- [x] Test competing causes with controlled experiments: proposal width, acceptance rate, recurrent/SWA rollback state volume, source publication and restore granularity, context/graph reuse, draft batching and arena pressure. A different arena is a separately labeled paired experiment at the same budget for both variants, not an unreported concession.
- [x] Identify and address the measured grouped restore-scheduling cost with a bounded implementation. Record unresolved snapshot/publication and proposal/verification costs without claiming the remaining MTP deficit is solved. Preserve the one-parent arena, selected-state ownership, native arithmetic, exact rollback and isolation, and existing phase/group scheduling policy.
- [x] Validate the selected solution or bounded proof of concept with repeated alternating-order common-window decode measurements and actual emitted token counts. Report aggregate speed versus the same-mode single-slot baseline, plus per-slot speed, acceptance, memory and transfer traffic. A solution claim requires evidence beyond an instrumented profile or a smaller amount of required work.
- [x] Preserve mapped legacy controls and pass final-source numerical, memory, lifecycle and affected image-path checks for the delivered change. Retain precision, memory/API checks, graph/PDL settings, input/model identities, output semantics and unsupported-mode guards. Keep earlier evidence identities and broader release limits explicit.
- [x] Record unresolved limits and explicit next implementation steps. If achieving the target requires changing an agreed ownership, scheduling or reference contract, present measured alternatives for a user decision rather than silently changing scope.

Exit, as accepted by the user: the measured restore-overlap implementation, source-generation guard and no-pending fast path are reviewed and qualified, with numerical/state/lifecycle correctness and transparent performance evidence. The original aggregate accepted-token target remains unmet for Qwen MTP3 and is accepted as a limitation at this stage's closure. Intentionally serialized over-budget cases and broader release claims remain separately scoped. No new full-context sweep, live production changes, delegation or pushes are authorized by this closure.

MS14 investigation checkpoint, 2026-11-13: the unchanged runtime was measured with 512 outputs and a common 32..480-token window. Aggregate/single decode ratios were 1.64 Qwen target-only, 0.764 Qwen MTP3, 2.32 Qwen DFlash7, 2.13 Qwen mixed K4V, 1.62 Gemma target-only and 1.49 Gemma MTP3. The DFlash ratios include much less speculative work per output, so they are not pure same-work batching gains. Qwen MTP generated more drafts and accepted fewer; early continuation differences confound a predictor-bug conclusion.

Matched target/draft CUDA traces show useful GPU batching (target kernel time per query row 6.689 to 4.573 ms, MTP draft 0.782 to 0.413 ms), but H2D traffic moves from drafting onto target verification. Total H2D per accepted output rises from 39.85 to 72.62 MiB. The existing recurrent-restore overlap path names serial contexts and checks sequence zero; its disjointness guard assumes target recurrent state is outside the serial draft arena. The grouped path needs selected-source readiness and protected non-overlapping parent-arena restore views, then asynchronous restore/prefetch during drafting. It must not bypass the guard or add private GPU state. Existing multi-sequence restore/publication banks are the reusable implementation base. A paired 2 GiB arena did not materially improve MTP3; reduced depth1 produces only a small same-depth gain and remains slower than the original single-slot MTP3, so neither is called the solution.

The concrete implementation and validation plan is recorded locally in `benchmarks/results/multi-slot-20261109/MS14_FINDINGS.json`. Source/runtime hashes close for all three initial diagnostic cohorts, which used unchanged C++/CUDA code. Existing documentation/release closeout remains deferred.

MS14 bounded prototype checkpoint, 2026-11-13: the existing persistent recurrent-state lease is checked against every draft lease before selected restores are queued. Early submission improves DFlash7 by 4.2-4.4% but leaves MTP flat: the bulk restore precedes MTP's own attention reads. Moving MTP submission after its final draft attention increases observed H2D/kernel overlap from 2.5% to 21.7%. Uninstrumented reversed-order pairs improve aggregate MTP3 from about 72.2 to 74.1 tokens/s (2.5-2.7%); the final DFlash7 pair improves 99.98 to 104.15 tokens/s (4.2%). Every paired token stream and draft counter remains exact. The 1-GiB arena and actual GPU memory are unchanged. This is a bounded short-context prototype, not a full-context sweep or release qualification.

MTP remains below its approximately 94-token/s single-slot control and the approximately 79-token/s two-slot target-only control. Recurrent snapshot publication/transfer volume and additional proposal/verification work remain unresolved; group next-consumer KV prefetch has not been implemented. The original aggregate-throughput target and MS14 completion remain open. Local code also fixes same-source metadata retirement after a queued MTP error while rejecting foreign request sources. The initial failed tests, frozen-runtime reproductions, exact test-only fixture deltas, mapped qualification and timing evidence remain under `benchmarks/results/multi-slot-20261109/`; `MS14_PROTOTYPE_RESULTS.json` records the current bounded evaluation. No live production service, model configuration, CUDA kernel, native arithmetic or safety setting was changed.

Bounded qualification passes 23 component GPU gates, normal and ASAN/UBSAN portable inventories with 55 executed tests and two expected CUDA-only skips each, and six strict zero-error memchecks including public failure after an actual restore enqueue. The public check returns HTTP 500 and recovers exact tokens and draft counts at the same native physical-slot layout. Strict checks retain API reporting all, exit99, graphs, default PDL and no filters. Nine cohort hash sets close over 15,767 pins; original failures, timeout and harness-shutdown attempts remain recorded. This qualification does not close the remaining MS14 performance or broader release requirements.

User clarification: the next TDD work targets these measured restore-overlap optimizations; it does not change KV quotas. The original implementation was a prototype followed by qualification, not a completed test-first MS14 implementation. The first regression-hardening slice adds dedicated MTP attention-order and DFlash prefetch tests. Three controlled missing/early-submission implementations fail the intended assertions; the measured prototype passes eight normal/ASAN GPU controls, two strict zero-error memchecks and both affected host tests in each build. The original asynchronous MTP gate remains separate. A stale DFlash unknown-generation negative was independently reproduced on the frozen runtime and corrected to require the explicit error, empty proposals and unchanged exact KV. `MS14_TDD_STATE.json` records the 13-case red/green cycle. Production source and KV allocation are unchanged; this step adds regression protection and does not relabel the earlier development history or close MS14.

The next actual test-first runtime change reproduces one stale-source approval on the unchanged prototype before modifying production code: a target cache can change semantic generation while keeping request identity and history length. Restore authorization now requires the completed target projection to remain current; the draft projection may still have an active append. Green-stage refactoring skips full isolation/source scans when no restore is pending. Both slots, all native MTP prefixes, existing Qwen/Gemma/K4/DFlash controls, 20 GPU gates, three strict zero-error memchecks and both portable inventories pass. Fixed-budget timing controls retain exact tokens/counters and unchanged memory. Final MTP paired deltas are -0.86%, -0.20% and +0.17%, with +0.31% variation in a prototype-self control; no statistical equivalence or new speedup is claimed. Aggregate throughput remains approximately 74 MTP and 104 DFlash tokens/s. Earlier observations remain recorded. `ms14-tdd-source-guard-v2/acceptance.json` maps the red, implementation, refactor, qualification and controls. KV allocation policy and the unresolved MS14 aggregate target remain unchanged.

MS14 final acceptance: the user accepted the delivered scope and closed the stage. Six additional final-source MTP/DFlash image compatibility gates pass, including two strict zero-error memchecks, using the existing native embedding fixtures. These cover MTP image continuation and full-parent overwrite, and DFlash image peer isolation, in normal and ASAN/UBSAN builds. Together with the final 20 text/state GPU gates and both 55-executed-test portable inventories, they qualify the delivered code; historical passes retain their original source/build identities. The unmet MTP performance target, small mixed timing deltas, unimplemented group next-consumer KV prefetch, unchanged KV quota policy, missing hardware and deferred release documentation remain explicit limits. `ms14-stage-audit/acceptance.json` records closeout. No further decode optimization is required for MS14 closure.

MS14 reopened by the user, 2026-11-14: continue investigating until a measured solution to the speculative aggregate-throughput deficit is found. The new unchanged-runtime Open-SWE baseline uses a 256K combined context budget, a 1-GiB arena, Q8/Q4 KV, DFlash depth seven, 512 outputs per member and two measured trials. At equal 8K histories, serial decode is150.7tokens/s and the two-slot aggregate is99.6tokens/s; draft acceptance is69.0% versus70.5%. At16K, serial is159.2tokens/s and aggregate76.3tokens/s, with75.4% versus76.2% acceptance. Thus lower overall draft acceptance does not explain these deficits. The larger sweep was explicitly paused by the user. These are one-trace results and use the common32..480 output-token wall interval, not prefill or unequal per-slot timers.

- [x] Attribute the 8K slowdown with matched CPU/CUDA traces; confirm the resident/ring mechanism at 16K.
- [x] Run controlled counterfactuals, retaining exact token/draft work for same-work claims and labeling memory/proposal changes separately.
- [x] Add reproducing tests before selected runtime fixes, preserve native arithmetic and ownership, and qualify affected paths and serial controls.
- [x] The user accepts the retained implementation as meeting the reopened MS14 target. Short-MTP and fixed-1-GiB 16K limits remain recorded. Final closure awaits the newly requested near-maximum-arena serial/parallel sweep.

The selected fixes remove transport and scheduling overhead: reuse resident KV across validated appends/rollback, refresh only live columns, gather exact encoded bytes in bounded CUDA launches, publish GPU-encoded rows directly to resident mirrors while canonical host copies finish, queue resident consumers on the owning backend, and reuse bounded native SET_ROWS graphs across recreated views of the same owned allocation. Ring waves preserve native column/reduction geometry and protected reuse. MTP catch-up uses the decode transport path without changing its native graph phase. There are no floating-point kernel, KV quota, logical-context or proposal-policy changes.

Final repeated uninstrumented results use one native-tokenized Open-SWE trace, 512 output tokens per request, separate warmup, and the common 32..480 output-token wall interval. Aggregate throughput is counted within each common interval before taking medians. Both 8K/16K serial and paired runs use the approved 256K combined context budget. The separate short diagnostic uses 513-token histories and a 4K context budget.

| Mode / history per slot | Arena MiB | Serial tokens/s | Parallel total tokens/s | Parallel / serial |
| --- | ---: | ---: | ---: | ---: |
| Qwen MTP3 / 8K | 1024 | 113.799 | 151.558 | 1.332 |
| Qwen DFlash7 / 8K | 1024 | 149.361 | 187.868 | 1.258 |
| Qwen DFlash7 / 16K | 1024 | 158.612 | 148.531 | 0.936 |
| Qwen DFlash7 / 16K | 1536 | 158.246 | 189.227 | 1.196 |
| Qwen MTP3 / 513 | 1024 | 94.179 | 93.779 | 0.996 |

The 8K DFlash old-runtime aggregate is 99.352 tokens/s in the final control cohort; original pre-investigation measurement was 99.602. All 512 output IDs and draft counters match old/new within the same parallel layout and trial. Native serial versus paired geometry may produce different continuations. At 8K, MTP draft acceptance is approximately 90.99% in both layouts; at 513 tokens it is 69.2% serial versus 41.9% paired. The short paired trace performs more verifier queries and approximately 2.2 times the recurrent-state H2D bytes per accepted output. Its MTP compute cannot hide all of those transfers under the current acceptance/catch-up order. Recurrent staging-depth increases and proposal suppression do not fix it; suppression reduces aggregate throughput to approximately 73.6 tokens/s and is rejected. A separate exact-work diagnostic moves accepted-prefix selection before MTP catch-up and starts isolated restores either after attention or immediately after GPU input staging. The final variant measures 91.55 / 92.10 / 91.85 tokens/s versus its 92.95 control; all 512 IDs and draft counters match. It is rejected and introduces no production scheduling changes. Earlier no-op eligibility probes and a loader failure remain separately classified. The final provenance write was interrupted; an independent full recomputation matches all original pins and is recorded in `ms14-reopened-catchup-order-screen-v5/provenance-after-recovered.json`. Increasing the 16K arena to 1536 MiB avoids the measured residency deficit but costs 512 MiB; it is a separately labeled memory counterfactual, not a same-budget solution or MS15 implementation.

Final unchanged-source qualification passes 49 GPU gates with 97,019 assertions, including 15 strict zero-error CUDA memchecks, and normal plus ASAN/UBSAN portable inventories with 55 executed tests and two expected GPU-only skips each. Actual CUDA byte-copy instrumentation, graph-view eviction, queued producer/consumer lifetime, native text/image continuation, all MTP acceptance prefixes, restore order/source guards and failure recovery are covered. Strict checks retain API reporting all, exit99, graphs/default PDL and no filters. The original additional-gates launcher-only exit127 is retained and resolved by separately recorded corrected runs; it is not counted as an executed GPU check.

Evidence under `benchmarks/results/multi-slot-20261109/`: `ms14-reopened-final-qualification-v1/acceptance.json`, `speed-comparison.json` and `speed-comparison.csv`; the same directory's `qualification/`, `additional/` and `additional-followup/` contain raw gates and source/runtime pins. The final normal runtime is byte-identical to the measured writer-view-cache candidate; frozen server/common build metadata remains truthful. Rejected conversion-workspace, ring-lookahead, staging-depth and proposal-policy experiments retain their original artifacts. These results cover one trace and one GPU, not dataset-wide performance.

The user accepted the retained implementation as satisfying MS14 and requested a new serial/parallel sweep before closure: 8K, 16K, 32K, 64K, 96K and near-128K per slot, using the current card's near-maximum arena and reporting each model/mode pair as soon as both runs finish. The six supported text modes are included; Gemma DFlash modes remain unsupported. The existing finder determines each mode/path's physical maximum; the user explicitly chose each path's own maximum minus 16 MiB, with both serial/parallel arena sizes reported. Their external allocations differ; initial failed serial launches at the parallel ceiling are preserved and superseded before measurement. This request resumes the larger sweep. MS15 remains planned and unimplemented. No production deployment, push or delegation is authorized by this reopening. The prior MS14 commit and acceptance evidence remain immutable historical records; new investigation artifacts are recorded separately.


MS14 requested-sweep checkpoint, 2026-11-14: all 24 Qwen model/mode/context comparisons completed with each path's own fitted maximum minus 16 MiB. Qwen aggregate throughput exceeds serial at 8K through 32K and falls below serial at 64K through near-128K in this cohort. Gemma target-only joint preflight fails borrowed graph-workspace allocation at a 256K budget; a fresh 128-token pair at 1,478 MiB reproduces the error seen at 2,048 tokens / 1,024 MiB. The specific cause (measurement, layout/binding or allocation) remains unproven; no valid P2 target-only maximum or throughput is claimed. Gemma MTP measurements completed through 96K; the user stopped the near-128K run and cancelled the queued direct-input follow-up. The runtime is unchanged, all 2,166 source/runtime/input pins match after stopping, and the GPU is idle. `ms14-accepted-max-arena-sweep-v3/user-stop.json` and `measured-comparison.csv` preserve 29 complete paired comparisons plus six serial-only comparisons. The incomplete point is explicitly stopped, not failed or passed. No further Gemma benchmarking or controls should run without user direction. MS15 remains unimplemented.


Qwen32K/64K follow-up: unchanged-runtime native metadata/copy/synchronization probes confirm dense committed history columns, not a logical-growth hole in the attention history. Equal two-slot resident capacities per owning attention layer are49,664tokens target-only,36,352MTP3 and34,816DFlash7 for the accepted maximum-arena profiles. Both32K histories fit;64K histories stream in all three probes. DFlash7 at64K uploads approximately1.53GiB across the16target attention layers per verification, with five observed CUDA stream synchronizations inside each attention call; resident32K calls have negligible H2D and no such waits. A serial near128K control uses a242-page adaptive ring and cross-layer overlap, versus15/14-page group rings and8/7-page waves, and has no stream synchronizations inside its per-layer streamed-attention calls (completion elsewhere remains necessary). Thus grouped ring policy/pipelining adds an implementation cost beyond the live KV footprint. Different query/state/workspace and acceptance costs remain explicit; no exact equal-throughput expectation follows from comparing128K total live KV. All eight128-output diagnostic cases match the first112 output IDs of the original512-output sweep; all cohort source/runtime/input pins close unchanged. A P1 observer expectation was corrected separately without changing runtime code; original harness failure is preserved. `ms14-qwen-ring-cliff-analysis.json` records scope and evidence. MS15 concerns physical quota reallocation; the committed-history projection is already dense. No runtime optimization was implemented by this investigation; Gemma remains stopped.

#### Adaptive double-buffer experiment

The user requested committing the retained runtime before this experiment. Commit `4a6fb3d24fac55ddc64e14368f40efbe467fec16` freezes that accepted MS14 implementation; the original qualified binaries retain their original build identities. The double-buffer experiment supplies the validated parallel reference for the later aggregate-ring decision. The paragraphs below describe that experiment; the current default and final results are recorded above. MS15 slot-quota redistribution remains unimplemented.

The shared serial/grouped residency policy reserves room for two complete streamed layer tails when the physical pool permits it. This is a minimum, not a fixed transfer size or ring ceiling. Deadline feedback can demote further resident pages to enlarge prefetch capacity, and clean feedback cannot shrink below the two-tail floor. The fixed arena and authoritative host histories are unchanged. Exact-fit pools can use an empty resident prefix to hold both tails. Infeasible tiny pools retain bounded streaming rather than losing admission.

Grouped decode now carries registered, multi-page prefetch across owning layers for both slots, using the existing ready/consumed CUDA events and FIFO planner. Stable future history can upload before its producer runs; provisional rows still require the correct layer's producer. Native column order, masks, query geometry and arithmetic remain unchanged. Target and owned MTP caches share one physical policy per slot pool; repartition drains both readers, invalidates physical mirrors, and adopts new views before frame publication. Serial feedback now includes the supported one-through-eight-token decode transport. Only completed frames can train grouped feedback; cancelled transfers and late telemetry retain separate identities.

Test-first evidence is under `benchmarks/results/multi-slot-20261109/double-buffer-policy-tdd-v1/` and `double-buffer-group-tdd-v1/`: baseline floor, grouped queue-ahead and serial multi-query feedback regressions were preserved before the respective fixes. Qualification includes native bit comparisons, canonical host bytes, arena canaries, mixed query widths, partial pages, failed transfers and shared-owner repartition. A full-model DFlash fixture exceeds VRAM under eager loading on both the frozen baseline and candidate; lazy-loading strict memcheck passes with zero errors. This experiment makes no speedup claim before matched serial/parallel cost measurements. Larger rings consume residency and may increase streamed bytes; both costs must be measured.

### MS15 - Admit arriving requests during prefill and batch their prefill execution (implemented; quality regression locally repaired)

User direction, 2026-11-16; implementation and qualification, 2026-11-16/17. A text request arriving during another slot's prefill can join at a completed microbatch boundary. Subsequent compatible chunks execute together, then prepared members enter the retained grouped decoder. Membership freezes when decode starts. This replaces the former 4,096-token growth-reserve and quota-rebalancing proposal; that proposal remains withdrawn. Earlier dated MS14 checkpoints remain historical records.

The source baseline is eb2ee04bb5d3194b0c3e1581eb30a65c7dfafa69. Frozen MS14 binaries and their original build stamps remain unchanged. MS15 is ready for review; this entry does not close the milestone, commit, push or deploy it.

#### Retained contract

| Item | MS15 contract |
| --- | --- |
| Initial request | Start immediately, without an artificial wait to collect a partner. |
| Admission window | Accept an eligible request into a free configured slot while the global phase is prefill. Inspect arrivals at completed microbatch boundaries and before committing the transition to decode. |
| Admission cutoff | Freeze membership when decode starts. Later arrivals, including replacements for completed members, wait for the next group. No decode-to-prefill transition or mixed prefill/decode execution. |
| Shared prefill | Submit real rows from both compatible unfinished prefills through shared native microbatches whenever both can legally progress. |
| Batch budget | Keep total logical and physical batch limits unchanged. With both limits at 256, a normal balanced chunk is 128 real tokens per slot. Use available capacity when only one unfinished prefill remains. |
| Unequal histories | Support different prompt lengths, cached prefixes and current positions. Respect native chunk constraints, including Qwen's equal sequence-row counts within recurrent microbatches. No fake tokens or position resets. |
| Completed prefill | Preserve the earlier slot's prepared state, existing first-token emission, pending sampled anchor and open response while its peer finishes. Subsequent decode waits for all remaining admitted prefills. |
| Resources | Same context limits and one GPU arena. Quote selected target, recurrent/SWA, speculative, mask, conversion and transfer resources before publishing a binding. |
| Canonical state | Preserve existing host addresses, KV bytes/frontiers, request generations, checkpoints, speculative carry, sampler/grammar/RNG state and output cursors. Joining never restarts the existing prefill. |
| Vision | Retain vision-before-prefill ordering. Late requests needing fresh image encoding wait for the next group; already encoded inputs may join only through a qualified prefill-ready path. Preserve existing image modes. |
| Supported modes | Qwen/Gemma target-only and supported MTP; Qwen DFlash7 and DFlash7+K4V. No new model pairing, slot count, kernel arithmetic, decode policy, growth reserve or quota-rebalancing policy. |

#### Retained implementation

- Request-group admission appends live members only during prefill, preserves existing generations/frontiers, and permits either prefill completion order. Vision preparation retains its existing ordering. Decode-phase arrivals remain queued even after a member finishes.
- The arena owner inspects arrivals before publishing final prefill completion. It parks/drains current work, initializes only the incoming namespace, builds a proposed group, quotes the actual target/auxiliary work and activates its binding before committing membership. Rejected admission keeps the existing request valid.
- The planner shares the existing total batch/ubatch budget: normally 128 real rows per slot within 256 total. It balances recurrent rows, caps unequal tails, stops at media markers, and uses full chunks when one unfinished source remains. Positions and histories remain independent; no padding tokens or arithmetic changes were added.
- Deferred attempts are bounded per configured slot and keyed by task, balanced row count and unfinished-member count. An identical infeasible attempt is not repeated every chunk. A changed tail shape is reconsidered; a prepared cached prefix survives those retries.
- A completed prefill preserves its first emitted token, pending anchor and response while peers finish. The existing grouped decoder, ring controller, context limits, host allocation and one fixed GPU arena are retained.
- Late image-bearing requests queue for the next group. This also conservatively queues fully cached image requests: an encoded-image late-join path is not qualified. Existing initial-group image execution remains supported.

#### TDD record

Evidence root: `benchmarks/results/multi-slot-20261109/ms15-prefill-tdd-v1/`. Original failed attempts are retained; fixture/setup failures are not counted as production reds.

| Red | Retained green |
| --- | --- |
| Original selection implementation fails the new join, completion-order and rejected-join epoch tests. | Full selection suite: 31 tests, 146,821 assertions; normal and ASAN/UBSAN pass. |
| New planner interface returning no progress fails seven assertions across four planner tests. This is an interface red, separate from the original-runtime reproduction. | Balanced/unequal tails, odd budgets, finished members, limits, rejection and exhaustive small-budget oracle pass. |
| Frozen MS14 completes staggered requests but has no shared prefill or joint decode. | All six supported text modes demonstrate shared prefill followed by grouped decode. |
| Text batching across an image marker returns HTTP 500. | Contiguous text spans stop at the marker; existing image plus text cases pass across all six modes. |
| A rejected 2 x 128 join is never reconsidered when a small final chunk becomes feasible. | A 384 MiB arena rejects the initial quote, then admits 2 x 4 rows and joint decode; cached-prefix and ASAN variants pass. |

- [x] Freeze source/runtime/configuration identities and reproduce the original scheduling failure.
- [x] Qualify deterministic membership, stale/reused generations, cancellation, completion order, decode cutoff and chunk-planner tests.
- [x] Prove actual GPU matrix work over both 128-row sources, exact target logits against matching native layouts, unchanged peer canonical state after incoming reset, stale-handle rejection and reuse of the same physical parent.
- [x] Qualify normal public JSON/SSE across Qwen target/MTP3/DFlash7/K4V and Gemma target/MTP3, including unequal prompts, swapped IDs, cached prefixes, cancellation/reuse, resource rejection and smaller-tail retry.
- [x] Qualify simultaneous arrivals, the committed decode cutoff, and late-vision queuing. The final small-tail retry covers an arrival accepted immediately before the peer completes; CPU ordering fixes admission before completion publication without an HTTP timing threshold.
- [x] Run affected image controls, host sanitizers, strict CUDA and fixed-budget streamed-history controls. Close source/runtime/input pins and review scope, bounds, ownership and failure recovery.
- [x] Run the focused SWE comparison after functional qualification, with no warmups, retries, reverse-order repetitions or comprehensive context sweep.

Native short-prefix tests pass both insertion orders for both models with exact full logit vectors; the final fixture passes strict CUDA memcheck with API reporting all, exit99, default graphs/PDL and no kernel filters. Forced-streaming tests use an 8,192-token existing prefix and prove live full-attention KV exceeds the available resident-pool upper bound: Qwen 224,919,552 versus 32,292,608 bytes; Gemma 281,149,440 versus 35,698,816 bytes. Both orders pass exact logits. References explicitly match native cache geometry and destination packing: the initial one-source SWA cache and later two-source cache differ physically. Earlier mismatched-reference failures remain recorded; no numerical tolerance or kernel change was used to make them pass.

Final ASAN/UBSAN host/control cases cover cached-prefix retry, image/cancellation/reuse with K4V and Gemma MTP prefix reuse. Strict public-server MTP and K4V controls have explicit zero-error summaries. ASAN instruments the rebuilt server/control code; the native core is the unchanged qualified binary. Exact native logit/state comparisons newly added here cover target prefill; speculative qualification combines supported-mode server execution, positive draft counters, lifecycle/strict controls and exact single-slot output/draft-counter preservation. It is not a new exhaustive bitwise inspection of every speculative internal tensor or poisoned-device fault path.

Gemma Q8/Q8 at a 1,024 MiB arena cannot fit shared prefill: its initial two-source quote requires 1,213,548,416 bytes versus 1,073,741,824 available. The preserved failed positive fixture is a capacity finding. A separate explicit fallback case passes separate prefill followed by joint decode at that budget; the 1,200 MiB case passes shared prefill. Qwen Q8/Q8, Q5/Q5 and FP16, and Gemma Q5/Q5 also pass. The speed comparison below uses Q8/Q4 and matching budgets, not that larger-format counterfactual.

#### Focused SWE performance

Two pinned SWE tasks, A and B, each supply 8,192 native prompt IDs and 512 outputs. Single-slot speed is the arithmetic mean of A and B. B arrives about one second after A, while A is still prefilling. Each model/mode has MS14/MS15 serial and staggered pairs: 24 runs, one trial each, no warmup/retry/reversed repetition. Model/sidecar files, input IDs, sampling, batch/ubatch 256 and budgets match within each comparison. Qwen uses C=262,144 and a 2,048 MiB arena; Gemma uses C=32,768 and a 1,024 MiB arena. Gemma's previously known 256K preflight issue is not resolved or reclassified by this shorter-context qualification.

| Model / mode | Single MS14 / MS15 tok/s | Single change | Staggered pair MS14 / MS15 seconds | Pair speedup | MS15 joint decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen Target only | 46.58 / 46.58 | -0.02% | 34.17 / 28.18 | 1.212x | 75.90 |
| Qwen MTP3 | 116.58 / 116.19 | -0.33% | 24.18 / 27.26 | 0.887x | 152.53 |
| Qwen DFlash7 | 146.88 / 146.82 | -0.03% | 25.22 / 37.27 | 0.677x | 176.04 |
| Qwen DFlash7+K4V | 146.89 / 146.48 | -0.28% | 25.08 / 36.98 | 0.678x | 174.09 |
| Gemma Target only | 39.63 / 39.63 | +0.00% | 44.87 / 42.01 | 1.068x | 56.02 |
| Gemma MTP3 | 77.62 / 77.64 | +0.02% | 31.75 / 34.31 | 0.925x | 118.65 |

The frozen MS14 staggered pairs have no overlapping decode window because B waits for A's group to finish. MS15 joint throughput uses a common window after both requests emit at least 32 tokens and before either reaches 480. It demonstrates access to the retained parallel decoder, not a new decoder-kernel speedup. Pair wall time includes queueing, prefill, planning, drains, rebind/refill and decoding. Different native paired shapes can change continuation and draft acceptance; parallel output is not asserted identical to serial. All six serial controls preserve all 512 output IDs and generated/accepted draft counters for both inputs exactly.

| Model / mode | A TTFT MS14 / MS15 seconds | B TTFT MS14 / MS15 seconds |
| --- | ---: | ---: |
| Qwen Target only | 5.78 / 11.90 | 21.83 / 13.32 |
| Qwen MTP3 | 6.74 / 15.72 | 17.77 / 17.71 |
| Qwen DFlash7 | 7.56 / 22.94 | 20.15 / 25.72 |
| Qwen DFlash7+K4V | 7.57 / 22.68 | 20.07 / 25.41 |
| Gemma Target only | 8.28 / 18.76 | 29.81 / 21.93 |
| Gemma MTP3 | 8.44 / 19.76 | 23.31 / 22.72 |

A joined prefill can delay A's first token and completion even when pair wall time improves. The table retains that latency cost. Raw SSE traces, uncached token counts, first-progress delays, server join/binding events, TTFT, completion times, draft acceptance and common-window calculations are in `swe-benchmark/measurements.json` and `swe-benchmark/latency-and-stage-events.json`. Progress spans are observations, not isolated GPU kernel time. Per-copy transfer instrumentation was not enabled in speed runs. These are observations for one RTX 5070 Ti and two tasks, not confidence intervals or a new full context sweep.

#### Performance acceptance remains open

Target-only pair completion improves on both models, but all four speculative pairs regress in end-to-end wall time. This is not a serial output or single-slot decode regression: the six serial means range from -0.331% to +0.023%, and IDs/draft counters match exactly. The common decode window improves aggregate throughput, while prefill/handoff costs outweigh the saved decode time for these 8K/512 speculative workloads. Qwen MTP draft acceptance is 95.707% versus 95.828%; Qwen DFlash7 acceptance is identical at 63.957%. Acceptance does not explain the large DFlash wall-time loss.

Source and event review identify mandatory checkpoint-driven drains and rebindings: `create_checkpoint()` parks the selected group before exporting state; with two unfinished sources, a checkpoint for either retires both. Shared 128-row chunks and different prompt frontiers also create additional tail shapes, and speculative contexts increase binding/restoration work. The Qwen DFlash case has nine prefill bindings versus six on the closed-admission baseline. This is a mechanism supported by the code and trace, not a measured attribution of every millisecond to checkpoints.

A separately labeled checkpoint-policy diagnostic runs each runtime once with checkpoints disabled. MS14/MS15 pair wall time is 23.79/30.04 seconds, compared with 25.22/37.27 under the retained default. MS15 prefill bindings drop from nine to five, but a deficit remains. This removes checkpoint drains and changes native batch tails; output IDs and draft counters also change. It cannot be called an isolated checkpoint-copy benchmark or used as a same-output acceptance result. Checkpoints remain enabled in production code and the main matrix. The remaining split among shared-prefill kernels, residency, state restoration and graph setup is not measured here. The implementation satisfies the scoped scheduling behavior; no universal latency gain or performance closure is claimed.

The rebuilt runtime contains all current server translation units plus the selection implementation, linked to the frozen qualified native/common/CUDA libraries. The manifests pin 109 compiled project inputs and verify 14 unchanged native runtime files; original build/version stamps are retained. This is not claimed as a fresh full native build or full CI/platform qualification. `qualification.json` records coverage, preserved failures and limits; `provenance-after.json` closes source, test, runtime and input identities. Production/container state was not changed. Milestone closure and committing remain separate user decisions.

#### Quality investigation and repair, 2026-11-17

The user reported poor answers and two apparently endless decodes. Both production tasks reached the requested 65,536-token cap, with EOS enabled and identical 99.732% draft acceptance. The original tasks completed before the authorized inference pause. The user subsequently requested keeping inference paused until a fixed build is ready; the UIs remain running. This is a confirmed numerical defect, not a benign token-choice difference.

The exact production image/settings reproduce it with two pinned SWE prompts at both 8K and 64K per slot. Sequential requests return coherent text; simultaneous requests emit token 14 (`/`) for all 32 outputs in both slots and their first top-token probabilities are all null/non-finite. With `n_probs=0`, temperature 1 still repeats `/`, while greedy sampling repeats token 18 (`3`). Thus probability reporting and random sampling do not explain the failure.

Root cause: all-empty grouped activation bypasses `restore_group_sequences`, so newly bound recurrent/bounded arena storage is not initialized. The memory can retain arbitrary previous-layout bytes, including NaNs. Mixed empty/nonempty restoration already clears before loading peers, so staggered joins avoided this particular gap. Ordinary single-slot execution also has native initialization; a one-source group can still fail if its reused arena bytes are poisoned.

The retained local correction restores nonempty saved states through the existing path. When saved states are empty, it clears fresh native recurrent/bounded storage, or the DFlash block memory, and synchronizes before execution. The same rule applies to target and auxiliary roles. It leaves existing histories, masks, quantization, kernels, context/arena budgets and sampling policy intact.

The new deterministic test binds and retires a recurrent-state view, resets the request generations, poisons only that retired state range with `0xff`, and activates fresh one-source and two-source prefills. The original core fails six finite/native-exact assertions; the same executable with the repaired core passes all 26 tests / 1,565 assertions. Full logit vectors match zero-initialized ordinary-native execution exactly. The strict CUDA run retains graphs, default PDL, API reporting all, exit99 and no filters and reports zero errors. Existing Qwen/Gemma restored-history native-exact controls and all six image/text modes with the new finite-probability assertion pass.

With only the corrected core library overlaid into the original production image, both parallel contexts now return coherent text and finite top probabilities. At 8K the slots have 29/28 distinct tokens across 32 outputs, versus 1/1 before; at 64K they have 29/26, versus 1/1 before. All four sequential 32-token controls retain exactly the same output IDs. The `n_probs=0` temperature1/0 controls also stop collapsing. These are focused correctness observations, not an overall model-quality score.

Residual batch-dependent differences remain measurable after repair. In the 8K first-token distributions the top token agrees for both inputs; maximum log-probability differences over the common top16 are 0.192 and 0.302. The leading probability changes from 0.99047 to 0.99156 for A and 0.62472 to 0.64771 for B. This comparison covers only the first next-token distribution and does not establish full-vocabulary KL or quality equivalence. The NaN/repeated-token collapse is separately eliminated by deterministic initialization.

This exposes a gap in the prior qualification: public completion counts alone can pass for garbage, and matching-shape/staggered native references did not test poisoned all-empty initialization. `--group-fresh-recurrent-only` now supplies that numerical regression, and `--check-finite-probs` adds an optional public-protocol guard without changing benchmark defaults. Original reds and the fixture-only compile correction remain preserved.

Evidence: `benchmarks/results/multi-slot-20261109/quality-drift-20261117-v1/investigation.json`, the original/corrected public responses, native red/green logs and the strict report. Only `llama-context.cpp` was rebuilt and relinked with verified qualified objects; CUDA kernels and all other runtime files remain unchanged. No new ASAN or overall accuracy qualification is claimed. At the qualification checkpoint the correction was local and production was paused. The user subsequently authorized committing and deploying the repair on 2026-11-17; deployment evidence is recorded separately. This follow-up does not close MS15's separate performance acceptance.

### MS16 - Generalize adaptive KV execution from two slots to configurable N (closed with qualification limits)

User direction, 2026-11-17: remove the two-slot restrictions identified in the implementation review. This stage generalizes the existing runtime, preserves its scheduling and numerical contracts, and qualifies more than two requests. It does not declare N-slot support merely by relaxing startup checks.

Implementation base: 19099a1053faf833e9fc9d8e528662beadc9290d, including the fresh recurrent/bounded-state initialization repair. Freeze its actual runtime libraries, model inputs and source identities before implementation. The current production image and SDK pairing are recorded in `/home/raymond/LLM/production-server-ms15-quality-19099a105-20261117/`. Earlier MS15 results retain their original scope; defining MS16 does not close MS15's separate performance acceptance.

#### Target and execution contract

Support `--parallel N` for the existing adaptive shared-arena Qwen/Gemma modes, subject to the native logical-sequence limit, checked metadata/host capacity and actual operation admission. Do not replace the two-slot ceiling with a new hard-coded ceiling of three or four. N=3 and N=4 are the initial hardware qualification targets; larger CPU metadata tests and boundary tests are also required. Runtime support claims must name the counts, modes and hardware actually qualified.

| Quantity or behavior | MS16 contract |
| --- | --- |
| Configured slots N | Independent logical request namespaces, generations, host histories, samplers, output cursors and checkpoints. Configured capacity does not imply all requests execute simultaneously. |
| Selected subset S | Any compatible subset of admitted members that fits the logical, native batch and measured physical contracts. Support noncontiguous IDs and subsets containing only high-numbered slots. |
| Context C | Keep C as each request's logical limit and the selected group's combined logical budget. Preserve up to N*C logical history capacity in host RAM using checked arithmetic and existing sparse backing; GPU execution still uses one fixed arena. |
| Resource failure | Select a smaller legal subset when joint work does not fit. Preserve parked histories and open responses. Reject unsupported configurations or an impossible minimum operation explicitly; never silently discard extra slots or disable an advertised mode. |
| Selection policy | Preserve greatest compatible progress-making cardinality, then smallest combined history, then deterministic slot order. Preserve equal logical append allowance over the selected count, event-driven reselection and the existing waiting policy. |
| Phase policy | Start the first request immediately. Admit prefill-ready text arrivals at completed microbatch boundaries before the decode commit; freeze membership during decode. Keep vision, prefill and decode separate and preserve late-image queuing. |
| Prefill budget | Keep total batch/ubatch limits unchanged. Balance real rows across selected compatible sources and respect native checkpoint/media boundaries and legal recurrent splitting. With a 256-token budget, three sources can use 85 rows each and four can use 64 each; a single unfinished source can use the full legal chunk. No fake padding tokens. |
| Decode and speculation | Derive query/output counts and per-source row maps from actual selected work. Retain per-request headroom, pending anchors, proposal clipping, acceptance, replay, rollback and stop accounting. Selected widths may differ. |
| Memory and transfers | Preserve one parent arena and the MS14 aggregate ring over combined per-layer demand. Resident slices remain source-specific. Host histories and model weights are shared/preserved as before; no private GPU KV bank or fixed arena is added per configured slot. |
| Numerical state | Fresh selected state must be initialized before use; restored state must retain exact native bytes and identities. Keep native masks, positions, quantization, arithmetic, precision and reduction plans. |
| Compatibility | Preserve the ordinary non-streaming server, the default serial path, established N=1/N=2 behavior, supported KV formats, still-image paths and public completion/persistence APIs. |

The stage covers Qwen/Gemma target-only and their supported MTP depths, Qwen DFlash through seven proposals, Qwen DFlash+K4V and the existing supported still-image pairings. It does not add Gemma DFlash, deeper drafts, new quants/model pairings, mixed prefill/decode, decode-time request admission, new fairness policy, quota relocation, multiple GPUs or new modality support.

Native batch, output, registration-source and workspace bounds remain enforced. Derive limits for the actual selected work instead of treating every existing constant two as a request ceiling. K/V planes, target/draft roles, temporal feedback banks and double buffering retain their own independent counts.

#### Reviewed restrictions and intended changes

| Area | Work required |
| --- | --- |
| Server grouped-mode detection | Replace the exact `n_parallel == 2` recognition in `tools/server/server-context.cpp` with validated adaptive multi-slot configuration. Preserve model-only/deferred startup for N and keep unrelated non-streaming/serial paths unchanged. |
| Scheduler and prefill | Derive request-group capacity from configured slots. Remove the literal two-member prefill cap and audit late-admission checks that require every unfinished member to be selected. Admission and execution selection remain distinct: legal subsets may progress while other admitted members wait. |
| Metadata identities | Replace target, embedded-MTP, shared-MTP and DFlash `{0,1}` declarations in `src/llama-context.cpp` with validated configured identities. Keep logical ID, selected ordinal, native row index and cache role separate. Declaring N identities must not allocate an execution-sized GPU bank for all N. |
| Core admission and quoting | Audit `src/llama-cparams.cpp`, grouped constructors/factories and workload quotation together. Replace pair-only guards with checked configured/selected counts and native/resource predicates, retaining overflow, uniqueness, format and model-limit checks. |
| Speculative/vision admission | Generalize the independent MTP, DFlash, model-memory-factory and vision checks in `common/`, `src/llama-model.cpp` and `tools/server/server-common.cpp`. Keep unsupported mode combinations rejected before request mutation. |
| Assistant rows and bounded views | Derive Gemma assistant batch/ubatch and output capacity from legal selected query work within the existing budgets. Remove the exact-two constructor condition and bounded-reader ceiling in `src/llama-kv-cache.cpp`; do not just set every assistant batch to N regardless of native limits. |
| Resource ownership | Verify recurrent/SWA state, speculative carries, native sampler state and publication/transfer leases for every selected owner. Quote all N-dependent costs, including unavoidable allocations outside the arena. Require explicit drains before replacing bindings. |
| Capacity reporting | Generalize grouped-mode detection in `/props`; retain separate per-slot C, selected-group C, N*C all-history capacity and selected-storage quantities. `/slots` must expose every configured ID without confusing admitted/waiting members with selected members. |
| Finder and qualification tools | Generalize `benchmarks/find-max-arena.py` and public/native harnesses beyond choices 1/2 and IDs 0/1. Generate actual k-member, high-ID, asymmetric and serialized-subset probes and require evidence that the claimed selections executed. |

Reuse the already count-based selector, page partitioner, slot table, recurrent per-sequence vectors and aggregate-ring arithmetic. Audit their callers and capacity assumptions rather than replacing them wholesale. The current selector searches combinations; assess search/quotation cost as N grows. Preserve the exact policy with valid pruning, reuse or resumable owner-thread work where needed. A silent pair cap, arbitrary candidate cutoff or stalled arena owner is not an acceptable scalability workaround.

#### Implementation progress - 2026-11-18

MS16 implementation and isolated qualification were active at this initial checkpoint. The repaired 19099a105 source/runtime and production container are frozen in `benchmarks/results/multi-slot-20261109/ms16-n-slots-tdd-20261118-v1/`. Every temporary inference pause follows an idle-slot check and restores the existing production container; the candidate is not deployed.

The retained changes validate configurable logical IDs through the native limit, keep selected-state allocation separate from N host histories, generalize target/MTP/DFlash/vision admission and metadata, and remove the two-member prefill limit. Native verification batch capacity is a separate selection bound derived from batch rows and the declared verification block; it does not reduce logical admission capacity or proposal depth. Exact history-based pruning avoids redundant equal-cost selection quotes without changing cardinality/history/ID ordering.

For selected counts above two, speculative sizing now enumerates native graph geometries instead of Cartesian per-slot combinations or slot-count shifts. Recurrent graphs use active sequence count, equal rows and output count; unified nonrecurrent graphs use flat row/output counts. Zero-output native fragments contribute scratch bounds without becoming invalid whole verification intents. GPU sampler activation allocations are conservatively included across possible active chains. Existing one-/two-source catalog paths remain intact. Wider aggregate verification uses the existing quoted native attention route when the legacy narrow decode workspace callback does not cover its row count; no CUDA arithmetic or kernels are changed.

Initial evidence, with failures preserved separately from setup errors:

| Gate | Current evidence |
| --- | --- |
| Behavioral reds | Frozen-baseline context/admission tests reject N=3/4/8/native-boundary declarations; five native higher-count/sparse cases reject group construction. Finder tests reject the old 1/2-only CLI. Separate selection tests expose redundant equal-history quotes and conflated logical/execution capacity. |
| CPU | Context-memory, speculation, vision admission, parser and exhaustive selection-policy controls pass. Finder and public-harness configuration tests cover higher counts, high IDs and native verification batch bounds. |
| Qwen public execution | Target-only, MTP3, DFlash7 and DFlash7+K4V each execute genuine three-/four-request shared prefill and decode at C=32768, b/ub=256, a fixed 2048 MiB arena and 16 outputs per request, with independent IDs and finite probabilities. These are functional probes, not throughput results. |
| Qwen native state | Three/four sources and configured four-slot selections `{3}`, `{1,3}`, `{0,2,3}` pass full-vocabulary finite/native-exact checks after retired-state poisoning, checkpoint round trips and mixed fresh/restored membership. Recurrent state exports match native bytes. Repaired one-/two-source fresh-state controls remain green. |
| Gemma Q3_K_M resources | The 2048 MiB and 1920 MiB profiles fail first-request arena allocation; at 1024 MiB, three/four logical requests finish in measured two-source subsets. The original generic native fixture used full SWA and was not an equivalent streaming control; its OOM is excluded from count qualification. A corrected bounded-SWA sparse two-source native control passes. The pinned target-only finder fits 1512 MiB and rejects 1513 MiB; MTP3 fits 1030 MiB and rejects 1031 MiB. Both use four configured slots and two selected requests, with actual runtime libraries pinned. Larger selections are not certified by either fit. At an explicit 1496 MiB target profile, corrected bounded-SWA native references pass three sources and sparse selections `{3}`, `{1,3}`, `{0,2,3}` with exact logits and snapshots after mapping only native ordinal cell IDs to logical IDs. The four-source ordinary reference fails a 282.29 MiB compute allocation; that is an unavailable gate, not a native pass. |
| Native attention | The three-/four-source matrix passes 384 profiles and 27311 assertions, including 24/32 query rows, supported native workspace fallback, dense/compact masks, all registered reader formats, host-byte preservation and canaries. A separate 12-profile resident-span red exposes the old 16-query direct-path ceiling; the one-line route fix passes the same binary with native-exact output and spans larger than one page. Both matrices pass strict CUDA memcheck with all API errors enabled, no filters and default graphs/PDL. No GGML/CUDA source is changed. |
| Lifecycle and safety | Normal and ASAN/UBSAN core/common/parser/selection suites pass. Sparse configured-eight MTP, decode-cutoff queuing, native batch subsets, logical overbudget subsets, stochastic independent output routing, cached-prefix/cancel/reuse and ordinary EOS/grammar/stop controls pass within their recorded profiles. The mixed-empty-header poison branch passes strict CUDA and the original failing public scenario passes host sanitizers. |
| Persistence | Source ID 3 from a four-slot context can load into destination ID 1 in a two-slot context with identical continuation logits. The test fails before the source-namespace guards are corrected and passes afterward, including malformed-source/destination controls, strict CUDA and ASAN/UBSAN. Destination capacity remains enforced. |
| Existing vision modes | Qwen target-only/MTP3/DFlash7/DFlash7+K4V and the established Gemma Q3_K_S target-only/MTP3 image profiles all pass initial image encoding followed by genuine four-request text decode with finite probabilities. A late image with swapped IDs waits for all three existing text requests to stop before its prefill. Q3_K_S evidence is not substituted for the Q3_K_M text profile. |
| Native assistant state | Gemma Q3_K_S three-/four-source MTP compares full native trace/features, proposal tokens, hidden carries and continuation logits over independent widths 1/2/3 and acceptance 0/1/3, including correction and subsequent drafting. Qwen MTP3 passes three-source native trace/features, carry, unequal acceptance and correction, including strict CUDA memcheck; the four-source ordinary RS3 reference fails a 108.74 MiB compute allocation. A separate, explicitly labeled MTP1 case passes three/four sources. Initial missing-head, missing draft-trim and callback-framing fixture failures are excluded from product reds. Strict Gemma Q3_K_S MTP3 at the unchanged 2048 MiB profile cannot allocate its arena under memcheck (exit 99, four reported allocation API errors); normal native results remain separate and that strict gate is open. |
| Remaining qualification | Close the final source/runtime/resource ledger and affected native auxiliary/format/API/platform controls, including explicit unavailable-hardware limits. Repaired N=1/N=2 compatibility and the initial focused four-input SWE measurements are recorded below. Large rejection-heavy inventories still require a bounded owner-thread search assessment; equal-history pruning alone does not make the arbitrary admission oracle polynomial. Do not close MS16 from process success or substitute unavailable count/mode evidence. |

Lifecycle qualification found an additional initialization bug: a parked zero-token request can contain a nonempty native serialization header. If it shares the next selection with a restored peer, testing byte-vector emptiness skips initialization of its new GPU storage. Reused graph bytes can then enter recurrent computation as NaNs. The deterministic mixed-header poison test fails on N=2/3/4 and sparse IDs before the fix; the retained change uses authoritative token count to identify empty histories. The same native tests pass bit-exactly afterward, and the original four-slot cached-prefix/cancellation/swapped-ID failure now completes with finite probabilities and successful cancelled-slot reuse. This is additional coverage beyond the 19099a105 all-fresh-state repair. Strict CUDA and host-sanitizer checks for this specific new branch are tracked separately from the earlier passing gates.

#### Focused integrated measurement - 2026-11-18

The initial runtime-v13 core matched the production sources in the working tree at that checkpoint; subsequent changes are recorded separately below. Normal and ASAN builds, test dependency hashes and unchanged qualified GGML/CUDA libraries are pinned in the local artifact directory. The later finder changes only repair runtime-library provenance for Python 3.10 and paths containing spaces. The 17 finder tests, 31 legacy finder tests (one existing skip), three cohort-harness tests and three independent measurement tests pass. Relevant LF/UTF-8/whitespace and 155 model-name mappings were checked; Python files parse as Python 3.10 syntax. The unavailable ty, editorconfig-checker and Python 3.10 interpreter are not reported as executed.

The focused SWE point uses 8192 natural history tokens and 512 generated tokens per request, C=65536, b/ub=256, one fixed arena per model/mode, no extra warmup, automatic retry, reverse-order repetition or context sweep. Independent serial A/B/C/D runs supply the arithmetic mean for each matching cohort: N2 uses A/B, N3 A/B/C, N4 A/B/C/D. Qwen uses 2048 MiB throughout; Gemma Q3_K_M uses 1496 MiB target-only and 1014 MiB MTP3, separately from the established Q3_K_S vision profile.

Each cell below is serial arithmetic mean / common-active aggregate accepted tokens per second. The common interval counts actual delivered tokens after every request has begun decode; it is not N*512 divided by an overlap duration. All 512 outputs remain in the recorded full-wall measurements.

| Model/mode | Two requests | Three requests | Four requests |
| --- | --- | --- | --- |
| Qwen target-only | 46.6 / 75.8 | 46.6 / 97.5 | 46.6 / 109.7 |
| Qwen MTP3 | 115.0 / 150.5 | 107.9 / 157.6 | 102.9 / 156.2 |
| Qwen DFlash7 | 142.5 / 171.9 | 125.8 / 153.6 | 114.1 / 136.1 |
| Qwen DFlash7+K4V | 143.1 / 170.0 | 126.2 / 149.9 | 114.4 / 137.4 |
| Gemma Q3_K_M target-only | 39.7 / 57.9 | 39.7 / 56.4 | 39.7 / unavailable (at most three selected) |
| Gemma Q3_K_M MTP3 | 76.7 / 111.0 | 80.7 / unavailable (at most two selected) | 79.9 / unavailable (at most two selected) |

All requests finish, including the physical-subset cases; those cases have no common all-request decode interval and are not labeled three-/four-way throughput. Whole-group request wall times are materially different from common-active decode speed:

| Model/mode | Two requests, seconds | Three requests, seconds | Four requests, seconds |
| --- | --- | --- | --- |
| Qwen target-only | 27.9 | 39.1 | 54.0 |
| Qwen MTP3 | 25.9 | 49.5 | 71.8 |
| Qwen DFlash7 | 36.3 | 72.6 | 106.0 |
| Qwen DFlash7+K4V | 36.4 | 72.6 | 105.4 |
| Gemma Q3_K_M target-only | 41.4 | 73.6 | 267.9 |
| Gemma Q3_K_M MTP3 | 34.6 | 168.8 | 305.9 |

These wall times include request admission, prefill and phase/setup waits but exclude process startup. The JSON/CSV also records TTFT, native prompt/decode timers, acceptance/provider counters, matching independent serial wall sums and sampled GPU/RSS resources. At four configured requests, sampled minimum free GPU memory was 823/469/241/241 MiB for Qwen target/MTP/DFlash/K4V and 21/15 MiB for Gemma target/MTP. These are sampled minima, not guarantees of transient peaks; OS VmPin/VmLck are not driver registration accounting.

All 18 matched frozen-19099 N=1/N=2 controls produce identical 512-token output streams. Observed decode-rate deltas range from -0.69% to +0.74%; these single-pass runs do not establish a statistical confidence interval. The initial simultaneous Qwen target pilot delivered B first to the baseline and A first to the candidate, changing the native prefill batches and outputs. An explicit A-progress-gated comparison matches both prefill progress arrays and all output tokens without a product change. Its N2 result is used above. The initial N3/N4 target pilot retains its simultaneous-arrival scope; other modes gate A's first prefill progress before launching peers.

Separate N3/N4 target-biased copy-route checks exercise both real K4V proposals and genuine neural DFlash forwards with independent owners, finite probabilities, exact constrained copied tokens and matching provider/API totals. They are functional routing checks, not performance or model-quality evidence. Native MTP prediction-forward counters are not populated in the same way as DFlash counters and zero is not evidence of no MTP computation.

The selected-subset cases expose significant prefill binding churn: Gemma target N4 records 128 prefill bindings; Gemma MTP N3/N4 record 95/131, compared with eight bindings for their N2 controls. Maximum TTFT reaches 224 seconds for target N4 and 148/282 seconds for MTP N3/N4. Exact per-component parking, registration, rebind and refill durations have not been isolated. The exact selection policy remains unchanged; these measured costs require further work, not a claim of improved end-to-end latency.

Local review artifacts: scaling-summary.json, scaling-summary.csv, scaling-throughput-and-latency.svg, final-source-review-ledger.json and final-source-style-checks.json under benchmarks/results/multi-slot-20261109/ms16-n-slots-tdd-20261118-v1/. Raw artifacts remain local. At this checkpoint, larger rejection-heavy selector inventories, affected native auxiliary/format/lifecycle coverage, unavailable native/sanitizer/platform cases and separate registration/stall accounting kept MS16 open. No commit, push, deployment or production configuration change was made; the original 19099 inference container and the other three services are restored.

#### Continued implementation and qualification - 2026-11-18

The initial matrix and plots above remain runtime-v13 measurements. The continued implementation adds a native mandatory-state upper bound before full subset quotation. It measures the selected recurrent bank and bounded target/owned-DFlash state using existing native layout functions, excludes the shared Gemma assistant alias from duplicate charges, and finds the largest possible count by monotone binary search. It allocates no executable bank and commits no canonical history. The bound is necessary, not sufficient: every selected operation still needs its complete native workspace and attention quote. Configured logical capacity, proposal depth and the exact cardinality/history/ID selection policy are preserved.

The original Gemma Q3_K_M MTP3 three-request case makes seven full quotes for an impossible three-source operation. That is the retained red. Native format tests compare the bound with exhaustive counts 1 through 256 for Q8/Q4, Q8/Q8, Q5_1/Q5_1 and F16/F16; exact-budget, one-byte-short and failure controls pass. Direct Qwen/Gemma native tests check sparse selection, no GPU parent or host commitment, and shared-assistant accounting, including strict CUDA memcheck. CPU selection replays with sparse IDs, unequal histories and ID-specific resource rejection preserve every selection and append allowance while reducing full quotes.

For the fixed 1014 MiB Gemma MTP3 four-request SWE point, the new runtime-v21 completes in 280.6 seconds versus 400.2 seconds for a frozen runtime-v13 control under the same observed external CPU contention: an observed 29.9% wall-time reduction. All four 512-token output streams and acceptance statistics are identical. Both still make 131 prefill bindings, so this removes impossible-count quotation work without resolving subset churn. An unrelated workload consumes roughly eight CPU cores and was left untouched; the original 305.9-second quiescent result is not used as a controlled comparison. Separate two-request controls under that load measure Qwen DFlash7+K4V at 136.7 versus 138.7 aggregate tokens/s and Gemma MTP3 at 103.8 versus 103.5. These are single-pass observations, not confidence intervals or a rerun of the full scaling matrix. No CPU build overlaps a timing run.

Higher logical counts also exposed three independent defects, each with preserved real failures:

| Defect | Retained correction and regression scope |
| --- | --- |
| Derived bounded-cache identity collides with an owning source | With eight configured slots, logical selection `{0,4}` gives canonical cache IDs 1 and 5; the old `first_id*4+1` also gives 5. Gemma target and Qwen DFlash7+K4V fail real requests. A checked, plan-local helper keeps the legacy ID when free, otherwise skips all target/auxiliary owning IDs and handles overflow. It does not mutate canonical IDs. Both public cases, cached/swapped Gemma controls and a native Gemma sparse mixed-state comparison pass with exact full logits and SWA bytes. |
| Image setup treats configured N as active S | N=17 with b/ub=16 aborts while adding a seventeenth setup token, although the actual image operation fits. Setup now quotes a bounded representative verification selection and separately the one actual image source through the same pending target/assistant factory used for ordinary admission. It preserves first-request immediacy and late-image queuing. Initial and late image tests using logical IDs 0 and 16 pass. |
| Output allocation treats configured N as required output rows | After image setup is fixed, N=17 with a 16-row output budget still asserts. Both the minimum reservation and ordinary-context initial reservation now respect the output-row limit; actual requested output rows remain validated. The same native test fails on the prior runtimes and passes on runtime-v27: eight teacher-forced steps at logical ID 16 produce finite, bit-identical full-vocabulary logits to compact native execution. No arena, quant, context or tolerance change is used to obtain the pass. |

Normal and ASAN/UBSAN runtime-v27 builds and all five CPU suites pass. All six established vision modes pass again with genuine four-request text decode after image processing. The N=17, b/ub=16 image/high-ID case passes strict CUDA memcheck with zero errors, all API reporting, exit99, no filters and default graphs/PDL; it also passes ASAN/UBSAN. Sparse configured-eight Gemma MTP3 and Qwen DFlash7+K4V pass ASAN/UBSAN with cached prefixes, cancellation, swapped IDs and successful slot reuse. The native output-row regression passes both normally and under host sanitizers. These runs retain the qualified GGML/CUDA libraries; host sanitizer instrumentation covers the changed core/common/server code and test executables.

The native Gemma Q3_K_M sparse `{0,4}` comparison at the unchanged 1496 MiB arena cannot allocate that parent under strict memcheck: exit99 and two reported CUDA allocation API errors. The native reference context is already destroyed before that allocation. Its normal native-exact logits/SWA result remains valid, but this strict case is an additional unavailable gate. The batch records five targeted safety passes and this one failure; it is not reported as all green. No resource profile is reduced to hide the failure.

The final 13 changed production C++/header paths match both build snapshots and compiler dependency hashes; all 12 GGML library entries match the qualified runtime-v13 libraries. The 20 finder/harness contracts, 31 legacy finder tests (one existing skip), three measurement tests and `git diff --check` pass. The first system-Python test invocation lacked NumPy; that setup failure is retained separately from the passing repository-venv run. Final results and hashes are in `continuation-source-review-ledger-v27.json` and `source-and-style-v27-review.json` in the local artifact directory. The initial runtime-v13 ledger remains historical and is not used as proof for later changed branches. The original production container and all four services are running again; no commit, push or candidate deployment was made.

At the November 18 checkpoint MS16 remained open. State-only pruning does not bound arbitrary workspace-only subset rejection, and the measured prefill rebinding/registration/refill costs still need isolation. Native Qwen four-source MTP3 and full DFlash references, native Gemma Q3_K_M four-source execution, the unchanged strict Gemma Q3_K_S MTP3 profile and non-CUDA platforms remain qualification gaps. A separately recorded 1K-context Qwen four-source MTP3 reference also runs out of memory before comparison; it is not retained as a reduced-context substitute for the original gate. No further context shrinking, precision change, forced proposal suppression or new fixed slot cap is introduced.

#### Completion and accepted qualification limits - 2026-11-19

The user explicitly approved finishing on the current 16 GB card with exact qualification limits. MS16 is complete within that boundary. Unavailable native-reference and sanitizer profiles below are not passes; configured logical capacity, fitting execution capacity and numerical qualification remain separate claims. This closes MS16 without changing the separate MS15 performance acceptance or deferred release work. At this qualification checkpoint the candidate was uncommitted and production remained on 19099a105. The user subsequently authorized committing the completed stage.

The final candidate is runtime-v31, with a separate ASAN/UBSAN build. In addition to the November 18 fixes, it closes the remaining selection-responsiveness and unnecessary snapshot-copy costs:

- Exact selection polls an optional continuation callback at visited branches and before publishing a result. Read-only resource selection uses the existing queue-yield mechanism; a new request, cancellation or shutdown can interrupt it so the owner can process the mutation. Interrupted selection, prefill plans and decode proposals preserve their outputs and request epochs. No partial best result, candidate cutoff, slot ceiling or approximate policy is introduced. Opaque admission can still require combinatorial total work; this is an exact-policy complexity limit, with responsive reads and cooperative interruption rather than an unresponsive control plane.
- The real red continues through 2509 resource quotes after interruption is requested at quote 7. The retained implementation stops at 7. A last-candidate regression also prevents publication when interruption arrives during the final quote. Deterministic queue tests retain declined cancellation exactly once. A controlled public quote barrier blocks `/slots` on runtime-v27; the corrected path serves it, handles the queued mutation and completes all four requests. Initial logging-level and raw-UTF-8-log fixture failures are preserved separately from the reviewed passing evidence.
- Parking previously copied every saved target and auxiliary payload, including inactive peers, before replacing selected entries. It now exports and validates all selected states, retires both roles, then publishes only those states through compile-time-checked nonthrowing moves. Inactive snapshots stay in place. Already-owned activation temporaries also move into their destination vectors. Export or retirement failure cannot partially publish the new snapshots.
- The allocation regression uses the same Gemma Q3_K_M MTP3 four-request, 1K SWE, 16-output, 1014 MiB profile before and after the change. Large allocations during parking outside native export fall from 78 / 18,648,217,456 bytes to zero. All output tokens are identical. This is an allocation/correctness test, not a throughput result or a smaller replacement for an unavailable native reference.

Request-phase instrumentation of the original runtime-v29 Gemma MTP3 four-request 8K/512-output point localizes the cost. Parking, activation and quotation consume 72.56, 52.48 and 26.21 seconds of exclusive host time respectively; their inclusive times overlap. The 1160 `cudaHostRegister` calls take 0.076 seconds. Host waits inside prefill/decode attention total 28.27/1.02 seconds, with 251.99/90.71 GB of observed H2D/D2H calls across the request phase. These counts include native-state traffic. Asynchronous copy bytes and dependency/computation waits are not isolated GPU DMA durations and cannot be summed as a critical path. No synchronization is added. These instrumented numbers explain the copy correction and are kept separate from the uninstrumented comparison below.

Final qualification includes:

| Gate | Final evidence and boundary |
| --- | --- |
| CPU and host safety | All six context-memory, selection, server-queue, speculative, vision and parser suites pass normally and under ASAN/UBSAN. The native fixtures compile with fatal warnings. |
| Target numerical formats | Qwen target-only passes full-vocabulary/native-state equality for three/four sources and sparse three-source membership in Q8/Q4, Q8/Q8, Q5_1/Q5_1 and F16/F16. Gemma Q3_K_M passes three-source and sparse equality in Q8/Q4 and Q5_1/Q5_1 at the established 1496 MiB arena. This is target-format qualification, not an exhaustive speculative-format matrix. |
| Speculative numerical state | Qwen MTP3 three-source native traces, hidden features, unequal acceptance, correction and carry pass normal, ASAN/UBSAN and strict CUDA checks. Established Gemma Q3_K_S MTP3 three-/four-source native controls pass. Earlier matching-shape operator, mixed copied/neural K4V and native MTP1 evidence retain their recorded scopes. |
| Native lifecycle | Reselection, failed-compute recovery, mixed empty headers and cross-capacity checkpoints pass; affected reselection/failure/checkpoint cases also pass host sanitizers. Strict Qwen default three-/four-source and sparse target checks report zero CUDA errors, as does the MTP3 three-source check. API reporting all, exit99, default graphs/PDL and no filters remain in use. |
| Vision and public output | All six established image modes again complete image processing followed by genuine four-request text decoding with finite probabilities. The logical-17, b/ub=16, high-ID image case passes host sanitizers. Cached-prefix/cancellation/swapped-ID/reuse controls pass under host sanitizers for Gemma MTP3 and Qwen DFlash7+K4V, including three-source decode in the latter. |
| Public persistence | Qwen target, embedded MTP3, DFlash7+K4V and Gemma MTP3 save from ID 7 with eight configured slots, relocate within that server, then restore into ID 1 after restarting with two slots. All continuations are identical. Malformed-file rejection followed by recovery, erased-slot reuse and byte-identical preservation of an inactive peer's checkpoint also pass. |
| Larger format subsets | At the unchanged 1496 MiB target profile, four logical Gemma Q3_K_M requests complete in up to three-source subsets with Q8/Q8 and one-source subsets with F16/F16. These are fitting runtime counts; they do not establish unavailable three-source ordinary-native comparisons. |
| Tools and provenance | The 20 finder/harness contracts, 31 legacy finder tests (one existing skip), three independent measurement tests, source/dependency identity checks and whitespace/syntax review pass. Final manifests pin the production C++/headers and native test inputs; qualified GGML/CUDA libraries are unchanged. |

The accepted unqualified profiles are explicit: ordinary-native Qwen MTP3 with four sources (108.74 MiB compute allocation failure at the original context; the separately recorded 1K attempt also fails), full three-/four-source Qwen DFlash7 native references, ordinary-native Gemma Q3_K_M with four sources, three-source Gemma Q8/Q8 (282.11 MiB compute allocation failure) and F16/F16 (2600 MiB KV allocation failure), strict Gemma Q3_K_S MTP3 at the unchanged 2048 MiB profile, and strict sparse Gemma Q3_K_M at the unchanged 1496 MiB profile. Existing normal native and public results are not substituted for those missing strict/reference results. Windows/HIP/MUSA execution and larger GPU selected counts remain unqualified. CPU logical-boundary tests cover 256 slots; GPU logical-ID tests reach 17 configured slots, while genuine four-way execution is qualified only for the named fitting profiles.

The last matched performance check uses runtime-v27 before these final changes versus runtime-v31 afterward, the same natural SWE A/B inputs, 8192 history tokens and 512 outputs per request, C=65536, b/ub=256 and the same per-mode arena. There are no extra warmups, retries, reverse-order repetitions or context sweep. Each two-request point has a genuine common decode interval. All paired output streams are identical; source counters agree with API totals. The serial controls and initial three-/four-request scaling matrix above retain their original identities and dates.

| Two-request model/mode | Aggregate tok/s before / after | Group wall seconds before / after |
| --- | --- | --- |
| Qwen target-only | 75.9 / 75.7 | 27.83 / 26.75 |
| Qwen MTP3 | 149.5 / 151.3 | 25.93 / 24.71 |
| Qwen DFlash7 | 171.4 / 171.4 | 36.01 / 34.96 |
| Qwen DFlash7+K4V | 171.3 / 171.2 | 36.27 / 34.45 |
| Gemma Q3_K_M target-only | 57.9 / 57.9 | 41.24 / 38.99 |
| Gemma Q3_K_M MTP3 | 111.1 / 110.8 | 34.41 / 31.88 |

Two-request decode changes range from -0.25% to +1.23%, while whole-request wall time falls 2.9-7.4%. The matched Gemma MTP3 four-request A/B/C/D point falls from 220.73 to 145.37 seconds, an observed 34.1% reduction with every 512-token stream unchanged. Both execute at most two sources at once and retain 131 prefill bindings; neither has an all-four common decode interval. These are single-pass observations, not confidence intervals or a claim that the snapshot correction speeds up steady-state kernels. Remaining subset rotation follows the preserved minimum-history policy; further scheduling-policy changes and GPU DMA critical-path analysis are future optimization work.

Final evidence is under benchmarks/results/multi-slot-20261109/ms16-n-slots-tdd-20261118-v1/, including completion-source-review-ledger-v31.json, final-comparison-v31-summary.json, native-final-v31-results.json, aux-final-v31-results.json, final-safety-v31-results.json and persistence results. Earlier v13/v27 ledgers and plots remain historical. At that checkpoint the existing production container was restored, all four compose services were running, and no candidate deployment, commit or push had been made.

#### Final closure after the extended context sweep

The subsequent Qwen sweep is complete and validated: 236 uncached runs comprise 164 matching serial references, 40 three-slot cases and 32 four-slot cases. All 72 parallel comparisons execute the requested count genuinely; all 412 requests generate 512 tokens. The four modes are target-only, MTP3, DFlash7 and DFlash7+K4V. A/B/C and A/B/C/D use distinct pinned natural SWE histories with arithmetic-mean serial references at the same actual history lengths, including shortened endpoints. The 256K combined budget covers three-slot points from 8K through 80K per request and four-slot points from 8K through 64K, in 8K increments. Each path uses its own fitted arena maximum minus 16 MiB; there are no benchmark warmups, repeated trials or automatic retries. Source/runtime/model/input pins close unchanged.

Results, validation and four aggregate-context plot files remain local under benchmarks/results/multi-slot-20261109/ms16-comprehensive-sweep-20261119-v1/. The one-/two-slot curves in those plots retain their historical MS14 identity; the three-/four-slot curves use MS16. They are not presented as a controlled runtime comparison. Production was restored afterward and all four compose services are running.

MS16 is closed with the accepted qualification limits above. The user authorized committing the implementation, regression tests and this closure record. Generated benchmark data, plots and unrelated untracked work are excluded from the implementation commit. This closure does not deploy the candidate, push the branch or close the separate MS15 performance acceptance.

#### TDD implementation stages

- [x] MS16-A - Freeze the repaired baseline and write real reds. Reproduce current rejection/omission of slot 2 and higher, the two-member scheduler/prefill caps, fixed metadata IDs, assistant capacity and finder restrictions. Keep baseline binaries and source stamps immutable; distinguish fixture/setup errors from behavioral reds.
- [x] MS16-B - Generalize logical identity and admission. Test configured N=1/2/3/4, larger CPU inventories, N at the native boundary and invalid/overflowing declarations. Exercise highest ID, sparse/permuted selections, duplicate/stale generations and N-dependent storage products before exposing the generalized mode switch.
- [x] MS16-C - Generalize deferred contexts and the selected resource catalog. Quote/bind actual S for target-only, recurrent/SWA and auxiliary roles. Test one selected high-ID request, genuine three-/four-source fitting work, smaller selections, exact/short grants and transactional activation failure with intact peers. Do not reserve all configured requests' execution state in VRAM at startup.
- [x] MS16-D - Generalize prefill and decode scheduling. Add deterministic simultaneous/staggered arrival, last-boundary arrival, completion, cancellation/reuse and decode-cutoff cases for three/four requests. Test 256-token totals, three-way 85-row chunks, four-way 64-row chunks, unequal tails, checkpoint/media boundaries and legal native splits. Prove real shared work and preserve every member's progress through reselection.
- [x] MS16-E - Qualify speculation and vision. Test independent widths/acceptance for three/four sources, high-ID sampler/output routing, mixed copied/neural K4V routes, target/assistant state and rollback/catch-up. Preserve vision ordering, per-image M-RoPE positions, fresh/cached/changed inputs and queued late images using the same parent arena.
- [x] MS16-F - Generalize API, persistence and finder integration. Cover N reported slots, waiting responses, N*C checked capacity, cache reuse, save/restore/restart, relocation into another valid ID and malformed payload rejection. Preserve existing two-slot artifacts; qualify cross-N restores where their geometry/format permits and report explicit incompatibility otherwise. Finder success must exercise the declared selection sizes and validate the next rejected capacity.
- [x] MS16-G - Close numerical, lifecycle, safety and compatibility gates, then measure performance. Retain every red/green record, source/runtime/input pin, actual native split and resource result. Keep unavailable coverage and inherited limitations explicit. Implementation or qualification is not complete merely because N requests returned the requested number of tokens.

#### Required qualification matrix

| Dimension | Required observations |
| --- | --- |
| Counts and identities | Configured N=1/2/3/4; selected sizes 1 through each admitted maximum. Include N=4 with S={3}, {1,3} and {0,2,3}, plus larger sparse/high-ID CPU cases. A configured four-slot server executing two requests is labeled two-way execution. |
| Histories and transitions | Equal/unequal histories, fresh all-empty groups, mixed empty/restored members, reused arenas, canceled/reused slots, selected-group total below/at/over C, growth exhaustion and permutations of completion order. Preserve parked members through repeated subset changes. |
| Numerical references | Full-vocabulary finite checks and native-exact logits, visible KV, hidden features and recurrent/SWA/speculative state with matching legal batch shapes and destination packing. Teacher-force identical inputs to localize divergence. Compare different layouts separately and investigate material drift instead of dismissing it because batching differs. |
| Initialization regression | Extend the 19099a105 poisoned-state test to three/four selected sources and affected target/auxiliary roles. Poison retired ranges, protect adjacent owners and test both all-empty and mixed restored states. Preserve the original one-/two-source red and green. |
| Public quality and stopping | JSON/SSE, simultaneous and staggered requests, finite pre-sampling probability checks, deterministic greedy and production-style stochastic controls, normal EOS/stop/usage accounting and no repeated-token collapse. Retain actual token IDs and raw scores where collected; a completion-count-only pass is insufficient. |
| Resource and failure paths | Same fixed arena per profile, actual selected-state/workspace/registration costs, resident and streamed histories, ring wrap and source/role separation. Exercise short grants, failed activation, stale bindings, cancellation and supported recovery without partial publication or damaged peers. |
| Mode/format coverage | All six existing text modes and affected still-image paths, supported speculative depths/source transitions, established Q8/Q4, Q8/Q8, Q5/Q5 and native-F16 controls according to changed dependencies. Keep target/draft/projector files and numerical rules fixed within a comparison. |
| Safety and legacy controls | Applicable host and ASAN/UBSAN controls, normal GPU and strict CUDA with API reporting all, exit99, default graphs/PDL and no kernel filters. Fresh N=1/N=2 controls against the repaired baseline and ordinary non-streaming compatibility remain mandatory. |

Use fitting native references without changing quants, positions, arithmetic or tolerances to obtain a pass. N=3/N=4 hardware qualification must name each supported model/mode and actual selected count. If the current RTX 5070 Ti cannot fit a genuine k-source operation, preserve the measured failure and verify correct smaller-subset execution; do not call that a k-way numerical or throughput pass. An unavailable required joint case remains a declared qualification gap, not a new software maximum or an automatic stage pass. Existing Gemma large-context preflight limitations are not implicitly resolved by MS16.

The N=1/N=2 regression comparison must include sources in their existing placements and batches so initialization or indexing changes cannot be hidden as permitted cross-layout variation. Checkpoint and generation semantics, accepted output IDs/counters where identical work permits exact comparison, resource minima and performance remain protected. No new output cap, forced proposal suppression, larger arena, smaller context or disabled feature may conceal a regression.

#### Performance and exit criteria

After functional qualification, run a focused scaling comparison using pinned distinct natural SWE inputs for each active request, including A/B and separate C/D inputs for three-/four-request cohorts. Keep the existing serial arithmetic-mean convention across those inputs. Freeze model/sidecar/projector, KV types, sampling, context, batch and arena per comparison; keep primary scaling measurements at a fixed arena and report fitted ceilings separately. Use 512 outputs per request for the measured cohort, with no automatic warmup, retry or reverse-order repetitions. Preserve crash evidence, fix the cause and resume the remaining cases; any rerun after a documented fix is recorded separately. This plan does not authorize a comprehensive context sweep.

Report configured N, actual selected counts, per-request history and total live context, serial mean speed, aggregate speed over one common active interval, acceptance/source statistics, TTFT, completion latency and whole-group wall time. Include admission, checkpoint, parking/restoration, rebind and refill costs. Record host commitment/registration and GPU minima; separate instrumented safety runs from speed runs. More configured slots or a higher aggregate rate does not establish a latency improvement or compensate for an existing-mode regression.

- [x] Pair-specific wiring and eligibility checks are removed or replaced by explicit native/resource contracts, with a reviewed distinction between request counts and unrelated two-element roles/buffers.
- [x] Qualified configurations can admit and preserve N independent requests and execute any legal selected subset without assuming IDs 0/1 or selected count two.
- [x] Genuine three-/four-source prefill/decode and the supported mode-specific cases have numerical/public execution evidence where admitted; remaining hardware/count limits are explicitly reported.
- [x] Fresh-state initialization, independent histories, stop/acceptance accounting, cache/persistence and one-/two-slot compatibility pass their required gates without hidden workarounds.
- [x] Finder/API/docs state actual capacities and qualification limits; the final source/runtime/input and regression ledger is closed, with measured scaling costs and unresolved findings visible.

MS16 exit: configurable N logical request slots, correctly quoted selected execution subsets, no hidden two-slot scheduler/identity/assistant assumptions, and explicit evidence for the supported counts/modes. Implementation was subsequently authorized, including temporary inference pauses for isolated GPU qualification and restoration of the existing server. The 2026-11-19 completion record and user-approved qualification limits above control closure. No candidate deployment, production configuration change, commit or push is implied; existing MS15 performance work remains separate.

### MS17 - Productize compositional numerical equivalence across supported combinations (planned)

User direction, clarified 2026-11-20: productize the numerical-equivalence approach so every supported combination works; there is no primary model, mode or slot count. Planning base: af64b74dec91b0f811eefc12742115c30ef7b5bf, including the MS16 implementation and its documented limits. The deployed three-slot Qwen DFlash7+K4V configuration is one regression profile in the complete matrix. Pin candidate/reference sources, runtime libraries, models, sampling, UVM policy and image budgets for every profile when implementation starts.

#### Product target and claim boundary

Deliver reusable, supported tooling and an automated qualification contract covering numerical computation, logical state and asynchronous ownership for every advertised combination. Keep reference rollback banks and intermediate records in host memory or streamed storage, and execute one original native layer or fused region at a time. Derive genuine selected counts, proposal widths, tensor geometry and role layouts from the configuration/capabilities; no model/count-specific manual runner edits are required. Reduce simultaneous reference residency rather than the numerical workload. Measure GPU, host and artifact-storage requirements across the supported geometry classes, not only the current production profile.

The product target is compositional numerical equivalence across the complete supported combination matrix. Each valid matrix cell needs an explicit qualification result and the supporting reference/composition evidence. Finite test inputs still do not establish a universal implementation-wide mathematical proof; stronger claims require their own formal argument and discharged assumptions. MS16 remains closed under its accepted limits, but those exceptions cannot waive missing MS17 coverage. A pass for a default profile or a representative subset does not close the milestone.

| Claim | Required meaning |
| --- | --- |
| Numerical regions | Matching semantic inputs, native batch geometry, positions, masks, dtypes, quantization, fusion boundaries, accumulator precision and reduction/dispatch plans produce identical defined outputs and state updates. |
| Logical-state relation | Candidate and reference describe the same request histories, token positions, valid KV, recurrent/SWA state, pending anchors, provisional rows, output cursors and required sampling state. Physical addresses and storage packing may differ through a validated ownership mapping. |
| Greedy behavior | Equal target scores with the same tie-breaking, filtering and stop rules emit the same tokens and preserve the state relation. |
| Stochastic behavior | The implemented verification and sampling algorithm preserves the target distribution under the same sampling transforms. Identical output strings from a shared seed require an additional RNG-coupling argument and are not implied by distribution equivalence. |
| Reference identity | An ordinary native path in this fork is labeled same-fork native. A stock claim additionally requires independently pinned stock target code and a validated extraction/compatibility bridge. Self-consistency is labeled separately. |

The composition argument is inductive: begin with related initial states; show each numerical region and each append/accept/reject/rollback/scheduling transition produces the required outputs and preserves that relation; then chain those checked transitions. Record the assumptions and coverage supporting each link. Correct components alone do not establish correct wiring or event ordering.

#### Supported combination matrix and product deliverables

Generate a versioned, machine-readable matrix from the product's capability and admission contracts. Do not silently narrow the matrix to cases convenient for the current card. Validate every supported cell through direct comparisons or an explicit composition of independently validated regions/transitions with matching assumptions; reuse must record why it applies to that cell. Continuous/data-dependent parameters need defined boundary classes and parameterized invariants rather than an assertion that every possible prompt has been enumerated.

| Dimension | Required product coverage |
| --- | --- |
| Model/mode | Qwen target-only, supported embedded MTP depths 1-3, DFlash depths 1-7, and DFlash+K4V; Gemma target-only and supported shared-KV MTP depths 1-3. Cover registered target/assistant/projector pairings and their advertised architecture/weight-format paths. |
| Proposal behavior | All supported declared depths and actual clipped widths, every legal accepted-prefix length, neural/copy provider selection and transitions, and supported K4V lookup/proposal/minimum-hit parameter boundary classes. Target-only and zero-proposal operations are first-class cases. |
| Configured/selected counts | Serial and arbitrary configured N within the native limit, every legal admitted selected geometry, sparse/high IDs and permutations, native batch limits and resource-driven subsets. N=1/2/3/4 and logical-boundary inventories are mandatory regressions, not a new ceiling. Count parameterization and identity/ownership invariants must also cover larger legal counts. |
| KV/attention | Every K/V pair, head geometry, conversion/writer path and native/optimized route declared supported by the active backend. Generate valid pairs from capabilities rather than assuming all CLI type combinations are executable. Assistant formats retain their declared inheritance/alias contracts. |
| Modality/phase | Text and all existing supported still-image pairings; image counts/aspects/admitted budgets, native positions/M-RoPE, fresh/cached/changed images, late-image queuing, image-to-text transitions and speculative image-feature routes. |
| Histories/memory | Short/long and equal/unequal histories, zero/restored/provisional state, resident/streaming/mixed layers, tails, ring wrap, registration boundaries, growth/reselection and near-context/arena boundaries. Preserve each profile's original numerical workload and declared memory policy. |
| Sampling/API/lifecycle | Greedy and supported stochastic transforms, filters/grammar, EOS/stop, streaming/non-streaming, prefix reuse, checkpoints/save/restore/restart/relocation, cancellation, slot reuse and error recovery with intact peers. |
| Build/backend | Every platform/backend that the product advertises, with explicit build/runtime/library identities and CI/resource requirements. Missing required hardware or a reference adapter is a release-blocking coverage gap, not an automatic successful skip. |

This includes all valid combinations of the supported dimensions. Existing invalid combinations, such as Gemma DFlash or unsupported modality/model pairings, remain explicit negative-admission tests; MS17 does not invent those features. An advertised valid configuration cannot be reclassified as unsupported merely because its verification is unfinished. Legitimate operation/resource rejection must preserve requests and select legal subsets or return the documented error. That proves admission behavior, not the numerical correctness of an unexecuted selected geometry; capacity and numerical coverage remain separate.

| Deliverable | Product requirement |
| --- | --- |
| Reference/replay entrypoints | Maintained project CLI/test entrypoints for reference capture, region replay, comparison and qualification, using existing graph/oracle/codec infrastructure where possible. Configuration-driven models, roles, counts and formats replace per-profile script patching. |
| Trace/comparison protocol | Versioned bounded records, source/config/weight/library provenance, checked schemas, streaming comparisons, deterministic reproduction commands and useful earliest-mismatch diagnostics. Validate malformed/truncated data and budget host/disk use. |
| Qualification orchestration | Capability-generated cases, explicit evidence reuse/assumptions, resumable progress without hiding failures, and per-cell exact-pass/failure/reference-gap/inconclusive status. A default-only or pairwise-only smoke suite cannot stand in for full valid-combination coverage. |
| CI and shipping gates | Fast CPU contract/codec/state/sampling regressions plus automated GPU region/integration/safety jobs, with sharding/resource tiers for the full matrix. Publish supported invocations, capability/coverage reports and failure guidance. Required jobs and missing cells block closure. |
| Runtime integration | Opt-in capture/validation attaches to the actual execution path without changing numerical behavior or silently serializing the pipeline. Normal production execution retains its behavior and resource/performance contracts when validation is disabled. |

#### Independent reference and staged replay

Select and pin a compatible stock target implementation, model/weight hashes, compiler/backend libraries and execution settings. The stock reference revision is not yet selected. Provide a valid reference adapter for each advertised architecture/operation. If upstream cannot represent one directly, record the gap and establish an independently audited compatible reference; the affected supported cell remains release-blocking until resolved. DFlash/K4V need not have an upstream drafter counterpart to establish target-output correctness, provided their proposals are always processed by a verified target/acceptance rule.

Reuse the existing native tensor/state oracles, framed exact-comparison infrastructure and operator tests where their source and input identities apply. Audit the reference adapter as part of the trusted boundary. Existing same-fork passes supply evidence but cannot independently validate modifications shared by both sides.

1. Start from zero/native initial state or an independently validated checkpoint. Generate reference prefill and subsequent histories from that state. Candidate-produced activations, KV, masks or frontiers must not become unverified expected inputs on both sides.
2. Extract the original layer or fused graph region with its actual global layer index, hparams, shapes, strides, masks, positions and native plan. Treat cross-layer fusions as one region where necessary. Changing the declared model size, precision, context, proposal width or execution layout to obtain a match invalidates the intended comparison.
3. Execute the whole selected-source geometry for that region, including each legal selected count and the full configured verification width, with the pending anchor and up to seven proposals where required by the protocol. Independent single-slot runs cannot silently replace the multi-source operation. Preserve legal native microbatch splits and record their input/output maps.
4. Compare the candidate's actual region inputs against independently derived reference inputs, then compare outputs, full-vocabulary target logits at the final head, quantized KV payloads and recurrent/SWA checkpoint updates. Chain trusted reference outputs into the next reference region. Preserve all needed prefill and carry paths, not only the last decode layer.
5. Retain only the current reference working set on GPU. Stream exact byte/value comparisons and release already checked intermediates to bound host memory and disk use. Hashes identify artifacts; they do not replace the numerical comparison. Keep failing inputs, outputs, state and the earliest differing logical coordinate for replay.

Require bit equality for defined payloads under matching native plans. Normalize only audited logical identity/storage mappings; do not normalize floating-point values or reorder a reduction to make it pass. Compare canaries and ownership separately from undefined scratch/padding. Distinguish legitimate masked infinities from unexpected non-finite activations or logits. Report first difference, differing-element count and absolute/ULP diagnostics without using them to introduce an unapproved tolerance. A decomposition that changes native arithmetic or cannot validate its bridge remains an unresolved reference gate.

#### State, speculation and asynchronous integration

- Build expected state from reference histories and an independent transition ledger. Check per-request identity/generation, logical position, K/V validity, recurrent/SWA state, target-feature carries, DFlash block positions/selectors, pending proposals, committed frontiers and accepted output cursors. Pending sampled anchors, committed KV, provisional verification rows and bonus/correction outputs require separate accounting.
- Exercise every accepted-prefix length for each supported draft depth, including lengths 0 through 7, immediate rejection, full acceptance and correction/bonus paths. Cover unequal per-slot outcomes and widths, source completion in different orders, context-end clipping, parking/restoration, checkpoint relocation, cancellation and reuse. Controlled proposal injection may isolate verification/rollback, but is labeled separately from real neural and K4V proposal execution.
- Confirm that rejected rows and stale generations are inaccessible to future logical attention, while accepted histories and inactive peers retain their exact state. Test fresh all-empty, mixed empty/restored and poisoned retired banks, including headers that encode zero history.
- Trace the actual asynchronous path and establish producer-write, transfer-completion, consumer-read and retirement ordering. Validate source/role ownership, leases, ring wrap and reuse fences. Delay copies and vary interleavings so stale-read and early-reuse defects are observable. Capture outputs on the real compute stream; adding a pre-consumer synchronization or replacing the pipeline with serialized replay cannot qualify the live ordering.
- Cover resident, streaming and mixed layers; equal/unequal histories; sparse/high IDs; selection changes and target/draft roles. Retain small live-server integration checks to verify that the traced transitions are the ones used by the deployed path. Public completion counts and finite top probabilities remain smoke evidence.

Audit the actual sampler path before choosing its correctness argument: target-resampling/match verification and probability-ratio rejection sampling have different obligations. Verify temperature, top-k/top-p, penalties, grammar/filtering, EOS/stop handling and independent request RNG ownership as applicable. Use exact small-vocabulary distributions and exhaustive branch cases to validate the implemented acceptance/correction rule, alongside the algebraic distribution-preservation argument. Distinguish that ideal-arithmetic argument from the implemented finite-precision/PRNG semantics and record the required assumptions or bounds; statistical non-rejection alone is not a proof. Neural and deterministic K4V proposals may differ from a reference drafter; target verification and any proposal probabilities used by the rule must remain correct. The distribution-preserving speculative-sampling framework is described in [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html); citing that result alone does not verify this implementation.

#### TDD implementation stages

- [ ] MS17-A - Freeze the complete capability matrix, claims and resource contracts. Pin candidate and independent reference sources, model/assistant/projector artifacts, libraries and settings. Enumerate all valid combinations and negative-admission cells, parameterized geometry/boundary classes and evidence requirements. Classify earlier oracles by independence. Measure region/host/disk working sets across the matrix and keep every unresolved reference/resource obligation visible.
- [ ] MS17-B - Productize and validate reference/replay entrypoints and artifacts. Provide maintained, configuration-driven project tooling and independent adapters for all supported architectures/roles. Use fitting native controls to validate extraction and composition, preserving fusions/plans. Add real reds for one-bit corruption, wrong masks/positions, owner/role swaps, stale generations and malformed/truncated records; verify intended failure reasons, bounded memory/storage and deterministic reproduction.
- [ ] MS17-C - Qualify numerical regions across all supported combinations. Replay embedding/input preparation, attention/conversion/writers, recurrent/SWA blocks, feed-forward/residual/normalization regions, feature extraction and logits for both model families, all modes/depths, supported format/geometry paths and legal selected counts. Include every supported image path. Every matrix cell needs direct or explicitly validated compositional evidence; no preferred/default profile substitutes for the rest.
- [ ] MS17-D - Qualify complete logical transitions across the matrix. Check every supported accepted-prefix length and width, unequal slot outcomes, rejection/correction/bonus, state carry, parking/restoration, image/text phase changes, checkpoint relocation, cancellation/reuse and failure publication against independent state. Detect output-cursor mutation, off-by-one frontiers and inactive-peer overwrite. Fix discovered defects with preserved red/green evidence and all affected existing-mode controls.
- [ ] MS17-E - Qualify live asynchronous integration for every applicable mode, role, format and phase. Stress real streams/events, delayed copies, resident/ring transitions, wraps, retirement and stale bindings. Demonstrate detection of omitted waits and premature reuse without synchronizing away the defect. Cover genuine neural/copy providers, all existing image paths and sparse/high-ID selected subsets.
- [ ] MS17-F - Close sampling and composition claims across the supported matrix. Tie every active greedy/stochastic path and proposal provider to its acceptance, correction, filtering, stopping and RNG contracts. Produce the initial-state relation, transition-preservation arguments and per-cell assumptions. Distinguish exact tensors, greedy identity, distributions and any stronger seed/formal claims; no missing cell is implied by another mode passing.
- [ ] MS17-G - Ship the tooling, automated matrix and completion gate. Integrate documented CLI/test workflows, fast regressions and sharded full GPU/build/safety jobs. Run normal and ASAN/UBSAN checks and strict CUDA with all API reporting, exit99, default graphs/PDL and no filters. Protect serial, N=2, higher counts and ordinary execution, including disabled-validation overhead. Close the per-combination ledger only after every supported cell and required platform job is qualified.

#### Acceptance and exit criteria

- [ ] Every valid supported model/mode/depth/format/count/modality combination has a valid independent reference chain and explicit qualification evidence at unchanged numerical settings and legal native geometry. Production profiles are included without replacing other combinations.
- [ ] Numerical outputs and all defined logical state match at the checked boundaries; trusted initial state and chained reference inputs prevent shared candidate errors from passing both sides.
- [ ] Verification/rollback, pending-anchor and output accounting preserve the relation for all accepted-prefix lengths and independently exercised slot outcomes, including real neural/copy routes and inactive peers.
- [ ] The live asynchronous pipeline passes its ordering/ownership gates, and defect-injection controls demonstrate sensitivity to stale reads, wrong routing, partial publication and early reuse.
- [ ] Greedy and stochastic correctness claims match the implemented algorithms. Same-seed identity, independent-stock equivalence and universal proof are claimed only where their additional obligations are actually met.
- [ ] Reusable project tooling, versioned artifacts, automated capability/matrix generation, CI integration and user/developer documentation are delivered; cases run without model- or count-specific manual code edits.
- [ ] Every supported matrix cell is qualified. Numerical failures, unsupported references, inconclusive cells and unavailable required hardware/jobs block closure. Small controls, default-only coverage or previous MS16 exceptions cannot mark missing combinations passed.

MS17 exit: a productized, reusable numerical-equivalence qualification system with maintained project tooling, automated CI/matrix coverage and qualified operation of all advertised combinations. Independent-reference provenance, exact region/state comparisons, sampling reasoning, live asynchronous ordering and explicit evidence reuse/claim limits are required throughout. There is no primary target and no closure based on one successful configuration. Preserve historical monolithic-reference OOM records while resolving the corresponding supported-cell reference obligations through the bounded approach or required qualification resources. Performance tuning is separate; correctness fixes and production-integration overhead regressions must be resolved through TDD and affected gates. This remains a plan; implementation, GPU qualification and production rollout have not started under this request.

### V3 release preparation - User documentation, staged CUDA builds and branch promotion (planned)

User direction, 2026-11-21: prepare this work for `feature/adaptive-kv-stream-v3`. Current implementation branch: `feature/adaptive-kv-multi-slot`; planning base: af64b74dec91b0f811eefc12742115c30ef7b5bf. Rewrite the README around the user-facing v1/v2/v3 concepts, with wording to be drafted together later. Publish Linux and Windows RTX 50 CUDA packages after a successful merge build, and build older GPU architectures on assigned weekdays through V100 and P100. This section records the plan; it does not rename a branch, change workflows, publish binaries, deploy a server or close MS17.

#### README narrative and scope

| Generation | User concept | Explanation boundary |
| --- | --- | --- |
| v1 | Full-attention history beyond available VRAM | Keep authoritative KV in host RAM, retain a GPU-resident prefix and stream the remainder through an adaptive ring while attention stays on the GPU. Explain RAM, VRAM and PCIe costs without implying constant throughput at every context length. |
| v2 | Speculative execution inside the existing adaptive infrastructure | Reuse the shared arena and phase-specific execution/storage machinery for target, draft and verification work. No separate draft KV streaming infrastructure or independently reserved full execution arena is required. Additional draft weights, state and applicable workspace still consume memory and must be included in fitting. |
| v3 | Multiple requests share an aggregate KV working set | Pack the selected sources' per-layer histories into a consecutive aggregate working set and map its resident portions and streamed suffixes onto the shared arena/ring. Preserve independent sequence identities, positions, masks, recurrent/SWA state and speculation. Storage aggregation does not let requests attend to each other's tokens. |

Plan the README in this order: the problem and three generation concepts; one small memory-placement illustration; supported models/modes/slot counts and measured limits; architecture-specific downloads; short serial and parallel quickstarts; arena fitting and combined-context semantics; performance and reproduction; links to developer details and historical results. Keep large kernel inventories and milestone history in development documentation. Explain the existing prefill admission window and decode membership policy where they affect a user. Separate configured context capacity, actual combined histories and generated-output allowance.

Retain existing benchmark provenance. Distinguish SWE central-window throughput from production whole-period rates and state sampling, acceptance, arena, source revision and per-slot/combined context. A branch rename is not evidence of universal stock equivalence or qualification on untested GPUs. Preserve MS16's accepted limits and MS17's separate productization obligations; do not silently mark MS17 complete.

#### Two CUDA build paths

1. Required PR/merge path: compile and link Linux x64 and Windows x64 for the RTX 50 / SM120 family only, using the existing pinned CUDA 13.0.1 toolchain and audited SM120a native-code handling. Exclude all older architecture code generation from this path. Preserve CPU, source, Python, server, sanitizer, UI, shared-library and package-startup checks according to their existing applicability. Publish the two CUDA packages only after the required merge/push checks succeed.
2. Weekly compatibility path: build the selected architecture profile for both operating systems, independently of merge readiness. Resolve the latest successfully validated/published v3 source commit once per run and pin that SHA through UI generation, configuration, all shards, linking, packaging and publication. Failures remain visible but do not make an otherwise successful RTX 50 merge wait for a weekly build. Keep a manual profile/SHA dispatch for diagnosis and recovery; rebuilding an already published identity must verify rather than replace its bytes.

Proposed weekly assignment, with both OS packages per profile:

| Day | GPU family | Native target | Toolkit |
| --- | --- | --- | --- |
| Monday | RTX 40 / Ada | SM89 | CUDA 13.0.1 |
| Tuesday | RTX 30 / consumer Ampere | SM86 | CUDA 13.0.1 |
| Wednesday | RTX 20 / T4 / Turing | SM75 | CUDA 13.0.1 |
| Thursday | V100 / Volta | SM70 | CUDA 12.9.1 |
| Friday | P100 / GP100 | SM60 | CUDA 12.9.1 |
| Saturday | P40 / GTX 10 Pascal compatibility | SM61 | CUDA 12.9.1 |
| Sunday | Additional datacenter coverage, if included in the release matrix | Separate SM80, SM90 and SM100 profiles | CUDA 13.0.1 |

The Sunday datacenter profiles are a proposed extension; the required older-card target in this request is coverage through V100/P100. Keep each profile a separate architecture build rather than recreating a universal fat binary. Use one profile table for dispatch, toolchain selection, package identity and documentation. Schedule away from the top of the hour, provisionally 03:17 America/Chicago, and make the weekday/timezone mapping explicit in tests.

CUDA 13 cannot compile targets older than SM75. Use the already supported pinned CUDA 12.9.1 installer for Volta/Pascal, including matching runtime/cuBLAS packaging and a compatible dependency set. P100 is SM60, not SM61: the existing manual `61-real;70-real` defaults omit it. Audit non-tensor-core paths, FP16/quantized attention dispatch, architecture guards and toolkit/library support; compiling an ordinary CUDA backend does not by itself establish that adaptive target/MTP/DFlash execution works on these devices. Record compile, package startup and physical-GPU runtime qualification separately, and resolve failures before advertising a supported combination.

GitHub schedules execute workflows from the repository default branch. Verify the actual default branch during migration; it was not available through the current metadata connection. If v3 will not be the default, put the scheduler entry on the default branch and dispatch the intended v3 workflow/ref explicitly. Never let a scheduler's default-branch `GITHUB_SHA` silently label binaries built from a different source SHA. See [GitHub schedule behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule), [NVIDIA toolkit guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/) and [legacy compute capabilities](https://developer.nvidia.com/cuda/gpus/legacy).

#### Build and publication implementation

- Reuse `.github/workflows/build-cuda-sharded.yml`, the setup action, `build.py`, `cuda_shards.py`, `setup_cuda.py` and the existing package verifier/publisher. Thread the source SHA, architecture profile and toolkit identity through every stage and artifact/cache key. Keep matching UI provenance and source/command/header validation. Preserve full format and D256/D512 coverage within each admitted architecture; do not remove inference features merely to shorten compilation.
- The current Linux defaults include SM75 PTX, SM86, SM89 and SM120; Windows already selects only SM120a. Retain sharding and inspect the remaining Windows critical path. Architecture reduction alone is not proof that the Windows timeout is fixed. Require actual hosted Linux and Windows completion, linked executables and packaged-startup evidence within the hosted job limits. Bound aggregate compiler concurrency across weekly profiles so weekly work does not starve merge builds.
- Make packages identify source SHA, OS, architecture profile and toolkit, with checksums, runtime dependencies and qualification scope. Both OS packages must validate before publishing that profile. Give scheduled profiles separate immutable release identities; do not append architecture assets to an already published immutable SHA release or replace its manifests. Existing v2 releases remain unchanged. A downloads table must show the exact commit available for each profile, since weekly binaries can lag the RTX 50 build.
- Update the active v2 branch filters, publication allowlist, release labels, workflow-policy tests, CI operating guide and download examples together. Preserve historical v2 benchmark and release references. Branch protection and scheduler registration/default-branch placement are explicit cutover checks, not effects assumed from changing YAML.

#### Implementation order and acceptance

- [ ] V3-A: add failing policy tests for RTX 50-only PR/merge selection, one scheduled profile per assigned weekday, CUDA 12.9 for SM60/61/70, explicit source SHA selection, matching two-OS packaging and profile-aware immutable publication. Cover rejected/missing/failed results and rerun behavior using existing CI test infrastructure.
- [ ] V3-B: implement the profile-aware fast build and release path, retain all other required checks, and obtain successful hosted Linux/Windows RTX 50 builds. Record actual critical-path timings and remaining compiler resource issues.
- [ ] V3-C: implement weekly/manual compatibility builds, validate each profile on both OSes, and publish only correctly labeled artifacts. Perform the relevant physical-GPU numerical/lifecycle qualification before claiming runtime support; unavailable hardware remains an explicit open gate.
- [ ] V3-D: draft and review the README together, replace stale single-request-only descriptions, document the implemented aggregate layout and actual supported matrix, and verify quickstarts/download links. This plan does not preempt the requested later wording discussion.
- [ ] V3-E: after the README and fast release path are accepted, promote `feature/adaptive-kv-multi-slot` to `feature/adaptive-kv-stream-v3`; align remote tracking, workflow targets, branch-protection checks, scheduler placement and active links. Verify a real v3 push/build/publication cycle and profile dispatch. Preserve commit history and historical v2 releases. Production changes are a separate task.

Exit: the v3 branch presents an accurate user-oriented explanation, delivers successful RTX 50 Linux/Windows merge packages promptly, and has a visible scheduled path and qualification ledger for the older architecture targets. CI compilation, hardware qualification and MS17 numerical-equivalence closure keep their distinct meanings.

### Deferred - Documentation and release review (previous MS14)

- [ ] Document opt-in configuration, context semantics, phases, the implemented prefill-only admission window and membership frozen for decode, pauses/heartbeats, proportional pages, host address/commit/pin accounting and actual mode/platform limits.
- [ ] Publish measured finder examples and reproducible performance summaries without committing raw benchmark artifacts or local server-AB harness additions.
- [ ] Consolidate status only after reviewing stage evidence; no complete checkbox based solely on a process exit.
- [ ] Review the final regression ledger and ensure no new failure, confirmed slowdown, capacity reduction or missing applicable gate is concealed by a scope change or marked resolved solely by explanation.
- [ ] Keep stage changes reviewable and use the user's one-sentence/title-only commit preference when implementation commits are authorized. Do not commit or push this planning document merely because it was requested.

Exit: all scoped combinations supported for two slots and, after MS16, explicitly qualified higher counts; single-slot compatibility preserved, no unqualified N-slot/physical-GPU/platform claims, and no production configuration changes without authorization.

## 12. Acceptance matrix and evidence rules

The following dimensions are required across appropriate stages; they need not form one blind expensive Cartesian server matrix.

| Dimension | Required cases |
| --- | --- |
| Selected slots | MS1-MS15 cover one/two on GPU. MS16 adds genuine three-/four-source fitting GPU work, sparse/high-ID subsets and larger/boundary N in CPU metadata/planner tests; report unavailable count/mode qualification explicitly. |
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

Keep three evidence roles explicit. Ordinary-native execution in this fork checks numerical agreement at equivalent execution shapes. The frozen pre-multi-slot V2 implementation checks preservation of existing functionality, resources and performance. MS17 additionally requires an independent stock-reference bridge and compositional state/ordering argument for a stock-equivalence claim; same-fork native agreement alone does not establish that claim.

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
| MS13 qualification / deferred release | Fresh integrated coverage for the final source and all required existing-mode comparisons; earlier passes are reused only under the explicit identity rules. |
| MS16 configured N / selected subsets | Repaired N=1/N=2 controls, high-ID and three-/four-source execution, poisoned fresh state, all affected target/speculative/vision roles, finder/API/persistence and bounded resource/failure behavior. |
| MS17 compositional numerical qualification | Independent reference provenance and extraction bridge; matching native regions and full-vocabulary/state checks; acceptance/rollback/sampling contracts; real asynchronous ownership/order; source/count/format coverage and explicit claim limits. |

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
