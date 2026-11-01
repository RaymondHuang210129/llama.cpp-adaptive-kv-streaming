# DFlash2 vision phase-arena implementation roadmap

Status: implementation in progress, 2026-11-01. DV0-DV3 are complete; DV4-DV9 remain pending. Public streamed DFlash2 plus mmproj admission remains closed. The user authorized implementation through DV9 with TDD, regression checks and one descriptive one-sentence commit per completed stage, without attribution trailers.

Source baseline: merged Adaptive KV Streaming V2 at `7425d14c3ae0bc357b2e5f51edff15fabbe17f70`. Recheck the named mechanisms when implementation starts. Historical evidence retains its measured source and binary identities when later Git metadata changes.

## User decisions

Recorded on 2026-11-01:

- Qualify pure DFlash2 plus mmproj first; K4V vision is a separate follow-up, not a condition for the first release.
- Correct image support with clearly reported performance is the release objective. A speedup over MTP3 is not required; existing supported-mode regressions still need investigation.
- For reference capacity, first try a validated smaller DFlash quantization, with the user-suggested HermiHg Q2_K_S-MIX checkpoint as the initial candidate. If the full native target/projector/smaller-draft configuration still cannot fit, split-process encoder/native-decoder references are approved, with their component-level scope stated explicitly. The delivered server remains a single process. Use the same draft checkpoint on both sides of each native/streamed comparison; target/projector weights and numerical checks do not change.

## Objective and release boundary

Support still-image/text requests with the already qualified Qwen3.8-27B target, matching projector and matching DFlash2 sidecar under the existing single shared phase arena. First qualify pure DFlash2 with mmproj. Treat optional DFlash2 plus K4V plus mmproj as a separate extension after the pure path passes; do not admit mixed vision merely because pure DFlash vision works.

Preserve target arithmetic, native attention masks and reduction order, DFlash convolution/selector behavior, exact cache formats and target sampling rules. Reuse MTMD preprocessing/encoding, the speculative dispatcher, bounded draft KV, recurrent rollback and the serial workspace owner. No second vision arena, new allocator, new attention kernel or speculative execution framework is expected.

Related contracts:

- [DFlash2 design](DFLASH2_ADAPTIVE_KV_ROADMAP.md).
- [DFlash2 and K4V design](DFLASH2_K4V_ADAPTIVE_KV_ROADMAP.md) and [current support report](docs/development/dflash2-k4v.md).
- [Memory infrastructure](DEVICE_MEMORY_INFRASTRUCTURE.md) and [consumer ownership](DEVICE_MEMORY_CONSUMERS_ROADMAP.md).
- [Existing vision behavior and limits](README.md#vision-support), [Qwen vision qualification](tools/server/tests/README.md#adaptive-kv-vision-qualification) and [Gemma vision ownership reference](GEMMA4_VISION_PHASE_ROADMAP.md).

### Initial scope

- The pinned Qwen target and original IncoAI DFlash2 pair in [the artifact manifest](benchmarks/data/dflash2-qwen27b-artifacts.json), plus the projector already qualified for that target. The user also approved qualifying a smaller DFlash weight encoding for native-reference capacity, as detailed below. Pin each sidecar and projector independently; this does not authorize changing the target or projector quantization.
- One CUDA GPU, one serial request, Flash Attention, GPU KV offload, an explicit device-local shared arena, `--fit off` and context shifting disabled.
- Physical-fit and ordinary performance work use UVM off. Do not silently offload weights, change quantization, shrink an image, reduce output count or change image-token limits to turn a failure into a pass.
- DFlash proposal lengths 1-7 and actual target verification widths TG1-TG8. Prediction remains within the sidecar's trained block size. Large image batches do not increase speculative width or recurrent rollback depth.
- Q8_0/Q4_0 as the first end-to-end profile, followed by Q8_0/Q8_0 and Q5_1/Q5_1. Adaptive drafting inherits target formats; explicit draft-cache overrides remain rejected.
- Still images, multiple images and interleaved image/text are release requirements within measured context, image and batch limits. Begin with one small image, then expand. The encoder's token budget and actual output-row count are not interchangeable.
- No simultaneous MTP, Gemma+DFlash, other target/draft pairs, DSpark/EAGLE, audio/video, parallel slots, multi-GPU, LoRA or K15/TG16 admission.
- Preserve all existing target-only/MTP vision, text-only DFlash/K4V, image processing and platform-specific launch behavior.

### Smaller-draft reference policy

The first reduced-footprint candidate is [HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF), file `Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf`. The publisher lists approximately 535 MiB / 561 MB. This is checkpoint size, not measured total runtime VRAM. The encoding mixes several tensor precisions; it changes draft weights, not the Q8/Q4 decoder-cache setting.

Before use, pin the repository revision and actual file SHA, audit target/tokenizer pairing, feature-layer mapping, dimensions, selector/convolution metadata, RoPE configuration and compiled weight-reader support. Preserve the original Q4_K_M artifact and its text regression baseline. Do not silently replace the production configuration.

There is an additional artifact-level image prerequisite: the repository's [metadata repair helper](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF/blob/main/add_dflash2_metadata_field.py) identifies `dflash.rope.dimension_sections` as relevant to image handling. Inspect the downloaded GGUF and compare it with the known-compatible drafter; the filename or model card alone is not proof the field is present and correct. Do not blindly execute the external patcher or infer draft sections from the target's spatial layout. If a repair is necessary, retain the original, review the change, prove tensor bytes unchanged and pin the derived file separately.

Reference order:

1. Qualify the smaller checkpoint's loader, text behavior and geometry, then attempt the full ordinary-native image control with that checkpoint.
2. Pair every native control with a streamed control using the identical smaller checkpoint, image embeddings/positions, KV types, batching and sampling. Existing Q4 controls remain their own cohort where fitting; Q2 draft vectors, proposals and acceptance counts need not equal Q4's.
3. Only after a genuine capacity failure of the smaller-draft full native configuration, use the approved split-process encoder export/native decoder import. Keep the same checkpoint and full-vector checks; retain actual end-to-end encoder/decoder tests on the streamed server.
4. If the failure is missing metadata, an unsupported kernel, wrong pairing or a numerical mismatch, diagnose it instead of treating split-process execution as a remedy. Any further target/projector quantization or offload change needs a separate decision.

## Findings that govern the design

| Finding at the baseline | Classification | Required response |
| --- | --- | --- |
| `common_streamed_dflash_config_error` rejects mmproj; vision startup and per-request guards admit only existing target/MTP combinations | Confirmed admission barriers | Open each guard narrowly after internal qualification; preserve text-only mixed support and unrelated rejections |
| `can_suspend_for_vision()` rejects the ordinary DFlash child role | Confirmed owner admission barrier | Admit a proven parent-owned bounded writer; never relabel DFlash as a shared-cache read-only assistant |
| DFlash KV and target rollback staging already use persistent parent regions | Existing reusable mechanism | Preserve their exact physical regions and account for them in every vision loan |
| `serial_features_match()` rejects every embedding row and currently identifies rows by token ID and temporal position | Confirmed feature-contract barrier | Distinguish deliberately skipped image batches from consumed features; validate the actual image/source identity when consuming features |
| Native common DFlash skips multirow embedding batches pinned to one temporal position | Existing native policy with a real capacity reason | Preserve and test this first; do not blindly insert all image rows into the draft cache |
| Parent workspace growth rejects a live serial child; target reservation is ordinarily token-input based | Confirmed restriction, profile-dependent failure risk | Measure supported image-input topology and retained feature outputs before child registration; do not assume a text quote is sufficient |
| Server image decoding already invokes `common_speculative_process` after target decoding | Existing reusable connection | Reuse the callback, verify ordering and propagate failures rather than adding a second image driver |
| K4V receives `get_text_tokens()`, which removes image placeholders | Separate mixed-mode semantic risk | Define media boundaries before optional mixed admission; filtered text offsets must never become target KV positions |
| Current vision suspension drains target compute and recurrent spill/restore completion | Existing completion boundary | Preserve it when admitting the DFlash child and the existing restore-overlap optimization |

These are source findings, not combined-runtime passes. The late-growth risk needs actual measurements; it must not be described as an observed OOM. Publisher support in another inference engine does not qualify this CUDA/arena path.

Source entry points:

- [Common DFlash admission](common/common.cpp): `common_streamed_dflash_config_error`, `common_streamed_dflash_verification_error`.
- [Server vision admission and token history](tools/server/server-common.cpp): `server_vision_arena_config_error`, `server_vision_arena_request_error`, `server_tokens::get_text_tokens`.
- [Server image driver](tools/server/server-context.cpp): `encode_mtmd_chunk`, `process_mtmd_chunk`, `add_mtmd_chunk`, speculative begin/draft and checkpoint handling.
- [DFlash input and image policy](common/speculative.cpp): `common_speculative_impl_draft_dflash::process`, `begin`, `draft` and the mixed dispatcher.
- [Target feature capture and graph measurement](src/llama-context.cpp): `serial_features_match`, `sched_reserve`, `graph_reserve`, layer-input extraction and KV suspension/resumption.
- [Shared owner](src/llama-context-memory.cpp): persistent reservations, `grow_workspace`, `can_suspend_for_vision`, `suspend_kv`, `quote_suspended_loans`, `lend_suspended`, `resume_kv` and serial handoffs.
- [Image encoder phase](tools/mtmd/mtmd-session.cpp): `mtmd_batch_encode_arena` and its failure-safe return path.
- [Vision measurements and workspace](tools/mtmd/mtmd.cpp) and [workspace lifetime](tools/mtmd/mtmd-workspace.cpp).
- [Position-correct image decoding](tools/mtmd/mtmd-helper.cpp) and [DFlash injection/prediction graphs](src/models/dflash.cpp).

## Decisions and invariants

### Image rows, positions and draft context

Keep these quantities separate throughout the implementation:

1. Physical target KV row count, including all image rows.
2. Target M-RoPE coordinates and the next decoder position.
3. Text-token history length passed to a lookup source.
4. Retained DFlash KV cells and their native positions.
5. Current proposal length and target recurrent rollback slots.

An image may have many rows sharing one temporal position. A position-based draft SWA cache cannot evict that same-position group merely because more rows arrive. A large group can exhaust bounded cell capacity even when the arena has spare bytes. Assigning artificial increasing positions would change semantics, not fix admission.

The first implementation preserves native pinned-image skipping. Later text features still come from the image-conditioned target. This does not establish good draft acceptance on image tasks; measure it. Confirm behavior for every MTMD chunk/microbatch shape, including one-row image tails. Existing first/last-position checks are not a general proof that all rows are pinned; characterize their actual input constraints before changing them.

If the native control itself fails a required image shape, preserve that failure and fix or explicitly resolve the native policy before using it as an oracle. Do not silently skip the profile. Full image-feature injection with a different draft position/window scheme is a separate design decision, not part of this initial compatibility work.

### Projector embeddings and DFlash features are different inputs

MTMD produces target input embeddings. DFlash consumes concatenated inputs from its declared target layers; for the current pairing, five target-layer feature vectors form each injection row. Do not feed raw projector outputs through the DFlash encoder or reuse Qwen MTP's raw-image/shifted-hidden convention as if the contracts were identical.

For any feature batch actually consumed, validation must establish the completed target forward, source generation, sequence, row extent, layer mapping and relevant input/position identity. Different images at identical positions are a required negative case. Pointer equality or text length alone is insufficient, and the checks must not depend on a projector buffer whose lifetime has ended.

A batch deliberately skipped under the native policy must not gather or inject stale features. Validate that the skip belongs to a supported image batch rather than turning every embedding callback into unconditional success. Preserve successful target-batch feature maintenance after every later verification, including a future K4V-supplied draft.

### Startup and workspace sizing

Use existing model/context setup and planning helpers. The intended ordering is:

1. Validate configuration, model/projector pairing, feature metadata, F and effective microbatch constraints without mutating an existing target's active configuration.
2. Load the target and retain the existing warmup-retirement-before-sidecar-load behavior.
3. Acquire sidecar and projector metadata without duplicate GPU weight loads. Enable the target's declared feature outputs before measuring relevant target graphs.
4. Measure token and image-input target graph shapes for the supported MTMD splitting/mask policy, including retained feature outputs, image tails and all admitted verification widths.
5. Resolve persistent bounded draft KV, recurrent staging and target/draft per-phase workspace requirements through the current parent planner before any child captures addresses into it.
6. Register and bind the DFlash child, measuring feature injection independently of token prediction. The target's F+1 verification limit must not be confused with an image's row count.
7. Configure the deferred projector and verify the final owner/projector compatibility and structured quota results before publishing a usable server.

If necessary metadata is currently obtained too late, separate metadata acquisition from execution registration using the existing initialization flow. Do not allocate the model twice or retain an extra warmup workspace to avoid ordering the steps correctly.

Prefer premeasurement over allowing arbitrary parent growth with live children. If measurements prove coordinated growth is necessary, retire affected executables, drain real users, preserve fixed regions and rebind every borrower transactionally. That expansion needs its own red/green lifetime tests; removing the `serial_children` growth guard is not an acceptable fix.

### One parent, serial phases

```text
Target text/image prefill and feature extraction
    -> native-policy feature injection or deliberate image skip
    -> DFlash prediction
    -> target verification and feature maintenance
    -> acceptance / bounded suffix repair / recurrent restoration

At a later image boundary:
    drain affected target/draft work and pending recurrent transfers
    -> retire temporary graph/ring bindings
    -> preserve fixed state and lend reclaimable parent space to vision
    -> encode image to owned host embeddings
    -> release encoder workspace and projector grants
    -> restore target prefill bindings
    -> decode image into target KV
    -> continue feature maintenance and speculative generation
```

Do not run the target, drafter or encoder concurrently in aliased graph scratch. Existing overlap of disjoint recurrent copies with draft computation can remain, provided vision entry joins all required copies before lending storage.

| Resource | Vision-phase lifetime and owner |
| --- | --- |
| Target and DFlash model weights | Remain loaded outside the arena; include both in physical fit |
| Target current recurrent state and host rollback snapshots | Retain existing owners; complete pending publication/restore before phase entry |
| DFlash bounded KV and cache metadata | Preserve authoritative bytes and positions in their existing persistent parent allocation |
| Parent rollback publication staging and other fixed regions | Preserve physical bounds and leases under the existing plan; reclaim only if separately proven reconstructible |
| Target full-attention history | Authoritative host history survives; device resident/ring contents may be reloaded under existing policy |
| Target/draft graph scratch and discardable device KV grants | Reclaim only after their last readers/copies complete and captures retire |
| Projector weights and encoder workspace | Temporary measured grants in the same parent, excluding every retained fixed region |
| Decoded image data, output embeddings and retained target features | Owned host/output allocations with explicit lifetime and separate accounting; not implicitly bounded by the device arena |

The parent must cover the maximum admitted serial workload plus its simultaneously live persistent resources. Total process VRAM also includes model weights, recurrent state, outputs and driver/executable allocations outside that parent. Use the existing layout and quote APIs rather than treating this description as a replacement sizing formula.

## Implementation stages

Each stage records a bounded failing case before its fix, then green coverage and the relevant regressions. A compiler error can establish a missing seam, but cannot substitute for a behavioral failure. OOM, unrelated fixture failure, malformed oracle input or process exit alone is not an expected numerical or ownership red.

### DV0 - Pin artifacts and establish native image-conditioned controls

Status: complete for initial native characterization. The approved Q2_K_S-MIX file at revision `2b4527438b07460dcf27713107aedf0c6853e850` has SHA256 `0c1a01efa952a752b8bf139e8beb934d07278422eefc2e0ba968a75ee8bed528`, matching tokenizer/feature geometry and the required `[64, 0, 0, 0]` draft RoPE sections. Its actual native GPU model buffer is 524.80 MiB. At context 2048 and b/ub256, F1 and F3 each complete four ordinary native controls (text, image, repeated image and two images), with actual drafting and 16 outputs per request. F7 cannot allocate the 927588992-byte projector buffer; this capacity result is preserved, not a numerical or memory pass. Its approved split reference remains a later F7 qualification gate.

The native-launch behavioral red had two missing-sidecar assertions and three invalid-configuration errors; all 11 host harness contracts now pass. Fresh CPU ownership/speculation suites pass 1807/484 assertions. A coherent 153-host-object build reuses one unchanged CUDA test helper and the qualified device libraries. The first transient-service preflight failed on Podman state; its evidence remains separate. The corrected inherited-scope run drained all three jobs and restored the same four service IDs/configurations with healthy inference. Raw receipts are local under `benchmarks/results/dflash2-vision-20261101/`, including `dv0-audit.json`; no benchmark data is added to the source tree.

Deliverables:

- Pin source, target, each sidecar, projector, tokenizer/template, fixtures, preprocessing, KV types and request settings. Complete the smaller-draft audit above, including image-relevant RoPE metadata and weight-kernel admission, before its capacity test. Reuse deterministic image fixtures and existing F16 projector references; do not introduce a new quality dataset implicitly.
- Capture ordinary native DFlash behavior for text-only, image-first, image-last/image-only where admitted, interleaved text/image and repeated images, at proposal lengths 1, 3 and 7 where native resources fit.
- Observe actual target rows/positions, injected versus skipped batches, feature-layer order, sidecar visible window, predictions and target verification widths.
- Characterize the last-text-feature behavior immediately before/after an image and the one-row image-tail case. Distinguish an intentional skip from missing callback execution.
- Record actual native capacity failures before scheduling an expensive matrix. The same target, sidecar, projector, native KV and rollback allocations may not coexist on the reference GPU even when phased streaming fits.
- First attempt full native image-server controls with the approved smaller DFlash checkpoint. If capacity still fails, record the exact gap and use the approved separate fresh-process encoder export and native decoder/DFlash import, preserving image embeddings, positions, weights and the full-vector oracle. This establishes component and native-arithmetic coverage, not a full ordinary-native server pass. A phase-managed native control also needs its own explicit label.
- Do not silently enable UVM/CPU offload, change target/projector quants or relabel component controls as whole-server equivalence. An independent failure of the native decoder stage remains a blocker to diagnose; splitting the encoder does not guarantee the decoder fits.

Exit: a reproducible native policy and fitting reference exist, with failure causes and unsupported native cases explicit. No streamed or public-API support claim yet.

### DV1 - Reproduce admission, feature and ownership barriers

Status: complete as a controlled red stage. The opt-in `--dflash-vision-readiness` fixture passes text feature maintenance and an actual streamed two-row M-RoPE embedding forward, then fails exactly three assertions: image provenance, native image-skip callback availability and bounded-writer vision admission. The same-position changed-image negative remains rejected. The run reports 1538 assertions, three intended failures and zero exceptions, with all owned work drained and the original services healthy. This is preserved behavioral red evidence, not a positive feature qualification. Existing admission, parent-growth rejection and owner tests remain in force.

Add focused tests in existing suites that separately demonstrate:

- The current startup and request guards reject mmproj+DFlash for the documented reason.
- A valid streamed image-derived batch is rejected by token-only feature provenance, after a valid text/native control.
- A populated DFlash bounded writer is rejected for vision suspension even though the necessary fixed regions exist.
- A growth request with an attached child rejects without moving resources or changing content; it must not be mistaken for backend OOM.
- Wrong image identity at equal positions, wrong layer mapping, stale generation and held reader/copy pins reject without public output or source mutation.

Public admission stays closed. Do not weaken correct invalid-input tests to make new positive tests pass.

Exit: each barrier has an independently attributable red and the valid setup preceding it is demonstrated.

### DV2 - Measure image-shaped target work before child attachment

Status: complete for the admitted startup catalog. A private four-argument factory carries mmproj intent while preserving the old two/three-argument interfaces. Image-input target graph quotes are included before serial child registration, with feature outputs enabled and decode reservation intent restored before TG measurements. Changing image intent invalidates the old reservation. Actual 64/256-microbatch controls measure 3/6 distinct image widths and execute the image target forward, then reach exactly the three deliberately unfixed DV1 barriers. CPU owner/speculation regressions pass 1807/484 assertions. The missing-overload compile red and the reservation controls remain local; no public image admission is opened by this stage.

Implement the startup ordering and measurement contract above using existing graph reservation and layout infrastructure.

Required cases:

- Token versus embedding input, Qwen M-RoPE positions, feature extraction enabled and disabled controls, zero-logit image rows and required final text outputs.
- At least two admitted microbatch sizes and ragged image chunks, including 1, microbatch-1, microbatch and microbatch+1 total image rows when actual preprocessing can produce them.
- F=1/3/7 with actual target widths TG1-TG8; include widths 5-7 rather than assuming TG8 is the worst footprint.
- Exact-fit, one-byte-short, alignment/overflow and constructor failures before/after sidecar load, with unchanged target state on rejected configurations.
- First image, an image after long generation, and a larger subsequent image: no unbudgeted late graph allocation or unsafe parent replacement.

Keep parent identity and fixed addresses stable after child registration. Any newly discovered shape must be admitted within the declared profile or fail with an actionable diagnostic before touching retained data.

Exit: supported image/text graph shapes have explicit quotas and execute inside the existing parent; old text-only minima and behavior are either preserved or transparently measured and reviewed.

### DV3 - Make streamed feature provenance compatible with the native image policy

Status: complete for the coordinated feature contract. Image-enabled targets retain owned input F32 bytes, input kinds/offsets and all M-RoPE coordinates for the last completed frame; matching is byte-exact rather than hash-only. Token-only paths keep their existing row metadata and do not copy embedding vectors. New forwards and full/sequence restores invalidate prior frames. Actual 64/256 controls now pass image provenance, changed-pixel/spatial/signed-zero negatives, original-byte restoration and stale-frame checks; only the deliberately pending owner-admission assertion remains. A one-row image tail additionally reaches real native-position draft injection, followed by text continuation. Ordinary and strict runs pass these checks; both strict component summaries report zero errors with only the known owner red. CPU owner/speculation regressions pass. These are frozen-input component controls, not actual encoder/public streamed qualification.

Implement a minimal batch/provenance contract rather than disabling `serial_features_match()`.

Required checks:

- Correctly recognize supported native-skipped image batches and preserve the same DFlash cache effect as native execution.
- For consumed features, verify producer completion, layer ordering, row association, sequence, generation and relevant multimodal positions/source identity.
- Use the existing extraction completion boundary; keep feature storage alive until injection completes.
- Preserve full-batch offsets across target microbatches and child injection chunks; getters currently retain all input rows independently of logits selection.
- Never substitute raw projector embeddings for extracted target-layer features.
- Reject same-size wrong-image data, stale callback views, changed row order, wrong spatial positions and invalid extents transactionally. Exercise normal text immediately afterward and compare native values.

If correcting an existing image skip predicate changes native behavior, separate that fix and its native qualification from the streaming adapter work. Preserve arithmetic and position conventions.

Exit: frozen-image and real-image target batches have native-matching feature/injection behavior, including required clean rejections.

### DV4 - Qualify DFlash writer suspension and vision loans

Extend owner admission only for the validated parent/child arrangement. The child's existing bounded-cache lease and registration must prove ownership; a generic boolean that admits arbitrary children is insufficient.

Required tests:

- A populated draft cache, persistent staging and target state survive suspension, writes through the entire permitted vision loan and resumption byte-for-byte.
- Exact maximum loan, one-byte-over, alignment, overlapping persistent regions, foreign parent/device, wrong child role, extra unsupported child and stale generation.
- Live execution pins, queued copies and a pending selected recurrent restore: either complete the required dependency or reject without mutation.
- No serial DFlash prediction or target compute can execute while vision owns aliased scratch.
- A failed child construction/registration leaves no dangling role, pinned lease or altered prefetch intent. Child/parent teardown in each supported order is safe.
- Measure, lend, projector reload, encode, workspace release and text-resume failures. Temporary grants must return, or the owner must close admission until reconstruction; no stale graph can execute after failed recovery.
- Repeat small/large/small vision loans and retain both parent address and persistent cache bytes. Observe generation/revision changes only where required.

Use existing `quote_suspended_loans`, `lend_suspended`, serial transitions and completion mechanisms. Never mark a DFlash writer as a Gemma-style shared-cache reader to pass admission.

Exit: the live three-scheduler lifecycle is qualified with no second device parent, fixed-region overwrite or pending-copy use-after-free.

### DV5 - Integrate the internal image-conditioned speculative loop

Use the existing MTMD post-target-decode callback and stock speculative driver. Initially expose this only through focused test entry points; public startup/request guards remain closed until the internal path is green.

Compare ordinary native and streamed runs under identical input, batching, position and verification geometry. Check complete relevant vectors and metadata, not just argmax tokens:

- Target selected-layer feature rows that are consumed, all target logits/hidden rows needed by verification and the DFlash selector/hidden outputs used to produce proposals.
- Target KV, bounded draft KV and recurrent state at committed frontiers, using existing framed oracle and bit-exact helpers.
- Every accepted-prefix length 0..F, correction/bonus tokens, empty prediction, short/early-stop blocks and context-end headroom.
- Image-conditioned text followed by rejection, then another image and another draft. The target remains the verifier; required DFlash feature maintenance still follows successful target forwards.
- Explicit image-skip counts and later feature continuity so a silent target-only fallback cannot pass as DFlash execution.

Read-only observation must complete producer work before examining host buffers. Do not hide numerical mismatches by skipping a shape, adding tolerances, changing splits or comparing different native batch layouts.

Exit: fitting native/internal streamed correctness is established for image-conditioned drafting and forced rejection, separately from public API behavior.

### DV6 - Open scoped startup, requests and the image-aware finder

After DV2-DV5 pass, update common, server startup, owner capability and per-request guards together for pure DFlash2+mmproj.

- Validate the actual target/sidecar/projector contract, F bounds, output capacity, image extents, effective microbatches and inherited cache formats.
- Request parameters cannot enable another drafter, change model identity, exceed startup proposal capacity or override draft KV types.
- Preserve target-only/MTP image admission and existing text-only DFlash+K4V. Mixed K4V plus mmproj remains explicitly rejected until the separate extension below passes.
- Extend the existing image-aware arena finder and existing public vision harness instead of building a new server-AB framework. Invalid combinations fail before starting a process or allocating weights.
- Image preflight retains the actual complete supplied image request. A configured projector with no image remains text execution, not encoder qualification. Search proofs, selected full validation and performance timing stay separate.
- Require real DFlash predictions and completed output accounting in positive controls. Rejections identify their actual cause; a device fault or incomplete response cannot count as a successful capacity test.
- Keep Linux/Windows executable, quoting, UTF-8, process cleanup and port-isolation behavior. Portable examples start their own server and do not assume a production container exists.

Exit: real image requests through the common public API work in the admitted pure DFlash profile; unsupported combinations still reject safely.

### DV7 - Complete image-aware lifecycle and failure recovery

Extend the existing slot/checkpoint/cache behavior for the combined path without introducing another serialization system.

Required scenarios:

- Fresh and cached requests, same image, same text with a changed image, changed image order, adjacent images and text between images.
- Small/large/small image requests, an image after substantial generation and repeated turns crossing the draft window and target rollback headroom.
- Cancellation/disconnect during encoder execution, image decoding, draft prediction and verification; the following clean request must match its own native reference.
- Save/restore, erase/restore, process restart and a changed request after restoration. Preserve both target and child state/frontiers and reconstruct disposable feature provenance appropriately.
- Truncated/corrupt state and stale feature records: exact rejection or explicit uncached reconstruction, with the previous state's survival claimed only when actually demonstrated.
- Malformed images, invalid dimensions/byte counts, image-budget and context limits, missing projector data, unsupported audio/video and unavailable old history.

Compare each cached/restored scenario with its matching native scenario, not an assumption that fresh and cached native layouts always produce identical outputs. A rewind into the middle of an image remains invalid. Never weaken old-prefix or bounded-window availability guards to force a cache hit.

Exit: image changes and state transitions cannot reuse stale features, draft KV, acceptance state or projector grants; long streamed-only tests are labeled self-consistency.

### DV8 - Complete numerical, memory and performance qualification

Run a staged matrix, preserving successful artifacts and expanding only after smaller controls pass.

| Dimension | Required coverage |
| --- | --- |
| Models | Pinned Qwen target/projector, approved smaller DFlash checkpoint with same-checkpoint native/streamed controls, and original Q4 draft regression coverage; target-only/MTP vision remain controls |
| Proposal widths | F=1/3/7 model controls; forced actual TG1-TG8 and all accepted prefixes in focused tests |
| KV formats | Q8_0/Q4_0, Q8_0/Q8_0, Q5_1/Q5_1 with matching native controls where physically fitting |
| Image shapes | Small/large, portrait/landscape, multiple/adjacent/interleaved, ragged decoder batches and one-row tails |
| Memory | Exact quota boundaries, persistent state, phase high-water marks, returned loans, cancellation and reconstruction |
| History | Short native-fitting controls, window-crossing generation, real long target-ring use and later-image handoffs |
| Supported-mode regressions | Text-only DFlash/K4V, embedded MTP, target-only vision, Gemma vision/shared-KV MTP and common owner tests |

Strict memory checks retain default PDL, API-all reporting, error exit 99 and no filters. Eager execution is limited to sanitizer runs; ordinary execution retains normal graphs. Require process success, completed numerical checks and explicit zero-error summaries. An instrumentation OOM/timeout is not a pass.

Use fresh-process native/export and streamed/import controls when needed, with validated framed oracles and exact input/output hashes. Prior sanitizer-only arena exceptions for other features do not automatically authorize a different DFlash vision profile. Preserve a failing profile, diagnose the resource/tool cause and obtain a decision before a material profile change.

Record total VRAM and host memory separately from the arena: sidecar/projector weights, recurrent snapshots, feature outputs, image embeddings and driver overhead all matter. Check idle baselines after repeated phase returns for retained resources; zero sampled free-VRAM difference is not proof of zero driver cost.

Performance measures ordinary uninstrumented requests. Separate image preprocessing/encoding, projector reload, target image/text prefill, drafting/verification, decode and total request latency. Record accepted tokens, actual neural forwards, target rounds, cache work and output counts. Compare the new text path with its pre-feature baseline, and image DFlash with matching image target-only/MTP/native controls as applicable. Fixed-arena and independently maximum-fit comparisons are separate cohorts.

Changing Q4 draft weights to Q2_K_S-MIX can change acceptance, prediction work and the maximum-fit arena. Keep weight-encoding comparisons separate from code-only optimization comparisons; publisher text benchmark results are not image-acceptance or speed guarantees on this GPU.

Report whether image-conditioned drafting actually reduces end-to-end work. Speedup over every alternative is not a functional-support requirement, but a severe regression needs diagnosis and explicit release scope. Poor acceptance must not be hidden by synthetic repetition, constrained sampling or a smaller image. Repeated timing samples are needed for small gain claims; profiler timing is never ordinary speed evidence.

Reuse source/binary-identical lower-level controls only with explicit exact provenance and unchanged commands/profiles. No CUDA arithmetic change is planned, so do not trigger a blind full legacy shader rebuild or remeasure clean compilation time. Run the affected host/API/platform builds and the existing reader guards appropriate to actual changes.

Exit: required functional/safety matrices pass, existing modes have no material unexplained regression, capacity limits and image performance are accurately scoped, and owned resources are drained.

### DV9 - Document and close pure DFlash2 vision support

- Update concise README concept/support/run entries and the DFlash guide; keep detailed measurements and implementation contracts in a linked vision report.
- Show the existing image-request API, a matching projector/sidecar launch, inherited cache types, explicit F, image-aware finder usage and the distinction between configured-projector text checks and real encoder execution.
- Document native-compatible image skipping, bounded draft state during vision, startup image-graph measurements, quota causes and all unsupported modes.
- Record tested hardware, image/batch/context limits, native versus streamed-only scope, strict-memory scope and actual acceptance/latency findings. Do not promise generic image-task acceleration or copy prior text fit budgets into image examples.
- Verify local links, command syntax and unchanged upstream README bytes. Keep descriptions merge-ready; commit subjects describe behavior in one sentence rather than using stage codes.
- Add the completed-stage ledger only after all mandatory checks pass. Public K4V+mmproj admission remains closed unless its independent extension is complete.

Exit: pure DFlash2+mmproj is reviewable and documented with accurate support boundaries; temporary services and GPU work have ended and any managed services are restored to their original healthy state.

## Separate follow-up: DFlash2 + K4V + mmproj

This follow-up is planned but is not a dependency for declaring pure DFlash2 vision complete. It must not be enabled implicitly by DV6.

### Mixed-mode contract

- Keep F and copied maximum K independent in 1-7; target verification/rollback capacity remains `max(F,K)+1` rows and `max(F,K)` candidate states respectively.
- K4V remains a CPU text-token map. Image embeddings and placeholder runs must not become ordinary copied candidates or image lookup keys.
- The initial conservative design is to prevent lookup keys or copied continuations from crossing a media boundary. Prefer a source-specific lookup view over changing the shared DFlash prompt or global target positions. Determine whether the existing map can enforce this using a post-last-image text segment before adding any new state.
- `dp.pos0` still comes from the actual target decoder frontier, never the length of filtered text or a K4V segment.
- Rebuild disposable map/acceptance state on generation replacement and relevant image changes. Preserve valid textual lookup within the admitted segment; no new GPU map or persistent second history is required.
- DFlash receives required target features after every successful verification even if K4V supplied the proposal. A lookup hit only skips neural prediction, not target verification or feature maintenance.

### Mixed-mode TDD and acceptance

1. CPU red cases: an apparent n-gram formed only by removing an image, a continuation spanning an image, identical text with a changed image, reset/restore and empty post-image text. A genuine repeated suffix after the image remains a positive case.
2. Common/request red cases: mixed vision remains closed until all limits, media identity and source-specific history contracts are validated. No forced generic read-only role or shared prompt-length substitution.
3. Internal controls: force K4V -> DFlash -> K4V, both-empty, rejection and every accepted prefix, including F3/K7 and F7/K3 with images. Verify supplying-source feedback and image-conditioned target/draft state.
4. Public API/lifecycle: real copied and neural proposals after images, changed-image/cache/save/restore/cancel behavior and exact per-scenario native controls.
5. Strict memory and ordinary performance: use the same reporting and failure rules as DV8. Copying/resource proofs are explicitly separated from natural image-task timings; no broad K4V gain is assumed.
6. Open only the matching mixed image pairing after these gates pass, then document its independent scope.

If boundary-aware lookup cannot be added without changing native shared history semantics, keep mixed vision closed and present the concrete API/design choice rather than weakening correctness checks.

## Expected file boundaries and review units

| Area | Prefer extending |
| --- | --- |
| Common admission/limits | `common/common.cpp`, existing declarations and speculative parameter helpers |
| DFlash features and source routing | `common/speculative.cpp`, minimal existing feature/provenance metadata |
| Context graph sizing and feature capture | `src/llama-context.cpp` and its private declarations |
| Parent/child phase lifetime | `src/llama-context-memory.cpp/.h`, existing bounded-cache and lease infrastructure |
| Server startup/requests/cache | `tools/server/server-common.cpp/.h`, `tools/server/server-context.cpp` |
| MTMD phase/execution | `tools/mtmd/mtmd-session.cpp`, `mtmd-workspace.cpp`, existing MTMD planning and helpers only when required |
| CPU/owner/numerical tests | Existing `tests/test-context-memory.cpp`, `tests/test-speculative.cpp`, `tests/test-kv-stream-context.cpp` and private DFlash fixture helpers |
| Public API/harness contracts | Existing `tools/server/tests/test_adaptive_vision.py`, its host harness tests and corresponding server tests |
| Finder | `benchmarks/find-max-arena.py` and an appropriate tracked test home outside the excluded server-AB tree |
| Documentation | README summary, a focused multimodal report, existing support guide and this ledger |

Suggested review boundaries are startup/provenance, persistent owner handoff, internal numerical integration, public requests/lifecycle and final documentation. Keep independent behavioral fixes separate when a native bug is discovered. A finished stage is not permission to publish or push; commit actions follow the user's instructions.

## Execution and evidence discipline

- Preserve unrelated files, the deferred K15 work and existing qualified runtime archives. Start implementation on a suitable working branch when authorized; do not edit source while a runner pins that source/binary cohort.
- Use the existing test and workload machinery. Follow CPU red/green, focused model controls, strict checks, public lifecycle and then performance; do not repeatedly launch the entire matrix after an unrelated harness failure.
- Preserve original failures, exact assertions, command/environment/model/library/image hashes, full responses and accepted partial results. Numerical mismatches require first-difference analysis; no tolerance/API relaxation, silent profile changes or fixture enlargement.
- Raw benchmark results, generated JSON/CSV data and server-AB test/harness changes remain local and are not added to the PR. Keep reusable source tests in the agreed tracked suites. Do not move raw data into another tracked directory to bypass this boundary.
- New request/server helpers must own their processes, refuse occupied ports, drain all descendants and leave no GPU worker behind. Record the primary error independently from cleanup/restoration failures.
- Ordinary examples never assume a production service. In the managed local qualification environment, stop only services covered by authorization, preserve exact IDs/configuration and restore their pre-test healthy state afterward.
- No agents, duplicate GPU jobs, pushes, production model/configuration changes, new codegen packaging or clean compile-time benchmarks. No prior artificial host-memory limit is reinstated by this plan.
- Avoid speculative performance claims. Same-mode exactness controls, cross-mode throughput, task quality, phase timing and hardware counters answer different questions.

## Completion checklist

- [x] DV0: artifacts and initial native image-conditioned controls pinned; F7 capacity limit preserved.
- [x] DV1: image provenance/callback and writer admission barriers reproduced after valid controls; existing guards retained.
- [x] DV2: image-input startup catalogs measured before child capture; both microbatch profiles validated.
- [x] DV3: native-policy feature behavior and exact owned image provenance pass, including one-row injection.
- [ ] DV4: persistent DFlash state and completion fences survive vision loans.
- [ ] DV5: image-conditioned speculative acceptance/correction matches fitting native controls.
- [ ] DV6: narrow common/server/request/finder integration works with actual images.
- [ ] DV7: cache, window, cancellation, state and process restart pass.
- [ ] DV8: required numerical/memory/regression/performance evidence audited.
- [ ] DV9: pure DFlash2 vision documentation and release scope complete; local resources restored.
- [ ] Optional mixed extension: media-bounded K4V separately qualified and documented, or explicitly left closed.

Pure DFlash2 vision is complete only when DV0-DV9 pass. A helper subgate, successful encoder call, configured projector, target-only fallback, or process exit does not satisfy whole-feature acceptance. Optional K4V vision has its own completion boundary.
