# DFlash2 vision phase-arena implementation roadmap

Status: DV0-DV26 complete, 2026-11-03, for the scoped pure/mixed Qwen DFlash2 still-image paths and common Qwen/Gemma controlled CUDA encoder workspace. Optional CPU K4V uses checked post-last-image text, fixed server policy and independent 1-7 limits. Numerical, device/host safety, lifecycle, sizing, regression/performance and documentation checks are complete. Historical stage receipts retain their measured source/binary identities; they are not relabelled as new passes after later changes.

Implementation extension: **DV18-DV26**, Qwen DFlash2 + K4V + mmproj. DV18-DV26 are complete, including image-aware lookup, internal/public/lifecycle controls, final safety/performance qualification and documented release limits. See the [implementation record](#k4v-vision-extension-dv18-dv26).

Initial DV0-DV9 source baseline: merged Adaptive KV Streaming V2 at `7425d14c3ae0bc357b2e5f51edff15fabbe17f70`. Recheck the named mechanisms when implementation starts. Historical evidence retains its measured source and binary identities when later Git metadata changes.

## User decisions

Recorded on 2026-11-01:

- Qualify pure DFlash2 plus mmproj first; K4V vision is a separate follow-up, not a condition for the first release.
- Correct image support with clearly reported performance is the release objective. A speedup over MTP3 is not required; existing supported-mode regressions still need investigation.
- For reference capacity, first try a validated smaller DFlash quantization, with the user-suggested HermiHg Q2_K_S-MIX checkpoint as the initial candidate. If the full native target/projector/smaller-draft configuration still cannot fit, split-process encoder/native-decoder references are approved, with their component-level scope stated explicitly. The delivered server remains a single process. Use the same draft checkpoint on both sides of each native/streamed comparison; target/projector weights and numerical checks do not change.

Extension decision, 2026-11-01: add a roadmap for arena ownership of application-controlled CUDA operator temporaries and explicit cuBLAS workspace, shared by Qwen and Gemma. Preserve the one-parent design and exact native arithmetic. Driver and opaque library state need separate accounting; planning this extension does not establish its runtime acceptance.

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

Status: complete for bounded-writer arena ownership. Serial DFlash construction registers the actual persistent cache lease by parent identity, resource/region, bounds, buffer type, base and size. Writer and shared-reader roles are mutually exclusive; failed null/workspace role declarations preserve the existing writer. The owner admits exactly one validated writer without auxiliary MTP. Actual 64/256 controls pass 1592 assertions each, including exact maximum loan and one-byte rejection, full-prefix GPU overwrite, byte-identical target/draft snapshots, retained-loan resume rejection and two successful resume/continuation cycles. Strict memcheck reports zero errors with all assertions passing. CPU owner/speculation regressions pass (1809/484 assertions), and original services are healthy. Actual projector failure/lifecycle and full-vector controls remain DV5-DV8 requirements.

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

Completed 2026-11-01: the existing framed oracle now exports real MTMD embeddings, tokens and four-coordinate positions and compares all consumed target features, logits/hidden rows, selector outputs, encoded target/draft KV and recurrent state. F1/F3/F7 cover all accepted prefixes and actual TG1-TG8, including phase-loan overwrite/resume; F3 also covers Q8_0/Q8_0 and Q5_1/Q5_1. Exact records are 504 (815636404 bytes) for F1, 907 (1474031420 bytes) for F3 Q8_0/Q4_0, and 1749 (2834713708 bytes) for F7. Three fresh strict decoder checks and the phase encoder check report zero errors with API-all/error99/default PDL. The native encoder and decoder components use the same image payload and weights; F7 is component-level native arithmetic coverage because full native F7 still cannot allocate the 927588992-byte projector. A corrected local builder rebuilds every static server-archive member; fresh coherent F1/F3 public native controls reproduce all prior tokens and F7 reproduces its capacity limit. Decoder CLI evidence does not link that archive. CPU image-framing checks pass with 1531 assertions. Local evidence: `benchmarks/results/dflash2-vision-20261101/` (encoder-input, image-vectors-v2, image-matrix and native-coherent gates).

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

Completed 2026-11-01: common and server admission allow pure DFlash1-7 with a matching external sidecar, inherited target cache formats and the registered bounded-writer owner. Mixed K4V vision, model/cache overrides, over-width requests and unsupported serial/phase configurations still reject. CPU red tests produced exactly 18 server failures, three common failures and one finder rejection before the narrow changes; green has 383 server assertions, 493 common assertions and 12 Python tests. Real public F1/F3/F7 each execute text, image, repeated-image and two-image requests with actual predictions. F1/F3 tokens and draft/accept/prompt counts exactly match their corresponding native requests; F7 public execution is paired with the separately qualified native-decoder reference, not a fitting full-native projector control. The existing finder completes an actual image request at 1024 MiB and returns 16 tokens with real drafting; this fixed-budget probe is not a maximum-fit search. Its first local driver used an image-token ceiling larger than the context and was rejected before model execution; the corrected driver sets a consistent ceiling without changing the actual image or request. Local public/finder cohorts preserve full provenance, original failures and healthy restoration of all managed services.

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

Completed 2026-11-01: expanded F1/F3 controls each compare 23 native and 23 streamed scenarios with exact token IDs and prompt/cache/draft/accept counts (5888 output IDs combined). They include image-only/first/last, portrait/narrow, changed/adjacent/later images, malformed image/audio/file-URL/context rejection, cancellation, complete state save/erase/restore, truncated-state recovery and restart. Corresponding 321935388-byte native/streamed state files have identical SHA256 digests in each mode. A separate F7 65536-capacity/48000-text-prefix/512-output/1024-MiB arena run passes 23 streamed-only scenarios, 11776 IDs and four state actions; its 1799655104-byte state contains 48584 tokens. Initial, state and restart logs prove real nonresident decode traffic (188-191 active pages, 66 ring slots). This long profile establishes lifecycle self-consistency, not native equivalence. The first long driver failed the mandatory RAM-cache proof because a 2 GiB host cache evicted its 1.84 GiB entry when intervening images arrived. The successful rerun keeps every assertion, image, history and GPU allocation unchanged and explicitly uses a 4 GiB host prompt cache; short profiles retain 2 GiB.

An existing projector test now attaches the real bounded DFlash writer and checks reload cancellation, an exception after encoder work is queued, no failed-output publication, byte-identical target/child state, successful exact encoder retry and eight native-exact continuation vectors plus real matching proposals. Its 93 assertions pass ordinarily, under ASAN/UBSAN and under strict device memcheck with zero errors. Thirteen host ASAN/UBSAN suites also pass; an obsolete all-mmproj-rejected argument assertion was updated to admit pure DFlash while retaining mixed-vision rejection. CUDA ASAN requires NVIDIA's documented `protect_shadow_gap=0` compatibility setting on this driver: an untraced no-model probe reproduces default initialization OOM and succeeds with that setting while retaining leak detection, UBSAN and halt-on-error. CPU ASAN suites retain their defaults. The first straced probe only established the known LSan/ptrace incompatibility and is not memory qualification. No numerical, API-reporting, arena, model or quantization checks were relaxed. Local evidence retains original failures, exact binaries, command/profile hashes and healthy service restoration.

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

Completed runtime qualification 2026-11-01: sixteen fresh device/model gates cover CUDA owner/workspace/recurrent recovery, graph-enabled owner memcheck, Qwen MTP exact vectors with both target and auxiliary ring traffic (2666 records), original Q4 DFlash and K4V switching/all widths, public F3 Q8_0/Q8_0 and Q5_1/Q5_1 image pairs, existing Qwen target-only/MTP3 and Gemma target-only/shared-KV MTP3 vision, fresh F7 image-vector memcheck, F3 image-vector ASAN, and the complete encoder/target/draft public path under strict memcheck and ASAN/UBSAN. All three strict logs report zero errors with unchanged API-all/error99/default PDL; the owner keeps CUDA graphs and eager execution is checker-only. Fourteen host ASAN/UBSAN suites pass, including the 139-assertion projector suite. The first shortened MTP recipe matched its vectors but correctly failed the required nonresident proof after decode reclaimed workspace; restoring the existing 16381-token/eight-round recipe, without changing its 32 MiB bootstrap pool or assertions, proves both rings. Original failing artifacts remain preserved.

Forty-four ordinary SWE benchmark requests form 22 baseline/current pairs with identical same-mode output IDs (11264 total), prompt/draft/source counts and physical decode layouts. The baseline is a coherent isolated build of 7425d14c3, including every static server-archive member. Three near-8-Ki pairs per Qwen mode give decode changes of -0.02%, +0.29%, +0.04% and +0.09% for target-only, MTP3, original Q4 DFlash7 and DFlash7+K4V; prefill changes stay within 0.09%. Three Gemma pairs use its own template/tokenizer (8191 tokens versus Qwen's 8186), with decode changes +0.17%/+1.24% and prefill within 0.07%. The four single 64524-token Qwen controls vary from -0.54% to +0.31% decode and within 0.05% prefill; they are not repeated statistical claims. No material regression is observed, and positive variation is not claimed as an optimization.

Ordinary fixed-1-GiB image runs use one warmup and three uncached 256-output samples per profile, plus two alternating-order target-only process pairs to investigate an initial small-prefill difference. Across three target-only process pairs, median changes are -0.35% prefill, -0.08% decode and +0.18% total latency, with individual total changes within 0.41%; the initial 1.3% prefill sample does not reproduce consistently. Existing MTP image work/outputs/layouts match, with paired decode change +0.01% and total latency +0.04%. The new Q2 DFlash3/7 profiles measure 71.95/63.62 tok/s and 3.91/4.39 s median request latency versus existing MTP3 at 72.77 tok/s/3.87 s. These pattern fixtures establish support and acceptance costs, not a universal image speedup or quality certificate. Normal warmup startup also has fitting full native/streamed image controls.

A final test review found that the old vision test harness accepted any failure log containing a vision banner. Six red subcases now prevent device faults/OOM from being counted as configuration rejections and recognize the specific DFlash guards; 17 host tests and nine real startup rejections plus native-matching recovery pass. This changes only the expected-rejection branch; an AST comparison proves all measured positive harness paths unchanged. The 45 Gemma host-harness tests and 132 other tracked benchmark host tests pass. With the user's explicit one-line exception for the existing benchmark test, the obsolete pure-DFlash image rejection now tests mixed DFlash+K4V image rejection instead. All 31 finder host tests complete successfully with one platform skip; the current pure/mixed contract also has the separate server-harness coverage. No other benchmark harness files or raw data are added, and numerical/API checks are unchanged. Raw data, benchmark harness additions and source/binary mappings remain local; measured binaries retain their actual pre-commit identities. Managed services are restored with unchanged IDs/configuration and healthy port 1234.

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

Completed 2026-11-01: the README gives brief pure-DFlash image support and the shared launch flow; the linked docs/multimodal/dflash2-vision.md report records pinned artifacts, one-parent lifetime/ownership, image-aware sizing, native versus component references, lifecycle/memory coverage, actual acceptance/latency and regression measurements. The K4V guide keeps mixed vision closed and links the pure path; infrastructure/consumer and original DFlash roadmaps identify their dated status as historical. No benchmark harness additions or raw measurement data are committed. Documentation validation checks 174 local links/anchors, 61 shell blocks and five finder examples. Existing Mermaid blocks are byte-identical (no new grammar/rendering claim). The original 7640-byte upstream README section retains SHA256 394fdd330057ca2d50a84097f2ae438ceeb70a1138a4fdc9cc301b79b0596ad6. Current vision/Gemma host contracts pass; the approved finder-test correction also passes all 31 tests with one platform skip. All owned GPU scopes are inactive and the original four managed services are restored healthy with unchanged identities/configuration.

- Update concise README concept/support/run entries and the DFlash guide; keep detailed measurements and implementation contracts in a linked vision report.
- Show the existing image-request API, a matching projector/sidecar launch, inherited cache types, explicit F, image-aware finder usage and the distinction between configured-projector text checks and real encoder execution.
- Document native-compatible image skipping, bounded draft state during vision, startup image-graph measurements, quota causes and all unsupported modes.
- Record tested hardware, image/batch/context limits, native versus streamed-only scope, strict-memory scope and actual acceptance/latency findings. Do not promise generic image-task acceleration or copy prior text fit budgets into image examples.
- Verify local links, command syntax and unchanged upstream README bytes. Keep descriptions merge-ready; commit subjects describe behavior in one sentence rather than using stage codes.
- Add the completed-stage ledger only after all mandatory checks pass. Public K4V+mmproj admission remains closed unless its independent extension is complete.

Exit: pure DFlash2+mmproj is reviewable and documented with accurate support boundaries; temporary services and GPU work have ended and any managed services are restored to their original healthy state.

## CUDA operator-scratch extension: DV10-DV17

Status: DV10-DV17 complete, 2026-11-02. This extension adds common CUDA workspace coverage for Qwen and Gemma without reopening DV0-DV9 or admitting K4V images. The starting source was `5927dd16b` plus the reviewed README follow-up. Each completed stage has its own descriptive commit and audited source/binary identity; compiled runtime metadata is retained when a later commit changes only documentation.

### Historical starting allocation boundary, before DV10

At the start of the extension, the automatic finder measured native image geometry and encoder graph storage, executed all seven source aspect ratios, and validated one common budget with serial image requests. At the measured maximum arenas, Qwen DFlash3 fits 2066 MiB with common image budget 48; Gemma MTP3 fits 2410 MiB with common budget 143. Those are executed capacity observations, not model quality scores, intrinsic resolution limits or full-262144-token prefill passes. See the [capacity report](docs/multimodal/vision-capacity.md).

Larger Qwen image probes passed the current graph/weight arena admission, then failed in `ggml_cuda_mul_mat_cublas` -> `ggml_cuda_pool_vmm::alloc` -> `cuMemCreate`. This established a real operator-private allocation outside the parent. DV10 subsequently measured its sizes, lifetimes and peak contributions.

| Allocation class | Starting path before DV10 | Extension contract |
| --- | --- | --- |
| Projector weights and scheduled encoder tensors/attention workspace | Existing vision-phase loans | Retain the same parent and existing persistent-region protections |
| Converted matmul inputs, temporary outputs and batched pointer arrays | `ggml_cuda_pool_alloc` through the backend's per-stream pool | Quote the selected native dispatch and bind checked arena subregions |
| Explicit cuBLAS workspace | `cublas_handle()` separately allocates with `cudaMalloc` | Supply a phase-owned workspace per active stream, with explicit ownership and completion |
| Other operator-private buffers discovered in the encoder graph | Inventory incomplete | Bound and migrate every reachable controlled path before declaring full coverage |
| Driver contexts, loaded code, CUDA graph executables and opaque library state | Outside the parent | Identify and account separately; do not claim a bound on all process VRAM |

The current CUDA code selects 32 MiB per initialized cuBLAS stream on the reference GPU. Preserve that configured size during the allocation migration, together with math mode, precision, algorithms and stream topology. A different library recommendation is not permission to shrink workspace to force a fit. Any later size/performance tuning is a separate experiment.

NVIDIA documents that caller-provided cuBLAS workspace requires at least 256-byte alignment, that `cublasSetStream` resets the workspace binding, and that concurrently used streams need safe workspace separation. Bind after setting the stream and retain storage through completion. See [cuBLAS workspace](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublassetworkspace), [stream selection](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublassetstream) and [reproducibility](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#results-reproducibility). Pin the installed toolkit/library versions; these references do not replace runtime qualification.

### Common design and acceptance invariants

- Implement a backend/operator workspace contract reusable by both models, with vision as the initial managed consumer. Derive sizes from actual tensors, native dispatch, dtype, strides, batching, device capabilities and execution topology; no Qwen/Gemma-specific scratch constant or model-specific CLI switch.
- Extend the existing graph measurement, parent quote, leases and execution bindings. The current attention external-workspace mechanism does not already cover general matmul temporaries. Introduce only the missing private query/bind seam; do not create another parent, allocator, persistent GPU cache or implicit global/thread-local pool override.
- Keep native conversions, chunking, tensor layouts, cuBLAS call sequence, compute types and reduction order unchanged. Moving an address is the initial change. Workspace/alignment changes can influence library dispatch, so exact native output checks remain mandatory.
- Validate every size product, sum, alignment and subrange before arithmetic or pointer publication. Include padding and all simultaneously live input/output/pointer buffers. A maximum across operators is sufficient only when their actual queue dependencies prove reuse safe; concurrent streams need disjoint live regions.
- Lease identity includes the parent, device/buffer type, physical range and generation. Do not overlap operator scratch with live graph tensors, protected SWA, DFlash KV, recurrent publication or another active consumer. Returning a host RAII object is not proof that its queued GPU work completed.
- Arena-bound execution has no hidden fallback to the native pool or a separate device allocation. Missing support or inadequate storage rejects before encoder work, with structured required/available/additional bytes where applicable. Ordinary unbound execution retains its native behavior.
- Track owned and borrowed cuBLAS pointers separately. Never `cudaFree` an arena view, retain a handle referencing returned KV storage, or leave an old owned 32 MiB workspace allocated while claiming it has been migrated. Drain and dispose of any previously created native pool/workspace only for the affected backend owner.
- CUDA graph capture/replay must retain valid addresses, ownership and generation. Retire captures and clear/rebind borrowed library state before returning vision grants. Reuse the existing phase-return and executor protocol for success, cancellation and failures.
- Prepare the supported library/kernel/handle state before final physical fitting where possible. Record opaque initialization costs separately; warmup cannot hide a retained GGML pool reservation or excuse unexplained request-time allocations.
- Prefer private host-side adapters and the current ownership infrastructure to broad CUDA header changes. If `common.cuh` changes, its dependent shader objects are affected even when arithmetic is unchanged: record the real rebuild inventory and qualify those dependencies. Reuse legacy objects only when relevant sources/headers/commands match; use the existing serial legacy-compilation policy where needed, without claiming older physical-GPU runtime coverage. No kernel/generator packaging refactor or clean compile-time benchmark is part of this extension.

```text
Same physical parent during vision:
  protected persistent state
  + projector weights
  + scheduled encoder tensors / attention workspace
  + bounded operator-private temporaries
  + explicit per-stream cuBLAS workspace

After completion and capture retirement:
  return temporary vision grants
  -> rebuild text bindings
  -> reuse reclaimable bytes for text workspace and device KV
```

The objective is complete ownership of the enumerated application-controlled vision workspace. It does not eliminate the model/assistant weights, protected KV, microbatch limits, driver state or physical VRAM limit. Peak capacity and throughput must be measured; neither a larger image budget nor a speedup is presumed.

### DV10 - Inventory and reproduce private encoder allocations

Status: complete for characterization, 2026-11-01. The opt-in CUDA host-adapter trace records operation/source shapes, dtypes/strides, context/stream identity, logical pool requests/frees, physical growth, retained capacity and explicit cuBLAS workspace. It adds no GPU waits or changed tensor arguments. Only the CUDA adapter and one host test object were rebuilt; shared CUDA headers and large D512 kernel units were unchanged. All 139 default projector host assertions pass.

Four encoder controls match complete native F32 embeddings before exactly one intended coverage failure each; the small Qwen control also has a strict repeat with zero memcheck errors and only that coverage failure. The trace finds both `MUL_MAT` and `FLASH_ATTN_EXT` private allocations, so matmul-only migration is insufficient. Observed single-stream retained pool capacities are 4 MiB for Qwen's 64-token image, 172 MiB for its 4096-token image, 20 MiB for Gemma's 64-token image and 24 MiB for its 256-token image; each also has a separate 32 MiB cuBLAS workspace. These are retained native pool capacities, not all simultaneously live bytes or all process VRAM. The larger Qwen control is encoder-only with target weights unloaded, not full-model 4096-image-token qualification.

Five actual server profiles (Qwen target-only/MTP3/Q2 DFlash3 and Gemma target-only/MTP3) each compare four native/arena scenarios with exact token IDs and prompt/cache/draft/accept counts: 20 paired cases, 640 generated IDs. Their completed encoder windows retain private pool requests and explicit cuBLAS workspace. Test-only CUDA API interception cross-checks direct allocations without changing their arguments/results or adding queue waits. All 14 gates pass their declared control or expected-red criteria; services restore healthy with unchanged identities/configuration.

The first large Qwen export stopped at a historical fixture parser cap of 1120 before GPU work; it is preserved as a fixture failure, not a coverage red. The common fixture now admits the requested native Qwen 4096-token input range; its parent budgets, native arithmetic and byte-exact oracle remain unchanged. A fixture-only native-control build changes only that input cap. A local NVCC dependency-receipt parsing failure is also preserved separately from the successful adapter compile/link. Evidence and hashes are retained locally under `benchmarks/results/vision-operator-scratch-20261101/`; no raw results are committed.

- Archive the qualified source/binaries and actual Qwen IQ4_XS + Q2 DFlash + F16 projector and Gemma Q3_K_S + Q8 assistant/projector profiles. Retain the current 48/143-budget results and failures as historical cohorts.
- Extend existing opt-in test diagnostics to attribute CUDA allocation requests to an encoder operation, backend/context, stream, size, allocation family and lifetime. Distinguish graph storage, GGML temporary pools, explicit cuBLAS workspace and opaque initialization. Inspect retained capacity as well as allocation-call counts.
- Trace actual target-only/MTP vision for both models and Qwen pure DFlash, starting with small controls and the preserved larger-image failure. Identify all pool-allocating operator paths reachable in the supported projector graphs, including relevant fused, quantized, contiguous/noncontiguous and batched paths.
- TDD: after valid native image/vector controls, assert zero unquoted controlled scratch allocations during the managed encoder window. Expect the known private paths to fail that assertion. An OOM or missing trace alone is not this behavioral red.

Exit: a bounded allocation inventory, reproducible coverage red and per-stream peak evidence exist. Instrumented timing is not an ordinary performance result.

### DV11 - Quote operator-private and library workspace requirements

Status: complete for the declared native vision dispatches, 2026-11-01. A private metadata-only contract now describes aligned operator requests and concurrent-stream peaks, with one explicit library workspace per stream. CUDA preparation caches actual kernel occupancy/configuration; subsequent queries read tensor metadata without device payloads or intercepted CUDA allocations. Unsupported operations/geometries return explicit status and preserve caller output. Native matmul route and precision rules are shared, the native 512 MiB F32 conversion chunk is retained, and Q8 MMQ selection is shared without removing the ordinary native path for other cases.

Four small/large Qwen/Gemma encoder controls compare complete native F32 embeddings and validate every traced private allocation against its quote: 617 quoted dispatches per Qwen fixture and 937 per Gemma fixture, with zero unsupported paths or mismatches. The strict small-Qwen repeat passes all controls with zero memcheck errors, unchanged API-all/error99/default PDL and checker-only eager execution. A standalone GPU test covers preparation, payload-free F16/Q8/attention queries, unsupported and invalid metadata, plus existing scheduler controls. All six device gates pass; six host ASAN/UBSAN suites also pass, including 582 workspace assertions. Pure query windows contain no intercepted CUDA allocations. Services are restored healthy.

The first Gemma run preserved exact native embeddings but reported 326 quote mismatches: MMQ's templated Stream-K launcher needs a fixup allocation in addition to quantized input. The corrected query follows the native tiling efficiency/fractional-tile rule; a measured example adds 4587520 bytes. An earlier CPU red also caught an incorrect proposed 256 MiB conversion bound against the actual native 512 MiB limit. Both failures are preserved, and neither was resolved by changing arithmetic, reducing workspace, enlarging arenas or relaxing assertions.

Only the CUDA adapters and existing head-72 tile unit were rebuilt. A source-local compile definition enables host metadata export for kernels already instantiated in that unit; the original tile header's active prefix, common CUDA header, generator and generated files remain unchanged. Unaffected objects use the original prefix with the added EOF block inactive; this is an explicit preprocessor-identity mapping rather than a claim of raw full-header identity. All 75 original head-72 device kernel names/resource records match. CUDA queries currently cover the observed plain F32/F16 head-72 vision attention and supported float/Q8 matmul paths; this does not advertise managed decoder support for every head geometry. Binding and phase integration are still DV12-DV14. Local evidence: `benchmarks/results/vision-operator-scratch-20261101/dv11-quotes-v2/` and `dv11-audit.json`.

- Add a side-effect-free requirements query through the existing private backend/workspace planning infrastructure. Share native dispatch decisions with execution so quotes cannot drift from the actual conversion/output path. Planning reads tensor metadata and capabilities, not device payloads.
- Cover each inventoried conversion, output and pointer-array interval, the existing bounded F32 conversion chunking, active-stream cuBLAS workspace and required alignment. Keep graph storage and operator scratch visible as separate components without counting bytes twice.
- TDD in existing workspace/owner suites: exact totals and offsets; dtype/stride/batch variations; already-compatible operands; overflow and padding limits; unsupported device/dispatch; concurrent-stream sums; unchanged output on rejected measurement. No GPU allocation is allowed during the query itself.
- Validate quotes against observed native peaks and actual dispatch on fitting operator fixtures. A quote is a coverage result only for its declared paths; an unknown allocation remains a blocking gap.

Exit: complete, checked requirements exist before binding. No runtime allocation path has been silently redirected yet.

### DV12 - Bind CUDA operator temporaries to arena leases

Status: complete for bounded native operator components, 2026-11-01. A private CUDA binding retains both the root phase loan and its nested scratch lease. Per-stream adapters supply the existing native pool requests from checked offsets in that grant; ordinary unbound execution keeps its native pool. Whole-graph admission checks supported dispatch, stream quotas and payload separation before compute. Invalid IDs/generations/types/layouts and short grants reject without output; managed execution owners cannot bypass the native contract. Pending work blocks release, completion joins active streams, and captured graphs retire before leases return.

Eight final GPU gates pass with current compiled dependencies and six host ASAN/UBSAN suites pass. The operator matrix covers all 24 inventoried matmul shapes and four strided head-72 attention shapes, repeated native-exact outputs, poisoned scratch, canaries, exact-size/one-byte-short grants, nested root ownership and a failure reported after real enqueue. A native three-branch fork/join uses four distinct stream regions and matches exact output in normal and strict captured execution. The eager strict matrix and small Qwen encoder strict control also report zero errors with default PDL and full API reporting. Traces show zero native-pool requests/growth while bound; all 75 head-72 kernel names/resource records match the prior build. Four complete encoder controls still match original native embeddings in ordinary pool mode.

The initial missing-binding API red and the later managed-owner admission bypass red are preserved. Early concurrency fixtures exposed reusable scheduler inputs and a self-referencing cast root with four consumers; stable dedicated tensor storage and a supported three-consumer native fork fixed the fixtures without changing arithmetic or weakening the required four-stream check. The final source/binary audit is `benchmarks/results/vision-operator-scratch-20261101/dv12-audit.json`; gates are in `dv12-qualification-v2/`. Explicit cuBLAS workspace is still outside the arena until DV13, and production MTMD hookup remains DV14. This component stage does not claim complete vision allocation coverage.

- Supply each quoted temporary from checked views of the existing arena grant. Extend the selected CUDA host adapters while retaining the native kernels and call order. Use the native pool only for ordinary unbound execution; do not replace `ctx.pool()` globally with a new allocator.
- TDD: valid bound operations first reproduce the DV10 pool-allocation red, then execute with zero native-pool allocation/growth. Test exact-size and one-byte-short grants, wrong device/owner/generation, overlaps, misalignment, canaries, poisoned scratch, repeated operations and queued failure/retry.
- Compare complete outputs byte-for-byte with the unchanged native operator across the inventoried precision, layout and batching paths. Include each projector's actual matmul geometries rather than using only small square matrices.
- Keep distinct scratch for concurrent streams and retain it until its last consumer completes. Test host return before GPU completion, same-stream reuse, cross-stream ordering and retirement while work is pending.

Exit: inventoried operator temporaries use bounded parent storage with exact native outputs, and insufficient/invalid bindings cannot allocate around the quota.

### DV13 - Make explicit cuBLAS workspace phase-owned

Status: complete for borrowed explicit cuBLAS workspace, 2026-11-01. Active handles use the quoted per-stream library regions with the native workspace size and TF32 math configuration. Borrowed pointers never enter the backend's owned `cudaFree` array. Quiescent native reservations retire before successful managed execution; preparation can initialize opaque handle state without allocating the explicit workspace. Returning to ordinary execution restores native owned workspace before capture. Pending work blocks retirement, captured graphs retire before detach, and failed setup or detach retains the root loan while admission stays closed. Destruction drains work and destroys handles before releasing borrowed storage if normal detach fails.

Twelve final GPU gates and six host ASAN/UBSAN suites pass with current source/object inputs. Normal and strict first-use/reused-handle tests match complete native output through managed/native transitions and destruction with queued work. Eight deterministic workspace failure cases cover before/after bind, release and destruction, including failure after two other streams have been configured or detached; handle-creation failure/retry also passes. The model-free failure suite has 175 assertions, and strict execution reports zero errors. All 24 inventoried matmul and four strided attention geometries retain exact output, including native return after arena destruction and four-stream capture/replay. Successful bound windows show zero native-pool escape and zero retained native-owned cuBLAS bytes; borrowed library addresses remain within the retained scratch grant. Four full ordinary encoder controls retain exact original native embeddings.

The initial library coverage red fails exactly once after valid native output. A local incremental-link script later omitted the MMQ helper by selecting reused objects by list position; the subsequent host link rejected the artifact before wide GPU qualification. The corrected build selects unique current objects by source identity and verifies relocations. Final evidence is `benchmarks/results/vision-operator-scratch-20261101/dv13-qualification/` and `dv13-audit.json`, with five zero-error strict gates. Native kernels and the shared CUDA context header remain unchanged. Opaque library/driver resources remain separately accounted, and complete MTMD phase admission/hookup is still DV14.

- Separate handle creation from the current unconditional owned-workspace allocation for managed execution. Preserve the existing configured workspace size and math configuration. Track borrowed versus owned pointers in initialization, rebind and destruction.
- Bind an aligned grant for each active handle/stream after stream selection. Ensure previously initialized native workspace/pool reservations are safely released or separately reported; merely adding an arena pointer while retaining the old allocation does not meet the memory objective.
- TDD: first-use and reused handles, native-to-managed and managed-to-native transitions, stream changes, multiple backend contexts, repeated capture/replay, workspace rebind failure and destructor cleanup. Assert no frees of borrowed addresses, no stale references after phase return, and no explicit out-of-arena workspace allocation on the managed path.
- Preserve exact outputs through all supported cuBLAS branches. Record opaque handle/library allocations separately; investigate unexpected allocation growth rather than declaring every library call covered.

Exit: both the GGML temporary pool path and the explicit cuBLAS workspace path identified in DV10 have qualified arena ownership and completion.

### DV14 - Integrate complete vision admission and phase recovery

Status: complete for common Qwen/Gemma CUDA vision phase integration, 2026-11-01. MTMD measures graph storage and backend-private/library requirements before admission, reserves disjoint operator banks within the existing compute loan, and retains the original root loan separately from nested scratch views. GGML receives a hard graph-placement limit while its complete root lease remains pinned. Zero-private-workspace graphs bind only the root, with no unused GPU bank. The complete original graph/source payload is revalidated after input copies. Existing serial target/vision ownership, protected recurrent/SWA storage and the physical parent remain authoritative.

The component quote reports graph, operator, library and alignment bytes separately. Worst-case root alignment is derived with the existing planner; the root may be 128-byte aligned while operator banks require 256-byte addresses. A GPU red caught the initial use of occupied bytes instead of the planner high-water bound: the correctly rejected fixture was missing 128 alignment bytes. The fix changed the quote, not the arena budget. Graph-only and operator-bank limits reject insufficient storage without native allocation around the quota.

Twenty-one accepted GPU/server gates and six host ASAN/UBSAN suites are audited in `benchmarks/results/vision-operator-scratch-20261101/dv14-audit.json`. Four complete encoder fixtures match original native F32 bytes with zero native-pool requests/growth and zero retained external explicit cuBLAS workspace; strict Qwen and Gemma controls report zero errors. The existing parent budgets remain unchanged, including the approved sanitizer-only Gemma profile. Protected Qwen+DFlash and Gemma targets retain exact KV/recurrent/SWA snapshots and native continuation outputs through cancellation, queued failure and retirement failure/retry. A full-phase red exposed stale embedding output after cleanup failure; an output transaction now publishes rows only after successful parent return. Failed retirement keeps the target closed and loans retained until explicit cleanup/resume, which the tests verify byte-for-byte.

Actual server validation covers Qwen target/MTP3/DFlash3 and Gemma target/MTP3, all seven common source aspect ratios, repeated/changed/multiple images, cache replay, socket cancellation, state erase/restore and process restart. All 109 paired completion scenarios match native token IDs and prompt/cache/draft/accept counts (3488 generated IDs across both variants); complete saved state files also have matching SHA-256 pairs. Processed-size cycles follow native preprocessing: Qwen includes 64/192/64 image tokens; Gemma includes 256/234/256 rather than assuming raw pixel size determines processed token order. Native references use their own preserved libraries.

Two harness failures are preserved: a zero-scratch fixture omitted the mandatory CPU fallback, and a 1 GiB host prompt cache evicted the native checkpoint before its restoration assertion. The fixture now supplies the CPU fallback and uses the established 2 GiB host cache; model weights, GPU arenas, exact oracles and API reporting remain unchanged. Seven earlier model gates are reused only because the corrected unused test binary was the sole changed runtime file; the explicit reuse mapping preserves original commands, profiles and hashes. Physical maximum-arena/resolution probing remains DV15; broader performance/platform checks remain DV16. Opaque driver/library resources remain outside the enumerated application-controlled workspace contract.

- Extend MTMD graph/phase requirements and existing suspended-loan admission to include all controlled operator and cuBLAS scratch. The parent must cover every simultaneously live graph/weight/scratch region and all protected persistent regions. Use the current planner, not a replacement manual fit formula.
- Revalidate startup ordering, native graph catalog/fusion choices, stream topology and graph revisions before the child captures addresses. Preserve Qwen's bounded-writer role, Gemma's shared-reader role, actual image provenance, SWA geometry and recurrent transfer fences.
- Add component diagnostics for graph bytes, operator scratch, cuBLAS workspace, padding, retained state and total required parent bytes. A changed minimum must be explained by actual measured requirements; do not enlarge fixtures or reduce workspace merely to pass.
- TDD: exact minimum/one-byte rejection before encoder enqueue; text -> vision -> text; small -> large -> small images; seven source ratios and portrait pairs; multiple images; changed images; capture/replay; cancel; queued conversion/GEMM failure; return failure; held grants; reconstruction; state restore and process restart. Verify source/KV/recurrent bytes and parent identity through recovery.
- Prove that all affected work and captures retire before temporary bytes return to the text KV pool. Failed cleanup must retain ownership and close admission until safe recovery.

Exit: both model families execute the complete controlled vision workspace inside one parent, with unchanged numerical/ownership contracts and no unexplained managed pool fallback.

### DV15 - Requalify automatic resolution probing and physical capacity

Status: complete, 2026-11-02. The versioned `vision-memory-v2` report and finder require graph, operator, explicit library and padding bytes to sum to the actual compute grant. Old graph-only reports, missing/bool/negative/inconsistent components and mismatched executed bytes reject. Opaque runtime allocations are explicitly excluded. The shared-MTP quota diagnostic is retained as a lower bound even when a separate-workspace fallback reaches the vision guard.

The eight capacity comparisons use one RTX 5070 Ti, native context capacity 262144, MTP3, Q8_0/Q4_0, b/ub256, UVM off, 2048-token bounded probes and 256 outputs. Qwen uses IQ4_XS with the matching F16 projector; Gemma uses Q3_K_S, its matching external Q8 assistant and local Q8 projector. Baseline/current controls use their own immutable libraries and unchanged weights. Fixed 2048 MiB parents validate all seven ratios at the respective 4096/256 image-budget ceilings on both implementations. The original fixed 1024 MiB Qwen attempt correctly rejects its 1197 MiB shared-MTP startup minimum; it is preserved rather than counted as a numerical pass.

| Profile | Baseline maximum MiB | Current maximum MiB | Current rejected neighbor MiB | Baseline common image budget | Current common image budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen embedded MTP3 | 2280 | 2314 | 2315 | 47 | 2186 |
| Gemma shared-KV MTP3 | 2410 | 2462 | 2463 | 143 | 256 |

The current Qwen individual ratio maxima reach budget 4096; its common setting is smaller because a combined larger sequence reaches an opaque target-decoder `cudaFuncSetAttribute` OOM after six successful, returned encoder phases. Those failures remain preserved. Gemma's current 256 ceiling is imposed by the tested microbatch. These are measured processed-image fits for seven source ratios, not visual-quality scores, arbitrary-image guarantees or native long-context equivalence. Fixed-parent and independently maximum-fit comparisons remain separate.

A publication-fidelity red proved that the old final text check used the bootstrap image setting while the generated command used the larger common setting. Final validation now runs after selecting that common setting and requires the exact published command. Each current maximum completes 261873 text tokens and 256 outputs, followed by seven image requests with 256 outputs each in the same process. The Qwen full text also reproduces every earlier text output ID and prompt/cache/draft/accept count. Default full search and opt-in selected validation retain the same publication check; failed final checks keep their receipts and publish no command.

The optional text resource probe has its own exact-launch test. Its source-accounting red exposed seven completed image requests omitted from expected draft totals; the fix includes their existing timings and still rejects generated or accepted count drift. The live fixed 2048 MiB control completes 8169 text tokens, eight outputs, the 32-output constrained text proof and all seven images. Host checks pass 44 vision and 182 benchmark tests, with one expected platform skip; type checking has the same five diagnostics as the unchanged baseline. No native arithmetic, GPU allocation policy or model encoding changes were made in this stage.

The audit checks 1447 artifact references, current native/model hashes, all ratio/common receipts and real +1 MiB rejections. Six accepted capacity gates are reused only with archived source, identical runtime/commands and a specialized AST proof that changed final-validation and opt-in accounting branches do not alter those probes; both published full-profile checks and the optional branch have fresh execution. Original services are restored with unchanged IDs/configurations and healthy port 1234. Local evidence: `benchmarks/results/vision-operator-scratch-20261101/` (`dv15-audit.json`, `dv15-v2-reuse-qualified.json`, `dv15-v3-reuse-qualified.json`, finder and copy qualification directories). Raw results remain local.


- Extend the existing server planning report and finder to use the complete phase quote, with versioned/validated component diagnostics. Configure the projector and generate native fixtures automatically; do not require a user-provided image or introduce a manual-image recommendation in README.
- Retain all seven ratios, native alignment, fixed decoder/batch configuration, actual processed dimensions, independent row settings and the common setting. Validate the common setting with all seven requests serially in one process after text; no command is published for incomplete coverage.
- TDD: the former graph-only quote must fail the new complete-admission assertion. Cover old/missing report fields, wrong component totals, a dry-admitted shape that cannot execute, physical OOM versus quota versus device faults, no common fit, partial output and artifact/hash preservation.
- Compare old and new at the same fixed parent and separately with independently detected maximum arenas on the same GPU. Moving previously external buffers into a fixed parent can raise its minimum; an improvement in total physical capacity must not be inferred from one budget column alone.
- Refit the exact published Qwen/Gemma profiles using bounded runtime probes, actual +1 MiB rejected neighbors and immutable receipts. Run the selected full text workload once per published profile when claiming full-history validation. Keep bounded probes and full-history passes distinct.
- Preserve native image ceilings and Gemma's whole-image microbatch/SWA constraints. A higher reported resolution must execute at unchanged model/KV settings; no silent quantization, offload, budget or assertion change.

Exit: the finder publishes measured resolutions using complete controlled-workspace admission, with actual execution, auditable upper boundaries and scoped remaining opaque costs.

### DV16 - Complete numerical, safety and performance qualification

Status: complete, 2026-11-02. The current-source audit covers 128 runtime gates, six focused ownership/replay gates and sixteen host ASAN/UBSAN suites. The 259 paired API scenarios comprise 196 full native-server pairs and 63 explicitly scoped component-reference pairs. Full native DFlash5-7 and Gemma MTP1-3 at image budget 1120 reach recorded physical capacity limits; their unchanged native encoder/decoder vector controls and end-to-end streamed references are kept separate from full native-server equivalence. All thirteen strict GPU gates report zero errors with default PDL, API-all reporting, error exit 99 and no filters. Capture-required owner/MTP2/concurrent-stream gates retain actual CUDA graphs; eager execution is checker-only elsewhere.

Ordinary Qwen MTP2 exposed a cached-writer destination lifetime bug during a small/large/small/changed/two-image sequence. A valid CPU red first completed a native-matching write, then proved the caller could free destination metadata while the writer still used it. The fix retains that buffer with the existing owning pointer until cached executors and graph contexts retire. It adds no GPU allocation, logical phase lease, kernel, arithmetic or synchronization policy. CPU/host-ASAN writer controls pass 4306 assertions; three fresh ordinary original-sequence replays and an actual graph-enabled strict replay match all seven native token/work-count responses. The earlier fixture and graph-environment mistakes remain preserved as failures, not positive qualification.

Eighteen alternating ordinary text process pairs use natural Open-SWE prefixes and 256 outputs. Same-mode tokens, prompt/cache/draft/accept counts, source statistics and physical decode layouts match. Across six modes, median paired prefill changes range from -0.073% to +0.118%, and decode from -0.615% to +0.215%. Five image profiles each compare three warmed uncached samples at three source sizes; Gemma native preprocessing produces two distinct processed geometries. Image total-latency changes range from -0.292% to +0.594%, and decode from -0.795% to +0.371%. Small encoder phases have measurable validation/ownership overhead; these checks establish no material end-to-end regression on the fixtures, not zero cost or a new throughput optimization.

Four image comparisons retain 2048 MiB parents. The archived DFlash baseline at that parent runs out of memory preparing a decoder CUDA kernel after the medium encoder phase returns; the candidate had not run. The fitting DFlash comparison uses the same derived 1605 MiB parent for both implementations, with the complete 4096-token square quote of 1682285312 bytes and an actual 1604 MiB rejection. All images, weights and numerical checks remain unchanged. Sampled 50 ms device peaks fall by about 202-204 MiB for Qwen and 62 MiB for Gemma; samples may miss shorter transients. The Gemma two-compatible-image checker profile correctly rejects its older 1536 MiB parent: complete requirements are 1633528448 bytes, so the strict replay uses the smallest whole-MiB fit, 1558 MiB. Ordinary 2048 MiB profiles are unchanged.

The long F7 48000-token image/cache/state/cancel/restart profile passes again, with real ring traffic and complete saved state; this is long lifecycle self-consistency. All-64 D512 spans/progressive readers, original Q4 DFlash and K4V text, Qwen MTP and both model families' protected state and recovery controls pass. The three affected CUDA adapter translation units compile into twelve actual SM61/70/75/86 code images using the existing offline CUDA 12.9 SDK, serially with native optimization flags. This is functional compilation, not older physical-GPU runtime or a full legacy-server link. The owned SDK container is removed and original services are healthy. Runtime, performance, compiled-object and source audits are retained locally under `benchmarks/results/vision-operator-scratch-20261101/`; raw benchmark artifacts are not part of the change.

- Fresh native/component image controls cover Qwen target-only/MTP1-3/pure DFlash1-7 and Gemma target-only/shared-KV MTP1-3. Retain Q8_0/Q4_0, Q8_0/Q8_0 and Q5_1/Q5_1 end-to-end controls. Use fitting full-native references first; the previously approved split-reference policy remains limited to genuine capacity failures.
- Compare full encoder embeddings and consumed target/draft feature, logit/hidden, KV and recurrent vectors byte-for-byte. Preserve native preprocessing, M-RoPE positions, sampling and per-scenario prompt/cache/draft/accept counts. Include higher-resolution and repeated-vision cases actually enabled by this extension.
- Run affected CPU ASAN/UBSAN suites, graph-enabled ownership checks, and strict encoder/public/decoder memcheck with default PDL, API-all reporting, error exit 99 and no filters. Eager execution is checker-only. Tool OOM/timeouts are findings, not passes; any material profile change needs an explicit scope decision.
- Recheck common allocator/executor, existing all-64 reader guards, Qwen MTP/DFlash/K4V text, Gemma text/MTP, phase recovery and ordinary unbound CUDA paths as appropriate to the actual dependency changes. Preserve K4V vision rejection and deferred K15 boundaries.
- Run repeated matched ordinary image requests for both families and several fitting image sizes, with exact same-mode outputs/work counts. Report complete request latency, encoder phase, prefill/decode throughput, peak/retained VRAM and allocation counts separately. Compare both equal-parent profiles and independently maximum-fit capacity; do not mix their interpretations.
- Investigate material existing-mode regressions before release. This is ownership/capacity work: no raw throughput gain is assumed, and profiler timings do not prove one. Audit source, model, toolchain and binary provenance; reuse earlier evidence only with exact relevant identity and unchanged commands/profiles.

Exit: the complete coverage claim is backed by current numerical, allocation, lifecycle and safety evidence, plus honest capacity/performance results and healthy restored local services.

### DV17 - Publish the controlled-workspace contract and close the extension

Status: complete, 2026-11-02. Fresh post-writer-fix finder examples retain the qualified compiled runtime and independently measure Qwen MTP3 at 2314 MiB (2315 rejected), common image budget 2186, and Gemma MTP3 at 2462 MiB (2463 rejected), common budget 256. Both use native context capacity 262144, Q8_0/Q4_0, b/ub256, UVM off, default 2240 MiB initial guess and automatic seven-ratio probing. Each final exact published command completes 261873 text tokens and 256 outputs followed by seven 256-output image requests. These are separate image prompts after full text, not long native equivalence or a global OOM/quality guarantee.

A subsequent actual finder CLI sweep verifies all fourteen printed ratio rows against prior native geometry and completed encoder execution. Qwen b/ub256 reproduces 2314 MiB and common budget 2186, with independent budget 4096 at every ratio. Gemma b/ub1280 refits to 2444 MiB (2445 rejected) and reaches common budget 1120 for every ratio. Its 2560-token bounded prefill and 256-output requests are explicitly separate from the full-history examples above. The follow-up audit checks 244 artifact references; no runtime source, binary, weight or numerical checks change. Focused reports include the larger-image table and command without relabelling it as full-context validation.

README remains a brief shared vision concept/support/launch guide, with real selected result fields and links to focused reports. The reports distinguish graph/operator/library/padding bytes, controlled versus opaque memory, independent versus common settings, fixed versus maximum parents, historical source cohorts and native versus component references. The old 48/143 examples are historical external-scratch limits. The high 1120 Gemma profile's new 2284 MiB seven-ratio minimum and the corrected 1558 MiB two-image checker minimum retain their exact rejected boundaries and scope.

The private memory guide documents preparation/query/bind/release, the retained root and graph-placement prefix, disjoint stream banks, native-size explicit cuBLAS workspace, captured-work retirement, failed-cleanup admission and buffer metadata versus logical leases. Raw local benchmarks, server-AB helpers and data are not added. The upstream README body is preserved byte-for-byte (7610 bytes, SHA256 `8e43128b52e5e99fcfd7b65b4d182681566f98f72e7e8f812f71847ef8ec1dbf`). Local paths/heading anchors, fences, launch-option spellings and the unchanged upstream body are checked; a Mermaid renderer is not installed, so unchanged diagrams are not claimed as newly renderer-qualified. Host checks complete 44 vision tests, 163 tracked benchmark tests and 19 retained local harness tests, with one expected platform skip. Original local services are restored healthy. K4V vision, other physical-GPU vision runtime and complete all-process VRAM control remain outside this extension.

- Update the focused vision/capacity reports and memory API documentation with exact controlled allocation classes, remaining driver/library costs, actual resolution results, changed minima and phase lifetime rules. Keep README concise and focused on automatic probing, supported configurations and launch commands.
- Replace the historical 48/143 examples only with new executed measurements; retain original failures/results with their source identity as historical evidence. State whether published values are fixed-parent, maximum-fit, bounded-preflight or full-history results. Do not turn resolution capacity into a visual-quality score.
- Validate Markdown links/commands and preserve the upstream README section byte-for-byte. Keep raw benchmarks and server-AB changes local, as required by the existing publication boundary.
- Close each stage only after its full acceptance criteria pass. Follow the user's one-sentence descriptive commit policy, with no stage-code-only titles, attribution trailers or pushes. Preserve unrelated working-tree changes, including the current README follow-up.

Exit: DV10-DV17 are complete for the declared controlled CUDA vision workspace. Model weights and identified opaque runtime costs remain separately accounted; no global all-process OOM guarantee or K4V-vision qualification is implied.

## K4V vision extension: DV18-DV26

Status: complete, 2026-11-03. DV18-DV26 qualify optional K4V on the existing Qwen DFlash2 image path; DV0-DV17 retain their separate foundation scopes. The initial implementation inspection used `af2b9cb8f`; current admission, source accounting and limits are described in the [focused report](docs/multimodal/dflash2-vision.md#optional-k4v-image-conditioned-text). Earlier closed-guard and pending-next-stage statements below describe their historical stage, not the current release.

### Scope and release policy

- Start with the already qualified Qwen3.8-27B IQ4_XS target, matching F16 projector and matching Q2_K_S-MIX DFlash2 sidecar. Keep the original Q4 sidecar as a separate text regression control. Pin the exact artifacts again; a different sidecar encoding can change acceptance and memory fit.
- One serial request on one CUDA GPU, Flash Attention, GPU KV offload, UVM off, an explicit shared arena, fit off and context shifting disabled. Preserve the three full-model KV reference pairs: Q8_0/Q4_0, Q8_0/Q8_0 and Q5_1/Q5_1. Draft formats remain inherited; even identical explicit overrides remain rejected.
- DFlash maximum F and K4V copied maximum K remain independently configured in 1-7. N remains the lookup-key length, not a table size or verification width. Keep existing positive-hit and N bounds. K15/TG16, K4V-only vision, MTP+K4V vision, Gemma+DFlash, other model pairs, audio/video, LoRA, parallel slots and multi-GPU are outside this extension.
- Support real still images, multiple images and interleaved text/image requests within measured image/context limits. K4V must actually supply valid post-image drafts in positive controls. Merely loading mmproj, accepting a flag or falling back to pure DFlash for every image request does not meet acceptance.
- Correct support with reported performance is the release objective; a speedup over pure DFlash or MTP3 is not presumed. Investigate material regressions in existing text/vision modes. Do not retune N/K/hit thresholds, reduce images or change model/KV precision to hide a failure.
- Reuse the completed bounded writer, source provenance, serial owner, rollback, prefetch/restore scheduling and controlled vision workspace. K4V stays in RAM. No new GPU map, attention kernel, allocator, second physical parent or speculative framework is expected.

### Current mechanisms and concrete gaps

| Source anchor | Current behavior | Extension requirement |
| --- | --- | --- |
| `common/common.cpp`: `common_streamed_dflash_config_error` | Explicitly rejects mmproj plus K4V | Open only the qualified mixed set after internal proof; keep pairing, width, format and owner guards |
| `common/common.cpp`: `common_streamed_dflash_verification_error` | Already derives target capacity from both F and K | Reuse it; do not size target verification from F alone |
| `tools/server/server-common.cpp`: vision configuration/request guards | The pure-DFlash predicate allows only NONE/DFlash types | Admit the exact configured NONE/DFlash/K4V set and validate its parameters; removing the common guard alone is insufficient |
| `tools/server/server-common.h/.cpp`: `server_tokens`, `get_text_tokens`, `find_chunk`, `pos_next` | Stores media extents/positions but filters NULL image rows from speculative text history | Derive a safe text suffix and preserve the existing full DFlash text snapshot and true decoder position |
| `tools/server/server-context.cpp`: draft setup and `common_speculative_begin` | Both currently receive filtered text without a media boundary | Supply a source-specific K4V lookup range at both generation begin and drafting; keep ownership through queue execution |
| `common/speculative.cpp`: `common_speculative_impl_ngram_map_k` | K4V begin/draft use the entire shared prompt; process is a no-op | Bind only K4V to its safe range; do not change DFlash's history or skip its process callback |
| `common/ngram-map.cpp/.h` | Map keys, cursors and values refer to positions in one contiguous token history | Preserve current selection/statistics semantics with a checked range view; invalidate indices when that range's identity changes |
| K4V zero-index sentinels (`key_map`, `value_idx`, `match_pos`) | Zero denotes unused/not found; current searches skip position zero | Define checked range/index encoding so the first real suffix occurrence is not accidentally lost or treated as a sentinel |
| `common/speculative.cpp`: DFlash process/draft | Validates completed features, skips multirow pinned image groups, injects text features; prediction uses actual `dp.pos0` | Preserve this policy even when K4V supplies proposals; raw image embeddings are not DFlash features |
| `common_speculative_get_state` / `set_state` | Existing interface handles one saved implementation state; K4V maps rebuild at generation boundaries | Reconstruct lookup metadata/maps from restored authoritative prompt; do not add a combined state format just to save disposable maps |
| `benchmarks/find-max-arena.py` | Rejects mixed mmproj; its mixed proof constructs an integer-token text prompt and runs before image bootstrap | Add an image-conditioned proof and complete request accounting, not just a parser exception |
| `tools/server/tests/test_adaptive_vision.py` | Launches pure DFlash/MTP image controls and expects mixed rejection | Extend the existing harness with explicit mixed settings, actual supplying-source observations and negative controls |

The existing text K4V fixes are already implemented: selected continuation emission, one-time statistics accounting, ambiguity/minimum-hit behavior and generation rebuilds have CPU tests. This work must not reintroduce the old bugs or replace the qualified scoring policy with a new heuristic.

### Mixed-mode contract

#### Source-specific text history

The first release uses the contiguous text suffix after the latest complete image chunk. Text before that image, including earlier otherwise-valid text segments, is intentionally not searched by K4V. This conservative scope prevents both lookup keys and copied values from crossing an image. Searching all older bounded segments is a later optional optimization with separate tests; it is not required to open the first mixed image release.

Prefer a checked non-owning range over the existing slot-owned `spec_prompt` text snapshot. Compute the suffix start while walking validated `server_tokens` media extents; do not guess it from literal marker strings or from the number of NULL rows alone. The default `get_text_tokens()` result and DFlash prompt remain unchanged. Reuse an existing suitable range utility, or add a minimal private contiguous token view with existing vector interfaces preserved as wrappers. Do not require C++20 `std::span`, make a second persistent conversation history, or copy full image-placeholder runs into a new map history.

The existing map uses zero-valued index sentinels. DV18-DV19 must choose and test a consistent range-aware index representation, including a real first suffix occurrence at offset zero. Preserve existing whole-prompt wrappers, use checked encoding or bounds where needed, and never solve an internal sentinel collision by adding a token to the model prompt.

The view describes only committed text, excluding the sampled anchor and any provisional draft. K4V combines its suffix with `id_last` using the existing lookup rule. The anchor is supplied once. Empty or too-short suffixes are legitimate lookup misses and fall through to DFlash; malformed media extents, stale generations or invalid tokens are errors/reset conditions, not ordinary misses. No artificial barrier token ID enters the vocabulary, target prompt or saved state. Native textual wrappers remain at their actual side of the media chunk; their treatment must be characterized by fixtures.

| Quantity | Meaning | Must not be substituted for |
| --- | --- | --- |
| Full server token index | Text plus all media storage rows | Model position under M-RoPE |
| Filtered DFlash prompt index | Existing text snapshot with image rows omitted | A valid K4V cross-image history |
| K4V lookup index | Checked position in the eligible suffix/view | Target/draft KV position |
| `server_tokens::pos_next()` / `dp.pos0` | Actual next model position | Text count, suffix length or image row count |

For example, two text tokens, a three-row image occupying two model positions, and three suffix tokens have eight storage rows, five filtered text tokens, a three-token K4V suffix and next model position seven. Use actual chunk metadata for this calculation; real images need not have that illustrative row/position relationship.

A pointer address, unchanged suffix length or identical text is not a sufficient generation identity. Bind the view to the existing request/prompt generation, media boundary and committed-prefix identity. Rebuild map keys/value statistics and retire pending supplier feedback on generation replacement, changed images, rewind, restore or changed suffix origin. Normal committed append within the same suffix must keep incremental statistics; do not clear or rebuild the map on every draft or merely because a snapshot's storage address changes. Keep lookup metadata per sequence even though the admitted server is serial.

#### Drafting, verification and features

```text
full committed prompt + real media metadata
    -> existing full DFlash text snapshot + true model frontier
    -> K4V-only view of text after the latest image
    -> K4V proposal, or DFlash prediction on a legitimate miss
    -> target verification at the original model frontier
    -> DFlash feature processing, regardless of proposal supplier
    -> native acceptance/correction and target/draft trimming
    -> next lookup from authoritative committed prompt
```

Use D = max(F,K): target verification/output capacity is D+1 rows and rollback depth is D; neural prediction remains F+1 rows. Image-input microbatch size does not become proposal width or full-vocabulary output count. Preserve all measured intermediate target widths TG1-TG8 and actual feature-injection capacity when K exceeds F. Short-context clamping may reduce actual proposals, never increase an admitted width or allocate around the quota.

Keep the existing cascade and supplier record. A copied draft skips only DFlash prediction, not target verification or required image-conditioned feature injection. Ambiguity/empty lookup falls through once; rejection does not retry DFlash against the old frontier in the same round. The target correction/bonus token is the next unprocessed anchor. Rebuild the next lookup from the server's final trimmed prompt rather than maintaining an independent manual copy of accepted history. A checkpoint replay must not consume acceptance feedback twice or reset a still-valid pending supplier midway through replay.

#### Recovery, identity and one-parent memory

| Event | Required lookup behavior | Target/draft requirement |
| --- | --- | --- |
| New generation, even identical/equal-length input | Reset using the current safe range, as text K4V already resets | Preserve existing common begin and sampler semantics |
| Successful verification/append | Refresh the borrowed view; preserve incremental prefix indices | Inject features, attribute acceptance and trim both contexts normally |
| Cached image/prompt or checkpoint restore | Derive boundary from restored whole-chunk metadata; discard stale map/feedback | Respect actual available history and the existing checkpoint selection |
| Changed, adjacent or later images | Move the suffix origin after the final complete image and reset lookup state | Preserve all image inputs/positions and completed feature provenance |
| Cancellation, erase, failed restore or source replacement | Drop pending candidates/view state before reuse; rebuild only from authoritative recovery | Drain or close the owner, with no stale output or speculative continuation |
| Vision loan/retirement failure | The host map cannot authorize target execution | Retain unresolved leases and keep the target closed until cleanup succeeds |
| Restart / serialized media placeholders | Rebuild from validated restored chunk extents, including placeholders with no pixels | Restore existing target/draft/media state; do not invent missing history |
| End of context / EOS / both sources empty | Honor actual remaining rows and clear stale supplier state | Keep native target-only fallback and end/cancel behavior |

K4V index, keys, values and view metadata stay in RAM. Measure capacity/growth and avoid a new full-prefix copy per lookup where the existing owned snapshot can be reused. GPU weights remain outside the parent. Target state, bounded DFlash KV, rollback staging and current vision weight/graph/operator/library/padding loans keep their existing owner and fences. Adding host lookup does not create another GPU consumer. F3/K7 can require more target rollback/output/workspace than pure F3, so remeasure rather than assuming mixed image admission is VRAM-neutral. Preserve the flexible prefetch and recurrent-restore overlap fixes and restore their prior hints at teardown.

### Staged implementation and acceptance

Each stage starts with focused red evidence, then a minimal fix and the affected regressions. Keep the public mixed vision barrier until internal policy, lifecycle and exact-model controls pass. A compile red proves a missing seam; geometry/OOM/fixture failures are findings, not the expected behavioral red. One descriptive one-sentence commit closes a whole qualified stage, with no attribution trailers or pushes. Preserve DV0-DV17 and unrelated work; implementation begins on a new feature branch from the qualified base when requested.

#### DV18 - Pin the mixed image contract and reproduce unsafe history joins

Status: complete as a controlled-red stage. The new feature branch preserves the qualified runtime/library hashes and original native image inputs. Three live GPU controls pass: seven pure DFlash native/streamed API scenarios, the real image-conditioned vector oracle (1580 assertions), and mixed-text source switching/all verification widths (4419 assertions). Current CPU speculative/server controls pass 493/386 assertions, and host-ASAN/UBSAN preserves the expected readiness red without memory errors. Original services are restored healthy.

The opt-in `test-server-vision --k4v-vision-readiness` reports exactly seven failures after valid controls: first post-image occurrence at index zero, a cross-image lookup key, a cross-image copied continuation, a pre-image value excluded by the initial suffix policy, and common/server/request mixed-image admission. The separate finder CPU red reports exactly one missing-admission failure. The ordinary released tests remain green and mixed vision remains closed. The first readiness attempt used an incorrect row/position assumption about the cheap CPU fixture; that fixture-only failure is preserved and is not counted as the behavioral red. The real native Qwen fixture has 64 image rows but eight model positions, with all four M-RoPE coordinates retained. Local evidence is under `benchmarks/results/k4v-vision-20261102/`; source/binary/model identities and original failing artifacts are preserved.

- Pin current source/binaries, target/projector/smaller-draft hashes, settings and existing pure-image/text-mixed controls. Confirm the native-compatible Q2 metadata again; use it on both sides of each new comparison.
- Characterize actual text wrappers, media extents, cached placeholders and all four M-RoPE coordinates. Record the precise safe-suffix extraction and generation lifetime before choosing the minimum private view seam.
- CPU reds must demonstrate the current filtered-history join: a key split by an image, a continuation split by an image, and a whole pre-image continuation that the initial suffix-only policy must exclude. A genuine repeated post-image suffix is the positive control.
- Add controlled common/server/finder reds for missing mixed-image support, after valid pure native/image controls. Keep current rejection as the released behavior; no guard-only feature opening.

Exit: source-anchored failure cases distinguish the new lookup policy from pairing, allocation or image-fixture failures; design and baseline artifacts are reviewable.

#### DV19 - Add a checked K4V-only suffix view

Status: complete. The bounded map overload borrows a suffix of the existing full text vector with checked absolute indices; it does not allocate a second history. Legacy begin/draft wrappers retain their previous behavior. A separate found flag and explicit first-eligible check handle real index zero without inventing a token. Search/statistics cursors exclude the prefix, bounds are checked before narrowing/access, stale shrink rejects, and ordinary append preserves incremental statistics. The new server snapshot walks validated whole media chunks and returns the same full filtered text plus its last-media offset; the original getter remains unchanged.

The missing getter/bounded-begin API compile red is preserved. Native and host ASAN/UBSAN pass 548 speculative and 430 server assertions, covering first/last/adjacent/multiple images, cached placeholders/serialization, short/empty/invalid ranges, N1/N1024, K1/K7, optional hashes, zero acceptance, prefix exclusion and growth. The readiness probe now has only the three expected admission reds; every history case passes. Three fresh current-library GPU controls pass again: pure native/streamed images, complete native image vectors and mixed-text switching/all widths. CUDA remains byte-identical. Generation/server dispatch integration is DV20; public mixed vision remains closed.

- Extend the existing server-token/speculative/ngram helpers with the smallest range seam. Borrow the existing owned text snapshot where possible; leave the ordinary shared prompt and position APIs unchanged. With no images and no boundary context, preserve existing text-only native and streamed K4V behavior.
- Apply the selected range consistently to map begin, key search, occurrence counting, value selection and acceptance indices. Keep indices/cursors in one declared range-aware space, use checked bounds before pointer arithmetic, and handle an empty view without null-pointer arithmetic. Prove first-suffix-token lookup and zero-sentinel handling; do not silently drop that occurrence.
- CPU TDD covers image-first/last/only, multiple/adjacent images, N=1 and N=1024, K=1 and K=7, suffix shorter than N, exactly sufficient history, collisions, ambiguity, selected values, saturation and zero acceptance. Include genuine copied proposals after images; globally disabling K4V is not a fix.
- Verify wrapper text behavior and malformed/truncated media-map rejection. Existing text map tests, other n-gram algorithms and CPU ASAN/UBSAN remain green. No map reset on each ordinary append.

Exit: unsafe joins and values are impossible within the declared suffix policy; exact text-only behavior and map accounting are preserved without a second persistent history.

#### DV20 - Wire generation, prompt-cache and checkpoint lifetime

Status: complete. The common dispatcher passes a checked per-sequence suffix/generation descriptor only to K4V; DFlash keeps the existing full text snapshot and true model frontier. Fresh rounds refresh the slot-owned snapshot; actual generation begin resets disposable lookup and supplier state. Replay keeps its pending source. Invalid ranges reject before begin mutation, and sequence resets are isolated. Native and ASAN/UBSAN/leak checks pass 563 speculative and 430 server assertions. The three public admission barriers remain closed.

The prior native server reports exactly one excluded-prefix-copy failure after valid real-image controls. Current native fresh/cached/changed-image requests preserve every forced output ID, reject pre-image copying and observe post-image copies plus genuine neural forwards. The initial four-output fixture ended before a second lookup; its failure is preserved separately. The corrected seven-output fixture changes no runtime or numerical check. Current pure-image native pairs, full image-vector import and mixed-text TG1-TG8 regressions also pass: five GPU gates including the expected red. Original services are restored healthy. Source, build and model hashes, logs and the qualified patch are retained locally in `benchmarks/results/k4v-vision-20261102/dv20-*`.

- Supply the K4V view at both generation begin and fresh-draft setup. Keep the snapshot alive and immutable through `queue_tasks.yield_to_queue`; DFlash receives its current full text prompt and actual `pos_next()`.
- Audit prompt clear/load, context checkpoint get/set, sequence removal, partial acceptance/replay, slot-file restore and restart. Reconstruct disposable K4V state from the authoritative prompt instead of serializing indices or extending the one-implementation state container unnecessarily.
- Reds cover same text/length with changed image, changed suffix origin, restored placeholders, rewind across an image, rejection replay, stale pending acceptance and cancellation followed by reuse. Validate that illegal mid-chunk rewinds and unavailable old history remain rejected.
- Prove no tentative token enters the next lookup and no supplier feedback survives a generation replacement. Reset only the affected sequence; do not call global begin on every round or disrupt another implementation's state.

Exit: source identity and acceptance lifetime survive every supported recovery path, or fail closed before output; no stale map/view/source pointer remains usable.

#### DV21 - Qualify actual image-conditioned source switching and rollback

Status: complete; public mixed vision remains closed until DV22. The existing image-conditioned framed oracle now accepts an independent K4V limit and bounds lookup to its post-image text. F3/K7, F7/K3 and F7/K7 pass complete native-decoder/streamed comparisons for copied/neural/copied switching, ambiguity, every accepted prefix for both sources, limited proposals and TG1-TG8. Each verification still processes target features into the DFlash cache. Full logits, hidden/layer vectors, selector values, visible image-aware KV, recurrent state, source counters and true frontiers match exactly. A one-row tail from the real encoder fixture injects features, while the original pinned multirow image policy remains checked. The streamed contexts retain the same physical parent across vision loans and source changes.

The real frozen encoder input and Q2 sidecar provide a component-level native decoder reference; they are not described as a complete native server with the projector loaded. The smaller-sidecar full-native F7 projector-capacity failure remains in the DV0 ledger. The initial context-end fixture confused image positions with occupied KV rows; its failure is retained separately. The corrected test fills the actual physical row limit, rejects an oversized append and resumes with the final fitting row without changing the context or arena. Four current GPU gates pass, including the unchanged pure-image vector oracle. Native framed controls are preserved in SHA-checked lossless archives, and every production library remains byte-identical to DV20. Source/build/model hashes and restored healthy services are audited locally in `benchmarks/results/k4v-vision-20261102/dv21-*`.

- Extend existing `test-speculative.cpp`, `kv-stream-dflash-test.h` and `test-kv-stream-context.cpp` controls; combine the real exported image fixture with the existing mixed-source program rather than adding a replacement inference driver.
- Force actual K4V -> DFlash -> K4V, misses/ambiguity, both-empty and every accepted prefix 0..L, including F3/K7, F7/K3 and F7/K7. Exercise actual TG1-TG8 and context/window boundaries with real post-image copied candidates and neural prediction forwards.
- Compare complete target features, logits/hidden rows, image inputs/positions, target/draft KV, recurrent state, correction frontier and source counters. Include one-row image injection and multirow pinned-image skipping. K4V-supplied verification must still invoke the existing DFlash feature processing.
- Use native attention/model arithmetic with the same new host boundary policy as the reference. Do not bless the old native filtered-history bug as an oracle. If the smaller-sidecar full native server cannot fit, retain its failure and use the approved encoder/native-decoder component reference with exact images, positions and source/decision transcripts. Keep independent CPU policy controls and end-to-end streamed tests.

Exit: mixed vision works internally with exact native/component values, correct supplied-source accounting and all rollback decisions, without a silent pure-DFlash/target-only substitute.

#### DV22 - Open only qualified startup and request admission

Status: complete. Common startup now admits the already validated DFlash2/K4V image pairing; the server request predicate accepts only its configured source set, matching draft model, inherited formats and bounded independent widths. K4V N/K/hit settings stay fixed per server, and request changes to either mixed source are rejected. Existing unsupported-mode/format/parallel/LoRA/media guards remain. CPU reds reproduced 12 missing-admission failures in both CLI orders and 12 partial-source-policy failures; the checked implementation passes 569 speculative and 526 server assertions in native and ASAN/UBSAN/leak runs, plus the readiness and argument controls. Old mixed-image rejection tests now distinguish qualified K7 from unsupported K8 instead of retaining the obsolete blanket barrier.

Seven current GPU gates pass: F3/K3, F3/K7, F7/K3 and F7/K7 public image route/resource cases, the pure-image regression, mixed-text TG1-TG8 exact switching, and exact one-byte-short joint quota/teardown. Fresh/cached/changed/later images and a missing text suffix produce the expected complete forced outputs and actual copied/neural counters; all controlled encoder scratch stays in the parent and vision loans return cleanly. F3 full-native pairs match outputs and counts; F7 full-native smaller-Q2 projector OOM is preserved, with DV21 native-decoder component values kept as the numerical reference. The first mixed-text retry ran out of temporary disk while writing its large oracle. Only test temporary stdio backing moved to RAM after read/write/seek/flush/EOF contract red/green checks; device execution, arena and exact comparisons did not change. Accepted artifacts and hashes remain intact and production is restored healthy. Evidence is in `benchmarks/results/k4v-vision-20261102/dv22-*`.

- Resolve F/K/D, outputs, rollback, image-input feature shapes and all width/workload quotes before child capture. Reuse `common_streamed_dflash_verification_error`, the serial bounded writer and existing image-aware construction; no private workspace fallback or Gemma read-role substitution.
- Update both common and server configuration/request predicates. Preserve model/projector matching, immutable sidecar, inherited cache formats, F/K bounds, effective microbatch limits and all unsupported-mode guards. Do not mistake a mixed configured set for the old pure-DFlash predicate.
- Initially freeze N/K/hit policy at server configuration unless per-request changes have explicit bounded/reset qualification. Requests cannot exceed admitted widths, replace models or enable an unconfigured source. Empty valid suffixes may fall through to DFlash; invalid provenance cannot.
- CPU/argument/server reds cover valid mixed images, both CLI source orders, F3/K7 and F7/K3, and negatives for K=0/8/48, short microbatch, wrong model/format overrides, K4V-only, Gemma, MTP mixing, parallel/audio/video/LoRA and model overrides. Invalid configurations reject before target mutation or inference.
- Execute exact startup/joint quota boundaries, one-byte/one-MiB rejection as appropriate, phase return and child teardown, with physical parent/persistent target/draft bytes unchanged across vision loans.

Exit: matching mixed image requests reach real source switching; all unrelated configurations retain their previous admission and no late growth occurs.

#### DV23 - Extend image-capacity fitting and real mixed route proofs

Status: complete. The finder now uses separate image-conditioned copied-suffix and genuine-miss resource requests instead of slicing image prompts into the old integer-token proof. Both retain the actual supplied/generated images; repeated-fragment tokenization and forced output IDs are checked. N/K/hit bounds determine proof history/output requirements, and insufficient history is a configuration error rather than VRAM OOM. Per-request source deltas use real generation-begin counters, so an earlier text prelude cannot stand in for image copying or neural prediction. Completed text preludes, main requests, seven-ratio bootstraps and extra image proofs are counted once; planned encoder phases include the two proof requests. Pure text and optional non-mixed checks retain their previous paths.

Four controlled host reds now pass; the full vision harness has 49 tests, and all 182 benchmark host tests pass with one expected skip. The previously authorized one-line finder-test exception now rejects K8 instead of the newly supported image pairing; no new server-AB harness or raw data is committed. Current live Qwen IQ4_XS/Q2/F16-projector F3/K7 and F7/K3 automatic fits each select 2110 MiB and reject 2111 MiB, at context 4096, b/ub256, Q8/Q4, UVM off, bounded 2048-token prefill and eight ordinary outputs. Both execute image copy/miss proofs and all seven independent/common ratio settings under the explicit image ceiling 256. The common image budget is 256. These bounded measurements do not replace earlier full-history examples or establish long native equivalence. The initial wider host suite run overlapped the finder port and failed for that reason; its serialized rerun passes. Source/core/model/probe hashes and healthy restored services are audited locally under `benchmarks/results/k4v-vision-20261102/dv23-*`.

- Remove the finder's blanket mixed/mmproj rejection only after a correct image-aware proof exists. Never pass an image prompt string/dictionary into the current integer-token slicing proof, and never call a text-only proof evidence of K4V after an image.
- Build a separate controlled image-plus-repeated-text-suffix resource probe, using the existing target-bias constraint where appropriate. Use a declared sequence of image-conditioned positive-copy and genuine-miss requests when needed; a repetition-only request need not naturally exercise both sources. Require actual copied and neural source activity, successful target verification and exact generated/draft/accept totals. Keep it separate from ordinary timing/quality workloads and respect configured N/K/hit bounds; insufficient proof history is not VRAM OOM.
- Preserve the user's actual image workload or generated seven-ratio profile. Explicitly track any additional proof encoder phases so strict phase-count checks remain exact. Account for every completed text prelude, image, route proof and common/bootstrap response once in cumulative source statistics.
- Reuse `vision-memory-v2`, native geometry, complete graph/operator/library/padding quotes, physical max+1 rejection and independent/common settings. Refit mixed mode rather than using a pure DFlash/MTP arena. A fit with copying never exercised is not a mixed resource qualification.
- Host tests cover supplied images, automatic ratios, no post-image text, common setting different from independent maxima, parser rejection and partial/missing counters. Execute representative Qwen F3/K7 and F7/K3 image fits. Run one full selected workload only when publishing that full-history claim; never silently label bounded preflight as full validation.

Exit: the tool prints executed resolutions and a safe ready-to-run command for the exact mixed configuration, with explicit source coverage and honest failure/validation scope.

#### DV24 - Complete public image, cache and restart lifecycle

Status: complete. The existing public vision harness now supports independent K4V parameters while preserving pure-mode defaults. It exercises genuine copied suffixes and neural misses alongside the ordinary image forms, cache/changed/adjacent/later images, malformed/oversized requests, low/high/low, complete save/erase/restore, truncated restore recovery and process restart. Cancellation now waits for actual generated IDs rather than native progress placeholders, and mixed tests cancel after copied or neural work before requiring an exact fresh text continuation.

An actual F3/K7 image rejection reproduced a native full-checkpoint bookkeeping defect: the replayed correction was counted as one accepted copied proposal. The common acceptance interface now separates the physical verified neural boundary from the logical original-proposal count; the old three-argument call remains. K4V/map feedback and source statistics use the logical count, while MTP/EAGLE retain the physical hidden-state row. The tempting logical-only prototype was rejected: an actual MTP3 image/full-checkpoint observer captured 1,497,376 bytes of embedding inputs, all bit-identical between the established baseline and corrected two-count implementation, and different in that unshipped prototype. The observer forwards the unchanged API and is a functional control, not a timing measurement. The first observer used the wrong public embedding setter; its missing-trace failure is retained separately.

Current native and ASAN/UBSAN/leak checks pass 578 speculative assertions and the server/argument controls; 52 vision, 46 Gemma and 182 benchmark host tests pass with one expected benchmark skip. Five GPU gates pass. F3/K3 has 30 corresponding native/streamed completion records per mode with exact tokens/draft/cache counts and byte-identical 321,082,524-byte saved states. F3/K7 streamed recovery and F7/K3 8-Ki window recovery each add 30 records; the latter preserves a complete 571,911,928-byte state beyond the draft window. Their larger full-native projector configurations retain their capacity limits and use the scoped DV21 component reference rather than a false full-server equivalence claim. The tracked replay control has strict per-mode source totals and output IDs, with native RS/F and streamed RS/max(F,K) recovery work explicitly distinguished. Source/model/build hashes and restored healthy services are audited locally under `benchmarks/results/k4v-vision-20261102/dv24-*`. Identical completed state payloads share read-only storage without losing any artifact paths or bytes.

- Extend the existing vision harness with explicit mixed parameters and source observations. Check all standard API image forms already supported by the feature, first/last/only/portrait/narrow/changed/adjacent/later images, positive copied suffixes and genuine no-copy fallback cases.
- Cover fresh/cached requests, low/high/low phases, save/erase/restore, malformed/truncated state, process restart, cancellation during encoder work and during either proposal source, error recovery and context end. Preserve complete state-file extent/token counts and native pairs where fitting.
- Compare corresponding native and streamed scenarios, not fresh versus cached/native cross-layout assumptions. A rebuilt map may have different proposal counts from uninterrupted generation; compare matching reset protocols and document that distinction.
- Require real copying and real DFlash prediction in positive lifecycle scenarios, correct supplier feedback, no stale image-conditioned features, returned vision loans and valid target continuation. Test held-loan/queued failure/reconstruction with the existing frame/owner checks.

Exit: the public mixed image path is functional and recoverable across the full supported lifecycle, with per-scenario tokens, counters and available state byte comparisons.

#### DV25 - Audit safety, regressions and useful performance

Status: complete. The final compiled runtime passes 34 accepted GPU numerical, memory and regression gates, including nine zero-error strict jobs. Device checks retain default PDL, all API errors, exit 99, all relevant child processes and no filters. The graph-owner gate keeps graph-disable variables absent and asserts real captures; eager execution is checker-only. Coverage includes widest target and neural mixed-image paths, MTP full-checkpoint replay, D256/D512 reader/operator matrices, writer/read-only/owner/copy/recurrent controls, Qwen MTP exact vectors, all three mixed native decoder-component transcripts, pure Qwen/Gemma regressions and three full-model cache pairs. Thirteen native and sixteen affected ASAN/UBSAN host suites pass; all 53 vision, 46 Gemma and 182 benchmark host tests pass with one expected benchmark skip.

The real mixed image ASAN lifecycle passes 30 native and 30 streamed records with identical corresponding outputs/work and complete saved states, no host sanitizer errors and confirmed CUDA model buffers. Its first attempt accidentally omitted the previously qualified `protect_shadow_gap=0` CUDA-ASAN setting and fell back to CPU. A no-model/no-arena diagnostic reproduces that default initialization failure in both baseline and current ASAN binaries with free VRAM, and both initialize CUDA using the accepted profile. Leak detection, halt-on-error and UBSAN stay enabled. A controlled host red additionally proved that a healthy CPU fallback could be mistaken for the requested GPU profile; the harness now fails closed before inference. The failed attempt is preserved and is not a GPU pass. Earlier accepted compiled-runtime results are adopted only after explicit review of that fail-fast-only harness delta, unchanged CXX/GPU/model/command identities and confirmed CUDA execution.

Ordinary performance uses a fixed 1-GiB arena, context capacity 8192, Q8/Q4, b/ub256, UVM off, fresh image requests, 128 outputs and three alternating paired trials, with one warmup per process. All 48 timed responses retain 6144 exact generated IDs within their comparisons, and all 12 paired physical decode-layout sequences match. Existing MTP3 and DFlash7 code-comparison median decode changes remain within 0.27%; no material existing-mode regression is observed in these scoped checks. N12/M7/H2 makes no eligible copies on the two ordinary fixtures and shows no meaningful gain. A separate N12/M7/H1 cohort accepts 98 copied code tokens, reduces neural forwards from 17 to 3 with the same 17 target verification passes, and improves the constructed post-image code-copy case by 9.98% paired-median decode (all three pairs +9.43% to +10.99%); request latency decreases about 2.7%. Captioning remains unchanged within noise. This is a deliberately constructed copy task, not an agent/image-quality result or a universal K4V speedup. The measured host index reserve is 1 MiB plus dynamic 88-byte key-statistic entries and existing history capacity on this build; the private lookup descriptor is 16 bytes. No GPU map or new device allocation is introduced.

Complete source/model/library/profile/command/loaded-library and result hashes, all real sanitizer reports, capacity findings and healthy service restoration are audited locally under `benchmarks/results/k4v-vision-20261102/dv25-*`. No CUDA kernel/header/generator changed, so unchanged legacy device objects are not needlessly rebuilt or represented as new physical-GPU qualification. Raw reports and measurement data remain local.

- Run affected CPU ASAN/UBSAN, actual graph-enabled owner/capture/lifetime checks and strict mixed encoder/target/draft/source-switching device memcheck. Keep default PDL, API-all reporting, error exit 99 and no filters. A presence-based graph-disable variable set to zero still disables graphs; capture-required gates need the variable absent and actual capture assertions. Eager execution is checker-only where explicitly scoped.
- Fresh regressions cover pure DFlash vision, mixed text, Qwen target/MTP vision and text, Gemma target/shared-KV vision, all-64 reader guards and three full-model cache pairs where affected. Preserve controlled encoder pool/library coverage, persistent state, source generations, phase errors and the cached-writer destination fix.
- Compare existing source versus candidate with K4V disabled to detect code regressions, and candidate pure DFlash versus candidate mixed mode to isolate K4V. Keep identical weights, images, prompts, KV types, F/K, batch sizes and fixed arenas within each relevant pair. Independently fitted arenas are a separate capacity comparison. Same-mode code comparisons require exact output/work/source counts and layouts. Pure-versus-mixed proposal counts differ by design; interpret equal-output timing separately and retain any unmatched responses without relaxing numerical checks. Do not mix Q2 image and Q4 text acceptance/timings.
- Measure ordinary end-to-end image latency, encoder/reload, text prefill/decode, lookup/indexing CPU cost, host reserved map/view capacity, neural prediction forwards, target passes, acceptance and physical decode layout. Controlled repetition proves paths/resources; natural or clearly labelled constructed multimodal workloads assess useful performance. Open-SWE text prefixes with inserted fixtures are not native multimodal dataset traces or agent task-success scores.
- Use several representative contexts/images first, repeat matched measurements only where a gain/regression needs confirmation, and investigate material existing-feature harm. Do not promise general K4V speedup or infer it from lookup hits, skipped kernels or instrumented timings. Preserve useful correct support even if its scope yields no speedup, with the limitation documented.
- Audit complete source/model/build/library/profile/probe hashes and every accepted error summary. No CUDA arithmetic/header/generator change is expected; if dependencies actually change, qualify the affected native/legacy objects proportionately without a clean compile-time benchmark or unsupported physical-GPU claim. Own/drain all child processes and restore any temporarily managed services.

Exit: current mixed vision is backed by numerical, memory, failure and ordinary performance evidence with no unresolved material regression or misleading reference/capacity claim.

#### DV26 - Publish and close K4V image support

Status: complete. The README now briefly lists matching mixed-image support and its standard launch flags, with detailed lookup/recovery/API, actual sizing and scoped performance in the focused vision/K4V/capacity reports. Related speculation, memory-consumer and test guides agree on post-last-image text, independent limits, per-generation/restore rebuilding, inherited formats, separate verified-boundary/proposal counts, one parent and unsupported pairings. The real 2110/2111-MiB bounded seven-ratio examples remain distinct from earlier full-context MTP examples. No user sample-image flag or running production service is required by the published automatic finder examples.

Local checks validate changed Markdown paths/anchors, shell syntax, balanced fences and all completed boxes. No local Markdown/Mermaid package is installed; every existing Mermaid block is byte-identical to the accepted baseline. The upstream README body remains exactly 7610 bytes with SHA256 `8e43128b52e5e99fcfd7b65b4d182681566f98f72e7e8f812f71847ef8ec1dbf`. Final host tests, compiled CXX/GPU/library identities and healthy service restoration are verified separately; binaries keep their measured pre-commit source mappings. Raw data and test runtime archives remain local, unrelated files are preserved, no push or production model/configuration change is made, and no owned GPU/build/watch work remains.

- Update the brief README support/launch examples, focused DFlash/K4V/vision reports, finder examples and this checklist. Document the post-last-image text-only lookup scope, legitimate early misses, rebuild-on-generation/restore policy, independent F/K bounds and native-versus-component references.
- Publish real measured arena/resolution/source-coverage results with bounded/full-history and performance scopes separated. Do not replace the fully validated earlier examples with an unvalidated larger profile. Keep raw benchmark results/data and server-AB additions local under the existing publication boundary.
- Verify actual commands/options, Markdown paths/anchors, any changed diagrams and upstream README byte preservation. Recheck code/binary provenance after documentation-only changes; do not relabel old builds as newly compiled commits.
- Check every completed DV18-DV26 box against its accepted gates, use descriptive one-sentence stage commits, leave unrelated files untouched and end with healthy services and no owned GPU/build/watch resources. No pushes or production model/configuration changes.

Exit: the matching Qwen DFlash2+K4V+mmproj pairing can be advertised with the declared lookup/verification/memory policy and reproducible qualified usage. Unrelated K4V/MTP/Gemma modes remain outside this release.

## Expected file boundaries and review units

| Area | Prefer extending |
| --- | --- |
| Common admission/limits | `common/common.cpp`, existing declarations and speculative parameter helpers |
| K4V history and media lifetime | `common/ngram-map.cpp/.h`, `common/speculative.cpp/.h`, existing `server_tokens` helpers and speculative call sites in `tools/server/server-context.cpp` |
| DFlash features and source routing | `common/speculative.cpp`, minimal existing feature/provenance metadata |
| Context graph sizing and feature capture | `src/llama-context.cpp` and its private declarations |
| Parent/child phase lifetime | `src/llama-context-memory.cpp/.h`, existing bounded-cache and lease infrastructure |
| Server startup/requests/cache | `tools/server/server-common.cpp/.h`, `tools/server/server-context.cpp` |
| MTMD phase/execution | `tools/mtmd/mtmd-session.cpp`, `mtmd-workspace.cpp`, existing MTMD planning and helpers only when required |
| CUDA operator scratch and cuBLAS ownership | `ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh`, and only other host adapters identified by DV10 |
| Workspace requirements/binding | Existing private `ggml/src/ggml-backend-memory.*`, execution bindings and `src/llama-memory-workspace.*` contracts; add only the missing generic operator seam |
| Scratch/stream/quote regression tests | Existing `tests/test-memory-workspace.cpp`, `test-context-memory.cpp`, `test-mtmd-workspace.cpp` and backend operator tests |
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
- [x] DV4: validated bounded-writer state survives complete vision loans and strict memory checks.
- [x] DV5: real encoder inputs, exact image-conditioned target/draft vectors and forced acceptance/correction match fitting native controls.
- [x] DV6: pure DFlash common/server/request/finder integration works with actual images; mixed K4V was excluded from that foundation stage.
- [x] DV7: image/cache/window/cancellation/state/restart and attached-writer encoder failure/retry pass.
- [x] DV8: scoped numerical/memory/regression/performance evidence audited; approved finder-test correction passes.
- [x] DV9: pure DFlash2 vision documentation and release scope complete; local resources restored; finder-test maintenance complete.
- [x] DV10: controlled encoder allocations inventoried and coverage red reproduced after exact native controls; private scratch remains to migrate.
- [x] DV11: declared operator/cuBLAS workspace requirements quoted with checked metadata, concurrency and native-allocation comparisons.
- [x] DV12: native CUDA operator components bound to parent leases with exact outputs, completion ownership and no native-pool escape; MTMD hookup remains DV14.
- [x] DV13: explicit cuBLAS workspace has qualified borrowed ownership, stream/capture lifetime and setup/retirement recovery; MTMD hookup remains DV14.
- [x] DV14: combined vision admission, phase transitions and failure recovery qualified for Qwen/Gemma, with exact native output and controlled CUDA storage inside one parent.
- [x] DV15: automatic per-ratio/common-resolution fitting requalified against complete controlled scratch and exact published full-profile execution.
- [x] DV16: fresh numerical, allocation, sanitizer, lifecycle and regression/performance evidence audited.
- [x] DV17: extended memory/capacity contract documented and local resources restored.
- [x] DV0-DV17 historical release boundary: that foundation excluded K4V+mmproj; the completed DV18-DV26 extension qualifies it separately.

### K4V vision extension checklist

- [x] DV18: current mixed-image artifacts, source contracts and unsafe history joins reproduced with valid controls.
- [x] DV19: checked K4V-only post-image text view passes positive, boundary and existing text-map CPU tests.
- [x] DV20: generation, media identity, cache/checkpoint and pending acceptance lifetime are integrated safely.
- [x] DV21: actual mixed image source switching, every accepted prefix and TG1-TG8 match scoped native/component controls.
- [x] DV22: only the qualified mixed startup/request set is admitted, with independent widths, exact quotas and preserved guards.
- [x] DV23: image-aware finder/ratio output and real post-image copy/neural resource proofs execute and account correctly.
- [x] DV24: public images, cache/state/cancel/restart/window lifecycle pass with real mixed source coverage.
- [x] DV25: current numerical/sanitizer/regression/performance and resource/provenance evidence is audited.
- [x] DV26: mixed-image support, limits and measured commands are documented, boxes checked and local resources restored.

Pure DFlash2 vision is complete only when DV0-DV9 pass. A helper subgate, successful encoder call, configured projector, target-only fallback, or process exit does not satisfy whole-feature acceptance. Optional K4V vision has its own completion boundary.

The controlled CUDA workspace extension is complete only when DV10-DV17 pass. DV0-DV9 remain complete for their recorded graph-workspace scope; neither the older feature ledger nor the completed automatic finder alone qualifies operator-private allocation coverage.

The K4V vision extension is complete only when DV18-DV26 pass and the matching public guards are opened by those qualified changes. The completed DV0-DV17 and text-only K7 ledgers do not themselves admit mixed image history.
