# Numerical continuity across KV state restoration

Saving each sequence separately used to retain that sequence's token order but discard its global KV row positions. For example, two prefill batches could leave `[A0..3, B0..3, A4..7, B4..7]`, while restoration produced `[A0..7, B0..7]`. Attention masks still isolated the requests, but the changed row placement changed floating-point reduction order. The resulting small attention differences could grow through later layers.

An independent CUDA build of upstream commit `10a60cf303566e10d6a7a2774c17d2085503d87b` reproduces this behavior with Qwen IQ4_XS: the short two-sequence fixture reaches maximum absolute logit differences of 0.367842 with F16 KV and 0.500316 with Q8V4 KV. Whole-context restoration is exact in that fixture.

Native KV snapshots now include a tagged layout record containing the original row indices, cache extent, and allocation head. Restoration keeps these coordinates when they fit and are available. Mirrored caches use the primary cache's destination coordinates and allocation head. Importing into a smaller or occupied destination still uses the existing allocation fallback and never overwrites a peer's rows.

Grouped arena state also retains each source's global row indices and allocation head. Rebinding a compatible group can therefore rebuild the original attention projection rather than concatenate its histories again. Group checkpoint version 3 persists this metadata.

The reader accepts the previous native representation and group checkpoint version 2. Those older snapshots lack the original global layout, so their restoration uses the previous placement policy. Older executables cannot read the new representations.

The guarantee is numerical continuity when restoring a compatible layout with the same execution geometry. It does not promise bitwise equality between different batch sizes, kernel choices, or an import that must relocate rows. Combining snapshots from different selections can require relocation when their coordinates conflict or their allocation heads differ. New metadata costs four host bytes per saved row in each representation, plus fixed headers, and stays in host memory.

Checkpoint scheduling can split prefill batches. A restoration comparison must retain the same batch boundaries; changing the checkpoint flag alone also changes the arithmetic workload.

Keeping interleaved native rows can fragment the reader's contiguous spans. For configurations supporting multiple slots, the existing encoded gather staging region therefore holds up to 64 pages within a 16 MiB budget (at least one page). This region is included in the arena quote and charged to the existing arena; the parent allocation does not grow. It leaves less arena space for resident and ring KV. Configurations supporting one slot retain one staging page.

Queued decode gathers several pages in native column order before submitting attention. Byte-copy kernels batch larger rectangles, and page-monotonic layouts retain streamed pages until their final use when ring capacity permits. Permuted layouts and small rings keep the existing fallback. These changes preserve KV bytes and attention column order; they do not change floating-point kernels.

The controlled three-slot Qwen DFlash7 measurement on RTX 5070 Ti recovered from 120.05 aggregate tokens/s for the initial numerical fix to 205.78 tokens/s, compared with 212.57 before the fix. This is a single-pass measurement at 8K SWE tokens per slot with a 2314 MiB arena, forcing 512 repeated `B` tokens per slot to hold output content fixed. The remaining 3.2% difference does not establish performance neutrality or predict long-context or agent workload performance.

Regression coverage:

- `test-kv-state-layout` checks sparse row placement, allocation heads, K/V payloads, occupied-row fallback, legacy input, mirrored caches, and invalid layout metadata without loading model weights. It can run under ASAN.
- `test-kv-stream-context --model MODEL --group-park-continuation-only` compares both native and grouped restoration directly with uninterrupted logits using F16 and Q8V4 KV.
- `--group-active-checkpoint-only` and `--group-active-checkpoint-f16` cover one through four selected sources, sparse IDs, compact/dense/RAM recurrent providers, prefix restoration, and export failures.
- `--group-active-dflash-checkpoint DRAFT` checks target and draft continuation after checkpoint export and restoration.
- `test-kv-stream-vector-spans --cuda-slot-group-workspace-gather` checks native attention bits, ring reuse, and arena canaries with interleaved and permuted layouts and two through four sources. `--smoke` selects a smaller matrix for CUDA memcheck.
- `--cuda-slot-group-reader-grid` exercises the raw reader across all supported KV type pairs.
