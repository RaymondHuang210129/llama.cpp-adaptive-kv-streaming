# Reusing selected prefill frames

Checkpoint export retains the active frame. A change in the next prefill batch's row or output counts previously still retired that frame, copied native state to the host, and reconstructed its arena grants. Checkpoint boundaries and unequal prompt tails can produce several such changes within one prefill phase.

The server now quotes a changed prefill workload using the same selected models and recurrent provider. Reuse requires that every measured graph, encoding, attention, publication, bounded-cache, current-state, and host region fit its existing grant. Each source must also stay within its admitted row capacity. Spare capacity in one region cannot cover an overflow in another. The normal admission path handles profiles that do not fit, and changes in cohort, phase, input representation, or causality.

Native per-sequence snapshots omit expired sliding-window rows. Retirement therefore also changes the bounded cache's metadata and can change its attention extent. Reusing the frame must preserve that pruning step to retain the same numerical behavior. `prepare_group_prefill_reuse` acquires the idle phase owner, prunes expired target and auxiliary rows, and leaves device bytes, live row positions, canonical host history, and allocation cursors intact. The server records the last actual shape so pruning occurs at the same boundaries as retirement did.

The first implementation passed short Qwen/Gemma numerical tests and the Qwen DFlash/MTP server tests, but diverged on an unequal three-request Gemma target-only fixture extending beyond its sliding window. A model-free failing test isolated the missing metadata pruning. After adding pruning, that same server fixture matched all output tokens. No numerical tolerance was relaxed.

Validation includes:

- `test-context-memory`: individual region and row limits, including smaller totals that overflow one region.
- `test-kv-state-layout`: pruning matches checkpoint metadata and preserves every stored K/V byte and allocation cursor.
- `test-kv-stream-context --model MODEL --group-prefill-reuse-only`: changing row counts with one through four selected sources, sparse IDs, unchanged grants, and bitwise native logits.
- Server comparisons with the committed implementation across Qwen/Gemma, supported target/draft modes, text/images, and one/three slots.
- CUDA memcheck, ASAN, and a late-arrival checkpoint test with late cuBLAS handle creation blocked.

The arena size and prefill batch boundaries remain unchanged. Reuse still pays for an exact host-side quote when the shape changes. This optimization removes eligible frame retirements; it does not eliminate admission work or required phase transitions.

On RTX 5070 Ti, Qwen IQ4_XS with vision weights loaded, DFlash7, a 2314 MiB arena, three 8K SWE prefixes, and 512 forced repeated `B` outputs per slot gave:

| Metric | Committed control | Frame reuse |
| --- | ---: | ---: |
| Initial text-prefill bindings | 5 | 1 |
| Mean request prefill time | 26.453 s | 24.597 s |
| Aggregate three-slot decode | 204.63 tokens/s | 206.29 tokens/s |

This was one pass per build, control followed by candidate, without warmups or reverse-order runs. Prefill time fell 7.0%; the 0.8% decode difference does not establish a decode speedup. All 1,536 forced output tokens and all per-slot draft/acceptance counts matched. Separate free-generation text/image comparisons matched 6,016 tokens across 12 supported Qwen/Gemma mode and slot-count cases. The late-arrival fixture used 8K/4K/2K prompts and retained checkpoints for all three requests with late cuBLAS handle creation disabled.
