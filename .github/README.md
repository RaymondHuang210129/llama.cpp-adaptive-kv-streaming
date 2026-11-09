# Fork CI and binary releases

`workflows/ci.yml` checks pull requests targeting `feature/adaptive-kv-stream-v2`, pushes to that branch, and manual runs. It combines the results in the `required` job. Configure branch protection against that job after its actual check name has appeared in a successful GitHub run.

## Required checks

| Configuration | Purpose | Release package |
|---|---|---|
| Linux CPU, shared and static | Compile, link and run portable tests | None |
| Windows CPU, shared | Compile, link DLLs and run portable tests | None |
| Linux CUDA, shared | Compile and link CUDA/server/tools; check packaged startup and dependencies | `.tar.gz` |
| Windows CUDA, shared | Compile and link CUDA/server/tools; check packaged startup and dependencies | `.zip` |

Temporarily, the main Windows CPU and CUDA jobs use `--fatal-warnings OFF` to collect the complete warning inventory. Warnings remain visible; compiler errors and test failures still fail CI. Linux builds and the build helper's default remain strict. Restore Windows warning promotion after the inventory is fixed.

Python typing, lint and host tests run on every event. Source consistency checks remain enabled. Server API, ASan/UBSan and UI tests run for relevant paths; CI or CMake changes request all suites. A manual run requests the full set. Draft PRs run lightweight checks; marking a PR ready triggers compilation and the applicable suites.

The aggregate rejects failed, cancelled, missing or unexpectedly skipped required jobs. Superseded PR runs are cancelled; different branch commits retain separate runs. Internal `cpu-test-*` archives transfer binaries to server tests and expire after one day. They are excluded from releases.

UI assets are built from the same checkout and passed to the C++ builds. Packages do not download upstream `latest` UI assets. The Linux compiler is GCC 13 on Ubuntu 24.04; Windows uses the MSVC environment on `windows-2022`. CMake 3.31.10 and Ninja 1.13.0 are pinned. Each CUDA shard uses one compiler process; Linux adds swap on the ephemeral runner for heavy compiler processes. Actual hosted build resource requirements still need validation.

### Parallel CUDA compilation

The Linux and Windows components call `build-cuda-sharded.yml`. A planning job reads Ninja's complete compilation database. Each private `kv-stream-*` translation unit gets its own shard; the remaining CUDA files are distributed across eight groups. Up to eight shards run concurrently per OS, using free standard hosted runners. Sharding preserves every selected architecture and format specialization without regrouping translation units.

Each shard returns compressed compiler objects and their source/command/dependency checksums. The final job rejects missing or duplicate objects, changed headers, mixed source/toolchain cohorts and invalid archives. A private CMake hook substitutes validated objects in the original source/link order. The final job compiles host code, links the complete CUDA backend, checks packaged startup and records shard provenance in the package manifest. It refuses to rebuild CUDA sources serially during final linking.

GitHub-hosted jobs have a [six-hour execution limit](https://docs.github.com/en/actions/reference/limits), so setting a single job to 24 hours does not extend it. Shards and final linking use at most 360 minutes per job, with a shared 24-hour deadline measured from plan creation and checked before and during compilation/linking. Setup and artifact/publication steps do not enforce that deadline, so it is not a strict workflow-wide wall-clock limit. An expired plan requires a new workflow run.

Temporary object archives use Actions artifact storage and expire after one day. Their compressed sizes are reported; free public-runner compute does not make [artifact storage unlimited](https://docs.github.com/en/billing/concepts/product-billing/github-actions). The native handoff self-test runs on each selected OS before the expensive CUDA compilation. Local C++ tests and a real NVCC SM75/SM120 object-handoff/link fixture pass; the complete hosted CUDA and Windows builds remain required. No GPU execution is implied by the fixture.

## CUDA coverage

The default toolkit is NVIDIA CUDA 13.0.1, installed from checksum-pinned redistributables without a driver. Linux PR and release builds use `75-virtual;86-real;89-real;120`: Turing PTX, native RTX 30/40 code, and native Blackwell code plus PTX. Existing CUDA CMake logic normalizes SM120 to SM120a for the native Blackwell path. Windows PR and release builds temporarily use only `120a` (native code plus architecture-specific PTX) while investigating compiler timeouts. These Windows CUDA packages target SM120a-compatible Blackwell GPUs; they do not contain CUDA code for older GPU architectures. Dedicated SM80 and SM90 targets are omitted from both defaults. Build manifests record the compiler and CMake architecture configuration. Custom `--cuda-architectures` selections remain available.

These hosted jobs compile CUDA and check executable startup, not GPU execution. They do not establish numerical correctness or performance on physical GPUs. Paid hosted GPU validation remains a separate decision. `build-cuda-compat.yml` provides manual CUDA 12.9.1 compilation for legacy SM61/SM70; those targets are not included in the initial CUDA 13 release packages.

`diagnose-cuda-compile.yml` is a manual probe for the two largest D512 runtime units. By default it compiles `kv-stream-mma-runtime-4-2.cu` for SM75 on both Linux and Windows, with the same optimization settings as the normal build. An explicit diagnostic target can investigate architectures outside an OS's current release profile. The `all` choice follows each OS's release defaults; select one architecture for a matched OS comparison. It records NVCC phase timings, compiler output and process memory/CPU samples; Windows also records system paging counters and private committed memory (which is not the number of bytes swapped). The collector must pass a telemetry preflight before compilation starts. This is a separate diagnostic profile, not an architectural change to release packages or an uninstrumented compilation-speed benchmark. Sampling failures and unfinished compilation are explicitly reported. The compiler deadline is 330 minutes to leave time for artifact upload within the six-hour job; no backend package is published.

For pre-merge investigation, add the `profile-cuda-compile` label to a PR targeting v2 to run the default probe. Other labels do not start it. This does not restart the normal CI workflow. Each probe uses one compiler runner per OS; start it when the normal CUDA jobs are idle to avoid exceeding the intended compiler concurrency. Manual dispatch becomes available when the workflow exists on the default branch.

Other retained platform workflows are manual. HIP/MUSA compilation is in `build-gpu-alternatives.yml`. Apple, Android, Vulkan, SYCL, OpenCL and other manual builds are not prerequisites for the two published CUDA packages. No workflow waits for an upstream or self-hosted runner.

## Commit-SHA releases

A successful v2 push publishes a GitHub prerelease tagged `sha-<full-40-character-sha>` containing:

- `llama-kv-stream-<sha>-linux-x64-cuda.tar.gz`
- `llama-kv-stream-<sha>-windows-x64-cuda.zip`
- `SHA256SUMS`
- `build-info.json`

The archives contain `llama-server`, `llama-cli`, `llama-bench`, the backend/shared libraries, required CUDA redistributables and dependency licenses. Models and GPU drivers are not bundled. Linux packages are built on Ubuntu 24.04; compatibility with older glibc distributions is not implied.

The manifest identifies the exact commit, run/attempt, build configuration, compiler versions and file/archive checksums. Both OS archives must pass validation before publication. Pull requests and manual runs cannot publish. The publisher stages assets in an owned draft and publishes only after checking their sizes and SHA-256 digests. A rerun verifies an existing complete release and preserves its original bytes; it never replaces a published SHA release. An interrupted draft may be resumed by the same Actions run ID. A draft owned by another run is retained for review rather than overwritten.

## Local checks and remaining rollout

```sh
python -m pip install PyYAML
python -m unittest discover -s .github/scripts/tests -v
actionlint
```

The migration has local Linux CPU, Python, package-startup and workflow-policy checks. Windows and CUDA builds, real release uploads and the final branch-protection check name require actual GitHub Actions runs. No GPU runtime result is implied by a local policy or packaging test. See the [migration ledger](../docs/development/ci-workflow-migration.md) for acceptance gates.

The vocabulary corpus is pinned to the last fixture revision before this fork's upstream tokenizer baseline. Update `LLAMA_TEST_VOCABS_REVISION` when rebasing tokenizer code, and qualify the complete new corpus. The shell driver propagates any fixture failure and rejects an empty corpus.
