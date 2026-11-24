# Fork CI and binary releases

`workflows/ci.yml` checks pull requests targeting `feature/adaptive-kv-stream-v3`, pushes to that branch, and manual runs. It combines the results in the `required` job. Configure branch protection against that job after its actual check name has appeared in a successful GitHub run.

## Required checks

| Configuration | Purpose | Release package |
|---|---|---|
| Linux CPU, shared and static | Compile, link and run portable tests | None |
| Windows CPU, shared | Compile, link DLLs and run portable tests | None |
| Linux CUDA, RTX 50 only | Compile and link CUDA/server/tools; check packaged startup and dependencies | `.tar.gz` |
| Windows CUDA, RTX 50 only | Compile and link CUDA/server/tools; check packaged startup and dependencies | `.zip` |

The required CUDA builds select `120a` on both operating systems, using CUDA 13.0.1 and CCCL v3.4.3. CUDA builds start after the CPU build/test/package and Python/source gates succeed, so already-failing commits do not consume the expensive compiler slots. They do not compile older architectures. Python typing, lint, host tests and source consistency checks remain enabled. Server API, ASan/UBSan and UI tests run for relevant paths; CI or CMake changes request all suites. A manual main-CI run requests the full set. Draft PRs run lightweight checks; marking a PR ready triggers compilation and the applicable suites.

The aggregate rejects failed, cancelled, missing or unexpectedly skipped required jobs. Superseded PR runs are cancelled; different branch commits retain separate runs. Internal `cpu-test-*` archives transfer binaries to server tests and expire after one day. They are excluded from releases.

Temporarily, Windows CPU/CUDA jobs use `--fatal-warnings OFF` to collect the warning inventory. Warnings remain visible; compiler errors and test failures still fail CI. Linux builds and the build helper's default remain strict. Restore Windows warning promotion after the inventory is fixed.

## CUDA coverage

`fork_ci.py` owns the architecture profile table used by configuration, toolkit setup, packaging and schedule selection. Every profile builds Linux x64 and Windows x64 separately. The weekly workflow runs at **03:17 America/Chicago**, with daylight-saving handling delegated to GitHub's timezone-aware schedule.

| Profile | GPU family | CUDA target | Toolkit | CCCL | Trigger |
|---|---|---|---|---|---|
| `rtx50` | RTX 50 | `120a` | 13.0.1 | v3.4.3 | Required PR/push CI |
| `rtx40` | RTX 40 / Ada | `89-real` | 13.0.1 | v3.4.3 | Monday |
| `rtx30` | RTX 30 / Ampere | `86-real` | 13.0.1 | v3.4.3 | Tuesday |
| `rtx20` | RTX 20 / T4 | `75-real` | 13.0.1 | v3.4.3 | Wednesday |
| `v100` | V100 / Volta | `70-real` | 12.9.1 | v2.8.2 | Thursday |
| `p100` | P100 / GP100 | `60-real` | 12.9.1 | v2.8.2 | Friday |
| `pascal` | GTX 10 / P40 | `61-real` | 12.9.1 | v2.8.2 | Saturday |

CUDA 12.9.1 supplies the Volta/Pascal compiler and runtime dependencies. P100 explicitly targets SM60; an SM61 build does not substitute for it. SDK manifests and downloads remain checksum-pinned. The older profiles use the CUDA-12-compatible CCCL series. Dedicated SM80/SM90/SM100 profiles remain outside this initial schedule.

`build-cuda-compat.yml` also accepts manual `profile` and optional full `source_sha` inputs. Leaving the SHA blank selects the latest successful V3 push CI with a published `rtx50` profile release. An explicit SHA must have that same release and successful CI provenance. The resolver rejects draft/incomplete releases, missing or redirected tags, mixed OS provenance, wrong toolchains and unrelated/failed CI runs. Complete one V3 push build before bootstrapping compatibility builds.

The selected SHA is resolved once, then used for UI checkout/build identity, every CUDA checkout, shard validation, artifact names and package provenance. It can differ from the scheduled workflow's `GITHUB_SHA`; the source SHA is carried separately as `OCEANUS_SOURCE_SHA`. The schedule must exist on the repository's default branch, currently V3. [GitHub schedule behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

These jobs compile CUDA and check packaged executable startup without a GPU. They do not establish numerical correctness or throughput on the corresponding physical cards. Hardware qualification remains separate from package availability.

### Parallel CUDA compilation

Both build paths reuse `build-cuda-sharded.yml`. A planning job reads Ninja's complete compilation database. Each private `kv-stream-*` translation unit gets its own shard; remaining CUDA files are distributed across the configured worker count. The merge path allows eight concurrent shards per OS; weekly/manual compatibility builds allow two per OS and share one compatibility-workflow concurrency group. Each shard uses one compiler process. All admitted attention formats and D256/D512 source units remain included.

Each shard returns compressed compiler objects and source/command/dependency checksums. Final linking rejects missing or duplicate objects, changed headers, mixed source/toolchain cohorts and invalid archives. The existing CMake hook substitutes validated objects in the original link order; it refuses to rebuild CUDA sources serially during final linking. Runner image labels are recorded per stage rather than required to match. Compiler, frontend, assembler and linker bytes remain part of the build identity; source, configuration, SDK and consumed-header checks remain strict. UI artifacts are named by their exact source SHA and come from the same checkout rather than upstream `latest` downloads.

GitHub-hosted jobs have a [six-hour execution limit](https://docs.github.com/en/actions/reference/limits). Shards and final linking use at most 360 minutes per job, with a shared 24-hour compilation/link deadline measured from plan creation. Setup and artifact/publication steps do not enforce that deadline, so it is not a strict workflow-wide wall-clock limit. Object archives expire after one day; their compressed sizes are reported.

Linux uses GCC 13 on Ubuntu 24.04 and an ephemeral swap file for heavy compiler processes. Windows uses MSVC on `windows-2022`. CMake 3.31.10 and Ninja 1.13.0 remain pinned. Native object-handoff checks run before expensive compilation on each OS. Windows was already SM120a-only before this change; successful hosted completion must still be measured rather than inferred from Linux's architecture reduction.

`diagnose-cuda-compile.yml` retains manual single-source/architecture profiling and the `profile-cuda-compile` PR label trigger, now targeting V3. It records compiler timings and memory samples without publishing packages. Its `all` architecture choice follows the new RTX 50 default; explicit architecture overrides remain available for diagnostics.

## Commit-SHA releases

A successful V3 push publishes an `rtx50` prerelease. Scheduled/manual compatibility runs publish their own profile only after both OS builds and packaged-startup checks succeed. Release tags are **`v3-<profile>-<full-40-character-sha>`**, containing:

- `oceanus-v3-<sha>-<profile>-linux-x64-cuda.tar.gz`
- `oceanus-v3-<sha>-<profile>-windows-x64-cuda.zip`
- `SHA256SUMS`
- `build-info.json`

The archives contain `llama-server`, `llama-cli`, `llama-bench`, backend/shared libraries, matching CUDA redistributables and dependency licenses. Models and GPU drivers are not bundled. Linux packages are built on Ubuntu 24.04; older glibc compatibility is not implied.

The manifest identifies source SHA, architecture profile, run/attempt, compiler/configuration, SDK receipt, shard provenance and checksums. Validation requires both OS packages to match the profile's architecture, CCCL and toolkit. Weekly binaries can lag the latest RTX 50 release: choose the profile and SHA shown in the release name and manifest.

Pull requests and manual main-CI runs cannot publish. The publisher stages assets in an owned draft and publishes only after validation. Reruns verify an existing complete release and preserve its original bytes. An interrupted draft may be resumed by the same Actions run ID; another run's draft is retained for review. Compatibility publication revalidates the source's successful RTX 50 release before writing its own release. Existing `sha-<sha>` V2 releases and assets are not modified.

## Local checks and remaining rollout

```sh
python -m pip install PyYAML
python -m unittest discover -s .github/scripts/tests -v
actionlint
```

Local checks cover policy, source/profile rejection cases, packaging recovery and real native object-handoff/link fixtures. Actual hosted Linux/Windows completion, all scheduled architecture builds, live publication and the final branch-protection check name remain rollout gates. Local policy tests do not close those gates. See the [migration ledger](../docs/development/ci-workflow-migration.md) and [V3 release plan](../MULTI_SLOT_PHASE_ARENA_ROADMAP.md#v3-release-preparation---user-documentation-staged-cuda-builds-and-branch-promotion-ci-implementation-hosted-rollout-pending).

The vocabulary corpus remains pinned to the last fixture revision before this fork's tokenizer baseline. Update `LLAMA_TEST_VOCABS_REVISION` when rebasing tokenizer code and qualify the complete new corpus. The shell driver propagates fixture failures and rejects an empty corpus.
