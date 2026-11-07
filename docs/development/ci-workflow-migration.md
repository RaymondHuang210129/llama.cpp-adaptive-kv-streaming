# GitHub Actions migration plan

Status: implemented locally on `feature/ci-workflow-migration`; hosted Windows/CUDA runs, live publication and branch-protection activation remain pending. See the [operating guide](../../.github/README.md).

Baseline: `2f63f96e3b8e0c249e3385fe378233b2ffceb528` on `feature/adaptive-kv-stream-v2`.

Confirmed release policy: use the full commit SHA as the package version and publish Linux and Windows CUDA binaries. CPU-only builds are retained for CI testing, not release downloads. Semantic versioning is not required for this pipeline.

Temporary diagnostic exception: the main Windows CPU/CUDA jobs collect warnings with warning promotion disabled, as requested. Linux and default helper invocations retain fatal warnings. Restore Windows fatal warnings after the complete inventory is fixed; strict Windows acceptance remains pending.

## 1. Requirements

- Require Linux and Windows compilation on pull requests targeting the v2 branch.
- Check CPU and CUDA configurations, including complete executable and shared-library linking.
- Preserve platform-independent tests, Python checks, source consistency checks, hosted server tests and hosted UI checks.
- Remove dependencies on self-hosted and upstream-specific runner services.
- Keep additional platform compilation available manually where standard hosted runners can perform it.
- Build downloadable binaries after successful checks on every v2 branch push, including merge commits.
- Publish immutable commit-SHA GitHub Releases containing Linux and Windows CUDA packages. Paid GPU provisioning remains a separate decision.
- Keep model arithmetic, kernel specialization, numerical assertions, model support and production configuration outside this migration.

The original audited branch contained 58 workflow YAML files and `bench.yml.disabled`. The disposition table below preserves that baseline inventory. GitHub also registers an automatic Dependency Graph workflow, which has no repository YAML file.

## 2. Baseline problems addressed by this migration

1. `build-cann.yml` has an empty `jobs:` mapping because all jobs are commented out. It fails workflow validation.
2. Most push filters still target `master`. Code Style and EditorConfig also restrict PRs to `master`.
3. Several workflows require specialized runners. The CPU workflow indirectly requires one through `build-cmake-pkg.yml`.
4. Some path filters miss relevant changes. CUDA PR builds watch CUDA files but not all callers under `src/`; the general sanitizer PR trigger watches only its own workflow file.
5. Some tests use `continue-on-error`, so an overall success can conceal failed coverage.
6. Several workflows rely on upstream Hugging Face cache services, publishing credentials or repository conventions.
7. Scheduled publishing, issue maintenance and hourly toolkit-cache jobs do not serve the requested PR/build pipeline.
8. Flake8 violations still need correction. The Python type-check and requirements workflows passed on the baseline commit; those fixes must be preserved.

Expired approval runs from contributor PRs are a separate GitHub approval issue. They must not be recorded as compiler/test failures or bypassed by executing contributor code through `pull_request_target`.

## 3. Workflow structure and event policy

Add `.github/workflows/ci.yml` as the entry point for:

- `pull_request` targeting `feature/adaptive-kv-stream-v2`, including updates to draft/ready state as needed.
- `push` to `feature/adaptive-kv-stream-v2`.
- `workflow_dispatch` for investigation and full validation.

Reuse existing workflow files and job steps through `workflow_call` where practical. Remove their duplicate automatic triggers when the orchestrator takes ownership. A second entry point is allowed only where it has a distinct purpose, such as manual additional-platform compilation.

The orchestrator must always start and report a stable `CI / required` result. Do not use top-level path filters that leave a required check absent. Any selective execution must happen inside the workflow, with an explicit required-job set for the event and changed paths.

Initial policy:

- Run the four Linux/Windows CPU/CUDA build jobs on every non-draft PR and every v2 push.
- Run portable host tests and Python checks on every non-draft PR and v2 push.
- Run ASan/UBSan, server API and UI checks for relevant changes; workflow/build-system changes request the full set.
- Run vendor/tokenizer/generated-document checks on their existing relevant inputs, with corrected branch handling.
- Keep a manual full-check mode that requests every retained required suite.
- Draft PRs may run lightweight feedback checks, but becoming ready for review must trigger the complete required set.

`CI / required` must fail for any required job that fails, is cancelled or is unexpectedly skipped. Intentional path-based skips are accepted only when the job was not required for that change. Enforce this policy in branch protection/rulesets after the new check names have produced real runs.

PR builds should validate the proposed merge with the v2 branch. Branch builds must use the exact pushed SHA. Record both the PR head and tested merge SHA where applicable.

Cancel superseded runs for the same PR. Preserve independent runs for v2 pushes; do not use a branch concurrency policy that discards intermediate commits awaiting their builds. Release uploads are scoped to the full SHA, so an older build cannot overwrite another commit's packages. Do not introduce a mutable rolling tag in this migration.

## 4. Required build and test coverage

| Check | Runner | Required result |
|---|---|---|
| Linux CPU | Standard hosted Ubuntu x64, pinned GCC | Compile all configured server/tool/test targets, link shared libraries, run portable CPU tests |
| Windows CPU | Standard hosted Windows x64, pinned MSVC | Compile and link server/tools/tests, including DLL imports/exports; run supported CPU tests |
| Linux CUDA | Standard hosted Ubuntu x64, pinned CUDA toolkit | Compile the CUDA backend, D256/D512 implementations and configured format specializations; link the complete server/tools |
| Windows CUDA | Standard hosted Windows x64, pinned CUDA toolkit/MSVC | Compile CUDA and link the complete server/tools with the intended DLL layout |
| CMake package | Standard hosted Ubuntu | Build/install the package and compile a consumer against the installed result |
| Python | Standard hosted Ubuntu | Run type checking, Flake8, requirements consistency and tracked benchmark/server host tests |
| Source consistency | Standard hosted Ubuntu | Preserve model naming, EditorConfig, vendor, tokenizer-hash and generated-operations-document checks |
| Sanitizers | Standard hosted Ubuntu | Run portable memory/streaming tests under ASan and UBSan; include selected server tests |
| Server API | Standard hosted Linux/Windows as supported | Exercise the existing API suite with small public fixtures and correct declared Python dependencies |
| UI | Standard hosted Ubuntu | Preserve the hosted UI build, source checks and applicable browser tests |

Additional details:

- Retain fatal compiler warnings. Do not add undefined-symbol linker allowances or disable D512/quantized paths to obtain a green build.
- Validate both static and shared-library configurations across the retained matrix. A static-only Windows build is insufficient to catch the previously observed export failures.
- Pin compiler, CUDA toolkit and action versions. Document the architecture list and supported toolkit floor; do not silently drop older supported compilation profiles.
- Set explicit compilation parallelism and record resource failures. First establish a working standard-runner build for the heavy D512 units. If one compiler process exceeds available RAM, document the measured limit and resolve runner capacity before making that job required; do not weaken optimization or coverage silently.
- Use fork-local GitHub caches with compiler/toolkit/architecture keys. A cache miss must still permit a successful build. No upstream HF cache token is required.
- Keep TSan available on hosted Linux in manual validation until its retained suite is demonstrated reliable; record any existing failure rather than treating it as a pass.
- Fix retained test failures before making the corresponding checks required. Do not replace `continue-on-error` with exclusions or relaxed assertions.
- Build embedded UI assets from the tested source when packaging them. Reuse the same UI artifact across builds; avoid silently fetching an unrelated upstream `latest` UI.
- Use small fixtures for automatic CI. Large-model equivalence, long-context sweeps and architecture-specific performance measurements remain separately scoped qualification.

## 5. CUDA execution coverage

CUDA compilation needs a toolkit but no GPU. It remains part of the required matrix regardless of GPU-service availability.

GitHub's documented hosted GPU option is a Tesla T4 with 16 GB VRAM on Ubuntu or Windows. GPU larger runners require an organization/enterprise on an eligible Team or Enterprise Cloud plan and are billed even for public repositories. The current personal-account repository therefore cannot obtain this by retaining an upstream runner label.

Do not keep queued jobs targeting `hf-jobs-*` or other upstream GPU labels. Add a fork-owned GPU workflow only after runner access and cost policy are settled. Until then, report CUDA runtime coverage as unavailable in automatic CI rather than green or silently passed.

If provisioned later:

- Start with manual GPU validation and explicit job timeouts/concurrency limits.
- Run synthetic D256/D512 native-equivalence, transport, ownership, quota, rollback and cancellation tests plus small-model smoke tests.
- Check capability admission against the actual device. An unsupported device/profile must be identified explicitly.
- Require exact assertions and ordinary API error reporting.
- A T4 result does not qualify Blackwell-specific branches or RTX 5070 Ti performance.
- Consider making selected GPU checks mandatory only after their reliability and cost have been measured.

## 6. Disposition of every existing workflow

"Manual" means compile/check coverage on a standard hosted runner, with no automatic PR/push/scheduled trigger. Native execution that requires unavailable hardware is not preserved as an implied pass.

| Existing file | Proposed disposition |
|---|---|
| `ai-issues.yml` | Remove upstream AI issue automation and its `opencode` runner dependency |
| `build-3rd-party.yml` | Manual hosted LLGuidance integration validation |
| `build-and-test-snapdragon.yml` | Manual hosted compilation; distinguish unavailable device execution |
| `build-android.yml` | Manual hosted Android compilation |
| `build-apple.yml` | Manual hosted Apple compilation; no implied physical GPU qualification |
| `build-cache.yml` | Remove hourly upstream toolkit-cache preparation |
| `build-cann.yml` | Remove invalid workflow; CANN source remains untouched |
| `build-cmake-pkg.yml` | Move to standard hosted Ubuntu; retain package-consumer validation |
| `build-cpu.yml` | Adapt into required Linux/Windows CPU build/test component |
| `build-cross.yml` | Manual hosted LoongArch/SpacemiT RISC-V cross-compilation |
| `build-cuda-ubuntu.yml` | Required Linux CUDA build/link component; move hosted HIP/MUSA compilation to manual validation |
| `build-cuda-windows.yml` | Required Windows CUDA build/link component; move hosted HIP compilation to manual validation |
| `build-ibm.yml` | Remove jobs requiring specialized IBM runners |
| `build-msys.yml` | Manual hosted MSYS2 compilation; remove weekly schedule |
| `build-opencl.yml` | Manual hosted OpenCL compilation |
| `build-openvino.yml` | Manual hosted compilation; remove specialized runner dependence |
| `build-riscv.yml` | Remove unavailable native runner job; retain applicable cross-compilation through `build-cross.yml` |
| `build-sanitize.yml` | Adapt to hosted ASan/UBSan plus explicit manual TSan validation |
| `build-sycl.yml` | Manual hosted SYCL compilation |
| `build-virtgpu.yml` | Manual hosted VirtGPU compilation |
| `build-vulkan.yml` | Manual hosted Vulkan compilation |
| `build-wasm.yml` | Manual hosted WebAssembly validation |
| `build-webgpu.yml` | Manual hosted WebGPU compilation/checks |
| `check-vendor.yml` | Retain portable consistency check with corrected event handling |
| `ci-self-hosted-cpu.yml` | Remove; preserve applicable portable tests in hosted CPU jobs |
| `ci-self-hosted-cuda.yml` | Remove upstream runner jobs; use required hosted compilation and the separately gated GPU proposal |
| `ci-self-hosted-kleidiai.yml` | Remove specialized runtime jobs |
| `ci-self-hosted-metal.yml` | Remove specialized runtime jobs |
| `ci-self-hosted-openvino.yml` | Remove specialized runtime jobs |
| `ci-self-hosted-vulkan.yml` | Remove specialized runtime jobs |
| `ci-self-hosted-webgpu.yml` | Remove specialized runtime jobs |
| `close-issue.yml` | Remove automatic stale-issue closure |
| `code-style.yml` | Retain portable source check; correct v2 PR/push handling |
| `copilot-setup-steps.yml` | Remove inherited assistant setup from the CI surface |
| `docker.yml` | Remove upstream daily container/tag publication; retain Dockerfiles and local deployment tooling |
| `editorconfig.yml` | Retain portable source check; correct v2 PR/push handling |
| `gguf-publish.yml` | Remove upstream Python-package publication |
| `hip-quality-check.yml` | Manual hosted HIP warning/spill validation |
| `labeler.yml` | Retain metadata-only PR labeling as a convenience; not a required code check |
| `make-release.yml` | Remove upstream release/tagging procedure |
| `models-check.yml` | Remove specialized backend-runner matrix; preserve applicable small hosted fixtures in server/CPU checks |
| `pr-draft-label.yml` | Retain metadata-only draft-label behavior; not a required code check |
| `pre-tokenizer-hashes.yml` | Retain portable conversion consistency check |
| `python-check-requirements.yml` | Retain required requirements check |
| `python-lint.yml` | Retain required lint check after fixing violations |
| `python-type-check.yml` | Retain required type check and its pinned checker/dependencies |
| `release-publish.yml` | Remove; consolidate the fork's immutable SHA publisher in `release.yml` |
| `release.yml` | Publish the tested Linux/Windows CUDA outputs together by commit SHA |
| `server-sanitize.yml` | Remove specialized runner workflow; preserve applicable tests in hosted sanitizer coverage |
| `server-self-hosted.yml` | Remove specialized runner workflow |
| `server.yml` | Retain hosted API tests, reusing the tested CPU build outputs where compatible |
| `ui-build-self-hosted.yml` | Remove specialized runner workflow |
| `ui-build.yml` | Retain hosted reusable UI build; remove upstream publishing dependencies |
| `ui-publish.yml` | Remove upstream HF UI publication |
| `ui-self-hosted.yml` | Remove specialized runner workflow |
| `ui.yml` | Retain hosted checks/tests and replace nonblocking failures with qualified required checks |
| `update-ops-docs.yml` | Retain generated-document consistency check |
| `winget.yml` | Remove upstream-only Winget publication |

Also remove `bench.yml.disabled` as obsolete workflow scaffolding. Leave GitHub's platform-managed Dependency Graph feature enabled. Remove local composite actions only after checking that no retained or manual workflow references them.

New manual components are `build-cuda-compat.yml` for CUDA 12.9.1 legacy compilation and `build-gpu-alternatives.yml` for the former HIP/MUSA compile jobs. Neither publishes packages or implies GPU runtime qualification.

Metadata-only label workflows may use `pull_request_target`, but must not check out or execute PR code. Build/test workflows use read-only permissions and do not require repository secrets.

## 7. Commit-SHA packages and GitHub Release publication

For every successful v2 push, including a merged PR, publish both initial x64 CUDA packages together. In the names below, `<sha>` is the full 40-character source commit SHA.

| Package | Release asset |
|---|---|
| Linux CUDA | `llama-kv-stream-<sha>-linux-x64-cuda.tar.gz` |
| Windows CUDA | `llama-kv-stream-<sha>-windows-x64-cuda.zip` |

Use `sha-<sha>` as the immutable Git tag and GitHub Release identifier, pointing at that exact commit. A shortened SHA may be used in the human-readable release title, but manifests, asset names and canonical identity use the full SHA. Treat these automatic v2 builds as continuous-build prereleases, without a semantic-version or mutable `latest` tag.

Also attach `SHA256SUMS` and `build-info.json`. The manifest lists the two required CUDA package assets and their checksums. Preserve upstream component/API version metadata and include the fork SHA; package naming does not require changing shared-library ABI/SONAME versions.

Package `llama-server`, `llama-cli`, `llama-bench` and the required shared libraries/runtime dependencies. Check the actual install output and preserve required dependency licenses. Do not bundle model weights, private benchmark data, API keys or local production configuration.

Build/package requirements:

- A merge is already a push to v2; do not add a second publication trigger on PR closure.
- Run the required checks on the exact branch SHA before declaring its packages successful.
- Reuse the same build definitions between PR and branch builds. Preserve provenance when test and distributable configurations differ.
- Keep temporary CPU test artifacts separate from release artifacts. Never include CPU-only archives in a GitHub Release.
- Record full SHA, build run/attempt, OS/architecture, compiler, CUDA toolkit/runtime, architecture list, build options and package SHA-256 checksums in a manifest.
- Smoke-test the packaged binaries and shared-library resolution, including Windows DLL dependencies, rather than testing only the build tree.
- Give temporary Actions artifacts an explicit retention policy. Successful branch packages are published as GitHub Release assets, rather than depending solely on Actions artifact retention.
- Keep privileged publication conditional on a trusted v2 push and successful required checks. PR jobs cannot publish releases or write repository tags.
- Create a draft release, upload and validate both CUDA archives plus the checksum/manifest assets, then publish it. A failed or missing Windows or Linux package must prevent publication of an incomplete release.
- Make reruns idempotent. If a complete release already exists for the SHA, verify its commit/provenance and report it; do not overwrite published assets. Different rerun outputs may remain separately identified Actions artifacts, but must not silently replace the original SHA release.
- If a run fails while staging a draft release, preserve diagnostics and complete or clean up only that run's identified draft state. Never remove another commit's published assets.

Commit-SHA GitHub Release publication is required in CI4. Publish the already tested packages and manifest rather than rebuilding a differently configured release. Do not claim publication is finished while only Actions artifacts exist. A future stable channel or signing scheme can be added separately without changing these immutable commit identities.

## 8. Implementation stages and acceptance gates

Checked items describe implemented/local evidence. Hosted execution and live release acceptance remain unchecked until observed on GitHub. Local validation has passed workflow linting, policy/package tests, Python checks, an actual Linux CPU build and all 114 CTests (111 passed, three expected GPU/CLI skips), plus UI build/check/lint. The broader CPU run exposed an existing single-sequence scratch underquote for Minimax/Kimi models; measurement now follows the existing native reservation rule. The tokenizer driver now propagates every fixture failure and pins its external corpus to the upstream code baseline rather than silently consuming newer incompatible fixtures.

### CI0 - Capture the baseline

- [x] Preserve the workflow inventory and current required-check/ruleset configuration where accessible.
- [x] Record which observed failures are source defects, approval expiry, unavailable runners or invalid workflow definitions.
- [x] Confirm retained workflow dependencies, including reusable workflows and composite actions.
- [x] Identify representative PR change sets: C++ core, CUDA, Windows/CMake, Python, UI, generated docs and documentation-only.

Acceptance: each of the 58 YAML files has an explicit disposition and no required test is removed without a destination or documented hardware limit.

### CI1 - Establish hosted Linux/Windows compilation

- [x] Rework CPU/CUDA build components and move CMake-package validation to a standard hosted runner.
- [x] Remove upstream cache credential requirements from these jobs.
- [x] Pin toolchains and use bounded parallelism, diagnostics and timeouts.
- [x] Preserve the full server/shared-library link, D512 compilation and configured format coverage.
- [ ] Demonstrate successful Linux CPU, Windows CPU, Linux CUDA and Windows CUDA builds on actual GitHub runners.

Acceptance: all four builds succeed on the same revision; the Windows jobs exercise DLL linking; resource failures are resolved rather than skipped. GPU execution is not claimed.

### CI2 - Preserve and qualify portable tests

- [x] Retain Python typing, requirements and tracked host tests; fix Flake8 violations without blanket suppressions.
- [x] Retain source/generated-document checks and hosted UI/API tests.
- [x] Move portable memory/streaming and applicable server sanitizer tests onto hosted Linux.
- [x] Remove inherited `continue-on-error` from checks once their expected passing baseline is established.
- [ ] Prove valid inputs pass and representative malformed inputs still fail in the existing tests.

Acceptance: retained checks run without self-hosted hardware or upstream secrets, and deliberate compile/test/lint failures produce failing job results.

### CI3 - Connect the orchestrator and retire inherited automation

- [x] Add the single PR/v2-push orchestrator and stable aggregate result.
- [x] Verify relevant-path classification and ready-for-review behavior.
- [x] Remove duplicate automatic triggers and obsolete runner/publisher/maintenance definitions according to the inventory.
- [x] Keep optional hosted platform checks manual; identify actual backend/device coverage separately.
- [x] Remove orphaned reusable-action references and ensure there are no empty-job workflows.

Acceptance: a PR has a complete, non-duplicated check set; no job waits on an upstream runner; all retained YAML definitions validate. A v2 push triggers the intended checks.

### CI4 - Package and publish tested branch builds by SHA

- [ ] Produce the two CUDA packages and manifests for a successful v2 push.
- [ ] Verify checksums, executable startup and packaged runtime-library resolution.
- [ ] Verify failed/cancelled required checks prevent successful-build publication.
- [ ] Verify PRs cannot invoke the publication path, and a merge produces only one branch build.
- [ ] Publish a `sha-<full-sha>` GitHub prerelease only after Linux and Windows CUDA archives, checksums and the manifest are all present and validated.
- [ ] Verify two successive successful pushes retain separate immutable releases and reruns do not overwrite existing published assets.
- [ ] Verify a failed package upload leaves no publicly published partial release.

Acceptance: each successful v2 commit has a public, immutable commit-SHA GitHub Release containing both tested CUDA packages and their provenance/checksums. A platform failure cannot yield a partially published release.

### CI5 - Document and enable required checks

- [x] Document the required matrix, manual checks, artifact downloads and coverage limits.
- [x] Run workflow linting such as `actionlint`, local script checks and the relevant host tests.
- [x] Exercise representative change sets and a deliberately failing change in controlled validation.
- [ ] Review the resulting GitHub jobs and logs, including actual Windows/CUDA builds.
- [ ] Update branch protection/rulesets to the new stable required result after successful checks exist; remove obsolete required-check names.

Acceptance: PRs cannot merge through a failed or missing required check, ordinary pushes produce the expected packages, and no unintended tag/package/issue automation remains active.

### CI6 - Optional hosted GPU execution

- [ ] After a runner/budget decision, provision and qualify optional hosted CUDA runtime tests as described above.

This is a separate follow-up. It must not be silently enabled, treated as already qualified or allowed to leave the core PR matrix queued indefinitely. Commit-SHA binary publication is part of CI4 and does not wait for paid GPU provisioning.

## 9. External references

- [GitHub-hosted runner availability](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [GPU larger-runner specifications](https://docs.github.com/en/actions/reference/runners/larger-runners)
- [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
- [Workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)

Runner availability and billing were checked against these sources on 2026-11-05. Recheck them before provisioning paid infrastructure.
