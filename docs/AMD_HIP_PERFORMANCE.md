# RX 7900 XTX support and performance evidence

This opt-in Linux `gfx1100` backend supersedes the initial support in
[PR #94](https://github.com/Niko1221/Strata/pull/94). It retains HIP runtime and
wave32 integer-dot compatibility, native mmap layout validation, and MTP, then
adds HIP MMQ, optional calibrated dense hipBLASLt GEMM, native prefill batching,
and the host-memory / SSD paths used by the measured configuration. HIP blocking
expert uploads use one reusable pinned staging buffer, including prompt-cache
refills, to avoid repeated pageable-source registrations.

## Reproduce the configuration

Build instructions are in [AMD_HIP.md](AMD_HIP.md). Use the repository-pinned
llama.cpp dependency; do not silently substitute another revision. Build with
`STRATA_PREFILL_MMQ=ON` and use these runtime variables for the measured arm:

```sh
export STRATA_PREFILL_MMQ=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1100-hipblaslt-100100.txt"
export STRATA_PREFILL_RING=96
export STRATA_IO_THREADS=32
```

The supplied table is calibrated for gfx1100 and hipBLASLt version 100100.
It is not a universal ROCm tuning table. The runtime guards architecture,
library version and actual shape/stride/workspace requirements, falling back
when a table entry is unavailable or incompatible. HIP MMQ is opt-in at runtime. Default CUDA selection is preserved.

Measured engine configuration: Orca Flash Next IQ3_XXS, native pack plus matching
GGUF/tokenizer/template, `--mmap-experts --resident-cpu-experts`, fixed ranked
`--expert-profile`, `--expert-cache auto`, `--prefill 8192`, `--spec 4`,
`--spec-min-p 0.5`, matching MTP runtime, `--max-context 262144`, `--kv int8`,
`--kv-resident 32768`, `--pool-workers 15`, `--adapt-every 0`, `--pcie-frac 0`,
and `--vram-reserve-mib 1024`. Keep PLE on SSD.

`--resident-cpu-experts` copies the complement of the static GPU cache to ordinary
RAM. It requires mmap and a fixed expert profile. The slots the prompt path may
borrow keep their experts in RAM too as far as the budget below allows (the rest
use the mapped fallback while borrowed and for their refill), and adaptive swaps
(`--adapt-every`) exchange experts between VRAM and the RAM copy without reading
the file. The copy is pageable; `--resident-experts` (CUDA, chosen by setup)
page-locks it and uses 4 GiB of headroom. Allocation needs sufficient available RAM;
on Linux this option requires readable standard cgroup-v2 mounts. For each
finite cgroup ancestor, the guard credits only `inactive_file` after subtracting
`file_dirty` and `file_writeback`, capped by current usage; it remains bounded by
the ancestor limit and host `MemAvailable`, with 8 GiB headroom. This accounts
for reclaimable clean file cache but cannot reserve memory against concurrent
system or process allocations.
The POSIX PLE path issues direct reads through a configurable worker pool.

Use a dedicated idle server, restart it between arms, and capture its engine log:

```sh
python3 tools/hip/bench_prefill.py \
  --model MODEL_NAME --url http://127.0.0.1:8080 \
  --engine-log /path/to/engine.log --label candidate --output candidate.json
```

The script uses the same synthetic source prompts and request order as the
reported measurements: a small warmup, then 140/280/140/280 functions, with one
follow-up after each fresh prompt. It enforces zero reused tokens for the fresh
prompts and records prefill, decode, wall time and stop reason. It requires the
unbuffered log to contain exactly one completed timing record per request. The
128-token output cap is intentional for throughput measurement, not task success.
Never use cancelled-request timing lines as throughput evidence.

## Final revision benchmark

Measured on 2026-09-29, source revision `9568e78da663e6b224ab96d57baf450717b36e73`.
HIP executable SHA-256: `705f5925a788bc227bad72abfb5466032e0879d752fea9afc3eb32051b35182b`.
The control and candidate below were both measured anew in this session; no
historical benchmark values are mixed into the table. The control binary SHA-256
is `4522d4937ca4a41ca294d31b60df8c2bc4df5b7a6caf65eaad10c036e08491f7`;
its source was an existing local runtime snapshot rather than a clean upstream commit.
Both arms use an RX 7900 XTX 24 GiB / gfx1100, Ryzen 9 7950X3D,
64 GiB installed RAM, Fedora-family Linux, ROCm 7.1 and a 272 W GPU cap.
Other model services are stopped for measurement and restored afterward.
Sampling: temperature 0, top-k 1, top-p 1, min-p 0, seed 42, reasoning disabled.

| Fresh request, execution order | Control prefill t/s | Candidate prefill t/s | Candidate output t/s | Candidate request wall time |
| --- | ---: | ---: | ---: | ---: |
| 4,210 tokens, first use | 240.0 | 447.5 | 55.5 | 11.729 s |
| 8,830 tokens, first use at this size | 484.8 | 873.7 | 55.7 | 12.420 s |
| 4,210 tokens, warmed | 457.2 | 901.1 | 56.7 | 6.937 s |
| 8,830 tokens, warmed | 478.9 | 966.5 | 59.2 | 11.314 s |

Each request generated exactly 128 tokens and finished at the intentional length
cap. All four fresh prompts had zero reused KV tokens. Fresh-prompt prefill was
1.80–2.02x the freshly measured existing AMD runtime; fresh-request wall time was
40–46% lower. This is a comparison with our existing HIP runtime, not unmodified
upstream, which does not provide this backend. The candidate also completed four
cached follow-ups in 2.95–3.26 seconds. Those follow-ups are not apples-to-apples
prefill comparisons: cache reuse differs with generated text and checkpoint
selection. Full sanitized measurements are in
[the fresh-run JSON](benchmarks/2026-09-29-gfx1100.json).

Before this revision, an intermediate package missing the HIP upload staging
buffer stalled on the 8,830-token request and was rejected. Restoring staging
allowed the complete sequence above to finish. That failed attempt is not
included in the throughput table. No watchdog limit was raised to obtain these
results.

Zero KV reuse is not equivalent to a cold filesystem cache. First-use speed
and warmed speed are reported separately. Individual observations do not
establish confidence intervals or a general rate at every context length.

## Correctness and quality limits

Both HIP and CUDA executables built from the measured source revision. CUDA was
compile-checked, not performance-tested by this run. The selected HIP CTest suite
passed **29/29** with the tuning table enabled, including expert-upload readback,
asynchronous handoff, QSA, MMQ, Lt GEMM, KV streaming and PLE reading. The excluded
`ple_parity` requires an external fixture; `platform_memory_test` requires a
larger locked-memory limit than the test account provides. Separately, three
real IQ3_XXS PLE matrix graph replays passed, and the POSIX direct-file test passed
queued reads, short EOF, wake, close/drain and reopen checks.

Development checks passed eight MMQ numerical comparisons, four actual Lt GEMM
comparisons, and native QSA/indexer/embedding parity including tail states,
chunk continuation and image overrides. Real quantized MMQ relative L2 error
was approximately 0.0026 against raw-FP32/dequantized-weight reference, reflecting
Q8 activation arithmetic; this is not bitwise numerical equivalence.

End-to-end coding quality is not established by these numerical checks or capped
throughput requests. Validate completed tasks with independent runtime tests; this
contribution makes no broad answer-quality or agentic-reliability claim.

The changes do not claim better model reasoning, verified full-context behavior,
end-to-end vision validation, Windows HIP, other AMD architectures, or mixed
AMD/NVIDIA execution. Existing CUDA multi-GPU code remains present but is not
validation of HIP multi-GPU support.

## Attribution and rejected experiments

The measurements above were taken before this backend was rebased onto engine 0.1.24. The PR's own batched
embedding gather and QSA indexer append (adapted from
[PR #108](https://github.com/Niko1221/Strata/pull/108), commit
`acd487233c0bbe2217a6881c5bb43f8a283b0de5`) were dropped in the rebase: 0.1.24 already does both on every
backend, bit-exact (C-2, C-4). #108's optional GDN parallel/split path is not included. Existing upstream MMQ orchestration is retained and enabled for
HIP with AMD architecture identification and the correct backend compilation.

Static 16K chunks, expanded FP16 expert tuning, and a 192-slot ring did not offer
a consistent end-to-end win in the evaluated workload. The table includes only
the selected 30 dense-shape rows. Microkernel speedups alone were not sufficient
to select a configuration. No increase in GPU power cap was used.

## gfx1103 (RDNA3 iGPU, Qwen3.8-Flash-Next GSQ-RCO Q2_0)

First measured on the integrated gfx1103 APU (12 CU per KFD / 6 per the
HIP runtime, 22.86 GiB GTT, 45 GiB RAM, ROCm 10.0.0~pre4, engine 0.1.29,
branch `gfx1103`).  See `docs/GFX1103.md` for the machine record, the
calibrated table (`tools/hip/gfx1103-hipblaslt-100401.txt`, 9 rows), and the
known failure modes.  Configuration: `--mmap-experts --expert-cache 6000`
(~7.72 GiB, explicit — never `auto`, which over-allocates off the
HIP runtime's 22.86 GiB figure), `--prefill 512 --spec 4 --spec-min-p 0.5`,
`--kv int8`, `--pool-workers 8`, greedy, tuned table via
`STRATA_HIPBLASLT_TUNING`.

One-shot generate mode (the `--serve` path needs the MTP draft-head GGUF,
which was not in the model transfer):

| Phase | Measured | Note |
| --- | --- | --- |
| Prefill, cold single chunk (44–73 tok) | 8.3–27.3 tok/s | first run after engine start streams 3.3k–4.5k experts from the CPU pool |
| Prefill, warm batched (1,162–1,364 tok, 3 chunks) | **68.8 tok/s** | consistent across two separate passes |
| Decode (`--spec 4`, suffix drafts) | 5.4–13.5 tok/s | greedy, 160-token replies; draft acceptance 0.21–0.49 |
| Expert pool rows (CPU AVX2) | 12.9–42.7 GB/s | gate/up + down phases; 21–64 ms/round |

Compared with the RX 7900 XTX (final revision: ~59 t/s decode, ~900 t/s
prefill on the reference model), the iGPU is in the expected range for a
shared-memory RDNA3 part: prefill is dominated by expert streaming from the
CPU pool (the `--resident-cpu-experts` mode does not fit this box — the 23.9
GiB complement exceeds available RAM), and decode is bound by the same pool
round-trip plus the untuned bf16 dense GEMM (the tuned-table bf16 dense row
returns NaN on this stack and is excluded by the accuracy gate — §9.10 in
`docs/GFX1103.md`; f16 dense would get ~3.15× from the table).

Smoke checks (4/4 PASS, `tools/hip/gfx1103_smoke.sh`): arithmetic 17×23+5 →
396; executable one-line Python printing 1275 (`print(sum(range(1, 51)))`);
system-marker recall; and a buried-fact recall at ~1,170-token batched
prefill (724913).  `ple_parity` is excluded: its fixtures are part of the
unpublished upstream suite.

A/B (post-reboot, 2026-09-30 01:31–01:33, warm page cache):

| Arm | Prefill (1,162 tok / 44 tok) | Decode | Note |
| --- | --- | --- | --- |
| Tuned table (19:49 pass) | 68.8 tok/s | 13.5 tok/s | longfill, 160-token reply |
| Tuned table (fresh 01:26 pass) | 68.6 tok/s | 13.6 tok/s | longfill, 160-token reply |
| **Untuned (no table)** | 73.8 tok/s | 12.0 tok/s | longfill, 60-token reply |
| **Pageable staging** (`STRATA_PAGEABLE_STAGING=1`) | 26.1 tok/s (44 tok) | 8.6 tok/s | no H2D failure; correct answer |

Two conclusions: (1) **end-to-end prefill on this iGPU is expert-streaming
bound** — CPU expert streaming is ~56% of the 3-chunk prefill wall time and
varies ±2.7 s with page-cache temperature, so the tuned table's GEMM benefit
(2.3–4.0× on f16 in microbenchmarks) does not move the end-to-end number
here; the tuned-vs-untuned end-to-end delta (68.7 vs 73.8) is within the
streaming spread. (2) **pageable staging is a validated fallback** for the
APU's flaky pinned H2D (no failures, correct output); its speed delta is
confounded by cache temperature, so no benefit is claimed.  A clean
GEMM-phase A/B needs `bench_prefill.py` (serve path — blocked on the MTP
head GGUF, `docs/GFX1103.md` §6.2).
