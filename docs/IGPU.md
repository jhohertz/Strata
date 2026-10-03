# Integrated GPU (APU) expert path: alias, don't copy

Branch `igpu-rework`, off `gfx1103` (Engine 0.1.36 base). Machine: Ryzen 7 8700G
+ Radeon 780M (gfx1103), 45 GiB RAM, Ubuntu 26.04, ROCm 10.0.0~pre4. Model and
configuration as in `docs/GFX1103.md` (Q2_0, `--expert-cache 6000`,
`--mmap-experts`, tuned table).

## The problem, measured

On an APU the GPU and the CPU share one DRAM behind one memory controller.
The engine's expert pipeline was built for a discrete card with a separate
VRAM and a PCIe link, and every stage that exists to move data across that
link becomes a **copy inside the shared DRAM** here:

- 0.1.36 longfill (1,162 tok prefill): `experts streamed 29494 (0 by DMA,
  host 20606.8 ms)` - **20.6 s of the 31.3 s prefill (66 %) is host-side
  expert streaming**, 0 blobs moved by a DMA engine (the DMA arm needs a
  pinned arena; this box's memlock rlimit is 8 MiB, so the arena cannot
  register and every fill is a CPU-mediated copy).
- Each of the 29,494 fills is `cudaMemcpy(H2D)` of a 1,382,400 B blob
  (`ExpertCache::fill_slot`, expert_cache.cpp:351): ~40 GiB of RAM→RAM
  traffic through the same controller the GEMMs run on.
- The 6,000-slot cache itself is 7.72 GiB of **copies of data the GPU can
  already read**: with `--mmap-experts` the whole experts.bin is GTT-mapped,
  and `FileExpertSource::blob(i)` already returns the host pointer
  (`complement_host + offsets[i]`).
- 0.1.36 vs 0.1.29 on this APU: longfill prefill **68.8 → 37.1/37.7 tok/s
  (−46 %)** while decode improved 13.5 → 19.4/19.5. The 0.1.36 prompt path
  ("borrows 262 cache slots", device-side plan) reworked the streaming
  machinery without changing the dGPU premise.
- Both 0.1.36 intermittent fault sites (`iq_dequant_gu_f16`,
  `prefill copy_i32`) sit in this fill/copy machinery.

## What already proves the iGPU-native assumption

The engine already reads host pointers from kernels on this exact APU in
three places, all working:

1. The verify flag-wait kernels spin on **GTT-mapped host flags**
   (`wait_flag_ge_kernel`, verify_kernels.cu:426) - CPU→GPU visibility
   through the shared DRAM is a live, load-bearing path (it needed only the
   gfx1103 `__nanosleep` no-op, `hip_compat/intrinsics.hpp:136`).
2. `copy_i32_from_mapped_unless` (verify_kernels.cu:503) - a kernel copies
   **from a mapped host source** into a device destination.
3. `--mmap-experts` ("the A/B arm of R2.1") - expert blobs served from the
   file mapping.

On amdgpu an iGPU *is* the host memory: a kernel given a host pointer reads
DRAM directly. Nothing in the pipeline needs a copy; the copies are an
assumption, not a requirement.

## The design: an aliasing expert cache

**In one sentence:** on an integrated device, a cache slot is a 64-bit
pointer into the GTT-mapped expert file, not a 1.38 MiB copy of it.

Components (all additive; the dGPU path is byte-identical):

1. **`ExpertCache` alias mode.** `open_sized` already builds
   `off[i] = off[i-1] + align256(bytes)` into one device arena. Alias mode
   allocates a device array of 64-bit pointers instead of the arena, and
   `fill_slot(slot, host_blob, ...)` becomes an **8-byte**
   `cudaMemcpyAsync` (or a batched pointer upload per ring turn) instead of
   a 1.38 MiB H2D. No device arena (−7.72 GiB of RAM), no pinned arena, no
   memlock requirement. The residency table, the profile and the
   `(layer, expert) → slot` bookkeeping are unchanged.
2. **Slot resolution.** Exactly two device-side consumers derive a pointer
   from `(cache_base, slot_off)`: `resident_plan_kernel`
   (verify_kernels.cu:433) and the grouped hit kernel
   (`moe_hit_grouped_s2_dev`, session.cpp:869). Each gains a pointer-table
   variant that reads `d_slot_ptr[slot]` instead of
   `cache_base + slot_off[slot]`; selected at runtime by a mode flag in the
   drive struct (no preprocessor, CUDA build untouched). Host-side
   bookkeeping that uses `slot_offsets()` for **sizes** (the borrow
   arithmetic in generate.cpp) is size-based and stays as-is - the alias
   mode reports the same per-slot byte counts.
3. **Detection and override.** `cudaDeviceGetAttribute(cudaDevAttrIntegrated)`
   (attribute 18, "device is integrated with host memory") selects alias
   mode automatically; `STRATA_IGPU_ALIAS=0|1` overrides (default auto).
   `cudaDevAttrIntegrated` is the same attribute on the HIP runtime, so one
   check covers both builds. A dGPU (RX 7900 XTX, Strix Halo *as dGPU
   partitioning* aside) never takes the new path.
4. **Ring/flag waits (phase 2).** The ring exists so the GPU waits until a
   slot's *copy* is done. In alias mode the data is already in DRAM; the
   "fill" is an 8-byte pointer write, so the per-slot flag spin
   (the kernel class that needed the `__nanosleep` no-op) can collapse to a
   single batch marker - or go away for experts that were already resident.
   This is where the intermittent faults (both sites in the fill machinery)
   are expected to die.

Memory-safety notes:

- **Coherence** is the same property the flag spins rely on; no new
  assumption.
- **Eviction**: a plan may reference file pages the OS later reclaims
  (45 GiB RAM, 31.6 GiB file). The GPU's user-page fault fetches the page
  from the page cache or disk - a latency spike, not a correctness break.
  The hot set (6,000 slots ≈ 8 GiB of the file) is exactly what the OS
  wants resident, and aliasing frees 7.72 GiB of private copies back to it.
  Measured in P0/P1, not assumed.
- **TLB**: the file mapping spans many PTEs; large pages (THP on the
  mapping) reduce GPU TLB pressure - measured in the P0 micro-benchmark.
- **Lifetime**: the mapping outlives the session (the file stays open);
  plans reference it only for the run.

## Experiment ladder (one step per fresh APU, gate first, no mid-run kills)

Machine rules from `docs/GFX1103.md` §9.11–9.14 apply: any fault stops the
session until a reboot.

- **P0 - no code.** (a) Baseline matrix on the 0.1.36 build:
  `--expert-cache 6000` (37.1/37.7 already) vs `--expert-cache 0` (no VRAM
  tier; CPU pool only - isolates the copy cost from the resident-compute
  benefit). (b) `STRATA_OLD_IQ_MMVQ=1` on longfill + arithmetic - the
  kernel-vs-streaming split for the −46 % and the faults. (c) Micro:
  dequant+GEMV on a blob read directly from the mmap (GTT) vs from a filled
  slot, with/without THP - the number that decides whether aliasing pays.
- **P1 - aliasing cache** (items 1–3). Success: longfill prefill ≥ 68.8
  tok/s (the 0.1.29 level) with 12/12 known-answer runs over 3 passes, no
  faults in ≥ 10 runs; `experts streamed … host` time collapses toward 0
  and 7.72 GiB leaves the RAM footprint.
- **P2 - ring/flag collapse** (item 4). Success: same numbers with the
  per-slot flag waits gone; the fault rate, if any remains, is re-measured
  against the P1 baseline.
- **P3 - prompt-path borrowing in alias mode** (the 0.1.36 "borrows 262
  slots" interaction: borrowed slots alias too).
- **P4 (stretch) - shrink the CPU pool.** With the GPU reading everything
  from DRAM, the AVX2 pool workers compete for the same controller; reduce
  `--pool-workers` and re-measure to find the split where CPU and GPU stop
  fighting.

## Upstream shape

Additive and gated: one device attribute check, one cache mode, two kernel
variants, one env override. Discrete cards keep the copy path byte-for-byte.
The pitch is a feature, not a workaround: every iGPU user (gfx1103 Phoenix,
the gfx1150–gfx1153 Strix Point family AMD already ships wheels for,
gfx1201 Strix Halo) runs today with the memlock workaround, the 7.7 GiB of
pointless copies and the copy-machinery fault surface; aliasing removes all
three. `--mmap-experts` stops being mandatory on APUs.

## P0 results (first boot, 2026-10-02 19:38 boot)

Gate green at 19:41. Two runs, one fault:

| run | config | prefill | decode | streaming | result |
| --- | --- | --- | --- | --- | --- |
| E1a 19:42 | `--expert-cache 6000`, cold page cache | 34.74 tok/s (33.45 s) | 18.75 tok/s | 29,494 streamed, 0 by DMA, host 22.9 s (68 % of prefill), resident 10,244 | PASS (exit 0) |
| E1b 19:45 | `--expert-cache 0` = **auto** (generate.cpp:2269 maps 0 to -1) | - | - | - | **FAIL: unspecified launch failure, exit 1** |

E1a confirms the 0.1.36 baseline on a fresh APU (37.1/37.7 warm → 34.7 cold) and
that the 16:44 degraded state was gone at 19:42. E1b is the first data point for
a second hypothesis: it sized to **24,576 slots (33.9 GiB of the 39.6 GiB free)**
- `auto` treats the APU's 48 GiB GTT like a dGPU's VRAM and fills nearly all of
it - and the fault landed in that configuration. Third fault overall (15:38
`iq_dequant_gu_f16`, 16:44 `prefill copy_i32`, 19:45 site not captured, exit
before the site line). Memory pressure from an oversized GTT arena may be a
trigger or a co-factor; indistinguishable from the baseline intermittency without
more runs, which the machine rules forbid after a fault.

**Revised P0 order (next fresh boot, gate first, 60 s+ between runs):**

1. E1c `STRATA_OLD_IQ_MMVQ=1` longfill + arithmetic (the kernel-vs-streaming
   split for the -46 % and the faults) - cold, as the first runs.
2. E1b' the no-cache pole done properly: a **small** fixed cache (`--expert-cache
   500`, ~0.69 GiB) instead of `0`/auto, so the pole is "almost no VRAM tier"
   without the 33.9 GiB arena.
3. E1a warm re-run (same 6000 config) to pair with the 34.7 cold number.
4. Micro-benchmark (GTT-read vs filled-slot dequant+GEMV, with/without THP) -
   written as a small test on this branch, built before the run window.

`--expert-cache 0` must be documented (and probably fixed upstream) as "auto"
for APU users: it silently sizes a 33.9 GiB arena on a box where `auto` is
known to over-allocate (GFX1103.md already says "never auto").

## P0 results (second boot, 2026-10-02 20:0x boot) - complete, all runs clean

Gate green before each phase. Five engine runs and two micro runs, no faults.

| run | config | prefill | decode | host streaming | streamed / resident |
| --- | --- | --- | --- | --- | --- |
| E1c arith, `STRATA_OLD_IQ_MMVQ=1`, cold | 6000 slots | 9.50 | 9.28 | 3.2 s / 4.6 s | 3,340 / 1,770 |
| E1c longfill, `STRATA_OLD_IQ_MMVQ=1`, cold | 6000 slots | 36.81 | 19.22 | 20.7 s / 31.6 s | 29,494 / 10,244 |
| E1a' longfill, default kernels, warm | 6000 slots | 37.95 | 19.52 | 20.1 s / 30.6 s | 29,494 / 10,244 |
| E1b' longfill, warm | **500 slots** | 37.78 | 19.00 | 20.9 s / 30.8 s | **39,239** / 499 |
| micro (1 GiB, 512 blocks) | - | - | - | H2D copy 35.8 GB/s | **GTT read 82.6 GB/s = VRAM read 82.6 GB/s**, checksums OK |

Conclusions:

1. **The kernels are exonerated.** The `STRATA_OLD_IQ_MMVQ=1` arm (0.1.29-era
   single-token kernels) shows the same ~37 tok/s warm / ~35 cold as the 0.1.36
   multi-token kernels, with identical host-streaming times. The -46 %
   regression vs 0.1.29 (68.8 warm) is in the shared streaming/caching
   machinery, not the kernels - and it is now reproducible at 37.1/37.7/37.95
   across two boots.
2. **Cache size is not a performance variable.** 500 slots (0.69 GiB) is
   throughput-identical to 6,000 slots (7.72 GiB) while streaming 33 % more
   fills (39,239 vs 29,494) - 54.1 GiB of blob traffic in the same 20.9 s as
   40.7 GiB. Copy bandwidth is not the constraint; the **per-expert host work
   is** (dequant + GEMV + issue, serialized through the pool workers). The
   ~20 s host time is the floor of the current architecture on this APU.
3. **The 0.1.36 prefill is ~20 s of pool/host expert work + ~11 s of GPU +
   dense work.** 0.1.29's 68.8 tok/s corresponds to a much smaller host share
   (its streaming was 1.26 s per the 0.1.29 A/B row); the 0.1.36 device-side
   plan rework moved expert work onto the host path. The fault sites (both in
   the fill machinery) sit in the same path.
4. **GTT reads are free.** The kernel reads an anonymous host buffer at
   exactly the VRAM-read rate (82.6 GB/s, flat over 9 iterations, checksums
   exact), while the H2D copy that exists to put data "where the GPU can read
   it" costs 35.8 GB/s for data already in that memory. MADV_HUGEPAGE was
   refused and made no difference - 4 K pages over 1 GiB are fine.
5. **Faults remain 3 total across two boots** (two identified sites in the
   fill machinery, one in the 24,576-slot auto config); 9 subsequent engine
   runs clean.

**Revised P1 design (replaces "slot = 8-byte pointer into a 6000-slot table"):**
the pointer table covers **all 24,576 experts** (24,576 x 8 B = 196 KiB on
device), not a 6,000-slot subset. With every expert simultaneously
"resident", the GPU computes **100 % of expert rows** and the CPU pool's
expert work - the measured ~20 s - retires to zero (or the pool becomes a
pure draft worker). The copy is only a symptom; the pool compute is the
disease. The bandwidth math: the longfill's 9,296 expert rows read 12.8 GiB
of blobs, ~0.16 s at the measured 82.6 GB/s; dense weights are read once.
Expect prefill to land in the 100+ tok/s region (measured in P1, not
promised), i.e. a 3-5x improvement over the current 37.8 and past 0.1.29's
68.8 - on the iGPU path only, the dGPU copy path unchanged.

P0 verdict: **the aliasing path is validated at the hardware level and the
target is now quantified**. Next: P1 implementation on this branch
(ExpertCache alias mode + full pointer table + the two resolution-kernel
variants + `cudaDevAttrIntegrated` detection + `STRATA_IGPU_ALIAS` override),
then the 3-pass smoke gate.

## P1: the aliasing expert cache (first pass)

Implemented on this branch (additive, gated: `cudaDevAttrIntegrated` detection +
`STRATA_IGPU_ALIAS=0|1` override; the dGPU copy path is unchanged code):

- `ExpertCache::open_aliased(layers, experts)`: one slot per (layer, expert),
  `blob_ = 8`, the device side is a 24,576 x 8 B = 192 KiB pointer table
  (`d_ptrs_`); `set_aliased_pointers()` uploads it with a D2H identity check.
  `slot_offsets()` returns the host table, `device_slot()` returns the host
  pointer, `fill_slot*()` are 8-byte pointer writes, `valid()`/`close()`
  alias-aware.
- Every existing "(cache_base, slot_off)" resolution point works without
  change, because `nullptr + ptr_table[slot]` is the host pointer itself: the
  host plan (the `GpuPlanSink` ternary), the device plan
  (`resident_plan_kernel`), the verify CUDA graph, and the prompt path's
  `device_slot()` lookups.
- The five per-entry decode kernels (`gu`, `down`, `gu_pair`, `down_pair`,
  `cpu_order_projection`) get a trailing `d_blob` parameter:
  `d_blob ? d_blob[slot] : blob_base + slot*blob_bytes`, plumbed through the
  four `moe_hit_grouped_s2*` wrappers, `ExpertDispatch`, and `TokenHits`.
  Default null = the copy path, byte-identical.
- `generate.cpp`: on an integrated device (or `STRATA_IGPU_ALIAS=1`) with
  stable non-transient blobs (the `--mmap-experts` file mapping), admit all
  24,576 experts, publish their file pointers, and skip the arena open, the
  profile prefill, and the prompt-path cache borrow (the prompt path then
  allocates its own buffers - the proven "no cache" dGPU path).

**First run (2026-10-02, boot 2, after the clean P0 session): HANG.**
`STRATA_IGPU_ALIAS=1`, arithmetic prompt (43 tokens), `timeout 600`: the
engine logged through "session is up", "token graph hit path: 24576 resident
experts", "prompt path allocates its own buffers", "prefill gemm: hipBLASLt
tuning enabled" (one dense GEMM completed), then produced **no output for 10
minutes until the timeout killed it**. No fault printed, no verify watchdog
(20 s) fired - so the hang is outside the verify window.

Desk analysis: every alias-specific kernel is a plain read (the micro proved
GTT reads; `gather_native` already degrades to byte-copies for 8-aligned
pointers, so alignment cannot fault); the verify window self-times-out at 20 s
per step; nothing in the prompt path spins. The one thing that is genuinely
new: **prompt-path kernels reading the 31.6 GiB GTT-mapped `experts.bin`
directly** - the same memory the 0.1.36 copy-path faults (`iq_dequant_gu_f16`
/ `prefill copy_i32`) sit in. Working theory: the APU's KFD/VM layer wedges
on sustained kernel reads of the GTT-mapped file (the simple 500 MiB micro
read passes; the MMQ-pattern access of 24,576 blobs does not).

After the timeout kill the APU must be rebooted (GPU work may have been
in flight). Next: `tools/hip/p1_alias_run.sh` - the same run with
`STRATA_TRACE=1` plus hang forensics (gpu_busy_percent, per-thread wchan,
gdb backtraces of every thread) captured automatically ~45 s into the
silence, then a kill. The backtraces separate "CPU parked in
cudaStreamSynchronize" (a GPU-side wedge) from "CPU parked in a condition
variable" (a host-thread deadlock in the pool/stager/PLE machinery).
