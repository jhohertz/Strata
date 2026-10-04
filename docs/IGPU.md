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

**Second run (2026-10-02 23:03, boot 3, gate green, `STRATA_TRACE=1`,
forensic monitor): the hang reproduced exactly, and the monitor nailed the
shape.** The trace shows `prompt chunk 0 of 44`, then silence; the iGPU
sat at **99 % busy for 45+ s** - a kernel on the GPU that never finishes,
not a host-side condition-variable deadlock (that would show ~0 % busy).
The first forensic pass attached gdb to the wrong process (the `timeout`
wrapper), so no backtraces yet; the monitor now targets the real engine
(`pgrep -x strata`), captures per-thread wchan/state, and lets the 900 s
timeout do the single kill.

Experiment ladder for the next two boots:
- **Boot 3, run A**: alias arithmetic, forensics v2. Expected: the main
  thread parked in a stream sync (GPU-side wedge) plus the last kernel
  before the hang.
- **Boot 4, run B**: the same run, but with the page cache **pre-warmed**
  first (a sequential read of the 31.6 GiB `experts.bin`). On the copy
  path the CPU faults the file pages in before the GPU reads; on the alias
  path the GPU's kernels fault 31 GiB of cold NVMe pages through the KFD
  GTT path for the first time in this codebase. If run B passes, the wedge
  is in the GPU-side page-fault path, and the fix is to pre-populate the
  pages on the host at alias open (one ~10 s sequential pass, or
  `MADV_POPULATE_READ` on the mapping) - the alias design stands. If run B
  hangs identically, it is sustained GTT kernel reads that wedge, and the
  strategy changes (bounce-buffer the prompt path, keep decode aliasing).

**Run 3 forensics (2026-10-02 23:13-23:27, boot 4, the same hang, twice
sampled - gdb is blocked by yama ptrace_scope without root, so wchan +
gpu_busy_percent did the work):**

- Phase 1 (23:12-23:22): main thread state **R** (running, not parked in a
  sync), iGPU **99 % busy**, pool/stager threads idle in futex.
- Phase 2 (23:22-23:27+): main thread still R (11 min of CPU time), iGPU
  **0 % busy** - the KFD queue died while the host keeps polling it.
- No "FileExpertSource: allocating ... GiB cache complement" line in the
  log: the complement RAM copy is **not** built in this configuration, so
  `blob(l,e)` returns the **mmap'd file pointer itself** - the kernel reads
  the 31.6 GiB `experts.bin` mapping **cold from NVMe, page by page, for
  the first time through the KFD/GTT path**.

That is the mechanism: 31.6 GiB = 8 million 4K pages. The copy path never
puts the GPU on that path (the pool's CPU reads fault the pages in at SSD
speed first; the kernel then reads warm RAM). A KFD fault at roughly
1 ms/page is hours, not seconds - phase 1 was that storm (or its wedge),
and phase 2 is the queue that stopped producing. Every bounded spin in the
prompt path was ruled out by reading: `Stager::wait` and `wait_issued`
only run for non-resident entries (all 24,576 are resident, the log says
so), the verify window self-times-out at 20 s, and the D-5 issuer thread
exits immediately on an empty seq.

**The fix, if the cold/warm micro confirms it, is one call:** pre-populate
the file's pages on the host at alias open (a sequential read pass, or
`madvise(MADV_POPULATE_READ)` on the mapping) - ~10 s once per boot.
`igpu_gtt_micro` gained a `--file` arm for exactly this: pass 1 reads the
whole mapped file cold (timed), an 8-thread CPU pass warms the cache and
produces the reference checksum, pass 2 reads warm (timed).  Boot 5
protocol: micro cold first (the cheap discriminator - no engine involved),
then the engine run.

**Run 4 (2026-10-02 23:29, boot 5): the cold micro faulted - cleanly.**
`igpu_gtt_micro --file packs/qwen38-flash-next-q2_0/experts.bin` (the first
GPU read of the 31.6 GiB file mapping on a fresh boot):
`hipStreamSynchronize: an illegal memory access was encountered`.  No
hang, no kill - the process self-terminated (exit 2), which is the mild
end of the failure spectrum; the APU is still treated as possibly
degraded and gets a reboot before more GPU work.

This replaces "wedged queue" with the actual first fault: **the iGPU's
KFD/GTT path cannot demand-fault a 31.6 GiB file mapping** (8 million 4K
pages, cold).  In the engine the same fault lands in the middle of a long
kernel stream, which is why it surfaced as the 99 % busy storm / dead
queue instead of a clean error.  This also gives the 0.1.36 copy-path
faults (`iq_dequant_gu_f16`, `prefill copy_i32`, 2 of 9 runs, moving
site) a candidate root cause: any first-touch GTT fault the KFD path
cannot serve is a launch failure, and which kernel it lands in depends on
where the physical pages happened to land.

Boot 6 protocol (one GPU budget, no kills planned):
1. gate.
2. **CPU-only** warm: `dd if=experts.bin of=/dev/null bs=8M` (no GPU).
3. micro `--file` (now warm): if it passes, kernel reads of
   RAM-backed file pages are fine and the fault was purely the cold
   demand path -> the engine fix is a host pre-population at alias open
   (~10 s sequential pass), the alias design stands.
4. engine alias arithmetic (file already warm from step 2) - the full
   P1 verdict.
If step 3 faults even warm, the GTT does not cover that address range at
all, and the alias target moves to a full 31.6 GiB anonymous-RAM
complement (built once by the CPU at startup, SSD speed) instead of the
file mapping - still zero H2D copies, still no pool work.

**Run 5 (2026-10-02 23:33, boot 6): warm, it faults too.**
Gate green; `dd` read the 31.6 GiB in 6.7 s (CPU-only, page cache warm);
the micro's full-file kernel read: `illegal memory access` again.  So the
cold/warm question is settled - **it is the mapping, not the page state.**
The data now: 1 GiB anonymous VMA (high user VA, ~0x7622...) = 82.6 GB/s
clean (P0); 260 MiB file VMA (the embedding table, same VA region) = clean
(every run); one 31.64 GiB file VMA at 0x762298400000-0x762a81400000 =
illegal access.  The remaining variable is the per-VMA size (or size
times type).

Boot 7 protocol - binary-search the per-VMA limit with file **slices**
(the micro gained `--slice <file> <off_gib> <size_gib>`: one VMA over
[off, off+size), one kernel read, checksummed, and `--anon <gib>` for the
anonymous-complement question).  Each PASS costs seconds; the first FAULT
costs the boot and brackets the limit from below.  Order:
  1. --slice 30.6 1     (1 GiB at the tail of the file)
  2. --slice 0 1        (1 GiB at the head)
  3. --slice 23.6 8     (8 GiB)
  4. --slice 15.8 16    (16 GiB)
  5. --slice 0 31       (31 GiB, near-full)
  6. --anon 32          (the fallback design's viability)
All-pass outcome: one contiguous 31.6 GiB VMA is the problem -> chunk the
alias open into several mmaps (e.g. 4 GiB each, pointers table updated
per chunk) - a small, clean change.  A fault at step N pins the limit and
the same chunking fix applies with the measured size.

**Runs 6-7 (2026-10-02 23:39-23:40, boot 7): even a 1 GiB file slice
faults - at both offsets.**  `--slice 30.6 1` (map 0x72c599200000) and
`--slice 0 1` (map 0x7f4301600000): both `illegal memory access`.  And the
260 MiB "mapped host memory" token embedding is **not a file VMA at all**
- `NativeEmbed` is `cudaHostAlloc` (pinned anonymous) or VRAM.  So: **no
kernel on this APU has ever read a file-backed VMA, and the ones that
have all fault.**  The KFD/GTT path on gfx1103 + ROCm 10.0.0~pre4 maps
anonymous (and pinned) host memory, not file mappings.  (This also means
the P0 82.6 GB/s number is the correct expectation for the alias reads:
they will run over anonymous RAM, exactly like the P0 arm.)

**The fix, implemented:** the alias table points at the **full RAM
complement** instead of the file.  The complement (`pin_cache_complement`)
builds one anonymous arena holding the experts the GPU cache does not
hold - so the alias open now pins it **before admitting any expert to the
alias cache** (an empty residency makes the complement cover all 24,576
experts; ~31.6 GiB, filled by the CPU at SSD speed, once, ~10-15 s), and
`blob()` then prefers the complement over the file mapping from then on.
The headroom is capped at 2 GiB for the alias run (the default 8 GiB -
sized for the ordinary modes - would reject the 31.6 GiB complement on a
45 GiB box).  Everything downstream (the pointer table, the kernels, the
plans) is unchanged: they already read "the blob pointer", which is now
anonymous RAM.

Boot 8 protocol:
1. gate.
2. `--anon 32`: does the KFD path map a 32 GiB **anonymous** VMA (the
   complement's shape)?  A fault means the arena must be chunked.
3. engine alias arithmetic (no dd needed: the complement fills from NVMe
   on the CPU at startup; the GPU never touches the file).

**Run 8 (2026-10-03 00:17, boot 8): a 32 GiB ANONYMOUS VMA faults too.**
`--anon 32` (map 0x77baf25ff010): `illegal memory access`.  So the
discriminator is not file-vs-anonymous - it is **the size of one VMA**
(the P0 1 GiB anonymous read passed at 82.6 GB/s; 32 GiB anonymous and
31.6 GiB file and 1 GiB file slices all fault).  The complement, as
designed, is one 31.6 GiB VMA: it cannot be mapped by the KFD path.

**Chunking design (to be validated in the micro before touching the
engine):** a `PROT_NONE` VA reservation of the full span, with the
committed chunks placed inside it separated by 64 KiB `PROT_NONE` gaps.
The gaps keep the committed chunks **separate VMAs** (the kernel merges
adjacent same-flag VMAs, which would defeat the purpose), and the KFD
path appears to map per VMA.  With per-chunk bases the dense
`complement_offsets_` stay valid once the gaps are folded into them, so
`blob()` / `has_resident` / the fill / the pointer table are unchanged.
`igpu_gtt_micro` gained `--fixed <va> <gib>` (anonymous at a chosen VA:
VA-window vs size) and `--chunks <n> <gib>` (the exact complement shape:
n committed chunks in a PROT_NONE reservation, every committed byte
read by the kernel).

Boot 9 protocol (one boot, ascending, stop at the first fault):
  1. gate
  2. size sweep: --anon 1, 2, 4, 8, 16   (brackets the per-VMA cap)
  3. chunk test at the sweep's ceiling: --chunks 4 <cap> (or 8 <cap/2>)
     - pass: chunking is validated; implement it in pin_cache_complement
       with the measured chunk size and run the engine.
     - fault: the KFD unit is not the VMA (a physical-range model); the
       complement must then be split into separate allocations (no
       shared reservation) - the pointer table already stores absolute
       pointers, so only the arena plumbing changes.

**Runs 9-13 (2026-10-03 00:4x-01:0x, boot of 00:42): the KFD model is
found, and the engine's hang gets a second data point.**

The "per-VMA size cap" was an artifact of the test shape.  The micro's
arms, in order:
- `--anon 1` with a 64 MiB H2D ping first: FAULT.
- `--anon 1` with a FULL 1 GiB H2D ping first: PASS (ping 17 GB/s, read
  82.6 GB/s).
- `--anon 32` with a full 32 GiB H2D ping first: PASS (ping 36.3 GB/s in
  946 ms, read 74.1 GB/s).
- `--reg 8` / `--reg 32`: hipHostRegister alone (no DMA at all) makes the
  whole region kernel-readable: PASS at 54.4 / 54.8 GB/s.

**The model (measured, gfx1103 + ROCm 10.0.0~pre4):** a host VMA is
kernel-readable only after the driver has been told about it -
hipHostRegister, or a DMA pass over the range (registration is per
touched range; a partial ping left the rest faulting).  Size is not a
limit (32 GiB passes); file-vs-anonymous is not a limit either (the file
slices faulted for lack of registration, not because they were files).

**The engine, with the complement registered at alias open, still hung**
(same shape: clean startup, "the 31.64 GiB expert complement is
registered", then 10 min of silence until the timeout).  Two variables
separate the passing micro from the engine:
1. **flags**: the micro used `hipHostRegisterDefault`; the engine's first
   attempt used `Mapped|Portable`.  (The engine is now switched to
   Default; the micro gained `--reg-mapped` to A/B the flags.)
2. **ordering**: the micro registered before any GPU traffic; the engine
   registers after ~1.5 GiB of weight loads (gigabytes of prior DMA).
   (The micro gained `--reg-after-dma`: a 1 MiB DMA first, then register.)

The engine now registers with `hipHostRegisterDefault` and falls back to
a full H2D touch pass if the driver refuses (`FileExpertSource::
register_complement_for_gpu`).  `tools/hip/p1_boot_run.sh` runs the boot
protocol: gate, `--reg-mapped 8`, `--reg-after-dma 8`, then the engine
alias arithmetic (and, on a pass, python on the same boot - clean runs do
not degrade the APU, so a green boot is a whole-smoke opportunity).

**Run 14 (2026-10-03 12:38, boot of 12:23): registration is fully
exonerated.**  `--reg-mapped 8` (the exact `Mapped|Portable` flags the
engine used): PASS at 54.4 GB/s.  `--reg-after-dma 8` (a 1 MiB DMA first,
register second - the engine's ordering): PASS.  Yet the engine alias run
timed out again (same silence).  The KFD registration story is closed:
the complement is readable, and the hang lives in the engine's code path.

New suspect: the alias forces the prompt path's **borrow=nullptr
("allocates its own buffers") branch** - the one branch the APU smoke has
never taken (every run so far borrowed from a 6000-slot arena; the
borrow math is `k+128 <= slots`, so `--expert-cache 50` on the plain COPY
path reaches the same branch with zero alias machinery).
`tools/hip/p1_boot_run2.sh` A/Bs it: copy path + `--expert-cache 50` +
`STRATA_IGPU_ALIAS=0` + trace first (the cheap reproducer if it is the
culprit), the alias run second.

**Run 15 (2026-10-03 13:18, boot of 12:55): the A/B is decisive.**
Copy path + `--expert-cache 50` (borrow=nullptr, the "own buffers" branch,
`STRATA_IGPU_ALIAS=0`): **PASS** (prefill 6.64, decode 8.21 tok/s - slow
on 50 slots, as expected).  The own-buffers branch is innocent.  The alias
run timed out again, same spot ("prompt chunk 0 of 44").  The only
material difference left: the alias prompt computes every expert through
`gather_native` reading the **host complement pointers** (the copy path's
resident computes read device-arena pointers).

Next boot (tools/hip/p1_boot_run3.sh):
1. micro `--scatter 32`: the engine's exact gather shape on a registered
   region - 24,576 CTAs, each reading its own 1.38 MiB blob at a scattered
   base (the complement's layout), one kernel.  A fault/hang here means
   the KFD path cannot serve that pattern (the pattern-free full reads
   pass) and the alias design bounces the prompt path (decode-only
   aliasing).
2. the engine, now with per-layer + chunk-done + token-loop traces
   (STRATA_TRACE) - the last line before the silence names the hang
   layer.

**Boot protocol 4 (tools/hip/p1_boot_run4.sh, the localizing boot):**
gate; micro `--scatter 32` (fault here = the pattern is unserveable,
decode-only aliasing is the design); then the traced engine **under gdb**
(gdb is the parent, so ptrace_scope=1 allows the attach): if the engine
hangs, gdb interrupts it at ~3 min and dumps `thread apply all bt 12` +
`info threads`; if it completes, no interrupt is sent and the boot is
preserved for a chained smoke.  One boot yields: the pattern answer, the
last trace line (which phase), and the host backtrace (which call).

**Run 16 (2026-10-03 15:41, boot of 15:18): the hang is LOCALIZED.**
- micro `--scatter 32`: **PASS at 54.5 GB/s** - 24,855 CTAs each reading
  their own 1.38 MiB blob on a registered 32 GiB, the engine's exact
  gather shape.  The KFD path serves the gather pattern fine; hypothesis
  (a) is dead.
- traced engine: clean startup, complement copied (31.64 GiB) and
  registered, "prompt chunk 0 of 44", "prompt layer 0", "prompt layer 1"
  - then silence.  **The hang is inside layer 1's body.**  Layer 1 is the
  PLE layer (the only layer with the PLE block: conv + table reads on the
  28.8 GiB file-mapped GGUF).

  What is new in alias mode at that point: 31.64 GiB of *pageable*
  complement plus the 28.8 GiB PLE file mapping plus the 31.64 GiB expert
  file mapping, with **8.14 GiB free** (traced).  The copy-50 run passed
  the same layer with a 69 MiB arena and ~40 GiB free.  Prime suspect:
  page fault / direct reclaim pressure around the PLE file-page path
  while 31.64 GiB of anonymous pages sit in the reclaim path.

  (The gdb run in this boot never attached: `-batch` gdb does not read
  the interrupt off stdin while `run` blocks.  Fixed: interactive gdb,
  the feeder SIGSTOPs the engine - not SIGINT, which could exit it -
  then `thread apply all bt 12` + `info threads` + `kill`, with correct
  timeout->gdb->engine PID walking and a 900 s process-group backstop.)

The engine now also traces inside the layer body: "PLE block done",
"attn+mlp done, experts begin", "moe done" (per layer, STRATA_TRACE) -
so the next run names the exact stage of layer 1, and the gdb
backtrace says whether the host is stuck in a page fault / reclaim or
the GPU is stuck in a kernel.

**Run 17 (2026-10-03 17:01, boot of 16:31): the root cause, by
backtrace.**
- scatter micro: PASS again (54.4 GB/s).
- traced engine: layer 0 completed fully ("PLE block done" x2 - the
  two halves - "attn+mlp done, experts begin", "moe done" - the alias
  gather + MMQ over the host complement WORK), then "prompt layer 1"
  and silence: the hang is inside the layer-1 PLE block.
- the gdb backtrace (SIGSTOP + `thread apply all bt`): the main thread
  is in `hsa_executable_freeze -> AmdHsaCodeLoader::FreezeExecutable ->
  RegionMemory::Freeze -> BlitKernel::SubmitLinearCopyCommand` waiting
  on an hsa signal - **a lazy kernel-module load** (the PLE postops'
  first launch) whose code-object blit to the GPU never completes.
  Two early HSA runtime threads (created at device init) are stuck in
  `ioctl(KFD, KFD_IOC_MEMORY_PREFAULT)` - the fault handlers.

  The story: the first kernel read of the 31.64 GiB *pageable*
  complement starts a GTT page-fault storm; in flight, the lazy module
  load's blit contends with the fault handlers and the KFD path deadlocks
  (no fault, no completion - the exact shape of every hang).

**Fix (in the tree):** the alias complement is now page-locked from
birth - `pin_cache_complement(..., pin=true, ...)` in the alias block
(the `cudaHostAlloc` path, the same one the 260 MiB embedding uses
successfully on this APU).  No lazy faults, no storm; the
register_complement_for_gpu step is a no-op on a full pin.  Fallback on
refusal: the old malloc + register path.

Boot protocol 5 (tools/hip/p1_boot_run5.sh): gate, traced engine under
gdb (backtrace again if it still hangs), and on a PASS it chains
python + marker + longfill on the same boot - a clean run does not
degrade the APU, so a green boot is the whole P1 characterization.

**Run 18 (2026-10-03 17:44, boot of 17:40): the pin did not help -
and the backtrace says why it could not.**
- "mapped pinned cache complement ready: resident 31.64 GiB,
  pinned 31.64 GiB; page-locked and mapped" - the complement was
  pinned from birth this time.
- Same hang, same spot, same backtrace: main thread in
  hsa_executable_freeze -> BlitKernel::SubmitLinearCopyCommand.
  The complement's pages are not what is faulting.

**The real trigger, identified from the tuning table:** the PLE
projection shapes are rows 6-9 of tools/hip/gfx1103-hipblaslt-100401.txt
(n=48/512, k=2560, ldy=96/512 - the per-token PLE GEMMs).  hipBLASLt
loads its algorithms LAZILY on a shape's first matmul, and the first
PLE-shape matmul of the run lands in the layer-1 PLE block, mid-prompt.
That lazy code-object load is the hsa_executable_freeze in the
backtrace; its blit deadlocks against the KFD fault handlers
(KFD_IOC_MEMORY_PREFAULT).  The statically-linked kernel modules load
fine at layer 0 (the alias MoE ran) - it is specifically the lazy
hipBLASLt load mid-prompt that wedges.

**Fix (opt-in, STRATA_HIPBLASLT_WARMUP=1):** one matmul per tuned row at
"tuning enabled" time (before the prompt, calm state) - all three
solution ids (5992/5387/5388) load at init, and the prompt's first
matmul of each shape is a plain launch of an already-loaded algorithm.
The dGPU path is untouched (no-op without the env var).  A/B stays
cheap: dropping the env var reproduces the hang.

**Run 19 (2026-10-03 18:27, boot of 18:00): the warmup did not
help either - which narrows the story further.**
- "hipBLASLt warmup: 9 of 9 tuned shapes resolved and launched before
  the prompt" - all nine tuned algorithms (solution ids 5992/5387/5388)
  loaded at init, in a calm state, without incident.
- Same hang, same backtrace (hsa_executable_freeze ->
  BlitKernel::SubmitLinearCopyCommand; two runtime threads in
  KFD_IOC_MEMORY_PREFAULT).  The layer-1 module load is not a tuned
  hipBLASLt algorithm.

  The consistent picture across runs 16-19: in alias mode the layer-0
  gather kernel reads all 31.64 GiB of the complement in a few seconds,
  which leaves the driver's fault handlers busy for minutes installing
  the complement's GTT page tables (they are lazy even when the region
  is page-locked).  Whatever module load lands while that drain is still
  running deadlocks its code-object blit against the KFD lock held by
  the fault handlers.  The copy-50 run never hit it: its complement's
  page tables are installed in 24,526 small DMA reads spread over 20 s
  (the pool streaming), never a storm.

**Fix (run 19, in the tree):** a full DMA read pass over the
complement right after it is built and registered - the micro's proven
recipe (a full H2D ping made its reads clean; a partial ping left the
rest faulting, so registration alone is not enough for the driver's
page tables).  ~1 s at the measured 36 GB/s, a 1 GiB temporary device
buffer, logged on success/failure.

**Run 20 (2026-10-03 18:49, boot of 18:47): the hang is GONE -
replaced by a memory-exhaustion abort that reveals the iGPU's
budget.**
- The prefault pass worked: "one DMA read pass over 31.64 GiB in
  898 ms" (35 GB/s, the measured H2D rate).
- But the 1 GiB H2D target + the pass itself ran up the iGPU's VRAM
  budget: the card's carve-out is **16 GiB** (mem_info_vram_total,
  idle-used 164 MB), and prefill then reported "device buffers for a
  chunk of 512 tokens do not fit" and aborted cleanly.  The
  "unspecified launch failure" lines came during teardown
  (hipModuleUnload) - a stale fault state, and they degrade the APU
  like any fault.
- Note: run 20 never reached the prompt, so it says nothing about the
  layer-1 module-load deadlock.

**Fixes for run 21:** the prefault's target is now 4 MiB (8,192 copies,
~1 s at the same rate, ~zero VRAM footprint).  And the gdb forensics
get the missing data if the layer-1 hang returns: `thread apply 1
bt 40` (through the libamdhip64 frames to the engine-side caller, so
the module load's identity is known - hipModuleLoadData = a
BLAS JIT cubin vs. a static kernel TU) plus `info proc mappings` (to
symbolize the library frames offline).

**Run 21 (2026-10-03 20:06, boot of 19:22): the 4 MiB prefault
fixed the VRAM exhaustion - and the layer-1 hang is back, with the
identity of the module load finally in hand.**
- Prefault: clean, ~1 s.  No "do not fit" - the prompt's buffers fit
  again.
- Same hang at "prompt layer 1".  The deep backtrace
  (`thread apply 1 bt 40`) finally reaches the engine:
  `#16 hipLaunchKernel -> #17 strata::kernels::native_ple_postops_batch
  -> #18 strata::prefill::Prefill::run -> #19 main`, under
  `hsa_executable_freeze -> BlitKernel::SubmitLinearCopyCommand`.

  **The module load is the lazy load of native_ple_postops.cu's code
  object, and its first launch is the prompt's layer-1 PLE block in
  every run** (the per-token variant only runs in the decode token
  loop, which starts after the prompt; the verify graph does not
  capture it).  The hipBLASLt warmup (run 19) and the PLE-table/GEMM
  work are all exonerated - this one launch is the deadlock trigger on
  this APU.

**Fix (run 21, in the tree):** one dummy launch of the PER-TOKEN
variant `native_ple_postops` (same TU, same code object) at init -
after "session is up", alias mode only, finite zero inputs, 342 KiB of
temporary device memory, followed by a device sync.  The per-token
variant is read-only in the PLE history (the batch variant advances
it), so the prompt starts with exactly the same state as before.  The
layer-1 launch is then an ordinary launch of an already-loaded code
object.

**Run 22 (2026-10-03 21:17, boot of 21:04): the "hang" was my own
warmup - a null-stream throw.**
- Prefault clean (1083 ms), "session is up" printed, then the process
  sat in `std::terminate -> abort` (caught mid-teardown by the
  forensics SIGSTOP): `__cxa_throw` out of
  `strata::kernels::native_ple_postops` - the first line of that
  function is `if (!stream) throw std::invalid_argument(...)`, and the
  dummy launch had passed stream 0.  The throw happened before any
  kernel launch, so the code object never loaded and this run says
  nothing about the init-time load hypothesis.

  Fix: the warmup creates a temporary stream, launches on it,
  stream-syncs (the load is what matters), destroys it.

**Run 23 (2026-10-03 22:10, boot of 21:28): the module-load
deadlock is FIXED; a new IMA surfaces in the expert section.**
- "layer 1 PLE block done" printed for BOTH halves - the
  native_ple_postops_batch launch that deadlocked in runs 16-21 now
  completes (the init-time code-object warmup works).
- Then: "prefill copy_i32: an illegal memory access" - a clean IMA
  (the process self-terminates).  The last trace predates the
  "experts begin" marker, so the fault is in the expert section's
  first kernels - the grp_mapped copies
  (copy_i32(m.grp_dev, m.ids, ...), copy_i32(m.slot_dev, m.grp_dev +
  grp_tk, ...)): kernels that read/write m.grp_dev, the
  cudaHostGetDevicePointer ALIAS of a small pinned host buffer.

  Two things make this path suspect:
  1. It is a branch the APU has never run - every copy-path run so far
     had misses and took the staging-ring branch; the alias mode (zero
     misses) is the first to take the grp_mapped host-grouping branch.
  2. The micro tested kernel reads of registered host memory through
     the HOST pointer, never through the device ALIAS.

  (The error is sticky-async, so an earlier kernel - the layer-1 QSA
  attention - could be the true fault site and the copy_i32 check only
  the first to report it.  The A/B below separates that: if the IMA
  moves with the grp path, the grp copies are it.)

  The engine already has a bypass: STRATA_GROUP_COPY=1 skips the grp
  allocation and uses plain cudaMemcpyAsync.  The micro gained
  --alias (kernel read through the device alias of a pinned region).
  Boot protocol 6: gate, --alias 8, engine alias arithmetic with
  STRATA_GROUP_COPY=1 (chaining the smoke on a pass).
