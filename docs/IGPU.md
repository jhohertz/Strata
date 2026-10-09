# Integrated GPU (APU) expert path: alias, don't copy

Branch `igpu-rework`, off `gfx1103` (Engine 0.1.36 base). Machine: Ryzen 7 8700G
+ Radeon 780M (gfx1103), 64 GiB RAM - 8 GiB carved out for the iGPU and 48 GiB of
the rest addressable by the GPU as the GTT pool (docs/GFX1103.md §1.1) - Ubuntu
26.04, ROCm 10.0.0~pre4. Model and
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
  budget: the card's carve-out read **16 GiB** at that scan (mem_info_vram_total,
  idle-used 164 MB; it is 8 GiB on the current BIOS setting, docs/GFX1103.md §1.1),
  and prefill then reported "device buffers for a
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

**Run 24 (2026-10-03 22:26, boot of 22:22): the alias is
innocent; the IMA moved; the hang was an unreported async IMA.**
- micro `--alias 8`: PASS - `hipHostGetDevicePointer` returns an
  **identity** pointer on this APU (device-alias == host pointer), and
  a kernel read through it runs at 54.7 GB/s.  The grp_dev alias
  mechanism is not a fault source.
- engine with STRATA_GROUP_COPY=1: the "prefill copy_i32" IMA is gone
  (those kernels no longer exist in this path), but the run still
  failed - it HUNG 900 s (the timeout), and the teardown revealed the
  error state: "illegal memory access".  The last trace is now
  "layer 1 attn+mlp done, experts begin" - past the point where run
  23 died.  So: an IMA in the expert section (quantize / bounds /
  group computes / combine) that no check() caught, leaving the stream
  dead and the run spinning until the timeout.  (Layer 0's expert
  section - same kernels, same complement - completed fine.)

  The non-staging expert section now has named, gated error checks
  (STRATA_PREFILL_CHECKS=1: after mmq quantize / after the bounds
  upload / after the expert computes / after moe_combine) - an async
  IMA now reports its section and layer and exits instead of hanging
  900 s.

## P1 result: the alias path works (runs 25-29, boots 2026-10-03 22:48 - 2026-10-04 02:22)

The expert-section IMA of runs 23-28 took five more boots to find, and it was
not a driver bug at all.

**Run 25 (MMQ off).** With STRATA_PREFILL_MMQ=0 the fault moved into the
non-MMQ subpath ("after the f16 products"), which ruled out the mmq gemm and
the ggml context pool.  dmesg finally showed the fault's shape:
`GCVM_L2_PROTECTION_FAULT_STATUS` gfxhub page faults at GPU VA 0x0, 0x1000,
0x2000, ... 0x6000 (4 KiB pages, PERMISSION_FAULTS 0x3, client 10/TCP) plus
`sq_intr type 2` - a NULL-base read, identical in the MMQ-on and MMQ-off
runs.

**Run 26 (the micro reproduces the workload).** A new `igpu_gtt_micro
--dequant <gib>` arm runs the engine's exact layer-0 dequant workload - 512
experts, each `iq_dequant_gu_f16` + `iq_dequant_f16` (Q2_0, n_ff=640,
n_embd=2560, the real 1,382,400 B blob with gate at 0, up at 460800, down at
921600 = `native_expert_layout`) - over a registered region after the
engine's 4 MiB-chunked prefault.  It PASSED (512/512, clean APU), so the
kernels and the GTT reads were exonerated.  The engine with per-kernel checks
failed at "layer 0 after dequant gu" - the same kernel the micro had just
passed.

**Run 27 (stream and pressure are innocent).** Micro variants: `nb`
(non-blocking stream, the engine's m.cs class) - PASS.  `nb` + 6 GiB of held
device memory (the engine's session-scratch pressure on the 16 GiB budget) -
PASS.  The engine with the new env-gated blocking compute stream
(STRATA_IGPU_BLOCKING_CS=1) still failed at the same check.

**Run 28 (scale and the smoking gun).** Micro `--dequant 32 nb 6` - a
32 GiB registered region (the engine's complement scale) with the nb stream
and the pressure - PASS.  Scale is innocent.  The engine's first-compute
pointer dump (moved before the launch; run 26's had sat after the fault
point and never printed) finally printed the launch's actual arguments:

    blob=(nil) blob+up=0x70800 dq_gu=0x75367b200000 ... (all valid)
    fmt gu=42 d=42 up_off=460800 down_off=921600 host_res=0

`blob_dev` - the expert pointer from the alias table - was NULL, while every
other pointer was a valid device VA.  The dequant read the gate region from
base 0 and the up region from 0x70800: the dmesg "NULL-base" fault was a
literal NULL pointer all along.

**Root cause.** `ExpertCache::set_aliased_pointers` (the batch table fill)
uploads the 24,576 pointers to `d_ptrs_` (device) and verifies the read-back,
but never filled `h_ptrs_` (host), which is the table `device_slot()` - the
prompt path's lookup - reads.  `h_ptrs_` sat at its `open_aliased` zeros, so
every prompt-path `device_slot()` returned NULL.  The decode path uses
`d_ptrs_` (verified, correct), which is why the table "looked" right and the
failure only ever appeared in the prompt's expert section.  The per-slot
`fill_slot` updates both tables; the batch fill forgot the host mirror.  One
line: `h_ptrs_.assign(host_pointers, host_pointers + n)` after the read-back
verification.

**Run 29 (P1 works).** With the fixed table, the prompt ran end to end for
the first time: 48 layers, 5110 resident expert computes, 0 staged, 0 blob
reads from the file.  Decode then needed one more line: the verify graph's
init guard rejected the alias tier (`cache_base == nullptr`), even though the
alias design (base null + `slot_off` = the per-slot host pointers, so every
"base + offset" resolution reduces to the pointer itself) was already wired
through the pool's plan builder (`drive.d.cache_slot_off =
xcache.slot_offsets()`, which returns the host pointer table in alias mode)
and `TokenHits::on()`.  The guard now accepts residency + either an arena or
the pointer table.  That was the last blocker.

All four Phase-B smokes pass in alias mode (STRATA_IGPU_ALIAS=1,
STRATA_GROUP_COPY=1, MMQ on, Q2_0 native pack, this machine):

| case      | verdict | checks on | checks off (clean) |
|-----------|---------|-----------|--------------------|
| arithmetic| 396 OK  | 11.4 pf / 12.9 dec tok/s | 12.4 pf / 11.8 dec tok/s |
| python    | OK      | 14.8 dec tok/s | - |
| marker    | OK      | 8.1 dec tok/s  | - |
| longfill  | 724913 OK | 35.9 pf / 20.5 dec tok/s | **38.9 pf / 20.5 dec tok/s** |

Prefill on the 1162-token prompt is 38.9 tok/s clean, against the 0.1.36
copy path's 34.7-37.95 on the same prompt, and decode 20.5 against the copy
path's 18.75-19.52.  The alias path removes the H2D expert copies entirely
(the run reports "0 exchanged with the VRAM tier, 0 blob reads from the
file"), but the prefill is not 2x faster: the earlier analysis found ~20.6 s
of ~31.3 s per prompt is host-side work (per-token PLE block, routing,
window handling), which the alias does not touch.  That is P2/P4
(collapsing the ring/flag waits, shrinking the CPU pool's surface).

Short-prompt prefill is still slow (44 tokens in 3.5 s, ~80 ms/token): with
one chunk the per-token host work is the whole prompt.  Also P2.

### What P1 left behind

- The alias path needs `--mmap-experts` (the complement is built from the
  file source), a file-backed pack, and runs with a 31.64 GiB
  page-locked + registered + prefaulted RAM complement and a 192 KiB device
  pointer table; the iGPU's 16 GiB VRAM budget then holds the session
  (~8 GiB) and the graph buffers.
- The debug machinery is gated and dGPU-untouched: STRATA_PREFILL_CHECKS
  (named section checks, now per-kernel in the expert compute),
  STRATA_GROUP_COPY, STRATA_HIPBLASLT_WARMUP, STRATA_IGPU_BLOCKING_CS,
  STRATA_TRACE (per-layer/per-section traces + the first-compute pointer
  dump), the PLE code-object warmup at init, and the 4 MiB-chunked
  complement prefault pass.
- The micro's `--dequant` arm (with `nb` and pressure options) stays: it is
  the engine's expert-section workload in a process without the engine, and
  it is what kept every hypothesis testable without burning a boot.

## P2.1: profile the alias prefill before optimizing it (P2.1, 2026-10-04)

The pre-P1 "20.6 s of 31.3 s is host-side" estimate predated alias mode, so P2
starts by re-measuring.  `STRATA_PREFILL_TIMING=1` (existing) gives the phase
breakdown; `tools/hip/p2_boot_run1.sh` ran longfill + arithmetic in alias mode.

| case | tokens | prefill | decode |
|---|---|---|---|
| longfill | 1162 | 30551.6 ms (38.0 tok/s), TTFT 31036 ms | 160 in 7819.8 ms (20.5 tok/s) |
| arithmetic | 44 | 3594.6 ms (12.2 tok/s), TTFT 4082 ms | 160 in 13684.5 ms (11.7 tok/s) |

**The alias prefill is GPU-bound, not host-bound.**  `GPU timeline == wall`
(30551 = 30551 ms); host staging 0 ms, host chunk setup 12 ms, PLE 9 ms.  The
old P2 target (host-side work) does not apply: the GPU is busy the entire
prompt.  (The 20.6 s estimate was of the copy path, whose H2D copies are now
gone.)  The phase split (longfill): gemm gate/up 12050 ms (39.4 %), "dequant"
7996 ms (26.2 %), gemm down 3938 ms (12.9 %), qsa 3213 (10.5 %), gdn 2413
(7.9 %), hc read 1271 (4.2 %), ple 1301 (4.3 %).

"Dequant" at 0.95 GB/s (7.05 GiB read in 8.0 s) looked like a GTT problem,
but the `--dequant` micro arm (512 experts, Q2_0, real blob geometry) measures
the dequant kernel in isolation:

| micro variant | read rate |
|---|---|
| GTT, malloc + hipHostRegister (legacy stream) | 6.32 GB/s |
| GTT + MADV_HUGEPAGE | 6.26 GB/s (huges refused) |
| device memory (H2D-copied blobs) | 7.53 GB/s |
| GTT, non-blocking stream (`nb`) | 6.26 GB/s |
| GTT, cudaHostAlloc (the engine's complement mapping) | 6.26 GB/s |

**Mapping kind, page size, stream type and pressure are all innocent**: the
engine's exact mapping (cudaHostAlloc, identity alias) dequants at the same
6.26 GB/s the micro gets everywhere.  The kernel's access pattern itself is
slow in isolation (single-digit GB/s) but that is not the engine's gap: the
engine runs the same kind of work ~7x slower (0.95 vs 6.3 GB/s).

The gap closed on re-reading the profile: with MMQ on (this run's mode), the
`kPfDequant` phase in `compute()` is marked at the top of the lambda and the
next mark is the group product - so **the "dequant" phase actually contains
the `gather_native` copies (GTT read + VRAM write, 16 experts per group), not
`iq_dequant_gu_f16`**.  7996 ms = 5110 gathers of 1.38 MB (7.05 GiB read +
7.05 GiB written) plus the inter-launch gaps.  The next P2 measurement is a
micro arm for `gather_native` itself, to split kernel time from host launch
gaps.

### The 0.1.29 baseline and the mmvq A/B

The -46 % prefill regression vs 0.1.29 (commit 2c110cd) was queued with
`STRATA_OLD_IQ_MMVQ=1` as the single lever.  Verified from git: 0.1.29's
expert `compute()` lambda is byte-identical to 0.1.36's, the llama.cpp pin is
the same commit in both (`3cf03257`), and the Q2_0 MMQ build is unchanged -
so the regression is not in the expert gemm path.  `p2_boot_run2.sh` runs the
A/B.  First pass (this boot): arithmetic PASS 396 with the old mmvq, prefill
3536.3 ms (12.4 tok/s) vs the 3594.6 ms (12.2 tok/s) baseline - **no
change**: the mmvq kernel is not the prefill regression (at least for the
short prompt).  longfill faulted in the hipBLASLt warmup on this pass
(`unspecified launch failure`, 719) - the known flake, now seen once on the
alias path (previously only on the copy path; the site moves, here a BLAS
solution launch).  Arithmetic ran clean at normal speed after the fault,
which is a data point on the flake's aftermath, not a license to keep going:
reboot per protocol before the A/B's longfill.

## P2.2: the mmvq A/B, clean boot (2026-10-04, boot after the 03:44 flake)

`p2_boot_run2.sh` on a fresh boot: gate, longfill + arithmetic, both with
`STRATA_OLD_IQ_MMVQ=1`, both **PASS** (724913, 396).  The warmup flake did not
reproduce.  Numbers vs the P2.1 baseline (new mmvq):

| case | new mmvq | old mmvq | delta |
|---|---|---|---|
| longfill prefill | 30551.6 ms (38.0) | 29735.7 ms (39.1) | -2.7 % |
| arithmetic prefill | 3594.6 ms (12.2) | 3551.4 ms (12.4) | -1.2 % |

**`iq_mmVQ` is not the prefill regression** (the deltas are boot-to-boot
noise, and in the improving direction).

## P2.3: the 0.1.29 baseline, rebuilt and profiled - the -46 % regression is a measurement artifact

The remaining suspect for the -46 % was "somewhere in the 35 % of non-expert
phases".  To settle it, the 0.1.29-era branch (commit `41da073`, the one that
measured 68.8) was rebuilt in a separate worktree (pack and profile symlinked,
`STRATA_PREFILL_TIMING=1` available there too)
and run with the exact 0.1.29-era smoke flags (`p2_boot_run3.sh`):

| config (same boot, same flags) | longfill prefill | staging |
|---|---|---|
| 0.1.29 copy, first run (file partly cold) | 33527.5 ms (**34.7**) | 22249 ms |
| 0.1.29 copy, warm re-run | 30904.5 ms (**37.6**) | 20001 ms |
| 0.1.36 copy (Sept-Oct characterization) | 37.1/37.7 | - |
| 0.1.36 alias (P1/P2.2) | **38.9/39.1** | 0 |
| 0.1.29 copy (Sept 30, "warm batched") | **68.8** | (not recorded) |

**On this boot, 0.1.36 alias is the fastest configuration in every case.**
The "-46 % regression" was an artifact of comparing against the 68.8, which
is not reproducible in any state on this boot (34.7-37.6 no matter how warm
the file gets).  The 68.8 was almost certainly a different machine state in
September (fully-warmed 31.64 GiB file with the CPU pool running at 5+ GB/s
instead of the 1.8 GB/s seen here - 40.7 GiB streamed per prompt through the
pool is what the copy path pays every time; the alias path pays it once at
init, hidden in the complement build), or a steady-state rate reading.  The
phase profiles say the same thing: 0.1.29's gemm phases are *slower* than
0.1.36's (gate/up 13505 vs 12050 ms, down 5281 vs 3938 ms) - the code did not
regress the expert gemm; the copy path's 20-22 s of per-prompt pool staging
did the rest.  One real (small) delta: qsa doubled (1580 -> 3213 ms, 5 % ->
10 % of the prompt) between 0.1.29 and 0.1.36 - noted, not worth a boot.

**P2 restated, with same-boot numbers:** the alias prefill is 30.5 s of
pure GPU time (wall == timeline), split gemm gate/up 39 %, the gather phase
26 %, gemm down 13 %, qsa 10 %, gdn 8 %.  The expert section is 78 % of the
prompt.  The next lever is the gather phase: a micro arm that times
`gather_native` itself (GTT read + VRAM write per expert) to split kernel
time from host inter-launch gaps - if the kernel streams at the ~54 GB/s a
flat copy gets, the 8.0 s is launch overhead and the fix is batching;
if it is ~1.8 GB/s like the pool, the MMQ gather pattern is the problem.

## P2.4: the gather micro, the expert-path A/Bs, and the host/GPU separation (2026-10-04)

**The gather micro (`--gather` arm):** 512 per-expert `copy16_kernel`
launches (the engine's exact kernel, from `src/prefill/moe_mmq.cu`) over the
pinned complement mapping, back-to-back as the engine's compute loop does:

| micro | rate |
|---|---|
| 512 per-expert gathers (GTT read + VRAM write) | 33.70 GB/s (21 ms) |
| one flat copy of the same bytes | 39.32 GB/s (18 ms) |

The per-expert launch pattern costs only ~9 % vs flat.  The engine's same
work runs at 0.95 GB/s - **35x slower than the micro's 33.7**.  The kernel
and the mapping are both fast in isolation; the engine's expert section is
spending its 8.0 s somewhere else.

**The expert-path A/Bs (all alias, longfill, same boot):**

| variant | prefill |
|---|---|
| MMQ on (default) | 38.9 / 39.1 tok/s (30551 / 29736 ms) |
| `STRATA_PREFILL_MMQ=0` (f16 dequant + tuned hipBLASLt) | 39.0 tok/s (29788 ms) |
| `STRATA_PF_FUSED=1` (0.1.36's fused Q2_0 prompt experts) | 38.6 tok/s (30096 ms) |

**Every expert compute path lands at ~39 tok/s.**  (Corrected in `docs/GFX1103.md`
§13: this build defines neither `STRATA_PREFILL_MMQ` nor `STRATA_PREFILL_FUSED`, so
all three rows above ran the same f16 path - the env toggles could not engage the
paths they named.  With MMQ compiled in the same prompt runs 2.44x faster.)
The total prefill is not
sensitive to which kernels do the expert math - so the ~30.5 s wall is set by
something shared (per-expert dispatch, per-layer syncs, or the non-expert
phases), not by the gemm choice.  This also reframes the "gemm gate/up 39 %"
phase: with MMQ off the same wall has no gemm phases at all, yet the total
is identical.

**The host/GPU separation (queued, then the flake):** added `STRATA_HOSTLOOP=1`
(gated, dGPU-untouched): per expert section (host grouping -> moe_combine)
it records the host wall time and the GPU span (two events on the compute
stream) and prints the sums at prompt end.  The first run died to the known
APU flake mid-way: `prefill: routed id out of range` (the per-layer ids
readback - a GPU result corrupted by the fault) followed by the
`hipModuleUnload failed: unspecified launch failure` teardown hang until the
timeout kill.  Per protocol the box stops after this; the boot run for the
hostloop measurement is queued below.

### The 04:2x-12:5x failure cluster - and the build that never was

Four consecutive 0.1.36 alias longfills faulted (04:2x, 12:1x, 12:3x, 12:5x),
all with the known site-moves signature: corrupted layer-0 ids readback (x2),
then `prefill gather_rows16: unspecified launch failure` right after the
9/9 hipBLASLt warmup, then teardown hangs (`hipModuleUnload failed`, timeout
kill).  dmesg: `device wedged, but recovered through reset` (the driver's GPU
reset succeeds - a new process can run after it, which the 0.1.29 contrast run
proved).

The first two were attributed to the new `STRATA_HOSTLOOP` instrumentation -
**wrong**.  The instrumented build never compiled: `hl_on`/`hl_a`/`hl_b`/`hl_t0`
were declared inside the host-grouping branch but used at a scope outside it
(the walk's `if (!stream_all)/else` closes before the push site), plus a
cudaEventElapsedTime float*/double* mismatch.  The 04:2x `make` failed with
exactly that error, but the command pipeline (`make | grep ... && run`) let the
run proceed against the **pre-instrumentation** binary (grep's exit status
masked make's).  All four failures were the original 0.1.36 binary, two of
them with the debug env off.  (The instrumentation is now fixed and actually
builds; `p2_boot_run4.sh` still runs it, unchanged.)

Corrected record, same machine:

| binary | longfills | faults |
|---|---|---|
| 0.1.36 alias (original) | 10 (2 boots) | 4 (all since 04:2x) |
| 0.1.29 copy (41da073 worktree) | 4 (2 boots) | 0 |

The fault sites are unrelated kernels (BLAS warmup solution, layer-0 router
ids, gather_rows16, copy_i32, iq_dequant_gu_f16 over the whole 0.1.31-0.1.36
history) - the site-moves signature of the chronic APU/driver flake, now
clustering: 7/7 clean 0.1.36 runs on one boot, then 4/4 faults across the
next two boots with 0.1.29 clean in between.  Whatever degrades the APU for
this engine's workload does not reset with a simple reboot, or the 0.1.36
prompt path trips a driver state the 0.1.29 path never enters.  Unresolved;
needs a longer clean period and the A/B below, not more same-day runs.

### The kernel-level signature: userptr restore storm + MES queue hang (2026-10-04 evening)

The journal (which survives reboots) gives the fault its real shape.  Both
the first fault (04:24, boot -4) and a later one (18:24, current boot):

```
amdgpu_amdkfd_restore_userptr_worker hogged CPU for >10000us N times   # N grows: 4..7..19..35
MES failed to respond to msg=REMOVE_QUEUE
MES might be in unrecoverable state, issue a GPU reset  ->  MODE2 reset succeeded
```

The KFD **userptr restore worker** (userptr = the registered-host-pointer
mapping, exactly the alias complement's mechanism) runs in >10 ms passes that
**never finish** in faulting runs, and the **MES** (the GPU's microengine
scheduler) then hangs removing a hardware queue (doorbell 0x1004) until the
driver does a MODE2 reset.  A *clean* 0.1.36 alias run (P2.2, 03:59) shows
the same restore worker, but it settles after 4-5 passes - the difference is
that in faulting runs the restore loop does not terminate.

Exonerated on the evening of 10-04 (all on the faulting machine):

| suspect | test | result |
|---|---|---|
| 31 GiB userptr range alone | micro `--gather 31` | PASS 32.2 GB/s |
| + 8 GiB VRAM session pressure | micro `--gather 31 8` | PASS 30.8 GB/s |
| SSD keepalive file churn during the prompt | engine + `STRATA_SSD_KEEPALIVE=0` | FAULT |
| reboot / 46 min idle | boot D, first run | FAULT |
| kernel/driver/firmware change | cmdline, dpkg, fwupd history | none in the window |

The 10/10-clean -> 0/5-fault transition happened **within one boot**, between
03:59 and 04:24 (boot -4), with 0.1.29 and 0.1.36 alias runs interleaved and
clean on both sides; it has since persisted across three reboots.  No
software artifact in the repo changed in that window.  The remaining suspects
are machine state that survives a reboot (a power rail / VRM / thermal
marginality on the APU package - the iGPU shares the die with the CPU - or
GPU firmware state if the reboots were not full power cycles).  The two
2-3 minute boots at 14:09/14:21 (the user was at the machine) did not change
the outcome either way.

### Queued (needs a FULL power cycle - hold the power button / unplug the PSU -
plus a long idle, and the next attempt logs thermals): `p2_boot_run6.sh`

Gate, then a 1 Hz `rocm-smi` temp/power logger in the background, then
longfill alias with `STRATA_HOSTLOOP=1` + `STRATA_PREFILL_TIMING=1`
(instrumentation now compiles) + `STRATA_SSD_KEEPALIVE=0` (exonerated but
harmless).  The thermal log answers the one remaining question the software
can ask: does the package approach its thermal/power limit in the faulting
window?  If the run faults with clean thermals, the evidence is for a
driver/firmware-level userptr path problem (record it as such, with the
journal excerpts) rather than a hardware one.  If it passes, the machine has
recovered and the host/GPU separation (P2's original goal) proceeds as
planned.

### Reboot log (the long tail)

Runs 23-29 were one IMA and eight reboots.  The sequence that worked, for
the record: dmesg first (the fault address beat every black-box guess),
then a micro that replicates the exact workload (kernel + geometry + GTT
state), then the one-shot pointer dump placed before the launch.  The three
reboots in between (25-28) each killed one suspect: the mmq path, the stream
type, and the VRAM/scale state.  All three turns out to have been innocent
of the actual bug (a missing `h_ptrs_` assignment) - but the dump that
proved it was built by the boot that came before them.

## P2.7: the micro exoneration table, the section shape, and the launch-cost picture (2026-10-04 late night, post power cycle)

The machine came up at 21:09 after the first FULL power cycle (hold the power button).
`p2_boot_run6.sh` ran the gate, then longfill alias with `STRATA_HOSTLOOP=1` +
`STRATA_PREFILL_TIMING=1` + `STRATA_SSD_KEEPALIVE=0`:

```
PASS  longfill
strata prefill timing: 1162 tokens, GPU timeline 32468 ms, wall 32468 ms, host staging 0 ms: ...
strata prefill hostloop: 144 expert sections, host 32255 ms, gpu span 24573 ms (max section host 1595.9 / gpu 237.5 ms)
therm.log: max 60.0 C, max 65.1 W
```

The three arithmetic smokes then passed (arithmetic/python/marker, all PASS).  The
thermal log was clean all the way (peak 60 C / 65 W - the same peak as the healthy
September characterization, far from the ~95 C throttle point).  **The full power
cycle fixed the fault; a plain reboot did not.**  The fault state survives reboots;
it needs the power to go fully off.

### The hostloop numbers, read correctly

host 32255 ms ~= wall 32468 ms, gpu span 24573 ms.  The first reading - "host-bound,
the host is 7.9 s behind" - was wrong.  The host time is almost entirely the
per-section `cudaStreamSynchronize` waiting for the GPU (max section host 1595.9 ms
against a 237.5 ms GPU span = the host blocked on the previous section's work plus
the sync).  The non-expert GPU phases (qsa 1.56 s, gdn 3.87 s, hc 1.26 s, router,
ple, attn) tile the rest: 24.57 + 7.9 = 32.5 s = the wall.  **The alias prefill is
GPU-saturated; there is no 7.9 s of host-bound slack to recover.**

### The section shape, measured (STRATA_TRACE)

`resident 39738` = the number of expert computations (the gather launches), and the
per-section trace line shows **173-284 routed experts per section, average 284**
(512-token chunk, 48 layers, top-k routing over 512 experts).  So per section:
~284 gathers + 18 groups x (gu + swiglu + quantize + dn + 2 memsets).  The earlier
"36 experts/section" arithmetic (35.5 = 5110/144) was a misread: 5110 was a P1-era
stat with a different meaning, not this count.

### Per-launch cost, recomputed with the right section shape

Per prompt: ~39,700 gathers + ~2,600 gu products + ~2,600 dn products + ~5,000 other
launches = ~50,000 launches over 32.5 s = 0.65 ms/launch wall.  The gemm phases alone
(16.0 s for ~5,200 products + swiglu + quantizes) average ~3 ms/product; the gather
phase (8.4 s for ~39,700 gathers) averages ~211 us/gather.  The micro's same gather
kernel over the same bytes runs at 45-70 us each (30-35 GB/s sustained).  **In-engine
launches cost ~3-10x the micro's, on the same machine state, with every process-state
variable the micro can reproduce ruled out below.**

### The micro exoneration table (all same boot, post power cycle, 31 GiB pinned unless noted)

| arm | result | verdict |
|---|---|---|
| `--gather 31` (per-expert, contiguous) | 32.17 GB/s | baseline |
| `--gather 31 sc` (SCATTERED offsets across the full 31 GiB, the engine's routing pattern) | 35.39 GB/s | scatter/TLB exon |
| `--gather 31 8` (+8 GiB VRAM session pressure) | 30.77 GB/s | VRAM pressure exon |
| `--gather 31 2s` (second non-blocking stream, the engine's m.copy) | 32.17 GB/s | queue count exon |
| `--gather 31 nb 2s` (the engine's exact stream config) | 32.17 GB/s | stream type exon |
| `--gather 31 sr` (engine rhythm: D2H + full sync every 16 experts) | 29.49 GB/s | sync rhythm exon |
| `--gather 31 nb sr` | 30.77 GB/s | sync rhythm exon |
| `--gather 31 fc` (31.64 GiB experts.bin ALSO in the page cache, the engine's RAM state) | 33.70 GB/s | page-cache coexistence exon |
| `--gather 31 thr 8` (8 busy-spin worker threads, the engine's pool) | 32.17 GB/s | CPU contention exon |

Nothing the micro can express reproduces the engine's ~211 us/gather.  The remaining
differences are the interleaved MMQ/BLAS product kernels, the count of loaded code
objects/buffers, and the engine's total process state.  The A/B on the f16 path
(`STRATA_PREFILL_MMQ=0`, same boot) is the telling datapoint: **its phase profile is
near-identical to the MMQ path** (dequant 8380 vs 8401 ms, gemm gu 12019 vs 12036 ms,
gemm dn 3986 vs 4003 ms) despite completely different kernels in every phase - the
wall is set by the number of launches and their in-engine cost, not by any kernel's
compute.  Every expert path (MMQ int8, f16+BLAS, PF_FUSED) landing at ~39 tok/s is
the same fact from the other side.

### The lever: fewer launches (STRATA_MMQ_GROUP)

`mmq_group()` now honors `STRATA_MMQ_GROUP` (default 16, the dGPU build unchanged):
the group buffers scale linearly (16 -> 64 experts: ~21 MB -> ~85 MB, trivial for the
16 GiB VRAM), and the per-group product launches drop 4x (18 groups/section -> 5).
If the in-engine launch cost is ~0.5-3 ms, collapsing the ~5,200 gemm-side launches
is the single biggest available win; the gather side (39,700 launches) is the bigger
prize but needs a batched-gather kernel (one launch per group instead of per expert).
Queued for the next boot as `p2_boot_run7.sh`: gate, then longfill with
STRATA_MMQ_GROUP=64 / 32 / 16 (HOSTLOOP + TIMING on all three), same-boot A/B.

## P2.8: the fault cluster returns on a healthy-looking boot (2026-10-04 23:13)

The post-power-cycle boot was clean for four engine runs (longfill PASS 35.8 tok/s,
longfill MMQ=0 PASS, three smokes PASS) and several micro runs, with clean thermals
(peak 60 C / 65 W during load).  Then the STRATA_TRACE longfill (the fifth engine
run) degraded: 71 of 144 sections completed in ~8 minutes (~6.7 s/section, 30x the
normal 0.22 s/section), and it faulted with the same signature as the Oct 4 04:2x
cluster:

```
Oct 04 23:13:09 kernel: amdgpu 0000:66:00.0: MES failed to respond to msg=REMOVE_QUEUE
Oct 04 23:13:09 kernel: amdgpu 0000:66:00.0: failed to remove hardware queue from MES, doorbell=0x1202
Oct 04 23:13:09 kernel: amdgpu 0000:66:00.0: MES might be in unrecoverable state, issue a GPU reset
Oct 04 23:13:09 kernel: amdgpu 0000:66:00.0: Failed to remove queue 2
Oct 04 23:13:09 kernel: amdgpu 0000:66:00.0: GPU reset begin!. Source:  3
```

with `error: unspecified launch failure` in the engine's stderr.  Three refinements
to the fault record:

1. **The fault happens at clean thermals.**  This run's load profile peaked at the
   same 60 C / 65 W as the passing runs.  The ~95 C throttle association from the
   first cluster is not a trigger.
2. **The fault is a spectrum with a degraded slow phase before the hard hang.**  This
   run ran at 6.7 s/section for ~8 minutes before faulting; the 04:2x-12:5x cluster
   showed the same slow-then-hang shape.  The machine enters the slow phase after an
   unpredictable number of healthy runs (10 on the Oct 3-4 boot, 4 on this one) and
   stays there (across reboots) until a full power cycle.
3. **The micro never shows the slow phase.**  Between and around the degraded/faulting
   engine runs, the micro's gather arm held 30-35 GB/s.  The degradation is specific
   to the engine process (its queue usage under the full 31.64 GiB complement +
   sustained product load), not to the machine's DRAM/GTT path.

Operating rule updated: after a full power cycle, a boot is good for roughly 4-5 heavy
engine runs before the degraded phase appears; budget experiments accordingly (the
MMQ_GROUP A/B is exactly that size), and treat any section running >1 s/section as
"stop now, do not reboot, power cycle" (a reboot does not clear it).

## P2.9: the thermal/idle theory is dead; the strategy changes (2026-10-05 00:3x, degraded boot)

The user (rightly) rejected the "cool it down for hours" protocol.  The evidence agrees:
P2.8's fault came at **60 C / 65 W, identical to the passing runs**, and the boot 0 run
(18:24, after a 49-minute idle) faulted anyway.  The idle did nothing; the thermals were
never at their limit.  `p2_boot_run6/7.sh`'s "long idle" instruction is retracted.

### What the degraded machine says right now (probe, 00:38)

After the 23:13 fault, a fresh longfill probe faulted within ~10 s of the prompt start:
`prefill: routed id out of range` at layer 0 (the corrupted-ids readback, the 04:2x
cluster's signature) + `unspecified launch failure` inside the hipBLASLt warmup, then the
same MES REMOVE_QUEUE hang (queues 0/1/2) + MODE2 reset.  The state machine has degrees:

| degree | behavior | observed |
|---|---|---|
| 0 | 0.22 s/section | first 4-10 runs of a post-power-cycle boot |
| 1 | ~6.7 s/section for minutes, then fault | the 23:13 trace run (71 sections in 8 min) |
| 2 | immediate fault (corrupted first-layer readback) | the 00:38 probe, after the reset |

**The GPU MODE2 reset does not heal the state - it deepens it** (degree 1 -> 2).  The
machine state survives warm reboots and is cleared only by a full power cycle, so the
accumulator lives in GPU firmware (the MES), not RAM.

**The micro runs at 33.7 GB/s on the degraded machine** - identical to healthy.  The
DRAM/GTT path is never the problem; the degradation is the driver's handling of the
engine's own surface (queues + the 31.64 GiB userptr under sustained load).  A GPU
devcoredump now exists at `/sys/class/drm/card1/device/devcoredump/data` (root-only) for
both the 23:13 and the 00:38 faults.

### Why the userptr must stay, and what changes

The memlock rlimit is 8 MiB (soft = hard): a 31.64 GiB complement cannot be `mlock`ed
without root, and the no-userptr variant (plain mmap + DMA-prefault, which P0 proved
kernel-readable) would be a pageable 31.64 GiB with no eviction protection on a 45 GiB
box.  `cudaHostAllocMapped` (the driver's page lock, no rlimit) or `hipHostRegister`
chunks are the only no-root page locks.  So the userptr stays; the exposure is reduced
instead:

- **`STRATA_IGPU_PIN_CHUNK_GIB` (new, expert_source.cpp)**: register the complement as
  N chunked userptr ranges (e.g. 4 x 8 GiB) instead of one 31.64 GiB range, each cut at
  an expert boundary.  The fault signature is the per-range restore worker
  (`amdgpu_amdkfd_restore_userptr_worker hogged CPU >10000us N times`; clean runs settle
  in 4-5 passes, faulting runs run 35+ and never terminate) - a quarter-sized range makes
  each pass a quarter of the work.  Fallback: if any chunk is refused, the arena is
  unmapped and the single-registration path runs (byte-identical to before).  Release
  unregisters each chunk by its own base.  Default off; dGPU untouched.
- **Live degradation guard (prefill.cpp, STRATA_HOSTLOOP=1)**: a section > 2 s of host
  wall after the first five prints one `DEGRADED` line (healthy ~0.2 s; degraded ~7 s,
  P2.8).  The run's result is flagged non-comparable instead of silently entering the
  numbers.

### The new protocol: probe, then burn in (`p2_boot_run8.sh`)

No idling, no guessing:

1. **Probe** - one longfill decides the machine's state in ~3 minutes.  Not clean ->
   capture the journal and stop (the reset is a power cycle, and only then).
2. **Burn in** - repeat longfills (60 s apart) and count the clean runs until the
   DEGRADED warning, a fault, or MAX_RUNS.  The count is the boot's run budget for the
   configuration.  Each run also captures the restore_worker's journal line count (the
   fault's leading indicator).
3. **A/B** - baseline boot (single registration, expected 4-5 clean runs, P2.8) vs a
   chunked boot (`CHUNK_GIB=8`).  If the chunked boot outlasts it (10+), the boot ritual
   is gone: a machine lasts a day of experiments per power cycle.  If not, the next
   variable is the chunk size, then the queue count (fold m.copy into m.cs in alias mode).

The file page cache is not a variable: the complement fill already `madvise(DONTNEED)` +
`posix_fadvise(DONTNEED)`s each file layer as it copies it (the `fc` micro arm's
coexistence was therefore not the engine's actual state).  No kernel/firmware updates
are pending for the 7.0.0-38-generic / ROCm 10.0.0~pre4 stack.

## P2.9b: the baseline budget is confirmed, and the timeout was making things worse (02:29-02:53 boot)

After a full power cycle, `p2_boot_run8.sh` (v1) ran the probe + burn-in:

| run | result | wall |
|---|---|---|
| probe | clean 37.5 tok/s | 63 s |
| 1-3 | clean 37.3-37.5 tok/s | 64-67 s each |
| 4 | **faulted at the first gather** (`prefill gather_rows16: unspecified launch failure`, t+46 s) | - |

**The 4-run budget is now confirmed twice** (P2.8: 4 clean then the slow phase; this boot: 4 clean
then an immediate first-gather fault).  The transition to the bad phase lands in ~1 minute
(between one run's end and the next's first expert section) and is invisible to the micro
(33.7 GB/s throughout).

**The v1 protocol made the fault worse.**  The faulting run does not exit: it hangs in its
error path after printing the failure.  v1's `timeout 900` therefore SIGTERM'd the hung
process 14 minutes 14 seconds after its fault - and the journal's MES REMOVE_QUEUE lines
landed exactly at the SIGTERM (02:52:51): the KILL, not the fault, drove the teardown's
queue-removal hang and the MODE2 reset.  A faulted engine process holds its KFD queues
until it exits; killing it mid-teardown is the worst possible moment.

Two consequences:

1. **v2 never kills** (`p2_boot_run8.sh` rewritten): strata runs in the background and its
   stderr is watched; a clean run exits on its own (~65 s warm), a fault is recognized in
   seconds, recorded, and the process is left alive (it dies with the power cycle).  The
   per-run forensics now count both the restore_userptr_worker lines and the MES lines in
   the run's window.
2. **The queue lifecycle is the prime accumulator candidate.**  Every engine run creates
   and removes ~3 KFD queues (default + m.cs + m.copy); a faulting run's removal is the one
   that hangs ("Failed to remove queue 0/1/2").  If even clean removals leave residual
   MES state, the 4-10 run budget is queue-table churn, not GPU work.  Two surface
   reductions are queued behind the chunk A/B: (a) the chunked userptr (P2.9), (b) one KFD
   queue in alias mode (fold m.copy and the blocking prefault memcpy into m.cs: 3 -> 1
   queue per run).

Machine state at the end of this segment: post-fault, post-kill - the next engine run will
fault immediately; no more engine runs on this boot.  The micro (33.7 GB/s) is not an
engine-state probe.

## P2.9c: the chunked pin's first flight (16:2x boot) - two real bugs, caught by the probe

The first CHUNK_GIB=8 probe died at t+27 s with `Aborted (core dumped)` - and the probe-first
protocol did its job: the machine spent nothing (death before the prompt, clean process exit),
and the fault was **in my new code, not the machine**.  The chunked registration itself worked
(`page-locked and mapped in 4 chunked registrations (8 GiB each)`); two bugs followed:

1. **A DMA that spans two registrations is refused.**  The chunk cuts are expert-boundary-aligned
   (multiples of 1,382,400 B, not 4 MiB-aligned), and the prefault pass copies in 4 MiB pieces
   from the arena base - so one copy straddled the first registration boundary at ~8.0 GiB and
   `cudaMemcpy` returned `invalid argument`.  Fix: the chunked pin records its ranges
   (`complement_dma_ranges()`), and the prefault pass walks them, never crossing a boundary.
2. **A refused DMA latches into `cudaGetLastError`.**  The prefault warning path did not clear
   it, so the next post-launch error check - the PLE warmup's GR RMSNorm - blamed a healthy
   launch, threw, hit a `noexcept` frame, and aborted.  The `terminate ... native GR RMSNorm
   launch: invalid argument` was the *second* code's crash for the *first* code's error.  Fix:
   the prefault pass clears the latched error; and this is a general lesson for the engine's
   check convention - every swallowed API failure must clear the latch or it misattributes the
   next check.

Fixed and re-run on the same boot (the budget was intact: no sustained GPU work had happened).

## P2.9d: the chunked pin does NOT extend the boot budget - the accumulator is per-session, not per-range (16:2x boot, continued)

After the P2.9c fixes, the CHUNK_GIB=8 burn-in ran on the same boot:

| session | what | result |
|---|---|---|
| 1 | the crash probe (died at warmup, before any prompt) | no budget spent at first sight |
| 2 | probe longfill | clean 37.3 tok/s |
| 3 | burn-in 1 | clean 37.2 |
| 4 | burn-in 2 | clean 37.4 |
| 5 | burn-in 3 | **faulted at the first gather (t+54 s)** - same site as the baseline boot |

**Fault on the 5th session - identical to the baseline boot** (probe + 3 clean, fault on the 5th
session there too; the crash probe's session counted here).  The chunked registration changed the
range layout 4x and nothing else moved: not the fault site, not the session count, not the speed
(37.2-37.4 vs 37.3-37.5 baseline - within noise).  **Verdict: the userptr range size is not the
accumulator.**  The restore-worker storm was a symptom of the bad phase, not its cause.

What the three boots do agree on: **~5 sessions that put sustained load on the userptr+queues
per power cycle**, independent of registration shape; small sessions (the smokes, and the micro's
context, which ran dozens per boot fault-free) cost less.  The remaining code-side candidate is
the KFD queue lifecycle itself (3 queues created/removed per session; a faulting session's removal
hangs in MES) - `STRATA_IGPU_ONE_QUEUE` (fold m.copy and the blocking prefault memcpy into m.cs:
3 -> 1 queue) would test it, but it is a deeper change and the budget it buys is uncertain.

**Practical decision: the budget is not a blocker - it is a planning constraint.**  ~4-5 big
sessions per power cycle with a probe that says (in 60 s) whether the boot is live is enough for
real development: one probe + a 3-arm A/B per boot, results at 65 s/run once the file cache is
warm.  The fault itself is driver territory (the signature, the session tally, and the two
mitigations tested are all recorded here).  The next boot therefore runs the experiment P2 was
always about - the launch-cost A/B (`STRATA_MMQ_GROUP` 64/32/16, `p2_boot_run9.sh`, v2 fault-safe
mechanics) - which fits the budget exactly: probe + 3 arms.

## P2.9e: a full power-off did NOT clear the state this time (17:06 boot) - shutdown-while-hung is the suspect

The 16:2x chunk boot ended with a faulted session, a failed queue eviction, and the faulted
process still hung (spinning) when the user shut the machine down at 16:44.  After ~22 minutes
fully off (journal: boot -1 ends 16:44:37, boot 0 starts 17:06:22), the fresh boot's FIRST
engine session faulted:

| session | t+ | signature |
|---|---|---|
| probe (run9, 17:08) | 54 s | restore storm (4->5->7->11->19 passes) at **17:08:57 - ~17 s in, during the complement build/prefault, not the prompt** - then `routed id out of range` at layer 0 + BLASLt launch failure; teardown: `Failed to evict queue 3`, `Failed to evict process queues`, `Failed to quiesce KFD`, MODE2 reset (succeeded) |
| probe retry (17:14) | - | `illegal memory access` - faulted, process hung |

With two hung processes holding queues the MES refused to evict, this boot is done; engine work
stops.  Three refinements:

1. **The storm clock**: on a cold-file first session the restore worker is doing its heaviest work
   during the build+prefault phase - that is where the storm starts (the earlier "storm during the
   prompt" reading was from warm-file sessions, where the build is fast and the storm lands in the
   prompt).  The storm is the registration/first-touch path losing a race with page-table work;
   settling in 4-5 passes vs blowing past 19 is a race outcome, not (only) a wear counter.
2. **`unrecoverable state` now has a mechanism-level description**: the MES refuses to evict the
   queues of a faulted process; the hung process keeps them; and - new this boot - **a shutdown
   performed while a process is in that state is a candidate for carrying the bad MES state
   across a power-off** (the driver's shutdown/suspend path touches the same queue machinery the
   fault just broke).  The earlier "power cycle fixed it" boots had also ended with hung/faulted
   processes, so the data is thin - but the "power cycle always fixes it" rule just took its first
   clear counterexample, and shutdown-from-faulted-state is the variable that changed.
3. **Engine behavior note**: the faulted process does not exit - it spins in the error path
   (STAT=R) forever.  A self-terminating fault path (hard exit after printing, no spin) would both
   stop the CPU burn and make "did the boot end dirty" unambiguous.  A usability fix, not a fault
   fix; queued low.

Operating rule updated: **if a session has faulted this boot, never shut down to "reset" - cut
AC (unplug / PSU switch, hold the power button ~10 s to discharge) and wait before the next
power-on.**  The first act on the next boot is the probe, which decides in ~60 s whether the
rule works; if a session-1 fault follows an AC-cut-with-discharge, the persistence is not in the
driver's shutdown path and the reset ritual theory dies entirely - the storm is just a race this
machine is losing more and more often, and the honest framing becomes "a flaky driver on a
faulted-state machine; budget experiments around probe results, one boot at a time."

## P2.10: the driver-race test matrix (2026-10-05, user-led: BIOS + module parameters)

Position: not hardware (faults at 60 C/65 W, micro always fast, MODE2 heals in seconds),
most likely the kernel driver's queue/userptr path - the fault's own strings are
driver-internal (`amdgpu_amdkfd_restore_userptr_worker`, `MES failed to respond to
msg=REMOVE_QUEUE`, `Failed to evict process queues`, `Failed to quiesce KFD`) and the
kernel even prints its own advice: "consider switching to WQ_UNBOUND" (a workqueue
maintainer note about this exact worker).

### BIOS items (change ONE per boot, probe first, record here)

| setting | try | why it is on the list |
|---|---|---|
| IOMMU (AMD-Vi) | off (or on - make it consistent) | the cmdline is self-contradictory (`iommu=off amd_iommu=on`); KFD SVM + IOMMU is a classic hang area; GTT/DMA mapping behavior changes with it |
| Global C-states Control | disabled | CPU deep sleep vs queue-doorbell/MES wake; the standard amdgpu-hang workaround; matches "breaks under sustained load" |
| Power Supply Idle Control | Typical Idle | the documented Ryzen instability setting (cTLP class) |
| Above 4G Decoding / Re-Size BAR (SAM) | toggle | changes how host memory is BAR/GTT-addressed for the iGPU |
| ErP / deep sleep in S5 | enabled | guarantees standby rails are cut - the honest version of "the power cycle"; if MES/SMU package state survived our power-off, ErP On removes the loophole |
| Memory Context Restore / Fast Boot | disabled | rules out cross-boot persistence of memory training/context state |
| iGPU UMA (VRAM) size | 4-8 GiB instead of 16 | the box reserves 16 GiB "VRAM" + GTT accounting to 48 GiB (KFD reports 51.5 GB of heaps on 45 GiB of RAM - oversubscribed); GTT allocation pressure is exactly what the restore worker churns through |
| BIOS/AGESA version | latest | AGESA updates fix iGPU/microcode instability on 7000-series |

### amdgpu module parameters (grub `GRUB_CMDLINE_LINUX` additions; one per boot, with the probe)

| parameter | purpose |
|---|---|
| `amdgpu.debug_evictions=1` | DIAGNOSTIC: logs the queue-eviction machinery the fault hangs in - the leading indicator becomes a labeled event |
| `amdgpu.mes=0` | THE TEST: if MES is the component hanging, removing it changes everything (gfx11 may fall back to the legacy KIQ path or refuse - visible at boot in dmesg; both answers are informative) |
| `amdgpu.gpu_recovery=0` | stops the MODE2 resets; the P2.9e evidence is that the resets deepen the state (degree 1 -> 2); without recovery a hung job just stays hung (worse for usability, better for forensics - and `queue_preemption_timeout_ms` / `lockup_timeout` modulate it) |
| `amdgpu.noretry=0` (or on) | the GPUVM fault retry path is what the restore worker walks; flipping retry changes the race |
| `amdgpu.max_num_of_queues_per_device` / `hws_max_conc_proc` | lower the per-process queue budget - the faulting teardown fails at "evict queue 3"; fewer queues per session = less MES queue-table churn |

### Boot journal for the test sequence

| boot | change | probe | result |
|---|---|---|---|
| (fill in) | | | |

## P2.11: the BIOS changeset resolves the fault cluster (17:46 boot) - 10/10 clean, zero restore-worker activity

The user's BIOS pass (IOMMU on everywhere, Global C-states disabled, ErP/S5 deep sleep on,
UMA 16 -> 8 GiB, ReBAR/Above-4G left on, BIOS already latest, no amdgpu module params yet)
changed the machine state in ways the journal shows directly:

| machine fact | before | after |
|---|---|---|
| VRAM (UMA carve) | 16 GiB | 8 GiB |
| host RAM the OS sees | 45 GiB | **53 GiB** (the carve comes out of RAM) |
| usable RAM for the workload | ~29 GiB vs 41.1 GiB of demand (31.64 pinned + ~8 session + 1.5 dense) - **chronic oversubscription** | 45 GiB vs 41.1 GiB - **fits with headroom** |
| KFD heap accounting | 51.5 GiB of heaps on 45 GiB of RAM | 56 GiB on 53 GiB (the accounting stays virtual, the *pressure* is what moved) |
| IOMMU | BIOS off + cmdline `iommu=off amd_iommu=on` (contradictory) | `iommu=on amd_iommu=on`, consistent |

The result on this boot: **7 longfill sessions clean (34.2-34.9 tok/s) + 3 smokes
(arithmetic/python/marker) PASS, and zero `restore_userptr_worker` lines and zero MES lines
for the whole boot.**  Every earlier boot showed 4-5 restore passes even in *clean* sessions;
this boot's fault subsystem never ran.  That is a qualitative change, not a longer budget.

The mechanical suspect is the oversubscription: the restore worker exists to walk and restore
the GTT page tables of the 31.64 GiB userptr; with ~12 GiB more of reclaim pressure on every
boot, that walk raced itself (settles in 4-5 passes vs runs away at 19+).  The other three
variables (IOMMU consistency, C-states, ErP) are on the list too but have no mechanism as
direct.  **Confirmation plan: 2-3 more boots, each with 5+ loaded sessions.  If a future boot
regresses, bisect in this order: UMA 16 vs 8 (the pressure hypothesis first), then IOMMU,
then C-states, then ErP** - one variable per boot, probe first, record in the P2.10 table.

### The launch-cost hypothesis dies (same boot)

`STRATA_MMQ_GROUP` 16 / 32 / 64 on the same boot: **34.5 / 34.2 / 34.2 tok/s with identical
phase profiles** (gemm gu 12.7 s, gather phase 8.8 s, gemm dn 4.7 s).  Collapsing the ~5,200
product launches 4x moved nothing - launch count is not the wall.  The prefill's ~34 s is:
the expert FLOPs at APU rate (17.5 s ~ 152 GFLOPS int8 for the fixed FLOPs), the gather phase
at ~1/5 of the micro's standalone rate (8.8 s for 54.8 GB moved - the one remaining unexplained
in-engine cost), and the non-expert phases (7.4 s).  The remaining single lever is the gather
phase itself (batched gather kernel, and the IOMMU-on access path it now runs through).

### The config tradeoff, recorded

IOMMU-on + 8 GiB UMA prefill: 34.2-34.9 tok/s vs the 16 GiB/IOMMU-off boots' 37.5-39.1.
The ~7% cost buys a fault-free machine (and 8 GiB back for the rest of the box).  For daily
use the fault-free config wins; both are now characterized so the choice is informed.

Protocol status: the 4-run budget no longer applies (keep the probe + DEGRADED guard as
standard hygiene until the 2-3 boot confirmation is in, then retire the ritual language from
the run scripts).  The fault work's deliverable is the diagnosis (this section + P2.6/P2.8-9e)
and the bisect ladder; the P2 performance work now runs on a machine that holds a boot.

## P2.12: the community's APU sizing rules name the mechanism - and this boot's later sessions correct P2.11

A community APU effort reported the exact failure mode we chased for three days, in their
words: `hipMemGetInfo()` reports the large GPU-addressable GTT pool **without subtracting the
engine's ordinary CPU allocations** (including its host expert arena); treating that capacity as
independent VRAM oversubscribes system memory and invokes the OOM killer.  Their four rules:
(1) label GTT capacity as shared GPU memory, (2) do not add shared GPU memory to system RAM when
deciding which model fits, (3) cap automatic expert-cache sizing on currently available host
memory, (4) leave 4 GiB for the OS and request-time CPU work.

Mapped onto this machine, the conservation arithmetic explains the whole era: the box has
**64 GiB of physical RAM**.  16-GiB-UMA boots: 45 GiB visible to the OS; the engine put the
31.64 GiB arena and the ~8 GiB session (the GTT pool - host RAM, not the dedicated carve; the
KFD heap read 48 GiB) and the ~1.5 GiB dense weights into that same 45 GiB: 31.64 + 8 + 1.5 +
~8 OS = ~49-51 vs 45 - **oversubscribed 4-6 GiB**, living in the chronic-reclaim zone where the
restore-worker race lives (it walks the 31.64 GiB userptr's GTT tables; under reclaim those
pages move out from under the walk).  8-GiB-UMA boots: 53.5 GiB visible; the same demand leaves
~4 GiB of headroom - fits, and the race stops contending.  That is why the BIOS pass (which
among other things halved the carve) changed the fault behavior, and it makes the earlier
"budget of 4-5 sessions" read as the reclaim zone's tolerance, not a hardware budget.

The engine's own #403 safety check had the right structure (MemAvailable minus a headroom) but a
blind spot: the alias path used `min(user headroom, 2 GiB)`, and MemAvailable does not know the
run is about to take another ~8 GiB of session from the same pool.  Now (this commit): the alias
headroom is **14 GiB** (~8 session + ~1.5 dense + 4 OS reserve, the community rule 4),
`STRATA_IGPU_HEADROOM_GIB` overrides per box, and startup prints the shared-memory accounting
(host RAM available vs complement + session + headroom) so a misfit is diagnosable then instead
of as chronic reclaim later.  On a 45-GiB box the 31.64 GiB complement is now refused with a
clear message instead of OOM territory.  Rule 2 (the model-fit picker in setup) is the same
conservation law and goes there when setup touches the APU path.

**Correction to P2.11, measured after the fact:** this boot did NOT stay clean.  After the
7 longfills + 3 smokes, sessions ~11-12 (the P2.12 verification smokes) began emitting
**degenerate repetition** (the arithmetic smoke's `17*23+5` answer became a `#$%#*` token loop)
- the fault family's silent-corruption stage: wrong outputs, clean process, zero
restore/MES lines in the journal.  The pre-change binary fails identically (it is the machine,
not the build).  So the honest statement: the BIOS changeset **extended the boot budget from
4-5 sessions to ~10-11**, roughly 2.5x, and left a silent-corruption stage before the hang -
a real, large, but not complete fix.  Two usable artifacts: (a) the arithmetic smoke is a
**canary** - run it between experiments; a degenerate answer means the boot is spent, and a
clean power-off (no hung process to kill - the ErP deep-sleep path the user enabled) is safe
and sufficient from a cleanly-degraded state; (b) the bisect ladder from P2.11 still stands for
finding what extends the budget further (UMA first - the conservation math makes it the
mechanical suspect - then IOMMU, C-states, ErP).

## P2.12b: fresh-boot confirmation, the headroom corrected, and the mapping-mode matrix

Fresh boot (19:05, normal power-off - no hung process, the ErP path), canary clean (396),
longfill 35.0 tok/s, zero restore/MES lines.

**The headroom arithmetic corrected (measured, this boot):** a full run's fixed host-RAM demand
beyond the 31.64 GiB complement is ~7 GiB (MemAvailable floor 14.8 of 53.5 during the run -
process + the model file's page cache + working set).  The session state is 0.17 GiB at 4096
cells and lives in the 8 GiB **dedicated carve**, not host RAM - the "~8 GiB session from GTT"
that P2.12 put in the conservation sum was the old-era binary's number (the 0.1.36 engine's
session is much smaller).  So the alias headroom is 8 GiB (4 OS reserve + 4 working set, the
community rules), not 14, and the accounting line now says what it measures.  The era failures
re-read as: 45 GiB visible - 31.64 pinned = 13.4 GiB for process + page cache + working set, a
working margin inside which the kernel reclaimed continuously - chronic reclaim, not OOM.

**The mapping-mode matrix (the micro's gather arm, 31 GiB, 512 experts x 1.38 MB, per-expert
launches):**

| mapping | per-expert gather | flat copy of the same bytes |
|---|---|---|
| pinned (hipHostAlloc Mapped, no userptr) | 27.22 GB/s | 35.39 GB/s |
| registered userptr (malloc + hipHostRegister - the engine's complement state) | 22.12 GB/s | 29.49 GB/s |
| pinned + registered | impossible - hipHostAlloc(Mapped) is already mapped; re-registration is refused | - |

The "pinned+registered" corner does not exist: the engine's complement is a page-locked malloc
registered as ONE userptr range, so the `reg` row IS the engine's mapping state.  userptr
registration costs ~19% here - but the engine's in-gather per-expert cost is 221 us (6.2 GB/s),
3.5-4.5x slower than the micro running the SAME mapping, SAME kernel, SAME size.  Every
pattern-level variable is now exonerated (scatter, streams, sync rhythm, file-cache
coexistence, spin threads, launch count via MMQ_GROUP, and now mapping mode): the in-engine
per-launch cost is a property of the engine PROCESS's KFD state - the same subsystem the
restore-worker race lives in.  The micro never faults; the engine degrades after ~10-11
sessions.  The two symptoms (slow launches, fault cluster) point at one object.  Closing the
last gap (which part of the process state) would need the cost measured inside the engine,
which the DEGRADED guard and the session budget make expensive; it is parked as the open
question, with the working characterization: prefill 34-35 tok/s = expert FLOPs at APU rate
(17.5 s) + gather at ~1/4 micro rate (8.8 s, process-state-bound) + non-expert (7.4 s).

## P2.12c: the fault returns on the 5th session of the 19:05 boot - and the micro's register cycles may be spending the budget

19:05 boot (normal power-off, ErP): canary clean (396), longfill 35.0, micro gather matrix
(three 31 GiB register -> DMA -> unregister cycles), longfill 34.9, longfill 34.9, then
**FAULT on the 5th session** at layer 19 of 48: `unspecified launch failure` in the hipblaslt
gemm, the process hung (left alive, no kill).  Two journal firsts: the driver **attempted queue
eviction at the fault** and failed (`amdgpu: Failed to evict queue 3`,
`remove_all_kfd_queues_mes: Failed to remove queue 2`) - the first time the journal has caught
the eviction attempt itself rather than its aftermath - and zero
`restore_userptr_worker` lines again for the whole boot.  tok/s was flat to the fault
(35.0 / 34.9 / 34.9): no visible ramp in wall time.

The previous boot ran 10 engine sessions clean with no micro between them; this one faulted
after 4 engine sessions + 3 micro runs, each of which did a 31 GiB `hipHostRegister` ->
4 MiB-step DMA pass -> `hipHostUnregister` cycle.  The two boots' totals agree with the
accumulator counting **userptr lifecycle work** (registration, DMA through the mapping,
unregistration, the driver's internal restore passes) rather than "engine sessions": ~6-7
session-equivalents either way.  The micro remains perf-safe (its rates do not degrade), but
it is NOT budget-neutral.  Working rule until proven otherwise: count a 31 GiB micro
register cycle as roughly one heavy engine session; on a boot meant for N engine runs, keep
the big micro runs out of the middle.

State after this boot: the machine is faulted with a hung process - **AC-cut power cycle
(unplug/PSU switch + hold the power button ~10 s), not a normal shutdown** (P2.9e: shutting
down from a faulted state may carry MES state across the power-off).

## P2.12d: the no-micro boot holds 8 clean session-equivalents (20:31 boot)

AC-cut power cycle after the 19:05 fault, fresh boot 20:31 (iommu on, 8 GiB carve,
MemAvailable 51.8 GiB at start).  Canary clean (396), then **seven longfills, all clean**:
34.9 / 34.9 / 34.8 / 34.9 / 34.9 / 34.6 / 34.9 tok/s, no degradation across the boot.  The
whole-boot fault indicators are all zero — `restore_userptr_worker`: 0, `MES failed`: 0,
`Failed to evict`: 0 (the only MES lines are the normal boot-time `vmid_mask`/`gfx_hqd_mask`
init at 20:31:32).

Eight full userptr-lifecycle equivalents (the canary and every longfill each do the 31.64 GiB
complement's register -> DMA -> the run -> unregister), no micro interleaved, and the machine
is still clean.  That is beyond the 6-7 estimate from P2.12c and more than double the old
era's 4-5.  The working rule now, measured: **a boot with no 31 GiB micro register cycles in
the middle holds at least 8 clean heavy engine sessions.**  The fault is not gone - it is the
same driver race, and the micro's repeated register/unregister teardown is what spent the
19:05 boot early - but a normal experimental boot has a usable budget.  State at stop: clean,
no hung process, a normal shutdown (ErP deep sleep) is safe.

## P2.13: the dense bf16 projections ran 2.4-7.3x slower than their best BLAS solution - tuned, +3% prefill

Fresh-boot profile of the 33.4 s prompt (STRATA_PREFILL_TIMING, 1162 tokens): the non-expert
7.6 s is hc/hyper-connection reads 1.59 s (per layer, 2 halves x norm + down 10240->320 + silu
+ up 320->10240 + inject 4->10240, D=10240 channels), gdn projections 1.64, qsa proj 1.23,
gdn out proj 1.15, qsa attn 0.60, router 0.62, recurrence 0.36, combine 0.19.  The experts are
26.4 s (dequant/gather 8.8, gemm gu 12.8, gemm dn 4.7) - 80%.

The hc/router gemms go through `bf16_proj` (hipBLASLt bf16); the qkv/gate/ssm_out gemms go
through `gm.native` (the native GGUF blocks dequantized to an f16 scratch each call, then an
f16 BLAS gemm - the f16 shapes were already tuned).  The bf16 dense shapes were NOT in the
tuning file: `tune_hipblaslt --case bf16,512,N,K,ldy` shows hipBLASLt's default heuristic
picking algorithms 2.36x-7.33x slower than the best candidate for every one of them
(qkv-shape 10240x2560: 22.3 -> 4.08 ms, 5.5x; gate 6144x2560: 6.1x; ssm_out 2560x6144: 7.3x;
router/indexer-q 512x2560: 3.6x; indexer-k 128x2560: 4.5x; hc-up 10240x320: 2.4x; hc-inject
10240x4: best candidate is 0.94x, so it stays on the default).  Added the six winning rows
(bucket 512 covers the 512/138 chunks): hc read 1593 -> 893 ms, router 616 -> 523, prefill
34.1 -> 33.1 s (34.9 -> 35.1 tok/s), arithmetic smoke still 396.  The gdn/qsa-proj/gdn-out
phases did NOT move (they are the `gm.native` f16 path, already tuned) - their wall is the
f16 gemm FLOPs plus the per-call W dequant.  Caching the dequantized W across the 3 chunks
would need ~5.5 GiB of per-layer f16 slots - it does not fit the 8 GiB carve (it would fit
the 16 GiB config), so on this box the remaining dense lever is a fused dequant+gemm kernel,
same family as the expert gemm work.

## P2.14: the OpenAI/Anthropic server - one prompt failed 4/4, and it was the alias cache's loan arithmetic

The serve path (serve/server.py + the engine's `--serve` loop) was untested: the one-shot
path had all of P1/P2, but nothing had ever sent a request through the loop.  First attempt
answered ("396", decode 16-17 tok/s, drafts accepted 82%), then one prompt failed
deterministically - "Write one line of Python that prints the sum of the integers 1 through
50" -> `prefill: routed id out of range`, 4/4 repeats, while two other prompts (arithmetic,
a marker phrase) passed.  The engine exits after a failed request and the server restarts
it per request, so every "passing" request ran on a fresh engine: the failure was prompt-
specific, not process state.

The engine alone reproduces it (a `GEN <ids>` line piped into `strata --serve`, no Python).
Instrumenting the router guard: at layer 0 the whole T x K routed-id table is garbage, the
router's logits are zero, and its input (the hyper-connection mix of the layer-0 hidden
state) is zero - while the embedding, read moments earlier, is healthy (8152 nonzero in
row 0).  Poisoning the residual buffer with an index pattern after the embedding broadcast
and re-reading it at the router: **the entire 2.98 MB buffer has been zeroed**, cleanly,
from element 0.  Section-by-section syncs around layer 0 put the overwrite in the GDN
section; per-step syncs put the first reported error on the very first GDN GEMM's
dequant; and the kernel log names the fault: a write at scratch+1.32 MB,
`PERMISSION_FAULTS: 0x5` (page not present) - the first write past the end of the mapped
region (the expert blob there is 1.38 MB).

The prompt path's buffers in serve mode are not allocated: they are **borrowed** from the
end of the expert cache - the loan math (`part_slots`/`part_bytes`) was written for
sized-slot caches, whose `slot_offsets()` is an (n+1)-entry array of byte offsets (the
last entry is the total).  The alias cache (P1) returns its n-entry **host-pointer table**
from the same accessor: the loop reads one entry past the end (0), `part_slots` returns 1
unconditionally, and `part_bytes` computes `bytes() - <host VA>` - a wrapped 1.8e19.  The
"16 GiB loan" is thus the region **starting at the last expert blob and ending nowhere**:
the carve lays the 64 MB dequant scratch, the BLAS workspace, the token buffers and the
streaming ring on top of the last expert and ~400 MB past the end of the 31.64 GiB
complement.  The dequant's first 1.38 MB silently overwrites the last expert blob; its
next write is the IMA; and the residual/ids buffers that live past the complement's end
read back as whatever the fault left there (zeros).  Whether the out-of-range write hits a
mapped or an unmapped page depends on what the ASLR happened to put there - which is why
other prompts sometimes got through with the last expert's data quietly destroyed (the
silent "plausible tokens" mode the loan code warns about).  The one-shot path was already
guarded in P1 (`!xcache.aliases()` - "no cache slots to borrow"); the serve path was not,
because nothing had exercised it.

The fix is the one-line guard on the serve loan planning: an alias cache holds the expert
**data** in the region its slots name, so there is nothing to lend and the prompt path
takes its own buffers (a log line says so at start).  Verified on this box: the 74-token
prompt that failed 4/4 now answers `print(sum(range(1, 51)))` (drafts 47/56); arithmetic
"396" with the checkpoint machinery working (69 tokens = 64 reused + 5 read); and a
1425-token prompt read in three 512-token chunks at 38.5 tok/s with a correct summary.
The serve config (strata-igpu-serve.json) and the launch script (tools/hip/igpu_serve_boot.sh)
commit with this; the MTP runtime files (mtp-q2_0/rt, dense.bin + experts.bin) are
generated, not committed: `python3 tools/mtp_rt.py --gguf <mtp-q2_0.gguf> --out mtp-q2_0/rt`.

The serve config is a local file (strata-*.json is gitignored); on this box it is:

    {
      "exe": "./build-hip/strata",
      "args": ["--serve", "--pack", "packs/qwen38-flash-next-q2_0",
               "--native", "<model shard 1.gguf>",
               "--ple-gguf", "<model shard 2.gguf>",
               "--mmap-experts", "--expert-profile", "data/expert-profile.bin",
               "--expert-cache", "6000", "--prefill", "2048", "--spec", "4",
               "--spec-min-p", "0.5", "--mtp", "mtp-q2_0/rt",
               "--max-context", "4096", "--pool-workers", "8",
               "--adapt-every", "0", "--pcie-frac", "0", "--vram-reserve-mib", "1024"],
      "env": {"STRATA_IGPU_ALIAS": "1", "STRATA_GROUP_COPY": "1", "STRATA_SSD_KEEPALIVE": "0",
              "STRATA_HIPBLASLT_WARMUP": "1", "STRATA_HIPBLASLT_TUNING": "$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"},
      "tokenizer": "packs/qwen38-flash-next-q2_0/tokenizer",
      "port": 8095,
      "model_name": "qwen38-flash-next-q2_0",
      "context": 4096
    }

(the engine needs `--engine strata` on the server command line, or the server falls
back to a mock engine; the port is whatever the server finds free - it reports the
URL it bound in its READY line.  `--spec 4` is the MTP draft head's speculation, and
`--mtp mtp-q2_0/rt` the runtime files it reads.)

The draft head is worth keeping on this card, with one caveat measured on 2026-10-08: on real prose it
raises decode from 10.36 tok/s (suffix drafts only) to 13.76 (`--spec 4`, 2.37 tokens per round against
1.50), even though the draft layer is a whole extra layer costing 25 ms per round and 795 MiB. On text
whose continuation is already present in the context, the free suffix drafter predicts better than the
learned head, and enabling MTP suppresses it - there the same setting measures 8.28 against 16.16. The
numbers and the knob sweep are section 20 of docs/GFX1103.md.

The debugging session ran forty-one full userptr-lifecycle equivalents on the 20:31 boot
(the canary, the one-shot reproductions, the serve-direct bisection, the server requests -
each engine process does the 31.64 GiB register -> DMA -> run -> unregister) and the whole-
boot fault indicators stayed at zero.  That is five times the "at least 8" measured in
P2.12d under the same no-micro rule - on this boot the BIOS changeset's effect looks like
more than a budget extension, though the protocol is unchanged: the fault is a driver
race, the budget is empirical, and an AC cut remains the only reset from a faulted state.

## The 0.1.40 port (igpu-40 branch)

Upstream moved 1159 commits in four minor versions (0.1.36 -> 0.1.40), and most of it
touched the same files as this work: `generate.cpp` +5,247 lines, `prefill.cpp` +1,531,
`expert_source.cpp` +1,314.  A commit-by-commit rebase was not an option, so the work was
ported onto upstream/main as the branch `igpu-40`: each changed file applied with
`git apply --3way` (12 of 19 clean), the nine conflicts resolved by hand, and the iGPU
functional delta (about 2,000 lines) re-verified on the GPU.

The port needed exactly one real code fix that the three-way merge could not see: the
upstream `resident_blob()` ends at a line the conflict block did not include, so its
closing brace was missing when `register_complement_for_gpu` was re-inserted after it.
Everything else was merge work.

What did not port, deliberately: `mmq_group()` (P2.11 measured it as slower than the
group gather it replaced, and it collides with upstream's rewritten `group_gather`);
the P2.6 prompt-parallelism experiments (superseded); and the debug instrumentation,
which was already gone.  The P2.14 serve loan guard did port (the `!xcache.aliases()`
check around the loan planning), and the alias table now feeds upstream's new
`live_slots_` residency bookkeeping as well as the old `slots_` array.

Validation on the 780M (17:23-17:45, after a 20:49 uptime boot), `build-hip40`:

| run | result |
| --- | --- |
| Phase B smoke (arithmetic, python, marker, longfill) | 4/4 PASS |
| serve, the 74-token python prompt of P2.14 (batched loan path) | PASS, 16.3 tok/s, 100% expert-cache hit |
| serve, 92-token prompt, 400 generated tokens (3+ prefill chunks) | PASS, 11.9-14.5 tok/s |

One caution the port session produced: a strata engine left running between measurements
holds its 31.64 GiB complement pinned, and the next process's headroom accounting sees
that as unavailable RAM.  With such a server still up, a fresh run pinned a **0.00 GiB**
complement, the alias table fell back to the file mapping, and every run died on an
illegal memory access at the first expert read (the file-backed VMA faults on this
iGPU, P1).  The fix is operational, not code: stop the old server before starting the
engine you are about to measure.

One comparison this session could not finish: the 0.1.36 one-shot decode was
18.8-20.5 tok/s, the 0.1.40 longfill above 14.97.  The two upstream commits that
touch the decode expert kernels both ship A/B toggles (`STRATA_OLD_GROUPED=1`
reselects the pre-e155b07 Q2_0 grouped and per-hit kernels; `STRATA_S2_SWIGLU_Q8=0`
keeps the pre-3281ac3 two-launch SwiGLU+quantize), so a fresh boot needs only a
canary plus three short decode runs to attribute the gap.

**Measured on the fresh boot (Oct 6, 21:56-23:25, zero journal fault lines, six
lifecycles), the gap is not a code regression.**  Same short prompt (the 44-token
arithmetic smoke), same flags, three kernel variants plus the old binary:

| run | decode | draft acceptance |
| --- | --- | --- |
| 0.1.40, default kernels | 9.15 tok/s | 53/79 (67%) |
| 0.1.40, `STRATA_OLD_GROUPED=1` | 9.07 | 54/79 (68%) |
| 0.1.40, `STRATA_S2_SWIGLU_Q8=0` | 9.10 | 54/82 (66%) |
| 0.1.36 binary, same prompt | 9.56 | 54/80 |

And the apples-to-apples longfill on the same boot: 0.1.40 35.70 prefill / 15.19
decode, 0.1.36 35.46 / 15.30, both 77/80 accepted - within 1% of each other.  The
18.8-20.5 figure was a different measurement condition (higher acceptance), not a
faster kernel: the upstream kernel changes are neutral on this APU, which e155b07
itself predicts (its new grid is below one block per SM on small parts).

**Canary refinement:** the spent-boot signal is a loop *instead of the answer*.
A post-answer `<|im_start|>` repetition in one-shot greedy mode is model behavior -
it was present on clean boots too (the 17:28 arithmetic run answered 396 correctly,
then looped) - and does not invalidate the timing numbers, only the acceptance
numbers, because the loop rounds are real rounds the draft head mispredicts.
Judge the canary by the answer, not by the tail.

## P2.15: the dp4a wall, quantified (Oct 6, fresh boot, device-only arms, zero fault lines)

| micro arm | what it measures | result |
| --- | --- | --- |
| `--dp4a 4 pinned` | read Q2_0 blocks at the full GTT rate, one dp4a per 4-byte word | 82.6 GB/s, 147 GFLOPS |
| `--dp4a-peak 512 200000` | the same inner loop in registers, no memory at all | **4660 GFLOPS** |

The 147 is the read-bound floor for Q2_0 (32 FLOP per 18-byte block x 82.6 GB/s), not the peak -
the engine's gemm phase (280-380 GFLOPS, each blob reused across ~14 tokens per row-group) already
sits above it, so the engine is compute-bound: **6-8% of the measured dp4a peak**. The dequant
phase's 8.8 s is the 7x write amplification (64 fp16 values written per 18-byte block read) at
~50 GB/s combined - overhead the fused path avoids, which then pays the compute wall instead.
P2.7's "every expert path lands at ~39 tok/s" is these two paths meeting the same wall from
opposite sides.

The lever, with a number attached: an RDNA3-shaped tile that reaches 30-40% of the 4.66 TFLOPS
peak puts the expert phase at ~3-4 s instead of 17.9 s, and prefill at ~100 tok/s. The next test
is the tile itself: `STRATA_PF_BK64` / the 128x256 64-k tile kernel, untested on gfx1103.

Two env levers tried on the same boot (longfill, timing on): `STRATA_PF_GEMM=1` refuses this
pack's shapes (the padded-X case needs the PF_PAD path, unsupported for the i-quant dense
tensors), and `STRATA_WMMA_GEMM=1` is flat - 34.7 vs 34.65 tok/s.  The dense phase is already
near hardware; the expert gemm has no env knob - its tile is compile-time constants, so the
6-8%-of-peak gap is a code change, not a setting.

## P2.16: the tile search - the shape is not the wall (same boot, device-only arm, zero fault lines)

The engine's grouped gate/up pattern run on device memory (blobs copied once, no host read),
512 blocks x 200 reps:

| variant | GFLOPS |
| --- | --- |
| engine default (word-major LDS, 32 rows, 256 threads) | **1243** |
| byte-major (pre-e155b07) | 1167 |
| 64 rows, 256 thr | 1237 |
| 32 rows, 512 thr | 1167 |
| 16 rows, 128 thr | 1220 |

The shape reaches 1243 GFLOPS - 4x the engine's 306 - and every tile variant lands within 6% of
the default, so the inner loop is already near its ceiling.  The gemm phase reads only ~813 MB of
codes (one blob-row read per group), so it is not read-bound either.  The 4x gap is the engine's
execution structure around the kernel: per-group launches, activation staging, and the
dequant-then-BLAS dependency (the engine's gemm phase is hipBLASLt on dequantized fp16, not this
dp4a shape).  The next test is `STRATA_PF_FUSED=1` with timing on a fresh boot - the fused path is
where the 1243 should show up, and P2.7 measured it at the same ~39 tok/s wall, so the loss is in
how the fused path is driven, not in its math.
