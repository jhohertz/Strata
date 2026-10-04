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
measured 68.8) was rebuilt in a worktree (`/home/jhohertz/co/Strata-029`,
pack and profile symlinked, `STRATA_PREFILL_TIMING=1` available there too)
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

**Every expert compute path lands at ~39 tok/s.**  The total prefill is not
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
