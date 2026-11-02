"""Routed CSF scale expansion with persistent CTAs and batched group decode (GLM-5.3 decode).

Replaces the per-(route, task) grid of ``nvfp4_csf._Pair`` for compact block planes
(``CompactBlockStoragePlane``: native codes, 3-byte group records, MMA-oriented output,
shared exceptions), optionally fused with W4A16 route packing. The stock grid launches capacity x tasks tiny CTAs; each re-reads the
route list to reject duplicates and then walks a chain of dependent loads (record bounds ->
first record) for a single 512-byte scale group, so the expansion runs at well under half
of DRAM bandwidth.

Here a persistent grid first builds the unique expert list once per CTA (presence flags in
shared memory), then each warp decodes ``batch`` consecutive groups of one expert at a time
with every independent load (codes, row bases, record bounds) issued before any is consumed.
The per-group arithmetic, exception patching and stores are the stock
``CompactBlockStoragePlane`` methods, so the expanded bytes are identical. CTA 0 also resets
the fused MoE barrier scalars, as the stock kernel does.

With a pack target the last CTA instead performs the small-prefix + sort route packing of
``route_pack.py`` (expert counts, block-padded offsets, per-expert route lists, block-to-expert
map), removing those two launches. Route order within an expert follows shared-memory atomics,
as the Triton sort kernel's global atomics do; each route is still one row of its expert block.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import atomic_add_shared_i32, ld_shared_v4_u32, st_shared_v4_u32
from b12x._lib.program_cache import program_cache
from b12x._lib.quant.csf_blocks import CompactBlockStoragePlane
from b12x._lib.utils import current_cuda_stream, make_ptr

MAX_PACK_ROUTES = 1024


def supported(first_geometry, second_geometry) -> bool:
    """Compact block planes with 3-byte group records, MMA output and shared exceptions."""
    for g in (first_geometry, second_geometry):
        if len(g) != 11 or g[3] not in (8, 9) or g[2] != 4:
            return False
        plane = CompactBlockStoragePlane(g)
        # groups_per_task / shared_exceptions only steer the stock decoder; this kernel
        # batches groups itself and always stages exceptions in shared memory.
        if (
            plane.destination != 1
            or plane.sections != 1
            or plane.storage.rows % 128
            or plane.storage.columns % 4
        ):
            return False
    return first_geometry[6] == second_geometry[6]


class _FastPlane(CompactBlockStoragePlane):
    def __init__(self, geometry):
        super().__init__(geometry)
        self.shared_exceptions = True  # exceptions are patched in the shared-memory stage

    @cute.jit
    def decode_unit(self, storage, output, expert, group0, stage, warp, lane, batch: cutlass.Constexpr):
        value_offset = self.group_value_offset(storage)
        words = []
        firsts = []
        ends = []
        positions = []
        values = []
        # Issue every independent load of the batch before consuming any.
        for b in cutlass.range_constexpr(batch):
            words.append(self.group_words(storage, expert, group0 + Int32(b), lane))
            first, end = self.group_record_bounds(storage, expert, group0 + Int32(b))
            firsts.append(first)
            ends.append(end)
        for b in cutlass.range_constexpr(batch):
            position, value = Int32(0), cutlass.Uint8(0)
            entry = firsts[b] + lane.to(Uint32)
            if entry < ends[b]:
                position, value = self.load_group_record(storage, entry, value_offset)
            positions.append(position)
            values.append(value)
        targets = []
        for b in cutlass.range_constexpr(batch):
            target = cute.recast_ptr(stage, dtype=cutlass.Uint8) + (warp * Int32(batch) + Int32(b)) * Int32(512)
            targets.append(target)
            w = words[b]
            st_shared_v4_u32(target.toint().to(Int32) + lane * Int32(16), w[0], w[1], w[2], w[3])
        cute.arch.sync_warp()
        for b in cutlass.range_constexpr(batch):
            if firsts[b] + lane.to(Uint32) < ends[b]:
                self.write_group_record(group0 + Int32(b), targets[b], positions[b], values[b])
            # Groups with more than 32 records (none observed) take the stock strided loop.
            self.patch_group_records(storage, group0 + Int32(b), lane, targets[b], firsts[b] + Uint32(32), ends[b])
        cute.arch.sync_warp()
        expert_output = output + Int64(expert) * Int64(self.rows * self.columns)
        for b in cutlass.range_constexpr(batch):
            v0, v1, v2, v3 = ld_shared_v4_u32(targets[b].toint().to(Int32) + lane * Int32(16))
            self.store_group(expert_output, group0 + Int32(b), lane, v0, v1, v2, v3)
        cute.arch.sync_warp()


class _FastPair:
    def __init__(self, first, second, warps, batch):
        self.first = _FastPlane(first)
        self.second = _FastPlane(second)
        self.experts = int(first[6])
        self.warps, self.batch = int(warps), int(batch)
        self.threads = self.warps * 32
        self.padded_experts = (self.experts + 31) // 32 * 32
        for plane in (self.first, self.second):
            if plane.groups % self.batch:
                raise ValueError("CSF group count must be a multiple of the warp batch")
        self.units13 = self.first.groups // self.batch
        self.units2 = self.second.groups // self.batch

    @cute.jit
    def __call__(
        self,
        s13: cute.Pointer,
        o13: cute.Pointer,
        s2: cute.Pointer,
        o2: cute.Pointer,
        ids_ptr: cute.Pointer,
        routes: Int32,
        grid: Int32,
        barrier_count_ptr: cute.Pointer,
        barrier_epoch_ptr: cute.Pointer,
        barrier_slots: Int32,
        packed_ptr: cute.Pointer,
        blocks_ptr: cute.Pointer,
        count_ptr: cute.Pointer,
        offsets_ptr: cute.Pointer,
        counts_ptr: cute.Pointer,
        pack_block_size: Int32,
        max_packed: Int32,
        max_blocks: Int32,
        pack: Int32,
        stream: cuda.CUstream,
    ):
        ids = cute.make_tensor(ids_ptr, cute.make_layout((routes,)))
        barrier_count = cute.make_tensor(barrier_count_ptr, cute.make_layout((barrier_slots,)))
        barrier_epoch = cute.make_tensor(barrier_epoch_ptr, cute.make_layout((barrier_slots,)))
        packed = cute.make_tensor(packed_ptr, cute.make_layout((max_packed,)))
        blocks = cute.make_tensor(blocks_ptr, cute.make_layout((max_blocks,)))
        count = cute.make_tensor(count_ptr, cute.make_layout((1,)))
        offsets = cute.make_tensor(offsets_ptr, cute.make_layout((self.experts + 1,)))
        counts = cute.make_tensor(counts_ptr, cute.make_layout((self.experts,)))
        self.kernel(
            s13, o13, s2, o2, ids, routes, barrier_count, barrier_epoch, barrier_slots,
            packed, blocks, count, offsets, counts, pack_block_size, max_packed, max_blocks, pack,
        ).launch(
            grid=(grid, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.jit
    def _pack(self, ids, routes, packed, blocks, count, offsets, counts, block_size, max_packed, max_blocks,
              counts_smem, prefix, ranks, tid, warp, lane):
        e = Int32(tid)
        while e < Int32(self.padded_experts):
            counts_smem[e] = Int32(0)
            e += Int32(self.threads)
        cute.arch.sync_threads()
        base_addr = counts_smem.iterator.toint().to(Int32)
        r = Int32(tid)
        while r < routes:
            raw = ids[r].to(Int32)
            rank = Int32(-1)
            if raw >= Int32(0) and raw < Int32(self.experts):
                rank = atomic_add_shared_i32(base_addr + raw * Int32(4), Int32(1))
            ranks[r] = rank
            r += Int32(self.threads)
        cute.arch.sync_threads()
        if warp == Int32(0):
            carry = Int32(0)
            for c in cutlass.range_constexpr(self.padded_experts // 32):
                n = counts_smem[Int32(c * 32) + lane]
                padded = (n + block_size - Int32(1)) // block_size * block_size
                inclusive = padded
                for d in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(inclusive, 1 << d)
                    if lane >= Int32(1 << d):
                        inclusive += up
                prefix[Int32(c * 32) + lane] = carry + inclusive - padded
                carry += cute.arch.shuffle_sync(inclusive, 31)
            if lane == Int32(0):
                prefix[Int32(self.padded_experts)] = carry
        cute.arch.sync_threads()
        total = prefix[Int32(self.padded_experts)]
        e = Int32(tid)
        while e < Int32(self.experts):
            offsets[e] = prefix[e] + counts_smem[e]
            counts[e] = counts_smem[e]
            e += Int32(self.threads)
        if tid == Int32(0):
            offsets[Int32(self.experts)] = total
            count[0] = total
        i = Int32(tid)
        while i < max_packed:
            packed[i] = routes
            i += Int32(self.threads)
        cute.arch.sync_threads()
        r = Int32(tid)
        while r < routes:
            rank = ranks[r]
            if rank >= Int32(0):
                packed[prefix[ids[r].to(Int32)] + rank] = r
            r += Int32(self.threads)
        b = Int32(tid)
        while b < max_blocks:
            row = b * block_size
            owner = Int32(-1)
            if row < total:
                owner = Int32(0)
                for ee in range(1, self.experts):
                    if prefix[Int32(ee)] <= row:
                        owner += Int32(1)
            blocks[b] = owner
            b += Int32(self.threads)

    @cute.kernel
    def kernel(
        self, s13, o13, s2, o2, ids, routes: Int32, barrier_count, barrier_epoch, barrier_slots: Int32,
        packed, blocks, count, offsets, counts, pack_block_size: Int32, max_packed: Int32, max_blocks: Int32,
        pack: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        # A PDL-launched successor (the fused MoE) may start its prologue now; its
        # griddepcontrol.wait still covers this grid's completion and memory flush.
        cute.arch.griddepcontrol_launch_dependents()
        decode_grid = Int32(grid)
        if pack != Int32(0):
            decode_grid = Int32(grid) - Int32(1)
        warp, lane = tid // Int32(32), tid % Int32(32)
        smem = cutlass.utils.SmemAllocator()
        flags = smem.allocate_tensor(Int32, cute.make_layout(self.padded_experts), byte_alignment=16)
        unique = smem.allocate_tensor(Int32, cute.make_layout(self.padded_experts + 1), byte_alignment=16)
        stage = smem.allocate_tensor(Uint32, cute.make_layout(self.warps * self.batch * 128), byte_alignment=16)
        prefix = smem.allocate_tensor(Int32, cute.make_layout(self.padded_experts + 1), byte_alignment=16)
        ranks = smem.allocate_tensor(Int32, cute.make_layout(MAX_PACK_ROUTES), byte_alignment=16)

        if pack != Int32(0) and Int32(block) == decode_grid:
            self._pack(ids, routes, packed, blocks, count, offsets, counts, pack_block_size, max_packed,
                       max_blocks, flags, prefix, ranks, tid, warp, lane)

        if block == Int32(0):
            slot = Int32(tid)
            while slot < barrier_slots:
                barrier_count[slot] = Int32(0)
                barrier_epoch[slot] = Int32(0)
                slot += Int32(self.threads)
        i = Int32(tid)
        while i < Int32(self.padded_experts):
            flags[i] = Int32(0)
            i += Int32(self.threads)
        cute.arch.sync_threads()
        r = Int32(tid)
        while r < routes:
            raw = ids[r].to(Int32)
            if raw >= Int32(0) and raw < Int32(self.experts):
                flags[raw] = Int32(1)
            r += Int32(self.threads)
        cute.arch.sync_threads()
        if warp == Int32(0):
            base = Int32(0)
            for c in cutlass.range_constexpr(self.padded_experts // 32):
                present = flags[Int32(c * 32) + lane] != Int32(0)
                ballot = cute.arch.vote_ballot_sync(present)
                if present:
                    unique[base + cute.arch.popc(ballot & cute.arch.lanemask_lt())] = Int32(c * 32) + lane
                base += cute.arch.popc(ballot)
            if lane == Int32(0):
                unique[Int32(self.padded_experts)] = base
        cute.arch.sync_threads()

        per_expert = Int32(self.units13 + self.units2)
        total = unique[Int32(self.padded_experts)] * per_expert
        unit = Int32(block) * Int32(self.warps) + warp
        if Int32(block) >= decode_grid:
            unit = total
        stride = decode_grid * Int32(self.warps)
        while unit < total:
            slot = unit // per_expert
            local = unit - slot * per_expert
            expert = unique[slot]
            if local < Int32(self.units13):
                self.first.decode_unit(s13, o13, expert, local * Int32(self.batch), stage.iterator, warp, lane, self.batch)
            else:
                self.second.decode_unit(
                    s2, o2, expert, (local - Int32(self.units13)) * Int32(self.batch),
                    stage.iterator, warp, lane, self.batch,
                )
            unit += stride


@program_cache
def compile_fast_pair(first, second, warps=8, batch=4):
    launch = _FastPair(first, second, warps, batch)
    key = (tuple(first), tuple(second), int(warps), int(batch))
    gmem8 = lambda: make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    i32 = lambda: make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    return b12x_compile(
        launch,
        gmem8(), gmem8(), gmem8(), gmem8(),
        i32(), 1, 1, i32(), i32(), 0,
        i32(), i32(), i32(), i32(), i32(), 1, 1, 1, 0,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.nvfp4_csf_fast_pair", 2, key),
    )


def launch_fast_pair(program, first, second, ids, out13, out2, barriers, grid, pack=None, pack_only=False):
    """``pack``: dict(packed, blocks, count, offsets, counts, block_size) runs route packing too;
    ``pack_only`` launches just that CTA (the MoE decodes the scales itself)."""
    if pack_only:
        grid = 0

    def ptr(t, dtype, align):
        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

    count = barriers[0] if barriers is not None else ids
    epoch = barriers[1] if barriers is not None else ids
    if pack is not None and ids.numel() > MAX_PACK_ROUTES:
        raise ValueError("fused route packing supports at most %d routes" % MAX_PACK_ROUTES)
    p = pack or {}
    dummy = ptr(ids, cutlass.Int32, 4)
    program(
        ptr(first.fixed, cutlass.Uint8, 16),
        ptr(out13, cutlass.Uint8, 16),
        ptr(second.fixed, cutlass.Uint8, 16),
        ptr(out2, cutlass.Uint8, 16),
        ptr(ids, cutlass.Int32, 4),
        int(ids.numel()),
        int(grid) + (1 if pack is not None else 0),
        ptr(count, cutlass.Int32, 4),
        ptr(epoch, cutlass.Int32, 4),
        0 if barriers is None else int(barriers[0].numel()),
        ptr(p["packed"], cutlass.Int32, 4) if p else dummy,
        ptr(p["blocks"], cutlass.Int32, 4) if p else dummy,
        ptr(p["count"], cutlass.Int32, 4) if p else dummy,
        ptr(p["offsets"], cutlass.Int32, 4) if p else dummy,
        ptr(p["counts"], cutlass.Int32, 4) if p else dummy,
        int(p.get("block_size", 1)),
        int(p["packed"].numel()) if p else 1,
        int(p["blocks"].numel()) if p else 1,
        1 if p else 0,
        current_cuda_stream(),
    )
