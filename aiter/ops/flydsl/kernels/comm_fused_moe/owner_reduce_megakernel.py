# SPDX-License-Identifier: Apache-2.0
"""Single-launch Stage2 producer/consumer kernel.

Producer workgroups run the natural-grid route GEMM. Completion is tracked per
N tile; the final ``service_groups`` producers become communication consumers
and execute the selected direct, reduce-broadcast, or reduce-scatter/all-gather
path. This keeps the native GEMM tiling and avoids an artificial window.
"""

import functools
import hashlib
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemPtr

from .. import buffer_ops, vector
from .. import communication_ops_utils as comm_ops
from ..mixed_moe_gemm_2stage_common import compile_mixed_moe_gemm2_common
from .collectives import (
    decode_scaled_fp8_f32,
    load_fp8_words,
    pack_fp8_words,
    store_fp8_words,
)
from .sync import peer_base


H = 7168
I = 384
E = 384
TOPK = 6
TP = 8
BLOCK = 256
SLOTS = 2
PRODUCER_COUNTER_STRIDE = 64


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _byte_ptr(addr):
    pointer = fx.PointerType.get(
        fx.Uint8.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=1,
    )
    return fx.inttoptr(pointer, fx.Int64(addr))


@dataclass(frozen=True)
class _RouteOutputEpilogue:
    """Owner-only route publication kept outside the shared GEMM builder."""

    fp8_fixed: bool
    device_coherent: bool

    @property
    def row_bytes(self) -> int:
        return H if self.fp8_fixed else H * 2

    def cshuffle_layout(self, tile_n: int) -> tuple[int, int]:
        if self.fp8_fixed and self.device_coherent and tile_n % 256 == 0:
            return 16, 16
        return min(tile_n // 32, 8), 32

    def store(
        self,
        *,
        row_ctx,
        col_g0,
        frag,
        e_vec: int,
        idx_to_llvm_ptr,
    ):
        _, row_byte_base, _ = row_ctx
        element_bytes = 1 if self.fp8_fixed else 2
        output_address = (
            row_byte_base + col_g0 * arith.constant(element_bytes, index=True)
        )
        if not self.fp8_fixed:
            output_pointer = idx_to_llvm_ptr(output_address)
            if self.device_coherent:
                if e_vec * element_bytes != 16:
                    raise RuntimeError(
                        "coherent BF16 route store requires one 16-byte vector"
                    )
                packed = vector.bitcast(T.vec(4, T.i32), frag)
                llvm_d.InlineAsmOp(
                    None,
                    [output_pointer, packed],
                    "global_store_dwordx4 $0, $1, off sc1",
                    "v,v",
                    has_side_effects=True,
                )
            else:
                raw = frag._value if hasattr(frag, "_value") else frag
                llvm_d.StoreOp(
                    raw,
                    output_pointer,
                    alignment=e_vec * element_bytes,
                    nontemporal=True,
                )
            return

        fragment = fx.Vector(frag)
        values = [
            fragment[element].to(fx.Float32)
            for element in range_constexpr(e_vec)
        ]
        packed_words = []
        for word in range_constexpr(e_vec // 4):
            element = word * 4
            packed = fx.Int32(0)
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                values[element],
                values[element + 1],
                packed,
                0,
            )
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                values[element + 2],
                values[element + 3],
                packed,
                1,
            )
            raw = packed._value if hasattr(packed, "_value") else packed
            packed_words.append(raw)
            if not self.device_coherent:
                llvm_d.StoreOp(
                    raw,
                    idx_to_llvm_ptr(
                        output_address + arith.constant(word * 4, index=True)
                    ),
                    alignment=4,
                    nontemporal=True,
                )

        if self.device_coherent:
            word_count = e_vec // 4
            if word_count not in (1, 2, 4):
                raise RuntimeError(
                    "coherent FP8 route store requires 1, 2, or 4 dwords"
                )
            packed_store = (
                packed_words[0]
                if word_count == 1
                else vector.from_elements(T.vec(word_count, T.i32), packed_words)
            )
            store_suffix = "" if word_count == 1 else f"x{word_count}"
            llvm_d.InlineAsmOp(
                None,
                [idx_to_llvm_ptr(output_address), packed_store],
                f"global_store_dword{store_suffix} $0, $1, off sc1",
                "v,v",
                has_side_effects=True,
            )


@dataclass(frozen=True)
class _OwnerGemm2ComposeConfig:
    block_threads: int
    persistent_groups: int | None
    output_epilogue: _RouteOutputEpilogue | None
    b_cache_modifier: int

    @staticmethod
    def emit_final_sync(iteration, one, iteration_count):
        final_iteration = scf.IfOp(
            arith.cmpi(
                CmpIPredicate.eq,
                iteration + one,
                iteration_count,
            )
        )
        with ir.InsertionPoint(final_iteration.then_block):
            fx.rocdl.s_waitcnt(0)
            scf.YieldOp([])


def _load_bf16(resource, offset, vector_width, cache_modifier):
    values = []
    for chunk in range_constexpr(vector_width // 8):
        loaded = fx.Vector(
            buffer_ops.buffer_load(
                resource,
                offset + fx.Int32(chunk * 8),
                vec_width=8,
                dtype=T.bf16,
                cache_modifier=cache_modifier,
            )
        )
        values.extend(loaded[element] for element in range_constexpr(8))
    return fx.Vector.from_elements(values, fx.BFloat16)


def _decode_unscaled_fp8_f32(words):
    """Decode packed FP8 directly to FP32 when no E8M0 scale is needed."""

    values = []
    for word in range_constexpr(len(words)):
        for half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_pk_f32_fp8(
                    T.vec(2, T.f32), words[word], bool(half)
                )
            )
            values.extend((pair[0], pair[1]))
    return values


def _decode_scaled_fp8_bf16(words, scale):
    """Decode packed FP8 directly to the BF16 final-output representation."""

    values = []
    for word in range_constexpr(len(words)):
        for half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16),
                    arith.unwrap(words[word]),
                    arith.unwrap(scale),
                    bool(half),
                )
            )
            values.extend((pair[0], pair[1]))
    return values


@dataclass(frozen=True)
class OwnerMegaKernelConfig:
    """Configuration for the single-launch owner producer/consumer kernel."""

    m: int
    tile_m: int = 16
    tile_n: int = 256
    tile_k: int = 128
    sort_block_m: int = 32
    compute_groups: int = 96
    block_threads: int = BLOCK
    vector_width: int = 16
    waves_per_eu: int = 0
    b_cache_modifier: int = 0
    route_store_scope: str = "device"
    local_load_cache_modifier: int = 1
    remote_load_cache_modifier: int = 1
    gather_load_cache_modifier: int = -1
    remote_store_cache_modifier: int = 0
    fp8_scale_exponent: int = 127
    n_tile_cohort: int = 0
    collective: str = "direct"
    service_groups: int = 1
    service_tile_group: int = 1
    producer_mode: str = "routes"
    flat_producer_grid: bool = False

    def __post_init__(self):
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            raise ValueError(
                "tile sizes must be positive, got "
                f"{(self.tile_m, self.tile_n, self.tile_k)}"
            )
        if H % self.tile_n:
            raise ValueError(f"H={H} must be divisible by tile_n={self.tile_n}")
        if self.sort_block_m % self.tile_m:
            raise ValueError(
                "sort_block_m must be divisible by tile_m, got "
                f"sort_block_m={self.sort_block_m}, tile_m={self.tile_m}"
            )
        if self.tile_n % self.vector_width:
            raise ValueError(
                "tile_n must be divisible by vector_width, got "
                f"tile_n={self.tile_n}, vector_width={self.vector_width}"
            )
        if self.b_cache_modifier not in (0, 1, 2, 3):
            raise ValueError(
                "b_cache_modifier must be one of (0, 1, 2, 3), got "
                f"{self.b_cache_modifier}"
            )
        if not 0 <= self.waves_per_eu <= 10:
            raise ValueError(
                f"waves_per_eu must be in [0, 10], got {self.waves_per_eu}"
            )
        if self.compute_groups <= 0:
            raise ValueError(
                f"compute_groups must be positive, got {self.compute_groups}"
            )
        if self.block_threads not in (128, 256):
            raise ValueError(
                f"block_threads must be 128 or 256, got {self.block_threads}"
            )
        num_waves = self.block_threads // 64
        if self.tile_n % (num_waves * 16):
            raise ValueError(
                "tile_n must provide an integral number of 16-column MFMA "
                "tiles per wave, got "
                f"tile_n={self.tile_n}, block_threads={self.block_threads}"
            )
        if self.route_store_scope not in ("default", "device"):
            raise ValueError(
                "route_store_scope must be 'default' or 'device', got "
                f"{self.route_store_scope!r}"
            )
        if self.local_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("local_load_cache_modifier must be in [0, 3]")
        if self.remote_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("remote_load_cache_modifier must be in [0, 3]")
        if self.gather_load_cache_modifier not in (-1, 0, 1, 2, 3):
            raise ValueError(
                "gather_load_cache_modifier must be -1 or in [0, 3]"
            )
        if self.remote_store_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("remote_store_cache_modifier must be in [0, 3]")
        if self.vector_width not in (8, 16):
            raise ValueError(
                "production owner-mega requires vector_width 8 or 16"
            )
        if not 0 <= self.fp8_scale_exponent <= 254:
            raise ValueError(
                "fp8_scale_exponent must be in [0, 254], got "
                f"{self.fp8_scale_exponent}"
            )
        if self.n_tile_cohort < 0:
            raise ValueError(
                f"n_tile_cohort must be non-negative, got {self.n_tile_cohort}"
            )
        if self.n_tile_cohort and self.n_tiles % self.n_tile_cohort:
            raise ValueError(
                "n_tile_cohort must divide n_tiles, got "
                f"n_tile_cohort={self.n_tile_cohort}, n_tiles={self.n_tiles}"
            )
        if self.collective not in (
            "direct",
            "rsag",
            "rs_broadcast",
        ):
            raise ValueError(
                "collective must be 'direct', 'rsag', or 'rs_broadcast', got "
                f"{self.collective!r}"
            )
        if not 1 <= self.service_groups <= self.compute_groups:
            raise ValueError(
                "service_groups must be in [1, compute_groups], got "
                f"service_groups={self.service_groups}, "
                f"compute_groups={self.compute_groups}"
            )
        if self.collective != "rsag" and self.service_groups != 1:
            raise ValueError(
                "direct and rs_broadcast collectives require service_groups=1"
            )
        if (
            self.collective == "rsag"
            and self.service_groups not in (1, 4, 8)
        ):
            raise ValueError(
                "production rsag service_groups must be 1, 4, or 8"
            )
        if self.service_tile_group <= 0 or self.n_tiles % self.service_tile_group:
            raise ValueError(
                "service_tile_group must be a positive divisor of n_tiles, got "
                f"service_tile_group={self.service_tile_group}, "
                f"n_tiles={self.n_tiles}"
            )
        if self.service_tile_group > 1 and (
            self.collective != "rsag"
            or self.service_groups == 1
        ):
            raise ValueError(
                "grouped service synchronization requires collective='rsag' "
                "and service_groups > 1"
            )
        if self.producer_mode not in (
            "routes",
            "routes_fp8_fixed",
            "atomic_shared",
        ):
            raise ValueError(
                "producer_mode must be 'routes', 'routes_fp8_fixed', or "
                "'atomic_shared', got "
                f"{self.producer_mode!r}"
            )
        if self.flat_producer_grid and self.collective == "direct":
            raise ValueError(
                "flat_producer_grid requires a dynamic collective path"
            )
        if self.flat_producer_grid and self.n_tile_cohort:
            raise ValueError(
                "flat_producer_grid and n_tile_cohort are mutually exclusive"
            )
        if self.collective == "direct" and self.producer_mode != "routes":
            raise ValueError(
                "direct production path requires producer_mode='routes'"
            )
        if (
            self.collective == "rs_broadcast"
            and self.producer_mode != "atomic_shared"
        ):
            raise ValueError(
                "rs_broadcast production path requires "
                "producer_mode='atomic_shared'"
            )
        if (
            self.uses_rsag
            and (
                self.m
                * self.tile_n
                // self.vector_width
            )
            % TP
        ):
            raise ValueError("rsag vector items must divide evenly across TP ranks")

    @property
    def n_tiles(self) -> int:
        return H // self.tile_n

    @property
    def uses_rsag(self) -> bool:
        return self.collective in ("rsag", "rs_broadcast")

    @property
    def shared_bf16_partials(self) -> bool:
        return self.collective == "rs_broadcast"

    @property
    def single_pass_direct(self) -> bool:
        return bool(
            self.collective == "direct"
            and self.m * self.tile_n // self.vector_width <= self.block_threads
        )

    @property
    def producer_rows(self) -> int:
        route_rows = self.m * TOPK
        # Sorting pads every non-empty expert independently to sort_block_m.
        # Account for multiple sort blocks per expert; the previous E-only
        # cap was valid only while no expert crossed the first block.
        max_sort_blocks = (
            route_rows
            if route_rows <= E
            else E + (route_rows - E) // self.sort_block_m
        )
        return max_sort_blocks * self.sort_block_m // self.tile_m

    @property
    def payload_bytes(self) -> int:
        return self.m * H * 2

    @property
    def partial_bytes(self) -> int:
        return self.m * H

    @property
    def reduced_shard_bytes(self) -> int:
        if not self.uses_rsag:
            return 0
        element_bytes = 2 if self.collective == "rs_broadcast" else 1
        return self.m * H * element_bytes // TP

    @property
    def reduced_offset(self) -> int:
        return SLOTS * self.partial_bytes

    @property
    def route_offset(self) -> int:
        return self.reduced_offset + SLOTS * self.reduced_shard_bytes

    @property
    def route_bytes(self) -> int:
        if self.producer_mode == "routes":
            return self.m * TOPK * H * 2
        if self.producer_mode == "routes_fp8_fixed":
            return self.m * TOPK * H
        return self.payload_bytes

    @property
    def output_offset(self) -> int:
        if self.producer_mode in ("routes", "routes_fp8_fixed"):
            return self.route_offset + self.route_bytes
        return self.route_offset

    @property
    def producer_done_offset(self) -> int:
        return self.output_offset + self.payload_bytes

    @property
    def epoch_offset(self) -> int:
        return self.gather_service_done_offset + self.n_tiles * 8

    @property
    def service_done_offset(self) -> int:
        return (
            self.producer_done_offset
            + self.n_tiles * PRODUCER_COUNTER_STRIDE
        )

    @property
    def reduce_done_offset(self) -> int:
        return self.service_done_offset + self.n_tiles * 8

    @property
    def gather_service_done_offset(self) -> int:
        return self.reduce_done_offset + self.n_tiles * 8

    @property
    def rank_ready_offset(self) -> int:
        return self.epoch_offset + self.n_tiles * 8

    @property
    def flat_base_offset(self) -> int:
        return _align_up(
            self.gather_done_offset
            + (
                self.n_tiles * TP * 4
                if self.uses_rsag
                else 0
            ),
            8,
        )

    @property
    def gather_done_offset(self) -> int:
        return self.owner_ready_offset + (
            self.n_tiles * TP * 4
            if self.uses_rsag
            else self.n_tiles * 4
        )

    @property
    def owner_ready_offset(self) -> int:
        return self.reduced_collective_ready_offset + self.n_tiles * 4

    @property
    def reduced_collective_ready_offset(self) -> int:
        return self.collective_ready_offset + self.n_tiles * 4

    @property
    def collective_ready_offset(self) -> int:
        return self.rank_ready_offset + self.n_tiles * TP * 4

    @property
    def workspace_bytes(self) -> int:
        return _align_up(self.flat_base_offset + 8, 256)


def _atomic_add_i32_agent(addr, value):
    return llvm_d.AtomicRMWOp(
        llvm_d.AtomicBinOp.add,
        comm_ops._to_ptr_global(addr),
        arith.unwrap(value),
        llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    ).res


def _wait_i32_system_until_at_least(addr, expected):
    def load():
        return llvm_d.LoadOp(
            ir.IntegerType.get_signless(32),
            comm_ops._to_ptr_global(addr),
            alignment=4,
            volatile_=True,
            ordering=llvm_d.AtomicOrdering.monotonic,
            syncscope=fx.rocdl.SyncScope.OneAs,
        ).result

    loop = scf.WhileOp([T.i32], [load()])
    before = ir.Block.create_at_start(loop.before, [T.i32])
    after = ir.Block.create_at_start(loop.after, [T.i32])
    with ir.InsertionPoint(before):
        current = before.arguments[0]
        waiting = arith.CmpIOp(
            arith.CmpIPredicate.slt, current, arith.unwrap(expected)
        ).result
        scf.ConditionOp(waiting, [current])
    with ir.InsertionPoint(after):
        llvm_d.InlineAsmOp(None, [], "s_sleep 1", "", has_side_effects=True)
        scf.YieldOp([load()])
    return loop.results[0]


def _wait_i32_agent_until_at_least(addr, expected, *, sleep=True):
    def load():
        return llvm_d.LoadOp(
            ir.IntegerType.get_signless(32),
            comm_ops._to_ptr_global(addr),
            alignment=4,
            volatile_=True,
            ordering=llvm_d.AtomicOrdering.monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        ).result

    loop = scf.WhileOp([T.i32], [load()])
    before = ir.Block.create_at_start(loop.before, [T.i32])
    after = ir.Block.create_at_start(loop.after, [T.i32])
    with ir.InsertionPoint(before):
        current = before.arguments[0]
        waiting = arith.CmpIOp(
            arith.CmpIPredicate.slt, current, arith.unwrap(expected)
        ).result
        scf.ConditionOp(waiting, [current])
    with ir.InsertionPoint(after):
        if sleep:
            llvm_d.InlineAsmOp(None, [], "s_sleep 1", "", has_side_effects=True)
        scf.YieldOp([load()])
    return loop.results[0]


def _store_i32_relaxed(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
    )


def _store_i32_agent_release(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.release,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def _store_i32_system_monotonic(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _store_i32_system_release(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _store_i64_relaxed(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=8,
    )


def _store_bf16(resource, offset, values, vector_width, cache_modifier=0):
    if vector_width == 8:
        buffer_ops.buffer_store(
            values, resource, offset, cache_modifier=cache_modifier
        )
        return
    for chunk in range_constexpr(vector_width // 8):
        chunk_values = fx.Vector.from_elements(
            [values[chunk * 8 + element] for element in range_constexpr(8)],
            fx.BFloat16,
        )
        buffer_ops.buffer_store(
            chunk_values,
            resource,
            offset + fx.Int32(chunk * 8),
            cache_modifier=cache_modifier,
        )


def _emit_direct_allreduce_service_tile(
    config,
    workspace,
    workspace_flat_base,
    shared_partial,
    shared_partial_flat_base,
    specialized_rank,
    n_tile,
    tid,
    service_group,
    service_smem_base,
):
    rank = fx.Int32(specialized_rank)
    payload_bytes = config.payload_bytes
    partial_bytes = config.partial_bytes
    reduce_items = config.m * config.tile_n // config.vector_width
    local_workspace_base = fx.Int64(ptrtoint(workspace))
    state_n_tile = (
        n_tile // fx.Int32(config.service_tile_group)
    ) * fx.Int32(config.service_tile_group)
    tile_byte_offset = fx.Int64(state_n_tile) * fx.Int64(8)
    epoch_address = (
        local_workspace_base
        + fx.Int64(config.epoch_offset)
        + tile_byte_offset
    )
    expected = fx.Int64(comm_ops.load_i64_global(epoch_address)) + fx.Int64(1)
    expected_i32 = fx.Int32(expected)
    slot = expected & fx.Int64(1)

    route_resource = buffer_ops.create_buffer_resource_from_addr(
        local_workspace_base + fx.Int64(config.route_offset),
        num_records_bytes=config.route_bytes,
    )
    output_resource = buffer_ops.create_buffer_resource_from_addr(
        local_workspace_base + fx.Int64(config.output_offset),
        num_records_bytes=payload_bytes,
    )
    shared_resource = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(ptrtoint(shared_partial)),
        num_records_bytes=payload_bytes,
    )
    if config.producer_mode == "atomic_shared":
        producer_resource = shared_resource
    else:
        producer_resource = route_resource
    partial_resource = buffer_ops.create_buffer_resource_from_addr(
        local_workspace_base + slot * fx.Int64(partial_bytes),
        num_records_bytes=partial_bytes,
    )
    if config.collective == "rsag":
        reduced_resource = buffer_ops.create_buffer_resource_from_addr(
            local_workspace_base
            + fx.Int64(config.reduced_offset)
            + slot * fx.Int64(config.reduced_shard_bytes),
            num_records_bytes=config.reduced_shard_bytes,
        )
    service_stride = config.block_threads * config.service_groups
    service_start = tid + service_group * fx.Int32(config.block_threads)
    retain_local_partials = (
        config.collective == "direct"
        and
        reduce_items % service_stride == 0
        and reduce_items <= 4 * service_stride
    )
    if config.uses_rsag:
        reuse_waiter = scf.IfOp(
            arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(TP))
        )
        with ir.InsertionPoint(reuse_waiter.then_block):
            gather_slot = state_n_tile * fx.Int32(TP) + tid
            _wait_i32_system_until_at_least(
                local_workspace_base
                + fx.Int64(config.gather_done_offset)
                + fx.Int64(gather_slot) * fx.Int64(4),
                expected_i32 - fx.Int32(SLOTS),
            )
            scf.YieldOp([])
        gpu.barrier()

    def emit_local_reduce_item(item):
        token = item // fx.Int32(config.tile_n // config.vector_width)
        tile_item = item - token * fx.Int32(
            config.tile_n // config.vector_width
        )
        output_offset = (
            token * fx.Int32(H)
            + n_tile * fx.Int32(config.tile_n)
            + tile_item * fx.Int32(config.vector_width)
        )
        if config.producer_mode == "atomic_shared":
            reduced_f32 = _load_bf16(
                producer_resource,
                output_offset,
                config.vector_width,
                config.local_load_cache_modifier,
            ).extf(T.vec(config.vector_width, T.f32))
        else:
            shared_values = _load_bf16(
                shared_resource,
                output_offset,
                config.vector_width,
                config.local_load_cache_modifier,
            ).extf(T.vec(config.vector_width, T.f32))

            def load_bf16_route(route_slot):
                route_offset = (
                    (token * fx.Int32(TOPK) + fx.Int32(route_slot))
                    * fx.Int32(H)
                    + n_tile * fx.Int32(config.tile_n)
                    + tile_item * fx.Int32(config.vector_width)
                )
                return _load_bf16(
                    producer_resource,
                    route_offset,
                    config.vector_width,
                    config.local_load_cache_modifier,
                ).extf(T.vec(config.vector_width, T.f32))

            def load_fp8_route(route_slot):
                column = (
                    n_tile * fx.Int32(config.tile_n)
                    + tile_item * fx.Int32(config.vector_width)
                )
                route_row_offset = (
                    token * fx.Int32(TOPK) + fx.Int32(route_slot)
                ) * fx.Int32(H)
                values = []
                for chunk in range_constexpr(config.vector_width // 8):
                    chunk_column = column + fx.Int32(chunk * 8)
                    words = load_fp8_words(
                        producer_resource,
                        (route_row_offset + chunk_column) // fx.Int32(4),
                        word_count=2,
                        load_width=2,
                        cache_modifier=config.local_load_cache_modifier,
                    )
                    values.extend(_decode_unscaled_fp8_f32(words))
                return fx.Vector.from_elements(values, fx.Float32)

            def load_route(route_slot):
                if const_expr(config.producer_mode == "routes_fp8_fixed"):
                    return load_fp8_route(route_slot)
                return load_bf16_route(route_slot)

            # Keep only two FP32 accumulation vectors live while decoding the
            # six route contributions. Materializing every route at once raises
            # VGPR pressure enough to reduce service-workgroup residency.
            local_even = shared_values + load_route(0)
            local_odd = load_route(1) + load_route(2)
            local_even = local_even + load_route(3)
            local_odd = local_odd + load_route(4)
            reduced_f32 = local_even + (local_odd + load_route(5))

        quant_scale = fx.Int32(
            (254 - config.fp8_scale_exponent) << 23
        ).bitcast(fx.Float32)
        packed_words = config.vector_width // 4
        packed = pack_fp8_words(reduced_f32, quant_scale, packed_words)
        store_fp8_words(partial_resource, output_offset, packed, packed_words)
        return packed

    retained_local_partials = []
    def emit_local_reduce_items():
        if retain_local_partials:
            for iteration in range_constexpr(reduce_items // service_stride):
                item = service_start + fx.Int32(iteration * service_stride)
                retained_local_partials.append(emit_local_reduce_item(item))
        else:
            local_reduce_loop = scf.ForOp(
                arith.index_cast(T.index, service_start),
                arith.constant(reduce_items, index=True),
                arith.constant(service_stride, index=True),
            )
            with ir.InsertionPoint(local_reduce_loop.body):
                emit_local_reduce_item(
                    arith.index_cast(T.i32, local_reduce_loop.induction_variable)
                )
                scf.YieldOp([])
        fx.rocdl.s_waitcnt(0)
        gpu.barrier()

    if not config.shared_bf16_partials:
        emit_local_reduce_items()

    def load_reduced_partials(
        offset,
        retained_local_partial=None,
        source_rotation=None,
        vector_width=None,
    ):
        load_vector_width = vector_width or config.vector_width

        def load_peer(peer, cache_modifier=None):
            if config.shared_bf16_partials:
                peer_resource = buffer_ops.create_buffer_resource_from_addr(
                    peer_base(shared_partial_flat_base, peer),
                    num_records_bytes=payload_bytes,
                )
                return _load_bf16(
                    peer_resource,
                    offset,
                    load_vector_width,
                    cache_modifier
                    if cache_modifier is not None
                    else (
                        config.remote_load_cache_modifier
                        if peer != specialized_rank
                        else config.local_load_cache_modifier
                    ),
                ).extf(T.vec(load_vector_width, T.f32))
            if const_expr(retain_local_partials and peer == specialized_rank):
                scale = fx.Int32(
                    config.fp8_scale_exponent << 23
                ).bitcast(fx.Float32)
                return fx.Vector.from_elements(
                    decode_scaled_fp8_f32(retained_local_partial, scale),
                    fx.Float32,
                )
            load_offset = offset
            peer_resource = buffer_ops.create_buffer_resource_from_addr(
                peer_base(workspace_flat_base, peer)
                + slot * fx.Int64(partial_bytes),
                num_records_bytes=partial_bytes,
            )
            if cache_modifier is None:
                cache_modifier = (
                    config.remote_load_cache_modifier
                    if peer != specialized_rank
                    else config.local_load_cache_modifier
                )
            words = load_fp8_words(
                peer_resource,
                load_offset // fx.Int32(4),
                word_count=load_vector_width // 4,
                load_width=load_vector_width // 4,
                cache_modifier=cache_modifier,
            )
            scale = fx.Int32(config.fp8_scale_exponent << 23).bitcast(
                fx.Float32
            )
            return fx.Vector.from_elements(
                decode_scaled_fp8_f32(words, scale),
                fx.Float32,
            )

        # Stagger peer traffic by owner rank.  For large FP8 shards, also use
        # the shard token so simultaneous waves on one rank do not serialize
        # against the same XGMI destination.
        if source_rotation is not None:
            first_peer = (
                rank + source_rotation + fx.Int32(1)
            ) & fx.Int32(TP - 1)
            second_peer = (
                rank + source_rotation + fx.Int32(2)
            ) & fx.Int32(TP - 1)
            reduced_even = load_peer(
                first_peer, config.remote_load_cache_modifier
            )
            reduced_odd = load_peer(
                second_peer, config.remote_load_cache_modifier
            )
            for peer_step in range_constexpr(3, TP + 1):
                peer = (
                    rank + source_rotation + fx.Int32(peer_step)
                ) & fx.Int32(TP - 1)
                if const_expr(peer_step % 2 == 1):
                    reduced_even = reduced_even + load_peer(
                        peer, config.remote_load_cache_modifier
                    )
                else:
                    reduced_odd = reduced_odd + load_peer(
                        peer, config.remote_load_cache_modifier
                    )
            reduced = reduced_even + reduced_odd
        else:
            # Seed one chain from the low-latency local partial and split the
            # seven rank-rotated remote loads over two accumulators.  This
            # preserves distributed peer traffic while shortening the serial
            # FP32 add dependency from eight values to two four-value chains.
            reduced_even = load_peer(specialized_rank)
            reduced_odd = load_peer((specialized_rank + 1) % TP)
            for peer_step in range_constexpr(2, TP):
                peer = (specialized_rank + peer_step) % TP
                if const_expr(peer_step % 2 == 0):
                    reduced_even = reduced_even + load_peer(peer)
                else:
                    reduced_odd = reduced_odd + load_peer(peer)
            reduced = reduced_even + reduced_odd
        return reduced

    def emit_direct_reduce():
        def emit_allreduce_item(item, retained_local_partial=None):
            token = item // fx.Int32(config.tile_n // config.vector_width)
            tile_item = item - token * fx.Int32(
                config.tile_n // config.vector_width
            )
            offset = (
                token * fx.Int32(H)
                + n_tile * fx.Int32(config.tile_n)
                + tile_item * fx.Int32(config.vector_width)
            )
            reduced = load_reduced_partials(
                offset,
                retained_local_partial,
            )
            _store_bf16(
                output_resource,
                offset,
                reduced.truncf(T.vec(config.vector_width, T.bf16)),
                config.vector_width,
            )

        if retain_local_partials:
            for iteration in range_constexpr(reduce_items // service_stride):
                item = service_start + fx.Int32(iteration * service_stride)
                emit_allreduce_item(
                    item,
                    retained_local_partials[iteration],
                )
        else:
            allreduce_loop = scf.ForOp(
                arith.index_cast(T.index, service_start),
                arith.constant(reduce_items, index=True),
                arith.constant(service_stride, index=True),
            )
            with ir.InsertionPoint(allreduce_loop.body):
                emit_allreduce_item(
                    arith.index_cast(T.i32, allreduce_loop.induction_variable)
                )
                scf.YieldOp([])

    def reset_tile_state_values():
        for tile_delta in range_constexpr(config.service_tile_group):
            _store_i32_relaxed(
                local_workspace_base
                + fx.Int64(config.producer_done_offset)
                + fx.Int64(
                    state_n_tile + fx.Int32(tile_delta)
                )
                * fx.Int64(PRODUCER_COUNTER_STRIDE),
                fx.Int32(0),
            )
        if config.service_groups > 1:
            _store_i32_relaxed(
                local_workspace_base
                + fx.Int64(config.service_done_offset)
                + tile_byte_offset,
                fx.Int32(0),
            )
            _store_i32_relaxed(
                local_workspace_base
                + fx.Int64(config.reduce_done_offset)
                + tile_byte_offset,
                fx.Int32(0),
            )
            _store_i32_relaxed(
                local_workspace_base
                + fx.Int64(config.gather_service_done_offset)
                + tile_byte_offset,
                fx.Int32(0),
            )
        _store_i64_relaxed(epoch_address, expected)

    def emit_rsag_reduce():
        collective_vector_width = config.vector_width

        def emit_gather_ack(barrier=True):
            gather_ack = scf.IfOp(
                arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(TP))
            )
            with ir.InsertionPoint(gather_ack.then_block):
                remote_slot = state_n_tile * fx.Int32(TP) + rank
                _store_i32_system_monotonic(
                    peer_base(workspace_flat_base, tid)
                    + fx.Int64(config.gather_done_offset)
                    + fx.Int64(remote_slot) * fx.Int64(4),
                    expected_i32,
                )
                scf.YieldOp([])
            if barrier:
                gpu.barrier()

        def publish_gather_completion():
            if config.service_groups == 1:
                if config.collective == "rs_broadcast":
                    # The preceding waitcnt + workgroup barrier already proves
                    # that every source load has retired.  Broadcast output is
                    # consumed only after kernel completion, so its stores do
                    # not need to be ordered by the source-reuse ACK.  Let the
                    # outer waitcnt drain the eight ACK stores without adding a
                    # second system fence and two more workgroup barriers.
                    emit_gather_ack()
                else:
                    gather_release = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                    )
                    with ir.InsertionPoint(gather_release.then_block):
                        comm_ops.fence_system_release()
                        scf.YieldOp([])
                    gpu.barrier()
                    emit_gather_ack()
            else:
                gather_done_address = (
                    local_workspace_base
                    + fx.Int64(config.gather_service_done_offset)
                    + tile_byte_offset
                )
                gather_publisher = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(gather_publisher.then_block):
                    comm_ops.fence_agent_release()
                    arrival = fx.Int32(
                        _atomic_add_i32_agent(
                            gather_done_address, fx.Int32(1)
                        )
                    )
                    SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).store(
                        arrival
                    )
                    scf.YieldOp([])
                gpu.barrier()
                gather_arrival = fx.Int32(
                    SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).load()
                )
                coordinator_condition = arith.cmpi(
                    CmpIPredicate.eq,
                    gather_arrival,
                    fx.Int32(
                        config.service_groups * config.service_tile_group - 1
                    ),
                )
                coordinator = scf.IfOp(coordinator_condition)
                with ir.InsertionPoint(coordinator.then_block):
                    gather_acquirer = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                    )
                    with ir.InsertionPoint(gather_acquirer.then_block):
                        comm_ops.fence_agent_acquire()
                        comm_ops.fence_system_release()
                        scf.YieldOp([])
                    gpu.barrier()
                    emit_gather_ack(barrier=False)
                    fx.rocdl.s_waitcnt(0)
                    gather_resetter = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                    )
                    with ir.InsertionPoint(gather_resetter.then_block):
                        reset_tile_state_values()
                        scf.YieldOp([])
                    scf.YieldOp([])

        def emit_reduce_scatter_items():
            vectors_per_token = config.tile_n // collective_vector_width
            shard_tokens = config.m // TP
            service_item = tid + service_group * fx.Int32(config.block_threads)
            first_shard_token = service_item // fx.Int32(vectors_per_token)
            vector_lane = service_item - first_shard_token * fx.Int32(
                vectors_per_token
            )

            reduce_scatter_loop = scf.ForOp(
                arith.index_cast(T.index, first_shard_token),
                arith.constant(shard_tokens, index=True),
                arith.constant(
                    config.block_threads
                    * config.service_groups
                    // vectors_per_token,
                    index=True,
                ),
            )
            with ir.InsertionPoint(reduce_scatter_loop.body):
                shard_token = arith.index_cast(
                    T.i32, reduce_scatter_loop.induction_variable
                )
                token = rank * fx.Int32(shard_tokens) + shard_token
                offset = (
                    token * fx.Int32(H)
                    + n_tile * fx.Int32(config.tile_n)
                    + vector_lane * fx.Int32(collective_vector_width)
                )
                reduced = load_reduced_partials(
                    offset,
                    source_rotation=(
                        shard_token
                        if config.collective == "rsag"
                        and config.service_groups > 1
                        else None
                    ),
                    vector_width=collective_vector_width,
                )
                reduced_bf16 = reduced.truncf(
                    T.vec(collective_vector_width, T.bf16)
                )
                if config.collective == "rsag":
                    # The local rank already owns this output shard. Publish
                    # it directly instead of reading and decoding it again in
                    # the subsequent all-gather.
                    _store_bf16(
                        output_resource,
                        offset,
                        reduced_bf16,
                        collective_vector_width,
                        config.remote_store_cache_modifier,
                    )
                reduced_offset = (
                    n_tile * fx.Int32(config.m * config.tile_n // TP)
                    + shard_token * fx.Int32(config.tile_n)
                    + vector_lane * fx.Int32(collective_vector_width)
                )
                if config.collective == "rs_broadcast":
                    for peer_step in range_constexpr(TP):
                        peer = (specialized_rank + peer_step + 1) % TP
                        if peer == specialized_rank:
                            peer_output_resource = output_resource
                        else:
                            peer_output_resource = (
                                buffer_ops.create_buffer_resource_from_addr(
                                    peer_base(workspace_flat_base, peer)
                                    + fx.Int64(config.output_offset),
                                    num_records_bytes=payload_bytes,
                                )
                            )
                        _store_bf16(
                            peer_output_resource,
                            offset,
                            reduced_bf16,
                            collective_vector_width,
                            config.remote_store_cache_modifier,
                        )
                else:
                    reduced_quant_scale = fx.Int32(
                        (254 - config.fp8_scale_exponent) << 23
                    ).bitcast(fx.Float32)
                    store_fp8_words(
                        reduced_resource,
                        reduced_offset,
                        pack_fp8_words(
                            reduced,
                            reduced_quant_scale,
                            collective_vector_width // 4,
                        ),
                        collective_vector_width // 4,
                    )
                scf.YieldOp([])
            fx.rocdl.s_waitcnt(0)
            gpu.barrier()

        emit_reduce_scatter_items()

        if config.collective == "rs_broadcast":
            publish_gather_completion()
            return

        def emit_reduced_exchange(propagate_acquire):
            reduced_exchange = scf.IfOp(
                arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(TP))
            )
            with ir.InsertionPoint(reduced_exchange.then_block):
                remote_slot = state_n_tile * fx.Int32(TP) + rank
                _store_i32_system_monotonic(
                    peer_base(workspace_flat_base, tid)
                    + fx.Int64(config.owner_ready_offset)
                    + fx.Int64(remote_slot) * fx.Int64(4),
                    expected_i32,
                )
                local_slot = state_n_tile * fx.Int32(TP) + tid
                _wait_i32_system_until_at_least(
                    local_workspace_base
                    + fx.Int64(config.owner_ready_offset)
                    + fx.Int64(local_slot) * fx.Int64(4),
                    expected_i32,
                )
                scf.YieldOp([])
            gpu.barrier()
            reduced_acquire = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
            )
            with ir.InsertionPoint(reduced_acquire.then_block):
                comm_ops.fence_system_acquire()
                scf.YieldOp([])
            if propagate_acquire:
                gpu.barrier()

        if config.service_groups == 1:
            reduced_release = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
            )
            with ir.InsertionPoint(reduced_release.then_block):
                comm_ops.fence_system_release()
                scf.YieldOp([])
            gpu.barrier()
            emit_reduced_exchange(True)
        else:
            reduce_done_address = (
                local_workspace_base
                + fx.Int64(config.reduce_done_offset)
                + tile_byte_offset
            )
            reduce_publisher = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
            )
            with ir.InsertionPoint(reduce_publisher.then_block):
                comm_ops.fence_agent_release()
                arrival = fx.Int32(
                    _atomic_add_i32_agent(reduce_done_address, fx.Int32(1))
                )
                SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).store(
                    arrival
                )
                scf.YieldOp([])
            gpu.barrier()
            reduce_arrival = fx.Int32(
                SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).load()
            )
            coordinator = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.eq,
                    reduce_arrival,
                    fx.Int32(
                        config.service_groups * config.service_tile_group - 1
                    ),
                )
            )
            with ir.InsertionPoint(coordinator.then_block):
                reduce_acquirer = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(reduce_acquirer.then_block):
                    comm_ops.fence_agent_acquire()
                    comm_ops.fence_system_release()
                    scf.YieldOp([])
                gpu.barrier()
                # Only lane 0 publishes reduced_collective_ready below. Its
                # system acquire is therefore ordered directly before the
                # agent-release publication; unlike the single-service path,
                # no other lane consumes remote reduced data here.
                emit_reduced_exchange(False)
                reduced_publisher = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(reduced_publisher.then_block):
                    _store_i32_agent_release(
                        local_workspace_base
                        + fx.Int64(config.reduced_collective_ready_offset)
                        + fx.Int64(state_n_tile) * fx.Int64(4),
                        expected_i32,
                    )
                    scf.YieldOp([])
                scf.YieldOp([])

            reduced_waiter = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
            )
            with ir.InsertionPoint(reduced_waiter.then_block):
                _wait_i32_agent_until_at_least(
                    local_workspace_base
                    + fx.Int64(config.reduced_collective_ready_offset)
                    + fx.Int64(state_n_tile) * fx.Int64(4),
                    expected_i32,
                )
                comm_ops.fence_agent_acquire()
                scf.YieldOp([])
            gpu.barrier()

        def emit_gather_source(source, source_start, source_stride):
            gather_vector_width = config.vector_width
            gather_cache_modifier = (
                config.remote_load_cache_modifier
                if config.gather_load_cache_modifier < 0
                else config.gather_load_cache_modifier
            )
            vectors_per_token = config.tile_n // gather_vector_width
            shard_tokens = config.m // TP
            first_token = source_start // fx.Int32(vectors_per_token)
            vector_lane = source_start - first_token * fx.Int32(
                vectors_per_token
            )
            source_resource = buffer_ops.create_buffer_resource_from_addr(
                peer_base(workspace_flat_base, source)
                + fx.Int64(config.reduced_offset)
                + slot * fx.Int64(config.reduced_shard_bytes),
                num_records_bytes=config.reduced_shard_bytes,
            )
            gather_loop = scf.ForOp(
                arith.index_cast(T.index, first_token),
                arith.constant(shard_tokens, index=True),
                arith.constant(
                    source_stride // vectors_per_token,
                    index=True,
                ),
            )
            with ir.InsertionPoint(gather_loop.body):
                source_token = arith.index_cast(
                    T.i32, gather_loop.induction_variable
                )
                offset = (
                    (
                        source * fx.Int32(shard_tokens)
                        + source_token
                    )
                    * fx.Int32(H)
                    + n_tile * fx.Int32(config.tile_n)
                    + vector_lane * fx.Int32(gather_vector_width)
                )
                reduced_offset = (
                    n_tile * fx.Int32(config.m * config.tile_n // TP)
                    + source_token * fx.Int32(config.tile_n)
                    + vector_lane * fx.Int32(gather_vector_width)
                )
                words = load_fp8_words(
                    source_resource,
                    reduced_offset // fx.Int32(4),
                    word_count=gather_vector_width // 4,
                    load_width=gather_vector_width // 4,
                    cache_modifier=gather_cache_modifier,
                )
                scale = fx.Int32(
                    config.fp8_scale_exponent << 23
                ).bitcast(fx.Float32)
                values = fx.Vector.from_elements(
                    _decode_scaled_fp8_bf16(words, scale),
                    fx.BFloat16,
                )
                _store_bf16(
                    output_resource,
                    offset,
                    values,
                    gather_vector_width,
                    config.remote_store_cache_modifier,
                )
                scf.YieldOp([])

        def emit_remote_gather_source(source, source_start, source_stride):
            remote_source = scf.IfOp(
                arith.cmpi(CmpIPredicate.ne, source, rank)
            )
            with ir.InsertionPoint(remote_source.then_block):
                emit_gather_source(source, source_start, source_stride)
                scf.YieldOp([])

        # SG4 assigns two source shards to each workgroup. Split the four
        # waves into two independent halves so both peers progress together;
        # unlike loading both peers in every lane, this keeps one live value
        # vector per thread and avoids the VGPR increase of the paired-load
        # experiment.
        if config.service_groups == 1:
            parallel_sources = 4
            threads_per_source = config.block_threads // parallel_sources
            source_lane = tid // fx.Int32(threads_per_source)
            for source_iteration in range_constexpr(TP // parallel_sources):
                source = (
                    n_tile
                    + source_lane
                    + fx.Int32(source_iteration * parallel_sources)
                ) % fx.Int32(TP)
                emit_remote_gather_source(
                    source,
                    tid % fx.Int32(threads_per_source),
                    threads_per_source,
                )
        elif config.service_groups == 4:
            sources_per_group = TP // config.service_groups
            threads_per_source = config.block_threads // sources_per_group
            source_lane = tid // fx.Int32(threads_per_source)
            source_phase = (
                n_tile + source_lane
            ) % fx.Int32(sources_per_group)
            source = (
                service_group
                + source_phase * fx.Int32(config.service_groups)
            )
            emit_remote_gather_source(
                source,
                tid % fx.Int32(threads_per_source),
                threads_per_source,
            )
        else:
            # Keep each service workgroup on one source rank at a time. The
            # old global item-stride mapping made every service workgroup pull
            # from the same peer concurrently. Source partitioning spreads
            # that traffic across peers and keeps the descriptor invariant in
            # the inner loop.
            for source_iteration in range_constexpr(
                TP // config.service_groups
            ):
                source_phase = (
                    n_tile + fx.Int32(source_iteration)
                ) % fx.Int32(TP // config.service_groups)
                source = (
                    service_group
                    + source_phase * fx.Int32(config.service_groups)
                )
                emit_remote_gather_source(
                    source, tid, config.block_threads
                )
        fx.rocdl.s_waitcnt(0)
        gpu.barrier()
        publish_gather_completion()

    if config.service_groups == 1:
        release_publisher = scf.IfOp(
            arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
        )
        with ir.InsertionPoint(release_publisher.then_block):
            comm_ops.fence_system_release()
            scf.YieldOp([])
        if config.single_pass_direct:
            gpu.barrier()

        rank_exchange = scf.IfOp(
            arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(TP))
        )
        with ir.InsertionPoint(rank_exchange.then_block):
            remote_slot = state_n_tile * fx.Int32(TP) + rank
            _store_i32_system_monotonic(
                peer_base(workspace_flat_base, tid)
                + fx.Int64(config.rank_ready_offset)
                + fx.Int64(remote_slot) * fx.Int64(4),
                expected_i32,
            )
            local_slot = state_n_tile * fx.Int32(TP) + tid
            ready_address = (
                local_workspace_base
                + fx.Int64(config.rank_ready_offset)
                + fx.Int64(local_slot) * fx.Int64(4)
            )
            if config.single_pass_direct:
                _wait_i32_agent_until_at_least(
                    ready_address,
                    expected_i32,
                    sleep=False,
                )
            else:
                _wait_i32_system_until_at_least(
                    ready_address,
                    expected_i32,
                )
            scf.YieldOp([])
        if not config.single_pass_direct:
            rank_acquire = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
            )
            with ir.InsertionPoint(rank_acquire.then_block):
                comm_ops.fence_system_acquire()
                scf.YieldOp([])
        gpu.barrier()
    else:
        service_done_address = (
            local_workspace_base
            + fx.Int64(config.service_done_offset)
            + tile_byte_offset
        )

        def publish_partials_and_exchange():
            service_publisher = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
            )
            with ir.InsertionPoint(service_publisher.then_block):
                comm_ops.fence_agent_release()
                arrival = fx.Int32(
                    _atomic_add_i32_agent(service_done_address, fx.Int32(1))
                )
                SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).store(
                    arrival
                )
                scf.YieldOp([])
            gpu.barrier()
            service_arrival = fx.Int32(
                SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).load()
            )
            coordinator = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.eq,
                    service_arrival,
                    fx.Int32(
                        config.service_groups * config.service_tile_group - 1
                    ),
                )
            )
            with ir.InsertionPoint(coordinator.then_block):
                service_acquirer = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(service_acquirer.then_block):
                    comm_ops.fence_agent_acquire()
                    local_ready_slot = state_n_tile * fx.Int32(TP) + rank
                    _store_i32_system_release(
                        local_workspace_base
                        + fx.Int64(config.rank_ready_offset)
                        + fx.Int64(local_ready_slot) * fx.Int64(4),
                        expected_i32,
                    )
                    scf.YieldOp([])
                gpu.barrier()

                rank_waiter = scf.IfOp(
                    arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(TP))
                )
                with ir.InsertionPoint(rank_waiter.then_block):
                    peer_ready_slot = state_n_tile * fx.Int32(TP) + tid
                    _wait_i32_system_until_at_least(
                        peer_base(workspace_flat_base, tid)
                        + fx.Int64(config.rank_ready_offset)
                        + fx.Int64(peer_ready_slot) * fx.Int64(4),
                        expected_i32,
                    )
                    scf.YieldOp([])
                gpu.barrier()
                rank_acquire = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(rank_acquire.then_block):
                    comm_ops.fence_system_acquire()
                    _store_i32_agent_release(
                        local_workspace_base
                        + fx.Int64(config.collective_ready_offset)
                        + fx.Int64(state_n_tile) * fx.Int64(4),
                        expected_i32,
                    )
                    scf.YieldOp([])
                scf.YieldOp([])

        publish_partials_and_exchange()

        def wait_for_collective():
            collective_waiter = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
            )
            with ir.InsertionPoint(collective_waiter.then_block):
                _wait_i32_agent_until_at_least(
                    local_workspace_base
                    + fx.Int64(config.collective_ready_offset)
                    + fx.Int64(state_n_tile) * fx.Int64(4),
                    expected_i32,
                )
                comm_ops.fence_agent_acquire()
                scf.YieldOp([])
            gpu.barrier()

    if config.uses_rsag:
        if config.service_groups > 1:
            wait_for_collective()
        emit_rsag_reduce()
    else:
        emit_direct_reduce()
    fx.rocdl.s_waitcnt(0)
    if not (config.uses_rsag and config.service_groups > 1):
        reset_condition = arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
        state_resetter = scf.IfOp(reset_condition)
        with ir.InsertionPoint(state_resetter.then_block):
            reset_tile_state_values()
            scf.YieldOp([])


@functools.cache
def _compile_owner_megakernel(
    config: OwnerMegaKernelConfig,
    specialized_rank: int,
):
    """Compile the production single-launch Stage2 owner pipeline."""

    if not 0 <= specialized_rank < TP:
        raise ValueError(f"invalid TP rank {specialized_rank}")

    n_tiles = config.n_tiles
    dynamic_producer = config.collective != "direct"
    gather_cache_tag = (
        "inherit"
        if config.gather_load_cache_modifier < 0
        else str(config.gather_load_cache_modifier)
    )
    rows_per_group = (
        config.producer_rows + config.compute_groups - 1
    ) // config.compute_groups
    launch_grid = (
        (config.compute_groups * n_tiles, 1, 1)
        if config.n_tile_cohort or config.flat_producer_grid
        else (n_tiles, config.compute_groups, 1)
    )
    cache_abi = "owner_mega_v1"
    cache_config = hashlib.sha256(repr(config).encode()).hexdigest()[:16]

    def compose(*, module_name, emit_gemm2, allocator):
        @flyc.kernel(
            name=(
                f"{module_name}_owner_mega_r{specialized_rank}_m{config.m}"
                f"_n{config.tile_n}_cg{config.compute_groups}"
                f"_bt{config.block_threads}_v{config.vector_width}"
                f"_bc{config.b_cache_modifier}"
                f"_rts{config.route_store_scope}"
                f"_glc{gather_cache_tag}"
                f"_rsc{config.remote_store_cache_modifier}"
                f"_fp8e{config.fp8_scale_exponent}"
                f"_ntc{config.n_tile_cohort}_{config.collective}"
                f"_sg{config.service_groups}_stg{config.service_tile_group}"
                f"_p{config.producer_mode}"
                f"_fpg{int(config.flat_producer_grid)}"
                f"_sbp{int(config.shared_bf16_partials)}"
            ),
            known_block_size=[config.block_threads, 1, 1],
        )
        def kernel(
            workspace: fx.Pointer,
            x: fx.Pointer,
            w: fx.Pointer,
            scale_x: fx.Pointer,
            scale_w: fx.Pointer,
            sorted_token_ids: fx.Pointer,
            expert_ids: fx.Pointer,
            sorted_weights: fx.Pointer,
            num_valid_ids: fx.Pointer,
            shared_partial: fx.Pointer,
            shared_partial_flat_base: fx.Int64,
            tokens: fx.Int32,
            model_dim: fx.Int32,
            inter_dim: fx.Int32,
            size_expert_ids: fx.Int32,
        ):
            local_workspace_base = fx.Int64(ptrtoint(workspace))
            route_output = _byte_ptr(
                local_workspace_base + fx.Int64(config.route_offset)
            )
            producer_output = (
                shared_partial
                if config.producer_mode == "atomic_shared"
                else route_output
            )
            tid = fx.Int32(gpu.thread_idx.x)
            base = allocator.get_base()

            def emit_gemm(block_id=None):
                emit_gemm2(
                    producer_output,
                    x,
                    w,
                    scale_x,
                    scale_w,
                    w,
                    scale_w,
                    sorted_token_ids,
                    expert_ids,
                    sorted_weights,
                    num_valid_ids,
                    shared_partial,
                    tokens,
                    model_dim,
                    inter_dim,
                    size_expert_ids,
                    block_id=block_id,
                )

            def emit_service(n_tile, service_group):
                workspace_flat_base = fx.Int64(
                    comm_ops.load_i64_global(
                        local_workspace_base
                        + fx.Int64(config.flat_base_offset)
                    )
                )
                _emit_direct_allreduce_service_tile(
                    config,
                    workspace,
                    workspace_flat_base,
                    shared_partial,
                    shared_partial_flat_base,
                    specialized_rank,
                    n_tile,
                    tid,
                    service_group,
                    base,
                )

            def emit_and_service(n_tile, block_id=None):
                def publish_completion():
                    completion_publisher = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                    )
                    with ir.InsertionPoint(completion_publisher.then_block):
                        ticket = fx.Int32(
                            _atomic_add_i32_agent(
                                local_workspace_base
                                + fx.Int64(config.producer_done_offset)
                                + fx.Int64(n_tile)
                                * fx.Int64(PRODUCER_COUNTER_STRIDE),
                                fx.Int32(1),
                            )
                        )
                        service_marker_ptr = SmemPtr(
                            base, 0, T.i32, shape=(1,)
                        )
                        service_begin = fx.Int32(
                            config.compute_groups - config.service_groups
                        )
                        is_service = arith.cmpi(
                            CmpIPredicate.uge,
                            ticket,
                            service_begin,
                        )
                        service_marker_ptr.store(
                            arith.select(
                                is_service,
                                ticket - service_begin + fx.Int32(1),
                                fx.Int32(0),
                            )
                        )
                        scf.YieldOp([])

                emit_gemm(block_id)
                # Dynamic producers are synchronized inside the GEMM emitter's
                # final persistent iteration. Fixed producers need the explicit
                # workgroup drain before publishing their completion ticket.
                if not dynamic_producer:
                    fx.rocdl.s_waitcnt(0)
                    gpu.barrier()
                publish_completion()
                gpu.barrier()

                service_marker = fx.Int32(
                    SmemPtr(base, 0, T.i32, shape=(1,)).load()
                )
                last_producer = scf.IfOp(
                    arith.cmpi(
                        CmpIPredicate.ugt,
                        service_marker,
                        fx.Int32(0),
                    )
                )
                with ir.InsertionPoint(last_producer.then_block):
                    service_group = service_marker - fx.Int32(1)
                    if config.service_groups > 1:
                        producer_waiter = scf.IfOp(
                            arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                        )
                        with ir.InsertionPoint(producer_waiter.then_block):
                            _wait_i32_agent_until_at_least(
                                local_workspace_base
                                + fx.Int64(config.producer_done_offset)
                                + fx.Int64(n_tile)
                                * fx.Int64(PRODUCER_COUNTER_STRIDE),
                                fx.Int32(config.compute_groups),
                            )
                            comm_ops.fence_agent_acquire()
                            scf.YieldOp([])
                    else:
                        if tid == fx.Int32(0):
                            comm_ops.fence_agent_acquire()
                    gpu.barrier()
                    emit_service(n_tile, service_group)
                    scf.YieldOp([])

            if config.flat_producer_grid:
                physical_block = fx.Int32(gpu.block_idx.x)
                n_tiles_i32 = fx.Int32(config.n_tiles)
                n_tile_index = physical_block % n_tiles_i32
                compute_group = physical_block // n_tiles_i32
                emit_and_service(
                    n_tile_index,
                    arith.index_cast(
                        T.index,
                        compute_group * n_tiles_i32 + n_tile_index,
                    ),
                )
            elif config.n_tile_cohort:
                physical_block = fx.Int32(gpu.block_idx.x)
                cohort_size = fx.Int32(config.n_tile_cohort)
                cohort_span = fx.Int32(
                    config.n_tile_cohort * config.compute_groups
                )
                cohort_base = physical_block // cohort_span * cohort_size
                within_cohort = physical_block % cohort_span
                n_tile_index = cohort_base + within_cohort % cohort_size
                compute_group = within_cohort // cohort_size
                emit_and_service(
                    n_tile_index,
                    arith.index_cast(
                        T.index,
                        compute_group * fx.Int32(config.n_tiles)
                        + n_tile_index,
                    ),
                )
            else:
                emit_and_service(fx.Int32(gpu.block_idx.x))

        def launch(
            workspace,
            shared_partial,
            shared_partial_flat_base,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            sorted_weights,
            num_valid_ids,
            tokens,
            model_dim,
            inter_dim,
            size_expert_ids,
            stream,
        ):
            allocator.finalized = False
            context = CompilationContext.get_current()
            with ir.InsertionPoint(context.gpu_module_body):
                allocator.finalize()
            if const_expr(config.waves_per_eu > 0):
                for op in context.gpu_module_body.operations:
                    if (
                        hasattr(op, "attributes")
                        and op.OPERATION_NAME == "gpu.func"
                    ):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, config.waves_per_eu
                        )
            kernel(
                workspace,
                x,
                w,
                scale_x,
                scale_w,
                sorted_token_ids,
                expert_ids,
                sorted_weights,
                num_valid_ids,
                shared_partial,
                shared_partial_flat_base,
                tokens,
                model_dim,
                inter_dim,
                size_expert_ids,
            ).launch(
                grid=launch_grid,
                block=(config.block_threads, 1, 1),
                stream=stream,
            )

        # FlyDSL's disk-cache identity does not include values captured by a
        # closure. Give every code-generating owner configuration a distinct
        # function name before wrapping it with jit, otherwise (for example)
        # an M=256 launch can reuse an M=64 code object.
        launch.__name__ = (
            f"launch_{cache_abi}_r{specialized_rank}_{cache_config}"
        )
        return flyc.jit(launch)

    route_producer = config.producer_mode in ("routes", "routes_fp8_fixed")
    output_epilogue = (
        _RouteOutputEpilogue(
            fp8_fixed=config.producer_mode == "routes_fp8_fixed",
            device_coherent=config.route_store_scope == "device",
        )
        if route_producer
        else None
    )
    compose.gemm2_config = _OwnerGemm2ComposeConfig(
        block_threads=config.block_threads,
        persistent_groups=(
            config.compute_groups if dynamic_producer else None
        ),
        output_epilogue=output_epilogue,
        b_cache_modifier=config.b_cache_modifier,
    )
    return compile_mixed_moe_gemm2_common(
        model_dim=H,
        inter_dim=I,
        experts=E,
        topk=TOPK,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        doweight_stage2=True,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        accumulate=not route_producer,
        persist_m=0 if dynamic_producer else rows_per_group,
        sort_block_m=config.sort_block_m,
        waves_per_eu=config.waves_per_eu or None,
        _compose_entry=compose,
    )


def compile_owner_megakernel(
    config: OwnerMegaKernelConfig,
    specialized_rank: int,
):
    return _compile_owner_megakernel(config, specialized_rank)
