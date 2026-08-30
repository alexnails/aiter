# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL DCP decode TopK merge tests."""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# The kernel hard-requires wave64 (see dcp_topk_merge._validate), so on a wave32
# target every test here would raise ValueError rather than skip.
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason=f"FlyDSL DCP TopK merge needs one of {SUPPORTED_GFX}",
)


def ref_dcp_merge(
    gathered_scores: torch.Tensor,  # fp32 [rows, W*k_loc]
    local_idx: torch.Tensor,  # i32  [rows, k_loc]
    block_table: torch.Tensor,  # i32  [rows, max_blocks]
    dcp_rank: int,
    k_loc: int,
    topk_tokens: int,
    page_size: int,
):
    """Oracle: global threshold select, keep own plane, map to physical slots.

    Tie rule mirrors the kernel: among candidates whose score equals the
    threshold exactly, the ones with the smallest flat candidate position win.
    torch.topk on a descending sort of (score, -position) gives that ordering.
    """
    rows = gathered_scores.shape[0]
    owned_slots = []
    counts = torch.zeros(rows, dtype=torch.int32, device=gathered_scores.device)
    for r in range(rows):
        sc = gathered_scores[r]
        finite = torch.isfinite(sc)
        n_valid = int(finite.sum())
        take = min(topk_tokens, n_valid)
        # Sort by score desc, ties broken by smaller flat position.
        order = torch.argsort(
            torch.where(finite, sc, torch.full_like(sc, -float("inf"))),
            descending=True,
            stable=True,
        )
        winners = order[:take]
        # Keep only winners from this rank's plane.
        mine = winners[(winners // k_loc) == dcp_rank]
        # Plane position -> local KV index -> physical slot.
        local_pos = (mine % k_loc).to(torch.int64)
        j = local_idx[r].to(torch.int64)[local_pos]
        assert bool((j >= 0).all()), "padding leaked into the winner set"
        slot = block_table[r].to(torch.int64)[j // page_size] * page_size + (
            j % page_size
        )
        # Compact in increasing flat-candidate order (deterministic).
        slot = slot[torch.argsort(mine, stable=True)]
        owned_slots.append(slot.to(torch.int32))
        counts[r] = slot.numel()
    return owned_slots, counts


def make_case(
    rows,
    world,
    k_loc,
    topk_tokens,
    page_size,
    max_blocks,
    n_local=None,
    tie_heavy=False,
    seed=0,
):
    """Build a self-consistent (scores, local_idx, block_table) triple.

    Every rank's plane gets k_loc candidates. Pass n_local < k_loc to emulate a
    short context: the tail of local_idx becomes -1, exactly as
    top_k_per_row_decode pads it the same way. Callers that want
    the matching -inf scores must starve the corresponding plane themselves --
    see test_short_context_padding_never_emitted (Task 6's short-context test
    does exactly this). The separation is intentional: make_case is responsible
    only for index/block-table consistency, not for score semantics.

    Multi-block slot coverage note: when n_local < page_size, every generated
    local index j satisfies j // page_size == 0, so only block_table[r, 0] is
    ever exercised. Callers writing short-context cases should pick
    n_local >= page_size if they want coverage of more than one block.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_cand = world * k_loc
    if tie_heavy:
        scores = torch.randint(
            -4, 5, (rows, n_cand), generator=g, dtype=torch.int32, device="cuda"
        ).float()
    else:
        scores = torch.randn(rows, n_cand, generator=g, device="cuda")
    fill = k_loc if n_local is None else min(k_loc, n_local)
    local_idx = torch.empty(rows, k_loc, dtype=torch.int32, device="cuda")
    for r in range(rows):
        # Local KV indices must stay in range for the slot formula: j // page_size
        # indexes block_table, so keep j < max_blocks * page_size.
        hi = min(max_blocks * page_size, max(fill, 1))
        perm = torch.randperm(hi, generator=g, device="cuda")[:fill]
        local_idx[r, :fill] = perm.to(torch.int32)
        local_idx[r, fill:] = -1
    block_table = torch.randint(
        0, 1000, (rows, max_blocks), generator=g, dtype=torch.int32, device="cuda"
    )
    return scores, local_idx, block_table


def test_reference_model_selects_global_topk():
    """The oracle keeps exactly this rank's share of the global winners."""
    rows, world, k_loc, topk, page = 2, 4, 8, 16, 4
    scores, local_idx, bt = make_case(rows, world, k_loc, topk, page, 64, seed=1)
    slots, counts = ref_dcp_merge(scores, local_idx, bt, 2, k_loc, topk, page)
    # Union across all ranks must be exactly `topk` winners per row.
    total = torch.zeros(rows, dtype=torch.int32, device="cuda")
    for r_id in range(world):
        _, c = ref_dcp_merge(scores, local_idx, bt, r_id, k_loc, topk, page)
        total += c
    assert torch.all(total == topk), f"ranks do not partition topk: {total}"
    # counts[r] must equal the number of slots returned for that row.
    for r in range(rows):
        assert int(counts[r]) == slots[r].numel(), (
            f"row {r}: counts[r]={int(counts[r])} but slots[r] has "
            f"{slots[r].numel()} elements"
        )
    assert len(slots) == rows


def test_kernel_owned_slots_match_reference():
    """Per-row owned slots and counts match the oracle, for every rank."""
    from aiter.ops.flydsl.dcp_topk_merge import _debug_staging

    rows, world, k_loc, topk, page = 4, 8, 64, 128, 16
    scores, local_idx, bt = make_case(rows, world, k_loc, topk, page, 64, seed=3)
    for rank in range(world):
        staging, counts = _debug_staging(scores, local_idx, bt, rank, k_loc, topk, page)
        want_slots, want_counts = ref_dcp_merge(
            scores, local_idx, bt, rank, k_loc, topk, page
        )
        torch.testing.assert_close(counts, want_counts, rtol=0, atol=0)
        for r in range(rows):
            n = int(counts[r])
            torch.testing.assert_close(
                staging[r, :n],
                want_slots[r],
                rtol=0,
                atol=0,
                msg=f"rank {rank} row {r} slot mismatch",
            )


@pytest.mark.parametrize("tie_heavy", [False, True])
def test_ranks_partition_the_global_topk(tie_heavy):
    """The W owned sets are disjoint and total exactly topk_tokens."""
    from aiter.ops.flydsl.dcp_topk_merge import _debug_staging

    rows, world, k_loc, topk, page = 4, 8, 64, 128, 16
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page, 64, tie_heavy=tie_heavy, seed=4
    )
    total = torch.zeros(rows, dtype=torch.int32, device="cuda")
    for rank in range(world):
        _, counts = _debug_staging(scores, local_idx, bt, rank, k_loc, topk, page)
        total += counts
    assert torch.all(total == topk), f"ranks do not partition topk: {total}"


def test_packed_output_and_indptr():
    """indptr is the exclusive cumsum of counts; indices are packed in row order."""
    import aiter

    rows, world, k_loc, topk, page = 5, 8, 64, 128, 16
    scores, local_idx, bt = make_case(rows, world, k_loc, topk, page, 64, seed=5)
    rank = 3
    want_slots, want_counts = ref_dcp_merge(
        scores, local_idx, bt, rank, k_loc, topk, page
    )

    staging = torch.empty(rows, k_loc, dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indices = torch.zeros(rows * k_loc, dtype=torch.int32)
    aiter.flydsl_dcp_topk_merge(
        scores,
        local_idx,
        bt,
        indices,
        indptr,
        counts,
        staging,
        rank,
        world,
        topk,
        page,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(counts, want_counts, rtol=0, atol=0)
    want_indptr = torch.zeros(rows + 1, dtype=torch.int32)
    want_indptr[1:] = torch.cumsum(want_counts, 0, dtype=torch.int32)
    torch.testing.assert_close(indptr, want_indptr, rtol=0, atol=0)
    for r in range(rows):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        torch.testing.assert_close(indices[lo:hi], want_slots[r], rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,world,k_loc,topk,page,tie_heavy,seed",
    [
        (4, 8, 64, 128, 16, True, 6),  # tie-heavy: many equal-to-threshold
        (7, 8, 128, 32, 16, False, 31),  # the shape C1 was reproduced on
    ],
)
def test_deterministic_across_runs(rows, world, k_loc, topk, page, tie_heavy, seed):
    """Repeated runs must be bitwise identical AND match the oracle.

    The kernel's block scan reuses one LDS buffer across back-to-back calls, so a
    missing barrier shows up only as a rare scheduling-dependent drift. Two runs
    are nowhere near enough to catch that; loop hard instead.
    """
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page, 512, tie_heavy=tie_heavy, seed=seed
    )
    # Saturate the GPU from a second stream. Whether waves of one block get
    # skewed enough to expose a missing LDS barrier is a scheduling accident;
    # co-resident load makes that accident reliable instead of rare.
    noise = torch.randn(8192, 8192)
    noise_stream = torch.cuda.Stream()
    for rank in range(world):
        want_slots, want_counts = ref_dcp_merge(
            scores, local_idx, bt, rank, k_loc, topk, page
        )
        ref = None
        for it in range(200 // world):
            with torch.cuda.stream(noise_stream):
                for _ in range(3):
                    noise @ noise
            indices, indptr, counts = _run(
                scores, local_idx, bt, rank, world, topk, page
            )
            torch.testing.assert_close(
                counts,
                want_counts,
                rtol=0,
                atol=0,
                msg=f"rank {rank} iter {it}: counts drifted from the oracle",
            )
            for r in range(rows):
                lo, hi = int(indptr[r]), int(indptr[r + 1])
                torch.testing.assert_close(
                    torch.sort(indices[lo:hi]).values,
                    torch.sort(want_slots[r]).values,
                    rtol=0,
                    atol=0,
                    msg=f"rank {rank} iter {it} row {r}: slots differ from oracle",
                )
            if ref is None:
                ref = (indices.clone(), indptr.clone())
            else:
                torch.testing.assert_close(
                    indices,
                    ref[0],
                    rtol=0,
                    atol=0,
                    msg=f"rank {rank} iter {it}: indices not reproducible",
                )
                torch.testing.assert_close(
                    indptr,
                    ref[1],
                    rtol=0,
                    atol=0,
                    msg=f"rank {rank} iter {it}: indptr not reproducible",
                )


def test_topk_larger_than_candidate_count():
    """topk_tokens > n_cand degenerates to "take everything", not garbage."""
    rows, world, k_loc, topk, page = 3, 2, 32, 128, 16
    scores, local_idx, bt = make_case(rows, world, k_loc, topk, page, 64, seed=11)
    for rank in range(world):
        indices, indptr, counts = _run(scores, local_idx, bt, rank, world, topk, page)
        want_slots, want_counts = ref_dcp_merge(
            scores, local_idx, bt, rank, k_loc, topk, page
        )
        torch.testing.assert_close(counts, want_counts, rtol=0, atol=0)
        # Every candidate is a winner, so each rank owns its whole plane.
        assert torch.all(counts == k_loc), f"rank {rank}: {counts} != {k_loc}"
        for r in range(rows):
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            torch.testing.assert_close(
                torch.sort(indices[lo:hi]).values,
                torch.sort(want_slots[r]).values,
                rtol=0,
                atol=0,
                msg=f"rank {rank} row {r} slot mismatch",
            )


def _atom_available():
    """Probe the SYMBOLS this test uses, not just the module.

    triton_filter_and_convert_dcp_index is the decode filter the fused op
    replaces; the companion ATOM change deletes it. Probing the module alone
    would make these cases fail rather than skip once that lands, so the two
    repos could not be merged in either order.
    """
    try:
        from atom.model_ops.dcp_ops import (  # noqa: F401
            dcp_global_pos,
            triton_filter_and_convert_dcp_index,
        )

        return True
    except Exception:  # noqa: BLE001  probe: any import failure means unavailable
        return False


@pytest.mark.skipif(
    not _atom_available(),
    reason="needs an ATOM that still has triton_filter_and_convert_dcp_index",
)
@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("page_size", [16, 64])
def test_matches_triton_filter_and_convert(interleave, page_size):
    """End-to-end parity with today's pipeline: same slots, same indptr.

    Builds the global top-k the old way (gather gids, merge, filter) and the new
    way (fused select + emit), then compares the packed outputs elementwise.
    """
    from atom.model_ops.dcp_ops import (
        dcp_global_pos,
        triton_filter_and_convert_dcp_index,
    )

    import aiter

    rows, world, k_loc, topk = 4, 8, 128, 256
    max_blocks = 512
    rank = 3
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page_size, max_blocks, seed=7
    )

    # --- old path: build global top-k over gids, then filter to owned ---
    gids = torch.empty(rows, world * k_loc, dtype=torch.int32)
    for w in range(world):
        j = local_idx.clamp(min=0).to(torch.int64)
        g = dcp_global_pos(j, w, world, interleave).to(torch.int32)
        gids[:, w * k_loc : (w + 1) * k_loc] = torch.where(local_idx >= 0, g, -1)
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    win = order[:, :topk]
    global_topk = torch.gather(gids, 1, win).contiguous()

    qo_indptr = torch.arange(rows + 1, dtype=torch.int32)
    g_kv_indptr = torch.zeros(rows + 1, dtype=torch.int32)
    old_indptr = torch.zeros(rows + 1, dtype=torch.int32)
    old_counts = torch.zeros(rows, dtype=torch.int32)
    old_indices = torch.zeros(rows * topk, dtype=torch.int32)
    triton_filter_and_convert_dcp_index(
        qo_indptr,
        g_kv_indptr,
        bt,
        global_topk,
        rank,
        world,
        page_size,
        out_kv_indptr=old_indptr,
        owned_counts=old_counts,
        NUM_TOPK_TOKENS=topk,
        out=old_indices,
        cp_kv_cache_interleave_size=interleave,
    )

    # --- new path ---
    staging = torch.empty(rows, k_loc, dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indices = torch.zeros(rows * topk, dtype=torch.int32)
    aiter.flydsl_dcp_topk_merge(
        scores,
        local_idx,
        bt,
        indices,
        indptr,
        counts,
        staging,
        rank,
        world,
        topk,
        page_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(indptr, old_indptr, rtol=0, atol=0)
    total = int(old_indptr[rows])
    torch.testing.assert_close(
        torch.sort(indices[:total]).values,
        torch.sort(old_indices[:total]).values,
        rtol=0,
        atol=0,
    )


def _run(scores, local_idx, bt, rank, world, topk, page):
    import aiter

    rows = scores.shape[0]
    k_loc = scores.shape[1] // world
    staging = torch.empty(rows, k_loc, dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indices = torch.zeros(rows * max(topk, k_loc), dtype=torch.int32)
    aiter.flydsl_dcp_topk_merge(
        scores,
        local_idx,
        bt,
        indices,
        indptr,
        counts,
        staging,
        rank,
        world,
        topk,
        page,
    )
    torch.cuda.synchronize()
    return indices, indptr, counts


def test_short_context_padding_never_emitted():
    """-inf / -1 padded candidates must not reach the output."""
    rows, world, k_loc, topk, page = 3, 4, 32, 64, 16
    # Use 20 real candidates (> page_size=16) so at least two block-table entries
    # are exercised: local indices 0..19 span j//page in {0, 1}.
    n_real = 20
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page, 64, n_local=n_real, seed=8
    )
    # Starve this rank's plane: only n_real real candidates, rest padded.
    rank = 1
    local_idx[:, n_real:] = -1
    scores[:, rank * k_loc + n_real : (rank + 1) * k_loc] = -float("inf")
    indices, indptr, counts = _run(scores, local_idx, bt, rank, world, topk, page)
    assert torch.all(counts <= n_real), f"emitted padded candidates: {counts}"
    # Compare emitted slots against reference — mere non-negativity is not enough.
    want_slots, _ = ref_dcp_merge(scores, local_idx, bt, rank, k_loc, topk, page)
    assert int(indptr[rows]) == int(
        counts.sum()
    ), "indptr tail must equal the total emitted count"
    for r in range(rows):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        got_row = torch.sort(indices[lo:hi]).values
        ref_row = torch.sort(want_slots[r]).values
        torch.testing.assert_close(
            got_row,
            ref_row,
            rtol=0,
            atol=0,
            msg=f"row {r}: emitted slots differ from reference",
        )


def test_zero_owned_is_valid():
    """A rank whose candidates all lose emits nothing, not garbage."""
    rows, world, k_loc, topk, page = 2, 4, 32, 8, 16
    scores, local_idx, bt = make_case(rows, world, k_loc, topk, page, 64, seed=9)
    rank = 2
    scores[:, rank * k_loc : (rank + 1) * k_loc] = -1e30
    _indices, indptr, counts = _run(scores, local_idx, bt, rank, world, topk, page)
    assert torch.all(counts == 0), f"expected no winners, got {counts}"
    assert int(indptr[rows]) == 0


@pytest.mark.parametrize("rows", [1, 17, 256])
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("page_size", [16, 64, 128])
def test_shape_sweep(rows, world, page_size):
    """Counts partition the global top-k AND the emitted slots match the oracle.

    Summing counts alone would pass even if the slots themselves were corrupt,
    so compare the emitted slots per row too (sorted: within-row order for a
    given rank is packed in flat-candidate order, but we only rely on the set).
    """
    k_loc, topk = 64, 128
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page_size, 512, seed=rows + world
    )
    total = torch.zeros(rows, dtype=torch.int32)
    for rank in range(world):
        indices, indptr, counts = _run(
            scores, local_idx, bt, rank, world, topk, page_size
        )
        want_slots, want_counts = ref_dcp_merge(
            scores, local_idx, bt, rank, k_loc, topk, page_size
        )
        torch.testing.assert_close(counts, want_counts, rtol=0, atol=0)
        for r in range(rows):
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            torch.testing.assert_close(
                torch.sort(indices[lo:hi]).values,
                torch.sort(want_slots[r]).values,
                rtol=0,
                atol=0,
                msg=f"rank {rank} row {r} slot mismatch",
            )
        total += counts
    expect = min(topk, world * k_loc)
    assert torch.all(total == expect), f"partition broken: {total} != {expect}"


def _ord(x: torch.Tensor) -> torch.Tensor:
    """float32 -> monotone int32 (mirrors _f32_to_ord in the kernel)."""
    b = x.view(torch.int32)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


@pytest.mark.parametrize("tie_heavy", [False, True])
def test_kernel_threshold_matches_reference(tie_heavy):
    """The block-local radix select finds the same threshold as a sort."""
    from aiter.ops.flydsl.dcp_topk_merge import _debug_threshold

    rows, world, k_loc, topk, page = 4, 8, 64, 128, 16
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page, 64, tie_heavy=tie_heavy, seed=2
    )
    got = _debug_threshold(scores, local_idx, bt, 0, k_loc, topk, page)
    want = torch.sort(_ord(scores), dim=-1, descending=True).values[:, topk - 1]
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("stable", [False, True])
def test_decode_topk_values_pad_with_neg_inf(stable):
    """`values=` writes each index's logit, and pads short rows with -inf.

    This is the precondition flydsl_dcp_topk_merge relies on: padding travels in
    the all-gathered score plane with no liveness flag of its own, so it has to
    lose the global threshold comparison on score alone. A regression to the old
    0.0 pad outranks real (negative) logits and makes this rank under-emit --
    silently, since nothing crashes. Measured before the fix: the owned
    partition collapsed from 256 entries to 30.
    """
    import aiter

    rows, n, k = 4, 4096, 512
    torch.manual_seed(17)
    logits = torch.randn(rows, n, dtype=torch.float32)
    # Rows shorter than k are the whole point -- that is where padding appears.
    lens = torch.tensor([0, 1, 100, n], dtype=torch.int32)

    idx = torch.full((rows, k), -777, dtype=torch.int32)
    vals = torch.full((rows, k), float("nan"), dtype=torch.float32)
    aiter.top_k_per_row_decode(
        logits,
        1,
        lens,
        idx,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        stable=stable,
        values=vals,
    )
    torch.cuda.synchronize()

    pad = idx < 0
    assert bool(
        (vals[pad] == -float("inf")).all()
    ), "padded slots must carry -inf, not 0.0"
    real = ~pad
    gathered = logits.gather(1, idx.clamp(min=0).to(torch.int64))
    torch.testing.assert_close(vals[real], gathered[real], rtol=0, atol=0)
    if pad.any() and real.any():
        assert (
            vals[pad].max() < vals[real].min()
        ), "padding must sort below all real scores"


def test_decode_topk_values_none_is_unchanged():
    """Passing values= must not perturb the indices the op already returned."""
    import aiter

    rows, n, k = 4, 4096, 256
    torch.manual_seed(19)
    logits = torch.randn(rows, n, dtype=torch.float32) - 3.0  # all negative
    lens = torch.tensor([0, 7, 100, n], dtype=torch.int32)

    without = torch.full((rows, k), -777, dtype=torch.int32)
    aiter.top_k_per_row_decode(
        logits,
        1,
        lens,
        without,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        stable=True,
    )
    with_vals = torch.full((rows, k), -777, dtype=torch.int32)
    vals = torch.full((rows, k), float("nan"), dtype=torch.float32)
    aiter.top_k_per_row_decode(
        logits,
        1,
        lens,
        with_vals,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        stable=True,
        values=vals,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(with_vals, without, rtol=0, atol=0)


@benchmark()  # call args become the table's left-hand columns
def test_dcp_topk_merge(rows, world, k_loc, topk, page, tie_heavy):
    """One rank's merge: global-threshold select -> owned, packed KV slots.

    Single-kernel-vs-reference shape: there is no second kernel to race. The
    sequence this op replaces lives in ATOM (a merge + a Triton filter), so it
    is not importable here as a candidate -- `ref_dcp_merge` is the oracle.
    """
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page, 4096, tie_heavy=tie_heavy, seed=42
    )
    rank = world // 2  # a middle plane: exercises the prior-equal sweep
    ref_slots, ref_counts = ref_dcp_merge(
        scores, local_idx, bt, rank, k_loc, topk, page
    )

    # Caller-owned buffers, as production allocates them (see ATOM's
    # dcp_decode_candidate_exchange_fused): the op allocates no device scratch.
    staging = torch.empty(rows, k_loc, dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indices = torch.zeros(rows * max(topk, k_loc), dtype=torch.int32)

    candidates = {
        "flydsl": lambda: aiter.flydsl_dcp_topk_merge(
            scores,
            local_idx,
            bt,
            indices,
            indptr,
            counts,
            staging,
            rank,
            world,
            topk,
            page,
        ),
    }

    n_cand = world * k_loc
    # Memory-side op: the radix select re-reads the candidate plane, and the
    # emit walks this rank's own plane plus its block table.
    nbytes = (
        rows * n_cand * scores.element_size()  # gathered scores
        + rows * k_loc * local_idx.element_size()  # local_idx
        + bt.numel() * bt.element_size()  # block table
        + rows * k_loc * 4 * 2  # staging + emitted indices
    )
    flops = 0  # selection + address arithmetic; no useful FLOPs to count

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        torch.cuda.synchronize()
        err = checkAllclose(
            ref_counts.to(dtypes.fp32),
            counts.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"{name}: owned_counts",
        )
        # The slot SET per row is the contract; within-row order is not.
        for r in range(rows):
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            checkAllclose(
                torch.sort(ref_slots[r]).values.to(dtypes.fp32),
                torch.sort(indices[lo:hi]).values.to(dtypes.fp32),
                rtol=0,
                atol=0,
                msg=f"{name}: row {r} slots",
                printLog=False,
            )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us > 0 else 0
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us > 0 else 0
        ret[f"{name} err"] = err
    return ret


# The perf sweep is driven by main(), not by pytest: its args are shape lists,
# so pytest would collect it as a zero-fixture test and error. The correctness
# tests above are the pytest surface of this file.
test_dcp_topk_merge.__test__ = False


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL DCP TopK merge unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL DCP decode TopK merge correctness + perf sweep",
    )
    # rows == num_decode_tokens: the decode batch this rank scheduled.
    parser.add_argument("-b", "--rows", type=int, nargs="*", default=[1, 32, 128, 256])
    parser.add_argument("-w", "--world", type=int, nargs="*", default=[8])
    # Production ships k_loc == topk == index_topk == 2048.
    parser.add_argument("-k", "--k-loc", type=int, nargs="*", default=[2048])
    parser.add_argument("--topk", type=int, nargs="*", default=[2048])
    parser.add_argument("-p", "--page", type=int, nargs="*", default=[64])
    parser.add_argument(
        "--tie-heavy", type=int, nargs="*", default=[0, 1], choices=[0, 1]
    )
    args = parser.parse_args()

    df = []
    for rows, world, k_loc, topk, page, tie in itertools.product(
        args.rows, args.world, args.k_loc, args.topk, args.page, args.tie_heavy
    ):
        df.append(test_dcp_topk_merge(rows, world, k_loc, topk, page, bool(tie)))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "FlyDSL DCP TopK merge summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
