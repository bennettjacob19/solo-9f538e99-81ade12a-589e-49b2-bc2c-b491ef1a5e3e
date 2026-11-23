"""补充边界：同流多消费者、bin 内最严格对齐、候选等待完整性、单张量。"""

from __future__ import annotations

from app.planner import service as service_module
from app.planner.solver import (
    build_base_edges,
    candidate_waits,
    reachability,
    solve_allocation,
)
from app.planner.service import plan

from conftest import assert_layout_valid, make_input


def test_same_stream_multiple_consumers_uses_last_one():
    # T 被 a2 和 a4 消费：a4 才是最后消费者。
    # - a5 出生的 U 与 T 生命周期不相交 -> 干扰矩阵中不干扰（允许复用）
    # - a3 出生、a4 消费的 V 与 T 同时在用 -> 干扰矩阵中必须干扰（禁止重叠）
    from app.planner.solver import build_base_edges, build_interference, reachability

    data = make_input(
        ["a1", "a2", "a3", "a4", "a5", "a6"],
        ["b1", "b2"],
        [
            {"name": "T", "producer": "a1", "consumers": ["a2", "a4"], "size": 100, "alignment": 1},
            {"name": "U", "producer": "a5", "consumers": ["a6"], "size": 100, "alignment": 1},
            {"name": "V", "producer": "a3", "consumers": ["a4"], "size": 100, "alignment": 1},
        ],
        budget=200,
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    interfere = build_interference(data, reach)
    i_t, i_u, i_v = 0, 1, 2
    assert interfere[i_t][i_u] is False       # T 先死 U 才生，可复用
    assert interfere[i_t][i_v] is True        # 同流内 T 活到 a4，V 生于 a3
    assert interfere[i_u][i_v] is False       # V 死于 a4，U 生于 a5

    out = plan(data)
    assert out.feasible and out.added_wait is None
    assert out.total_bytes == 200             # T/V 必须占两块，U 复用其中一块
    assert_layout_valid(data, out, True)


def test_bin_uses_strictest_alignment_for_shared_offset():
    # 三个生命周期互不相交的张量，对齐分别为 4/128/8；
    # 共享偏移必须同时满足三者，即起始偏移按 128 对齐
    data = make_input(
        ["a1", "a2", "a3", "a4"],
        ["b1", "b2"],
        [
            {"name": "T1", "producer": "a1", "consumers": ["a2"], "size": 10, "alignment": 4},
            {"name": "T2", "producer": "a2", "consumers": ["a3"], "size": 10, "alignment": 128},
            {"name": "T3", "producer": "a3", "consumers": ["a4"], "size": 10, "alignment": 8},
        ],
        budget=1000,
    )
    out = plan(data)
    assert out.feasible
    assert out.offsets["T1"] == out.offsets["T2"] == out.offsets["T3"]
    assert out.offsets["T1"] % 128 == 0
    assert_layout_valid(data, out, True)


def test_candidate_waits_complete_and_cross_stream_only():
    data = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 10, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 10, "alignment": 1},
        ],
        budget=1000,
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    cands = candidate_waits(data, edges, reach)
    pairs = {(w.waiter, w.waitee) for w in cands}
    # 2x2 全部跨流有序对，且没有任何流内候选
    expected = {
        ("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2"),
        ("b1", "a1"), ("b1", "a2"), ("b2", "a1"), ("b2", "a2"),
    }
    assert pairs == expected
    # 枚举顺序确定性：按 (waiter, waitee) 升序
    keys = [(w.waiter, w.waitee) for w in cands]
    assert keys == sorted(keys)


def test_candidate_waits_excludes_redundant_and_cyclic():
    # 已有等待 b2 等 a1（边 a1->b2）：
    # - b2 等 a1 本身冗余（已可达）；a1 等 b2 会成环；二者都不应出现
    data = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 10, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 10, "alignment": 1},
        ],
        budget=1000,
        waits=[("b2", "a1")],
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    pairs = {(w.waiter, w.waitee) for w in candidate_waits(data, edges, reach)}
    assert ("b2", "a1") not in pairs  # 冗余
    assert ("a1", "b2") not in pairs  # 成环


def test_single_tensor():
    data = make_input(
        ["a1", "a2"], ["b1"],
        [{"name": "T", "producer": "a1", "consumers": ["a2"], "size": 77, "alignment": 16}],
        budget=80,
    )
    out = plan(data)
    assert out.feasible
    assert out.offsets == {"T": 0}
    assert out.total_bytes == 77
    assert out.added_wait is None
    assert_layout_valid(data, out, True)


def test_bins_group_by_same_start_offset_sorted():
    # bins 必须按相同起始偏移分组：箱按偏移升序、箱内张量 id 字典序
    data = make_input(
        ["a1", "a2", "a3", "a4", "a5"],
        ["b1", "b2"],
        [
            {"name": "TS1", "producer": "a1", "consumers": ["a2"], "size": 1, "alignment": 1},
            {"name": "TS2", "producer": "b1", "consumers": ["b2"], "size": 1, "alignment": 1},
            {"name": "TB", "producer": "a3", "consumers": ["a4"], "size": 100, "alignment": 1},
        ],
        budget=100,
        waits=[("a3", "b2")],
    )
    out = plan(data)
    assert out.feasible
    # TB 与 TS1 同偏移 0 一组，TS2 单独偏移 1 一组
    assert out.bins == [["TB", "TS1"], ["TS2"]]
    # 与 offsets 自洽
    for group in out.bins:
        starts = {out.offsets[name] for name in group}
        assert len(starts) == 1
    grouped = {name for group in out.bins for name in group}
    assert grouped == set(out.offsets)


def test_infeasible_response_has_empty_offsets_and_bins():
    # 两个 100B 跨流干扰张量，任何单等待最多把占用降到 100；预算 50 必无解
    data = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=50,
    )
    out = plan(data)
    assert not out.feasible
    assert out.offsets == {}
    assert out.bins == []
    assert out.best_effort_total_bytes == 100


def test_layout_solved_once_per_interference_relation():
    # 紧预算迫使枚举全部候选等待；不同等待产生相同干扰关系时不得重复求解
    data = make_input(
        ["a1", "a2", "a3"], ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=1,
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    n_candidates = len(candidate_waits(data, edges, reach))
    assert n_candidates > 2

    orig = solve_allocation
    calls = {"n": 0}

    def counting(inner_data, interfere, upper_bound=None):
        calls["n"] += 1
        return orig(inner_data, interfere, upper_bound)

    service_module.solve_allocation = counting
    try:
        plan(data)
    finally:
        service_module.solve_allocation = orig

    # 该实例只有“干扰/不干扰”两种关系：原依赖 1 次 + 新增关系至多 1 次
    assert calls["n"] <= 2
    assert calls["n"] < n_candidates


def test_search_size_independent_of_byte_magnitude():
    # 干扰关系不变、字节数整体放大，求解次数与结果结构不变
    import time

    base = make_input(
        ["a1", "a2", "a3", "a4"], ["b1", "b2"],
        [
            {"name": "T1", "producer": "a1", "consumers": ["a2"], "size": 13, "alignment": 4},
            {"name": "T2", "producer": "a2", "consumers": ["a3"], "size": 71, "alignment": 8},
            {"name": "T3", "producer": "a3", "consumers": ["a4"], "size": 29, "alignment": 2},
        ],
        budget=10**6,
    )
    huge = make_input(
        ["a1", "a2", "a3", "a4"], ["b1", "b2"],
        [
            {"name": "T1", "producer": "a1", "consumers": ["a2"], "size": 13 * 10**9, "alignment": 4},
            {"name": "T2", "producer": "a2", "consumers": ["a3"], "size": 71 * 10**9, "alignment": 8},
            {"name": "T3", "producer": "a3", "consumers": ["a4"], "size": 29 * 10**9, "alignment": 2},
        ],
        budget=10**30,
    )
    t0 = time.perf_counter()
    out_base = plan(base)
    t_base = time.perf_counter() - t0
    t0 = time.perf_counter()
    out_huge = plan(huge)
    t_huge = time.perf_counter() - t0
    # 布局结构（偏移比例）一致，耗时同量级（不允许按字节扫描）
    assert out_base.bins == out_huge.bins
    assert t_huge < max(0.5, t_base * 5 + 0.1)
