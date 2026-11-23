"""规划编排：原依赖优先，排不下才枚举“加一条不成环等待”的方案。

同一请求中，每种张量干扰关系（干扰矩阵 + 尺寸/对齐）只做一次布局求解：
不改变干扰关系的等待直接跳过；产生相同干扰关系的等待复用缓存。对后续
关系传入“当前最优占用”作为剪枝上界；由于该上界是已实现的可行占用、不
低于任何关系的真实最优，受限求解仍返回真实最优，缓存始终精确。
"""

from __future__ import annotations

from .models import AllocationResult, PlanOutcome, PlanningInput, Wait
from .solver import (
    build_base_edges,
    build_interference,
    candidate_waits,
    interference_signature,
    reachability,
    solve_allocation,
    validate_input,
    validate_tensor_lifecycles,
)


def plan(data: PlanningInput) -> PlanOutcome:
    validate_input(data)
    base_edges = build_base_edges(data)
    nodes = data.all_operators()
    base_reach = reachability(base_edges, nodes)

    # 干扰关系签名 -> 该关系下的**精确**最优布局；同一请求中每种关系只求一次
    layout_cache: dict[tuple, AllocationResult] = {}

    base_interfere = build_interference(data, base_reach)
    validate_tensor_lifecycles(data, base_reach)
    base_signature = interference_signature(data, base_interfere)
    base_result = solve_allocation(data, base_interfere)
    layout_cache[base_signature] = base_result

    # 1) 原依赖下能排进预算：保持并行，不加任何等待
    if base_result.feasible:
        return PlanOutcome(
            feasible=True,
            total_bytes=base_result.total_bytes,
            budget_bytes=data.budget_bytes,
            offsets=base_result.offsets,
            added_wait=None,
            note="原依赖下即可在预算内完成显存复用，未新增跨流等待",
            bins=base_result.bins,
        )

    # 2) 排不下：枚举至多一条新增的、不成环的跨流等待。
    # 候选按 (waiter, waitee) 字典序枚举；只在占用严格更小时更新，
    # 因此同占用下自然保留字典序最小的等待，结果确定。
    best = base_result
    best_wait: Wait | None = None
    for wait in candidate_waits(data, base_edges, base_reach):
        edges = set(base_edges)
        # waiter 等 waitee 完成 -> happens-before 边 waitee -> waiter
        edges.add((wait.waitee, wait.waiter))
        reach = reachability(edges, nodes)
        validate_tensor_lifecycles(data, reach)
        interfere = build_interference(data, reach)
        signature = interference_signature(data, interfere)

        # 干扰关系未变 -> 布局与原依赖完全相同，不可能改善占用
        if signature == base_signature:
            continue

        cached = layout_cache.get(signature)
        if cached is not None:
            result = cached
        else:
            # 传入当前最优占用作为剪枝上界：只需判断该关系能否严格更优。
            # 剪枝上界始终不低于该关系的真实最优（它是已实现的可行占用），
            # 因此求解返回的仍是真实最优，受限与否结果都可安全写入缓存。
            result = solve_allocation(
                data, interfere, upper_bound=best.total_bytes
            )
            layout_cache[signature] = result
        if result.total_bytes < best.total_bytes:
            best = result
            best_wait = wait

    if best.feasible:
        assert best_wait is not None
        return PlanOutcome(
            feasible=True,
            total_bytes=best.total_bytes,
            budget_bytes=data.budget_bytes,
            offsets=best.offsets,
            added_wait=best_wait,
            note=(
                f"原依赖下最小占用 {base_result.total_bytes} 字节超出预算，"
                f"新增一条跨流等待 {best_wait.waiter!r} 等待 "
                f"{best_wait.waitee!r} 完成后，占用降至 {best.total_bytes} 字节"
            ),
            bins=best.bins,
        )

    # 3) 加一条等待仍排不下：无解。报告最小占用，偏移与 bins 留空。
    return PlanOutcome(
        feasible=False,
        total_bytes=best.total_bytes,
        budget_bytes=data.budget_bytes,
        offsets={},
        added_wait=best_wait,
        note=(
            "即使新增一条不成环的跨流等待，最小显存占用仍超出预算，无解"
            if best_wait is not None
            else "不存在可新增的合法跨流等待，且原依赖下最小占用超出预算，无解"
        ),
        best_effort_total_bytes=best.total_bytes,
        bins=[],
    )
