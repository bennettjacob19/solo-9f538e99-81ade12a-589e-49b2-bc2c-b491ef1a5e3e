"""依赖分析与最优显存布局求解器。

只使用 Python 标准库。张量数上限为 8。

布局模型
========
两个张量允许在字节上重叠，当且仅当在依赖偏序中一个张量的最后消费者
happens-before 另一个张量的生产者（见 ``build_interference``）。其余张量对
构成“干扰边”：在某些合法时序下会同时在用，字节区间不得重叠。

与旧实现的区别
--------------
旧实现把“互不干扰”的张量强行放进同一个 bin、强制共享**同一个起始偏移**，
再按 bin 拼接。这会错过两类合法布局：

1. 两个互不干扰的张量本可复用同一段显存的**不同偏移**（一个占头部、另一个
   占尾部的洞），bin 划分却只能让它们同起点或完全错开；
2. 对齐取整产生的碎片无法被后续张量利用。

本实现改为枚举张量的**放置顺序**（n! 种，n<=8 时最多 40320 种）：每个
张量在其放置时刻的候选偏移只有 0 与所有已放置张量（含非干扰张量，以便
利用它们尾部的洞）尾部的对齐值，取其中最小的可行值。枚举全部顺序即覆盖
每个最优布局对应的最左放置序列，从而得到允许碎片化留洞的精确最优布局。

搜索规模只取决于张量数 n（O(n!·n²)）与各张量的尺寸/对齐，**不随显存
字节数增长**：候选偏移只有 0 和已放置张量的对齐尾部，从不按字节扫描。
"""

from __future__ import annotations

from itertools import permutations

from .models import AllocationResult, PlanningError, PlanningInput, TensorSpec, Wait

MAX_OPERATORS = 12
MAX_TENSORS = 8


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def validate_input(data: PlanningInput) -> None:
    if len(data.streams) != 2:
        raise PlanningError("必须且只能提供两条 CUDA 流")

    all_ops: list[str] = []
    for stream_id, ops in data.streams.items():
        if not ops:
            raise PlanningError(f"流 {stream_id!r} 不能为空")
        if len(set(ops)) != len(ops):
            raise PlanningError(f"流 {stream_id!r} 内算子 id 重复: {ops}")
        all_ops.extend(ops)

    if len(all_ops) > MAX_OPERATORS:
        raise PlanningError(f"算子总数 {len(all_ops)} 超过上限 {MAX_OPERATORS}")
    if len(set(all_ops)) != len(all_ops):
        raise PlanningError("两条流之间算子 id 不允许重复")

    if len(data.tensors) > MAX_TENSORS:
        raise PlanningError(f"张量总数 {len(data.tensors)} 超过上限 {MAX_TENSORS}")
    if not data.tensors:
        raise PlanningError("至少需要规划一个张量")

    op_set = set(all_ops)
    tensor_names: set[str] = set()
    for t in data.tensors:
        if t.name in tensor_names:
            raise PlanningError(f"张量 id 重复: {t.name}")
        tensor_names.add(t.name)
        if t.producer not in op_set:
            raise PlanningError(f"张量 {t.name} 的生产算子 {t.producer!r} 不存在")
        if not t.consumers:
            raise PlanningError(f"张量 {t.name} 必须至少有一个消费算子")
        if len(set(t.consumers)) != len(t.consumers):
            raise PlanningError(f"张量 {t.name} 的消费算子有重复")
        for c in t.consumers:
            if c not in op_set:
                raise PlanningError(f"张量 {t.name} 的消费算子 {c!r} 不存在")
        if t.producer in t.consumers:
            raise PlanningError(f"张量 {t.name} 的生产算子不能同时是消费算子")
        if t.size <= 0:
            raise PlanningError(f"张量 {t.name} 的字节数必须为正整数")
        if t.alignment <= 0 or (t.alignment & (t.alignment - 1)) != 0:
            raise PlanningError(
                f"张量 {t.name} 的对齐 {t.alignment} 不是 2 的幂"
            )

    if data.budget_bytes <= 0:
        raise PlanningError("显存预算必须为正整数")

    wait_pairs: set[tuple[str, str]] = set()
    for w in data.waits:
        if w.waiter not in op_set or w.waitee not in op_set:
            raise PlanningError(
                f"跨流等待引用了不存在的算子: {w.waiter!r} -> {w.waitee!r}"
            )
        if w.waiter == w.waitee:
            raise PlanningError("跨流等待不能是算子等待自身")
        pair = (w.waiter, w.waitee)
        if pair in wait_pairs:
            raise PlanningError(f"重复的跨流等待: {w.waiter!r} -> {w.waitee!r}")
        wait_pairs.add(pair)


# --------------------------------------------------------------------------- #
# 依赖图 / 可达性
# --------------------------------------------------------------------------- #
def build_base_edges(data: PlanningInput) -> set[tuple[str, str]]:
    """流内顺序边 + 已有跨流等待边。

    Wait(waiter=u, waitee=v) 表示 u 必须等 v 完成，即 v happens-before u，
    因此依赖边方向为 v -> u。
    """
    edges: set[tuple[str, str]] = set()
    for ops in data.streams.values():
        for a, b in zip(ops, ops[1:]):
            edges.add((a, b))
    for w in data.waits:
        edges.add((w.waitee, w.waiter))
    return edges


def reachability(
    edges: set[tuple[str, str]], nodes: list[str]
) -> dict[str, frozenset[str]]:
    """返回每个节点经有向边可到达的后继节点集合（不含自身）。

    边 u->v 表示 u happens-before v，因此 v 在 reach[u] 中。
    若图存在环会抛出 PlanningError。
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1

    # Kahn 拓扑排序；处理 u 时把 u 及其祖先传播给其每个后继 v，
    # 因此 ancestors[v] 最终是“能到达 v 的全部节点”
    ancestors: dict[str, set[str]] = {n: set() for n in nodes}
    queue = [n for n in nodes if indeg[n] == 0]
    visited = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        visited += 1
        for v in adj[u]:
            ancestors[v].update(ancestors[u])
            ancestors[v].add(u)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if visited != len(nodes):
        raise PlanningError("依赖图存在环，给定的跨流等待不成环约束被破坏")

    # 把“v 的祖先”翻转成“u 的后继可达集”
    reach: dict[str, set[str]] = {n: set() for n in nodes}
    for v, anc in ancestors.items():
        for u in anc:
            reach[u].add(v)
    return {n: frozenset(reach[n]) for n in nodes}


# --------------------------------------------------------------------------- #
# 张量生命周期与干扰图
# --------------------------------------------------------------------------- #
def _happens_before(
    reach: dict[str, frozenset[str]], a: str, b: str
) -> bool:
    """算子 a 是否 happens-before 算子 b。"""
    return a == b or b in reach[a]


def _tensor_born(data: PlanningInput, t: TensorSpec) -> str:
    return t.producer


def _tensor_last_consumers(data: PlanningInput, t: TensorSpec) -> tuple[str, ...]:
    """可能成为“最后完成的消费者”的算子集合。

    同一流内的多个消费者，靠后的一定最后完成，只保留每个流里最靠后的那个；
    不同流的消费者之间无偏序时都要保留。
    """
    stream_of: dict[str, str] = {}
    position: dict[str, int] = {}
    for stream_id, ops in data.streams.items():
        for i, op in enumerate(ops):
            stream_of[op] = stream_id
            position[op] = i

    latest_per_stream: dict[str, str] = {}
    for c in t.consumers:
        s = stream_of[c]
        if s not in latest_per_stream or position[c] > position[latest_per_stream[s]]:
            latest_per_stream[s] = c
    return tuple(latest_per_stream.values())


def _tensor_dead_before_born(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
    dead: TensorSpec,
    born: TensorSpec,
) -> bool:
    """dead 的整个生命周期是否严格先于 born 的生产（所有合法时序下）。"""
    born_op = _tensor_born(data, born)
    # dead 的每一个“可能最后完成”的消费者都必须 happens-before born 的生产者
    return all(
        _happens_before(reach, c, born_op)
        for c in _tensor_last_consumers(data, dead)
    )


def build_interference(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
) -> list[list[bool]]:
    """干扰矩阵：interfere[i][j] 为 True 表示存在两张量同时在用的合法时序。"""
    n = len(data.tensors)
    interfere = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = data.tensors[i], data.tensors[j]
            a_before_b = _tensor_dead_before_born(data, reach, a, b)
            b_before_a = _tensor_dead_before_born(data, reach, b, a)
            if not (a_before_b or b_before_a):
                interfere[i][j] = interfere[j][i] = True
    return interfere


def validate_tensor_lifecycles(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
) -> None:
    """生产必须 happens-before 每一个消费，否则输入本身就是数据竞争。"""
    for t in data.tensors:
        for c in t.consumers:
            if not _happens_before(reach, t.producer, c):
                raise PlanningError(
                    f"张量 {t.name} 的生产算子 {t.producer!r} 不在消费算子 "
                    f"{c!r} 之前，缺少依赖（存在非法的数据竞争）"
                )


# --------------------------------------------------------------------------- #
# 干扰关系签名：同一请求中相同干扰关系只求解一次
# --------------------------------------------------------------------------- #
def interference_signature(
    data: PlanningInput,
    interfere: list[list[bool]],
) -> tuple:
    """可哈希的干扰关系签名。

    干扰矩阵决定“谁能与谁重叠”，而尺寸与对齐决定布局代价；同一请求里
    张量集合不变，因此签名 = 上三角干扰位 + 每个张量的 (尺寸, 对齐)。
    """
    n = len(data.tensors)
    bits = tuple(
        bool(interfere[i][j])
        for i in range(n)
        for j in range(i + 1, n)
    )
    specs = tuple((t.size, t.alignment) for t in data.tensors)
    return (n, bits, specs)


# --------------------------------------------------------------------------- #
# 最优布局：枚举放置顺序 + 每步最左可行对齐偏移
# --------------------------------------------------------------------------- #
def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def solve_allocation(
    data: PlanningInput,
    interfere: list[list[bool]],
    upper_bound: int | None = None,
) -> AllocationResult:
    """在给定干扰图下求满足对齐的最小连续显存布局（精确最优）。

    枚举全部 n! 个放置顺序（``itertools.permutations`` 按字典序产出）。
    每个顺序中，张量的候选偏移只有 0 与所有**已放置**张量（含非干扰张量，
    以便钻进它们尾部的洞）尾部按本张量对齐向上取整的值，取其中最小的可行
    值；可行性只检查干扰张量，非干扰张量允许字节重叠。

    候选点数量与显存字节数无关，搜索规模只取决于张量数 n：
    O(n!·n²)，n<=8 时最多 40320 个顺序。

    剪枝：先用下标序贪心造一个可行解作为初始上界；若调用方给出
    ``upper_bound``（只关心是否存在占用不超过它的布局），取二者中较小者
    作为初始上界。单个放置顺序内跨度单调不减，部分跨度一旦**严格大于**
    当前上界即可中止该顺序。上界随更优解不断收紧。注意：传入的
    upper_bound 只用于剪枝，返回值始终是真实最小占用（剪枝不会误杀等于
    上界的最优解），调用方无需区分。

    完备性：设存在最优布局 L，按 L 中起始偏移从小到大（并列按张量 id）作为
    放置顺序，L 给每个张量的偏移在其放置时刻都是一个可行候选，而“取最左
    可行偏移”不会比 L 的偏移更靠右——故该顺序的贪心放置占用不大于 L，
    枚举全部顺序必然复现最优占用。

    同占用下的确定性 tie-break：偏移向量 (off_0, ..., off_{n-1}) 字典序最小。
    """
    tensors = data.tensors
    n = len(tensors)
    sizes = [t.size for t in tensors]
    aligns = [t.alignment for t in tensors]

    # 预计算每个张量的干扰邻接位掩码，加速可行性判定
    conflict_mask = [0] * n
    for i in range(n):
        m = 0
        for j in range(n):
            if interfere[i][j]:
                m |= 1 << j
        conflict_mask[i] = m

    def place_one(i: int, offsets: list[int], placed_mask: int) -> int:
        # 候选起点：0，以及每个已放置张量 j（含非干扰张量）的对齐尾部。
        # 非干扰张量的尾部必须纳入，否则会漏掉其后面洞里的可行位置。
        # n<=8，用线性插入维护有序候选，避免集合/排序开销。
        candidates = [0]
        placed_bits = placed_mask
        ai = aligns[i]
        while placed_bits:
            b = placed_bits & -placed_bits
            j = b.bit_length() - 1
            c = _align_up(offsets[j] + sizes[j], ai)
            if c not in candidates:
                candidates.append(c)
            placed_bits ^= b
        candidates.sort()

        cbm = conflict_mask[i] & placed_mask
        si = sizes[i]
        for c in candidates:
            end = c + si
            # 只要求与已放置的干扰张量区间不相交；非干扰张量允许重叠
            bits = cbm
            ok = True
            while bits:
                b = bits & -bits
                j = b.bit_length() - 1
                if not (end <= offsets[j] or offsets[j] + sizes[j] <= c):
                    ok = False
                    break
                bits ^= b
            if ok:
                return c
        raise AssertionError("候选 0 始终可行，不可能走到这里")

    # 初始上界：下标序贪心最左放置，必然可行
    greedy_off = [0] * n
    mask = 0
    for i in range(n):
        greedy_off[i] = place_one(i, greedy_off, mask)
        mask |= 1 << i
    greedy_total = max(greedy_off[i] + sizes[i] for i in range(n))
    incumbent = greedy_total
    best_offsets: tuple[int, ...] = tuple(greedy_off)

    # 调用方给定上界时用它加强剪枝；只取不大于贪心得分的值，
    # 避免初始最优记录被一个无法实现的上界污染
    prune_bound = incumbent
    if upper_bound is not None and upper_bound < prune_bound:
        prune_bound = upper_bound

    for order in permutations(range(n)):
        offsets = [0] * n
        placed_mask = 0
        span = 0
        aborted = False
        for i in order:
            c = place_one(i, offsets, placed_mask)
            offsets[i] = c
            placed_mask |= 1 << i
            end = c + sizes[i]
            if end > span:
                span = end
                # 安全剪枝：本顺序跨度已严格超剪枝上界，最终只会更大
                if span > prune_bound:
                    aborted = True
                    break
        if aborted:
            continue

        off_tuple = tuple(offsets)
        if span < incumbent or (span == incumbent and off_tuple < best_offsets):
            incumbent = span
            best_offsets = off_tuple
            # 更优解收紧剪枝上界（不低于真实最优，剪枝始终安全）
            if incumbent < prune_bound:
                prune_bound = incumbent

    offsets_out = {
        tensors[i].name: best_offsets[i] for i in range(n)
    }
    bins_out = _bins_from_offsets(tensors, best_offsets)

    return AllocationResult(
        feasible=incumbent <= data.budget_bytes,
        total_bytes=incumbent,
        offsets=offsets_out,
        bins=bins_out,
    )


def _bins_from_offsets(
    tensors: list[TensorSpec], offsets: tuple[int, ...]
) -> list[list[str]]:
    """按相同起始偏移把张量分箱；箱按起始偏移升序，箱内 id 字典序。"""
    groups: dict[int, list[str]] = {}
    for i, t in enumerate(tensors):
        groups.setdefault(offsets[i], []).append(t.name)
    return [sorted(groups[start]) for start in sorted(groups)]


# --------------------------------------------------------------------------- #
# 候选跨流等待枚举
# --------------------------------------------------------------------------- #
def candidate_waits(
    data: PlanningInput,
    base_edges: set[tuple[str, str]],
    reach: dict[str, frozenset[str]],
) -> list[Wait]:
    """枚举所有可能新增的、不成环且非冗余的跨流等待边。

    规则：
    - 必须跨流（流内顺序边已存在）；
    - 边 u->v 若已存在（u 已可达 v）则冗余，跳过；
    - 若 v 已可达 u，加入 u->v 必然成环，跳过；
    - 自环跳过。
    按 (waiter, waitee) 字典序返回，保证结果确定性。
    """
    stream_of: dict[str, str] = {}
    for stream_id, ops in data.streams.items():
        for op in ops:
            stream_of[op] = stream_id

    out: list[Wait] = []
    nodes = data.all_operators()
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if stream_of[u] == stream_of[v]:
                continue
            if v in reach[u]:
                continue  # 已有依赖（边或传递可达），冗余
            if u in reach[v]:
                continue  # 反向已可达，加入即成环
            out.append(Wait(waiter=u, waitee=v))
    out.sort(key=lambda w: (w.waiter, w.waitee))
    return out
