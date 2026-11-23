# CUDA 双流显存离线规划服务

在边缘推理部署前，对两条 CUDA 流的算子/张量做**离线显存复用规划**。
给定两条流的算子顺序、已有的跨流等待（依赖）、每个张量的生产算子与全部消费算子、
字节数、偏移对齐要求与显存预算，接口返回每个张量在同一块显存中的字节偏移；
当原依赖下任何合法时序都无法满足预算时，允许新增**一条不成环**的跨流等待后再规划，
仍然排不下则报告无解。

## 规划语义（核心规则）

1. 每个张量从其**生产算子开始**到**最后一个消费算子执行完成为止**都算在用（live）。
2. 在**任何符合依赖关系的时序**下，同时在用的两个张量所占字节区间不得重叠。
   实现上用偏序（happens-before）判定：张量 `a` 先于 `b` 死亡、`b` 才出生，即
   `last_consumer(a)` 可到达 `producer(b)`，二者才允许复用同一段显存；方向反过来同理。
   两个方向都不可达时，存在让二者同时在用的合法时序，必须错开。
3. 流内按提交顺序建立全序依赖；已有跨流等待是额外的有向边。整条依赖图必须是 DAG。
4. 仅当原依赖下最优布局仍超预算时，才允许增加一条跨流等待边；只考虑不成环的边。
   在所有可行方案中取占用最小的，占用相同取（等待方算子 id, 被等方算子 id）字典序最小的。
5. 每个张量的偏移必须满足自身对齐要求（对齐量必须为 2 的幂）。
6. 约束：算子总数 ≤ 12，张量总数 ≤ 8，两条流各自的算子 id 不允许重复。

## 布局求解

- 两个张量“不干扰”只意味着**允许**字节重叠，并不强制共享同一个起始偏移：
  互不干扰的张量可以落在同一段显存的不同偏移（例如大张量区间内的对齐洞），
  从而得到比“按组共享起始偏移再拼接”更小的占用。
- 求解器枚举全部放置顺序（n!，n≤8 时最多 40320 种），每个张量在放置时刻只
  考虑 `0` 与已放置张量尾部按自身对齐向上取整的候选，取最小可行偏移。
  候选数量与显存字节数无关，**搜索规模不随字节数增长**。
- 同一请求中每种张量干扰关系（干扰矩阵 + 尺寸/对齐签名）只求解一次：
  不改变干扰关系的等待直接跳过，产生相同关系的等待复用缓存布局。

## 目录

```
app/
  main.py          FastAPI 入口与请求模型
  planner/
    models.py      输入校验后的内部数据结构
    solver.py      依赖图、干扰图与最优显存布局（标准库实现）
    service.py     规划编排：原依赖优先，必要时加一条等待（含关系级缓存）
tests/             pytest 自测（含属性化随机校验与暴力预言机对照）
Dockerfile
docker-compose.yml
requirements.txt
```

## 本地运行

```bash
docker compose up --build
# 打开 http://localhost:8000/docs 提交规划请求
```

不用 Docker 时：

```bash
python3 -m venv --without-pip .venv && .venv/bin/python <(curl -sS https://bootstrap.pypa.io/get-pip.py)
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 自测

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## 请求示例

`POST /plan`，请求体见 `examples/request.json`，响应字段：

- `feasible`: 是否在预算内排下
- `total_bytes`: 本次布局实际占用的连续字节数
- `budget_bytes`: 回显预算
- `offsets`: 张量 id -> 字节偏移
- `bins`: 按相同起始偏移分组的张量 id（箱按偏移升序，箱内 id 字典序）
- `added_wait`: 新增的跨流等待，未新增为 `null`
- `note`: 说明信息
- 无解时 `feasible=false`，`offsets` 与 `bins` 均为空，并通过
  `best_effort_total_bytes` 给出（加至多一条等待后）能达到的最小占用，
  便于调用方判断预算缺口。
