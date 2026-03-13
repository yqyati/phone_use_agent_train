# InfiniteWeb GUI Agent GRPO Training with verl

基于 [InfiniteWeb-Dataset](https://github.com/...) 环境，使用 verl 框架对 Qwen3-VL 做 GRPO 强化学习训练。

与 AndroidWorld 版本相比的核心优势：
- **并发更高**：Playwright 浏览器 ~150 MB/实例 vs Android 模拟器 ~2 GB → 同等内存下可跑 4× 并发
- **数据规模大**：626 个网站 × 平均 6-7 个任务 ≈ **4000+ 训练任务**（约为 AndroidWorld 的 17×）
- **无需 Docker/KVM**：浏览器 worker 就是普通 Python 进程，无硬件虚拟化依赖
- **数据准备离线**：直接读数据集文件，无需先运行服务器

## 文件结构

```
examples/wb_agent/
├── config/
│   ├── wb_agent_grpo.yaml                        # GRPO 训练配置
│   └── interaction_config/
│       └── wb_agent_interaction_config.yaml      # Worker 端口池配置（64个）
├── data_preprocess/
│   └── prepare_wb_agent_data.py                  # 离线生成 train/test.parquet
├── env_server/
│   └── web_env_server.py                         # Playwright + FastAPI 环境服务器
├── run_wb_agent_grpo.sh                          # 一键启动训练
├── wb_agent_reward.py                            # 奖励函数
└── README.md

verl/interactions/
└── web_world_interaction.py                      # 核心 Interaction 类
```

---

## 快速开始

### 第一步：安装依赖

```bash
pip install playwright fastapi uvicorn aiofiles
#playwright install chromium
export PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/data/android/yqy/tests/wbagent/chrome-linux64/chrome
```


### 第二步：启动浏览器 Worker 池

启动 64 个并行 worker，每个 worker 独占一个 Playwright Chromium 实例：

```bash
DATASET_DIR=/data/android/yqy/tests/wbagent/InfiniteWeb-Dataset

for i in $(seq 1 64); do
  python examples/wb_agent/env_server/web_env_server.py \
    --port $((6000+i)) \
    --dataset-dir "$DATASET_DIR" &
done
```

等待所有 worker 就绪：

```bash
sleep 5
for i in $(seq 1 64); do
  curl -s http://localhost:$((6000+i))/health && echo " port $((6000+i)) OK"
done
```

> **注意**：每个 worker 约占 150-200 MB 内存。64 个 worker 共需约 10-13 GB。
> 若内存有限可减少 worker 数量，同时相应缩短 `ports` 列表。

### 第三步：生成训练数据（只需运行一次）

数据准备脚本直接读取数据集文件，**无需 worker 运行**：

```bash
cd /path/to/verl

python examples/wb_agent/data_preprocess/prepare_wb_agent_data.py \
  --dataset-dir /path/to/InfiniteWeb-Dataset \
  --output-dir ~/data/wb_agent_verl \
  --train-ratio 0.9 \
  --seed 42
```

输出：
```
~/data/wb_agent_verl/
├── train.parquet   # ~3600 条
└── test.parquet    # ~400 条
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset-dir` | `~/workspace/InfiniteWeb-Dataset` | 数据集根目录 |
| `--output-dir` | `~/data/wb_agent_verl` | 输出目录 |
| `--train-ratio` | 0.9 | 训练集比例 |
| `--seed` | 42 | 随机种子 |
| `--max-websites` | -1 | 限制网站数量（-1 = 全部，调试时可设为 10） |

### 第四步：启动训练

```bash
cd /path/to/verl

MODEL_PATH=/path/to/Qwen3-VL-7B-Instruct \
  bash examples/wb_agent/run_wb_agent_grpo.sh
```

常用环境变量覆盖：

```bash
MODEL_PATH=/path/to/model \
TRAIN_BATCH_SIZE=64 \
N_GPUS=8 \
OFFLOAD=True \
  bash examples/wb_agent/run_wb_agent_grpo.sh
```

也可以直接用 Hydra 覆盖任意配置项：

```bash
bash examples/wb_agent/run_wb_agent_grpo.sh \
  actor_rollout_ref.rollout.n=4 \
  trainer.total_epochs=5
```

---

## 配置说明

### 修改 Worker 数量

编辑 `config/interaction_config/wb_agent_interaction_config.yaml`，调整 `ports` 列表：

```yaml
config:
  ports: [6001, 6002, ..., 6128]   # 128 个 worker
  max_steps: 25                    # 每个 episode 最大步数
  success_threshold: 1.0           # 提前终止的成功阈值
```

同时需要相应启动对应数量的 worker 进程。

### 修改 GRPO group size

编辑 `config/wb_agent_grpo.yaml`：

```yaml
actor_rollout_ref:
  rollout:
    n: 8   # 同一 task 采样 8 条轨迹，越大 advantage 估计越准但吞吐越低
```

### 调整浏览器视口

Worker 启动时指定（默认 1280×800）：

```bash
python env_server/web_env_server.py \
  --port 6001 \
  --dataset-dir /path/to/data \
  --viewport-width 1280 \
  --viewport-height 800
```

---

## 系统要求

| 组件 | 最低要求 | 推荐 |
|---|---|---|
| GPU | 8× A100 40GB | 8× A100 80GB |
| CPU | 32 核 | 64 核 |
| 内存 | 64 GB（Worker）+ 128 GB（训练） | 256 GB |
| 磁盘 | 50 GB（数据集）+ 200 GB NVMe（训练） | 500 GB NVMe |
| Worker 进程数 | 8 | 64-128 |

> 无需 KVM / Docker / 硬件虚拟化支持。Worker 进程为普通 Python 进程，可在任意 Linux/macOS 上运行。

---

## 训练流程

```
verl GRPO 训练循环
    ↓ 每个 task 采样 8 条轨迹（rollout.n=8）
ToolAgentLoop 多轮状态机
    ↓ 每轮调用
WebWorldInteraction.generate_response()
    ↓ asyncio.to_thread（同步 HTTP → 异步）
web_env_server（64 个 Playwright worker）
    ↓ 返回页面截图 + localStorage score
verl 计算 group advantage → 更新模型参数
```

每轮交互的消息格式：

- **User**（初始）：`Task: <goal>`
- **Assistant**：`<action>{"action_type": "click", "x": 640, "y": 400}</action>`
- **User**（环境反馈）：`![page](data:image/png;base64,...)\nStep 1/25. Score: 0.0\nWhat is your next action?`
- **Assistant**：下一步动作
- ...（最多 25 轮）

---

## 评分机制

奖励信号来自 InfiniteWeb 的 `rewritten_tasks.json` 中 `ground_truth.target_ids`：

1. 每步执行后，通过 Playwright `page.evaluate()` 读取当前 `localStorage` 全量数据
2. 检查每个 `target_id`（目标商品/房源/预订 ID 等）是否出现在 localStorage 中
3. 返回 `命中数 / 总目标数`（0.0–1.0）

**示例（Hawaii 度假租房任务）**：
```json
"target_ids": ["oahu_oceanfront_1br_budget"]
```
当 agent 成功将该房源加入购物车或开始预订时，此 ID 会被 `business_logic.js` 写入 localStorage，score 从 0.0 → 1.0。

---

## 与 AndroidWorld 对比

| 对比项 | AndroidWorld | **wb_agent** |
|---|---|---|
| 环境 | Android 模拟器（KVM） | Playwright Chromium |
| 并发环境内存 | ~2 GB / 实例 | ~150 MB / 实例 |
| 并发上限（256 GB） | ~32 | **~128** |
| 环境启动时间 | 30–60 秒 | 1–3 秒 |
| 训练任务数 | ~230 | **~4000+（约 17×）** |
| 数据准备 | 需要运行 Docker 容器 | **离线读文件** |
| 动作坐标系 | Android 截图坐标 | 浏览器截图坐标（1280×800） |
| 奖励密度 | Sparse（最终步） | Sparse（最终步，可扩展为 dense） |
