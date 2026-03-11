# AndroidWorld GRPO Training with verl

基于 [AndroidWorld](https://github.com/google-research/android_world) 环境，使用 verl 框架对 Qwen3-VL 做 GRPO 强化学习训练。

## 文件结构

```
examples/android_world/
├── config/
│   ├── android_world_grpo.yaml                          # GRPO 训练配置
│   └── interaction_config/
│       └── android_world_interaction_config.yaml        # Docker 端口池配置
├── data_preprocess/
│   └── prepare_android_world_data.py                    # 生成 train/test.parquet
├── run_android_world_grpo.sh                            # 一键启动训练
└── README.md

verl/interactions/
└── android_world_interaction.py                         # 核心 Interaction 类
```

---

## 快速开始

### 第一步：构建 AndroidWorld Docker 镜像

```bash
git clone https://github.com/google-research/android_world.git
cd android_world
docker build -t android_world:latest .
```

### 第二步：启动 Docker 容器

启动 32 个并行容器，每个容器对应一个 Android 模拟器实例：

```bash
for i in $(seq 1 32); do
  docker run -d --privileged --name aw_worker_$i \
    -p $((5000+i)):5000 \
    android_world:latest
done
```

等待所有容器启动完成（模拟器冷启动需 2-5 分钟）：

```bash
# 检查容器是否健康
for i in $(seq 1 32); do
  curl -s http://localhost:$((5000+i))/health && echo " port $((5000+i)) OK"
done
```

### 第三步：生成训练数据（只需运行一次）

```bash
cd /path/to/verl

python examples/android_world/data_preprocess/prepare_android_world_data.py \
  --port 5001 \
  --n_task_combinations 3 \
  --seed 42 \
  --output_dir ~/data/android_world_verl
```

输出：
```
~/data/android_world_verl/
├── train.parquet   # ~180 条（77 task types × 3 instances × 0.8）
└── test.parquet    # ~45 条
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | 5001 | 用于查询任务列表的容器端口 |
| `--n_task_combinations` | 3 | 每个任务类型的随机参数实例数 |
| `--seed` | 42 | 随机种子 |
| `--train_ratio` | 0.8 | 训练集比例 |
| `--output_dir` | `~/data/android_world_verl` | 输出目录 |

### 第四步：启动训练

```bash
cd /path/to/verl

MODEL_PATH=/path/to/Qwen3-VL-7B-Instruct \
  bash examples/android_world/run_android_world_grpo.sh
```

常用环境变量覆盖：

```bash
MODEL_PATH=/path/to/model \
TRAIN_BATCH_SIZE=64 \
N_GPUS=8 \
OFFLOAD=True \          # 显存不足时开启 CPU offload
  bash examples/android_world/run_android_world_grpo.sh
```

也可以直接用 Hydra 覆盖任意配置项：

```bash
bash examples/android_world/run_android_world_grpo.sh \
  actor_rollout_ref.rollout.n=4 \
  trainer.total_epochs=5
```

---

## 配置说明

### 修改容器数量

编辑 `config/interaction_config/android_world_interaction_config.yaml`，调整 `ports` 列表：

```yaml
config:
  ports: [5001, 5002, ..., 5064]   # 64 个容器
  max_steps: 20                    # 每个 episode 最大步数
```

### 修改 GRPO group size

编辑 `config/android_world_grpo.yaml`：

```yaml
actor_rollout_ref:
  rollout:
    n: 8   # 同一 task 采样 8 条轨迹，越大 advantage 估计越准但吞吐越低
```

---

## 系统要求

| 组件 | 最低要求 | 推荐 |
|---|---|---|
| GPU | 8× A100 40GB | 8× A100 80GB |
| CPU | 64 核 | 96 核 |
| 内存 | 128 GB | 256 GB |
| 磁盘 | 500 GB NVMe SSD | 1 TB NVMe SSD |
| Docker 容器数 | 8 | 32-64 |

> **注意**：每个 Android 模拟器容器需要约 2 GB 内存 + 8 GB 磁盘，且必须开启 KVM 硬件虚拟化（`/dev/kvm` 存在）。

---

## 训练流程

```
verl GRPO 训练循环
    ↓ 每个 task 采样 8 条轨迹（rollout.n=8）
ToolAgentLoop 多轮状态机
    ↓ 每轮调用
AndroidWorldInteraction.generate_response()
    ↓ asyncio.to_thread（同步 HTTP → 异步）
AndroidWorld Docker 容器（32 个）
    ↓ 返回截图 + score
verl 计算 group advantage → 更新模型参数
```

每轮交互的消息格式：
- **User**（初始）：`Task: <goal>`
- **Assistant**：`<action>{"action_type": "click", "x": 500, "y": 1000}</action>`
- **User**（环境反馈）：`![screen](data:image/png;base64,...)\nStep 1/20. Score: 0.0\nWhat is your next action?`
- **Assistant**：下一步动作
- ...
