# AndroidWorld GRPO 项目深度技术文档

> 写给下一个接手此项目的 AI。本文档覆盖所有模块的底层逻辑，读完即可完整理解整个系统。

---

## 一、项目全景

本项目将 **AndroidWorld**（真实 Android 模拟器任务基准）接入 **verl**（字节跳动的生产级 RL 训练框架），对 **Qwen3-VL**（多模态视觉语言模型）做 **GRPO**（Group Relative Policy Optimization）强化学习训练。

**核心思路**：让模型在真实 Android 模拟器里执行任务（点击、输入、滚动等），靠任务完成率作为 reward 驱动学习，没有任何人工标注轨迹。

### 文件结构总览

```
verl/
├── verl/interactions/
│   └── android_world_interaction.py     ← 核心：模型与环境的桥梁
└── examples/android_world/
    ├── PROJECT_DEEP_DIVE.md             ← 本文件
    ├── android_world_reward.py          ← reward 函数
    ├── run_android_world_grpo.sh        ← 一键启动脚本
    ├── config/
    │   ├── android_world_grpo.yaml      ← 训练超参配置
    │   └── interaction_config/
    │       └── android_world_interaction_config.yaml  ← Docker 端口池配置
    └── data_preprocess/
        └── prepare_android_world_data.py  ← 生成 train/test.parquet

外部依赖（独立仓库）：
android_world/
└── android_world/
    ├── server/android_server.py         ← Docker 容器内的 FastAPI 服务
    ├── task_evals/                      ← 116个Android任务 + 90个MiniWoB任务
    └── registry.py                      ← 任务注册表
```

---

## 二、物理架构：训练机 ↔ Docker 容器

```
训练机（GPU 服务器）
├── verl 训练进程（8× A100）
│   └── AndroidWorldInteraction（asyncio 协程池）
│       └── _AndroidEnvHttpClient × N（同步 HTTP，运行在线程池里）
│
└── Docker 容器集群（同一台机器或局域网）
    ├── aw_worker_1  → host:5001 → 容器:5000 → FastAPI → Android 模拟器 #1
    ├── aw_worker_2  → host:5002 → 容器:5000 → FastAPI → Android 模拟器 #2
    ├── ...
    └── aw_worker_32 → host:5032 → 容器:5000 → FastAPI → Android 模拟器 #32
```

**每个 Docker 容器是完全独立的 Android 模拟器实例**，容器间不共享任何状态。容器启动命令：

```bash
docker run -d --privileged --name aw_worker_$i \
  -p $((5000+i)):5000 android_world:latest
```

`--privileged` 是必须的，因为 Android 模拟器需要 KVM 硬件虚拟化支持。

---

## 三、Docker 容器内部：android_server.py

**文件**：`android_world/server/android_server.py`

容器内运行一个 FastAPI 服务（uvicorn 监听 :5000），封装了 AndroidWorld 环境的所有操作。

### API 端点完整列表

| 端点 | 方法 | 功能 | 底层调用 |
|------|------|------|----------|
| `/health` | GET | 健康检查 | 检查 env 是否初始化 |
| `/reset` | POST | 重置模拟器 | `env.reset(go_home=True)` |
| `/screenshot` | GET | 获取截图 | `env.get_state(wait_to_stabilize)` → numpy array |
| `/execute_action` | POST | 执行动作 | `JSONAction(**dict)` → `env.execute_action()` → ADB |
| `/suite/reinitialize` | GET | 重新初始化任务套件 | `suite_utils.create_suite(n_task_combinations, seed)` |
| `/suite/task_list` | GET | 获取所有任务名 | `list(suite.keys())` |
| `/suite/task_length` | GET | 某任务类的实例数 | `len(suite[task_type])` |
| `/task/initialize` | POST | 初始化特定任务 | `suite[task_type][task_idx].initialize_task(env)` |
| `/task/tear_down` | POST | 清理任务状态 | `suite[task_type][task_idx].tear_down(env)` |
| `/task/score` | GET | 查询任务得分 | `suite[task_type][task_idx].is_successful(env)` |
| `/task/goal` | GET | 获取任务描述文本 | `suite[task_type][task_idx].goal` |

**关键细节**：
- `/screenshot` 返回的是 `{"pixels": [flat int list]}`，是 numpy RGB array 序列化后的结果
- `/execute_action` 接收 JSON dict，服务端转成 `JSONAction` 对象再通过 ADB 执行
- `/task/score` 调用任务自带的 `is_successful(env)` 方法，这是 Python 代码检查 App 数据库/文件系统状态，不是字符串匹配

---

## 四、核心模块：android_world_interaction.py

**文件**：`verl/interactions/android_world_interaction.py`

这是整个项目最核心的文件，继承 verl 的 `BaseInteraction`，实现模型与环境的所有交互逻辑。

### 4.1 端口池机制（agent-environment 一对一绑定）

```python
# __init__：所有端口预填入 asyncio.Queue
self._port_queue = asyncio.Queue()
for port in [5001, 5002, ..., 5032]:
    self._port_queue.put_nowait(port)

# start_interaction：原子性地拿一个端口
port = await self._port_queue.get()   # 若无空闲则阻塞等待（背压）

# finalize_interaction：还回端口
self._port_queue.put_nowait(port)
```

`asyncio.Queue.get()` 是协程安全的原子操作。多个协程并发时，每个协程拿到不同端口，保证一个容器同时只服务一个 agent。

若并发 rollout 数（如 128 条轨迹）超过容器数（32），多余的协程在 `get()` 处自动排队，不会崩溃，只是等待。

### 4.2 实例状态字典

每个 agent 通过 `instance_id`（UUID）与其专属状态绑定：

```python
self._instances[instance_id] = {
    "port": 5003,                    # 专属容器端口
    "client": _AndroidEnvHttpClient, # 指向该容器的 HTTP 客户端
    "task_type": "SimpleSmsSend",
    "task_idx": 2,
    "step": 0,                       # 当前步数
    "initial_screenshot": "data:image/png;base64,...",  # 仅 step=0 时存在
    "done": False,
    "final_score": None,
}
```

### 4.3 同步 HTTP → 异步的处理方式

HTTP 调用是同步阻塞的（requests 库），但 verl 的 agent loop 是 asyncio 异步的。
解决方案：`asyncio.to_thread()` 把同步调用放到线程池执行，不阻塞事件循环：

```python
await asyncio.to_thread(client.reset, True)
await asyncio.to_thread(client.execute_action, action_dict)
pixels = await asyncio.to_thread(client.get_screenshot, True)
```

### 4.4 截图编码

`/screenshot` 返回的 numpy 像素数组需要编码成模型可以理解的格式：

```python
def _encode_screenshot(pixels: np.ndarray) -> str:
    img = Image.fromarray(pixels.astype(np.uint8))  # numpy → PIL
    buf = io.BytesIO()
    img.save(buf, format="PNG")                      # PIL → PNG bytes
    b64 = base64.b64encode(buf.getvalue()).decode()  # bytes → base64
    return f"data:image/png;base64,{b64}"            # → data URL
```

最终嵌入 user 消息的 markdown 图片格式：`![screen](data:image/png;base64,...)`

Qwen3-VL 能解析这种 base64 data URL 格式的图片。

### 4.5 动作解析

模型输出格式：`<action>{"action_type": "click", "x": 500, "y": 800}</action>`

```python
def _parse_action(text: str) -> dict:
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    # 失败时 fallback：尝试找裸 JSON，再失败返回 wait 动作
    return {"action_type": "wait"}  # 最终兜底
```

### 4.6 BaseInteraction 四个方法的完整逻辑

**`start_interaction(instance_id, task_type, task_idx)`**
1. `await queue.get()` 拿端口（可能阻塞）
2. `POST /reset` → `POST /task/initialize` → `GET /screenshot`
3. 把初始截图存入实例字典，等第一次 `generate_response` 时注入
4. 失败时归还端口，抛异常

**`generate_response(instance_id, messages)`**

分三个分支：

```
step == 0:
    → 返回初始截图 + "What is your next action?"
    → reward=0.0, done=False（不执行任何动作）

assistant 最后一条 = status:success:
    → 直接查 /task/score 得到真实得分
    → done=True，触发 cleanup

其他（正常步骤）:
    → 解析 <action> → POST /execute_action
    → GET /screenshot（新状态）
    → GET /task/score（当前得分）
    → 判断 done = score >= 1.0 or step >= max_steps
    → 若 done：pop instance，调用 _cleanup_instance 归还端口
    → 返回新截图 + 反馈文字
```

返回值：`(done: bool, obs_text: str, score: float, metadata: dict)`

**`calculate_score(instance_id)`**
- 若 done：返回 `final_score`（已缓存）
- 否则：实时调用 `/task/score`

**`finalize_interaction(instance_id)`**
- `POST /task/tear_down` → `POST /reset` → `queue.put_nowait(port)`
- 内部调用 `_cleanup_instance`，异常时也确保端口归还（在 `finally` 块里）

---

## 五、verl 框架内部：数据如何流动

### 5.1 整体调用链

```
ray_trainer.py（训练主循环）
    ↓ 每个 training step
AgentLoopManager.generate_sequences(batch)
    ↓ 并发处理每个 sample
AgentLoopWorker._run_agent_loop()
    ↓ 根据 config.agent.default_agent_loop = "tool_agent"
ToolAgentLoop.run_agent_loop()   ← 多轮状态机
    ↓ 每轮 INTERACTING 状态
AndroidWorldInteraction.generate_response()
    ↓ HTTP
Docker 容器
    ↓ episode 结束
RewardLoopManager → NaiveRewardManager → android_world_reward.compute_score()
    ↓ rm_scores 写入 batch
GRPO advantage 计算 → actor 参数更新
```

### 5.2 ToolAgentLoop 状态机

```
PENDING
  ↓ apply_chat_template → 准备 prompt_ids
GENERATING
  ↓ sGLang 生成一条 assistant 回复
  ↓ 如果有 tool_calls → PROCESSING_TOOLS
  ↓ 如果有 interaction → INTERACTING
INTERACTING
  ↓ interaction.generate_response()
  ↓ 把返回的 obs_text 作为 user 消息追加
  ↓ 把 score 追加到 turn_scores
  ↓ done=True → TERMINATED
  ↓ done=False → 回到 GENERATING
TERMINATED
  ↓ 收集 output（prompt_ids, response_ids, response_mask）
  ↓ extra_fields = {"turn_scores": [...], "tool_rewards": [...]}
```

**重要**：ToolAgentLoop **不设置** `AgentLoopOutput.reward_score`（保持 None）。
reward 通过 RewardLoopManager 异步计算（见下节）。

### 5.3 Reward 流向（这是最容易弄错的部分）

```
episode 结束后：
AgentLoopWorker._compute_score(output)
    ↓ enable_async_reward = (reward_loop_worker_handles is not None)
    ↓ 因为 use_rm=False，所以 enable_agent_reward_loop=True（ray_trainer.py:829）
    ↓ reward_loop_worker_handles = reward_loop_manager.reward_loop_workers ← 始终存在！
    ↓
NaiveRewardManager.run_single(data)
    ↓ tool_extra_fields = data.non_tensor_batch["tool_extra_fields"]
    ↓ extra_info.update(tool_extra_fields)  ← turn_scores 在这里传入
    ↓
android_world_reward.compute_score(
    data_source="android_world",
    solution_str=<decoded response>,
    ground_truth=<task_type>,
    extra_info={"turn_scores": [0.0, 0.0, 0.5, 1.0], ...}
)
    ↓ return float(turn_scores[-1])  ← 只用最后一步的环境得分
    ↓
output.reward_score = 1.0
    ↓
_postprocess():
rm_scores = zeros_like(response_mask)
rm_scores[sample_i, last_token_pos] = reward_score  ← reward 只打在最后一个 token
batch["rm_scores"] = rm_scores
    ↓
GRPO: advantage_i = reward_i - mean(group_rewards)
```

**关键理解**：
- `turn_scores` 里的中间步骤 score（如 [0.0, 0.0, 0.5]）**对训练无贡献**，只是日志
- 真正训练信号只有 `turn_scores[-1]`（episode 最终得分）
- reward 只赋给 response 序列的**最后一个 token**，前面所有 token 的 `rm_scores = 0`
- 这是标准的 sparse reward 设定

### 5.4 GRPO advantage 计算

同一个 task（task_type + task_idx）采样 8 条轨迹（`rollout.n=8`）：

```
轨迹0: reward=1.0 → advantage = 1.0 - 0.25 = +0.75
轨迹1: reward=0.0 → advantage = 0.0 - 0.25 = -0.25
轨迹2: reward=0.0 → advantage = 0.0 - 0.25 = -0.25
轨迹3: reward=0.5 → advantage = 0.5 - 0.25 = +0.25
...（共8条）
group_mean = 0.25
```

advantage 有正有负，负 advantage 轨迹的 loss 会降低这些动作序列的概率。

---

## 六、训练数据：parquet 文件结构

### 6.1 数据生成流程

```
prepare_android_world_data.py
    ↓ 连接一个运行中的 Docker 容器（--port 5001）
    ↓ GET /suite/reinitialize(n_task_combinations=N, seed=42)
    ↓ GET /suite/task_list → 获取所有任务类名
    ↓ 对每个 (task_type, task_idx)：GET /task/goal → 自然语言描述
    ↓ 随机 8:2 split（用 numpy rng，可复现）
    ↓ 保存 train.parquet + test.parquet
```

### 6.2 每行数据的字段

```python
{
    "data_source": "android_world",          # reward 函数用这个识别
    "prompt": '[                             # JSON 字符串，存 chat messages
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Task: Send SMS to Alice saying Hello"}
    ]',
    "ability": "android_agent",              # 仅标注，不影响训练
    "reward_model": '{"style": "rule", "ground_truth": "SimpleSmsSend"}',
    "extra_info": '{                         # 关键：包含 interaction_kwargs
        "task_type": "SimpleSmsSend",
        "task_idx": 2,
        "goal": "Send SMS to Alice saying Hello",
        "interaction_kwargs": {              # ← ToolAgentLoop 用这个初始化 interaction
            "name": "android_world",         # ← 匹配 interaction_config 里的 name
            "task_type": "SimpleSmsSend",
            "task_idx": 2
        }
    }'
}
```

### 6.3 interaction_kwargs 的路由机制

1. `extra_info["interaction_kwargs"]["name"]` = `"android_world"`
2. ToolAgentLoop 在 `interaction_config_path` 指向的 YAML 里找 `name: "android_world"` 的条目
3. 实例化 `AndroidWorldInteraction`，调用 `start_interaction(**interaction_kwargs)`

### 6.4 任务数量与多样性

- **AndroidWorld 原生任务**：116 个任务类，参数随机生成
  - 消息文本瓶颈：只有 40 条固定句子（`RANDOM_SENTENCES`）
  - 日历日期固定在 October 2023
  - 推荐 `n_task_combinations=5~10`，超过 20 收益递减
- **MiniWoB 任务**：90 个任务类，参数由 JavaScript 每次动态生成
  - 无重复问题，`n_task_combinations=20` 每条都不同
  - 任务类型：Web 表单、按钮点击、邮件、订票、社交媒体等
  - 用法：`--task_family miniwob`（当前配置默认用 `android_world`）
- **两者合并**可达 206 种任务类型，显著提升多样性

---

## 七、System Prompt 与动作格式

模型被要求以固定格式输出动作：

```
<action>{"action_type": "click", "x": 500, "y": 800}</action>
```

支持的动作类型（完整列表）：

| 动作 | 参数 | 说明 |
|------|------|------|
| `click` | x, y | 点击坐标 |
| `input_text` | text | 键盘输入 |
| `scroll` | direction: up/down/left/right | 滚动 |
| `long_press` | x, y | 长按 |
| `navigate_home` | 无 | Home 键 |
| `navigate_back` | 无 | Back 键 |
| `keyboard_enter` | 无 | 回车键 |
| `open_app` | app_name | 打开应用 |
| `wait` | 无 | 等待（也是解析失败的兜底） |
| `status` | goal_status: "success" | 宣告任务完成 |

`status` 动作是特殊的：触发 ToolAgentLoop 立即查询 `/task/score` 并结束 episode，不执行实际操作。

---

## 八、配置文件详解

### 8.1 android_world_grpo.yaml 关键参数

```yaml
actor_rollout_ref:
  rollout:
    n: 8                          # GRPO group size：同一 task 采 8 条轨迹
    temperature: 1.0              # 生成时的温度，需要 > 0 保证多样性
    agent:
      default_agent_loop: tool_agent  # 必须有！否则 interaction 不会被调用
    multi_turn:
      enable: True
      max_assistant_turns: 20    # 必须与 interaction_config 里的 max_steps 匹配

reward:
  custom_reward_function:
    path: examples/android_world/android_world_reward.py
    name: compute_score          # 必须配置！否则 default_compute_score 对
                                 # data_source="android_world" 抛 NotImplementedError
```

### 8.2 android_world_interaction_config.yaml 关键参数

```yaml
interaction:
  - name: "android_world"        # 与 extra_info.interaction_kwargs.name 匹配
    class_name: "verl.interactions.android_world_interaction.AndroidWorldInteraction"
    config:
      ports: [5001, ..., 5032]   # 必须与实际启动的容器数量匹配
      max_steps: 20              # 必须与 max_assistant_turns 匹配
      host: "localhost"
      success_threshold: 1.0     # 得分 >= 此值视为成功
```

---

## 九、常见问题与排查

### Q1: 训练报 KeyError: 'rm_scores'
**原因**：`reward.custom_reward_function.path` 未配置，`default_compute_score` 对 `"android_world"` 抛 `NotImplementedError`，导致 `reward_score=None`，`rm_scores` 没写入 batch。
**修复**：确认 `android_world_grpo.yaml` 里有 `reward.custom_reward_function` 配置块。

### Q2: 报错 "Interaction 'android_world' not found"
**原因**：`data.extra_info.interaction_kwargs.name` 与 `interaction_config.yaml` 里的 `name` 字段不匹配。
**修复**：两处都应为 `"android_world"`。

### Q3: 所有 reward 都是 0
**排查**：
1. 容器是否正常运行？`curl http://localhost:5001/health`
2. `turn_scores` 是否有值？看 wandb 的 `extra_fields/turn_scores` 日志
3. `max_steps` 是否太少？任务还没完成就结束了

### Q4: 训练跑一段时间后全部卡住（端口泄漏）
**原因**：`generate_response` 在 episode 通过 `score >= threshold` 或 `step >= max_steps` 结束时，原先没有调用 `_cleanup_instance`，端口永不归还。`ToolAgentLoop` 也从不调用 `finalize_interaction`。32 个端口跑完后，所有 rollout 协程永久阻塞在 `await queue.get()`。
**已修复**：`generate_response` 中 `done=True` 分支现在会先 `self._instances.pop(instance_id)` 再 `await self._cleanup_instance(inst)`，端口可正常归还。

### Q5: 端口池暂时耗尽（all ports busy，无卡死）
**原因**：并发 rollout 数 > 容器数，`queue.get()` 阻塞等待。
**处理**：这是正常的背压机制，不是 bug。要提高并发度需增加 Docker 容器数并更新 `ports` 列表。

### Q6: 训练启动即崩溃（AttributeError / TypeError 在数据加载阶段）
**原因**：`prepare_android_world_data.py` 曾用 `json.dumps()` 把 `prompt`、`extra_info`、`reward_model` 存为 JSON 字符串，但 verl 多处代码（`rl_dataset.py`、`tool_agent_loop.py`、`naive.py`）直接以 Python list/dict 方式访问这些字段：
- `row_dict["extra_info"].get("interaction_kwargs")` → 字符串无 `.get()` → `AttributeError`
- `data_item.non_tensor_batch["reward_model"]["ground_truth"]` → 字符串下标 → `TypeError`

**已修复**：三处 `json.dumps()` 已移除，字段直接存原生 Python list/dict。

### Q7: 截图图片 base64 太长导致 prompt 超过 max_prompt_length
**修复**：调整 `data.max_prompt_length`（当前 2048），或降低截图分辨率（需修改 server 端）。

---

## 十、扩展方向

### 加入 MiniWoB 任务

修改数据生成脚本同时生成两种任务，合并 parquet：

```bash
# 生成 AndroidWorld 数据
python prepare_android_world_data.py --task_family android_world \
  --n_task_combinations 5 --output_dir ~/data/aw_part

# 生成 MiniWoB 数据
python prepare_android_world_data.py --task_family miniwob \
  --n_task_combinations 20 --output_dir ~/data/miniwob_part

# 合并（Python）
import pandas as pd
df = pd.concat([pd.read_parquet("aw_part/train.parquet"),
                pd.read_parquet("miniwob_part/train.parquet")])
df.to_parquet("combined/train.parquet", index=False)
```

### Dense Reward

当前是 sparse reward（只用最后一步）。改成 dense reward 需修改 `generate_response`：
- 每步加步骤惩罚（鼓励快速完成）
- 格式奖励（动作 JSON 格式合法 +0.05）
- 中间状态奖励（某些任务支持，如删除了部分条目）

`turn_scores` 里的每步得分已经记录，只需在 `android_world_reward.py` 里改用 `sum(turn_scores)` 或加权求和而非只取最后一个值，同时在 `generate_response` 里针对每步构造有意义的中间分数。

### 多机训练

增加 `trainer.nnodes > 1`，Docker 容器可以分布在多台机器上，修改 `host` 参数或用 `host:port` 直接指定完整地址（需修改 interaction 代码支持每个端口独立的 host）。
