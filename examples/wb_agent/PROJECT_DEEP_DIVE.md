# InfiniteWeb GUI Agent GRPO 项目深度技术文档

> 写给下一个接手此项目的 AI。本文档覆盖所有模块的底层逻辑，读完即可完整理解整个系统。

---

## 一、项目全景

本项目将 **InfiniteWeb-Dataset**（626 个自包含静态网站 + ~4000 个带 ground truth 的任务）接入 **verl**（字节跳动生产级 RL 训练框架），对 **Qwen3-VL**（多模态视觉语言模型）做 **GRPO**（Group Relative Policy Optimization）强化学习训练。

**核心思路**：让模型在真实浏览器里执行 Web 任务（点击、输入、滚动、导航等），靠 localStorage 状态是否符合 ground truth 作为 reward 驱动学习，没有任何人工标注轨迹。

与 AndroidWorld 版本的最大结构差异：

| 维度 | AndroidWorld | **wb_agent** |
|------|-------------|-------------|
| 运行环境 | Docker + KVM Android 模拟器 | 普通 Python 进程 + Playwright Chromium |
| Worker 资源消耗 | ~2 GB RAM / 实例 | ~150 MB RAM / 实例 |
| 并发能力 | 受 KVM 限制，~32 | 受内存限制，~128 |
| 数据准备 | 需运行容器才能查询任务列表 | 离线读文件，无需 worker |
| reward 来源 | `is_successful(env)` 查 App 数据库 | `page.evaluate()` 查 localStorage |
| 截图格式 | numpy flat array → base64 转换 | Playwright 直接输出 base64 PNG |

### 文件结构总览

```
verl/
├── verl/interactions/
│   └── web_world_interaction.py        ← 核心：模型与环境的桥梁
└── examples/wb_agent/
    ├── PROJECT_DEEP_DIVE.md            ← 本文件
    ├── wb_agent_reward.py              ← reward 函数
    ├── run_wb_agent_grpo.sh            ← 一键启动脚本
    ├── config/
    │   ├── wb_agent_grpo.yaml          ← 训练超参配置
    │   └── interaction_config/
    │       └── wb_agent_interaction_config.yaml  ← Worker 端口池配置
    ├── data_preprocess/
    │   └── prepare_wb_agent_data.py    ← 离线生成 train/test.parquet
    └── env_server/
        └── web_env_server.py           ← Playwright + FastAPI 环境服务器

外部依赖：
InfiniteWeb-Dataset/
├── 1_hawaii_vacation_rent/
│   ├── index.html                      ← 数据已内嵌，无需 fetch 外部资源
│   ├── business_logic.js               ← 所有交互逻辑，操作 localStorage
│   ├── rewritten_tasks.json            ← 任务列表 + ground_truth
│   └── website_data.json               ← 原始数据（已注入 index.html）
├── ...（626 个网站目录）
```

---

## 二、物理架构：训练机 ↔ Worker 进程

```
训练机（GPU 服务器）
├── verl 训练进程（8× A100）
│   └── WebWorldInteraction（asyncio 协程池）
│       └── _WebEnvHttpClient × N（同步 HTTP，运行在线程池里）
│
└── Browser Worker 进程集群（同一台机器）
    ├── web_env_server.py --port 6001 → Playwright Chromium #1  → 当前任务网站
    ├── web_env_server.py --port 6002 → Playwright Chromium #2  → 当前任务网站
    ├── ...
    └── web_env_server.py --port 6064 → Playwright Chromium #64 → 当前任务网站
```

**每个 worker 进程是完全独立的 Playwright 浏览器实例**，各自管理独立的 `BrowserContext`，浏览器间不共享 localStorage 或 Cookie。

Worker 启动方式（无需 root / Docker / KVM）：

```bash
python examples/wb_agent/env_server/web_env_server.py \
  --port 6001 \
  --dataset-dir /path/to/InfiniteWeb-Dataset
```

---

## 三、Worker 内部：web_env_server.py

**文件**：`examples/wb_agent/env_server/web_env_server.py`

每个 worker 是一个 FastAPI + Playwright 的单进程服务，暴露与 AndroidWorld 对齐的 HTTP API，使得 interaction 层无需感知底层差异。

### 3.1 API 端点完整列表

| 端点 | 方法 | 功能 | 底层调用 |
|------|------|------|----------|
| `/health` | GET | 健康检查 | 检查 browser 是否初始化 |
| `/reset` | POST | 关闭当前 context，创建新 context | `browser.new_context()` |
| `/screenshot` | GET | 截图当前 viewport | `page.screenshot(type="png")` → base64 |
| `/execute_action` | POST | 执行浏览器动作 | Playwright mouse/keyboard API |
| `/task/initialize` | POST | 加载指定网站和任务 | `page.goto(file:// URL)` |
| `/task/tear_down` | POST | 清理当前任务 | `context.close()` |
| `/task/score` | GET | 查询任务得分 | `page.evaluate(localStorage)` → 比对 target_ids |
| `/task/goal` | GET | 获取任务描述文本 | 读 `rewritten_tasks.json` |

与 AndroidWorld server 的关键差异：
- `/screenshot` 直接返回 `{"screenshot_b64": "data:image/png;base64,…"}` 而非 numpy 数组，省去一次编码转换
- `/task/score` 是纯 Python/JS 计算，不需要查询 App 数据库
- `/task/initialize` 使用 `file://` 协议加载本地 HTML，无需起 HTTP 文件服务器

### 3.2 网站加载机制

```python
# 直接用 file:// 协议，Playwright 支持 localStorage 在 file:// 页面使用
index_url = website_dir.resolve().as_uri() + "/index.html"
await page.goto(index_url, wait_until="domcontentloaded")
```

每个 `index.html` 顶部内嵌了完整的数据初始化脚本：

```html
<script>
  if (!localStorage.getItem('dataInitialized')) {
      localStorage.setItem("destinations", "[{...}]");
      localStorage.setItem("rental_properties", "[{...}]");
      // ... 所有数据均内嵌，无需 fetch 外部资源
      localStorage.setItem('dataInitialized', 'true');
  }
</script>
```

页面加载即自动完成数据初始化，**不依赖任何网络请求**。

### 3.3 评分机制详解

InfiniteWeb 的业务逻辑（`business_logic.js`）会在用户完成关键操作时把目标对象的 ID 写入 localStorage（例如：选中房源 → `bookings` 数组追加 `{"property_id": "oahu_oceanfront_1br_budget", ...}`）。

评分函数利用这一特性：

```python
# 1. 读取当前所有 localStorage 内容
ls_raw = await page.evaluate(
    "() => JSON.stringify(Object.assign({}, localStorage))"
)
ls_text = ls_raw.lower()

# 2. 检查 ground_truth.target_ids 是否出现在 localStorage 中
target_ids = ["oahu_oceanfront_1br_budget"]  # 来自 rewritten_tasks.json
matches = sum(1 for tid in target_ids if tid.lower() in ls_text)

# 3. 返回命中比例
score = matches / len(target_ids)  # 0.0 ~ 1.0
```

**得分语义**：
- `0.0`：任务完全未完成，目标对象未被操作
- `0.5`：多目标任务（如同时添加 Maui 和 Oahu 两个房源），完成了一个
- `1.0`：全部 target_ids 均出现在 localStorage，任务完全完成

**局限性**：当前是字符串包含检查，对于 target_id 较短的情况（如 `id`、`home`）存在误判风险。实际数据集中的 ID 均为具体业务 slug（如 `kauai_entire_2br_hanalei_home`），长度足够，误判率极低。

### 3.4 动作执行细节

```python
# click：基于截图像素坐标，与 Android 完全对齐
await page.mouse.click(action.x, action.y)

# input_text：向当前焦点元素输入
await page.keyboard.type(text, delay=30)  # delay=30ms 模拟真实输入速度

# scroll：在当前鼠标位置滚动
dx, dy = {"up": (0,-400), "down": (0,400), "left": (-400,0), "right": (400,0)}
await page.mouse.wheel(dx, dy)

# navigate_back：浏览器返回
await page.go_back(wait_until="domcontentloaded")

# navigate_home：回到当前网站的 index.html
await page.goto(file_url, wait_until="domcontentloaded")
```

每次动作后，server 等待 `domcontentloaded`（超时 2s，失败不报错），给 JavaScript 时间处理事件。

---

## 四、核心模块：web_world_interaction.py

**文件**：`verl/interactions/web_world_interaction.py`

继承 verl 的 `BaseInteraction`，逻辑与 `android_world_interaction.py` 基本一致，差异点如下：

### 4.1 端口池机制（与 AndroidWorld 完全相同）

```python
# __init__：所有端口预填入 asyncio.Queue
self._port_queue = asyncio.Queue()
for port in [6001, 6002, ..., 6064]:
    self._port_queue.put_nowait(port)

# start_interaction：原子性地拿一个端口
port = await self._port_queue.get()   # 若无空闲则阻塞等待（背压）

# _cleanup_instance：还回端口（在 finally 块保证必执行）
self._port_queue.put_nowait(port)
```

### 4.2 实例状态字典

```python
self._instances[instance_id] = {
    "port": 6003,
    "client": _WebEnvHttpClient,      # 指向该 worker 的 HTTP 客户端
    "website_id": "1_hawaii_vacation_rent",
    "task_idx": 0,
    "step": 0,
    "initial_screenshot": "data:image/png;base64,…",  # 直接可用，无需转换
    "done": False,
    "final_score": None,
}
```

### 4.3 截图处理的简化

AndroidWorld 的截图路径：
```
Docker → {"pixels": flat_int_list} → np.array() → PIL.Image → PNG → base64 data URL
```

wb_agent 的截图路径（worker 已完成所有转换）：
```
Playwright → PNG bytes → base64 data URL → 直接使用
```

`_encode_screenshot()` 函数在 wb_agent 中不存在，`get_screenshot()` 直接返回 `str`（data URL）。

### 4.4 BaseInteraction 四个方法的完整逻辑

**`start_interaction(instance_id, website_id, task_idx)`**
1. `await queue.get()` 拿端口
2. `POST /reset` → `POST /task/initialize` → `GET /screenshot`
3. 把初始截图（已是 base64 data URL）存入实例字典
4. 失败时归还端口，抛异常

**`generate_response(instance_id, messages)`**

```
step == 0:
    → 返回初始截图 + "What is your next action?"
    → reward=0.0, done=False

assistant 最后一条 = status:success:
    → GET /task/score → 真实得分
    → done=True，触发 cleanup

其他（正常步骤）:
    → _parse_action() 解析 <action>
    → POST /execute_action
    → GET /screenshot（新状态，直接可用）
    → GET /task/score
    → 判断 done = score >= 1.0 or step >= max_steps
    → 若 done：pop instance → _cleanup_instance → 归还端口
    → 返回新截图 + 反馈文字
```

返回值：`(done: bool, obs_text: str, score: float, metadata: dict)`

**`calculate_score(instance_id)`**
若 done 返回缓存的 `final_score`，否则实时调用 `/task/score`。

**`finalize_interaction(instance_id)`**
`POST /task/tear_down` → `queue.put_nowait(port)`，`finally` 保证端口必然归还。

---

## 五、verl 框架内部：数据流动

与 AndroidWorld 版本完全相同，此处仅列出差异点，其他参见 AndroidWorld 的 PROJECT_DEEP_DIVE.md。

### 5.1 整体调用链

```
ray_trainer.py（训练主循环）
    ↓
AgentLoopManager → ToolAgentLoop（多轮状态机）
    ↓ 每轮 INTERACTING 状态
WebWorldInteraction.generate_response()
    ↓ asyncio.to_thread（同步 HTTP → 异步）
web_env_server（64 个 Playwright worker）
    ↓ 返回 base64 截图 + localStorage score
NaiveRewardManager → wb_agent_reward.compute_score()
    ↓ return float(turn_scores[-1])
GRPO advantage 计算 → actor 参数更新
```

### 5.2 Reward 流向

```
episode 结束后 turn_scores = [0.0, 0.0, 0.0, 1.0]
    ↓
wb_agent_reward.compute_score(
    extra_info={"turn_scores": [0.0, 0.0, 0.0, 1.0]}
)
    ↓ return 1.0
    ↓
rm_scores[last_token_pos] = 1.0  ← reward 只打在最后一个 token
    ↓
GRPO: advantage_i = reward_i - mean(group_rewards)
```

**中间步骤的 score**（如 `[0.0, 0.0, 0.0]`）不参与 loss 计算，只记录在日志中。如需 dense reward，修改 `wb_agent_reward.py` 改用加权求和（见第十节）。

### 5.3 GRPO advantage 计算

同一 task（`website_id + task_idx`）采样 8 条轨迹：

```
轨迹0: reward=1.0 → advantage = 1.0 - 0.375 = +0.625
轨迹1: reward=0.0 → advantage = 0.0 - 0.375 = -0.375
轨迹2: reward=1.0 → advantage = 1.0 - 0.375 = +0.625
轨迹3: reward=0.0 → advantage = 0.0 - 0.375 = -0.375
...（共 8 条）
group_mean = 0.375
```

---

## 六、训练数据：parquet 文件结构

### 6.1 数据生成流程（离线，无需 worker）

```
prepare_wb_agent_data.py
    ↓ 遍历 dataset_dir 下所有子目录
    ↓ 读取每个目录的 rewritten_tasks.json
    ↓ 提取 tasks[].instruction 作为 goal
    ↓ numpy rng 打乱 + 9:1 train/test split
    ↓ 保存 train.parquet + test.parquet
```

无需任何运行中的 server，速度极快（626 个网站 < 5 秒完成）。

### 6.2 每行数据的字段

```python
{
    "data_source": "wb_agent",          # reward 函数用这个识别
    "prompt": [                         # Python list（非 JSON 字符串！）
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Task: For a 4-night stay on Oahu, "
                                    "find an oceanfront 1BR condo under $250..."}
    ],
    "ability": "web_agent",
    "reward_model": {"style": "rule", "ground_truth": "1_hawaii_vacation_rent"},
    "extra_info": {                     # Python dict（非 JSON 字符串！）
        "split": "train",
        "index": 0,
        "website_id": "1_hawaii_vacation_rent",
        "task_idx": 0,
        "goal": "For a 4-night stay on Oahu...",
        "interaction_kwargs": {         # ← ToolAgentLoop 用这个初始化 interaction
            "name": "wb_agent",         # ← 匹配 interaction_config 里的 name
            "website_id": "1_hawaii_vacation_rent",
            "task_idx": 0
        }
    }
}
```

> **重要**：`prompt`、`extra_info`、`reward_model` 必须存为原生 Python list/dict，不能 `json.dumps()` 成字符串。verl 内部在多处直接以 dict/list 方式访问这些字段，字符串会导致 `AttributeError`。

### 6.3 interaction_kwargs 的路由机制

1. `extra_info["interaction_kwargs"]["name"]` = `"wb_agent"`
2. ToolAgentLoop 在 `interaction_config_path` 的 YAML 里找 `name: "wb_agent"` 的条目
3. 实例化 `WebWorldInteraction`，调用 `start_interaction(**interaction_kwargs)`
4. 即 `start_interaction(name="wb_agent", website_id="...", task_idx=0)`（`name` 被 `**kwargs` 吸收，不报错）

### 6.4 任务数量与多样性

| 统计维度 | 数值 |
|---------|------|
| 网站总数 | 626 |
| 平均每网站任务数 | ~6-7 |
| 总任务实例数 | ~4000+ |
| 行业覆盖 | 电商、医疗、教育、政务、房产、SaaS 等 |
| 任务类型 | 筛选商品、预订、创建 wishlist、发送消息、注册账号等 |
| 每网站平均页面数 | ~16 个 HTML 页面 |

626 个网站已覆盖足够多样的 UI 模式，无需像 AndroidWorld 一样用 `n_task_combinations` 来扩展数量。

---

## 七、System Prompt 与动作格式

模型被要求以固定格式输出动作：

```
<action>{"action_type": "click", "x": 640, "y": 400}</action>
```

动作坐标系：**浏览器 viewport 像素坐标**，原点在左上角，x 向右，y 向下。viewport 默认 1280×800。

支持的动作类型：

| 动作 | 参数 | 对应 Playwright 调用 |
|------|------|---------------------|
| `click` | x, y | `page.mouse.click(x, y)` |
| `input_text` | text | `page.keyboard.type(text)` |
| `scroll` | direction: up/down/left/right | `page.mouse.wheel(dx, dy)` |
| `navigate_back` | 无 | `page.go_back()` |
| `navigate_home` | 无 | `page.goto(index.html)` |
| `keyboard_enter` | 无 | `page.keyboard.press("Enter")` |
| `long_press` | x, y | `mouse.down()` + sleep(0.8s) + `mouse.up()` |
| `wait` | 无 | `asyncio.sleep(1.0)` |
| `status` | goal_status: "success" | interaction 层处理，不发给 server |

`status:success` 是特殊动作：ToolAgentLoop 识别后立即调用 `/task/score` 并结束 episode，不会调用 `/execute_action`。

**解析容错**：

```python
def _parse_action(text):
    # 优先：<action>{...}</action>
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    # fallback：裸 JSON 对象
    if not match:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
    # 最终兜底：wait（避免 episode 崩溃）
    return {"action_type": "wait"}
```

---

## 八、配置文件详解

### 8.1 wb_agent_grpo.yaml 关键参数

```yaml
actor_rollout_ref:
  rollout:
    n: 8                          # GRPO group size：同一 task 采 8 条轨迹
    temperature: 1.0              # 必须 > 0 保证 8 条轨迹有足够多样性
    agent:
      default_agent_loop: tool_agent  # 必须有！否则 interaction 不会被调用
    multi_turn:
      enable: True
      max_assistant_turns: 25    # 必须 >= interaction_config 里的 max_steps
                                 # 若小于 max_steps，ToolAgentLoop 会先截断
reward:
  custom_reward_function:
    path: examples/wb_agent/wb_agent_reward.py
    name: compute_score          # 必须配置！否则对 data_source="wb_agent"
                                 # 抛 NotImplementedError
```

### 8.2 wb_agent_interaction_config.yaml 关键参数

```yaml
interaction:
  - name: "wb_agent"             # 与 extra_info.interaction_kwargs.name 严格匹配
    class_name: "verl.interactions.web_world_interaction.WebWorldInteraction"
    config:
      ports: [6001, ..., 6064]   # 必须与实际启动的 worker 进程数量匹配
      max_steps: 25              # 每个 episode 最大步数
      host: "localhost"
      success_threshold: 1.0
```

### 8.3 Worker 端口数量与 GRPO 并发的关系

```
所需最小 worker 数 = train_batch_size × rollout.n / 平均 episode 步数

例：batch_size=128, n=8, 平均步数=12
  → 同时需要 128×8 = 1024 个并发 episode slot
  → 但每个 episode 平均 12 步，每步耗时约 1-2s
  → 实际并发需求 ≈ 1024 / 12 ≈ 85 个 worker
  → 配置 96 个 worker 有 10% 余量
```

---

## 九、常见问题与排查

### Q1：训练报 `KeyError: 'rm_scores'`

**原因**：`reward.custom_reward_function.path` 未配置，`default_compute_score` 对 `"wb_agent"` 抛 `NotImplementedError`，导致 `reward_score=None`，`rm_scores` 未写入 batch。

**修复**：确认 `wb_agent_grpo.yaml` 有 `reward.custom_reward_function` 配置块。

---

### Q2：报错 "Interaction 'wb_agent' not found"

**原因**：`extra_info.interaction_kwargs.name` 与 `interaction_config.yaml` 里的 `name` 不匹配，大小写或拼写差异即报错。

**修复**：两处都必须为 `"wb_agent"`，包括 `prepare_wb_agent_data.py` 中 `interaction_kwargs["name"]` 的赋值。

---

### Q3：所有 reward 都是 0

**排查顺序**：

```bash
# 1. Worker 是否正常运行
curl http://localhost:6001/health

# 2. 初始化任务是否成功
curl -X POST "http://localhost:6001/task/initialize?website_id=1_hawaii_vacation_rent&task_idx=0"

# 3. 初始 score（应为 0）
curl "http://localhost:6001/task/score"

# 4. 看 wandb 日志里 turn_scores 是否全是 [0.0, 0.0, ...]
#    若是，说明 agent 执行了动作但 localStorage 没变 → 检查网站加载是否正常
```

常见原因：
- `dataset_dir` 路径错误，`index.html` 未加载
- `target_ids` 的值在 localStorage 里是嵌套 JSON 对象内部，字符串搜索未命中 → 检查 `business_logic.js` 写入 localStorage 的具体 key 和格式

---

### Q4：训练跑一段时间后全部卡住（端口泄漏）

**原因**：`generate_response` 在 `done=True` 时未调用 `_cleanup_instance`，端口永不归还，所有协程永久阻塞在 `await queue.get()`。

**检查**：确认 `generate_response` 中所有 `done=True` 的分支都有：
```python
self._instances.pop(instance_id, None)
await self._cleanup_instance(inst)
```

`_cleanup_instance` 内部的 `finally` 块保证即使 `/task/tear_down` 失败，端口也必然归还：
```python
finally:
    self._port_queue.put_nowait(port)
```

---

### Q5：`file://` 协议下 localStorage 跨页面不共享

**原因**：Chromium 把不同 `file://` 路径视为不同 origin，导致 `index.html` 初始化的 localStorage 在 `search_results.html` 中读不到。

**现象**：点击搜索后跳转到 `search_results.html`，页面显示空白或报错。

**修复**：Playwright 默认使用的是不强制隔离 file:// origin 的模式，实测同目录下的多个 HTML 页面共享 localStorage。若仍有问题，可切换为 HTTP 服务方案：

```python
# 在 web_env_server.py 中改用 aiohttp 静态服务器
from aiohttp import web
runner = web.AppRunner(web.Application())
runner.app.router.add_static("/", website_dir)
await runner.setup()
site = web.TCPSite(runner, "localhost", file_server_port)
await site.start()
await page.goto(f"http://localhost:{file_server_port}/index.html")
```

---

### Q6：`AttributeError: 'str' object has no attribute 'get'`（数据加载阶段崩溃）

**原因**：`prepare_wb_agent_data.py` 把 `prompt`、`extra_info`、`reward_model` 用 `json.dumps()` 存成了 JSON 字符串，但 verl 内部以 dict/list 方式直接访问。

**修复**：确保这三个字段存的是 Python 原生 list/dict，不做 `json.dumps()`。

---

### Q7：截图 base64 太长导致 prompt 超过 `max_prompt_length`

1280×800 的 PNG base64 约 500KB-1MB（取决于页面复杂度）。默认 `max_prompt_length=2048` 按 token 算，但图片 token 数由模型的视觉 tokenizer 决定，Qwen3-VL 对于 1280×800 图片约消耗 1280 个 vision token。

**修复选项**：
- 降低 viewport 分辨率（推荐 1024×768 或更低）
- 增大 `max_prompt_length`（如 4096）
- 使用 JPEG 格式减小图片体积（在 `web_env_server.py` 中改 `type="jpeg", quality=80`）

---

### Q8：Worker 进程崩溃后 port 泄漏（port 已归还但 worker 已死）

**现象**：port 返回队列，`_WebEnvHttpClient.health()` 返回 False，之后所有分配到该 port 的 episode 都报 `ConnectionRefusedError`。

**处理**：在 `start_interaction` 中加健康检查：
```python
if not await asyncio.to_thread(client.health):
    self._port_queue.put_nowait(port)
    raise RuntimeError(f"Worker on port {port} is not healthy")
```

配合外部进程守护（如 `supervisord` 或 systemd）自动重启崩溃的 worker。

---

## 十、扩展方向

### Dense Reward

当前是 sparse reward（只用 `turn_scores[-1]`）。中间步骤的 score 已记录在 `turn_scores` 里，可以直接改 `wb_agent_reward.py` 使用：

```python
# 方案1：每步有进展就给奖励（鼓励探索）
def compute_score(...):
    if not turn_scores:
        return 0.0
    # 如果中间有过 score > 0 的步骤，给一个折扣奖励
    max_mid = max(turn_scores[:-1]) if len(turn_scores) > 1 else 0.0
    final = turn_scores[-1]
    return final + 0.1 * max_mid  # final score 为主，中间进展加分

# 方案2：步骤惩罚（鼓励快速完成）
step_penalty = 0.01 * len(turn_scores)
return max(0.0, turn_scores[-1] - step_penalty)

# 方案3：差分奖励（鼓励每步都有进展）
diff_rewards = [turn_scores[i] - turn_scores[i-1]
                for i in range(1, len(turn_scores))]
return sum(diff_rewards) + turn_scores[-1]
```

### 扩充数据集：跨数据集混合训练

InfiniteWeb + AndroidWorld MiniWoB 混合训练，利用两种数据的互补性（web 表单任务 vs 移动端操作任务）：

```bash
# 生成 wb_agent 数据
python examples/wb_agent/data_preprocess/prepare_wb_agent_data.py \
  --output-dir ~/data/wb_agent_part

# 生成 MiniWoB 数据（需要 AndroidWorld 容器运行）
python examples/android_world/data_preprocess/prepare_android_world_data.py \
  --task_family miniwob --n_task_combinations 20 \
  --output_dir ~/data/miniwob_part

# 合并
python -c "
import pandas as pd
train = pd.concat([
    pd.read_parquet('~/data/wb_agent_part/train.parquet'),
    pd.read_parquet('~/data/miniwob_part/train.parquet'),
])
train.to_parquet('~/data/combined/train.parquet', index=False)
"
```

注意：混合训练时 `interaction_config.yaml` 需要同时配置两个 interaction：
```yaml
interaction:
  - name: "wb_agent"
    class_name: "verl.interactions.web_world_interaction.WebWorldInteraction"
    ...
  - name: "android_world"
    class_name: "verl.interactions.android_world_interaction.AndroidWorldInteraction"
    ...
```

### 多机训练

```bash
# 机器 A（GPU 训练机 + worker）
for i in $(seq 1 64); do
  python web_env_server.py --port $((6000+i)) --host 0.0.0.0 ...
done

# 机器 B（额外 worker 机器）
for i in $(seq 65 128); do
  python web_env_server.py --port $((6000+i)) --host 0.0.0.0 ...
done
```

`interaction_config.yaml` 目前所有 port 共用同一个 `host`，若需要多机需改造 `WebWorldInteraction.__init__` 支持 `{host, port}` 元组列表。

### 更精确的评分器

当前 target_id 字符串匹配存在边界情况。更健壮的方案：

```python
# 解析 localStorage 的已知 key 结构，精确查找
ls_dict = json.loads(ls_raw)

# 示例：检查 bookings 数组里是否有 target property_id
bookings = json.loads(ls_dict.get("bookings", "[]"))
if any(b.get("property_id") == target_id for b in bookings):
    return 1.0
```

这需要对每类网站的 localStorage schema 建立解析规则，工作量较大，适合在 score = 0 的任务上针对性优化。
