# 各模块模型安排推荐

本项目通过 **两种模型档位** 统一调度：`chat_high_quality`（大脑型）与 `chat_low_cost`（劳工型）。  
以下为各模块的调用归属及推荐模型配置。

---

## 一、当前代码中的调用归属

| 模块 | 子环节 | 使用接口 | 说明 |
|------|--------|----------|------|
| **Analyzer** | 智能抽样 (sampler) | high | 关键章节筛选，需强推理 |
| | 元协议生成 (protocol_generator) | high | 逻辑红线与术语，需高质量 |
| | 逻辑回溯 (backtrack) | high | 因果一致性判断 |
| | 冲突整合 (refiner) | high | 冲突消解与合并 |
| | 双重校对 (double_check) | high | 逻辑/风格审核 |
| | 滑动窗口提取 (extractor) | **low** | 批量 JSON 抽取，偏劳工 |
| **Writer** | 战略层 / 节拍表 (strategy_layer) | high | 大纲编排 |
| | 逻辑层 (logic_layer) | high | 因果守门员 |
| | 3 走向 + Consultant + Critique (branch_simulation, graph_workflow) | high | 分支与评分、批判 |
| | 草稿层 (draft_layer) | **low** | 长文生成，语感与性价比 |
| **User Center** | 心理侧写 / 偏好补丁 (psychology_expert) | high | 用户画像与偏好推理 |
| | 意图解析 (intent_analyzer) | high | 续写/改写/同类意图解析 |
| | 类似体验 (recommendation) | high | 换皮构思 |
| **Evolution** | Simulator 评分 (simulator) | high | 文风/逻辑/剧情三维打分 |
| | 工程诊断 (engineering) | high | Action Plan 生成 |
| | 计划转编辑 (orchestrator plan_to_edits) | high | 根据诊断改码建议生成补丁 |
| **仿写/逆向重构** | 逻辑适配与章节设计 (logic_master) | **WRITING_REASONING_MODEL** | 默认 deepseek-reasoner，与 HIGH 一致 |
| | 正文生成 (writing_service) | **WRITING_DRAFT_MODEL** | 默认 qwen-max，需 DASHSCOPE_API_KEY |
| **Analyzer（可选）** | 逐章提取加速 | **ANALYZER_FAST_MODEL** | 若配置则优先用于 JSON 抽取，否则用 LOW |

---

## 二、环境变量与推荐模型

分析/写作/用户心理/演化 仍以 **ANALYZER_HIGH_MODEL** 与 **ANALYZER_LOW_MODEL** 为主；仿写流水线额外支持**写作双模型**，便于逻辑与文笔分离。

| 变量 | 控制范围 | 推荐模型（国产） | 推荐模型（OpenAI） |
|------|----------|------------------|--------------------|
| **ANALYZER_HIGH_MODEL** | 所有 `chat_high_quality` 调用（分析/战略/逻辑/批判/演化） | 见下表「高质量」 | gpt-4o |
| **ANALYZER_LOW_MODEL** | 所有 `chat_low_cost` 调用（批量提取、草稿） | 见下表「低成本」 | gpt-4o-mini |
| **WRITING_REASONING_MODEL** | 仿写逻辑适配（LogicMaster） | deepseek-reasoner | 未配置则用 ANALYZER_HIGH_MODEL |
| **WRITING_DRAFT_MODEL** | 仿写正文生成（WritingAgent） | qwen-max | 未配置则需 DASHSCOPE_API_KEY |
| **ANALYZER_FAST_MODEL** | 可选；逐章提取时若配置则优先用于加速 | glm-4-flash / deepseek-chat | 未配置则用 ANALYZER_LOW_MODEL |

---

## 三、按场景的推荐组合

### 1. 性价比优先（推荐日常使用）

- **高质量（ANALYZER_HIGH_MODEL）**：`deepseek-reasoner`（DeepSeek-R1 推理）  
  - 协议、抽样、回溯、战略层、逻辑层、Critique、用户画像、Simulator、工程诊断均用此模型。
- **低成本（ANALYZER_LOW_MODEL）**：`deepseek-chat`（DeepSeek-V3）  
  - 批量提取、草稿层生成。

特点：成本可控，逻辑与评分质量好。

---

### 2. 长上下文优先（前 50 章宏观扫描、长摘要）

- **高质量**：`moonshot-v1-128k`（Kimi 128k）  
  - 适合 analyzer 中「整书/长段」一次喂入的环节（如抽样、协议生成）。
- **低成本**：`deepseek-chat` 或 `glm-4-flash`。

特点：可少做截断，长文理解更好。

---

### 3. 成本极简（能跑即可）

- **高质量**：`glm-4` 或 `deepseek-chat`（用 chat 顶替推理）  
  - 逻辑/推理略弱于 R1，但可完成全流程。
- **低成本**：`glm-4-flash`（智谱免费额度友好）。

---

### 4. 质量优先（不差钱 / 关键上线）

- **高质量**：`gpt-4o`（需 OPENAI_API_KEY）  
  - 将 `ANALYZER_HIGH_MODEL=gpt-4o` 即可，其余不变。
- **低成本**：`gpt-4o-mini` 或继续用 `deepseek-chat`。

---

## 四、.env 配置示例

```bash
# 国产性价比组合（推荐）
ANALYZER_HIGH_MODEL=deepseek-reasoner
ANALYZER_LOW_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-...

# 仿写整书（逻辑 R1 + 文笔 Qwen，可选）
# WRITING_REASONING_MODEL=deepseek-reasoner
# WRITING_DRAFT_MODEL=qwen-max
# DASHSCOPE_API_KEY=sk-...

# 长上下文组合（Kimi）
# ANALYZER_HIGH_MODEL=moonshot-v1-128k
# MOONSHOT_API_KEY=...

# 极简成本（智谱）
# ANALYZER_HIGH_MODEL=glm-4
# ANALYZER_LOW_MODEL=glm-4-flash
# ZHIPU_API_KEY=...

# OpenAI 质量优先
# ANALYZER_HIGH_MODEL=gpt-4o
# ANALYZER_LOW_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...
```

---

## 五、小结

- **大脑型（high）**：分析协议/抽样/回溯/整合、写作战略与逻辑与批判、用户心理与意图、演化评分与诊断与改码建议 → 统一用 **ANALYZER_HIGH_MODEL**，推荐 **DeepSeek-R1** 或 **Kimi-128k**。
- **劳工型（low）**：批量抽取、草稿生成 → 统一用 **ANALYZER_LOW_MODEL**，推荐 **DeepSeek-V3** 或 **GLM-4-Flash**。
- **仿写流水线**：逻辑适配建议 **WRITING_REASONING_MODEL=deepseek-reasoner**，正文生成建议 **WRITING_DRAFT_MODEL=qwen-max**（需 DASHSCOPE_API_KEY）；未配置时推理沿用 ANALYZER_HIGH_MODEL。
- **可选加速**：**ANALYZER_FAST_MODEL** 用于逐章提取时可进一步降低成本或延迟；未配置则使用 ANALYZER_LOW_MODEL。
