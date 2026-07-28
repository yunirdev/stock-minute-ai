# TradingAgents 本地运行配置

当前集成使用独立 Python 进程，避免 TradingAgents 的 LangChain/LangGraph
依赖进入主交易 Runtime 的虚拟环境。

## 已安装组件

- TradingAgents：官方 `v0.3.1`
- 源码：`D:\tradingagents-runtime\TradingAgents`
- Python：`D:\tradingagents-runtime\.venv\Scripts\python.exe`
- 缓存：`D:\tradingagents-runtime\cache`
- 结果：`D:\tradingagents-runtime\results`
- 记忆：`D:\tradingagents-runtime\memory\trading_memory.md`

主应用通过 `trader/tradingagents_worker.py` 启动专用 Python，并以带标记的
JSON 返回分析结果。第三方依赖崩溃或超时不会把 LangChain 导入主 Runtime。

## 本地模型

当前使用不复制权重的 TradingAgents 运行别名，避免原版模型按最大原生
上下文预分配后溢出到 CPU：

- `tradingagents-qwen3-14b:16k`：快速分析，16384 上下文。
- `tradingagents-qwen3.6:32k`：深度辩论和最终决策，32768 上下文。

对应 Modelfile 位于 `D:\tradingagents-runtime`; reusable repo templates: `docs/tradingagents/Modelfile.qwen2.5-32k` and `docs/tradingagents/Modelfile.qwen3.6-32k`。qwen3.6 质量更高，但在当前
机器上会同时使用 CPU/GPU；每日深度候选建议先设置为 2–3 只。

## 项目配置

`.env` 使用以下非密钥配置：

```dotenv
TRADINGAGENTS_PROJECT_DIR=D:\tradingagents-runtime\TradingAgents
TRADINGAGENTS_PYTHON=D:\tradingagents-runtime\.venv\Scripts\python.exe
TRADINGAGENTS_LLM_PROVIDER=ollama
TRADINGAGENTS_DEEP_MODEL=tradingagents-qwen3.6:32k
TRADINGAGENTS_QUICK_MODEL=tradingagents-qwen3-14b:16k
TRADINGAGENTS_BACKEND_URL=http://127.0.0.1:11434/v1
TRADINGAGENTS_TIMEOUT_SECONDS=7200
TRADINGAGENTS_CACHE_DIR=D:\stock-minute-ai\.tmp\tradingagents\cache
TRADINGAGENTS_RESULTS_DIR=D:\stock-minute-ai\.tmp\tradingagents\results
TRADINGAGENTS_MEMORY_LOG_PATH=D:\stock-minute-ai\.tmp\tradingagents\memory\trading_memory.md
TRADINGAGENTS_OUTPUT_LANGUAGE=Chinese
TRADINGAGENTS_CHECKPOINT_ENABLED=true
TRADINGAGENTS_TEMPERATURE=0.1
```

Ollama 的 OpenAI 兼容地址必须包含 `/v1`。

## 验证

```powershell
ollama ps
.venv\Scripts\python.exe -m pytest tests\test_tradingagents_adapter.py tests\test_tradingagents_worker.py -q
.venv\Scripts\python.exe -m trader.daily_research --symbols MSFT --deep-limit 1 --force
```

端到端验证已完成：MSFT 成功经过快速模型、深度模型、中文 JSON 传输和主应用
结果解析。TradingAgents 的五档评级在主应用中映射为：

- Buy / Overweight → BUY
- Hold → HOLD
- Underweight / Sell → SELL

## 升级

升级 TradingAgents 时，先在独立目录切换到明确的官方标签，再用 `uv pip
install --python D:\tradingagents-runtime\.venv\Scripts\python.exe
D:\tradingagents-runtime\TradingAgents` 重新安装。升级后先运行专项测试和一只
股票的端到端验证，不要直接把新依赖安装进主项目 `.venv`。
