# quant

日频量化研究工作台，由三个相互独立、可单独安装的子系统组成：

| 子包 | 职责 | 运行环境 |
| --- | --- | --- |
| `quant.factor_research` | 因子计算、因子表达式搜索、模型训练与回测报告 | 本地 Python 3.11 |
| `quant.market_data` | 基于 DuckDB 的本地行情库、查询客户端与网页服务 | 本地 Python 3.11 |
| `quant.qmt_downloader` | 通过大 QMT 内置 Python 落盘日线、财务与除权数据 | 大 QMT 内置 Python |

## 目录结构

```text
quant/
├─ src/quant/            # 全部库代码，src layout
│  ├─ cli/               # 命令行入口，只做参数解析与装配
│  ├─ config/            # 路径与环境变量的统一解析
│  ├─ factor_research/   # 因子研究
│  ├─ market_data/       # 行情库与服务
│  └─ qmt_downloader/    # 大 QMT 数据落盘
├─ scripts/              # 不经 pip 安装、需直接运行的入口（大 QMT 编辑器等）
├─ configs/              # 配置样例，按子系统分目录
├─ docs/                 # 详细文档
├─ tests/                # 测试，目录结构与 src/quant 对应
└─ pyproject.toml        # 打包、依赖与工具链配置
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[research,service,dev]"
```

创建虚拟环境后必须先激活，`python` 才会指向仓库内的 `.venv`；下文所有命令都假定
当前会话已激活该环境。

`quant.qmt_downloader` 只依赖 pandas，在大 QMT 内置 Python 中无需安装，
由 `scripts/qmt_run_downloader.py` 把 `src` 目录注入 `sys.path` 后直接导入。

## 常用命令

文档中的命令一律写成 `python -m <模块>`，在仓库根目录、已激活 `.venv` 的会话中
执行。表格第二列是 `pip install -e .` 之后可用的等价短命令别名。

| 命令 | 等价短命令 | 说明 |
| --- | --- | --- |
| `python -m quant.cli.factor_demo` | `quant-factor-demo` | 因子实验、回测与评估报告 |
| `python -m quant.cli.grid_search` | `quant-grid-search` | 因子表达式网格/遗传搜索 |
| `python -m quant.cli.build_daily_store` | `quant-build-daily-store` | 大 QMT 日线 CSV 增量入库 |
| `python -m quant.cli.market_check` | `quant-market-check` | 审计入库后的日线库 |
| `python -m quant.cli.market_server` | `quant-market-server` | 启动本地行情网页服务 |
| `python -m quant.cli.qmt_self_check` | `quant-qmt-self-check` | 全量审计 QMT 落盘数据 |
| `python -m quant.cli.single_factor_test` | `quant-single-factor-test` | 单因子快速试算 |
| `python -m quant.cli.ifind_download` | `quant-ifind-download` | 下载同花顺分钟行情 |
| `python -m quant.market_data.build_database` | `quant-build-market-db` | 从年度压缩包构建 5 分钟库 |

示例：

```powershell
python -m quant.cli.factor_demo --config configs\factor_research\example.yaml
```

行情库路径优先取环境变量 `MARKET_DB_PATH`，未设置时回落到 `D:\量化\market.duckdb`；
所有命令都支持用 `--database` 显式覆盖。

## 开发

```powershell
python -m pytest            # 全量测试
python -m ruff check .      # 静态检查
python -m mypy              # 类型检查
```

开发规范、因子与模型的实现约束、防未来数据泄漏要求见 [AGENTS.md](AGENTS.md)。

## 文档

- [docs/factor_research.md](docs/factor_research.md)：因子研究框架使用说明
- [docs/factor_grid_search.md](docs/factor_grid_search.md)：因子搜索设计与用法
- [docs/cli_arguments.md](docs/cli_arguments.md)：全部命令行参数速查
- [docs/market_data.md](docs/market_data.md)：行情库与服务接口
- [docs/qmt_downloader.md](docs/qmt_downloader.md)：大 QMT 数据落盘工具
- [docs/roadmap.md](docs/roadmap.md)：待办与计划
