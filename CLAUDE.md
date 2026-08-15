# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本项目的开发规范、目录说明和验证要求统一维护在 `AGENTS.md`，请以该文件为唯一来源。
本文件只做导航，不复制 `AGENTS.md` 的内容，避免两份规范出现分歧。

下面单独一行的 `@AGENTS.md` 是 Claude Code 的文件导入语法，会话启动时会把该文件
全文注入上下文。Claude Code 只原生读取 `CLAUDE.md`，不会自动加载 `AGENTS.md`，
普通 Markdown 链接（`[AGENTS.md](AGENTS.md)`）也只是链接、不加载内容，因此这一行
不能删，否则下面所有"先读 AGENTS.md"的指引都会落空。

注意：`@` 路径写在句中同样会触发导入，导入解析只跳过反引号包裹的行内代码和代码块。
在正文里提到该路径时必须像本段这样用反引号包住，否则会重复注入整份文件。

@AGENTS.md

## 动手前必读

无论任务多小，先读 `AGENTS.md` 的这三节，它们描述的约束一旦违反会直接导致运行时
报错或数据泄漏，且不会被本地测试之外的手段发现：

- **分层约束**：`quant.qmt_downloader` 及其导入链必须兼容大 QMT 内置 Python
  （早于 3.7，禁止 `from __future__ import annotations`、禁止变量注解）；
  `scripts/qmt_*.py` 必须保持纯 ASCII。两条都有专门的测试守护。
- **防止未来数据泄漏**：任何触及因子、数据集、实验的改动都受这一节约束。
- **验证要求**：修改完成后必须自行创建 sub-agent 做独立 code review，
  review 通过前不得提交；交付说明需列出 review 结论、验证命令与 commit hash。

## 按任务定位规范

| 任务 | 先读 |
| --- | --- |
| 新增/修改因子 | `AGENTS.md` 因子开发规范 + [docs/factor_research.md](docs/factor_research.md) |
| 因子表达式搜索、DSL 算子 | `AGENTS.md` 因子搜索开发规范 + [docs/factor_grid_search.md](docs/factor_grid_search.md) |
| 新增模型 | `AGENTS.md` 模型开发规范（模型经工厂注册注入，不改实验主流程） |
| 命令行参数 | [docs/cli_arguments.md](docs/cli_arguments.md)；CLI 层只做装配，业务逻辑下沉库层 |
| 行情库 / 网页服务 | [docs/market_data.md](docs/market_data.md) |
| 大 QMT 落盘与自检 | [docs/qmt_downloader.md](docs/qmt_downloader.md) |
| 拆分超大模块 | `AGENTS.md` 模块规模与拆分约定（>700 行拆包，mixin 状态契约集中在 `base.py`） |

三个子系统的职责分工与安装方式见 [README.md](README.md)。

## 环境与命令补充

解释器：仓库内 `.venv`（Python 3.11.9）。受限终端偶尔报找不到 `python`，
这是沙箱权限问题，见 `AGENTS.md` 本地 Python 环境一节，不要建议重装。

`AGENTS.md` 验证要求之外常用的两条：

```powershell
# 单个测试 / 单个用例
.venv\Scripts\python.exe -m pytest tests\factor_research\test_metrics.py -q
.venv\Scripts\python.exe -m pytest "tests\qmt_downloader\test_downloader.py::DownloaderTests::test_config_rejects_reverse_date_range" -q

# 类型检查（CI 中不阻断，本地可选）
.venv\Scripts\python.exe -m mypy
```

CI（[.github/workflows/ci.yml](.github/workflows/ci.yml)，windows-latest）依次执行
`ruff check .` → `pytest -q` → `mypy`（`continue-on-error`）。本地至少跑通前两步。
