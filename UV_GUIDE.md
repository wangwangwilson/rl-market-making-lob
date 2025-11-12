# UV 环境管理和部署指南

## 📖 目录

- [什么是 UV](#什么是-uv)
- [安装 UV](#安装-uv)
- [快速开始](#快速开始)
- [详细使用指南](#详细使用指南)
- [常见任务](#常见任务)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

---

## 什么是 UV

[uv](https://github.com/astral-sh/uv) 是一个**极快的Python包和项目管理器**，由Astral（Ruff的开发团队）开发。

### 为什么选择 UV？

| 特性 | UV | pip/venv | conda |
|------|-----|----------|-------|
| **速度** | ⚡ 10-100倍更快 | 慢 | 中等 |
| **磁盘空间** | 节省（全局缓存） | 浪费 | 浪费 |
| **依赖解析** | 快速准确 | 慢，有时不准确 | 慢 |
| **工具链管理** | ✅ 内置 | ❌ 需要pyenv | ✅ 内置 |
| **锁文件** | ✅ 自动生成 | ❌ 需要pip-tools | ❌ |
| **Rust实现** | ✅ | ❌ | ❌ |

### 核心优势

1. **极快的包安装** - 并行下载和安装
2. **全局缓存** - 包只下载一次，多项目共享
3. **精确的依赖解析** - 避免依赖冲突
4. **内置Python版本管理** - 无需pyenv或conda
5. **零配置** - 开箱即用

---

## 安装 UV

### 方法1: 官方安装脚本（推荐）

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip
pip install uv
```

### 方法2: 使用包管理器

```bash
# macOS (Homebrew)
brew install uv

# Linux (apt)
sudo apt install uv

# Windows (winget)
winget install astral-sh.uv
```

### 验证安装

```bash
uv --version
# 输出: uv 0.5.x (或更高版本)
```

---

## 快速开始

### 🚀 30秒快速启动

```bash
# 1. 进入项目目录
cd /workspace

# 2. 创建虚拟环境并安装所有依赖
uv venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 3. 同步依赖（根据pyproject.toml）
uv sync

# 4. 运行测试
uv run python test_crypto_data.py

# 5. 运行回测
uv run python run_crypto_backtest.py
```

### 🎯 一行命令运行（无需激活环境）

```bash
# 直接运行脚本
uv run test_crypto_data.py

# 直接运行回测
uv run run_crypto_backtest.py

# 下载数据
uv run example_tardis_download.py
```

---

## 详细使用指南

### 1. 创建和管理虚拟环境

#### 创建虚拟环境

```bash
# 使用默认Python版本（来自.python-version）
uv venv

# 指定Python版本
uv venv --python 3.12

# 指定环境名称
uv venv my_env
```

#### 激活虚拟环境

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# 鱼壳（Fish Shell）
source .venv/bin/activate.fish
```

#### 退出环境

```bash
deactivate
```

### 2. 安装依赖

#### 安装项目依赖（推荐）

```bash
# 安装pyproject.toml中定义的所有依赖
uv sync

# 只安装核心依赖（不含可选依赖）
uv sync --no-dev

# 包含开发依赖
uv sync --all-extras
```

#### 安装单个包

```bash
# 安装包并自动添加到pyproject.toml
uv add pandas

# 安装特定版本
uv add "pandas>=2.3.0"

# 安装为开发依赖
uv add --dev pytest

# 安装可选依赖组
uv add --optional notebook jupyter
```

#### 移除包

```bash
# 移除包
uv remove pandas

# 移除开发依赖
uv remove --dev pytest
```

### 3. 锁定依赖

```bash
# 生成uv.lock文件（精确记录所有依赖版本）
uv lock

# 更新锁文件（获取最新兼容版本）
uv lock --upgrade

# 更新特定包
uv lock --upgrade-package pandas
```

### 4. Python版本管理

```bash
# 列出可用的Python版本
uv python list

# 安装特定Python版本
uv python install 3.12

# 使用特定版本
uv python pin 3.12

# 查看当前Python版本
uv python show
```

### 5. 运行脚本和命令

```bash
# 在虚拟环境中运行Python脚本
uv run python script.py

# 运行命令
uv run pytest

# 使用缩写
uv run test_crypto_data.py

# 传递参数
uv run python run_crypto_backtest.py --episodes 50
```

### 6. 工具管理

```bash
# 安装全局工具（不污染项目环境）
uv tool install black
uv tool install ruff

# 运行工具
uv tool run black .
uv tool run ruff check .

# 列出已安装的工具
uv tool list

# 移除工具
uv tool uninstall black
```

---

## 常见任务

### 任务1: 初次设置项目

```bash
# 克隆或进入项目
cd /workspace

# 创建虚拟环境
uv venv

# 激活环境
source .venv/bin/activate

# 安装所有依赖
uv sync --all-extras

# 验证安装
uv run python -c "import torch; print(torch.__version__)"
```

### 任务2: 运行数据测试

```bash
# 方式1: 激活环境后运行
source .venv/bin/activate
python test_crypto_data.py

# 方式2: 直接使用uv run（推荐）
uv run python test_crypto_data.py

# 方式3: 使用脚本命令（如果配置了）
uv run crypto-test
```

### 任务3: 运行完整回测

```bash
# 确保有数据文件
ls crypto_data/scaled_data.csv

# 运行回测
uv run python run_crypto_backtest.py

# 或使用配置的命令
uv run crypto-backtest

# 查看结果
ls crypto_backtest_results/
```

### 任务4: 下载Tardis数据

```bash
# 交互式下载
uv run python example_tardis_download.py

# 或使用配置的命令
uv run crypto-download
```

### 任务5: 开发和调试

```bash
# 安装开发依赖
uv sync --all-extras

# 运行Python交互式环境
uv run ipython

# 或启动Jupyter
uv run jupyter notebook
```

### 任务6: 代码质量检查

```bash
# 使用uv tool安装代码检查工具（一次性）
uv tool install ruff
uv tool install black
uv tool install mypy

# 格式化代码
uv tool run black .

# 检查代码风格
uv tool run ruff check .

# 类型检查
uv tool run mypy *.py
```

### 任务7: 添加新依赖

```bash
# 添加运行时依赖
uv add requests

# 添加开发依赖
uv add --dev pytest-mock

# 添加可选依赖
uv add --optional tensorboard

# 查看已安装的包
uv pip list
```

### 任务8: 更新依赖

```bash
# 更新所有包到最新兼容版本
uv lock --upgrade

# 更新特定包
uv lock --upgrade-package torch

# 同步更新后的依赖
uv sync
```

### 任务9: 导出依赖（兼容性）

```bash
# 导出为requirements.txt
uv pip freeze > requirements.txt

# 只导出直接依赖
uv pip compile pyproject.toml -o requirements.txt
```

### 任务10: 清理环境

```bash
# 删除虚拟环境
rm -rf .venv

# 清理缓存
uv cache clean

# 重新创建环境
uv venv
uv sync
```

---

## 项目特定命令

### 数据处理流程

```bash
# 1. 测试数据处理（使用模拟数据）
uv run python test_crypto_data.py

# 2. 下载真实数据（可选）
uv run python example_tardis_download.py

# 3. 运行回测
uv run python run_crypto_backtest.py

# 4. 查看结果
cat crypto_backtest_results/backtest_stats.csv
```

### 完整工作流

```bash
#!/bin/bash
# 完整的实验工作流

# 设置
uv venv
source .venv/bin/activate
uv sync

# 测试
echo "运行数据测试..."
uv run python test_crypto_data.py

# 回测
echo "运行回测..."
uv run python run_crypto_backtest.py

# 结果
echo "回测完成，查看结果："
cat crypto_backtest_results/backtest_stats.csv
ls -lh crypto_backtest_results/*.png
```

---

## 故障排除

### 问题1: uv命令未找到

```bash
# 检查安装
which uv

# 重新安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 添加到PATH（如果需要）
export PATH="$HOME/.cargo/bin:$PATH"
```

### 问题2: Python版本不匹配

```bash
# 查看项目要求的版本
cat .python-version

# 安装所需版本
uv python install 3.12

# 固定版本
uv python pin 3.12
```

### 问题3: 依赖冲突

```bash
# 删除锁文件重新解析
rm uv.lock

# 重新锁定
uv lock

# 同步
uv sync
```

### 问题4: 包安装失败

```bash
# 清理缓存
uv cache clean

# 重新安装
uv sync --reinstall

# 查看详细错误
uv sync --verbose
```

### 问题5: 虚拟环境损坏

```bash
# 删除并重建
rm -rf .venv
uv venv
uv sync
```

### 问题6: CUDA/PyTorch问题

```bash
# 安装CPU版本PyTorch
uv add torch --index-url https://download.pytorch.org/whl/cpu

# 或安装CUDA版本
uv add torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 最佳实践

### 1. 使用锁文件

```bash
# 始终提交uv.lock到版本控制
git add uv.lock

# 在CI/CD中使用锁文件
uv sync --frozen  # 不更新锁文件
```

### 2. 分离开发和生产依赖

```python
# pyproject.toml
[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]

# 生产环境
uv sync

# 开发环境
uv sync --all-extras
```

### 3. 使用工具隔离

```bash
# 不要将开发工具添加到项目依赖
# 使用 uv tool 管理
uv tool install black
uv tool install ruff
```

### 4. 缓存优化

```bash
# 定期清理旧缓存
uv cache prune

# 查看缓存大小
du -sh ~/.cache/uv
```

### 5. 脚本命令

```python
# 在pyproject.toml中定义脚本
[project.scripts]
test = "pytest:main"
lint = "ruff:main"

# 使用
uv run test
uv run lint
```

### 6. 多项目管理

```bash
# 为每个项目使用独立的虚拟环境
cd project1 && uv venv
cd project2 && uv venv

# uv会自动使用全局缓存，节省空间
```

---

## 性能对比

### 包安装速度

| 任务 | pip | conda | uv |
|------|-----|-------|-----|
| 安装numpy | 3.2s | 8.5s | **0.3s** |
| 安装pandas | 5.1s | 12.3s | **0.5s** |
| 安装torch | 45s | 120s | **8s** |
| 安装项目所有依赖 | 120s | 300s | **15s** |

### 磁盘空间使用

```
pip/venv方式:
project1/.venv: 500MB
project2/.venv: 500MB
project3/.venv: 500MB
总计: 1.5GB

uv方式:
~/.cache/uv: 550MB (全局缓存)
project1/.venv: 50MB (链接)
project2/.venv: 50MB (链接)
project3/.venv: 50MB (链接)
总计: 700MB (节省53%)
```

---

## UV vs 其他工具

### UV vs pip

```bash
# pip方式
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 慢，无依赖解析

# uv方式
uv venv
uv sync  # 快，精确的依赖解析
```

### UV vs Poetry

```bash
# Poetry
poetry install  # 慢，但功能丰富

# uv
uv sync  # 极快，功能相当
```

### UV vs Conda

```bash
# Conda
conda create -n myenv python=3.12
conda activate myenv
conda install pandas numpy  # 慢

# uv
uv venv --python 3.12
uv add pandas numpy  # 快
```

---

## 高级用法

### 1. 工作区（Workspace）

```toml
# pyproject.toml
[tool.uv.workspace]
members = ["packages/*"]
```

### 2. 私有仓库

```bash
# 配置私有PyPI源
uv pip install --index-url https://pypi.company.com/simple package
```

### 3. 离线安装

```bash
# 下载所有包
uv pip download -r requirements.txt -d wheels/

# 离线安装
uv pip install --no-index --find-links wheels/ -r requirements.txt
```

### 4. 约束文件

```bash
# constraints.txt
numpy<2.0.0

# 使用约束
uv sync --constraint constraints.txt
```

---

## CI/CD 集成

### GitHub Actions

```yaml
name: Test

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Setup environment
        run: |
          uv venv
          uv sync
      
      - name: Run tests
        run: uv run pytest
```

### Docker

```dockerfile
FROM python:3.12-slim

# 安装uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY pyproject.toml .
COPY . .

# 安装依赖
RUN uv sync --frozen --no-dev

# 运行
CMD ["uv", "run", "python", "run_crypto_backtest.py"]
```

---

## 常见问题（FAQ）

### Q1: uv和pip可以一起使用吗？

**A:** 可以，但不推荐。uv管理的环境可以使用pip，但可能导致依赖不一致。

### Q2: uv.lock应该提交到版本控制吗？

**A:** 是的！锁文件确保团队成员使用相同的依赖版本。

### Q3: 如何迁移现有项目到uv？

```bash
# 1. 创建pyproject.toml（如果没有）
uv init

# 2. 从requirements.txt导入
uv add -r requirements.txt

# 3. 生成锁文件
uv lock

# 4. 同步环境
uv sync
```

### Q4: uv支持editable安装吗？

```bash
# 支持，使用-e标志
uv pip install -e .
```

### Q5: 如何使用特定的PyPI镜像？

```bash
# 使用环境变量
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 或在pyproject.toml中配置
[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
```

---

## 总结

### UV的核心优势

1. ⚡ **极快** - 比pip快10-100倍
2. 💾 **节省空间** - 全局缓存机制
3. 🔒 **精确依赖** - 自动生成锁文件
4. 🐍 **Python管理** - 内置版本管理
5. 🛠️ **现代化** - Rust实现，活跃维护

### 推荐工作流

```bash
# 一次性设置
uv venv && uv sync

# 日常开发
uv run python script.py

# 添加依赖
uv add package-name

# 更新依赖
uv lock --upgrade && uv sync

# 代码检查
uv tool run black .
```

### 进一步学习

- 📚 官方文档: https://docs.astral.sh/uv/
- 💬 GitHub: https://github.com/astral-sh/uv
- 🎥 视频教程: [YouTube搜索 "uv python"]

---

**享受极速的Python开发体验！** 🚀
