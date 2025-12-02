# WBR 量化交易系统

基于 QMT (国金量化交易平台) 的 Python 量化交易系统

## 环境要求

- Python 3.12
- Windows 操作系统 (QMT 平台依赖) + [QMT 客户端](https://download.gjzq.com.cn/gjty/organ/gjzqqmt.rar)已安装
- uv 包管理器

### 安装 uv

方式一：使用官方安装脚本（推荐）

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

方式二：使用 pip 安装（作为 Python 包）
如果你更习惯使用 pip，也可以这样安装它（但安装后，uv 内部也会使用这个已存在的 Python 环境）：

```powershell
pip install uv
```

### 初始化并安装依赖

使用 uv 自动创建虚拟环境并安装所有依赖：

```powershell
# uv 会自动读取 pyproject.toml 并安装依赖
uv sync
# 或者手动激活虚拟环境
.\.venv\Scripts\Activate.ps1
```

#### TA-Lib 安装说明

Windows 下安装 TA-Lib 可能需要额外步骤：

```powershell
# 方法1：使用 uv 直接安装（推荐）
uv pip install TA-Lib

# 方法2：如果方法1失败，先下载预编译文件
# 从 https://github.com/cgohlke/talib-build/releases 下载对应版本
# 然后安装：
uv pip install TA_Lib-0.6.8-cp312-cp312-win_amd64.whl
```

## 实盘运行

> 非首次运行可直接跳到第 5 步

1. 复制 [env.template.py](configs/env.template.py) 为 `env.py`，并修改为实际的配置项
2. 复制 [_copy.ps1](_copy.ps1) 到 gjqmt\\bin.x64 目录下。
3. 执行 _copy.ps1 脚本以准备截取登录信息
4. 登录QMT，勾选极简模式
5. 检查 _linkMini 是否成功生成
6. 运行 `.\run.ps1` 启动实盘交易

## 开发指引

### 性能分析

使用 line_profiler 进行性能分析：

```python
from line_profiler import profile

@profile
def your_function():
  # 你的代码
  pass
```

## 项目架构

详见：https://ai.feishu.cn/wiki/NaSewvuQ8iRpQfkpzWMckZRSnmh

## 许可证

本项目采用私有许可，仅供内部使用。
