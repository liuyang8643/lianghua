# 用于实盘运行的 PowerShell 脚本

$currentDirectory = $PSScriptRoot
Set-Location -Path $currentDirectory

# 初始化代码
git fetch origin main
git reset --hard origin/main

# 设置 Python 环境
$env:PYTHONPATH = "$currentDirectory"
python --version

# 安装依赖
uv sync --locked

# 运行 Python 脚本
$workingDirectory = "$currentDirectory\trading"
Set-Location -Path $workingDirectory
& python "$workingDirectory\watchdog.py"
