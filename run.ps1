param(
    [Parameter(Mandatory = $true)]
    [string]$Bundle,

    [Parameter(Mandatory = $true)]
    [string]$Runtime,

    [string]$DecisionDate = "",
    [string]$Account = "",
    [string]$Journal = "",
    [switch]$InitializeNewChain
)

# 用于实盘运行的 PowerShell 脚本。代码更新和版本切换必须在启动前独立完成。

$currentDirectory = $PSScriptRoot
Set-Location -Path $currentDirectory

$env:PYTHONPATH = "$currentDirectory"
python --version

$watchdogArgs = @(
    "-m", "trade.watchdog",
    "--bundle", $Bundle,
    "--runtime", $Runtime,
    "--prepare-snapshot",
    "--execute"
)
if ($DecisionDate) {
    $watchdogArgs += @("--date", $DecisionDate)
}
if ($Account) {
    $watchdogArgs += @("--account", $Account)
}
if ($Journal) {
    $watchdogArgs += @("--journal", $Journal)
}
if ($InitializeNewChain) {
    $watchdogArgs += "--initialize-new-chain"
}

# `uv run --locked` 只按已封存锁文件启动，不修改代码或依赖解析。
& uv run --locked python @watchdogArgs
