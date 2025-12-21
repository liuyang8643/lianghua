# 1. 获取目标文件夹的绝对路径
$TargetName = ".cache"
$TargetFolder = Join-Path (Get-Location) $TargetName
$AbsolutePath = [System.IO.Path]::GetFullPath($TargetFolder)

# 检查目录是否存在
if (-not (Test-Path $AbsolutePath))
{
    Write-Host "❌ 错误: 未在当前目录下找到 $TargetName 文件夹。" -ForegroundColor Red
    Write-Host "搜索路径: $AbsolutePath" -ForegroundColor Gray
    exit
}

# 2. 安全确认阶段
Write-Host "------------------------------------------------"
Write-Host "您即将永久删除以下目录及其中的所有内容（不可恢复）：" -ForegroundColor Red
Write-Host "$AbsolutePath" -ForegroundColor Yellow
Write-Host "------------------------------------------------"
Write-Host "请按 [任意键] 确认并开始删除，或直接关闭窗口/按 Ctrl+C 取消：" -NoNewline

# 等待按键输入
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host "`n`n🚀 正在启动极限清理..." -ForegroundColor Cyan

# 3. 准备工作
$startTime = Get-Date
$EmptyDir = Join-Path $env:TEMP "empty_$( Get-Random )"
New-Item -Path $EmptyDir -ItemType Directory -Force | Out-Null

# 4. 执行 Robocopy 镜像删除任务
# 参数解释：/MIR 镜像 | /MT:128 满线程 | /NFL /NDL /NJH /NJS /NC /NS /NP 全静默
# 使用 Start-Process 异步启动以便监控心跳
$process = Start-Process robocopy -ArgumentList "`"$EmptyDir`" `"$AbsolutePath`" /MIR /MT:128 /R:0 /W:0 /NFL /NDL /NJH /NJS /NC /NS /NP" `
           -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\robodelete.log"

# 5. 心跳进度打印 (不影响性能)
Write-Host "🧹 正在销毁小文件序列... " -ForegroundColor Yellow
while (-not $process.HasExited)
{
    $elapsed = (Get-Date) - $startTime
    $timeStr = "{0:mm\:ss}" -f $elapsed
    Write-Host "`r[已运行: $timeStr] 系统后台正在全速处理中，请勿关闭... " -NoNewline -ForegroundColor Gray
    Start-Sleep -Seconds 1
}

# 6. 最后收尾 (删除空壳目录)
Remove-Item $AbsolutePath -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item $EmptyDir -Force -ErrorAction SilentlyContinue

$totalTime = "{0:mm\:ss}" -f ((Get-Date) - $startTime)
Write-Host "`n`n✅ 清理完成！" -ForegroundColor Green
Write-Host "总耗时: $totalTime"
