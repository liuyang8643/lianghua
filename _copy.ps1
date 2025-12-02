$lastOutputTime = Get-Date

while ($true) {
    if (Test-Path linkmini) {
        Copy-Item linkMini _linkMini
        Write-Host "finish"
        break
    }

    $currentTime = Get-Date
    $timeDifference = ($currentTime - $lastOutputTime).TotalSeconds

    if ($timeDifference -ge 1) {
        Write-Host "continue"
        $lastOutputTime = $currentTime
    }

    Start-Sleep -Milliseconds 50 # 延迟 50 毫秒
}