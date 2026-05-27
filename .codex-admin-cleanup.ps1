$ErrorActionPreference = 'Continue'

function Get-TreeBytes {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum
    if ($null -eq $sum) { return 0 }
    return [int64]$sum
}

function Clear-Contents {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $before = Get-TreeBytes $Path
    Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    $after = Get-TreeBytes $Path
    [pscustomobject]@{
        Path = $Path
        FreedGB = [math]::Round((($before - $after) / 1GB), 2)
        RemainingGB = [math]::Round(($after / 1GB), 2)
    }
}

$beforeDrive = Get-PSDrive -Name C
Write-Output "START_FREE_GB=$([math]::Round($beforeDrive.Free / 1GB, 2))"

$services = 'bits', 'wuauserv', 'dosvc'
foreach ($svc in $services) {
    try { Stop-Service -Name $svc -Force -ErrorAction SilentlyContinue } catch {}
}

$results = @()
$paths = @(
    'C:\Windows\SoftwareDistribution\Download',
    'C:\Windows\Temp',
    'C:\ProgramData\Microsoft\Windows\WER\ReportArchive',
    'C:\ProgramData\Microsoft\Windows\WER\ReportQueue',
    'C:\ProgramData\Microsoft\Windows\WER\Temp',
    'C:\ProgramData\Microsoft\Windows\DeliveryOptimization\Cache'
)
foreach ($path in $paths) {
    $result = Clear-Contents $path
    if ($result) { $results += $result }
}

try {
    Delete-DeliveryOptimizationCache -Force -ErrorAction SilentlyContinue
} catch {}

Write-Output 'DISM_START'
& dism.exe /Online /Cleanup-Image /StartComponentCleanup /NoRestart
Write-Output "DISM_EXIT=$LASTEXITCODE"

foreach ($svc in $services) {
    try { Start-Service -Name $svc -ErrorAction SilentlyContinue } catch {}
}

$results | Where-Object { $_.FreedGB -gt 0 } | Sort-Object FreedGB -Descending |
    Format-Table -AutoSize | Out-String | Write-Output

$afterDrive = Get-PSDrive -Name C
Write-Output "END_FREE_GB=$([math]::Round($afterDrive.Free / 1GB, 2))"
