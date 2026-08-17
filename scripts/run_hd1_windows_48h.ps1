param(
    [ValidateSet('preflight','smoke','canonical','all','status')]
    [string]$Phase = 'status',
    [string]$Python = '.\.venv-pinn\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot
if (![System.IO.Path]::IsPathRooted($Python)) { $Python = Join-Path $RepoRoot $Python }
$ResultDir = Join-Path $RepoRoot 'results\hd1_windows_high_dimensional_pinn_48h'
$StatusDir = Join-Path $ResultDir 'status'
$LogDir = Join-Path $ResultDir 'logs'
$Experiment = Join-Path $RepoRoot 'experiments\hd1_windows_high_dimensional_pinn_48h.py'
$Config = Join-Path $RepoRoot 'configs\hd1_windows_high_dimensional_pinn_48h.json'
$FocusedTest = Join-Path $RepoRoot 'tests\test_hd1_high_dimensional_pinn.py'
$CanonicalOrder = @('d1_SUP','d1_HVI','d2_SUP','d2_HVI','d5_SUP','d5_HVI','d10_SUP','d10_HVI')
$PidValue = $PID
$InvocationText = $MyInvocation.Line
$Commit = (& git -C $RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'git rev-parse failed' }

function Ensure-RuntimeDirs {
    New-Item -ItemType Directory -Force -Path $ResultDir,$StatusDir,$LogDir | Out-Null
    $probe = Join-Path $ResultDir '.write_probe'
    'ok' | Set-Content -Encoding ascii $probe
    Remove-Item $probe -Force
}

function Invoke-Logged {
    param([string]$Name, [string[]]$Arguments)
    Ensure-RuntimeDirs
    $log = Join-Path $LogDir ("{0}_{1}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'),$Name)
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Name failed with exit code $code; see $log" }
}

function Write-RunnerStatus {
    param([string]$Name,[string]$State,[datetime]$Started,[string]$Message='')
    Ensure-RuntimeDirs
    $envInfo = @{
        python = $Python
        powershell = $PSVersionTable.PSVersion.ToString()
        os = [System.Environment]::OSVersion.VersionString
        cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
    }
    $payload = [ordered]@{
        phase=$Name; status=$State; message=$Message; pid=$PidValue
        start_time=$Started.ToString('o'); end_time=(Get-Date).ToString('o')
        command=$InvocationText; git_commit=$Commit; environment=$envInfo
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $StatusDir "runner_$Name.json")
}

function Invoke-PhaseWithStatus {
    param([string]$Name,[scriptblock]$Body)
    $started = Get-Date
    try {
        & $Body
        Write-RunnerStatus $Name 'passed' $started
    } catch {
        Write-RunnerStatus $Name 'failed' $started $_.Exception.Message
        throw
    }
}

function Assert-PreflightPassed {
    $p = Join-Path $StatusDir 'preflight.json'
    if (!(Test-Path $p)) { throw 'preflight_not_passed: preflight.json missing' }
    $s = Get-Content -Raw $p | ConvertFrom-Json
    if ($s.status -ne 'passed' -or $s.git_commit -ne $Commit) { throw 'preflight_not_passed_or_stale' }
}

function Assert-SmokePassed {
    $p = Join-Path $StatusDir 'smoke.json'
    if (!(Test-Path $p)) { throw 'smoke_not_passed: smoke.json missing' }
    $s = Get-Content -Raw $p | ConvertFrom-Json
    if ($s.status -ne 'passed' -or $s.git_commit -ne $Commit) { throw 'smoke_not_passed_or_stale' }
}

function Invoke-Preflight {
    if (!(Test-Path $Python)) { throw "Python not found: $Python" }
    if (!(Test-Path $Config)) { throw "Config not found: $Config" }
    Ensure-RuntimeDirs
    Invoke-Logged 'environment' @('-c', "import sys,torch,numpy,scipy; print('python',sys.version.split()[0]); print('torch',torch.__version__); print('numpy',numpy.__version__); print('scipy',scipy.__version__); print('cuda_available',torch.cuda.is_available()); print('torch_cuda',torch.version.cuda); assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print('gpu',torch.cuda.get_device_name(0)); print('gpu_total_gib',p.total_memory/1024**3); assert p.total_memory >= 7*1024**3")
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $nvidiaLog = Join-Path $LogDir ((Get-Date -Format 'yyyyMMdd_HHmmss') + '_nvidia_smi.log')
        & nvidia-smi 2>&1 | Tee-Object -FilePath $nvidiaLog
        if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
    } else { throw 'nvidia-smi not found' }
    Invoke-Logged 'cpu_oracle_smoke' @('-c', "import sys,numpy as np; sys.path[:0]=['src']; from solvers.cn_psor import SolverConfig,solve_option; r=solve_option(SolverConfig(m=32,n=32,q=.02,sigma=.25)); assert r.values.dtype==np.float64 and np.isfinite(r.values).all(); print('cpu_float64_oracle_smoke=passed')")
    Invoke-Logged 'focused_tests' @('-m','pytest','-q','-p','no:cacheprovider',$FocusedTest)
    Invoke-Logged 'preflight' @($Experiment,'--phase','preflight')
}

function Invoke-Smoke {
    Assert-PreflightPassed
    Invoke-Logged 'smoke' @($Experiment,'--phase','smoke')
}

function Get-GpuHours {
    $total = 0.0
    foreach ($job in $CanonicalOrder) {
        $p = Join-Path $StatusDir "canonical_$job.json"
        if (Test-Path $p) {
            $s = Get-Content -Raw $p | ConvertFrom-Json
            if ($s.status -eq 'completed' -and $s.git_commit -eq $Commit -and $null -ne $s.metrics.training_seconds) {
                $total += [double]$s.metrics.training_seconds / 3600.0
            }
        }
    }
    return $total
}

function Test-DimensionHasCorrectModel {
    param([int]$Dimension)
    foreach ($variant in @('SUP','HVI')) {
        $p = Join-Path $StatusDir ("canonical_d{0}_{1}.json" -f $Dimension,$variant)
        if (!(Test-Path $p)) { continue }
        $s = Get-Content -Raw $p | ConvertFrom-Json
        if ($s.status -ne 'completed' -or $s.git_commit -ne $Commit) { continue }
        $m = $s.metrics
        $finite = (-not [double]::IsNaN([double]$m.normalized_price_rmse) -and -not [double]::IsInfinity([double]$m.normalized_price_rmse) -and -not [double]::IsNaN([double]$m.trusted_pde_p95) -and -not [double]::IsInfinity([double]$m.trusted_pde_p95) -and -not [double]::IsNaN([double]$m.gamma_contraction_rmse) -and -not [double]::IsInfinity([double]$m.gamma_contraction_rmse))
        if ($finite -and [double]$m.obstacle_violation_max -le 1e-6 -and [double]$m.peak_cuda_gib -lt 7.0) { return $true }
    }
    return $false
}

function Invoke-Canonical {
    Assert-PreflightPassed
    Assert-SmokePassed
    $previousDimension = $null
    foreach ($job in $CanonicalOrder) {
        $dimension = [int](($job -split '_')[0].Substring(1))
        if ($null -ne $previousDimension -and $dimension -gt $previousDimension) {
            if (!(Test-DimensionHasCorrectModel $previousDimension)) {
                throw "dimension_ladder_stopped: both d$previousDimension models failed correctness"
            }
        }
        $hours = Get-GpuHours
        if ($hours -ge 24.0) { throw "gpu_hour_budget_blocked: $hours GPU-hours already consumed" }
        Invoke-Logged $job @($Experiment,'--phase','canonical-job','--job',$job)
        $hours = Get-GpuHours
        if ($hours -gt 24.0) { throw 'gpu_hour_budget_blocked: canonical total exceeded 24 GPU-hours' }
        $previousDimension = $dimension
    }
    Invoke-Logged 'finalize' @($Experiment,'--phase','finalize')
}

try {
    switch ($Phase) {
        'preflight' { Invoke-PhaseWithStatus 'preflight' { Invoke-Preflight } }
        'smoke'     { Invoke-PhaseWithStatus 'smoke' { Invoke-Smoke } }
        'canonical' { Invoke-PhaseWithStatus 'canonical' { Invoke-Canonical } }
        'all'       {
            $allStarted=Get-Date
            Invoke-PhaseWithStatus 'preflight' { Invoke-Preflight }
            Invoke-PhaseWithStatus 'smoke' { Invoke-Smoke }
            Invoke-PhaseWithStatus 'canonical' { Invoke-Canonical }
            Write-RunnerStatus 'all' 'passed' $allStarted
        }
        'status'    { & (Join-Path $PSScriptRoot 'check_hd1_windows_status.ps1') -Python $Python; exit $LASTEXITCODE }
    }
} catch {
    if ($Phase -eq 'all') {
        if ($null -eq $allStarted) { $allStarted=Get-Date }
        Write-RunnerStatus 'all' 'failed' $allStarted $_.Exception.Message
    }
    Write-Error $_
    exit 1
}
exit 0
