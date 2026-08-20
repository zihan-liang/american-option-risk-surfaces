param(
    [ValidateSet("all", "test", "preflight", "canonical", "stress", "finalize", "status")]
    [string]$Phase = "all",

    [string]$Python = ".\.venv-pinn\Scripts\python.exe",

    [switch]$IncludeStress100
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Experiment = Join-Path $RepoRoot "experiments\hd3_matched_windows_entrypoint.py"
$ResultRoot = Join-Path $RepoRoot "results\hd3_matched_american_put_comparison"
$LogDir = Join-Path $ResultRoot "runtime\logs"

function Invoke-PythonPhase {
    param([string]$Name, [string[]]$ExtraArgs = @())
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $LogFile = Join-Path $LogDir "$Name-$Stamp.log"
    Write-Host "`n===== HD3 MATCHED: $Name =====" -ForegroundColor Cyan

    # Windows PowerShell 5.1 can promote native stderr to a terminating
    # NativeCommandError when ErrorActionPreference is Stop.  Temporarily use
    # Continue so the complete Python traceback reaches both the console and
    # the durable log, then enforce failure from the native exit code.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PythonExe $Experiment "--phase" $Name @ExtraArgs 2>&1 | Tee-Object -FilePath $LogFile
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "Phase '$Name' exited with code $ExitCode. Re-run the same command to resume. Log: $LogFile"
    }
}

Push-Location $RepoRoot
try {
    if ([System.IO.Path]::IsPathRooted($Python)) {
        $PythonExe = $Python
    }
    else {
        $PythonExe = Join-Path $RepoRoot $Python
    }
    if (-not (Test-Path $PythonExe)) {
        throw "Python interpreter not found: $PythonExe"
    }
    if (-not (Test-Path $Experiment)) {
        throw "Experiment entry point not found: $Experiment"
    }
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    Write-Host "Repository : $RepoRoot"
    Write-Host "Python     : $PythonExe"
    Write-Host "Branch     : $(git branch --show-current)"
    Write-Host "Commit     : $(git rev-parse HEAD)"

    if ($Phase -eq "test" -or $Phase -eq "all") {
        Write-Host "`n===== HD3 MATCHED: focused CPU-safe tests =====" -ForegroundColor Cyan
        & $PythonExe -m pytest "tests\test_hd3_matched_american_put_comparison.py" -q
        if ($LASTEXITCODE -ne 0) { throw "Focused tests failed." }
        & $PythonExe -m compileall -q `
            "experiments\hd3_matched_american_put_comparison.py" `
            "experiments\hd3_matched_windows_entrypoint.py" `
            "tests\test_hd3_matched_american_put_comparison.py"
        if ($LASTEXITCODE -ne 0) { throw "Compile check failed." }
    }

    if ($Phase -eq "preflight" -or $Phase -eq "all") {
        Invoke-PythonPhase "preflight"
    }

    if ($Phase -eq "canonical" -or $Phase -eq "all") {
        Invoke-PythonPhase "canonical"
    }

    if ($Phase -eq "stress") {
        Invoke-PythonPhase "stress"
    }
    elseif ($Phase -eq "all" -and $IncludeStress100) {
        Invoke-PythonPhase "stress"
    }

    if ($Phase -eq "finalize" -or $Phase -eq "all") {
        $FinalizeArgs = @()
        if ($IncludeStress100) { $FinalizeArgs += "--include-stress" }
        Invoke-PythonPhase "finalize" $FinalizeArgs
    }

    if ($Phase -eq "status") {
        Invoke-PythonPhase "status"
    }

    if ($Phase -eq "all") {
        Write-Host "`n===== FINAL STATUS =====" -ForegroundColor Green
        & $PythonExe $Experiment "--phase" "status"
        if ($LASTEXITCODE -ne 0) { throw "Final status check failed." }
        Write-Host "`nCanonical matched run completed. Review results before committing." -ForegroundColor Green
        Write-Host "Result directory: $ResultRoot"
    }
}
finally {
    Pop-Location
}
