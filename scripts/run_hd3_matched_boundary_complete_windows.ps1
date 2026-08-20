param(
    [ValidateSet("all", "test", "preflight", "canonical", "finalize", "status")]
    [string]$Phase = "all",

    [string]$Python = ".\.venv-pinn\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Experiment = Join-Path $RepoRoot "experiments\hd3_matched_boundary_complete_windows_entrypoint.py"
$ResultRoot = Join-Path $RepoRoot "results\hd3_matched_american_put_boundary_complete"
$LogDir = Join-Path $ResultRoot "runtime\logs"
$ExpectedBranch = "exp/hd3-matched-american-put-final-boundary-complete"

function Invoke-PythonPhase {
    param([string]$Name, [string[]]$ExtraArgs = @())
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $LogFile = Join-Path $LogDir "$Name-$Stamp.log"
    Write-Host "`n===== HD3 BOUNDARY-COMPLETE: $Name =====" -ForegroundColor Cyan

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
        throw "Phase '$Name' exited with code $ExitCode. Re-run the same command to resume only this final experiment. Log: $LogFile"
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
    $Branch = (git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) {
        throw "Refusing final run from branch '$Branch'. Expected '$ExpectedBranch'."
    }
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    Write-Host "Repository : $RepoRoot"
    Write-Host "Python     : $PythonExe"
    Write-Host "Branch     : $Branch"
    Write-Host "Commit     : $(git rev-parse HEAD)"
    Write-Host "Results    : $ResultRoot"

    if ($Phase -eq "test" -or $Phase -eq "all") {
        Write-Host "`n===== HD3 BOUNDARY-COMPLETE: focused CPU-safe tests =====" -ForegroundColor Cyan
        & $PythonExe -m pytest "tests\test_hd3_matched_boundary_complete.py" -q
        if ($LASTEXITCODE -ne 0) { throw "Focused tests failed." }

        Write-Host "`n===== HD3 BOUNDARY-COMPLETE: compile check =====" -ForegroundColor Cyan
        & $PythonExe -m compileall -q `
            "experiments\hd3_matched_boundary_complete_core.py" `
            "experiments\hd3_matched_american_put_boundary_complete.py" `
            "experiments\hd3_matched_boundary_complete_windows_entrypoint.py" `
            "tests\test_hd3_matched_boundary_complete.py"
        if ($LASTEXITCODE -ne 0) { throw "Compile check failed." }
    }

    if ($Phase -eq "preflight" -or $Phase -eq "all") {
        Invoke-PythonPhase "preflight"
    }

    if ($Phase -eq "canonical" -or $Phase -eq "all") {
        Invoke-PythonPhase "canonical"
    }

    if ($Phase -eq "finalize" -or $Phase -eq "all") {
        Invoke-PythonPhase "finalize"
    }

    if ($Phase -eq "status") {
        Invoke-PythonPhase "status"
    }

    if ($Phase -eq "all") {
        Write-Host "`n===== FINAL STATUS =====" -ForegroundColor Green
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $PythonExe $Experiment "--phase" "status" 2>&1
            $ExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($ExitCode -ne 0) { throw "Final status check failed." }
        Write-Host "`nFinal boundary-complete matched run completed. Review results before committing." -ForegroundColor Green
        Write-Host "Result directory: $ResultRoot"
    }
}
finally {
    Pop-Location
}
