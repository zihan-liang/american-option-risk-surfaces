param(
    [ValidateSet("all", "preflight", "hd2", "hd3", "finalize", "status")]
    [string]$Phase = "all",

    [string]$Python = ".\.venv-pinn\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
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

    $LogDir = Join-Path $RepoRoot "results\hd_pipeline\runtime\logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $LogFile = Join-Path $LogDir "release-pipeline-$Phase-$Stamp.log"

    & $PythonExe "experiments\hd_pipeline_windows_release.py" "--phase" $Phase 2>&1 |
        Tee-Object -FilePath $LogFile

    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        Write-Error "HD2->HD3 release phase '$Phase' exited with code $ExitCode. Re-run the same command to resume. Log: $LogFile"
        exit $ExitCode
    }
    exit 0
}
finally {
    Pop-Location
}
