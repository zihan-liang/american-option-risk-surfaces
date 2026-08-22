# Usage: .\scripts\run_binomial_benchmark_windows.ps1 -Phase all -Workers 4
param(
    [ValidateSet('preflight','probe','confirm','all')]
    [string]$Phase = 'all',
    [int]$Workers = 4,
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($Python)) {
    $candidate = Join-Path $RepoRoot '.venv-pinn\Scripts\python.exe'
    if (Test-Path $candidate) {
        $Python = $candidate
    } else {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $cmd) {
            throw 'Python interpreter not found. Pass -Python <path-to-python.exe>.'
        }
        $Python = $cmd.Source
    }
}

if (-not (Test-Path $Python)) {
    $resolvedCommand = Get-Command $Python -ErrorAction SilentlyContinue
    if ($null -eq $resolvedCommand) {
        throw "Python interpreter not found: $Python"
    }
    $Python = $resolvedCommand.Source
}

Write-Host "Repository: $RepoRoot"
Write-Host "Python:     $Python"
Write-Host "Phase:      $Phase"
Write-Host "Workers:    $Workers"

& $Python -c "import sys, numpy; print('python =', sys.version.split()[0]); print('numpy =', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Python/NumPy environment check failed.' }

& $Python -m unittest tests.test_binomial_benchmark_convergence -v
if ($LASTEXITCODE -ne 0) { throw 'Focused unit tests failed.' }

& $Python experiments\binomial_benchmark_convergence.py --phase preflight --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Benchmark preflight failed.' }

if ($Phase -ne 'preflight') {
    & $Python experiments\binomial_benchmark_convergence.py --phase $Phase --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Benchmark phase failed: $Phase" }
}

Write-Host ''
Write-Host 'Result directory:'
Write-Host (Join-Path $RepoRoot 'results\binomial_benchmark_convergence')
Write-Host 'Run is resumable: rerun the same command after interruption.'
