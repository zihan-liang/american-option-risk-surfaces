param(
    [string]$CommitMessage = "results(hd3): add matched American-put single-seed run",
    [string]$TargetRepoUrl = "https://github.com/zihan-liang/american-option-risk-surfaces.git"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResultRoot = Join-Path $RepoRoot "results\hd3_matched_american_put_comparison"
$ExpectedBranch = "exp/hd3-matched-american-put-comparison"

Push-Location $RepoRoot
try {
    $Branch = (git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) {
        throw "Refusing to commit results from branch '$Branch'. Expected '$ExpectedBranch'."
    }
    if (-not (Test-Path (Join-Path $ResultRoot "final_status.json"))) {
        throw "final_status.json is missing. Run finalization successfully before uploading."
    }
    $Final = Get-Content (Join-Path $ResultRoot "final_status.json") -Raw | ConvertFrom-Json
    if ($Final.status -ne "completed") {
        throw "Final status is not completed."
    }

    if ($TargetRepoUrl -notmatch "github\.com[:/]zihan-liang/american-option-risk-surfaces(?:\.git)?$") {
        throw "Refusing to push to unexpected repository URL: $TargetRepoUrl"
    }

    $Files = Get-ChildItem $ResultRoot -Recurse -File | Where-Object {
        $_.FullName -notlike "*\runtime\*"
    }
    if (-not $Files) {
        throw "No compact result artifacts found."
    }
    $TooLarge = $Files | Where-Object { $_.Length -ge 50MB }
    if ($TooLarge) {
        $Names = ($TooLarge | ForEach-Object { "$($_.FullName) ($([math]::Round($_.Length/1MB,2)) MB)" }) -join "`n"
        throw "Refusing to stage artifacts >= 50 MB:`n$Names"
    }

    foreach ($File in $Files) {
        # Compatible with Windows PowerShell 5.1 / .NET Framework (no Path.GetRelativePath dependency).
        $Prefix = $RepoRoot.TrimEnd('\') + '\'
        if (-not $File.FullName.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Artifact is unexpectedly outside the repository: $($File.FullName)"
        }
        $Relative = $File.FullName.Substring($Prefix.Length)
        if ($File.Extension -eq ".npz") {
            git add -f -- $Relative
        }
        else {
            git add -- $Relative
        }
        if ($LASTEXITCODE -ne 0) { throw "git add failed for $Relative" }
    }

    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw "git diff --cached --check failed." }

    Write-Host "`n===== STAGED RESULT SUMMARY =====" -ForegroundColor Cyan
    git diff --cached --stat
    if ($LASTEXITCODE -ne 0) { throw "Unable to show staged diff." }

    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "No new compact results to commit."
    }
    else {
        git commit -m $CommitMessage
        if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
    }

    Write-Host "`nPushing matched results to: $TargetRepoUrl" -ForegroundColor Cyan
    git -c http.version=HTTP/1.1 push $TargetRepoUrl "HEAD:$ExpectedBranch"
    if ($LASTEXITCODE -ne 0) { throw "git push failed." }

    Write-Host "`nMatched comparison results pushed successfully." -ForegroundColor Green
    git rev-parse HEAD
}
finally {
    Pop-Location
}
