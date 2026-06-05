# bump-version.ps1 — Bump version across Cargo.toml + installer.iss, tag, and push
# Usage: .\scripts\bump-version.ps1 <major|minor|patch> [-NoPush]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("major","minor","patch")]
    [string]$Bump,

    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

$cargoPath = "crates/ai-gateway/Cargo.toml"
$issPath   = "scripts/installer.iss"

# --- Read current version from Cargo.toml ---
$cargoContent = Get-Content $cargoPath -Raw
if ($cargoContent -match 'version\s*=\s*"(\d+)\.(\d+)\.(\d+)"') {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    $patch = [int]$Matches[3]
} else {
    throw "Could not parse version from $cargoPath"
}

$oldVersion = "$major.$minor.$patch"

# --- Compute new version ---
switch ($Bump) {
    "major" { $major++; $minor = 0; $patch = 0 }
    "minor" { $minor++; $patch = 0 }
    "patch" { $patch++ }
}

$newVersion = "$major.$minor.$patch"
Write-Host "Bumping: $oldVersion -> $newVersion" -ForegroundColor Cyan

# --- Update Cargo.toml (first occurrence of version = "x.y.z") ---
$cargoContent = $cargoContent -replace "version\s*=\s*`"$oldVersion`"", "version = `"$newVersion`""
Set-Content $cargoPath -Value $cargoContent -NoNewline
Write-Host "  Updated $cargoPath" -ForegroundColor Green

# --- Update installer.iss ---
$issContent = Get-Content $issPath -Raw
$issContent = $issContent -replace "#define MyAppVersion\s+`"$oldVersion`"", "#define MyAppVersion   `"$newVersion`""
Set-Content $issPath -Value $issContent -NoNewline
Write-Host "  Updated $issPath" -ForegroundColor Green

# --- Generate changelog from commits since last tag ---
$lastTag = git describe --tags --abbrev=0 HEAD~ 2>$null
if (-not $lastTag) { $lastTag = (git rev-list --max-parents=0 HEAD) }

$commits = git log "$lastTag..HEAD" --pretty=format:"- %s" --no-merges | Where-Object { $_ -notmatch "^- chore: bump version" }

$lines = @("## What's Changed", "")
$features = $commits | Where-Object { $_ -match "^- feat" }
$fixes    = $commits | Where-Object { $_ -match "^- fix" }
$other    = $commits | Where-Object { $_ -notmatch "^- feat" -and $_ -notmatch "^- fix" }

if ($features) { $lines += "### Features"; $lines += $features; $lines += "" }
if ($fixes)    { $lines += "### Fixes";    $lines += $fixes;    $lines += "" }
if ($other)    { $lines += "### Other";    $lines += $other;    $lines += "" }

$lines += "**Full Changelog**: https://github.com/fdanobey/OBEY-api-gateway/compare/$lastTag...v$newVersion"

$changelog = $lines -join "`n"

Write-Host "`nChangelog:" -ForegroundColor Cyan
Write-Host $changelog

# --- Git commit + tag (annotated with changelog via temp file) ---
git add $cargoPath $issPath
git commit -m "chore: bump version to $newVersion"

$tagMsgFile = [System.IO.Path]::GetTempFileName()
[System.IO.File]::WriteAllText($tagMsgFile, $changelog, (New-Object System.Text.UTF8Encoding($false)))
git tag -a "v$newVersion" -F $tagMsgFile
Remove-Item $tagMsgFile -Force
Write-Host "  Tagged v$newVersion (annotated)" -ForegroundColor Green

if (-not $NoPush) {
    git push
    git push origin "v$newVersion"
    Write-Host "  Pushed to origin (CI release will trigger)" -ForegroundColor Green
} else {
    Write-Host "  Skipped push (-NoPush). Run: git push && git push origin v$newVersion" -ForegroundColor Yellow
}

Write-Host "`nDone: v$newVersion" -ForegroundColor Cyan
