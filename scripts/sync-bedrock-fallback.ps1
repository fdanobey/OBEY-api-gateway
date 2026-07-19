<#
.SYNOPSIS
Verifies the generated AWS Bedrock fallback catalogs against AWS documentation.
.DESCRIPTION
Docs-first drift verifier. Reads exact model IDs and compatibility from a
committed snapshot or a user-supplied JSON file, validates every generated
fallback block, and emits a compact JSON summary. Live AWS listings are
advisory and never remove a documented active model.

Exit codes: 0 no drift; 1 drift found; 2 advisory AWS check unavailable;
3 source/snapshot parse failure; 4 script error.
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$StatePath,
    [string]$FromCache,
    [switch]$ValidateFixtures,
    [string[]]$Regions = @('us-east-1', 'us-west-2')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$RepoRoot = Split-Path $PSScriptRoot -Parent
$SourcePath = Join-Path $RepoRoot 'crates\ai-gateway\src\providers\bedrock.rs'
$UiPath = Join-Path $RepoRoot 'crates\ai-gateway\src\admin\static\index.html'
$DefaultSnapshot = Join-Path $PSScriptRoot 'fixtures\bedrock\catalog.json'
if ([string]::IsNullOrWhiteSpace($StatePath)) {
    $StatePath = Join-Path $PSScriptRoot '.cache\bedrock-drift-state.json'
}

$Catalogs = [ordered]@{
    mantle_chat = [ordered]@{ rust_begin = '// BEGIN BEDROCK MANTLE CHAT FALLBACK MODELS'; rust_end = '// END BEDROCK MANTLE CHAT FALLBACK MODELS'; ui_begin = '// BEGIN BEDROCK MANTLE CHAT FALLBACK MODELS'; ui_end = '// END BEDROCK MANTLE CHAT FALLBACK MODELS'; api = 'chat_completions'; endpoint = 'bedrock-mantle' }
    mantle_responses = [ordered]@{ rust_begin = '// BEGIN BEDROCK MANTLE RESPONSES FALLBACK MODELS'; rust_end = '// END BEDROCK MANTLE RESPONSES FALLBACK MODELS'; ui_begin = '// BEGIN BEDROCK MANTLE RESPONSES FALLBACK MODELS'; ui_end = '// END BEDROCK MANTLE RESPONSES FALLBACK MODELS'; api = 'responses'; endpoint = 'bedrock-mantle' }
    mantle_messages = [ordered]@{ rust_begin = '// BEGIN BEDROCK MANTLE MESSAGES FALLBACK MODELS'; rust_end = '// END BEDROCK MANTLE MESSAGES FALLBACK MODELS'; ui_begin = '// BEGIN BEDROCK MANTLE MESSAGES FALLBACK MODELS'; ui_end = '// END BEDROCK MANTLE MESSAGES FALLBACK MODELS'; api = 'messages'; endpoint = 'bedrock-mantle' }
    runtime = [ordered]@{ rust_begin = '// BEGIN BEDROCK RUNTIME FALLBACK MODELS'; rust_end = '// END BEDROCK RUNTIME FALLBACK MODELS'; ui_begin = '// BEGIN BEDROCK RUNTIME FALLBACK MODELS'; ui_end = '// END BEDROCK RUNTIME FALLBACK MODELS'; api = 'converse'; endpoint = 'bedrock-runtime' }
}

function Write-Summary {
    param([hashtable]$Summary)
    $Summary | ConvertTo-Json -Depth 10 -Compress
}

function Get-DelimitedBlock {
    param([string]$Text, [string]$Begin, [string]$End)
    $start = $Text.IndexOf($Begin, [System.StringComparison]::Ordinal)
    $finish = $Text.IndexOf($End, [System.StringComparison]::Ordinal)
    if ($start -lt 0 -or $finish -le $start) { throw "Missing generated block: $Begin" }
    $Text.Substring($start, ($finish + $End.Length) - $start)
}

function Get-RustIds {
    param([string]$Block)
    @([regex]::Matches($Block, 'id:\s*"(?<id>[^"]+)"') | ForEach-Object { $_.Groups['id'].Value })
}

function Get-UiIds {
    param([string]$Block)
    @([regex]::Matches($Block, "'(?<id>[^']+)'") | ForEach-Object { $_.Groups['id'].Value })
}

function Read-State {
    if (-not (Test-Path -LiteralPath $StatePath)) { return @{} }
    $raw = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    $result = @{}
    if ($raw.confirmations) {
        foreach ($property in $raw.confirmations.PSObject.Properties) { $result[$property.Name] = [int]$property.Value }
    }
    $result
}

function Write-State {
    param([hashtable]$Confirmations)
    $parent = Split-Path $StatePath -Parent
    if ($parent -and -not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $payload = [ordered]@{ updated_at = [datetime]::UtcNow.ToString('o'); confirmations = $Confirmations }
    [System.IO.File]::WriteAllText($StatePath, ($payload | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
}

function Get-AdvisoryLiveIds {
    param([string[]]$TargetRegions)
    $ids = @{}
    if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { return $ids }
    foreach ($region in $TargetRegions) {
        try {
            $json = & aws bedrock list-foundation-models --region $region --output json 2>$null
            if ($LASTEXITCODE -ne 0) { continue }
            $parsed = $json | ConvertFrom-Json
            foreach ($summary in @($parsed.modelSummaries)) {
                if ($summary.modelId) { $ids[[string]$summary.modelId] = [string]$summary.modelLifecycle.status }
            }
        } catch { }
    }
    $ids
}

try {
    $snapshotPath = if ($FromCache) { $FromCache } else { $DefaultSnapshot }
    if (-not (Test-Path -LiteralPath $snapshotPath)) { throw "Catalog snapshot not found: $snapshotPath" }
    $snapshot = Get-Content -LiteralPath $snapshotPath -Raw | ConvertFrom-Json
    if (-not $snapshot.models) { throw 'Catalog snapshot has no models array.' }

    $source = [System.IO.File]::ReadAllText($SourcePath)
    $ui = [System.IO.File]::ReadAllText($UiPath)
    $state = Read-State
    $nextState = @{}
    $drift = New-Object System.Collections.Generic.List[object]
    $catalogSummary = [ordered]@{}

    foreach ($catalogName in $Catalogs.Keys) {
        $definition = $Catalogs[$catalogName]
        $rustIds = @(Get-RustIds (Get-DelimitedBlock $source $definition.rust_begin $definition.rust_end))
        $uiIds = @(Get-UiIds (Get-DelimitedBlock $ui $definition.ui_begin $definition.ui_end))
        $expected = @($snapshot.models | Where-Object {
            $_.lifecycle -eq 'ACTIVE' -and $_.endpoint -eq $definition.endpoint -and $_.api -eq $definition.api -and -not $_.preview -and -not $_.moderation
        } | ForEach-Object { [string]$_.id } | Sort-Object -Unique)

        $missingRust = @($expected | Where-Object { $_ -notin $rustIds })
        $missingUi = @($expected | Where-Object { $_ -notin $uiIds })
        $extraRust = @($rustIds | Where-Object { $_ -notin $expected })
        $extraUi = @($uiIds | Where-Object { $_ -notin $expected })

        foreach ($id in $extraRust) {
            $key = "$catalogName|$id"
            $nextState[$key] = if ($state.ContainsKey($key)) { $state[$key] + 1 } else { 1 }
        }
        foreach ($id in @($state.Keys | Where-Object { $_ -like "$catalogName|*" -and $_ -notin $nextState.Keys })) { $nextState[$id] = 0 }

        if ($missingRust.Count -or $missingUi.Count -or $extraRust.Count -or $extraUi.Count) {
            $drift.Add([ordered]@{ catalog = $catalogName; missing_rust = $missingRust; missing_ui = $missingUi; extra_rust = $extraRust; extra_ui = $extraUi })
        }
        $catalogSummary[$catalogName] = [ordered]@{ expected = $expected.Count; rust = $rustIds.Count; ui = $uiIds.Count }
    }

    Write-State $nextState
    $live = Get-AdvisoryLiveIds $Regions
    $summary = [ordered]@{
        status = if ($drift.Count) { 'drift' } else { 'current' }
        dry_run = [bool]$DryRun
        source = $snapshot.source
        snapshot_date = $snapshot.generated_at
        catalogs = $catalogSummary
        drift = $drift.ToArray()
        advisory_live_models = $live.Count
        removal_confirmations = $nextState
        fixture_validation = [bool]$ValidateFixtures
    }
    Write-Summary $summary
    if ($drift.Count) { exit 1 }
    if ($ValidateFixtures) { exit 0 }
    if ($live.Count -eq 0 -and $env:BEDROCK_REQUIRE_ADVISORY_LIVE -eq 'true') { exit 2 }
    exit 0
} catch {
    Write-Summary ([ordered]@{ status = 'error'; message = $_.Exception.Message })
    exit 3
}
