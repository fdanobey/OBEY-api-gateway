<#
.SYNOPSIS
Synchronizes the curated NVIDIA hosted NIM fallback catalog.
.DESCRIPTION
Exit codes: 0 no drift; 1 drift (rewritten unless -DryRun); 2 missing key;
3 auth/upstream hard failure; 4 script logic error. Candidate filtering excludes
obvious non-chat IDs (embed, rerank, reward, vl-, vision, OCR, parse, PII, ASR,
TTS, STT, grouponly, retriever). HTTP 429 counts as available; 5xx/timeouts are
transient and reset confirmation. -FromCache performs no network requests.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 20)]
    [int]$TopN = 3,
    [switch]$DryRun,
    [string]$StatePath,
    [string]$FromCache,
    [switch]$ValidateFixtures
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ([string]::IsNullOrWhiteSpace($StatePath)) {
    $StatePath = Join-Path $PSScriptRoot '.cache\nim-drift-state.json'
}
$BaseUrl = 'https://integrate.api.nvidia.com/v1'
$SourcePath = Join-Path (Split-Path $PSScriptRoot -Parent) 'crates\ai-gateway\src\providers\nvidia_nim.rs'
$BeginMarker = '// BEGIN NVIDIA NIM FALLBACK MODELS'
$EndMarker = '// END NVIDIA NIM FALLBACK MODELS'

function Write-JsonSummary {
    param([hashtable]$Summary)
    $Summary | ConvertTo-Json -Depth 8 -Compress
}

function Get-HttpStatusCode {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)
    try {
        return [int]$ErrorRecord.Exception.Response.StatusCode
    } catch {
        return 0
    }
}

function Get-CurrentFallbackEntries {
    param([string]$Path)
    $source = [System.IO.File]::ReadAllText($Path)
    $start = $source.IndexOf($BeginMarker, [System.StringComparison]::Ordinal)
    $end = $source.IndexOf($EndMarker, [System.StringComparison]::Ordinal)
    if ($start -lt 0 -or $end -le $start) {
        throw 'Could not find the NVIDIA NIM generated fallback block.'
    }
    $block = $source.Substring($start, ($end + $EndMarker.Length) - $start)
    $matches = [regex]::Matches($block, 'id:\s*"(?<id>[^"]+)"[\s\S]*?owned_by:\s*"(?<owner>[^"]+)"[\s\S]*?supports_vision:\s*(?<vision>true|false)[\s\S]*?context_window:\s*(?<context>None|Some\([\d_]+\))[\s\S]*?max_completion_tokens:\s*(?<completion>None|Some\([\d_]+\))[\s\S]*?source_url:\s*"(?<url>[^"]+)"')
    if ($matches.Count -eq 0) {
        throw 'The NVIDIA NIM generated block did not contain parseable entries.'
    }
    @($matches | ForEach-Object {
        [pscustomobject]@{
            id = $_.Groups['id'].Value
            owned_by = $_.Groups['owner'].Value
            supports_vision = $_.Groups['vision'].Value -eq 'true'
            context_window = Convert-RustOptionToNumber $_.Groups['context'].Value
            max_completion_tokens = Convert-RustOptionToNumber $_.Groups['completion'].Value
            source_url = $_.Groups['url'].Value
        }
    })
}

function Convert-RustOptionToNumber {
    param([string]$Value)
    if ($Value -eq 'None') { return $null }
    [uint32](($Value -replace '^Some\(', '' -replace '\)$', '') -replace '_', '')
}

function Convert-NumberToRustOption {
    param($Value)
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) { return 'None' }
    "Some($([uint32]$Value))"
}

function Get-AllCatalogModels {
    param([string]$ApiKey)
    $headers = @{ Authorization = "Bearer $ApiKey"; Accept = 'application/json' }
    $models = @()
    $nextUrl = "$BaseUrl/models"
    $visited = @{}
    while ($nextUrl) {
        if ($visited.ContainsKey($nextUrl)) { throw 'NVIDIA model pagination loop detected.' }
        $visited[$nextUrl] = $true
        try {
            $page = Invoke-RestMethod -Method Get -Uri $nextUrl -Headers $headers -TimeoutSec 30
        } catch {
            $status = Get-HttpStatusCode $_
            if ($status -in 401, 403) { throw "AUTH:$status" }
            throw "Failed to fetch NVIDIA model catalog (HTTP $status): $($_.Exception.Message)"
        }
        $models += @($page.data | Where-Object { $_.id } | ForEach-Object { [string]$_.id })
        $hasMore = $page.has_more -eq $true
        $token = @($page.next_page_token, $page.next_token, $page.pagination_token, $page.after | Where-Object { $_ })[0]
        if ($hasMore -or $token) {
            if (-not $token) { throw 'INCOMPLETE:NVIDIA catalog reports more pages without a continuation token.' }
            $separator = if ($BaseUrl.Contains('?')) { '&' } else { '?' }
            $nextUrl = "$BaseUrl/models${separator}page_token=$([uri]::EscapeDataString([string]$token))"
        } else {
            $nextUrl = $null
        }
    }
    @($models | Sort-Object -Unique)
}

function Test-IsChatCandidate {
    param([string]$Id)
    $Id -notmatch '(?i)(embed|rerank|reward|vl-|vision|ocr|parse|pii|asr|tts|stt|grouponly|retriever)'
}

function Test-ChatModel {
    param([string]$Id, [string]$ApiKey)
    $headers = @{ Authorization = "Bearer $ApiKey"; Accept = 'application/json'; 'Content-Type' = 'application/json' }
    $body = @{
        model = $Id
        messages = @(@{ role = 'user'; content = 'Respond only with OK.' })
        max_tokens = 8
        stream = $false
    } | ConvertTo-Json -Depth 5 -Compress
    try {
        $response = Invoke-RestMethod -Method Post -Uri "$BaseUrl/chat/completions" -Headers $headers -Body $body -TimeoutSec 30
        if ($response.choices -or $response.model) { return 'available' }
        return 'skipped'
    } catch {
        $status = Get-HttpStatusCode $_
        switch ($status) {
            429 { return 'available' }
            401 { return 'auth' }
            403 { return 'auth' }
            404 { return 'absent' }
            400 { return 'skipped' }
            422 { return 'skipped' }
            default { return 'transient' }
        }
    }
}

function Convert-ParameterCount {
    param([string]$Text)
    if ($Text -match '(?i)(?<value>\d+(?:\.\d+)?)\s*(?<unit>[KMBT])(?:illion)?') {
        $multiplier = switch ($Matches['unit'].ToUpperInvariant()) {
            'K' { 1e3 }
            'M' { 1e6 }
            'B' { 1e9 }
            'T' { 1e12 }
        }
        return [double]$Matches['value'] * $multiplier
    }
    0
}

function Get-ModelMetadata {
    param([string]$Id)
    # NVIDIA build pages replace dots with underscores in the model-name segment
    # e.g. nvidia/llama-3.1-nemotron-70b-instruct -> nvidia/llama-3_1-nemotron-70b-instruct
    $parts = $Id -split '/', 2
    $sluggedName = if ($parts.Count -eq 2) { "$($parts[0])/$($parts[1] -replace '\.', '_')" } else { $Id -replace '\.', '_' }
    $url = "https://build.nvidia.com/$sluggedName"
    $metadata = [ordered]@{
        id = $Id
        owned_by = ($Id -split '/', 2)[0]
        source_url = $url
        updated = [datetime]::MinValue.ToString('o')
        release_date = [datetime]::MinValue.ToString('o')
        capability_count = 0
        total_parameters = 0
        active_parameters = 0
        supports_vision = $false
        context_window = $null
        max_completion_tokens = $null
        metadata_available = $false
    }
    try {
        $content = (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 15 -Headers @{ Accept = 'text/markdown,text/plain,text/html' }).Content
        $metadata.metadata_available = $true
        if ($content -match '(?im)^updated:\s*["'']?(?<date>[^\r\n"'']+)') {
            $parsedDate = [datetime]::MinValue
            if ([datetime]::TryParse($Matches['date'].Trim(), [ref]$parsedDate)) { $metadata.updated = $parsedDate.ToUniversalTime().ToString('o') }
        }
        if ($content -match '(?im)^(release_date|release date):\s*["'']?(?<date>[^\r\n"'']+)') {
            $parsedDate = [datetime]::MinValue
            if ([datetime]::TryParse($Matches['date'].Trim(), [ref]$parsedDate)) { $metadata.release_date = $parsedDate.ToUniversalTime().ToString('o') }
        } elseif ($content -match '(?im)(Release Date|build\.nvidia\.com):?\s*[^\r\n]*(?<date>\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})') {
            $parsedDate = [datetime]::MinValue
            if ([datetime]::TryParse($Matches['date'], [ref]$parsedDate)) { $metadata.release_date = $parsedDate.ToUniversalTime().ToString('o') }
        }
        $capabilityCount = 0
        foreach ($capability in @('Function Calling', 'Structured Output', 'Reasoning')) {
            if ($content -match "(?i)$([regex]::Escape($capability)):\s*(Supported|Yes|True)") { $capabilityCount++ }
        }
        if ($content -match '(?i)(Input Types?|Input):[^\r\n]*(Image|Video)|Vision[^\r\n]*Supported') {
            $metadata.supports_vision = $true
            $capabilityCount++
        }
        $metadata.capability_count = $capabilityCount
        if ($content -match '(?im)Active Parameters:[^\r\n]*(?<params>\d+(?:\.\d+)?\s*[KMBT](?:illion)?)') {
            $metadata.active_parameters = Convert-ParameterCount $Matches['params']
        }
        if ($content -match '(?im)(Total Parameters|Parameters):[^\r\n]*(?<params>\d+(?:\.\d+)?\s*[KMBT](?:illion)?)') {
            $metadata.total_parameters = Convert-ParameterCount $Matches['params']
        }
        if ($metadata.active_parameters -eq 0) { $metadata.active_parameters = $metadata.total_parameters }
        if ($content -match '(?im)(Context Length|Input Context Length[^:]*):\s*(?<context>[\d,]+)') {
            $metadata.context_window = [uint32](($Matches['context']) -replace ',', '')
        }
        if ($content -match '(?im)(Max(?:imum)? (?:Output|Completion) Tokens):\s*(?<completion>[\d,]+)') {
            $metadata.max_completion_tokens = [uint32](($Matches['completion']) -replace ',', '')
        }
    } catch {
        Write-Warning "Could not read NVIDIA model metadata for ${Id}: $($_.Exception.Message)"
    }
    [pscustomobject]$metadata
}

function Rank-AvailableModels {
    param([object[]]$Models)
    # Deterministic score ordering: release-date rank * 100, active-parameter
    # rank * 10, capability count * 5; lexical id ascending breaks final ties.
    @($Models | Sort-Object -Property @(
        @{ Expression = {
            $parsed = [datetime]::MinValue
            if ($_.updated -and [datetime]::TryParse([string]$_.updated, [ref]$parsed)) { $parsed } else { [datetime]::MinValue }
        }; Descending = $true },
        @{ Expression = { [double]$_.parameters }; Descending = $true },
        @{ Expression = { [int]$_.capability_count }; Descending = $true },
        @{ Expression = { [string]$_.id }; Descending = $false }
    ))
}

function Read-State {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return @{} }
    $raw = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $state = @{}
    if ($raw.confirmations) {
        foreach ($property in $raw.confirmations.PSObject.Properties) { $state[$property.Name] = [int]$property.Value }
    }
    $state
}

function Write-State {
    param([string]$Path, [hashtable]$Confirmations)
    $parent = Split-Path $Path -Parent
    if ($parent -and -not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $payload = [ordered]@{ updated_at = [datetime]::UtcNow.ToString('o'); confirmations = $Confirmations }
    [System.IO.File]::WriteAllText($Path, ($payload | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
}

function New-FallbackBlock {
    param([object[]]$Entries)
    try { $gitRevision = (& git -C (Split-Path $PSScriptRoot -Parent) rev-parse --short HEAD 2>$null).Trim() } catch { $gitRevision = 'unknown' }
    if ([string]::IsNullOrWhiteSpace($gitRevision)) { $gitRevision = 'unknown' }
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add($BeginMarker)
    $lines.Add("/// Probe provenance: catalog=$BaseUrl/models; probed=$([datetime]::UtcNow.ToString('o')); git_rev=$gitRevision")
    $lines.Add('pub const NVIDIA_NIM_FALLBACK_MODELS: &[NimFallbackModel] = &[')
    foreach ($entry in $Entries) {
        $lines.Add('    NimFallbackModel {')
        $lines.Add("        id: `"$($entry.id)`",")
        $lines.Add("        owned_by: `"$($entry.owned_by)`",")
        $lines.Add("        supports_vision: $(([string][bool]$entry.supports_vision).ToLowerInvariant()),")
        $lines.Add("        context_window: $(Convert-NumberToRustOption $entry.context_window),")
        $lines.Add("        max_completion_tokens: $(Convert-NumberToRustOption $entry.max_completion_tokens),")
        $lines.Add("        source_url: `"$($entry.source_url)`",")
        $lines.Add('    },')
    }
    $lines.Add('];')
    $lines.Add($EndMarker)
    $lines -join "`r`n"
}

function Set-FallbackBlock {
    param([string]$Path, [object[]]$Entries, [switch]$WhatIf)
    $source = [System.IO.File]::ReadAllText($Path)
    $start = $source.IndexOf($BeginMarker, [System.StringComparison]::Ordinal)
    $end = $source.IndexOf($EndMarker, [System.StringComparison]::Ordinal)
    if ($start -lt 0 -or $end -le $start) { throw 'Could not find generated NVIDIA NIM block.' }
    $end += $EndMarker.Length
    $replacement = New-FallbackBlock $Entries
    $updated = $source.Substring(0, $start) + $replacement + $source.Substring($end)
    if ($updated -eq $source) { return $false }
    if (-not $WhatIf) {
        [System.IO.File]::WriteAllText($Path, $updated, (New-Object System.Text.UTF8Encoding($false)))
    }
    return $true
}

function Invoke-FixtureValidation {
    $ranked = @(Rank-AvailableModels @(
        [pscustomobject]@{ id = 'z/old'; updated = '2025-01-01T00:00:00Z'; capability_count = 3; parameters = 1; supports_vision = $false },
        [pscustomobject]@{ id = 'a/new'; updated = '2026-01-01T00:00:00Z'; capability_count = 1; parameters = 1; supports_vision = $false }
    ))
    if ($ranked[0].id -ne 'a/new') { throw 'Ranking fixture failed.' }

    $state = @{}
    $status = 'absent'
    $state['current/model'] = if ($status -eq 'absent') { 1 } else { 0 }
    if ($state['current/model'] -ne 1) { throw 'First-run confirmation fixture failed.' }
    $state['current/model'] = if ($status -eq 'absent') { $state['current/model'] + 1 } else { 0 }
    if ($state['current/model'] -ne 2) { throw 'Second-run confirmation fixture failed.' }
    $status = 'transient'
    $state['current/model'] = if ($status -eq 'absent') { $state['current/model'] + 1 } else { 0 }
    if ($state['current/model'] -ne 0) { throw 'Transient reset fixture failed.' }

    Write-JsonSummary @{ status = 'fixtures_valid'; ranking = $ranked.id }
    exit 0
}

try {
    if ($ValidateFixtures) { Invoke-FixtureValidation }

    $current = @(Get-CurrentFallbackEntries $SourcePath)
    $cacheData = $null
    if ($FromCache) {
        $cacheData = Get-Content -LiteralPath $FromCache -Raw | ConvertFrom-Json
        $catalogIds = @($cacheData.catalog_ids)
        $probeResults = @{}
        foreach ($property in $cacheData.probe_results.PSObject.Properties) { $probeResults[$property.Name] = [string]$property.Value }
        $metadataById = @{}
        foreach ($item in @($cacheData.metadata)) { $metadataById[$item.id] = $item }
    } else {
        if ([string]::IsNullOrWhiteSpace($env:NVIDIA_API_KEY)) {
            Write-JsonSummary @{ status = 'error'; reason = 'NVIDIA_API_KEY is required' }
            exit 2
        }
        try {
            $catalogIds = @(Get-AllCatalogModels $env:NVIDIA_API_KEY)
        } catch {
            if ($_.Exception.Message -match '^AUTH:') { Write-JsonSummary @{ status = 'error'; reason = $_.Exception.Message }; exit 3 }
            if ($_.Exception.Message -match '^INCOMPLETE:') { Write-JsonSummary @{ status = 'error'; reason = $_.Exception.Message }; exit 3 }
            throw
        }
        $probeResults = @{}
        $metadataById = @{}
        foreach ($id in @($catalogIds | Where-Object { Test-IsChatCandidate $_ })) {
            $probe = Test-ChatModel $id $env:NVIDIA_API_KEY
            if ($probe -eq 'auth') { Write-JsonSummary @{ status = 'error'; reason = "Authentication failed while probing $id" }; exit 3 }
            $probeResults[$id] = $probe
            if ($probe -eq 'available') { $metadataById[$id] = Get-ModelMetadata $id }
        }
    }

    if (-not $FromCache) {
        $cacheDir = Join-Path $PSScriptRoot '.cache'
        if (-not (Test-Path -LiteralPath $cacheDir)) { New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null }
        $cachePayload = [ordered]@{ generated_at = [datetime]::UtcNow.ToString('o'); catalog_ids = $catalogIds; probe_results = $probeResults; metadata = @($metadataById.Values) }
        $cachePath = Join-Path $cacheDir ("nim-catalog-{0}.json" -f [datetime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
        [System.IO.File]::WriteAllText($cachePath, ($cachePayload | ConvertTo-Json -Depth 10), (New-Object System.Text.UTF8Encoding($false)))
    }

    $confirmations = Read-State $StatePath
    $confirmedAbsent = New-Object System.Collections.Generic.List[string]
    $transientCurrent = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $current) {
        $status = if (-not ($catalogIds -contains $entry.id)) { 'absent' } elseif ($probeResults.ContainsKey($entry.id)) { $probeResults[$entry.id] } else { 'transient' }
        if ($status -eq 'absent' -or $status -eq 'skipped') {
            $prior = if ($confirmations.ContainsKey($entry.id)) { [int]$confirmations[$entry.id] } else { 0 }
            $confirmations[$entry.id] = $prior + 1
            if ($confirmations[$entry.id] -ge 2) { $confirmedAbsent.Add($entry.id) }
        } else {
            $confirmations[$entry.id] = 0
            if ($status -eq 'transient') { $transientCurrent.Add($entry.id) }
        }
    }
    Write-State $StatePath $confirmations

    $availableMetadata = @()
    foreach ($id in $probeResults.Keys) {
        if ($probeResults[$id] -eq 'available') {
            if ($metadataById.ContainsKey($id)) {
                $availableMetadata += $metadataById[$id]
            } else {
                $availableMetadata += [pscustomobject]@{
                    id = $id; owned_by = ($id -split '/', 2)[0]; source_url = "https://build.nvidia.com/$id"
                    updated = [datetime]::MinValue.ToString('o'); release_date = [datetime]::MinValue.ToString('o'); capability_count = 0
                    total_parameters = 0; active_parameters = 0
                    supports_vision = $false; context_window = $null; max_completion_tokens = $null; metadata_available = $false
                }
            }
        }
    }
    $ranked = @(Rank-AvailableModels $availableMetadata)
    $proposed = New-Object System.Collections.Generic.List[object]
    foreach ($entry in $current) {
        if (-not $confirmedAbsent.Contains($entry.id)) { $proposed.Add($entry) }
    }
    foreach ($candidate in $ranked) {
        if ($proposed.Count -ge $TopN) { break }
        if (-not ($proposed.id -contains $candidate.id)) { $proposed.Add($candidate) }
    }

    if ($proposed.Count -lt $TopN) {
        Write-JsonSummary @{ status = 'error'; reason = 'Not enough confirmed hosted chat models to preserve fallback size'; current = $current.id; confirmed_absent = $confirmedAbsent }
        exit 3
    }
    if ($transientCurrent.Count -gt 0) {
        Write-JsonSummary @{ status = 'transient'; reason = 'Curated model probe was transient; no rewrite allowed'; transient_models = $transientCurrent; confirmations = $confirmations }
        exit 0
    }
    $proposed = @($proposed | Select-Object -First $TopN)
    $proposalDir = Join-Path $PSScriptRoot '.cache'
    if (-not (Test-Path -LiteralPath $proposalDir)) { New-Item -ItemType Directory -Path $proposalDir -Force | Out-Null }
    [System.IO.File]::WriteAllText((Join-Path $proposalDir 'nim-fallback-proposed.json'), ($proposed | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))
    $changedIds = (@($current.id) -join "`n") -ne (@($proposed.id) -join "`n")
    if (-not $changedIds) {
        Write-JsonSummary @{ status = 'unchanged'; current = $current.id; confirmations = $confirmations; available_candidates = $ranked.Count }
        exit 0
    }

    $changed = Set-FallbackBlock -Path $SourcePath -Entries $proposed -WhatIf:$DryRun
    Write-JsonSummary @{
        status = if ($DryRun) { 'change_proposed' } else { 'changed' }
        dry_run = [bool]$DryRun
        current = $current.id
        proposed = $proposed.id
        confirmed_absent = $confirmedAbsent
        transient_current = $transientCurrent
        source_changed = $changed
    }
    exit 1
} catch {
    Write-JsonSummary @{ status = 'error'; reason = $_.Exception.Message; type = $_.Exception.GetType().FullName }
    exit 4
}
