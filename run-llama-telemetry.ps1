[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$ServerPath,

    [ValidateSet("x64", "ARM64")]
    [string]$Architecture = $(if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq [System.Runtime.InteropServices.Architecture]::Arm64) { "ARM64" } else { "x64" }),

    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Configuration = "Release",

    [string]$HostAddress = "127.0.0.1",

    [ValidateRange(1, 65535)]
    [int]$Port = 8080,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$ContextLength = 8192,

    [string]$GpuLayers = "auto",

    [ValidateRange(1, [int]::MaxValue)]
    [int]$ParallelSlots = 1,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$BatchSize = 2048,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$UBatchSize = 512,

    [string]$SpecType = "none",

    [string]$SpecModelPath,

    [string]$MtpModelPath,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$SpecDraftMax = 3,

    [ValidateRange(0, [int]::MaxValue)]
    [int]$SpecDraftMin = 0,

    [switch]$ContentLogging,

    [ValidateRange(1, 4096)]
    [int]$EventBufferMiB = 64,

    [switch]$Background,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArguments
)

$ErrorActionPreference = "Stop"
if (-not $ServerPath) {
    $architectureName = $Architecture.ToLowerInvariant()
    $serverCandidates = @(
        (Join-Path $PSScriptRoot "build-telemetry-$architectureName\bin\$Configuration\llama-server.exe"),
        (Join-Path $PSScriptRoot "build-telemetry-$architectureName\bin\llama-server.exe"),
        (Join-Path $PSScriptRoot "build-telemetry-$architectureName-cuda-clang-v2\bin\llama-server.exe")
    )
    $ServerPath = $serverCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $ServerPath) {
        throw "No $Architecture llama-server telemetry build was found. Run build-telemetry.ps1 -Architecture $Architecture first, or pass -ServerPath."
    }
}
$server = (Resolve-Path -LiteralPath $ServerPath).Path
$model = (Resolve-Path -LiteralPath $ModelPath).Path

if ($SpecModelPath -and $MtpModelPath) {
    throw "Use either -SpecModelPath or -MtpModelPath, not both."
}
if ($MtpModelPath) {
    $SpecType = "draft-mtp"
    $SpecModelPath = $MtpModelPath
}

$arguments = @(
    "--model", $model,
    "--host", $HostAddress,
    "--port", $Port,
    "--ctx-size", $ContextLength,
    "--n-gpu-layers", $GpuLayers,
    "--parallel", $ParallelSlots,
    "--batch-size", $BatchSize,
    "--ubatch-size", $UBatchSize,
    "--metrics"
)

if ($SpecType -and $SpecType -ne "none") {
    $arguments += @("--spec-type", $SpecType)
    if ($SpecType.StartsWith("draft-")) {
        $arguments += @("--spec-draft-n-max", $SpecDraftMax)
        if ($SpecDraftMin -gt 0) {
            $arguments += @("--spec-draft-n-min", $SpecDraftMin)
        }
    }
}
if ($SpecModelPath) {
    $arguments += @("--spec-draft-model", (Resolve-Path -LiteralPath $SpecModelPath).Path)
}
if ($AdditionalArguments) {
    $arguments += $AdditionalArguments
}

$previousContent = $env:LLAMA_TELEMETRY_CONTENT
$previousBuffer = $env:LLAMA_TELEMETRY_EVENT_BUFFER_MIB
$env:LLAMA_TELEMETRY_CONTENT = if ($ContentLogging) { "1" } else { "0" }
$env:LLAMA_TELEMETRY_EVENT_BUFFER_MIB = $EventBufferMiB.ToString([Globalization.CultureInfo]::InvariantCulture)

if (-not $Background) {
    Write-Host "Starting llama-server telemetry on http://${HostAddress}:$Port"
    try {
        & $server @arguments
        $serverExitCode = $LASTEXITCODE
    } finally {
        $env:LLAMA_TELEMETRY_CONTENT = $previousContent
        $env:LLAMA_TELEMETRY_EVENT_BUFFER_MIB = $previousBuffer
    }
    exit $serverExitCode
}

function ConvertTo-ProcessArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + $Value.Replace('"', '\"') + '"'
}

$processArguments = ($arguments | ForEach-Object { ConvertTo-ProcessArgument ([string]$_) }) -join " "
try {
    $process = Start-Process -FilePath $server -ArgumentList $processArguments -WorkingDirectory (Split-Path $server) -WindowStyle Hidden -PassThru
} finally {
    $env:LLAMA_TELEMETRY_CONTENT = $previousContent
    $env:LLAMA_TELEMETRY_EVENT_BUFFER_MIB = $previousBuffer
}
$ready = $false
for ($attempt = 0; $attempt -lt 100; $attempt++) {
    if ($process.HasExited) {
        throw "llama-server exited during startup with code $($process.ExitCode)."
    }
    try {
        $health = Invoke-RestMethod -Uri "http://${HostAddress}:$Port/health" -TimeoutSec 1
        if ($health.status -eq "ok") {
            $ready = $true
            break
        }
    } catch {
    }
    Start-Sleep -Milliseconds 100
}
if (-not $ready) {
    if (-not $process.HasExited) {
        $process.Kill()
    }
    throw "llama-server did not become healthy at http://${HostAddress}:$Port within 10 seconds. Process ID: $($process.Id)"
}

Write-Host "llama-server is ready. Process ID: $($process.Id)"
Write-Host "Telemetry: http://${HostAddress}:$Port/telemetry/v1/capabilities"
$process
