[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [Parameter(Mandatory = $true)]
    [string]$ApiKeyFile,

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

function Test-LoopbackHost([string]$Address) {
    if ([string]::IsNullOrWhiteSpace($Address)) {
        return $false
    }

    return $Address.Trim().ToLowerInvariant() -in @("127.0.0.1", "::1", "localhost")
}

function Resolve-ProtectedApiKeyFile([string]$Path) {
    $pathRoot = if ([string]::IsNullOrWhiteSpace($Path)) { $null } else { [System.IO.Path]::GetPathRoot($Path) }
    if ([string]::IsNullOrWhiteSpace($Path) -or
        -not [System.IO.Path]::IsPathRooted($Path) -or
        [string]::IsNullOrEmpty($pathRoot) -or
        $pathRoot -eq [System.IO.Path]::DirectorySeparatorChar.ToString()) {
        throw "-ApiKeyFile must name an absolute protected local file."
    }

    try {
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    } catch {
        throw "-ApiKeyFile must name an existing protected local file."
    }

    if ($item.PSProvider.Name -ne "FileSystem" -or
        $item.PSIsContainer -or
        $item.FullName.StartsWith("\\") -or
        (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "-ApiKeyFile must name an absolute protected local file."
    }

    try {
        $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().User
        if ($null -eq $currentUser) {
            throw "The current Windows user SID is unavailable."
        }

        $system = [Security.Principal.SecurityIdentifier]::new(
            [Security.Principal.WellKnownSidType]::LocalSystemSid,
            $null)
        $administrators = [Security.Principal.SecurityIdentifier]::new(
            [Security.Principal.WellKnownSidType]::BuiltinAdministratorsSid,
            $null)
        $allowed = @($currentUser.Value, $system.Value, $administrators.Value)

        $acl = Get-Acl -LiteralPath $item.FullName -ErrorAction Stop
        if (-not $acl.AreAccessRulesProtected) {
            throw "The API-key file ACL inherits access rules."
        }

        $owner = $acl.GetOwner([Security.Principal.SecurityIdentifier])
        if ($owner -isnot [Security.Principal.SecurityIdentifier] -or -not $owner.Equals($currentUser)) {
            throw "The API-key file is not owned by the current user."
        }

        $rules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
        if ($rules.Count -ne $allowed.Count) {
            throw "The API-key file ACL has an unexpected access rule."
        }

        $present = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($rule in $rules) {
            $sid = $rule.IdentityReference
            if ($rule.IsInherited -or
                $sid -isnot [Security.Principal.SecurityIdentifier] -or
                $sid.Value -notin $allowed -or
                $rule.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow -or
                $rule.InheritanceFlags -ne [Security.AccessControl.InheritanceFlags]::None -or
                $rule.PropagationFlags -ne [Security.AccessControl.PropagationFlags]::None -or
                (($rule.FileSystemRights -band [Security.AccessControl.FileSystemRights]::FullControl) -ne [Security.AccessControl.FileSystemRights]::FullControl)) {
                throw "The API-key file ACL has an unexpected access rule."
            }
            [void]$present.Add($sid.Value)
        }

        foreach ($identity in $allowed) {
            if (-not $present.Contains($identity)) {
                throw "The API-key file ACL is missing a required servicing principal."
            }
        }
    } catch {
        if ($_.Exception.Message -like "The API-key file*") {
            throw
        }
        throw "-ApiKeyFile could not be verified as a protected local file."
    }

    return $item.FullName
}

function Assert-SafeAdditionalArguments([string[]]$Arguments) {
    foreach ($argument in @($Arguments)) {
        if ($argument -match "^(--(?:host|api-key|api-key-file|props))(?:=|$)") {
            throw "-AdditionalArguments cannot override --host, --api-key-file, or --props."
        }
    }
}

function Get-LoopbackServerUri([string]$Address, [int]$ServerPort) {
    if ($Address -eq "::1") {
        return "http://[::1]:$ServerPort"
    }

    return "http://${Address}:$ServerPort"
}

if (-not (Test-LoopbackHost $HostAddress)) {
    throw "Telemetry control requires -HostAddress 127.0.0.1, ::1, or localhost."
}
if ($ContentLogging) {
    throw "-ContentLogging is deprecated. Enable request_content with authenticated POST /props from LlamaScope."
}

$HostAddress = $HostAddress.Trim()
$apiKeyFile = Resolve-ProtectedApiKeyFile $ApiKeyFile
Assert-SafeAdditionalArguments $AdditionalArguments

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
    "--metrics",
    "--props",
    "--api-key-file", $apiKeyFile
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

$environmentOverrides = @{
    "LLAMA_API_KEY" = $null
    "LLAMA_ARG_API_KEY_FILE" = $null
    "LLAMA_TELEMETRY_CONTENT" = $null
    "LLAMA_TELEMETRY_OUTPUT_TOKENS" = $null
    "LLAMA_TELEMETRY_TOKEN_CANDIDATES" = $null
    "LLAMA_TELEMETRY_MOE_ROUTING" = $null
    "LLAMA_TELEMETRY_PROMPT_PERPLEXITY" = $null
    "LLAMA_TELEMETRY_KV_PRESSURE_DETAIL" = $null
    "LLAMA_TELEMETRY_GPU_GPM" = $null
    "LLAMA_TELEMETRY_EVENT_BUFFER_MIB" = $EventBufferMiB.ToString([Globalization.CultureInfo]::InvariantCulture)
}
$previousEnvironment = @{}
foreach ($name in $environmentOverrides.Keys) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable($name, $environmentOverrides[$name], [EnvironmentVariableTarget]::Process)
}

function Restore-LauncherEnvironment {
    foreach ($name in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], [EnvironmentVariableTarget]::Process)
    }
}

$serverUri = Get-LoopbackServerUri $HostAddress $Port

if (-not $Background) {
    Write-Host "Starting llama-server telemetry on $serverUri"
    try {
        & $server @arguments
        $serverExitCode = $LASTEXITCODE
    } finally {
        Restore-LauncherEnvironment
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
    Restore-LauncherEnvironment
}
$ready = $false
for ($attempt = 0; $attempt -lt 100; $attempt++) {
    if ($process.HasExited) {
        throw "llama-server exited during startup with code $($process.ExitCode)."
    }
    try {
        $health = Invoke-RestMethod -Uri "$serverUri/health" -TimeoutSec 1
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
    throw "llama-server did not become healthy at $serverUri within 10 seconds. Process ID: $($process.Id)"
}

Write-Host "llama-server is ready. Process ID: $($process.Id)"
Write-Host "Telemetry: $serverUri/telemetry/v1/capabilities"
$process
