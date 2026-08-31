[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Assert-Contains([string]$Text, [string]$Expected) {
    if ($Text.IndexOf($Expected, [StringComparison]::Ordinal) -lt 0) {
        throw "Expected launcher source to contain '$Expected'."
    }
}

function Assert-DoesNotContain([string]$Text, [string]$Unexpected) {
    if ($Text.IndexOf($Unexpected, [StringComparison]::Ordinal) -ge 0) {
        throw "Launcher source must not contain '$Unexpected'."
    }
}

function Assert-Matches([string]$Text, [string]$Pattern) {
    if ($Text -notmatch $Pattern) {
        throw "Expected launcher source to match '$Pattern'."
    }
}

function Get-ThrownMessage([scriptblock]$Action) {
    try {
        & $Action
    } catch {
        return $_.Exception.Message
    }

    throw "Expected the launcher to fail before starting a server."
}

function Set-OwnerProtectedTestAcl([string]$Path) {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($null -eq $currentUser) {
        throw "The current Windows user SID is unavailable."
    }

    $acl = Get-Acl -LiteralPath $Path
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($rule in @($acl.Access)) {
        [void]$acl.RemoveAccessRuleAll($rule)
    }
    $acl.SetOwner($currentUser)

    foreach ($identity in @(
        $currentUser,
        [Security.Principal.SecurityIdentifier]::new([Security.Principal.WellKnownSidType]::LocalSystemSid, $null),
        [Security.Principal.SecurityIdentifier]::new([Security.Principal.WellKnownSidType]::BuiltinAdministratorsSid, $null))) {
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
            $identity,
            [Security.AccessControl.FileSystemRights]::FullControl,
            [Security.AccessControl.InheritanceFlags]::None,
            [Security.AccessControl.PropagationFlags]::None,
            [Security.AccessControl.AccessControlType]::Allow))
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $repositoryRoot "run-llama-telemetry.ps1"
$documentation = Join-Path $repositoryRoot "tools\server\README-telemetry.md"

$tokens = $null
$parseErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile($launcher, [ref]$tokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count -ne 0) {
    $parseErrors | ForEach-Object { Write-Error $_.Message }
    throw "run-llama-telemetry.ps1 has PowerShell parse errors."
}

$source = [IO.File]::ReadAllText($launcher)
Assert-Matches $source '(?s)\[Parameter\(Mandatory = \$true\)\]\s*\[string\]\$ApiKeyFile'
Assert-Contains $source "function Resolve-ProtectedApiKeyFile"
Assert-Contains $source "IsPathRooted"
Assert-Contains $source "GetAccessRules"
Assert-Contains $source '"--props",'
Assert-Contains $source '"--api-key-file", $apiKeyFile'
Assert-Contains $source '"LLAMA_API_KEY" = $null'
Assert-Contains $source '"127.0.0.1", "::1", "localhost"'
Assert-Contains $source "function Assert-SafeAdditionalArguments"
Assert-Contains $source "api-key-file|props"
Assert-Contains $source "-ContentLogging is deprecated"
Assert-DoesNotContain $source "LLAMA_TELEMETRY_CONTENT = if"

$documentationSource = [IO.File]::ReadAllText($documentation)
Assert-Contains $documentationSource "-ApiKeyFile C:\secure\llama-server.key"
Assert-Contains $documentationSource 'authenticated `POST /props`'

$keyFile = Join-Path ([IO.Path]::GetTempPath()) "llama-telemetry-launcher-$([Guid]::NewGuid().ToString('N')).key"
try {
    New-Item -ItemType File -Path $keyFile -ErrorAction Stop | Out-Null
    Set-OwnerProtectedTestAcl $keyFile

    $invalidHost = Get-ThrownMessage {
        & $launcher -ModelPath "C:\model-not-opened.gguf" -ApiKeyFile $keyFile -HostAddress "0.0.0.0"
    }
    if ($invalidHost -notlike "Telemetry control requires -HostAddress*") {
        throw "The launcher did not fail closed for a non-loopback host."
    }

    $unsafeOverride = Get-ThrownMessage {
        & $launcher -ModelPath "C:\model-not-opened.gguf" -ApiKeyFile $keyFile -AdditionalArguments "--api-key"
    }
    if ($unsafeOverride -notlike "-AdditionalArguments cannot override*") {
        throw "The launcher did not reject an API-key command-line override."
    }

    $validatedKey = Get-ThrownMessage {
        & $launcher -ModelPath "C:\model-not-opened.gguf" -ApiKeyFile $keyFile -ServerPath "C:\server-not-opened.exe"
    }
    if ($validatedKey -notlike "Cannot find path*") {
        throw "The launcher did not reach executable validation after protected key-file validation."
    }
} finally {
    Remove-Item -LiteralPath $keyFile -Force -ErrorAction SilentlyContinue
}

Write-Output "run-llama-telemetry static contract: passed"
