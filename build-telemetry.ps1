[CmdletBinding()]
param(
    [ValidateSet("Auto", "On", "Off")]
    [string]$Cuda = "Auto",

    [ValidateSet("x64", "ARM64")]
    [string]$Architecture = "x64",

    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Configuration = "Release",

    [string]$BuildDirectory,

    [string]$CudaArchitecture,

    [string]$LlvmPath,

    [int]$Jobs = 2
)

$ErrorActionPreference = "Stop"
$sourceDirectory = $PSScriptRoot
if (-not $BuildDirectory) {
    $BuildDirectory = "build-telemetry-$($Architecture.ToLowerInvariant())"
}
$buildPath = if ([System.IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory
} else {
    Join-Path $sourceDirectory $BuildDirectory
}

$cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
if (-not $cmake) {
    $cmakeCandidates = @(
        "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        (Join-Path ${env:ProgramFiles} "CMake\bin\cmake.exe"),
        (Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"),
        (Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
    )
    $cmakePath = $cmakeCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
    if (-not $cmakePath) {
        throw "CMake was not found. Install CMake or the Visual Studio 2022 C++ CMake tools."
    }
} else {
    $cmakePath = $cmake.Source
}

$nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if (-not $nvcc -and $env:CUDA_PATH) {
    $candidate = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path -LiteralPath $candidate) {
        $nvcc = Get-Item -LiteralPath $candidate
    }
}
if (-not $nvcc) {
    $localCuda = Join-Path $env:USERPROFILE "Tools\CUDA\v13.4\bin\nvcc.exe"
    if (Test-Path -LiteralPath $localCuda) {
        $nvcc = Get-Item -LiteralPath $localCuda
    }
}

$cudaEnabled = switch ($Cuda) {
    "On" {
        if (-not $nvcc) {
            throw "CUDA was requested, but nvcc.exe was not found. Install the NVIDIA CUDA Toolkit or use -Cuda Off."
        }
        $true
    }
    "Off" { $false }
    default { [bool]$nvcc }
}
$nvccPath = if ($nvcc) {
    if ($nvcc.Source) { $nvcc.Source } else { $nvcc.FullName }
} else {
    $null
}

$commonConfigureArguments = @(
    "-DGGML_CUDA=$(if ($cudaEnabled) { 'ON' } else { 'OFF' })",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TESTS=ON",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_UI=OFF",
    "-DLLAMA_USE_PREBUILT_UI=OFF",
    "-DLLAMA_BUILD_TOOLS=ON"
)

$singleConfiguration = $false
if ($Architecture -eq "ARM64") {
    $llvmCandidates = @(
        $LlvmPath,
        $env:LLVM_PATH,
        (Join-Path $env:USERPROFILE "Tools\LLVM\22.1.8"),
        (Join-Path ${env:ProgramFiles} "LLVM")
    )
    $resolvedLlvm = $llvmCandidates |
        Where-Object { $_ -and (Test-Path -LiteralPath (Join-Path $_ "bin\clang-cl.exe")) } |
        Select-Object -First 1
    if (-not $resolvedLlvm) {
        throw "ARM64 llama.cpp builds require clang-cl. Install LLVM for Windows on Arm or pass -LlvmPath."
    }

    $ninjaCandidates = @(
        (Join-Path (Split-Path $cmakePath -Parent) "..\..\Ninja\ninja.exe"),
        "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
        (Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe")
    )
    $ninjaPath = $ninjaCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
    if (-not $ninjaPath) {
        throw "Ninja was not found. Install the Visual Studio C++ CMake tools."
    }

    $vcVarsCandidates = @(
        "C:\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        (Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
        (Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat")
    )
    $vcVars = $vcVarsCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $vcVars) {
        throw "Visual Studio 2022 ARM64 C++ build tools were not found."
    }
    & cmd.exe /d /s /c "`"$vcVars`" arm64 >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }

    $visualCppRoot = Split-Path (Split-Path (Split-Path $vcVars -Parent) -Parent) -Parent
    $arm64Cl = Get-ChildItem (Join-Path $visualCppRoot "Tools\MSVC") -Directory |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "bin\Hostarm64\arm64\cl.exe" } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if (-not $arm64Cl) {
        throw "The Visual Studio ARM64 cl.exe host compiler was not found."
    }

    $clangCl = Join-Path $resolvedLlvm "bin\clang-cl.exe"
    $env:PATH = "$(Split-Path $clangCl -Parent);$(Split-Path $ninjaPath -Parent);$env:PATH"
    if ($cudaEnabled) {
        $cudaRoot = Split-Path (Split-Path $nvccPath -Parent) -Parent
        $env:CUDA_PATH = $cudaRoot
        $env:PATH = "$(Split-Path $nvccPath -Parent);$env:PATH"
        if (-not $CudaArchitecture) {
            $CudaArchitecture = "121"
        }
    }

    $configureArguments = @(
        "-S", $sourceDirectory,
        "-B", $buildPath,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=$Configuration",
        "-DCMAKE_C_COMPILER=$clangCl",
        "-DCMAKE_CXX_COMPILER=$clangCl",
        "-DGGML_NATIVE=OFF"
    ) + $commonConfigureArguments
    if ($cudaEnabled) {
        $configureArguments += @(
            "-DCMAKE_CUDA_COMPILER=$nvccPath",
            "-DCMAKE_CUDA_HOST_COMPILER=$($arm64Cl.Replace('\', '/'))",
            "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitecture"
        )
    }
    $singleConfiguration = $true
} else {
    $configureArguments = @(
        "-S", $sourceDirectory,
        "-B", $buildPath,
        "-G", "Visual Studio 17 2022",
        "-A", $Architecture
    ) + $commonConfigureArguments
    if ($cudaEnabled -and $CudaArchitecture) {
        $configureArguments += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitecture"
    }
}

Write-Host "Configuring llama-server telemetry ($Architecture, $Configuration, CUDA=$cudaEnabled)"
& $cmakePath @configureArguments
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE."
}

Write-Host "Building llama-server"
$buildArguments = @("--build", $buildPath, "--target", "llama-server", "--parallel", $Jobs)
if (-not $singleConfiguration) {
    $buildArguments += @("--config", $Configuration)
}
& $cmakePath @buildArguments
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed with exit code $LASTEXITCODE."
}

$artifactCandidates = @(
    (Join-Path $buildPath "bin\$Configuration\llama-server.exe"),
    (Join-Path $buildPath "bin\llama-server.exe")
)
$artifact = $artifactCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $artifact) {
    throw "Build completed without llama-server.exe under $buildPath\bin."
}

$artifactDirectory = Split-Path $artifact -Parent
if ($Architecture -eq "ARM64" -and $resolvedLlvm) {
    $openMpRuntime = Get-ChildItem (Join-Path $visualCppRoot "Redist\MSVC") -Recurse -File -Filter "libomp140.aarch64.dll" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($openMpRuntime) {
        Copy-Item -LiteralPath $openMpRuntime.FullName -Destination $artifactDirectory -Force
    } else {
        Write-Warning "The ARM64 OpenMP runtime libomp140.aarch64.dll was not found under the Visual C++ redistributables."
    }
}
if ($cudaEnabled) {
    $cudaRoot = Split-Path (Split-Path $nvccPath -Parent) -Parent
    $runtimeArchitecture = $Architecture.ToLowerInvariant()
    $cudaRuntimeDirectories = @(
        (Join-Path $cudaRoot "bin\$runtimeArchitecture"),
        (Join-Path $cudaRoot "bin")
    ) | Where-Object { Test-Path -LiteralPath $_ }
    foreach ($runtimeName in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
        $runtime = $cudaRuntimeDirectories |
            ForEach-Object { Get-ChildItem -LiteralPath $_ -Filter $runtimeName -File -ErrorAction SilentlyContinue } |
            Select-Object -First 1
        if ($runtime) {
            Copy-Item -LiteralPath $runtime.FullName -Destination $artifactDirectory -Force
        }
    }
}

Write-Host "Built: $artifact"
Get-Item -LiteralPath $artifact
