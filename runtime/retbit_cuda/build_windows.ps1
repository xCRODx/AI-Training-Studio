$ErrorActionPreference = 'Stop'

$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe'
$hostCompiler = Get-ChildItem -LiteralPath 'C:\Program Files (x86)\Microsoft Visual Studio' `
    -Filter cl.exe -Recurse -ErrorAction Stop |
    Where-Object { $_.FullName -match 'Hostx64\\x64' } |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty FullName

foreach ($tool in @($nvcc, $hostCompiler)) {
    if ([string]::IsNullOrWhiteSpace($tool) -or -not (Test-Path -LiteralPath $tool -PathType Leaf)) {
        throw "Ferramenta CUDA não encontrada: $tool"
    }
}

$source = Join-Path $PSScriptRoot 'retbit_cuda.cu'
$artifactRoot = Join-Path (Split-Path (Split-Path $PSScriptRoot)) 'artifacts\retbit-cuda'
$buildRoot = Join-Path $PSScriptRoot 'build-windows'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
$output = Join-Path $buildRoot 'retbit_cuda_sm120.dll'

& $nvcc $source -o $output --shared -O3 --use_fast_math -arch=sm_120 `
    -ccbin (Split-Path $hostCompiler) -allow-unsupported-compiler
if ($LASTEXITCODE -ne 0) { throw "Build do kernel RetBit CUDA falhou ($LASTEXITCODE)." }
$runtimeHash = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.Substring(0, 12).ToLowerInvariant()
$deployed = Join-Path $artifactRoot "retbit_cuda_sm120-$runtimeHash.dll"
if (-not (Test-Path -LiteralPath $deployed -PathType Leaf)) {
    Copy-Item -LiteralPath $output -Destination $deployed -ErrorAction Stop
}
Write-Output "Kernel RetBit CUDA publicado: $deployed"
