$ErrorActionPreference = 'Stop'

$cmake = 'C:\Program Files\CMake\bin\cmake.exe'
$clang = 'C:\Program Files\LLVM\bin\clang++.exe'
$nmake = Get-ChildItem -LiteralPath 'C:\Program Files (x86)\Microsoft Visual Studio' `
    -Filter nmake.exe -Recurse -ErrorAction Stop |
    Where-Object { $_.FullName -match 'Hostx64\\x64' } |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$resourceCompiler = Get-ChildItem -LiteralPath 'C:\Program Files (x86)\Windows Kits\10\bin' `
    -Filter rc.exe -Recurse -ErrorAction Stop |
    Where-Object { $_.FullName -match '\\x64\\rc\.exe$' } |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty FullName

foreach ($tool in @($cmake, $clang, $nmake, $resourceCompiler)) {
    if ([string]::IsNullOrWhiteSpace($tool) -or -not (Test-Path -LiteralPath $tool -PathType Leaf)) {
        throw "Ferramenta de build não encontrada: $tool"
    }
}

$source = $PSScriptRoot
$build = Join-Path $source 'build-windows'
$configureArguments = @(
    '-S', $source,
    '-B', $build,
    '-G', 'NMake Makefiles',
    "-DCMAKE_MAKE_PROGRAM=$nmake",
    "-DCMAKE_CXX_COMPILER=$clang",
    "-DCMAKE_RC_COMPILER=$resourceCompiler",
    '-DCMAKE_BUILD_TYPE=Release'
)

& $cmake @configureArguments
if ($LASTEXITCODE -ne 0) { throw "Configuração CMake falhou ($LASTEXITCODE)." }

& $cmake --build $build
if ($LASTEXITCODE -ne 0) { throw "Build RetBit falhou ($LASTEXITCODE)." }

& (Join-Path (Split-Path $cmake) 'ctest.exe') --test-dir $build --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Testes RetBit falharam ($LASTEXITCODE)." }

$builtRuntime = Join-Path $build 'retbit_cli_build.exe'
if (-not (Test-Path -LiteralPath $builtRuntime -PathType Leaf)) {
    throw "Executável RetBit não foi produzido: $builtRuntime"
}
$runtimeRoot = Join-Path (Split-Path (Split-Path $source)) 'artifacts\retbit-runtime'
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
$runtimeHash = (Get-FileHash -LiteralPath $builtRuntime -Algorithm SHA256).Hash.Substring(0, 12).ToLowerInvariant()
$deployedRuntime = Join-Path $runtimeRoot "retbit_cli-$runtimeHash.exe"
if (-not (Test-Path -LiteralPath $deployedRuntime -PathType Leaf)) {
    Copy-Item -LiteralPath $builtRuntime -Destination $deployedRuntime -ErrorAction Stop
}
Write-Output "Runtime publicado: $deployedRuntime"
