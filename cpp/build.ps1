# Build the C++ engine into src/prs/_engine*.pyd.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File cpp\build.ps1
# Needs: MSVC Build Tools (C++ workload) and the `prs` conda env (cmake, ninja, pybind11, eigen).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "MSVC Build Tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

# find conda even when it is not on PATH (plain PowerShell)
$conda = $env:CONDA_EXE
if (-not $conda) { $conda = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe" }
$env_prefix = (& $conda run -n prs python -c "import sys; print(sys.prefix)").Trim()

# cmd /c only runs the first line of a multi-line string, so build one line
$setup = "call `"$vcvars`" >nul 2>nul"
$configure = "cmake -S `"$root\cpp`" -B `"$root\cpp\build`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
             "-DPython_EXECUTABLE=`"$env_prefix\python.exe`" -DCMAKE_PREFIX_PATH=`"$env_prefix\Library`""
$build = "cmake --build `"$root\cpp\build`" --parallel"
$cmd = "$setup && $configure && $build"
$env:PATH = "$env_prefix;$env_prefix\Library\bin;$env_prefix\Scripts;" +
            "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer;$env:PATH"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Host "Built:" (Get-ChildItem "$root\src\prs\_engine*.pyd").Name
