@echo off
setlocal EnableDelayedExpansion

:: =====================================================================
:: build_npu.bat — configure MSVC and build the Intel NPU C++ extension.
:: Usage: build_npu.bat [--no-install] [--clean]
::   --no-install : only `build_ext --inplace`, skip `pip install -e .`
::   --clean      : remove build/ and *.pyd artifacts before rebuilding
:: Env knobs (forwarded to setup.py):
::   NPU_NO_BUILD_EXT=1  : skip the C++ extension (pure-Python install)
::   NPU_VERBOSE_BUILD=1 : verbose OpenVINO discovery logging
:: =====================================================================

set "NO_INSTALL=0"
set "CLEAN=0"
for %%a in (%*) do (
  if "%%a"=="--no-install" set "NO_INSTALL=1"
  if "%%a"=="--clean" set "CLEAN=1"
)

:: --- 0. Sanity checks ------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 'python' not found on PATH. Install Python 3.10+ and retry.
  exit /b 1
)
python -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.10+ is required. Your version:
  python --version
  exit /b 1
)

:: --- 1. Locate Visual Studio -----------------------------------------
set "VS_PATH="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
  )
)

if defined VS_PATH (
  set "VCVARS_PATH=!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat"
) else (
  :: Common default locations for VS 2022 / Build Tools (no personal paths).
  for %%p in (
    "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  ) do (
    if exist %%p (
      set "VCVARS_PATH=%%~p"
      goto :found_vcvars
    )
  )
)
:found_vcvars

if not defined VCVARS_PATH (
  echo [ERROR] No Visual Studio C++ toolchain found.
  echo         Install "Desktop development with C++" from:
  echo           https://visualstudio.microsoft.com/downloads/
  echo         Or install Build Tools only:
  echo           winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools"
  exit /b 1
)

if not exist "%VCVARS_PATH%" (
  echo [ERROR] vcvars64.bat not found at %VCVARS_PATH%.
  echo         Re-run the Visual Studio Installer and add the C++ workload.
  exit /b 1
)

call "%VCVARS_PATH%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] vcvars64.bat failed. Try running this script from a "Developer Command Prompt".
  exit /b 1
)
where cl >nul 2>nul
if errorlevel 1 (
  echo [ERROR] cl.exe still not on PATH after vcvars64.bat. Check your VS installation.
  exit /b 1
)

:: --- 2. Build ---------------------------------------------------------
SET DISTUTILS_USE_SDK=1
cd /d "%~dp0intel_npu_lib"

if "%CLEAN%"=="1" (
  echo Cleaning previous build artifacts...
  if exist build rmdir /s /q build
  del /q /s *.pyd >nul 2>&1
)

echo Building C++ extension inplace...
python setup.py build_ext --inplace > build_log.txt 2>&1
if errorlevel 1 (
  echo [ERROR] C++ extension compilation failed.
  echo         See 'intel_npu_lib\build_log.txt' for details.
  echo         Tip: set NPU_VERBOSE_BUILD=1 and re-run for OpenVINO discovery logs.
  exit /b 1
)
echo [SUCCESS] C++ extension built inplace.

if "%NO_INSTALL%"=="1" (
  echo Skipping pip install (--no-install).
  exit /b 0
)

echo Installing package in editable mode...
python -m pip install --no-build-isolation --no-deps -e . >> build_log.txt 2>&1
if errorlevel 1 (
  echo [ERROR] pip editable install failed. See build_log.txt.
  exit /b 1
)

echo [SUCCESS] intel_npu_acceleration installed. Verify with:
echo   python -m intel_npu_acceleration --info
