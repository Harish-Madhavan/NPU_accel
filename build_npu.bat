@echo off
setlocal

:: Try to find vcvars64.bat automatically using vswhere
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
  set "VS_PATH=%%i"
)

if defined VS_PATH (
  set "VCVARS_PATH=%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
) else (
  echo Could not find Visual Studio installation via vswhere.
  echo Falling back to default path...
  set "VCVARS_PATH=E:\IDE\Visual studio IDE\vs26\VC\Auxiliary\Build\vcvars64.bat"
)

if exist "%VCVARS_PATH%" (
  call "%VCVARS_PATH%"
) else (
  echo vcvars64.bat not found at %VCVARS_PATH%. Please check your Visual Studio installation.
  exit /b 1
)

SET DISTUTILS_USE_SDK=1
cd /d "%~dp0intel_npu_lib"
echo Building and installing intel_npu_lib in development mode...
pip install -e .[dev]
