@echo off
setlocal
rem faultbench launcher: finds the sibling SADE venv's Python, sets PYTHONUTF8,
rem and forwards all arguments to `python -m faultbench`.
set PYTHONUTF8=1
set "PY=%~dp0..\SADE-NetworkAgent\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo SADE venv not found at %PY% -- run the Setup steps in README.md first.
  exit /b 1
)
"%PY%" -m faultbench %*
exit /b %ERRORLEVEL%
