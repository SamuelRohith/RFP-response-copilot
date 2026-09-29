@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo The virtual environment was not found.
  echo Follow the one-time setup steps in README.md first.
  pause
  exit /b 1
)

"%PYTHON_EXE%" -m streamlit run "%PROJECT_DIR%app.py"
pause
