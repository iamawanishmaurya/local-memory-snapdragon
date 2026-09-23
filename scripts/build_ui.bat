@echo off
rem Build the shadcn-admin based dashboard into ui/dist, served by the FastAPI app.
set "PATH=%~dp0..\tools\node-v22.14.0-win-x64;%PATH%"
cd /d "%~dp0..\ui"
if not exist node_modules (
  echo Installing UI dependencies...
  call npm install --no-fund --no-audit
)
call npm run build
echo.
echo Done. Start Local Memory with: python -m local_memory.main
