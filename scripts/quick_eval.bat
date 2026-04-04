@echo off
setlocal
set SNAPSHOT=%1
if "%SNAPSHOT%"=="" set SNAPSHOT=data/case4_dataset_snapshot.jsonl
powershell -ExecutionPolicy Bypass -File "%~dp0quick_eval.ps1" -SnapshotPath "%SNAPSHOT%"
endlocal

