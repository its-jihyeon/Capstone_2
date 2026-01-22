@echo off
title HEAT Project - FastAPI Server
cd /d %~dp0

echo [1/3] 
call .venv311\Scripts\activate

echo [2/3] 
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

echo [3/3] 
pause
