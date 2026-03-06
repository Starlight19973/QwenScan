#!/bin/bash
cd /home/testusr4/QwenScan
source venv/bin/activate
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8081
