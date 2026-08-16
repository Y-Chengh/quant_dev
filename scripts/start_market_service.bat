@echo off
rem 启动本地行情服务。需先在项目根目录执行 python -m pip install -e ".[service]"
cd /d "%~dp0.."
python -m quant.cli.market_server %*
