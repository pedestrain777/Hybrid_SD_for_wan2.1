#!/bin/bash
# 停止所有VBench 2.0评测进程

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}停止所有VBench 2.0评测进程...${NC}"
echo ""

for gpu in 4 5 6 7; do
    PID_FILE="vbench2_gpu${gpu}.pid"
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat ${PID_FILE})
        if ps -p ${PID} > /dev/null 2>&1; then
            echo -e "${YELLOW}停止 GPU ${gpu} (PID: ${PID})...${NC}"
            kill ${PID}
            sleep 1
            if ps -p ${PID} > /dev/null 2>&1; then
                echo -e "${RED}进程未响应，强制停止...${NC}"
                kill -9 ${PID}
            fi
            echo -e "${GREEN}✓ GPU ${gpu} 已停止${NC}"
        else
            echo -e "${YELLOW}GPU ${gpu} 进程未运行 (PID: ${PID})${NC}"
        fi
    else
        echo -e "${YELLOW}GPU ${gpu} 未找到PID文件${NC}"
    fi
done

echo ""
echo -e "${GREEN}所有进程已停止${NC}"
