#!/bin/bash
# VBench 2.0 评测监控脚本

OUTPUT_DIR="results/vbench/vbench2_half"
LOG_FILE="vbench2_half_generation.log"
TOTAL_PROMPTS=611

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VBench 2.0 评测进度监控${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查进程是否运行
if [ -f "vbench2_half.pid" ]; then
    PID=$(cat vbench2_half.pid)
    if ps -p ${PID} > /dev/null 2>&1; then
        ELAPSED=$(ps -p ${PID} -o etime= | tr -d ' ')
        CPU=$(ps -p ${PID} -o %cpu= | tr -d ' ')
        MEM=$(ps -p ${PID} -o %mem= | tr -d ' ')
        echo -e "${GREEN}✓ 进程运行中${NC}"
        echo -e "  PID: ${PID}"
        echo -e "  运行时间: ${ELAPSED}"
        echo -e "  CPU: ${CPU}%"
        echo -e "  内存: ${MEM}%"
    else
        echo -e "${YELLOW}✗ 进程未运行 (PID: ${PID})${NC}"
    fi
else
    echo -e "${YELLOW}✗ 未找到PID文件${NC}"
fi
echo ""

# 统计已生成视频数
if [ -d "${OUTPUT_DIR}/videos" ]; then
    GENERATED=$(find ${OUTPUT_DIR}/videos -name "*.mp4" 2>/dev/null | wc -l)
    PROGRESS=$(echo "scale=2; ${GENERATED} * 100 / ${TOTAL_PROMPTS}" | bc)
    REMAINING=$((TOTAL_PROMPTS - GENERATED))

    echo -e "${BLUE}生成进度:${NC}"
    echo -e "  已完成: ${GENERATED} / ${TOTAL_PROMPTS} (${PROGRESS}%)"
    echo -e "  剩余: ${REMAINING}"

    if [ ${GENERATED} -gt 0 ]; then
        # 估算剩余时间
        AVG_TIME=26.8  # 分钟/视频
        REMAINING_TIME=$(echo "scale=1; ${REMAINING} * ${AVG_TIME} / 60" | bc)
        REMAINING_DAYS=$(echo "scale=1; ${REMAINING_TIME} / 24" | bc)
        echo -e "  预计剩余时间: ${REMAINING_TIME}小时 (${REMAINING_DAYS}天)"
    fi
else
    echo -e "${YELLOW}输出目录不存在${NC}"
fi
echo ""

# GPU状态
echo -e "${BLUE}GPU状态:${NC}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -1
echo ""

# 最新日志
if [ -f "${LOG_FILE}" ]; then
    echo -e "${BLUE}最新日志 (最后10行):${NC}"
    tail -10 ${LOG_FILE}
else
    echo -e "${YELLOW}日志文件不存在${NC}"
fi
