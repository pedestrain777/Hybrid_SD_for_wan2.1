#!/bin/bash
# VBench 2.0 多GPU监控脚本

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VBench 2.0 多GPU评测进度监控${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 总进度统计
TOTAL_PROMPTS=611
TOTAL_GENERATED=0

# 检查每个GPU的进程和进度
for gpu in 4 5 6 7; do
    echo -e "${BLUE}=== GPU ${gpu} ===${NC}"

    # 检查进程
    PID_FILE="vbench2_gpu${gpu}.pid"
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat ${PID_FILE})
        if ps -p ${PID} > /dev/null 2>&1; then
            ELAPSED=$(ps -p ${PID} -o etime= | tr -d ' ')
            echo -e "${GREEN}✓ 进程运行中${NC} (PID: ${PID}, 运行时间: ${ELAPSED})"
        else
            echo -e "${RED}✗ 进程未运行${NC} (PID: ${PID})"
        fi
    else
        echo -e "${YELLOW}✗ 未找到PID文件${NC}"
    fi

    # 统计已生成视频
    OUTPUT_DIR="results/vbench/vbench2_half_gpu${gpu}"
    if [ -d "${OUTPUT_DIR}/videos" ]; then
        GENERATED=$(find ${OUTPUT_DIR}/videos -name "*.mp4" 2>/dev/null | wc -l)
        TOTAL_GENERATED=$((TOTAL_GENERATED + GENERATED))

        # 读取该GPU的总prompts数
        PROMPT_FILE="VBench-master/VBench-2.0/prompts/VBench2_half_gpu${gpu}.txt"
        if [ -f "${PROMPT_FILE}" ]; then
            GPU_TOTAL=$(wc -l < ${PROMPT_FILE})
            PROGRESS=$(echo "scale=1; ${GENERATED} * 100 / ${GPU_TOTAL}" | bc)
            echo -e "进度: ${GENERATED} / ${GPU_TOTAL} (${PROGRESS}%)"
        else
            echo -e "进度: ${GENERATED} 个视频"
        fi
    else
        echo -e "${YELLOW}输出目录不存在${NC}"
    fi

    echo ""
done

# 总体进度
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}总体进度${NC}"
echo -e "${GREEN}========================================${NC}"
OVERALL_PROGRESS=$(echo "scale=1; ${TOTAL_GENERATED} * 100 / ${TOTAL_PROMPTS}" | bc)
REMAINING=$((TOTAL_PROMPTS - TOTAL_GENERATED))
echo -e "已完成: ${TOTAL_GENERATED} / ${TOTAL_PROMPTS} (${OVERALL_PROGRESS}%)"
echo -e "剩余: ${REMAINING}"

if [ ${TOTAL_GENERATED} -gt 0 ]; then
    # 估算剩余时间（4卡并行）
    AVG_TIME=26.8  # 分钟/视频
    REMAINING_TIME=$(echo "scale=1; ${REMAINING} * ${AVG_TIME} / 60 / 4" | bc)
    echo -e "预计剩余时间: ${REMAINING_TIME}小时"
fi
echo ""

# GPU状态
echo -e "${BLUE}GPU状态:${NC}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | grep -E "^[4567],"
echo ""

# 磁盘使用
echo -e "${BLUE}磁盘使用:${NC}"
du -sh results/vbench/vbench2_half_gpu* 2>/dev/null | awk '{print "  "$0}'
