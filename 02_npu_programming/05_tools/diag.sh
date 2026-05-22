#!/bin/bash
# phase2/diag.sh — 收集昇腾硬件与软件环境诊断信息
# 运行后输出一份完整的硬件规格和软件版本报告
# 用法: bash diag.sh [npu_id]

# 检测必要工具是否存在
if ! command -v npu-smi &>/dev/null; then
    echo "错误: npu-smi 未找到，请确认驱动已正确安装" >&2
    exit 3
fi

# 检查 NPU 数量并校验传入的 NPU_ID
NPU_ID="${1:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"
# 处理多设备环境变量 (如 "0,1,2")，取第一个值
NPU_ID="${NPU_ID%%,*}"
# 验证是否为非负整数
if ! [[ "$NPU_ID" =~ ^[0-9]+$ ]]; then
    echo "错误: NPU_ID 必须为非负整数，当前值: $NPU_ID" >&2
    exit 2
fi
TOTAL_NPU=$(npu-smi info -l 2>/dev/null | grep -c "NPU\|Chip" || echo 0)
if (( TOTAL_NPU > 0 && NPU_ID >= TOTAL_NPU )); then
    echo "错误: NPU $NPU_ID 不存在，共检测到 $TOTAL_NPU 张卡" >&2
    exit 2
fi

# 清理 3 天前的旧诊断目录
find /tmp -maxdepth 1 -type d -name "npu-diag-*" -mtime +3 -exec rm -rf -- {} + 2>/dev/null || true

OUTPUT_DIR="/tmp/npu-diag-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR" || { echo "错误: 无法创建输出目录 $OUTPUT_DIR" >&2; exit 1; }

echo "=== 昇腾环境诊断 (NPU $NPU_ID) ==="
echo "输出目录: $OUTPUT_DIR"
echo ""

# ── 软件版本 ──
echo "[1/8] 操作系统版本..."
cat /etc/os-release | head -3 > "$OUTPUT_DIR/os.txt"

echo "[2/8] CANN 与驱动版本..."
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
else
    echo "警告: set_env.sh 未找到，CANN 环境可能未正确设置" >&2
fi
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg > "$OUTPUT_DIR/cann_version.txt" 2>/dev/null
cat /usr/local/Ascend/driver/version.info > "$OUTPUT_DIR/driver_version.txt" 2>/dev/null
cat /usr/local/Ascend/firmware/version.info > "$OUTPUT_DIR/firmware_version.txt" 2>/dev/null

echo "[3/8] 版本兼容性检查..."
TOOLBOX_PATH="${ASCEND_TOOLBOX_PATH:-/usr/local/Ascend/toolbox/6.0.0}"
ASCEND_DMI="${TOOLBOX_PATH}/Ascend-DMI/bin/ascend-dmi"
export LD_LIBRARY_PATH=${TOOLBOX_PATH}/Ascend-DMI/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
$ASCEND_DMI --compatible > "$OUTPUT_DIR/compatible.txt" 2>&1

# ── 硬件信息 ──
echo "[4/8] 卡间拓扑..."
npu-smi info -l > "$OUTPUT_DIR/topology.txt"

echo "[5/8] NPU $NPU_ID 硬件规格 (ascend-dmi)..."
$ASCEND_DMI --info --detail --fmt json > "$OUTPUT_DIR/hardware_detail_npu${NPU_ID}.json" 2>&1

echo "[6/8] NPU $NPU_ID 物理信息 (board)..."
npu-smi info -t board -i "$NPU_ID" > "$OUTPUT_DIR/board_npu${NPU_ID}.txt"

echo "[7/8] NPU $NPU_ID ECC 与 PCIe 错误..."
npu-smi info -t ecc -i "$NPU_ID" > "$OUTPUT_DIR/ecc_npu${NPU_ID}.txt"
npu-smi info -t pcie-err -i "$NPU_ID" > "$OUTPUT_DIR/pcie_err_npu${NPU_ID}.txt"

# ── 性能基线 ──
echo "[8/8] NPU $NPU_ID HBM 带宽测试 (d2d)..."
$ASCEND_DMI --bw -t d2d -d "$NPU_ID" -q > "$OUTPUT_DIR/bw_d2d_npu${NPU_ID}.txt" 2>&1

echo ""
echo "=== 诊断完成 ==="
echo "完整输出: $OUTPUT_DIR"
echo ""
echo "关键信息速览:"
echo "---"
grep -E "Version|version" "$OUTPUT_DIR/cann_version.txt" 2>/dev/null | head -3
grep -E "Health|health" "$OUTPUT_DIR/hardware_detail_npu${NPU_ID}.json" 2>/dev/null | head -1 || \
  echo "  (JSON 输出，请直接查看 $OUTPUT_DIR/hardware_detail_npu${NPU_ID}.json)"
echo "  HBM 带宽: 见 $OUTPUT_DIR/bw_d2d_npu${NPU_ID}.txt"
