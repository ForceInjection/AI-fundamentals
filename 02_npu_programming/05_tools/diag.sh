#!/bin/bash
# phase2/diag.sh — 收集昇腾硬件与软件环境诊断信息
# 运行后输出一份完整的硬件规格和软件版本报告
# 用法: bash diag.sh [npu_id]

NPU_ID="${1:-7}"
OUTPUT_DIR="/tmp/npu-diag-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=== 昇腾环境诊断 (NPU $NPU_ID) ==="
echo "输出目录: $OUTPUT_DIR"
echo ""

# ── 软件版本 ──
echo "[1/8] 操作系统版本..."
cat /etc/os-release | head -3 > "$OUTPUT_DIR/os.txt"

echo "[2/8] CANN 与驱动版本..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg > "$OUTPUT_DIR/cann_version.txt" 2>/dev/null
cat /usr/local/Ascend/driver/version.info > "$OUTPUT_DIR/driver_version.txt" 2>/dev/null
cat /usr/local/Ascend/firmware/version.info > "$OUTPUT_DIR/firmware_version.txt" 2>/dev/null

echo "[3/8] 版本兼容性检查..."
export LD_LIBRARY_PATH=/usr/local/Ascend/toolbox/6.0.0/Ascend-DMI/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
/usr/local/Ascend/toolbox/6.0.0/Ascend-DMI/bin/ascend-dmi --compatible > "$OUTPUT_DIR/compatible.txt" 2>&1

# ── 硬件信息 ──
echo "[4/8] 卡间拓扑..."
npu-smi info -l > "$OUTPUT_DIR/topology.txt"

echo "[5/8] NPU $NPU_ID 硬件规格 (ascend-dmi)..."
/usr/local/Ascend/toolbox/6.0.0/Ascend-DMI/bin/ascend-dmi --info --detail --fmt json > "$OUTPUT_DIR/hardware_detail_npu${NPU_ID}.json" 2>&1

echo "[6/8] NPU $NPU_ID 物理信息 (board)..."
npu-smi info -t board -i "$NPU_ID" > "$OUTPUT_DIR/board_npu${NPU_ID}.txt"

echo "[7/8] NPU $NPU_ID ECC 与 PCIe 错误..."
npu-smi info -t ecc -i "$NPU_ID" > "$OUTPUT_DIR/ecc_npu${NPU_ID}.txt"
npu-smi info -t pcie-err -i "$NPU_ID" > "$OUTPUT_DIR/pcie_err_npu${NPU_ID}.txt"

# ── 性能基线 ──
echo "[8/8] NPU $NPU_ID HBM 带宽测试 (d2d)..."
/usr/local/Ascend/toolbox/6.0.0/Ascend-DMI/bin/ascend-dmi --bw -t d2d -d "$NPU_ID" -q > "$OUTPUT_DIR/bw_d2d_npu${NPU_ID}.txt" 2>&1

echo ""
echo "=== 诊断完成 ==="
echo "完整输出: $OUTPUT_DIR"
echo ""
echo "关键信息速览:"
echo "---"
grep -E "Version|version" "$OUTPUT_DIR"/cann_version.txt 2>/dev/null | head -3
grep -E "Health|health" "$OUTPUT_DIR"/hardware_detail_npu${NPU_ID}.json 2>/dev/null | head -1 || \
  echo "  (JSON 输出，请直接查看 $OUTPUT_DIR/hardware_detail_npu${NPU_ID}.json)"
echo "  HBM 带宽: 见 $OUTPUT_DIR/bw_d2d_npu${NPU_ID}.txt"
