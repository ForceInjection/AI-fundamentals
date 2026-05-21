#!/bin/bash
# phase5/atc_convert.sh — ATC 模型转换示例 (PyTorch → ONNX → OM)
# 用法: bash atc_convert.sh [model.pth]
#       不提供 .pth 时自动创建 ResNet-50 并导出

set -e
MODEL_PTH="${1:-}"
INPUT_SIZE=224
OUTPUT_NAME="resnet50_910B3"

# ── 加载环境 ──
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/npu-learning/venv/bin/activate

echo "=== ATC 模型转换 ==="

# ── Step 1: 准备 ONNX ──
ONNX_FILE="${OUTPUT_NAME}.onnx"

if [ -n "$MODEL_PTH" ] && [ -f "$MODEL_PTH" ]; then
    echo "[1/2] 使用已有模型: $MODEL_PTH"
    python3 -c "
import torch
import torchvision.models as models
model = models.resnet50(weights=None)
model.load_state_dict(torch.load('$MODEL_PTH', map_location='cpu'))
model.eval()
dummy = torch.randn(1, 3, $INPUT_SIZE, $INPUT_SIZE)
torch.onnx.export(model, dummy, '$ONNX_FILE',
    input_names=['input'], output_names=['output'], opset_version=11)
print('ONNX 导出完成')
"
else
    echo "[1/2] 创建 ResNet-50 随机权重并导出 ONNX..."
    python3 -c "
import torch
import torchvision.models as models
model = models.resnet50(weights=None)
model.eval()
dummy = torch.randn(1, 3, $INPUT_SIZE, $INPUT_SIZE)
torch.onnx.export(model, dummy, '$ONNX_FILE',
    input_names=['input'], output_names=['output'], opset_version=11)
print('ONNX 导出完成 (随机权重)')
"
fi

ONNX_SIZE=$(du -h "$ONNX_FILE" | cut -f1)
echo "  ONNX 模型: $ONNX_FILE ($ONNX_SIZE)"

# ── Step 2: ONNX → OM ──
echo "[2/2] ATC 编译 ONNX → OM..."
atc --model="$ONNX_FILE" \
    --framework=5 \
    --output="$OUTPUT_NAME" \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,${INPUT_SIZE},${INPUT_SIZE}" \
    --input_format=NCHW \
    --log=error

OM_FILE="${OUTPUT_NAME}.om"
if [ -f "$OM_FILE" ]; then
    OM_SIZE=$(du -h "$OM_FILE" | cut -f1)
    echo "  OM 模型已生成: $OM_FILE ($OM_SIZE)"
    echo ""
    echo "=== 转换成功 ==="
    echo "  输入:  $ONNX_FILE ($ONNX_SIZE)"
    echo "  输出:  $OM_FILE ($OM_SIZE)"
    echo "  芯片:  Ascend 910B3"
    echo "  形状:  1×3×${INPUT_SIZE}×${INPUT_SIZE}"
else
    echo "ATC 转换失败，检查 atc 日志"
    exit 1
fi
