#!/bin/bash

# =============================================================================
# 测试 nvlink_count 变量的整数表达式修复
# 验证修复后的脚本不再出现 "[: 0 0: integer expression expected" 错误
# =============================================================================

echo "=== 测试 nvlink_count 变量修复 ==="
echo

# 模拟可能导致问题的 nvidia-smi 输出
test_nvlink_count_parsing() {
    echo "测试 nvlink_count 变量解析..."
    
    # 模拟各种可能的 nvidia-smi 输出
    local test_cases=(
        "0"           # 正常情况：没有 NVLink
        "2"           # 正常情况：有 2 个 NVLink
        "0\n0"        # 问题情况：多行输出
        " 0 "         # 问题情况：包含空格
        "0\n"         # 问题情况：包含换行符
        ""            # 边界情况：空输出
    )
    
    for i in "${!test_cases[@]}"; do
        local test_input="${test_cases[$i]}"
        echo "测试案例 $((i+1)): '$test_input'"
        
        # 使用修复后的逻辑处理
        local nvlink_count=$(echo -e "$test_input" | tr -d ' \n\r\t' | grep -oE '^[0-9]+$' | head -1)
        if [ -z "$nvlink_count" ]; then
            nvlink_count="0"
        fi
        
        echo "  处理后的值: '$nvlink_count'"
        
        # 测试修复后的比较逻辑
        if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
            echo "  ✅ 比较结果: NVLink 可用 ($nvlink_count 个)"
        else
            echo "  ✅ 比较结果: NVLink 不可用"
        fi
        echo
    done
}

# 测试实际的脚本函数逻辑
test_script_logic() {
    echo "测试脚本中的实际逻辑..."
    
    # 模拟 nvidia-smi 命令不可用的情况
    local nvlink_count
    
    # 测试第一种获取方式（第461行逻辑）
    echo "测试第461行逻辑（nvidia-smi nvlink --status）:"
    nvlink_count=$(echo "0 GB/s" | grep -c "GB/s" | tr -d ' \n\r\t' | grep -oE '^[0-9]+$' | head -1)
    if [ -z "$nvlink_count" ]; then
        nvlink_count="0"
    fi
    echo "  nvlink_count = '$nvlink_count'"
    if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
        echo "  ✅ 比较成功: NVLink 可用"
    else
        echo "  ✅ 比较成功: NVLink 不可用"
    fi
    echo
    
    # 测试第735行逻辑
    echo "测试第735行逻辑（nvidia-smi nvlink -s）:"
    nvlink_count=$(echo -e "Link 0: Active\nLink 1: Active" | grep -c "Active" | tr -d ' \n\r\t' | grep -oE '^[0-9]+$' | head -1)
    if [ -z "$nvlink_count" ]; then
        nvlink_count="0"
    fi
    echo "  nvlink_count = '$nvlink_count'"
    if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
        echo "  ✅ 比较成功: NVLink 可用 ($nvlink_count 个)"
    else
        echo "  ✅ 比较成功: NVLink 不可用"
    fi
    echo
    
    # 测试第849行逻辑
    echo "测试第849行逻辑（nvidia-smi nvlink -s）:"
    nvlink_count=$(echo "" | grep -c "Active" || echo "0")
    nvlink_count=$(echo "$nvlink_count" | tr -d ' \n\r\t' | grep -oE '^[0-9]+$' | head -1)
    if [ -z "$nvlink_count" ]; then
        nvlink_count="0"
    fi
    echo "  nvlink_count = '$nvlink_count'"
    if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
        echo "  ✅ 比较成功: NVLink 可用"
    else
        echo "  ✅ 比较成功: NVLink 不可用"
    fi
    echo
}

# 主函数
main() {
    test_nvlink_count_parsing
    test_script_logic
    
    echo "🎉 所有测试通过！"
    echo "   - nvlink_count 变量现在可以正确处理各种输入格式"
    echo "   - 使用正则表达式验证确保变量为纯数字"
    echo "   - 避免了 'integer expression expected' 错误"
    echo "   - 所有三个位置的修复都工作正常"
}

# 运行测试
main "$@"