#!/bin/bash

# =============================================================================
# DNS 解析逻辑测试脚本
# 功能：用于验证 NCCL 多节点配置中的 DNS 解析功能
# =============================================================================

echo "=== DNS 解析逻辑测试脚本 ==="
echo "测试时间: $(date)"
echo

# 测试函数：DNS 解析逻辑
test_dns_resolution() {
    local hostname="$1"
    local expected_result="$2"
    
    echo "--- 测试主机名: $hostname ---"
    
    # 使用与 YAML 中相同的解析逻辑
    echo "正在使用 ping 解析 IP 地址..."
    
    # 主要解析方法
    RESOLVED_ADDR=$(ping -c 1 -W 2 "$hostname" 2>/dev/null | grep "PING" | sed -n 's/.*(\([0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\)).*/\1/p')
    
    # 如果上述方法失败，尝试另一种提取方式
    if [ -z "$RESOLVED_ADDR" ]; then
        echo "尝试备用解析方法..."
        RESOLVED_ADDR=$(ping -c 1 -W 2 "$hostname" 2>/dev/null | head -1 | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}')
    fi
    
    # 验证解析结果和 IP 地址格式
    if [ -z "$RESOLVED_ADDR" ]; then
        echo "⚠️ 错误: 无法解析主机地址 $hostname"
        echo "使用域名作为备选方案"
        RESOLVED_ADDR="$hostname"
        return 1
    elif [[ ! "$RESOLVED_ADDR" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        echo "⚠️ 警告: 解析结果格式异常: $RESOLVED_ADDR"
        echo "使用域名作为备选方案"
        RESOLVED_ADDR="$hostname"
        return 1
    else
        echo "✅ 成功解析 IP: $RESOLVED_ADDR"
        
        # 验证 IP 地址的每个段是否在有效范围内 (0-255)
        IFS='.' read -ra IP_PARTS <<< "$RESOLVED_ADDR"
        for part in "${IP_PARTS[@]}"; do
            if [ "$part" -gt 255 ] || [ "$part" -lt 0 ]; then
                echo "⚠️ 警告: IP 地址段超出范围: $part"
                echo "使用域名作为备选方案"
                RESOLVED_ADDR="$hostname"
                return 1
            fi
        done
        
        echo "✅ IP 地址格式验证通过"
        
        # 如果提供了期望结果，进行比较
        if [ -n "$expected_result" ]; then
            if [ "$RESOLVED_ADDR" = "$expected_result" ]; then
                echo "✅ 解析结果与期望一致: $RESOLVED_ADDR"
            else
                echo "⚠️  解析结果与期望不同: 实际=$RESOLVED_ADDR, 期望=$expected_result"
            fi
        fi
        
        return 0
    fi
}

# 测试 ping 命令输出格式分析
analyze_ping_output() {
    local hostname="$1"
    
    echo "--- 分析 ping 输出格式: $hostname ---"
    
    # 执行 ping 并显示原始输出
    echo "原始 ping 输出:"
    ping -c 1 -W 2 "$hostname" 2>/dev/null || echo "ping 命令失败"
    echo
    
    # 测试不同的解析方法
    echo "方法1 - sed 正则表达式:"
    method1=$(ping -c 1 -W 2 "$hostname" 2>/dev/null | grep "PING" | sed -n 's/.*(\([0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\)).*/\1/p')
    echo "结果: '$method1'"
    
    echo "方法2 - grep -oE:"
    method2=$(ping -c 1 -W 2 "$hostname" 2>/dev/null | head -1 | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}')
    echo "结果: '$method2'"
    
    echo "方法3 - awk 提取:"
    method3=$(ping -c 1 -W 2 "$hostname" 2>/dev/null | grep "PING" | awk -F'[()]' '{print $2}')
    echo "结果: '$method3'"
    
    echo
}

# 验证 IP 地址格式的函数
validate_ip_format() {
    local ip="$1"
    
    echo "--- 验证 IP 地址格式: $ip ---"
    
    # 检查基本格式
    if [[ "$ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        echo "✅ 基本格式检查通过"
        
        # 检查每个段的范围
        IFS='.' read -ra IP_PARTS <<< "$ip"
        valid=true
        for i in "${!IP_PARTS[@]}"; do
            part="${IP_PARTS[$i]}"
            if [ "$part" -gt 255 ] || [ "$part" -lt 0 ]; then
                echo "⚠️ 第$((i+1))段超出范围: $part (应在0-255之间)"
                valid=false
            else
                echo "✅ 第$((i+1))段有效: $part"
            fi
        done
        
        if [ "$valid" = true ]; then
            echo "✅ IP 地址完全有效: $ip"
            return 0
        else
            echo "⚠️ IP 地址无效: $ip"
            return 1
        fi
    else
        echo "⚠️ 基本格式检查失败: $ip"
        return 1
    fi
}

# 主测试流程
main() {
    echo "开始 DNS 解析逻辑测试..."
    echo
    
    # 测试常见的主机名
    # 格式: "hostname:expected_success" (true/false)
    test_cases=(
        "localhost:true"
        "google.com:true"
        "github.com:true"
        "kubernetes.default.svc.cluster.local:false"
        "nccl-multinode-0.nccl-multinode.default:false"
        "invalid-hostname-that-should-fail.local:false"
    )
    
    echo "=== 测试用例执行 ==="
    for test_case in "${test_cases[@]}"; do
        IFS=':' read -r hostname expected_success <<< "$test_case"
        
        if test_dns_resolution "$hostname"; then
            if [ "$expected_success" = "true" ]; then
                echo "✅ [PASS] $hostname 解析成功 (符合预期)"
            else
                echo "⚠️ [WARN] $hostname 解析成功 (预期失败，可能环境支持解析)"
            fi
        else
            if [ "$expected_success" = "false" ]; then
                echo "✅ [PASS] $hostname 解析失败 (符合预期)"
            else
                echo "❌ [FAIL] $hostname 解析失败 (预期成功)"
                exit 1
            fi
        fi
        echo
    done
    
    echo "=== ping 输出格式分析 ==="
    analyze_ping_output "google.com"
    
    echo "=== IP 地址格式验证测试 ==="
    # 格式: "ip:expected_valid" (true/false)
    ip_test_cases=(
        "192.168.1.1:true"
        "10.0.0.1:true"
        "172.16.0.1:true"
        "8.8.8.8:true"
        "256.1.1.1:false"    # 无效：超出范围
        "192.168.1:false"    # 无效：格式错误
        "192.168.1.1.1:false" # 无效：段数错误
        "84:false"           # 无效：之前遇到的问题
    )
    
    for test_case in "${ip_test_cases[@]}"; do
        IFS=':' read -r ip expected_valid <<< "$test_case"
        
        if validate_ip_format "$ip"; then
            if [ "$expected_valid" = "true" ]; then
                echo "✅ [PASS] $ip 验证通过 (符合预期)"
            else
                echo "❌ [FAIL] $ip 验证通过 (预期失败)"
                exit 1
            fi
        else
            if [ "$expected_valid" = "false" ]; then
                echo "✅ [PASS] $ip 验证失败 (符合预期)"
            else
                echo "❌ [FAIL] $ip 验证失败 (预期成功)"
                exit 1
            fi
        fi
        echo
    done
    
    echo "=== 测试总结 ==="
    echo "✅ DNS 解析逻辑测试完成"
    echo "📝 请检查上述输出，确认解析逻辑工作正常"
    echo "🔧 如发现问题，请根据测试结果调整 YAML 配置中的解析逻辑"
}

# 如果脚本被直接执行
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi