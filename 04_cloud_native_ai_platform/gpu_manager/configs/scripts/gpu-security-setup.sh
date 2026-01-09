#!/bin/bash

# GPU 安全设置脚本
# 用于配置 GPU 环境的安全策略和访问控制

set -euo pipefail

# 配置变量
GPU_GROUP="gpu"
DOCKER_GROUP="docker"
SECURITY_CONFIG_DIR="/etc/gpu-security"
LOG_FILE="/tmp/gpu-security-setup.log"
BACKUP_DIR="/tmp/gpu-security-backup-$(date +%Y%m%d-%H%M%S)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

# 检查权限
check_permissions() {
    if [ "$EUID" -ne 0 ]; then
        log_error "此脚本需要 root 权限运行"
        exit 1
    fi
}

# 创建备份
create_backup() {
    log_info "创建配置备份..."
    mkdir -p "$BACKUP_DIR"
    
    # 备份重要配置文件
    local files_to_backup=(
        "/etc/group"
        "/etc/passwd"
        "/etc/sudoers"
        "/etc/udev/rules.d/*nvidia*"
        "/etc/docker/daemon.json"
    )
    
    for file_pattern in "${files_to_backup[@]}"; do
        if ls $file_pattern 1> /dev/null 2>&1; then
            cp $file_pattern "$BACKUP_DIR/" 2>/dev/null || true
        fi
    done
    
    log_success "备份创建完成: $BACKUP_DIR"
}

# 创建 GPU 用户组
create_gpu_group() {
    log_info "创建 GPU 用户组..."
    
    if ! getent group "$GPU_GROUP" > /dev/null 2>&1; then
        groupadd "$GPU_GROUP"
        log_success "GPU 用户组 '$GPU_GROUP' 创建成功"
    else
        log_info "GPU 用户组 '$GPU_GROUP' 已存在"
    fi
}

# 配置设备权限
configure_device_permissions() {
    log_info "配置 GPU 设备权限..."
    
    # 创建 udev 规则
    cat > /etc/udev/rules.d/70-nvidia-gpu-security.rules << EOF
# NVIDIA GPU 设备安全规则
# 设置 GPU 设备的用户组和权限

# NVIDIA 控制设备
KERNEL=="nvidia_ctl", GROUP="$GPU_GROUP", MODE="0660"

# NVIDIA GPU 设备
KERNEL=="nvidia[0-9]*", GROUP="$GPU_GROUP", MODE="0660"

# NVIDIA UVM 设备
KERNEL=="nvidia-uvm", GROUP="$GPU_GROUP", MODE="0660"
KERNEL=="nvidia-uvm-tools", GROUP="$GPU_GROUP", MODE="0660"

# NVIDIA 模式设置设备
KERNEL=="nvidia-modeset", GROUP="$GPU_GROUP", MODE="0660"

# NVIDIA 持久化守护进程
KERNEL=="nvidia-persistenced", GROUP="$GPU_GROUP", MODE="0660"
EOF
    
    # 重新加载 udev 规则
    udevadm control --reload-rules
    udevadm trigger
    
    log_success "GPU 设备权限配置完成"
}

# 配置用户访问控制
configure_user_access() {
    log_info "配置用户访问控制..."
    
    # 创建安全配置目录
    mkdir -p "$SECURITY_CONFIG_DIR"
    
    # 创建用户访问策略文件
    cat > "$SECURITY_CONFIG_DIR/gpu-access-policy.conf" << EOF
# GPU 访问策略配置
# 定义哪些用户和组可以访问 GPU 资源

# 允许的用户组
allowed_groups=$GPU_GROUP,docker

# 允许的用户（逗号分隔）
# allowed_users=user1,user2

# 最大并发 GPU 进程数（每用户）
max_processes_per_user=10

# 最大 GPU 内存使用率（百分比）
max_memory_usage_percent=90

# 是否启用审计日志
enable_audit_log=true

# 审计日志文件
audit_log_file=/var/log/gpu-access.log
EOF
    
    log_success "用户访问控制配置完成"
}

# 配置 Docker GPU 安全
configure_docker_gpu_security() {
    log_info "配置 Docker GPU 安全..."
    
    # 检查 Docker 是否安装
    if ! command -v docker &> /dev/null; then
        log_warn "Docker 未安装，跳过 Docker GPU 安全配置"
        return
    fi
    
    # 备份现有 Docker 配置
    if [ -f "/etc/docker/daemon.json" ]; then
        cp "/etc/docker/daemon.json" "/etc/docker/daemon.json.backup"
    fi
    
    # 创建或更新 Docker 配置
    local docker_config='{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "runc",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "security-opts": [
    "apparmor=docker-default",
    "seccomp=default"
  ],
  "userns-remap": "default",
  "no-new-privileges": true
}'
    
    echo "$docker_config" > /etc/docker/daemon.json
    
    # 重启 Docker 服务
    systemctl restart docker
    
    log_success "Docker GPU 安全配置完成"
}

# 配置 NVIDIA 持久化模式
configure_nvidia_persistence() {
    log_info "配置 NVIDIA 持久化模式..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi 未找到，跳过持久化模式配置"
        return
    fi
    
    # 启用持久化模式
    nvidia-smi -pm 1
    
    # 创建 systemd 服务
    cat > /etc/systemd/system/nvidia-persistenced.service << 'EOF'
[Unit]
Description=NVIDIA Persistence Daemon
Wants=syslog.target

[Service]
Type=forking
PIDFile=/var/run/nvidia-persistenced/nvidia-persistenced.pid
ExecStart=/usr/bin/nvidia-persistenced --verbose
ExecStopPost=/bin/rm -rf /var/run/nvidia-persistenced
User=nvidia-persistenced
Group=nvidia-persistenced

[Install]
WantedBy=multi-user.target
EOF
    
    # 创建专用用户
    if ! id "nvidia-persistenced" &>/dev/null; then
        useradd -r -s /sbin/nologin nvidia-persistenced
    fi
    
    # 启用并启动服务
    systemctl enable nvidia-persistenced
    systemctl start nvidia-persistenced
    
    log_success "NVIDIA 持久化模式配置完成"
}

# 配置资源限制
configure_resource_limits() {
    log_info "配置资源限制..."
    
    # 创建 systemd 用户切片配置
    mkdir -p /etc/systemd/system/user-.slice.d
    
    cat > /etc/systemd/system/user-.slice.d/gpu-limits.conf << 'EOF'
[Slice]
# GPU 用户资源限制
# 限制每个用户的系统资源使用

# 内存限制（8GB）
MemoryMax=8G
MemoryHigh=6G

# CPU 限制
CPUQuota=400%

# 进程数限制
TasksMax=1000

# IO 限制
IOWeight=100
EOF
    
    # 重新加载 systemd 配置
    systemctl daemon-reload
    
    log_success "资源限制配置完成"
}

# 配置审计日志
configure_audit_logging() {
    log_info "配置审计日志..."
    
    # 创建日志目录
    mkdir -p /var/log/gpu-security
    
    # 创建 GPU 访问监控脚本
    cat > /usr/local/bin/gpu-access-monitor.sh << 'EOF'
#!/bin/bash

# GPU 访问监控脚本
# 记录 GPU 设备的访问和使用情况

LOG_FILE="/var/log/gpu-security/gpu-access.log"
PID_FILE="/var/run/gpu-access-monitor.pid"

# 检查是否已经运行
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "GPU 访问监控已在运行"
    exit 1
fi

# 记录 PID
echo $$ > "$PID_FILE"

# 清理函数
cleanup() {
    rm -f "$PID_FILE"
    exit 0
}

trap cleanup SIGTERM SIGINT

# 监控循环
while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 记录当前 GPU 使用情况
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
        echo "[$timestamp] GPU_STATUS: $gpu_info" >> "$LOG_FILE"
        
        # 记录 GPU 进程
        gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader)
        if [ -n "$gpu_processes" ]; then
            echo "[$timestamp] GPU_PROCESSES: $gpu_processes" >> "$LOG_FILE"
        fi
    fi
    
    # 记录设备访问
    if [ -e /dev/nvidia0 ]; then
        device_access=$(lsof /dev/nvidia* 2>/dev/null | tail -n +2 || true)
        if [ -n "$device_access" ]; then
            echo "[$timestamp] DEVICE_ACCESS: $device_access" >> "$LOG_FILE"
        fi
    fi
    
    sleep 30
done
EOF
    
    chmod +x /usr/local/bin/gpu-access-monitor.sh
    
    # 创建 systemd 服务
    cat > /etc/systemd/system/gpu-access-monitor.service << 'EOF'
[Unit]
Description=GPU Access Monitor
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/gpu-access-monitor.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF
    
    # 启用服务
    systemctl enable gpu-access-monitor
    systemctl start gpu-access-monitor
    
    log_success "审计日志配置完成"
}

# 配置防火墙规则
configure_firewall() {
    log_info "配置防火墙规则..."
    
    # 检查防火墙服务
    if systemctl is-active --quiet ufw; then
        # UFW 配置
        log_info "配置 UFW 防火墙规则..."
        
        # 允许本地 GPU 监控端口
        ufw allow from 127.0.0.1 to any port 9400 comment "DCGM Exporter"
        ufw allow from 127.0.0.1 to any port 3476 comment "NVIDIA ML"
        
    elif systemctl is-active --quiet firewalld; then
        # Firewalld 配置
        log_info "配置 Firewalld 防火墙规则..."
        
        # 创建 GPU 服务定义
        cat > /etc/firewalld/services/gpu-monitoring.xml << 'EOF'
<?xml version="1.0" encoding="utf-8"?>
<service>
  <short>GPU Monitoring</short>
  <description>GPU monitoring and management services</description>
  <port protocol="tcp" port="9400"/>
  <port protocol="tcp" port="3476"/>
</service>
EOF
        
        firewall-cmd --reload
        firewall-cmd --permanent --add-service=gpu-monitoring --zone=internal
        firewall-cmd --reload
        
    else
        log_warn "未检测到支持的防火墙服务"
    fi
    
    log_success "防火墙规则配置完成"
}

# 创建安全检查脚本
create_security_check_script() {
    log_info "创建安全检查脚本..."
    
    cat > /usr/local/bin/gpu-security-check.sh << 'EOF'
#!/bin/bash

# GPU 安全检查脚本
# 检查 GPU 环境的安全配置状态

echo "=== GPU 安全状态检查 ==="
echo "检查时间: $(date)"
echo

# 检查用户组
echo "1. 检查 GPU 用户组:"
if getent group gpu > /dev/null 2>&1; then
    echo "   ✅ GPU 用户组存在"
    echo "   成员: $(getent group gpu | cut -d: -f4)"
else
    echo "   ❌ GPU 用户组不存在"
fi
echo

# 检查设备权限
echo "2. 检查设备权限:"
if [ -e /dev/nvidia0 ]; then
    ls -la /dev/nvidia* | while read line; do
        echo "   $line"
    done
else
    echo "   ⚠️  未找到 NVIDIA 设备"
fi
echo

# 检查 udev 规则
echo "3. 检查 udev 规则:"
if [ -f /etc/udev/rules.d/70-nvidia-gpu-security.rules ]; then
    echo "   ✅ GPU 安全 udev 规则存在"
else
    echo "   ❌ GPU 安全 udev 规则不存在"
fi
echo

# 检查 Docker 配置
echo "4. 检查 Docker GPU 配置:"
if [ -f /etc/docker/daemon.json ]; then
    if grep -q "nvidia" /etc/docker/daemon.json; then
        echo "   ✅ Docker NVIDIA 运行时已配置"
    else
        echo "   ⚠️  Docker NVIDIA 运行时未配置"
    fi
else
    echo "   ⚠️  Docker 配置文件不存在"
fi
echo

# 检查监控服务
echo "5. 检查监控服务:"
if systemctl is-active --quiet gpu-access-monitor; then
    echo "   ✅ GPU 访问监控服务运行中"
else
    echo "   ❌ GPU 访问监控服务未运行"
fi
echo

# 检查日志文件
echo "6. 检查日志文件:"
if [ -f /var/log/gpu-security/gpu-access.log ]; then
    log_size=$(du -h /var/log/gpu-security/gpu-access.log | cut -f1)
    echo "   ✅ GPU 访问日志存在 (大小: $log_size)"
else
    echo "   ❌ GPU 访问日志不存在"
fi
echo

echo "=== 检查完成 ==="
EOF
    
    chmod +x /usr/local/bin/gpu-security-check.sh
    
    log_success "安全检查脚本创建完成"
}

# 添加用户到 GPU 组
add_user_to_gpu_group() {
    local username="$1"
    
    if [ -z "$username" ]; then
        log_error "用户名不能为空"
        return 1
    fi
    
    if ! id "$username" &>/dev/null; then
        log_error "用户 '$username' 不存在"
        return 1
    fi
    
    usermod -a -G "$GPU_GROUP" "$username"
    log_success "用户 '$username' 已添加到 GPU 组"
}

# 移除用户从 GPU 组
remove_user_from_gpu_group() {
    local username="$1"
    
    if [ -z "$username" ]; then
        log_error "用户名不能为空"
        return 1
    fi
    
    gpasswd -d "$username" "$GPU_GROUP"
    log_success "用户 '$username' 已从 GPU 组移除"
}

# 清理函数
cleanup() {
    log_info "GPU 安全设置完成"
    log_info "配置文件位置:"
    log_info "  - 安全配置: $SECURITY_CONFIG_DIR"
    log_info "  - 日志文件: $LOG_FILE"
    log_info "  - 备份目录: $BACKUP_DIR"
    log_info "  - 安全检查: /usr/local/bin/gpu-security-check.sh"
}

# 显示帮助信息
show_help() {
    cat << EOF
GPU 安全设置脚本

用法: $0 [选项] [参数]

选项:
  install         安装完整的 GPU 安全配置
  group           仅创建 GPU 用户组
  permissions     仅配置设备权限
  docker          仅配置 Docker GPU 安全
  monitoring      仅配置监控和审计
  firewall        仅配置防火墙规则
  add-user USER   添加用户到 GPU 组
  remove-user USER 从 GPU 组移除用户
  check           运行安全检查
  help            显示此帮助信息

环境变量:
  GPU_GROUP       GPU 用户组名称 (默认: gpu)
  DOCKER_GROUP    Docker 用户组名称 (默认: docker)

示例:
  $0 install
  $0 add-user alice
  $0 check
  GPU_GROUP=nvidia $0 install

注意: 此脚本需要 root 权限运行

EOF
}

# 主函数
main() {
    local action=${1:-help}
    local username=${2:-}
    
    log_info "开始 GPU 安全设置: $action"
    
    case $action in
        install)
            check_permissions
            create_backup
            create_gpu_group
            configure_device_permissions
            configure_user_access
            configure_docker_gpu_security
            configure_nvidia_persistence
            configure_resource_limits
            configure_audit_logging
            configure_firewall
            create_security_check_script
            ;;
        group)
            check_permissions
            create_gpu_group
            ;;
        permissions)
            check_permissions
            configure_device_permissions
            ;;
        docker)
            check_permissions
            configure_docker_gpu_security
            ;;
        monitoring)
            check_permissions
            configure_audit_logging
            ;;
        firewall)
            check_permissions
            configure_firewall
            ;;
        add-user)
            check_permissions
            if [ -z "$username" ]; then
                log_error "请提供用户名"
                show_help
                exit 1
            fi
            add_user_to_gpu_group "$username"
            ;;
        remove-user)
            check_permissions
            if [ -z "$username" ]; then
                log_error "请提供用户名"
                show_help
                exit 1
            fi
            remove_user_from_gpu_group "$username"
            ;;
        check)
            if [ -f /usr/local/bin/gpu-security-check.sh ]; then
                /usr/local/bin/gpu-security-check.sh
            else
                log_error "安全检查脚本不存在，请先运行安装"
            fi
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
    
    log_success "GPU 安全设置操作完成"
}

# 信号处理
trap cleanup EXIT

# 执行主函数
main "$@"