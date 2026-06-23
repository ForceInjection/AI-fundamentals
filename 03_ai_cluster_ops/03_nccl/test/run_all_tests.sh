#!/bin/bash
# NCCL Benchmark 脚本单元测试运行器

VERSION="3.0"; SCRIPT_NAME="NCCL Benchmark Test Suite"
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_RESULTS_DIR="/tmp/nccl_test_results_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$TEST_RESULTS_DIR/test_suite.log"

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
QUICK_MODE=false; VERBOSE_MODE=false

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; PURPLE='\033[0;35m'; NC='\033[0m'

log()      { mkdir -p "$(dirname "$MAIN_LOG")" 2>/dev/null; echo -e "$1" | tee -a "$MAIN_LOG"; }
log_info() { log "${BLUE}[INFO]${NC} $1"; }
log_pass() { log "${GREEN}[SUCCESS]${NC} $1"; }
log_err()  { log "${RED}[ERROR]${NC} $1"; }

suites=(
    "syntax:语法检查:test_syntax_basic.sh"
    "args:参数解析:test_arg_parsing.sh"
    "config:配置管理器:test_config_manager.sh"
    "optimization:优化级别:test_optimization_levels.sh"
    "pxn:PXN 模式:test_pxn_mode.sh"
)

show_help() {
    cat << EOF
$SCRIPT_NAME v$VERSION — 验证 nccl_benchmark.sh 脚本逻辑

用法: $0 [选项]

选项:
  -h, --help          帮助
  --list               列出套件
  --suite NAME         运行指定套件
  -q, --quick          快速模式
  --verbose            详细输出

套件: syntax / config / optimization / pxn
EOF
}

list_suites() {
    log "${PURPLE}=== 可用测试套件 ===${NC}"
    for s in "${suites[@]}"; do
        IFS=':' read -r name desc script <<< "$s"
        [ -f "$TEST_DIR/$script" ] && log "✅ $name — $desc" || log "❌ $name — 脚本缺失"
    done
}

run_one() {
    local name="$1" desc="$2" script="$3"
    TOTAL=$((TOTAL + 1))
    log ""; log "${PURPLE}=== $name ===${NC}"

    local spath="$TEST_DIR/$script" slog="$TEST_RESULTS_DIR/${name}.log"
    [ ! -f "$spath" ] && { SKIPPED=$((SKIPPED + 1)); log "跳过: 脚本不存在"; return 0; }
    chmod +x "$spath" 2>/dev/null

    local start=$(date +%s) exit_code=0
    if [ "$VERBOSE_MODE" = true ]; then
        bash "$spath" 2>&1 | tee "$slog"; exit_code=${PIPESTATUS[0]}
    else
        bash "$spath" > "$slog" 2>&1; exit_code=$?
    fi
    local dur=$(($(date +%s) - start))

    local pc=$(grep -c "\[PASS\]" "$slog" 2>/dev/null || echo 0)
    local fc=$(grep -c "\[FAIL\]" "$slog" 2>/dev/null || echo 0)
    local sum="(${pc} passed, ${fc} failed, ${dur}s)"

    if [ $exit_code -eq 0 ]; then
        PASSED=$((PASSED + 1)); log "${GREEN}[SUITE-PASS]${NC} $name $sum"
    else
        FAILED=$((FAILED + 1)); log "${RED}[SUITE-FAIL]${NC} $name $sum"
        [ "$VERBOSE_MODE" = false ] && log "  查看: $slog"
    fi
}

# ── parse args ──
TARGET_SUITE=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) show_help; exit 0 ;;
        --list) list_suites; exit 0 ;;
        -q|--quick) QUICK_MODE=true ;;
        --verbose) VERBOSE_MODE=true ;;
        --suite) TARGET_SUITE="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# ── run ──
mkdir -p "$TEST_RESULTS_DIR"
log "${PURPLE}=== NCCL Benchmark 脚本测试 ===${NC}"
log_info "结果目录: $TEST_RESULTS_DIR"

if [ -n "$TARGET_SUITE" ]; then
    for s in "${suites[@]}"; do
        IFS=':' read -r name desc script <<< "$s"
        [ "$name" = "$TARGET_SUITE" ] && run_one "$name" "$desc" "$script"
    done
else
    for s in "${suites[@]}"; do
        IFS=':' read -r name desc script <<< "$s"
        run_one "$name" "$desc" "$script"
    done
fi

# ── report ──
log ""; log "${PURPLE}=== 报告 ===${NC}"
log "  总计: $TOTAL, 通过: $PASSED, 失败: $FAILED, 跳过: $SKIPPED"
[ "$FAILED" -eq 0 ] && log_pass "全部通过" || log_err "存在失败"
exit $FAILED
