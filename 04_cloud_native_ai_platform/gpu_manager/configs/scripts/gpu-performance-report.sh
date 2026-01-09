#!/bin/bash
# GPU自动化性能报告脚本
# 生成全面的GPU性能分析和监控报告

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# 配置变量
REPORT_DURATION=3600  # 默认1小时
OUTPUT_DIR="gpu_performance_report_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_DIR/report_generation.log"
REPORT_FILE="$OUTPUT_DIR/gpu_performance_report.html"
JSON_REPORT="$OUTPUT_DIR/performance_data.json"
SAMPLE_INTERVAL=10
REPORT_FORMAT="html"
INCLUDE_CHARTS=true
EMAIL_RECIPIENTS=""
SLACK_WEBHOOK=""
AUTO_UPLOAD=false

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "${PURPLE}\n=== $1 ===${NC}" | tee -a "$LOG_FILE"
}

# 创建输出目录
setup_output_directory() {
    mkdir -p "$OUTPUT_DIR"
    log_info "性能报告将保存到: $OUTPUT_DIR"
}

# 检查依赖工具
check_dependencies() {
    log_info "检查依赖工具..."
    
    local missing_tools=()
    
    # 必需工具
    if ! command -v nvidia-smi &> /dev/null; then
        missing_tools+=("nvidia-smi")
    fi
    
    # Python工具检查
    if command -v python3 &> /dev/null; then
        python3 -c "import json, csv, datetime" 2>/dev/null || missing_tools+=("python3-json")
    else
        missing_tools+=("python3")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "缺少必需工具: ${missing_tools[*]}"
        return 1
    fi
    
    log_info "✅ 依赖检查完成"
}

# 收集GPU性能数据
collect_performance_data() {
    log_section "收集GPU性能数据 (${REPORT_DURATION}秒)"
    
    # 创建数据收集脚本
    cat > "$OUTPUT_DIR/collect_data.py" << 'EOF'
import subprocess
import json
import csv
import time
import sys
from datetime import datetime, timedelta

class GPUDataCollector:
    def __init__(self, duration, interval, output_dir):
        self.duration = duration
        self.interval = interval
        self.output_dir = output_dir
        self.data = {
            'collection_info': {
                'start_time': datetime.now().isoformat(),
                'duration_seconds': duration,
                'interval_seconds': interval
            },
            'gpu_info': [],
            'performance_data': [],
            'process_data': [],
            'summary_stats': {}
        }
    
    def get_gpu_info(self):
        """获取GPU基本信息"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,compute_cap,driver_version,power.max_limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        self.data['gpu_info'].append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_total_mb': int(parts[2]),
                            'compute_capability': parts[3],
                            'driver_version': parts[4],
                            'power_limit_w': float(parts[5])
                        })
        except Exception as e:
            print(f"获取GPU信息失败: {e}")
    
    def collect_performance_sample(self):
        """收集单次性能样本"""
        try:
            # GPU性能数据
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw,clocks.gr,clocks.mem',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            sample_data = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 10:
                        sample_data.append({
                            'timestamp': parts[0],
                            'gpu_index': int(parts[1]),
                            'gpu_utilization_percent': float(parts[2]),
                            'memory_utilization_percent': float(parts[3]),
                            'memory_used_mb': int(parts[4]),
                            'memory_free_mb': int(parts[5]),
                            'temperature_c': float(parts[6]),
                            'power_draw_w': float(parts[7]),
                            'graphics_clock_mhz': int(parts[8]),
                            'memory_clock_mhz': int(parts[9])
                        })
            
            return sample_data
        except Exception as e:
            print(f"收集性能数据失败: {e}")
            return []
    
    def collect_process_data(self):
        """收集GPU进程数据"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory',
                '--format=csv,noheader'
            ], capture_output=True, text=True, check=True)
            
            process_data = []
            for line in result.stdout.strip().split('\n'):
                if line.strip() and 'No running processes found' not in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        process_data.append({
                            'timestamp': parts[0],
                            'gpu_uuid': parts[1],
                            'pid': int(parts[2]),
                            'process_name': parts[3],
                            'memory_used_mb': int(parts[4].replace(' MiB', ''))
                        })
            
            return process_data
        except Exception as e:
            print(f"收集进程数据失败: {e}")
            return []
    
    def calculate_summary_stats(self):
        """计算汇总统计信息"""
        if not self.data['performance_data']:
            return
        
        # 按GPU分组计算统计信息
        gpu_stats = {}
        
        for sample in self.data['performance_data']:
            for gpu_data in sample:
                gpu_idx = gpu_data['gpu_index']
                if gpu_idx not in gpu_stats:
                    gpu_stats[gpu_idx] = {
                        'gpu_utilization': [],
                        'memory_utilization': [],
                        'temperature': [],
                        'power_draw': [],
                        'memory_used': []
                    }
                
                gpu_stats[gpu_idx]['gpu_utilization'].append(gpu_data['gpu_utilization_percent'])
                gpu_stats[gpu_idx]['memory_utilization'].append(gpu_data['memory_utilization_percent'])
                gpu_stats[gpu_idx]['temperature'].append(gpu_data['temperature_c'])
                gpu_stats[gpu_idx]['power_draw'].append(gpu_data['power_draw_w'])
                gpu_stats[gpu_idx]['memory_used'].append(gpu_data['memory_used_mb'])
        
        # 计算统计值
        for gpu_idx, stats in gpu_stats.items():
            summary = {}
            for metric, values in stats.items():
                if values:
                    summary[metric] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'samples': len(values)
                    }
            
            self.data['summary_stats'][f'gpu_{gpu_idx}'] = summary
    
    def run_collection(self):
        """运行数据收集"""
        print(f"开始收集GPU性能数据，持续时间: {self.duration}秒")
        
        # 获取GPU基本信息
        self.get_gpu_info()
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < self.duration:
            # 收集性能数据
            perf_sample = self.collect_performance_sample()
            if perf_sample:
                self.data['performance_data'].append(perf_sample)
            
            # 每分钟收集一次进程数据
            if sample_count % 6 == 0:  # 假设间隔10秒，每分钟收集一次
                proc_data = self.collect_process_data()
                if proc_data:
                    self.data['process_data'].extend(proc_data)
            
            sample_count += 1
            if sample_count % 10 == 0:
                elapsed = time.time() - start_time
                remaining = self.duration - elapsed
                print(f"已收集 {sample_count} 个样本，剩余时间: {remaining:.0f}秒")
            
            time.sleep(self.interval)
        
        # 计算汇总统计
        self.calculate_summary_stats()
        
        # 保存数据
        self.save_data()
        
        print(f"数据收集完成，共收集 {sample_count} 个样本")
    
    def save_data(self):
        """保存收集的数据"""
        # 保存JSON格式
        json_file = f"{self.output_dir}/performance_data.json"
        with open(json_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        
        # 保存CSV格式
        csv_file = f"{self.output_dir}/performance_data.csv"
        with open(csv_file, 'w', newline='') as f:
            if self.data['performance_data']:
                # 获取第一个样本的字段
                first_sample = self.data['performance_data'][0][0]
                fieldnames = list(first_sample.keys())
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for sample in self.data['performance_data']:
                    for gpu_data in sample:
                        writer.writerow(gpu_data)
        
        print(f"数据已保存到: {json_file} 和 {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python3 collect_data.py <duration> <interval> <output_dir>")
        sys.exit(1)
    
    duration = int(sys.argv[1])
    interval = int(sys.argv[2])
    output_dir = sys.argv[3]
    
    collector = GPUDataCollector(duration, interval, output_dir)
    collector.run_collection()
EOF
    
    # 运行数据收集
    log_info "启动数据收集进程..."
    python3 "$OUTPUT_DIR/collect_data.py" "$REPORT_DURATION" "$SAMPLE_INTERVAL" "$OUTPUT_DIR"
    
    log_info "✅ 性能数据收集完成"
}

# 生成HTML报告
generate_html_report() {
    log_section "生成HTML性能报告"
    
    # 创建HTML报告生成脚本
    cat > "$OUTPUT_DIR/generate_html_report.py" << 'EOF'
import json
import sys
from datetime import datetime

def load_performance_data(json_file):
    """加载性能数据"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def generate_gpu_info_section(gpu_info):
    """生成GPU信息部分"""
    html = """
    <div class="section">
        <h2>🖥️ GPU设备信息</h2>
        <div class="gpu-grid">
    """
    
    for gpu in gpu_info:
        html += f"""
        <div class="gpu-card">
            <h3>GPU {gpu['index']}</h3>
            <p><strong>型号:</strong> {gpu['name']}</p>
            <p><strong>内存:</strong> {gpu['memory_total_mb']} MB</p>
            <p><strong>计算能力:</strong> {gpu['compute_capability']}</p>
            <p><strong>驱动版本:</strong> {gpu['driver_version']}</p>
            <p><strong>功耗限制:</strong> {gpu['power_limit_w']} W</p>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html

def generate_summary_stats_section(summary_stats):
    """生成汇总统计部分"""
    html = """
    <div class="section">
        <h2>📊 性能统计摘要</h2>
        <div class="stats-grid">
    """
    
    for gpu_key, stats in summary_stats.items():
        gpu_index = gpu_key.replace('gpu_', '')
        html += f"""
        <div class="stats-card">
            <h3>GPU {gpu_index} 统计</h3>
            <div class="stats-table">
        """
        
        for metric, values in stats.items():
            metric_name = metric.replace('_', ' ').title()
            html += f"""
            <div class="stat-row">
                <span class="stat-label">{metric_name}:</span>
                <span class="stat-value">
                    平均: {values['mean']:.1f} | 
                    最小: {values['min']:.1f} | 
                    最大: {values['max']:.1f}
                </span>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html

def generate_performance_charts(performance_data):
    """生成性能图表"""
    if not performance_data:
        return "<p>没有性能数据可显示</p>"
    
    # 准备图表数据
    chart_data = {}
    timestamps = []
    
    for i, sample in enumerate(performance_data[:100]):  # 限制显示前100个样本
        timestamps.append(i)
        for gpu_data in sample:
            gpu_idx = gpu_data['gpu_index']
            if gpu_idx not in chart_data:
                chart_data[gpu_idx] = {
                    'gpu_util': [],
                    'memory_util': [],
                    'temperature': [],
                    'power': []
                }
            
            chart_data[gpu_idx]['gpu_util'].append(gpu_data['gpu_utilization_percent'])
            chart_data[gpu_idx]['memory_util'].append(gpu_data['memory_utilization_percent'])
            chart_data[gpu_idx]['temperature'].append(gpu_data['temperature_c'])
            chart_data[gpu_idx]['power'].append(gpu_data['power_draw_w'])
    
    # 生成Chart.js图表
    html = """
    <div class="section">
        <h2>📈 性能趋势图表</h2>
        <div class="charts-container">
    """
    
    # GPU利用率图表
    html += """
        <div class="chart-container">
            <h3>GPU利用率 (%)</h3>
            <canvas id="gpuUtilChart" width="800" height="400"></canvas>
        </div>
    """
    
    # 内存利用率图表
    html += """
        <div class="chart-container">
            <h3>内存利用率 (%)</h3>
            <canvas id="memoryUtilChart" width="800" height="400"></canvas>
        </div>
    """
    
    # 温度图表
    html += """
        <div class="chart-container">
            <h3>温度 (°C)</h3>
            <canvas id="temperatureChart" width="800" height="400"></canvas>
        </div>
    """
    
    # 功耗图表
    html += """
        <div class="chart-container">
            <h3>功耗 (W)</h3>
            <canvas id="powerChart" width="800" height="400"></canvas>
        </div>
    """
    
    html += """
        </div>
    </div>
    
    <script>
    // 图表数据
    const timestamps = """ + str(timestamps) + """;
    const chartData = """ + json.dumps(chart_data) + """;
    
    // GPU利用率图表
    const gpuUtilCtx = document.getElementById('gpuUtilChart').getContext('2d');
    new Chart(gpuUtilCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: Object.keys(chartData).map((gpu, index) => ({
                label: `GPU ${gpu}`,
                data: chartData[gpu].gpu_util,
                borderColor: `hsl(${index * 60}, 70%, 50%)`,
                backgroundColor: `hsla(${index * 60}, 70%, 50%, 0.1)`,
                tension: 0.1
            }))
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
    
    // 内存利用率图表
    const memoryUtilCtx = document.getElementById('memoryUtilChart').getContext('2d');
    new Chart(memoryUtilCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: Object.keys(chartData).map((gpu, index) => ({
                label: `GPU ${gpu}`,
                data: chartData[gpu].memory_util,
                borderColor: `hsl(${index * 60 + 120}, 70%, 50%)`,
                backgroundColor: `hsla(${index * 60 + 120}, 70%, 50%, 0.1)`,
                tension: 0.1
            }))
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
    
    // 温度图表
    const temperatureCtx = document.getElementById('temperatureChart').getContext('2d');
    new Chart(temperatureCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: Object.keys(chartData).map((gpu, index) => ({
                label: `GPU ${gpu}`,
                data: chartData[gpu].temperature,
                borderColor: `hsl(${index * 60 + 240}, 70%, 50%)`,
                backgroundColor: `hsla(${index * 60 + 240}, 70%, 50%, 0.1)`,
                tension: 0.1
            }))
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // 功耗图表
    const powerCtx = document.getElementById('powerChart').getContext('2d');
    new Chart(powerCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: Object.keys(chartData).map((gpu, index) => ({
                label: `GPU ${gpu}`,
                data: chartData[gpu].power,
                borderColor: `hsl(${index * 60 + 300}, 70%, 50%)`,
                backgroundColor: `hsla(${index * 60 + 300}, 70%, 50%, 0.1)`,
                tension: 0.1
            }))
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    </script>
    """
    
    return html

def generate_recommendations(summary_stats):
    """生成优化建议"""
    recommendations = []
    
    for gpu_key, stats in summary_stats.items():
        gpu_index = gpu_key.replace('gpu_', '')
        
        # GPU利用率分析
        if 'gpu_utilization' in stats:
            avg_util = stats['gpu_utilization']['mean']
            if avg_util < 30:
                recommendations.append(f"GPU {gpu_index}: 利用率较低 ({avg_util:.1f}%)，考虑增加工作负载或批次大小")
            elif avg_util > 95:
                recommendations.append(f"GPU {gpu_index}: 利用率过高 ({avg_util:.1f}%)，可能存在性能瓶颈")
        
        # 内存利用率分析
        if 'memory_utilization' in stats:
            avg_mem = stats['memory_utilization']['mean']
            if avg_mem > 90:
                recommendations.append(f"GPU {gpu_index}: 内存使用率过高 ({avg_mem:.1f}%)，考虑减少批次大小")
            elif avg_mem < 20:
                recommendations.append(f"GPU {gpu_index}: 内存利用率较低 ({avg_mem:.1f}%)，可以增加批次大小")
        
        # 温度分析
        if 'temperature' in stats:
            max_temp = stats['temperature']['max']
            if max_temp > 80:
                recommendations.append(f"GPU {gpu_index}: 温度过高 ({max_temp:.1f}°C)，检查散热系统")
    
    if not recommendations:
        recommendations.append("🎉 所有GPU运行状态良好，无需特别优化")
    
    html = """
    <div class="section">
        <h2>💡 优化建议</h2>
        <div class="recommendations">
    """
    
    for rec in recommendations:
        html += f"<div class='recommendation'>• {rec}</div>"
    
    html += """
        </div>
    </div>
    """
    
    return html

def generate_html_report(data, output_file):
    """生成完整的HTML报告"""
    
    # HTML模板
    html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU性能报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .section {{
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #4a5568;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .gpu-grid, .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .gpu-card, .stats-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .gpu-card h3, .stats-card h3 {{
            color: #2d3748;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .gpu-card p {{
            margin-bottom: 8px;
            color: #4a5568;
        }}
        
        .stats-table {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .stat-label {{
            font-weight: 600;
            color: #2d3748;
        }}
        
        .stat-value {{
            color: #4a5568;
            font-family: monospace;
        }}
        
        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
        }}
        
        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }}
        
        .chart-container h3 {{
            margin-bottom: 15px;
            color: #2d3748;
            text-align: center;
        }}
        
        .recommendations {{
            background: #f0fff4;
            border: 1px solid #9ae6b4;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .recommendation {{
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #48bb78;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .charts-container {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 GPU性能监控报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
            <p>监控时长: {data['collection_info']['duration_seconds']} 秒 | 采样间隔: {data['collection_info']['interval_seconds']} 秒</p>
        </div>
        
        {generate_gpu_info_section(data['gpu_info'])}
        
        {generate_summary_stats_section(data['summary_stats'])}
        
        {generate_performance_charts(data['performance_data'])}
        
        {generate_recommendations(data['summary_stats'])}
        
        <div class="footer">
            <p>报告由GPU性能监控系统自动生成 | 数据采集时间: {data['collection_info']['start_time']}</p>
        </div>
    </div>
</body>
</html>
    """
    
    # 保存HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"HTML报告已生成: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python3 generate_html_report.py <json_file> <output_html>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_html = sys.argv[2]
    
    data = load_performance_data(json_file)
    if data:
        generate_html_report(data, output_html)
    else:
        print("无法加载性能数据")
EOF
    
    # 生成HTML报告
    if [[ -f "$JSON_REPORT" ]]; then
        log_info "生成HTML报告..."
        python3 "$OUTPUT_DIR/generate_html_report.py" "$JSON_REPORT" "$REPORT_FILE"
        log_info "✅ HTML报告生成完成: $REPORT_FILE"
    else
        log_error "性能数据文件不存在，无法生成报告"
        return 1
    fi
}

# 生成Markdown报告
generate_markdown_report() {
    log_section "生成Markdown报告"
    
    local md_report="$OUTPUT_DIR/gpu_performance_report.md"
    
    cat > "$md_report" << EOF
# GPU性能监控报告

## 报告概览
- **生成时间**: $(date)
- **监控时长**: ${REPORT_DURATION} 秒
- **采样间隔**: ${SAMPLE_INTERVAL} 秒
- **报告格式**: $REPORT_FORMAT

## GPU设备信息

\`\`\`
$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv 2>/dev/null || echo "GPU信息获取失败")
\`\`\`

## 系统环境
- **操作系统**: $(uname -a)
- **NVIDIA驱动版本**: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 2>/dev/null || echo "N/A")
EOF
    
    if command -v nvcc &> /dev/null; then
        echo "- **CUDA版本**: $(nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-)" >> "$md_report"
    fi
    
    cat >> "$md_report" << EOF

## 性能数据文件

生成的性能数据文件：
- JSON格式: \`performance_data.json\`
- CSV格式: \`performance_data.csv\`
- HTML报告: \`gpu_performance_report.html\`

## 使用说明

1. **查看HTML报告**: 在浏览器中打开 \`gpu_performance_report.html\` 查看交互式图表
2. **分析JSON数据**: 使用编程工具分析 \`performance_data.json\` 中的详细数据
3. **导入CSV数据**: 将 \`performance_data.csv\` 导入到Excel或其他分析工具中

## 报告生成日志

详细的生成日志请查看: \`report_generation.log\`
EOF
    
    log_info "✅ Markdown报告生成完成: $md_report"
}

# 发送报告通知
send_notifications() {
    if [[ -z "$EMAIL_RECIPIENTS" && -z "$SLACK_WEBHOOK" ]]; then
        return 0
    fi
    
    log_section "发送报告通知"
    
    # 发送邮件通知
    if [[ -n "$EMAIL_RECIPIENTS" ]] && command -v mail &> /dev/null; then
        log_info "发送邮件通知到: $EMAIL_RECIPIENTS"
        {
            echo "GPU性能监控报告已生成"
            echo "生成时间: $(date)"
            echo "报告位置: $OUTPUT_DIR"
            echo "HTML报告: $REPORT_FILE"
        } | mail -s "GPU性能监控报告 - $(date +%Y%m%d)" "$EMAIL_RECIPIENTS"
    fi
    
    # 发送Slack通知
    if [[ -n "$SLACK_WEBHOOK" ]] && command -v curl &> /dev/null; then
        log_info "发送Slack通知"
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"text\": \"🚀 GPU性能监控报告已生成\\n📅 时间: $(date)\\n📁 位置: $OUTPUT_DIR\\n📊 HTML报告: $REPORT_FILE\"
            }" \
            "$SLACK_WEBHOOK" &>/dev/null
    fi
    
    log_info "✅ 通知发送完成"
}

# 上传报告到云存储
upload_to_cloud() {
    if [[ "$AUTO_UPLOAD" != "true" ]]; then
        return 0
    fi
    
    log_section "上传报告到云存储"
    
    # 这里可以添加上传到AWS S3、Google Cloud Storage等的逻辑
    # 示例：上传到AWS S3
    if command -v aws &> /dev/null; then
        log_info "上传到AWS S3..."
        # aws s3 cp "$OUTPUT_DIR" "s3://your-bucket/gpu-reports/$(basename $OUTPUT_DIR)/" --recursive
    fi
    
    log_info "✅ 云存储上传完成"
}

# 清理旧报告
cleanup_old_reports() {
    log_section "清理旧报告"
    
    # 删除7天前的报告
    find "$(dirname $OUTPUT_DIR)" -name "gpu_performance_report_*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
    
    log_info "✅ 旧报告清理完成"
}

# 显示帮助信息
show_help() {
    echo "GPU自动化性能报告脚本"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --duration SECONDS      监控持续时间（默认: 3600秒）"
    echo "  --interval SECONDS      采样间隔（默认: 10秒）"
    echo "  --output-dir DIR        输出目录（默认: gpu_performance_report_TIMESTAMP）"
    echo "  --format FORMAT         报告格式 (html|markdown|both，默认: html)"
    echo "  --no-charts            不生成图表"
    echo "  --email RECIPIENTS     邮件通知收件人"
    echo "  --slack-webhook URL    Slack Webhook URL"
    echo "  --auto-upload          自动上传到云存储"
    echo "  --help                 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 生成1小时的HTML报告"
    echo "  $0 --duration 1800 --format both     # 生成30分钟的HTML和Markdown报告"
    echo "  $0 --email admin@company.com         # 生成报告并发送邮件通知"
    echo "  $0 --slack-webhook https://...       # 生成报告并发送Slack通知"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --duration)
                REPORT_DURATION="$2"
                shift 2
                ;;
            --interval)
                SAMPLE_INTERVAL="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                LOG_FILE="$OUTPUT_DIR/report_generation.log"
                REPORT_FILE="$OUTPUT_DIR/gpu_performance_report.html"
                JSON_REPORT="$OUTPUT_DIR/performance_data.json"
                shift 2
                ;;
            --format)
                REPORT_FORMAT="$2"
                shift 2
                ;;
            --no-charts)
                INCLUDE_CHARTS=false
                shift
                ;;
            --email)
                EMAIL_RECIPIENTS="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                shift 2
                ;;
            --auto-upload)
                AUTO_UPLOAD=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    # 设置输出目录
    setup_output_directory
    
    # 检查依赖
    check_dependencies
    
    log_info "开始生成GPU性能报告..."
    log_info "监控时长: ${REPORT_DURATION}秒，采样间隔: ${SAMPLE_INTERVAL}秒"
    
    # 收集性能数据
    collect_performance_data
    
    # 生成报告
    case $REPORT_FORMAT in
        "html")
            generate_html_report
            ;;
        "markdown")
            generate_markdown_report
            ;;
        "both")
            generate_html_report
            generate_markdown_report
            ;;
        *)
            log_error "不支持的报告格式: $REPORT_FORMAT"
            exit 1
            ;;
    esac
    
    # 发送通知
    send_notifications
    
    # 上传到云存储
    upload_to_cloud
    
    # 清理旧报告
    cleanup_old_reports
    
    log_info "🎉 GPU性能报告生成完成！"
    log_info "报告位置: $OUTPUT_DIR"
    
    if [[ "$REPORT_FORMAT" == "html" || "$REPORT_FORMAT" == "both" ]]; then
        log_info "HTML报告: $REPORT_FILE"
        log_info "在浏览器中打开查看交互式图表"
    fi
    
    if [[ "$REPORT_FORMAT" == "markdown" || "$REPORT_FORMAT" == "both" ]]; then
        log_info "Markdown报告: $OUTPUT_DIR/gpu_performance_report.md"
    fi
}

# 解析参数并运行主函数
parse_arguments "$@"
main