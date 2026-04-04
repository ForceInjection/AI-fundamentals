#!/bin/bash
# 多轮指代消解对话系统启动脚本
# Multi-turn Coreference Resolution Dialogue System Startup Script

echo "🚀 启动多轮指代消解对话系统..."
echo "Starting Multi-turn Coreference Resolution Dialogue System..."

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行 setup.sh"
    echo "❌ Virtual environment not found, please run setup.sh first"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

echo "✅ 虚拟环境已激活"
echo "✅ Virtual environment activated"

# 检查参数
if [ "$1" = "test" ]; then
    echo "🧪 运行系统测试..."
    echo "🧪 Running system tests..."
    python -c "
import asyncio
from example_usage import IntegratedDialogueSystem

async def test_dialogue():
    system = IntegratedDialogueSystem()
    print('🚀 开始对话测试...')
    
    # 测试对话1
    response1 = await system.process_user_input('你好，我是张三')
    print(f'响应1: {response1}')
    
    # 测试对话2 - 指代消解
    response2 = await system.process_user_input('他在哪里工作？')
    print(f'响应2: {response2}')
    
    # 获取最终统计
    stats = system.get_system_stats()
    print(f'\\n📊 测试完成统计:')
    print(f'- 对话轮次: {stats[\"conversation_turns\"]}')
    print(f'- 当前轮次ID: {stats[\"current_turn_id\"]}')
    
    return True

# 运行测试
try:
    result = asyncio.run(test_dialogue())
    print('\\n✅ 对话系统测试成功！')
except Exception as e:
    print(f'❌ 对话测试失败: {e}')
    import traceback
    traceback.print_exc()
"
elif [ "$1" = "demo" ]; then
    echo "🎯 运行演示程序..."
    echo "🎯 Running demo program..."
    python example_usage.py
elif [ "$1" = "api" ]; then
    echo "🌐 启动API服务器..."
    echo "🌐 Starting API server..."
    python system_integration.py
else
    echo "📖 使用说明:"
    echo "📖 Usage:"
    echo "  ./run_system.sh test    # 运行系统测试"
    echo "  ./run_system.sh demo    # 运行演示程序"
    echo "  ./run_system.sh api     # 启动API服务器"
    echo ""
    echo "💡 或者直接激活虚拟环境:"
    echo "💡 Or activate virtual environment directly:"
    echo "  source venv/bin/activate"
fi