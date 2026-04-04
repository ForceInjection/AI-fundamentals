#!/bin/bash
# 多轮指代消解对话系统环境设置脚本
# Multi-turn Coreference Resolution Dialogue System Setup Script

echo "🔧 设置多轮指代消解对话系统环境..."
echo "🔧 Setting up Multi-turn Coreference Resolution Dialogue System environment..."

# 检查Python版本
echo "📋 检查Python版本..."
echo "📋 Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python3 未安装，请先安装Python 3.8+"
    echo "❌ Python3 not installed, please install Python 3.8+ first"
    exit 1
fi

# 创建虚拟环境
echo "🏗️ 创建虚拟环境..."
echo "🏗️ Creating virtual environment..."
python3 -m venv venv --clear

if [ $? -ne 0 ]; then
    echo "❌ 虚拟环境创建失败"
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✅ 虚拟环境创建成功"
echo "✅ Virtual environment created successfully"

# 激活虚拟环境
echo "🔌 激活虚拟环境..."
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
echo "⬆️ Upgrading pip..."
./venv/bin/pip install --upgrade pip

# 安装依赖包
echo "📦 安装依赖包..."
echo "📦 Installing dependencies..."

if [ -f "requirements_simple.txt" ]; then
    echo "使用简化版依赖文件 requirements_simple.txt"
    echo "Using simplified requirements file requirements_simple.txt"
    ./venv/bin/pip install -r requirements_simple.txt
else
    echo "使用标准依赖文件 requirements.txt"
    echo "Using standard requirements file requirements.txt"
    ./venv/bin/pip install -r requirements.txt
fi

if [ $? -ne 0 ]; then
    echo "⚠️ 部分依赖包安装失败，但核心功能可能仍然可用"
    echo "⚠️ Some dependencies failed to install, but core functionality may still work"
else
    echo "✅ 依赖包安装成功"
    echo "✅ Dependencies installed successfully"
fi

# 测试核心模块
echo "🧪 测试核心模块导入..."
echo "🧪 Testing core module imports..."
./venv/bin/python -c "
try:
    from performance_optimization import PerformanceOptimizer
    from memory_management import MemoryManager
    from multimodal_coref import MultimodalCoreferenceResolver
    from example_usage import IntegratedDialogueSystem
    print('✅ 所有核心模块导入成功')
    print('✅ All core modules imported successfully')
except Exception as e:
    print('❌ 模块导入失败:', str(e))
    print('❌ Module import failed:', str(e))
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 环境设置完成！"
    echo "🎉 Environment setup completed!"
    echo ""
    echo "📖 使用说明:"
    echo "📖 Usage:"
    echo "  ./run_system.sh test    # 运行系统测试"
    echo "  ./run_system.sh demo    # 运行演示程序"
    echo "  ./run_system.sh api     # 启动API服务器"
    echo ""
    echo "💡 或者手动激活虚拟环境:"
    echo "💡 Or manually activate virtual environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "📝 注意: 如需使用OpenAI API，请设置环境变量:"
    echo "📝 Note: To use OpenAI API, please set environment variable:"
    echo "  export OPENAI_API_KEY='your-api-key'"
else
    echo "❌ 环境设置失败"
    echo "❌ Environment setup failed"
    exit 1
fi