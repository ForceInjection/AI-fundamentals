#!/bin/bash

# 批量为所有HTML页面添加PPT模式支持
# 此脚本会在每个page_*.html文件的</body>标签前添加ppt-mode.js引用

echo "开始为所有页面添加PPT模式支持..."

# 获取当前目录
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 计数器
count=0

# 遍历所有page_*.html文件
for file in "$CURRENT_DIR"/page_*.html; do
    if [ -f "$file" ]; then
        # 检查文件是否已经包含ppt-mode.js引用
        if ! grep -q "ppt-mode.js" "$file"; then
            echo "正在处理: $(basename "$file")"
            
            # 使用sed在</body>标签前添加script引用
            # 创建临时文件
            temp_file="${file}.tmp"
            
            # 在</body>前添加script标签
            sed 's|</body>|<script src="ppt-mode.js"></script>\n</body>|g' "$file" > "$temp_file"
            
            # 替换原文件
            mv "$temp_file" "$file"
            
            count=$((count + 1))
        else
            echo "跳过 $(basename "$file") - 已包含PPT模式支持"
        fi
    fi
done

echo "完成！共处理了 $count 个文件。"
echo "所有页面现在都支持PPT模式了！"
echo ""
echo "使用说明："
echo "1. 打开任意页面，按F5键进入PPT模式"
echo "2. 使用方向键或空格键切换页面"
echo "3. 按O键查看页面概览"
echo "4. 按F11键切换全屏"
echo "5. 按ESC键退出PPT模式"
echo ""
echo "或者直接打开 index.html 开始演示！"