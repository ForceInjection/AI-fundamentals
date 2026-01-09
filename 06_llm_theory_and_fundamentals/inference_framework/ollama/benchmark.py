import time
import ollama

# 配置参数
model = "deepseek-r1:32b"
prompt = "请介绍一下中国的历史人物屈原。"
num_trials = 3  # 运行次数

print("🔥 预热模型中，请稍候...")
ollama.chat(model=model, messages=[{"role": "user", "content": "你好"}])  # 预热
print("✅ 预热完成，开始正式推理...\n")

# 存储 token 速率
speeds = []

for i in range(1, num_trials + 1):
    print(f"🚀 运行第 {i} 轮推理...")

    # 记录开始时间
    start_time = time.time()

    # 进行推理，**禁用流式输出**
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=False)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 提取统计数据
    eval_count = response.get("eval_count", 0)  # 生成的 Token 数
    eval_duration = response.get("eval_duration", 1) / 1e9  # 生成时间（秒）
    prompt_eval_count = response.get("prompt_eval_count", 0)  # 提示词 Token 数
    prompt_eval_duration = response.get("prompt_eval_duration", 1) / 1e9  # 提示词处理时间（秒）
    total_duration = response.get("total_duration", 1) / 1e9  # 总耗时（秒）

    # 计算 Token 速率
    speed = eval_count / eval_duration if eval_duration > 0 else 0
    speeds.append(speed)

    print(f"📜 提示词 Token 数: {prompt_eval_count}, 处理耗时: {prompt_eval_duration:.2f} 秒")
    print(f"📜 生成 Token 数: {eval_count}, 生成耗时: {eval_duration:.2f} 秒")
    print(f"⏱️  总推理时间: {total_duration:.2f} 秒")
    print(f"⚡ Token 速率: {speed:.2f} tokens/sec")
    print("📢 **模型输出:**")
    print(response.get("message", {}).get("content", ""))  # 输出完整文本
    print("-" * 80 + "\n")

# 计算平均速度
avg_speed = sum(speeds) / len(speeds)
print(f"🏆 平均 Token 速率: {avg_speed:.2f} tokens/sec")