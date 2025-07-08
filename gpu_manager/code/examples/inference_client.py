# inference_client.py
import tritonclient.http as httpclient
import numpy as np

def run_inference():
    # 连接到HAMi管理的推理服务
    client = httpclient.InferenceServerClient(url="model-serving-service:8000")
    
    # 准备输入数据
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    
    # 执行推理（HAMi自动进行资源调度）
    outputs = [httpclient.InferRequestedOutput("output")]
    response = client.infer("resnet50", inputs, outputs=outputs)
    
    # 获取结果
    result = response.as_numpy("output")
    return result

if __name__ == '__main__':
    result = run_inference()
    print(f"Inference result shape: {result.shape}")