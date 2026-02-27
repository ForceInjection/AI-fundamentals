# Qwen2-VL-7B-Instruct 昇腾部署指南

## 1. 背景

### 1.1 模型简介

Qwen2-VL-7B-Instruct 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM），可以以图像、文本、视频作为输入，并以文本作为输出。该模型支持多种分辨率和宽高比的图像理解，以及超过20分钟的视频理解能力。

### 1.2 版本兼容性

| 组件       | 版本要求            |
| ---------- | ------------------- |
| MindIE     | 1.0.0+              |
| CANN       | 8.0.RC1+            |
| Atlas 硬件 | 800I A2 32G/64G     |
| 操作系统   | OpenEuler 24.03 LTS |
| Python     | 3.11+               |

---

## 2. 环境准备

### 2.1 前置条件检查

在开始部署前，请确认以下条件：

1. **硬件检查**：确认 Atlas 800I A2 设备正常工作

   ```bash
   npu-smi info
   ```

2. **驱动检查**：确认昇腾驱动已正确安装

   ```bash
   cat /usr/local/Ascend/driver/version.info
   ```

3. **Docker 环境**：确认 Docker 已安装并可正常使用

### 2.2 模型准备

目前提供的MindIE镜像预置了 Qwen2-VL-7B-Instruct 模型推理脚本，无需使用本仓库自带的atb_models中的代码。

### 2.3 镜像下载与加载

前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：`1.0.0-800I-A2-py311-openeuler24.03-lts`

完成之后，请使用`docker images`命令确认查找具体镜像名称与标签。

---

## 3. 硬件要求

### 3.1 Atlas 800I A2 32G 配置

- **最低要求**：1台 Atlas 800I A2 32G 服务器
- **推荐配置**：8卡并行部署
- **内存要求**：至少 64GB 系统内存
- **存储要求**：至少 50GB 可用存储空间

### 3.2 Atlas 800I A2 64G 配置

- **最低要求**：1台 Atlas 800I A2 64G 服务器
- **推荐配置**：4卡或8卡并行部署
- **内存要求**：至少 128GB 系统内存
- **存储要求**：至少 50GB 可用存储空间

---

## 4. 容器部署

### 4.1 创建容器

#### 4.1.1 基础启动命令

```bash
docker run -dit -u root \
--name ${容器名} \
-e ASCEND_RUNTIME_OPTIONS=NODRV \
--privileged=true \
-v /home/path:/home/path \
-v /data:/data \
-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
-v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
-v /usr/local/sbin/:/usr/local/sbin \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/dcmi:/usr/local/dcmi \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--shm-size=100g \
-p ${映射端口}:22 \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
${MindIE镜像名称:标签} \
/bin/bash
```

#### 4.1.2 参数说明

| 参数                     | 说明                     | 示例值                                                            |
| ------------------------ | ------------------------ | ----------------------------------------------------------------- |
| `${容器名}`              | 自定义容器名称           | `qwen2-vl-container`                                              |
| `/home/路径`             | 宿主机数据目录路径       | `/home/user/data`                                                 |
| `${映射端口}`            | SSH端口映射              | `2222`                                                            |
| `${MindIE镜像名称:标签}` | 下载的MindIE镜像完整名称 | `ascendhub.huawei.com/xxx:1.0.0-800I-A2-py311-openeuler24.03-lts` |

#### 4.1.3 启动示例

```bash
docker run -dit -u root \
--name qwen2-vl-container \
-e ASCEND_RUNTIME_OPTIONS=NODRV \
--privileged=true \
-v /home/user/models:/home/user/models \
-v /data:/data \
-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
-v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
-v /usr/local/sbin/:/usr/local/sbin \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/dcmi:/usr/local/dcmi \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--shm-size=100g \
-p 2222:22 \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
your-mindie-image:1.0.0-800I-A2-py311-openeuler24.03-lts \
/bin/bash
```

### 4.2 进入容器并安装依赖

#### 4.2.1 进入容器

```bash
docker exec -it ${容器名} bash
```

#### 4.2.2 安装Python依赖

```bash
cd /usr/local/Ascend/atb-models
pip install -r requirements/models/requirements_qwen2_vl.txt
```

---

## 5. 纯模型推理

### 5.1 配置推理脚本

修改 `/usr/local/Ascend/atb-models/examples/models/qwen2_vl/run_pa.sh` 脚本中的关键参数：

#### 5.1.1 Atlas 800I A2 32G 配置

```bash
# 设置卡数，Atlas-800I-A2-32G必须八卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 模型权重路径（请确保路径存在且包含完整的模型文件）
model_path="/data/Qwen2-VL-7B-Instruct/"

# 批次大小，底层使用continuous batching逻辑
max_batch_size=1

# 最大输入长度，输入长视频或者较大分辨率图片时，需要设置较大的值
# kv cache会根据最大输入长度、最大输出长度以及bs进行预分配，设置太大会影响吞吐
max_input_length=8192

# 最大输出长度
max_output_length=80

# 输入图片或视频文件路径（支持格式：jpg/png/jpeg/mp4/wmv/avi）
input_image="/data/test_images/sample.jpg"

# 用户prompt，默认放置在图片后
input_text="Explain the details in the image."

# 数据集路径（优先级比input_image高）
# 若要推理整个数据集，在base_cmd入参中添加 --dataset_path $dataset_path
dataset_path="/data/test_images"

# 共享内存name保存路径，任意位置的一个txt即可
shm_name_save_path="./shm_name.txt"
```

#### 5.1.2 Atlas 800I A2 64G 配置

```bash
# 设置卡数，Atlas-800I-A2-64G四卡或八卡均可
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 8卡配置
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3        # 4卡配置

# 其他参数可根据需要调整batch_size等
max_batch_size=4  # 64G版本可以设置更大的batch_size
```

#### 5.1.3 参数详细说明

| 参数名              | 说明              | 推荐值              | 注意事项               |
| ------------------- | ----------------- | ------------------- | ---------------------- |
| `max_batch_size`    | 批处理大小        | 32G: 1-4；64G: 4-32 | 影响内存使用和吞吐量   |
| `max_input_length`  | 最大输入token长度 | 8192                | 处理长视频时可适当增加 |
| `max_output_length` | 最大输出token长度 | 80-512              | 根据应用需求调整       |
| `input_image`       | 输入文件路径      | 绝对路径            | 确保文件存在且格式支持 |

### 5.2 运行推理

```bash
bash /usr/local/Ascend/atb-models/examples/models/qwen2_vl/run_pa.sh
```

### 5.3 性能基准测试

#### 5.3.1 Atlas 800I A2 32G 性能数据

**测试配置**：

- `max_batch_size=4`
- `max_input_length=8192`
- `max_output_length=80`
- `input_image="/data/test_images/1902x1080.jpg"`

**测试结果**：

- 总token数：320
- 总耗时：7.44秒
- **吞吐量：43 tokens/s**

#### 5.3.2 Atlas 800I A2 64G 性能数据

**测试配置**：

- `max_batch_size=32`
- `max_input_length=8192`
- `max_output_length=80`
- `input_image="/data/test_images/1902x1080.jpg"`

**测试结果**：

- 总token数：2560
- 总耗时：25.912秒
- **吞吐量：98.79 tokens/s**

> **注意**：更详细的性能数据（如首token时延、内存使用等）请参考终端performance输出。实际性能可能因硬件配置、模型版本、输入数据等因素而有所差异。

---

## 6. 服务化推理

### 6.1 配置服务

#### 6.1.1 编辑配置文件

```bash
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

#### 6.1.2 关键配置项说明

```json
{
  "Version": "1.0.0",
  "LogConfig": {
    "logLevel": "Info",
    "logFileSize": 20,
    "logFileNum": 20,
    "logPath": "logs/mindservice.log"
  },
  "ServerConfig": {
    "ipAddress": "127.0.0.1",
    "port": 1040, // 主服务端口，可自定义
    "managementPort": 1041, // 管理端口，可自定义
    "metricsPort": 1042, // 监控端口，可自定义
    "maxLinkNum": 200,
    "httpsEnabled": false,
    "timeoutInSeconds": 300
  },
  "BackendConfig": {
    "npuDeviceIds": [[0, 1, 2, 3, 4, 5, 6, 7]], // NPU设备ID配置
    "ModelDeployConfig": {
      "maxSeqLen": 50000,
      "maxInputTokenLen": 50000,
      "truncation": false,
      "ModelConfig": [
        {
          "modelInstanceType": "Standard",
          "modelName": "qwen2_vl", // 建议使用qwen2_vl便于benchmark测试
          "modelWeightPath": "/data/Qwen2-VL-7B-Instruct", // 模型权重路径
          "worldSize": 8, // 并行度，需与npuDeviceIds数量一致
          "npuMemSize": 8, // KV cache分配，单位GB，需给VIT预留显存空间
          "maxBatchSize": 32,
          "maxInputTokens": 8192,
          "maxOutputTokens": 512
        }
      ]
    },
    "ScheduleConfig": {
      "maxPrefillTokens": 50000,
      "maxIterTimes": 4096,
      "maxBatchSize": 32,
      "maxWaitingTimeInMs": 100
    }
  }
}
```

#### 6.1.3 配置参数优化建议

| 硬件配置          | npuMemSize | maxBatchSize | worldSize | 说明                 |
| ----------------- | ---------- | ------------ | --------- | -------------------- |
| Atlas 800I A2 32G | 8-10       | 4-8          | 8         | 保守配置，稳定性优先 |
| Atlas 800I A2 64G | 16-20      | 16-32        | 8         | 性能配置，吞吐量优先 |

> **重要提示**：`npuMemSize` 切勿设置为 -1，需要给 VIT（视觉编码器）预留足够的显存空间。

### 6.2 启动服务

#### 6.2.1 设置环境变量

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=7896
```

#### 6.2.2 启动服务进程

```bash
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

### 6.2.3 检查服务状态

```bash
# 检查服务是否正常启动
curl http://127.0.0.1:1041/health

# 查看服务日志
tail -f /usr/local/Ascend/mindie/latest/mindie-service/logs/mindservice.log
```

### 6.3 API 接口测试

#### 6.3.1 VLLM 兼容接口

```bash
curl -X POST http://127.0.0.1:1040/generate \
-H "Content-Type: application/json" \
-d '{
  "prompt": [
    {
      "type": "image_url",
      "image_url": "file:///data/test_images/sample.jpg"
    },
    {
      "type": "text",
      "text": "Explain the details in the image."
    }
  ],
  "max_tokens": 512,
  "stream": false,
  "do_sample": true,
  "repetition_penalty": 1.00,
  "temperature": 0.01,
  "top_p": 0.001,
  "top_k": 1,
  "model": "qwen2_vl"
}'
```

#### 6.3.2 OpenAI 兼容接口

```bash
curl -X POST http://127.0.0.1:1040/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "qwen2_vl",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "file:///data/test_images/sample.jpg"
          }
        },
        {
          "type": "text",
          "text": "Explain the details in the image."
        }
      ]
    }
  ],
  "max_tokens": 512,
  "stream": false,
  "do_sample": true,
  "repetition_penalty": 1.00,
  "temperature": 0.01,
  "top_p": 0.001,
  "top_k": 1
}'
```

#### 6.3.3 流式输出示例

```bash
curl -X POST http://127.0.0.1:1040/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "qwen2_vl",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "描述这张图片的内容"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "file:///data/test_images/sample.jpg"
          }
        }
      ]
    }
  ],
  "max_tokens": 512,
  "stream": true,
  "temperature": 0.7
}'
```

---

## 7. 故障排除

### 7.1 常见问题及解决方案

1. **服务启动失败**
   - 检查NPU设备状态：`npu-smi info`
   - 检查端口占用：`netstat -tlnp | grep 1040`
   - 查看详细日志：`tail -f logs/mindservice.log`

2. **内存不足错误**
   - 降低 `npuMemSize` 参数
   - 减少 `maxBatchSize` 设置
   - 检查模型文件完整性

3. **推理速度慢**
   - 调整 `maxBatchSize` 参数
   - 优化 `maxInputTokenLen` 设置
   - 检查硬件资源使用情况

4. **图片加载失败**
   - 确认图片路径正确且可访问
   - 检查图片格式是否支持
   - 验证文件权限设置
