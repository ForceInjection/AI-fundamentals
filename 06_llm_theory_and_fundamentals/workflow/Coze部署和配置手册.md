# Coze 部署和配置手册

## 1. 系统概述与架构

### 1.1 平台概述

Coze Studio 是一个基于 Docker 容器化部署的 AI 应用开发平台，采用微服务架构设计。本手册详细介绍了 Coze Studio 的部署架构、组件配置和部署步骤。

### 1.2 系统架构

Coze Studio 采用分层架构设计，包含以下核心组件：

#### 1.2.1 应用层

- **coze-web**: 前端 Web 应用，基于 Nginx 提供静态资源服务
- **coze-server**: 后端 API 服务，提供业务逻辑处理

#### 1.2.2 数据存储层

- **MySQL**: 关系型数据库，存储业务数据
- **Redis**: 缓存数据库，提供高性能缓存服务
- **Elasticsearch**: 搜索引擎，提供全文检索功能
- **MinIO**: 对象存储服务，存储文件和媒体资源
- **Milvus**: 向量数据库，支持 AI 向量检索

#### 1.2.3 基础设施层

- **etcd**: 分布式键值存储，提供服务发现和配置管理
- **NSQ**: 消息队列系统，包含 `nsqlookupd`、`nsqd`、`nsqadmin` 三个组件

### 1.3 系统组件详细配置

本节将基于 [Docker Compose 配置](../../08_agentic_system/multi_agent/multi_agent_system/docker-compose.yml) 对各组件的详细配置进行说明。

#### 1.3.1 MySQL 数据库

**镜像版本**: `mysql:8.4.5`

**端口配置**:

- 内部端口: 3306
- 外部端口: 未暴露（仅内网访问，端口已注释）

**环境变量**:

```bash
MYSQL_ROOT_PASSWORD=your_secure_password  # 管理员密码
MYSQL_DATABASE=opencoze           # 数据库名称
MYSQL_USER=coze                   # 业务用户名
MYSQL_PASSWORD=coze123            # 业务用户密码
```

**存储配置**:

- 数据目录: `./data/mysql:/var/lib/mysql`
- 初始化脚本: `./volumes/mysql/schema.sql`

**健康检查**:

- 检查间隔: 10 秒
- 超时时间: 5 秒
- 重试次数: 5 次
- 启动等待: 30 秒

#### 1.3.2 Redis 缓存

**镜像版本**: `bitnami/redis:8.0`

**端口配置**:

- 内部端口: 6379
- 外部端口: 未暴露（仅内网访问，端口已注释）

**环境变量**:

```bash
REDIS_AOF_ENABLED=no              # 关闭 AOF 持久化
REDIS_PORT_NUMBER=6379            # 服务端口
REDIS_IO_THREADS=4                # IO 线程数
ALLOW_EMPTY_PASSWORD=yes          # 允许空密码
```

**存储配置**:

- 数据目录: `./data/bitnami/redis:/bitnami/redis/data`

**健康检查**:

- 检查间隔: 5 秒
- 超时时间: 10 秒
- 重试次数: 10 次

#### 1.3.3 Elasticsearch 搜索引擎

**镜像版本**: `bitnami/elasticsearch:8.18.0`

**端口配置**:

- 内部端口: 9200
- 外部端口: 未暴露（仅内网访问，端口已注释）

**特殊配置**:

- 安装 `analysis-smartcn` 中文分词插件
- 自动初始化索引结构
- 支持自定义配置文件

**存储配置**:

- 数据目录: `./data/bitnami/elasticsearch:/bitnami/elasticsearch/data`
- 配置文件: `./volumes/elasticsearch/elasticsearch.yml`
- 插件包: `./volumes/elasticsearch/analysis-smartcn.zip`

#### 1.3.4 MinIO 对象存储

**镜像版本**: `minio/minio:RELEASE.2025-06-13T11-33-47Z-cpuv1`

**端口配置**:

- API 端口: 9000（内部，端口已注释）
- 控制台端口: 9001（内部，端口已注释）

**环境变量**:

```bash
MINIO_ROOT_USER=minioadmin        # 管理员用户名
MINIO_ROOT_PASSWORD=minioadmin123 # 管理员密码
MINIO_DEFAULT_BUCKETS=opencoze,milvus # 默认存储桶
```

**存储配置**:

- 数据目录: `./data/minio:/data`
- 默认图标: `./volumes/minio/default_icon/`
- 插件图标: `./volumes/minio/official_plugin_icon/`

#### 1.3.5 etcd 配置中心

**镜像版本**: `bitnami/etcd:3.5`

**端口配置**:

- 客户端端口: 2379（内部，端口已注释）
- 节点通信端口: 2380（内部，端口已注释）

**环境变量**:

```bash
ETCD_AUTO_COMPACTION_MODE=revision    # 自动压缩模式
ETCD_AUTO_COMPACTION_RETENTION=1000   # 保留版本数
ETCD_QUOTA_BACKEND_BYTES=4294967296   # 后端存储配额（4GB）
ALLOW_NONE_AUTHENTICATION=yes         # 允许无认证访问
```

#### 1.3.6 Milvus 向量数据库

**镜像版本**: `milvusdb/milvus:v2.5.10`

**端口配置**:

- 服务端口: 19530（内部，端口已注释）
- 健康检查端口: 9091（内部，端口已注释）

**依赖服务**:

- etcd（配置存储）
- MinIO（数据存储）

**环境变量**:

```bash
ETCD_ENDPOINTS=etcd:2379              # etcd 连接地址
MINIO_ADDRESS=minio:9000              # MinIO 连接地址
MINIO_BUCKET_NAME=milvus              # 存储桶名称
```

#### 1.3.7 NSQ 消息队列

**镜像版本**: `nsqio/nsq:v1.2.1`

**组件说明**:

1. **nsqlookupd**: 服务发现组件
   - 端口: 4160（TCP）、4161（HTTP）（端口已注释）
2. **nsqd**: 消息队列守护进程
   - 端口: 4150（TCP）、4151（HTTP）（端口已注释）
3. **nsqadmin**: Web 管理界面
   - 端口: 4171（HTTP）（端口已注释）

#### 1.3.8 Coze Server 后端服务

**镜像版本**: `cozedev/coze-studio-server:latest`

**端口配置**:

- API 端口: 8888（内部，端口已注释）
- 管理端口: 8889（内部，端口已注释）

**依赖服务**:

- MySQL（数据存储）
- Redis（缓存）
- Elasticsearch（搜索）
- MinIO（文件存储）
- Milvus（向量检索）

#### 1.3.9 Coze Web 前端服务

**镜像版本**: `cozedev/coze-studio-web:latest`

**端口配置**:

- HTTP 端口: 80（容器内部）
- 外部访问端口: 8888（可配置）

**配置文件**:

- Nginx 主配置: `./nginx/nginx.conf`
- 代理配置: `./nginx/conf.d/default.conf`

#### 1.3.10 网络拓扑

所有服务运行在同一个 Docker 网络 `coze-network` 中，采用桥接模式：

```text
┌─────────────────────────────────────────────────────────────┐
│                    coze-network (bridge)                    │
├─────────────────┬───────────────────────────────────────────┤
│   Frontend      │              Backend Services             │
├─────────────────┼───────────────────────────────────────────┤
│ coze-web:80     │ coze-server:8888                         │
│ (→ :8888)       │                                           │
├─────────────────┼───────────────────────────────────────────┤
│   Data Layer    │            Infrastructure                 │
├─────────────────┼───────────────────────────────────────────┤
│ mysql:3306      │ etcd:2379,2380                           │
│ redis:6379      │ nsqlookupd:4160,4161                     │
│ elasticsearch:  │ nsqd:4150,4151                           │
│   9200          │ nsqadmin:4171                            │
│ minio:9000,9001 │                                           │
│ milvus:19530,   │                                           │
│   9091          │                                           │
└─────────────────┴───────────────────────────────────────────┘
```

#### 1.3.11 端口列表总结

| 服务          | 内部端口    | 外部端口 | 协议     | 说明                 |
| ------------- | ----------- | -------- | -------- | -------------------- |
| coze-web      | 80          | 8888     | HTTP     | Web 前端访问         |
| coze-server   | 8888, 8889  | -        | HTTP     | API 服务（内网）     |
| mysql         | 3306        | -        | TCP      | 数据库（内网）       |
| redis         | 6379        | -        | TCP      | 缓存（内网）         |
| elasticsearch | 9200        | -        | HTTP     | 搜索引擎（内网）     |
| minio         | 9000, 9001  | -        | HTTP     | 对象存储（内网）     |
| etcd          | 2379, 2380  | -        | TCP      | 配置中心（内网）     |
| milvus        | 19530, 9091 | -        | TCP/HTTP | 向量数据库（内网）   |
| nsqlookupd    | 4160, 4161  | -        | TCP/HTTP | 消息队列发现（内网） |
| nsqd          | 4150, 4151  | -        | TCP/HTTP | 消息队列（内网）     |
| nsqadmin      | 4171        | -        | HTTP     | 消息队列管理（内网） |

---

## 2. 部署前准备

### 2.1 本地存储目录规划

在开始部署之前，我们需要创建以下本地存储目录用于数据持久化。

注意：所有目录都基于项目根目录(/)的相对路径。

#### 2.1.1 数据存储目录

```bash
# 创建数据存储根目录
mkdir -p ./data

# MySQL 数据库存储
mkdir -p ./data/mysql

# Redis 缓存数据存储
mkdir -p ./data/bitnami/redis

# Elasticsearch 搜索引擎数据存储
mkdir -p ./data/bitnami/elasticsearch

# MinIO 对象存储数据
mkdir -p ./data/minio
```

#### 2.1.2 配置文件目录

```bash
# 创建配置文件根目录
mkdir -p ./volumes

# MySQL 初始化脚本目录
mkdir -p ./volumes/mysql

# Elasticsearch 配置文件目录
mkdir -p ./volumes/elasticsearch

# MinIO 图标资源目录
mkdir -p ./volumes/minio/default_icon
mkdir -p ./volumes/minio/official_plugin_icon

# Nginx 配置文件目录
mkdir -p ./nginx/conf.d
```

#### 2.1.3 目录权限设置

```bash
# 设置数据目录权限（确保 Docker 容器可以读写）
sudo chown -R 1001:1001 ./data/bitnami
sudo chmod -R 755 ./data

# 设置配置文件目录权限
sudo chmod -R 755 ./volumes
sudo chmod -R 755 ./nginx
```

#### 2.1.4 目录结构总览

完成目录创建后，项目根目录下的存储结构如下：

```text
coze-studio/
├── data/                           # 数据持久化根目录
│   ├── mysql/                      # MySQL 数据库文件
│   ├── bitnami/
│   │   ├── redis/                  # Redis 缓存数据
│   │   └── elasticsearch/          # Elasticsearch 索引数据
│   └── minio/                      # MinIO 对象存储数据
├── volumes/                        # 配置文件根目录
│   ├── mysql/
│   │   └── schema.sql              # MySQL 初始化脚本
│   ├── elasticsearch/
│   │   ├── elasticsearch.yml       # Elasticsearch 配置
│   │   └── analysis-smartcn.zip    # 中文分词插件
│   └── minio/
│       ├── default_icon/           # 默认图标资源
│       └── official_plugin_icon/   # 官方插件图标
├── nginx/                          # Nginx 配置目录
│   ├── nginx.conf                  # 主配置文件
│   └── conf.d/
│       └── default.conf            # 代理配置文件
└── docker-compose.yml              # Docker Compose 配置文件
```

**注意事项**：

1. **磁盘空间**：确保 `./data` 目录所在磁盘有足够空间（建议至少 50GB）
2. **备份策略**：定期备份 `./data` 和 `./volumes` 目录
3. **权限管理**：避免使用 `root` 权限运行容器，确保目录权限正确设置
4. **路径一致性**：所有路径必须与 `docker-compose.yml` 中的卷映射配置保持一致

> 注意：真实部署的时候，需要考虑对数据盘做 RAID 或者进行数据备份，以免数据丢失。

---

## 3. 部署步骤

### 3.1 环境准备（单节点）

#### 3.1.1 系统要求

- 操作系统：Linux (Ubuntu 18.04+, CentOS 7+)
- CPU：4 核心以上
- 内存：8GB 以上（推荐 16GB）
- 磁盘：50GB 以上可用空间
- 网络：稳定的互联网连接

#### 3.1.2 安装 Docker 和 Docker Compose

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose-plugin

# CentOS/RHEL
sudo yum install docker docker-compose-plugin

# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到docker组
sudo usermod -aG docker $USER
newgrp docker
```

#### 3.1.3 配置系统参数

```bash
# 增加文件描述符限制
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 配置内核参数（Elasticsearch需要）
echo "vm.max_map_count=262144" >> /etc/sysctl.conf
sysctl -p

# 创建必要的目录
sudo mkdir -p /opt/coze-studio/{data,logs,config}
sudo chown -R $USER:$USER /opt/coze-studio
```

### 3.2 下载和配置

#### 3.2.1 获取项目文件

```bash
# 克隆项目代码
git clone https://github.com/coze-dev/coze-studio.git
cd coze-studio/docker

# 或者下载发布包
wget https://github.com/coze-dev/coze-studio/releases/latest/download/coze-studio-docker.tar.gz
tar -xzf coze-studio-docker.tar.gz
cd coze-studio-docker
```

#### 3.2.2 配置环境变量

```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件
vim .env

# 必须修改的配置项：
WEB_LISTEN_ADDR=0.0.0.0:8888  # 允许外部访问
MYSQL_ROOT_PASSWORD=your_secure_password
MINIO_ROOT_PASSWORD=your_secure_password
PLUGIN_AES_AUTH_SECRET=your_16_byte_secret
PLUGIN_AES_STATE_SECRET=your_16_byte_secret
PLUGIN_AES_OAUTH_TOKEN_SECRET=your_16_byte_secret
```

#### 3.2.3 准备初始化文件

```bash
# 检查必要的目录和文件
ls -la volumes/
ls -la volumes/mysql/
ls -la volumes/elasticsearch/
ls -la volumes/minio/

# 如果缺少初始化文件，创建基本结构
mkdir -p volumes/{mysql,elasticsearch,minio,nginx}
mkdir -p data/{mysql,redis,elasticsearch,minio}
```

### 3.3 启动服务

#### 3.3.1 分步启动（推荐）

```bash
# 第一步：启动基础设施服务
docker-compose up -d mysql redis etcd

# 等待服务就绪（约30-60秒）
docker-compose logs -f mysql
# 看到 "ready for connections" 后按 Ctrl+C 退出

# 第二步：启动存储和搜索服务
docker-compose up -d elasticsearch minio milvus

# 等待服务就绪（约60-120秒）
docker-compose ps

# 第三步：启动消息队列
docker-compose up -d nsqlookupd nsqd nsqadmin

# 第四步：启动应用服务
docker-compose up -d coze-server

# 等待后端服务就绪
docker-compose logs -f coze-server
# 看到服务启动成功日志后按 Ctrl+C 退出

# 第五步：启动前端服务
docker-compose up -d coze-web
```

#### 3.3.2 一键启动

```bash
# 启动所有服务
docker-compose up -d

# 查看启动进度
docker-compose logs -f
```

### 3.4 验证部署

#### 3.4.1 检查服务状态

```bash
# 查看所有服务状态
docker-compose ps

# 查看异常服务日志（如果有服务显示 "Exit" 或 "Restarting"）
docker-compose logs [service_name]
```

**服务状态说明**：`Up` = 正常运行，`Exit` = 异常退出，`Restarting` = 重启中

#### 3.4.2 健康检查

```bash
# 快速验证关键服务
docker-compose exec mysql mysql -u root -p -e "SELECT 1" && \
docker-compose exec redis redis-cli ping && \
curl -f http://localhost:8888/health && \
echo "✅ 所有服务正常"
```

详细的监控和故障排除请参考第 7 章。

#### 3.4.3 访问应用

```bash
# 检查Web服务
curl -I http://localhost:8888

# 在浏览器中访问
# http://your-server-ip:8888
```

### 3.5 初始化配置

#### 3.5.1 创建管理员账户

访问 `http://your-server-ip:8888` 进行首次设置：

- 设置管理员邮箱和密码
- 配置基本系统设置
- 添加 AI 模型配置

#### 3.5.2 配置 AI 模型

在管理界面中配置：

- 添加聊天模型（如 `GPT`、`Claude` 等）
- 配置嵌入模型
- 设置模型参数和限制

#### 3.5.3 测试功能

- 创建测试 Bot
- 上传测试文档到知识库
- 进行对话测试

### 3.6 多节点部署（生产环境）

**架构设计**:

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Application   │    │   Data Layer    │
│    (2-3 台)     │    │     Cluster     │    │    Cluster      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ Nginx/HAProxy   │───▶│ coze-web × 3+   │    │ MySQL Master    │
│ + Keepalived    │    │ coze-server × 3+│    │ MySQL Slaves×2+ │
│                 │    │                 │    │ Redis Cluster×3+│
│                 │    │                 │    │ ES Cluster × 3+ │
│                 │    │                 │    │ MinIO Cluster×4+│
│                 │    │                 │    │ Milvus Cluster×3│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**机器配置建议**:

1. **负载均衡层 (2-3 台)**

   - **配置**: 4C8G，50GB SSD
   - **组件**: Nginx/HAProxy + Keepalived
   - **说明**: 采用主备模式，确保高可用性

2. **应用集群层 (最少 3 台)**

   - **配置**: 8C16G，100GB SSD
   - **组件**: coze-web + coze-server
   - **扩展**: 根据并发量水平扩展

3. **数据层集群 (9-15 台)**

   **MySQL 集群 (3 台)**:

   - **Master**: 8C32G，500GB SSD
   - **Slave**: 8C16G，500GB SSD × 2

   **Redis 集群 (3 台)**:

   - **配置**: 4C16G，200GB SSD
   - **模式**: 3 主 3 从模式

   **Elasticsearch 集群 (3 台)**:

   - **配置**: 8C32G，1TB SSD
   - **角色**: Master + Data + Ingest

   **MinIO 集群 (4 台)**:

   - **配置**: 4C16G，2TB HDD
   - **模式**: 分布式模式，4 节点

   **Milvus 集群 (3 台)**:

   - **配置**: 8C32G，500GB SSD
   - **组件**: Query Node + Data Node + Index Node

**部署策略**:

1. **负载均衡层**

   - 使用 Nginx 或 HAProxy 进行负载均衡
   - 配置 SSL 终止和健康检查
   - 实现会话保持（如需要）

2. **应用层扩展**

   - coze-web: 无状态，可水平扩展
   - coze-server: 无状态，可水平扩展
   - 使用容器编排工具（Kubernetes/Docker Swarm）

3. **数据层高可用**
   - MySQL: 主从复制或 MySQL Cluster
   - Redis: Redis Cluster 或 Sentinel 模式
   - Elasticsearch: 多节点集群
   - MinIO: 分布式模式
   - Milvus: 集群模式
   - etcd: 3/5/7 节点集群

**配置示例**（Kubernetes）:

```yaml
# 应用层 Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coze-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: coze-server
  template:
    metadata:
      labels:
        app: coze-server
    spec:
      containers:
        - name: coze-server
          image: cozedev/coze-studio-server:latest
          ports:
            - containerPort: 8888
          env:
            - name: MYSQL_HOST
              value: "mysql-service"
            - name: REDIS_ADDR
              value: "redis-service:6379"
          # 其他环境变量...
```

---

## 4. 配置说明

### 4.1 基础服务配置

#### 4.1.1 服务器配置

```bash
# 服务器监听地址和端口
export LISTEN_ADDR=":8888"                    # 服务器监听端口
export LOG_LEVEL="debug"                      # 日志级别
export MAX_REQUEST_BODY_SIZE=1073741824       # 最大请求体大小（1GB）
export SERVER_HOST="http://localhost${LISTEN_ADDR}"  # 服务器主机地址
export USE_SSL="0"                            # 是否启用SSL
export SSL_CERT_FILE=""                       # SSL证书文件路径
export SSL_KEY_FILE=""                        # SSL私钥文件路径
```

#### 4.1.2 数据库配置

```bash
# MySQL 数据库配置
export MYSQL_ROOT_PASSWORD=root               # MySQL root 密码
export MYSQL_DATABASE=opencoze                # 数据库名称
export MYSQL_USER=coze                        # 数据库用户名
export MYSQL_PASSWORD=coze123                 # 数据库密码
export MYSQL_HOST=mysql                       # 数据库主机
export MYSQL_PORT=3306                        # 数据库端口
export MYSQL_DSN="${MYSQL_USER}:${MYSQL_PASSWORD}@tcp(${MYSQL_HOST}:${MYSQL_PORT})/${MYSQL_DATABASE}?charset=utf8mb4&parseTime=True"
export ATLAS_URL="mysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DATABASE}?charset=utf8mb4&parseTime=True"

# Redis 配置
export REDIS_AOF_ENABLED=no                   # AOF 持久化
export REDIS_IO_THREADS=4                     # IO 线程数
export ALLOW_EMPTY_PASSWORD=yes               # 允许空密码
export REDIS_ADDR="redis:6379"                # Redis 地址
export REDIS_PASSWORD=""                      # Redis 密码（可选）
```

#### 4.1.3 消息队列配置

```bash
# 消息队列类型配置
export COZE_MQ_TYPE="nsq"                     # nsq / kafka / rmq
export MQ_NAME_SERVER="nsqd:4150"             # NSQ 服务器地址

# RocketMQ 配置（如使用 RocketMQ）
export RMQ_ACCESS_KEY=""                      # RocketMQ 访问密钥
export RMQ_SECRET_KEY=""                      # RocketMQ 密钥
```

### 4.2 存储配置

#### 4.2.1 文件上传组件配置

```bash
# 文件上传组件类型
export FILE_UPLOAD_COMPONENT_TYPE="storage"   # storage / imagex
```

#### 4.2.2 火山引擎 ImageX 配置

```bash
# 火山引擎 ImageX（如使用）
export VE_IMAGEX_AK=""                        # ImageX Access Key
export VE_IMAGEX_SK=""                        # ImageX Secret Key
export VE_IMAGEX_SERVER_ID=""                 # ImageX 服务器ID
export VE_IMAGEX_DOMAIN=""                    # ImageX 域名
export VE_IMAGEX_TEMPLATE=""                  # ImageX 模板
export VE_IMAGEX_UPLOAD_HOST="https://imagex.volcengineapi.com"  # ImageX 上传地址
```

#### 4.2.3 对象存储配置

```bash
# 存储组件配置
export STORAGE_TYPE="minio"                   # minio / tos / s3
export STORAGE_UPLOAD_HTTP_SCHEME="http"      # http / https（如果网站使用https，必须设置为https）
export STORAGE_BUCKET="opencoze"              # 存储桶名称

# MinIO 配置
export MINIO_ROOT_USER=minioadmin             # MinIO 用户名
export MINIO_ROOT_PASSWORD=minioadmin123      # MinIO 密码
export MINIO_DEFAULT_BUCKETS=milvus           # 默认存储桶
export MINIO_AK=$MINIO_ROOT_USER              # MinIO Access Key
export MINIO_SK=$MINIO_ROOT_PASSWORD          # MinIO Secret Key
export MINIO_ENDPOINT="minio:9000"            # MinIO 端点
export MINIO_API_HOST="http://${MINIO_ENDPOINT}"  # MinIO API 地址

# TOS 配置（火山引擎对象存储）
export TOS_ACCESS_KEY=                        # TOS Access Key
export TOS_SECRET_KEY=                        # TOS Secret Key
export TOS_ENDPOINT=https://tos-cn-beijing.volces.com  # TOS 端点
export TOS_BUCKET_ENDPOINT=https://opencoze.tos-cn-beijing.volces.com  # TOS 存储桶端点
export TOS_REGION=cn-beijing                  # TOS 区域

# S3 配置（AWS S3 兼容存储）
export S3_ACCESS_KEY=                         # S3 Access Key
export S3_SECRET_KEY=                         # S3 Secret Key
export S3_ENDPOINT=                           # S3 端点
export S3_BUCKET_ENDPOINT=                    # S3 存储桶端点
export S3_REGION=                             # S3 区域
```

#### 4.2.4 Elasticsearch 配置

```bash
# Elasticsearch 搜索引擎配置
export ES_ADDR="http://elasticsearch:9200"    # Elasticsearch 地址
export ES_VERSION="v8"                        # Elasticsearch 版本
export ES_USERNAME=""                         # Elasticsearch 用户名（可选）
export ES_PASSWORD=""                         # Elasticsearch 密码（可选）
```

#### 4.2.5 向量存储配置

```bash
# 向量存储类型
export VECTOR_STORE_TYPE="milvus"             # milvus / vikingdb

# Milvus 向量数据库配置
export MILVUS_ADDR="milvus:19530"             # Milvus 地址
export MILVUS_USER=""                         # Milvus 用户名（可选）
export MILVUS_PASSWORD=""                     # Milvus 密码（可选）

# VikingDB 向量数据库配置（火山引擎）
export VIKING_DB_HOST=""                      # VikingDB 主机
export VIKING_DB_REGION=""                    # VikingDB 区域
export VIKING_DB_AK=""                        # VikingDB Access Key
export VIKING_DB_SK=""                        # VikingDB Secret Key
export VIKING_DB_SCHEME=""                    # VikingDB 协议
export VIKING_DB_MODEL_NAME=""                # VikingDB 模型名称（如未设置需配置嵌入模型）
```

### 4.3 AI 模型配置

`AI` 模型配置是 `Coze Studio` 的核心部分，包括**嵌入模型**、**聊天模型**和**内置聊天模型**的配置。

以下表格列出了 `Coze Studio` 中所有可能需要配置的模型类型：

| 模型类别         | 模型类型       | 支持的提供商                                | 主要用途                  | 配置前缀                                                                                           |
| ---------------- | -------------- | ------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------- |
| **嵌入模型**     | 文本嵌入       | Ark、OpenAI、Ollama、Gemini、HTTP           | 知识库向量化、文档检索    | `ARK_EMBEDDING_`、`OPENAI_EMBEDDING_`、`OLLAMA_EMBEDDING_`、`GEMINI_EMBEDDING_`、`HTTP_EMBEDDING_` |
| **聊天模型**     | 对话生成       | Ark、OpenAI、DeepSeek、Ollama、Qwen、Gemini | Agent 对话、Workflow 执行 | `MODEL_`                                                                                           |
| **内置聊天模型** | NL2SQL         | Ark、OpenAI、DeepSeek、Ollama、Qwen、Gemini | 自然语言转 SQL 查询       | `NL2SQL_BUILTIN_CM_`                                                                               |
|                  | 消息重写       | Ark、OpenAI、DeepSeek、Ollama、Qwen、Gemini | 查询意图理解和重写        | `M2Q_BUILTIN_CM_`                                                                                  |
|                  | 图像标注       | Ark、OpenAI、DeepSeek、Ollama、Qwen、Gemini | 图像内容识别和标注        | `IA_BUILTIN_CM_`                                                                                   |
|                  | 工作流知识召回 | Ark、OpenAI、DeepSeek、Ollama、Qwen、Gemini | 工作流中的知识检索        | `WKR_BUILTIN_CM_`                                                                                  |
| **Rerank 模型**  | 结果重排序     | VikingDB、RRF                               | 搜索结果重新排序          | `VIKINGDB_RERANK_`                                                                                 |
| **OCR 模型**     | 文字识别       | 火山引擎、PaddleOCR                         | 图像文字提取              | `VE_OCR_`、`PADDLEOCR_OCR_`                                                                        |

**配置说明**：

1. **嵌入模型**：用于将文本转换为向量表示，是知识库功能的基础
2. **聊天模型**：用于生成对话回复，支持配置多个模型实例
3. **内置聊天模型**：为特定功能提供专用的 AI 能力，可独立配置
4. **Rerank 模型**：用于优化搜索结果的相关性排序
5. **OCR 模型**：用于从图像中提取文本内容

**配置优先级**：

- 内置聊天模型支持功能级别的独立配置（如 `NL2SQL_BUILTIN_CM_TYPE`）
- 如果功能级别配置不存在，则回退到通用配置（如 `BUILTIN_CM_TYPE`）
- 这种设计允许为不同功能配置不同的模型，提高系统灵活性

#### 4.3.1 嵌入模型配置

嵌入模型用于知识库向量化，支持多种接入方式。如果向量数据库自带嵌入功能（如 `VikingDB`），则无需配置。

```bash
# 嵌入模型基础配置
export EMBEDDING_TYPE="ark"                   # ark / openai / ollama / gemini / http
export EMBEDDING_MAX_BATCH_SIZE=100           # 批处理大小
```

**说明：**

`EMBEDDING_TYPE` 环境变量是嵌入模型选择的核心配置项，系统会根据该变量的值来决定使用哪种嵌入模型提供商：

- **ark**：使用字节跳动 Ark 嵌入模型，需要配置 `ARK_EMBEDDING_*` 相关环境变量
- **openai**：使用 OpenAI 嵌入模型，需要配置 `OPENAI_EMBEDDING_*` 相关环境变量
- **ollama**：使用 Ollama 本地嵌入模型，需要配置 `OLLAMA_EMBEDDING_*` 相关环境变量
- **gemini**：使用 Google Gemini 嵌入模型，需要配置 `GEMINI_EMBEDDING_*` 相关环境变量
- **http**：使用自定义 HTTP 嵌入模型，需要配置 `HTTP_EMBEDDING_*` 相关环境变量

系统在启动时会读取 `EMBEDDING_TYPE` 的值，并根据该值初始化对应的嵌入模型实例。如果设置了不支持的值，系统会返回错误："`init knowledge embedding failed, type not configured`"。

`EMBEDDING_MAX_BATCH_SIZE` 用于控制批处理的大小，影响嵌入向量生成的性能和内存使用。

**Ark 嵌入模型（推荐）**：

```bash
# Ark 嵌入模型配置（火山引擎/字节跳动）
export ARK_EMBEDDING_BASE_URL=""              # （必填）Ark 嵌入模型基础 URL
export ARK_EMBEDDING_MODEL=""                 # （必填）Ark 嵌入模型名称
export ARK_EMBEDDING_API_KEY=""               # （必填）Ark 嵌入模型 API 密钥
export ARK_EMBEDDING_DIMS="2048"              # （必填）Ark 嵌入模型维度
export ARK_EMBEDDING_API_TYPE=""              # （可选）API 类型：text_api / multi_modal_api，默认 text_api
```

**OpenAI 嵌入模型**：

```bash
# OpenAI 嵌入模型配置
export OPENAI_EMBEDDING_BASE_URL=""           # （必填）OpenAI 嵌入模型基础 URL
export OPENAI_EMBEDDING_MODEL=""              # （必填）OpenAI 嵌入模型名称
export OPENAI_EMBEDDING_API_KEY=""            # （必填）OpenAI 嵌入模型 API 密钥
export OPENAI_EMBEDDING_BY_AZURE=false        # （可选）是否使用 Azure OpenAI
export OPENAI_EMBEDDING_API_VERSION=""        # （可选）Azure OpenAI API 版本
export OPENAI_EMBEDDING_DIMS=1024             # （必填）OpenAI 嵌入模型维度
export OPENAI_EMBEDDING_REQUEST_DIMS=1024     # （可选）请求中的嵌入维度，如果 API 不支持指定维度则留空
```

**Ollama 嵌入模型**：

```bash
# Ollama 嵌入模型配置
export OLLAMA_EMBEDDING_BASE_URL=""           # （必填）Ollama 嵌入模型基础 URL
export OLLAMA_EMBEDDING_MODEL=""              # （必填）Ollama 嵌入模型名称
export OLLAMA_EMBEDDING_DIMS=""               # （必填）Ollama 嵌入模型维度
```

**Gemini 嵌入模型**：

```bash
# Gemini 嵌入模型配置
export GEMINI_EMBEDDING_BASE_URL=""                  # （必填）Gemini 嵌入模型基础 URL
export GEMINI_EMBEDDING_MODEL="gemini-embedding-001" # （必填）Gemini 嵌入模型名称
export GEMINI_EMBEDDING_API_KEY=""                   # （必填）Gemini 嵌入模型 API 密钥
export GEMINI_EMBEDDING_DIMS=2048                    # （必填）Gemini 嵌入模型维度
export GEMINI_EMBEDDING_BACKEND="1"                  # （必填）Gemini 后端：1 为 BackendGeminiAPI / 2 为 BackendVertexAI
export GEMINI_EMBEDDING_PROJECT=""                   # （可选）Gemini 项目
export GEMINI_EMBEDDING_LOCATION=""                  # （可选）Gemini 位置
```

**HTTP 嵌入模型**：

```bash
# HTTP 嵌入模型配置
export HTTP_EMBEDDING_ADDR=""             # （必填）HTTP 嵌入模型地址
export HTTP_EMBEDDING_DIMS=1024           # （必填）HTTP 嵌入模型维度
```

#### 4.3.2 聊天模型配置

用于 `Agent` 和 `Workflow` 的聊天模型配置，支持添加多个模型（通过后缀数字区分）。

```bash
# 聊天模型配置（可添加多个，通过后缀数字区分）
export MODEL_PROTOCOL_0="ark"             # 协议类型
export MODEL_OPENCOZE_ID_0="100001"       # 记录 ID
export MODEL_NAME_0=""                    # 显示名称
export MODEL_ID_0=""                      # 连接用的模型名称
export MODEL_API_KEY_0=""                 # 模型 API 密钥
export MODEL_BASE_URL_0=""                # 模型基础 URL
```

#### 4.3.3 内置聊天模型配置

内置聊天模型是系统内部使用的专用模型，主要用于支持系统核心功能的自动化处理，与用户在 `Agent` 和 `Workflow` 中选择的聊天模型不同。

**功能作用**：

- **知识库处理**：自然语言转 SQL（`nl2sql`）查询
- **消息优化**：消息重写和查询优化（`messages2query`）
- **图像理解**：图像内容标注和分析（`image annotation`）
- **工作流增强**：工作流中的知识召回和推理（`workflow knowledge recall`）

**与普通聊天模型的区别**：

- 普通聊天模型：用户在创建 `Agent` 和 `Workflow` 时可选择的对话模型
- 内置聊天模型：系统后台自动调用，用户无需手动选择，专门处理系统内部任务

支持为特定功能配置专用模型（通过前缀区分）：

- **nl2sql**: `NL2SQL_`
- **messages2query**: `M2Q_`
- **image annotation**: `IA_`
- **workflow knowledge recall**: `WKR_`

**使用示例**：

如果要为 NL2SQL 功能配置专用的 Ark 模型，可以设置：

```bash
# 为 NL2SQL 功能配置专用模型
export NL2SQL_BUILTIN_CM_TYPE="ark"
export NL2SQL_BUILTIN_CM_ARK_API_KEY="your_nl2sql_api_key"
export NL2SQL_BUILTIN_CM_ARK_MODEL="your_nl2sql_model"
export NL2SQL_BUILTIN_CM_ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"

# 为消息重写功能配置专用模型
export M2Q_BUILTIN_CM_TYPE="openai"
export M2Q_BUILTIN_CM_OPENAI_API_KEY="your_m2q_api_key"
export M2Q_BUILTIN_CM_OPENAI_MODEL="gpt-4"
export M2Q_BUILTIN_CM_OPENAI_BASE_URL="https://api.openai.com/v1"
```

**工作机制**：

- 系统会优先读取带前缀的环境变量（如 `NL2SQL_BUILTIN_CM_TYPE`）
- 如果前缀环境变量不存在，则回退到通用配置（如 `BUILTIN_CM_TYPE`）
- 这样可以为不同功能配置不同的模型，提高系统的灵活性和性能

```bash
# 内置聊天模型基础配置
export BUILTIN_CM_TYPE="ark"              # openai / ark / deepseek / ollama / qwen / gemini
```

**Ark 内置聊天模型**：

```bash
# Ark 内置聊天模型配置
export BUILTIN_CM_ARK_API_KEY=""          # （必填）Ark API 密钥
export BUILTIN_CM_ARK_MODEL=""            # （必填）Ark 模型名称
export BUILTIN_CM_ARK_BASE_URL=""         # （必填）Ark 基础 URL
```

**OpenAI 内置聊天模型**：

```bash
# OpenAI 内置聊天模型配置
export BUILTIN_CM_OPENAI_BASE_URL=""      # （必填）OpenAI 基础 URL
export BUILTIN_CM_OPENAI_API_KEY=""       # （必填）OpenAI API 密钥
export BUILTIN_CM_OPENAI_BY_AZURE=false   # （可选）是否使用 Azure
export BUILTIN_CM_OPENAI_MODEL=""         # （必填）OpenAI 模型名称
```

**DeepSeek 内置聊天模型**：

```bash
# DeepSeek 内置聊天模型配置
export BUILTIN_CM_DEEPSEEK_BASE_URL=""    # （必填）DeepSeek 基础 URL
export BUILTIN_CM_DEEPSEEK_API_KEY=""     # （必填）DeepSeek API 密钥
export BUILTIN_CM_DEEPSEEK_MODEL=""       # （必填）DeepSeek 模型名称
```

**Ollama 内置聊天模型**：

```bash
# Ollama 内置聊天模型配置
export BUILTIN_CM_OLLAMA_BASE_URL=""      # （必填）Ollama 基础 URL
export BUILTIN_CM_OLLAMA_MODEL=""         # （必填）Ollama 模型名称
```

**Qwen 内置聊天模型**：

```bash
# Qwen 内置聊天模型配置
export BUILTIN_CM_QWEN_BASE_URL=""        # （必填）Qwen 基础 URL
export BUILTIN_CM_QWEN_API_KEY=""         # （必填）Qwen API 密钥
export BUILTIN_CM_QWEN_MODEL=""           # （必填）Qwen 模型名称
```

**Gemini 内置聊天模型**：

```bash
# Gemini 内置聊天模型配置
export BUILTIN_CM_GEMINI_BACKEND=""       # （必填）Gemini 后端
export BUILTIN_CM_GEMINI_API_KEY=""       # （必填）Gemini API 密钥
export BUILTIN_CM_GEMINI_PROJECT=""       # （可选）Gemini 项目
export BUILTIN_CM_GEMINI_LOCATION=""      # （可选）Gemini 位置
export BUILTIN_CM_GEMINI_BASE_URL=""      # （必填）Gemini 基础 URL
export BUILTIN_CM_GEMINI_MODEL=""         # （必填）Gemini 模型名称
```

### 4.4 功能组件配置

#### 4.4.1 代码运行器配置

```bash
# 代码运行器类型
export CODE_RUNNER_TYPE="local"               # sandbox / local
export CODE_RUNNER_TIMEOUT_SECONDS="60"       # 执行超时时间（秒）
export CODE_RUNNER_MEMORY_LIMIT_MB="100"      # 内存限制（MB）
export CODE_RUNNER_ALLOW_NET="cdn.jsdelivr.net" # 允许的网络访问

# 沙箱配置（当使用 sandbox 时）
export CODE_RUNNER_ALLOW_ENV=""                # 允许的环境变量
export CODE_RUNNER_ALLOW_READ=""               # 允许读取的路径
export CODE_RUNNER_ALLOW_WRITE=""              # 允许写入的路径
export CODE_RUNNER_ALLOW_RUN=""                # 允许执行的命令
export CODE_RUNNER_ALLOW_NET="cdn.jsdelivr.net"  # 允许的网络访问
export CODE_RUNNER_TIMEOUT_SECONDS=""          # 执行超时时间
export CODE_RUNNER_MEMORY_LIMIT_MB=""          # 内存限制
```

#### 4.4.2 OCR 配置

```bash
# OCR 类型配置
export OCR_TYPE="ve"                          # ve / paddleocr

# 火山引擎 OCR 配置
export VE_OCR_AK=""                           # 火山引擎 OCR Access Key
export VE_OCR_SK=""                           # 火山引擎 OCR Secret Key

# PaddleOCR 配置
export PADDLEOCR_OCR_API_URL=""               # PaddleOCR API 地址
```

#### 4.4.3 Rerank 模型配置

```bash
# Rerank 模型类型配置
export RERANK_TYPE="rrf"                      # vikingdb / rrf / ark / openai / http

# VikingDB Rerank 配置
export VIKINGDB_RERANK_HOST="api-knowledgebase.mlp.cn-beijing.volces.com"
export VIKINGDB_RERANK_REGION="cn-north-1"
export VIKINGDB_RERANK_AK=""                  # VikingDB Rerank Access Key
export VIKINGDB_RERANK_SK=""                  # VikingDB Rerank Secret Key
export VIKINGDB_RERANK_MODEL="base-multilingual-rerank"  # 支持 m3-v2-rerank

# 注意：目前仅支持 VikingDB 和 RRF 两种 Rerank 模型
```

#### 4.4.4 文档解析器配置

```bash
# 文档解析器类型
export PARSER_TYPE="builtin"                  # builtin / paddleocr

# PaddleOCR 结构化解析配置
export PADDLEOCR_STRUCTURE_API_URL=""         # PaddleOCR 结构化 API 地址
```

### 4.5 安全配置

#### 4.5.1 密码安全

```bash
# 使用强密码
export MYSQL_ROOT_PASSWORD="$(openssl rand -base64 32)"
export MINIO_ROOT_PASSWORD="$(openssl rand -base64 32)"

# 插件加密密钥（必须是 16/24/32 字节）
export PLUGIN_AES_AUTH_SECRET="^*6x3hdu2nc%-p38"        # 16字节示例
export PLUGIN_AES_STATE_SECRET="osj^kfhsd*(z!sno"       # 16字节示例
export PLUGIN_AES_OAUTH_TOKEN_SECRET="cn+$PJ(HhJ[5d*z9" # 16字节示例
```

#### 4.5.2 网络安全

```bash
# 启用 HTTPS（与6.1.1中的SSL配置相同，生产环境建议启用）
export USE_SSL="1"
export SSL_CERT_FILE="/path/to/cert.pem"
export SSL_KEY_FILE="/path/to/key.pem"
```

#### 4.5.3 用户注册控制

```bash
# 禁用用户注册
export DISABLE_USER_REGISTRATION="true"       # 设置为true禁用注册
# 允许注册的邮箱白名单（逗号分隔）
export ALLOW_REGISTRATION_EMAIL="admin@company.com,user@company.com"
```

#### 4.5.4 CORS 配置

```bash
# CORS 跨域配置
export CORS_ALLOW_ORIGINS="*"                  # 允许的源（生产环境建议具体指定）
export CORS_ALLOW_METHODS="GET,POST,PUT,DELETE,OPTIONS"  # 允许的HTTP方法
export CORS_ALLOW_HEADERS="*"                  # 允许的请求头
```

#### 4.5.5 其他安全配置

```bash
# 请求限制
export MAX_REQUEST_BODY_SIZE=1073741824        # 最大请求体大小（1GB）
export REQUEST_TIMEOUT="30s"                   # 请求超时时间

# 日志配置
export LOG_LEVEL="info"                        # 生产环境建议使用info级别
export LOG_FORMAT="json"                       # 日志格式
```

---

## 5. 运维管理

### 5.1 监控与维护

#### 5.1.1 健康检查

每个服务都配置了健康检查机制：

```yaml
# 示例：MySQL 健康检查
healthcheck:
  test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
  interval: 10s # 检查间隔
  timeout: 5s # 超时时间
  retries: 5 # 重试次数
  start_period: 30s # 启动等待时间
```

#### 5.1.2 日志管理

**日志管理**：

详细的日志查看和管理命令请参考附录中的服务管理命令。

**日志轮转配置**（在 docker-compose.yml 中添加）：

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

#### 5.1.3 数据备份

详细的备份策略和自动化脚本请参考本章节 5.4.3 的备份策略。

**快速备份**：

详细的备份命令请参考附录中的数据库操作和维护命令。

#### 5.1.4 性能优化

1. **资源限制**

   详细的资源限制配置请参考本章节 5.4.1 的生产环境配置。

2. **存储优化**

   ```bash
   # 使用 SSD 存储
   # 配置合适的文件系统（ext4/xfs）
   # 启用数据库查询缓存
   # 优化 Elasticsearch 堆内存设置
   ```

---

### 5.2 故障排除

#### 5.2.1 常见问题

1. **服务启动失败**

   - 检查服务状态：参考附录中的服务管理命令
   - 查看错误日志：参考附录中的日志查看命令
   - 检查端口占用：参考附录中的系统监控命令

2. **数据库连接问题**

   - 检查 MySQL 服务状态：参考附录中的数据库操作命令
   - 检查网络连通性：`docker-compose exec coze-server ping mysql`

3. **存储空间不足**
   - 检查磁盘使用情况：参考附录中的系统监控命令
   - 清理 Docker 缓存：参考附录中的清理和维护命令

#### 5.2.2 性能问题诊断

1. **资源使用监控**

   - 查看容器资源使用：参考附录中的系统监控命令
   - 查看系统负载：参考附录中的系统监控命令

2. **数据库性能**
   - MySQL 慢查询日志：`docker-compose exec mysql mysql -u root -p -e "SHOW VARIABLES LIKE 'slow_query_log';"`
   - Redis 性能监控：参考附录中的数据库操作命令

---

### 5.3 升级指南

#### 5.3.1 版本升级步骤

1. **备份数据**

   - 备份数据库：参考附录中的数据库操作命令
   - 备份配置文件：`cp -r . ../coze-studio-backup-$(date +%Y%m%d)`

2. **更新镜像**

   - 拉取最新镜像：`docker-compose pull`
   - 停止和启动服务：参考附录中的服务管理命令

3. **验证升级**
   - 检查服务状态：参考附录中的服务管理命令
   - 检查应用功能：`curl -f http://localhost:8888/health`

#### 5.3.2 回滚策略

```bash
# 如果升级失败，回滚到备份版本
docker-compose down
cp -r ../coze-studio-backup-20240101/* .
docker-compose up -d

# 恢复数据库
docker-compose exec -T mysql mysql -u root -p < backup_before_upgrade.sql
```

---

### 5.4 最佳实践

#### 5.4.1 生产环境配置

1. **安全加固**

   ```bash
   # 使用强密码
   MYSQL_ROOT_PASSWORD="$(openssl rand -base64 32)"
   MINIO_ROOT_PASSWORD="$(openssl rand -base64 32)"

   # 限制Web访问（使用反向代理）
   WEB_LISTEN_ADDR="0.0.0.0:8888"

   # 禁用用户注册
   DISABLE_USER_REGISTRATION="true"
   ```

2. **性能优化**

   ```bash
   # 调整JVM参数（Elasticsearch）
   ES_JAVA_OPTS="-Xms2g -Xmx2g"

   # 增加连接池大小
   MYSQL_MAX_CONNECTIONS=200

   # 优化Redis配置
   REDIS_MAXMEMORY="1gb"
   REDIS_MAXMEMORY_POLICY="allkeys-lru"
   ```

3. **资源限制**

   在 `docker-compose.yml` 中添加资源限制：

   ```yaml
   services:
     mysql:
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: "1.0"
           reservations:
             memory: 1G
             cpus: "0.5"
   ```

#### 5.4.2 监控告警

1. **集成 Prometheus 监控**

   ```yaml
   # 添加到docker-compose.yml
   prometheus:
     image: prom/prometheus:latest
     ports:
       - "9090:9090"
     volumes:
       - ./prometheus.yml:/etc/prometheus/prometheus.yml

   grafana:
     image: grafana/grafana:latest
     ports:
       - "3000:3000"
     environment:
       - GF_SECURITY_ADMIN_PASSWORD=admin
   ```

2. **健康检查脚本**

   ```bash
   #!/bin/bash
   # health_check.sh

   services=("mysql" "redis" "elasticsearch" "minio" "milvus" "coze-server" "coze-web")

   for service in "${services[@]}"; do
       if ! docker-compose ps $service | grep -q "Up"; then
           echo "ALERT: $service is down!"
           # 发送告警通知
       fi
   done
   ```

#### 5.4.3 备份策略

1. **自动化备份脚本**

   ```bash
   #!/bin/bash
   # backup.sh

   BACKUP_DIR="/opt/backups/$(date +%Y%m%d)"
   mkdir -p $BACKUP_DIR

   # MySQL备份
   docker-compose exec -T mysql mysqldump -u root -p$MYSQL_ROOT_PASSWORD --all-databases > $BACKUP_DIR/mysql.sql

   # MinIO备份
   docker-compose exec -T minio mc mirror /data $BACKUP_DIR/minio/

   # 配置文件备份
   cp .env $BACKUP_DIR/
   cp docker-compose.yml $BACKUP_DIR/

   # 压缩备份
   tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
   rm -rf $BACKUP_DIR

   # 清理旧备份（保留7天）
   find /opt/backups -name "*.tar.gz" -mtime +7 -delete
   ```

2. **定时备份**

   ```bash
   # 添加到crontab
   0 2 * * * /opt/coze-studio/backup.sh
   ```

#### 5.4.4 高可用部署

1. **负载均衡配置**

   ```nginx
   # nginx.conf
   upstream coze_backend {
       server 192.168.1.10:8888;
       server 192.168.1.11:8888;
       server 192.168.1.12:8888;
   }

   server {
       listen 80;
       server_name coze.example.com;

       location / {
           proxy_pass http://coze_backend;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

2. **数据库集群**

   ```yaml
   # MySQL主从配置示例
   mysql-master:
     image: mysql:8.0
     environment:
       MYSQL_REPLICATION_MODE: master
       MYSQL_REPLICATION_USER: replicator
       MYSQL_REPLICATION_PASSWORD: replicator_password

   mysql-slave:
     image: mysql:8.0
     environment:
       MYSQL_REPLICATION_MODE: slave
       MYSQL_MASTER_HOST: mysql-master
       MYSQL_REPLICATION_USER: replicator
       MYSQL_REPLICATION_PASSWORD: replicator_password
   ```

#### 5.4.5 故障恢复

1. **快速恢复脚本**

   ```bash
   #!/bin/bash
   # recovery.sh

   echo "Starting Coze Studio recovery..."

   # 停止所有服务
   docker-compose down

   # 清理异常容器
   docker system prune -f

   # 恢复配置
   if [ -f ".env.backup" ]; then
       cp .env.backup .env
   fi

   # 重新启动服务
   docker-compose up -d

   echo "Recovery completed. Please check service status."
   ```

2. **数据恢复**

   ```bash
   # 从备份恢复MySQL
   docker-compose exec -T mysql mysql -u root -p$MYSQL_ROOT_PASSWORD < backup/mysql.sql

   # 恢复MinIO数据
   docker-compose exec minio mc mirror backup/minio/ /data/
   ```

---

## 6. 本地模型服务部署

本章节介绍如何部署和配置本地模型服务，以 Ollama 为例，提供完全离线的 AI 模型解决方案。本地模型服务可以降低对外部 API 的依赖，提高数据安全性和响应速度。

### 6.1 Ollama 服务部署

#### 6.1.1 Ollama 容器配置

在 `docker-compose.yml` 中添加 Ollama 服务：

```yaml
# Ollama 本地模型服务
ollama:
  image: ollama/ollama:latest
  container_name: coze-ollama
  restart: unless-stopped
  ports:
    - "11434:11434" # Ollama API 端口
  volumes:
    - ollama_data:/root/.ollama # 模型数据存储
  environment:
    - OLLAMA_HOST=0.0.0.0
    - OLLAMA_ORIGINS=*
  networks:
    - coze-network
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu] # GPU 支持（可选）
```

#### 6.1.2 模型下载和管理

**下载常用模型**：

```bash
# 进入 Ollama 容器
docker-compose exec ollama bash

# 下载聊天模型
ollama pull llama3.1:8b          # Meta Llama 3.1 8B
ollama pull qwen2.5:7b           # 阿里通义千问 2.5 7B
ollama pull gemma2:9b            # Google Gemma 2 9B
ollama pull mistral:7b           # Mistral 7B

# 下载嵌入模型
ollama pull nomic-embed-text     # 英文嵌入模型
ollama pull bge-m3               # 多语言嵌入模型
ollama pull mxbai-embed-large    # 高质量嵌入模型

# 下载中文优化模型
ollama pull qwen2.5:14b          # 更大参数的中文模型
ollama pull chatglm3:6b          # 清华 ChatGLM3
```

**查看已安装模型**：

```bash
# 列出所有模型
ollama list

# 查看模型详情
ollama show llama3.1:8b
```

### 6.2 Coze Studio 配置

#### 6.2.1 嵌入模型配置

使用 Ollama 作为嵌入模型提供者：

```bash
# Ollama 嵌入模型配置
export EMBEDDING_TYPE="ollama"                    # 设置为 ollama
export OLLAMA_EMBEDDING_BASE_URL="http://ollama:11434"  # Ollama 服务地址
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"  # 嵌入模型名称
export OLLAMA_EMBEDDING_DIMS="768"               # 模型维度（根据具体模型调整）
export EMBEDDING_MAX_BATCH_SIZE="50"             # 批处理大小（字符串格式）
```

**常用嵌入模型维度参考**：

| 模型名称          | 维度 | 特点             |
| ----------------- | ---- | ---------------- |
| nomic-embed-text  | 768  | 英文优化，轻量级 |
| bge-m3            | 1024 | 多语言支持       |
| mxbai-embed-large | 1024 | 高质量嵌入       |

#### 6.2.2 聊天模型配置

配置 Ollama 聊天模型供 Agent 和 Workflow 使用：

```bash
# 聊天模型配置（可配置多个）
export MODEL_PROTOCOL_0="ollama"                 # 协议类型
export MODEL_COZE_ID_0="200001"                  # 记录 ID
export MODEL_NAME_0="Llama 3.1 8B"               # 显示名称
export MODEL_ID_0="llama3.1:8b"                  # 模型名称
export MODEL_API_KEY_0="ollama"                  # API 密钥（可为任意值）
export MODEL_BASE_URL_0="http://ollama:11434"    # Ollama 服务地址

# 第二个模型配置
export MODEL_PROTOCOL_1="ollama"
export MODEL_COZE_ID_1="200002"
export MODEL_NAME_1="通义千问 2.5 7B"
export MODEL_ID_1="qwen2.5:7b"
export MODEL_API_KEY_1="ollama"
export MODEL_BASE_URL_1="http://ollama:11434"

# 第三个模型配置
export MODEL_PROTOCOL_2="ollama"
export MODEL_COZE_ID_2="200003"
export MODEL_NAME_2="Gemma 2 9B"
export MODEL_ID_2="gemma2:9b"
export MODEL_API_KEY_2="ollama"
export MODEL_BASE_URL_2="http://ollama:11434"
```

#### 6.2.3 内置聊天模型配置

配置 Ollama 作为内置功能的聊天模型：

```bash
# 内置聊天模型配置
export BUILTIN_CM_TYPE="ollama"                  # 设置为 ollama
export BUILTIN_CM_OLLAMA_BASE_URL="http://ollama:11434"  # Ollama 服务地址
export BUILTIN_CM_OLLAMA_MODEL="qwen2.5:7b"      # 内置功能使用的模型
```

**针对特定功能的模型配置**：

> **注意**：如需使用这些功能，可以根据需要添加相应的环境变量。

```bash
# NL2SQL 功能专用模型（自然语言转SQL）
export NL2SQL_BUILTIN_CM_TYPE="ollama"
export NL2SQL_BUILTIN_CM_OLLAMA_BASE_URL="http://ollama:11434"
export NL2SQL_BUILTIN_CM_OLLAMA_MODEL="qwen2.5:14b"  # 使用更大的模型提高准确性

# 消息重写功能专用模型（Messages to Query）
export M2Q_BUILTIN_CM_TYPE="ollama"
export M2Q_BUILTIN_CM_OLLAMA_BASE_URL="http://ollama:11434"
export M2Q_BUILTIN_CM_OLLAMA_MODEL="llama3.1:8b"

# 图像标注功能专用模型（Image Annotation，需要多模态模型）
export IA_BUILTIN_CM_TYPE="ollama"
export IA_BUILTIN_CM_OLLAMA_BASE_URL="http://ollama:11434"
export IA_BUILTIN_CM_OLLAMA_MODEL="llava:7b"      # 多模态模型

# 工作流知识召回专用模型（Workflow Knowledge Recall）
export WKR_BUILTIN_CM_TYPE="ollama"
export WKR_BUILTIN_CM_OLLAMA_BASE_URL="http://ollama:11434"
export WKR_BUILTIN_CM_OLLAMA_MODEL="qwen2.5:7b"
```

### 6.3 性能优化配置

#### 6.3.1 GPU 加速配置

**NVIDIA GPU 支持**：

```yaml
# docker-compose.yml 中的 GPU 配置
ollama:
  image: ollama/ollama:latest
  container_name: coze-ollama
  restart: unless-stopped
  ports:
    - "11434:11434"
  volumes:
    - ollama_data:/root/.ollama
  environment:
    - OLLAMA_HOST=0.0.0.0
    - OLLAMA_ORIGINS=*
    - CUDA_VISIBLE_DEVICES=0,1 # 指定使用的 GPU
  networks:
    - coze-network
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

**环境变量配置**：

```bash
# GPU 内存管理
export OLLAMA_GPU_MEMORY_FRACTION=0.8    # 使用 80% 的 GPU 内存
export OLLAMA_NUM_PARALLEL=2             # 并行处理数量
export OLLAMA_MAX_LOADED_MODELS=3        # 最大加载模型数
```

#### 6.3.2 CPU 优化配置

```bash
# CPU 优化配置
export OLLAMA_NUM_THREAD=8               # CPU 线程数
export OLLAMA_NUMA=true                  # 启用 NUMA 优化
export OMP_NUM_THREADS=8                 # OpenMP 线程数
```

#### 6.3.3 内存管理配置

```bash
# 内存管理
export OLLAMA_MAX_QUEUE=512              # 最大队列长度
export OLLAMA_KEEP_ALIVE=5m              # 模型保持加载时间
export OLLAMA_LOAD_TIMEOUT=5m            # 模型加载超时时间
```

### 6.4 模型管理最佳实践

#### 6.4.1 模型选择建议

**模型选择建议**：

| 硬件配置            | 推荐聊天模型              | 推荐嵌入模型      | 说明     |
| ------------------- | ------------------------- | ----------------- | -------- |
| 低配置（8GB RAM）   | qwen2.5:7b, llama3.1:8b   | nomic-embed-text  | 基础功能 |
| 中配置（16GB RAM）  | qwen2.5:14b, gemma2:9b    | bge-m3            | 平衡性能 |
| 高配置（32GB+ RAM） | qwen2.5:32b, llama3.1:70b | mxbai-embed-large | 最佳效果 |

**场景化模型选择**：

- **中文优化**：qwen2.5 系列、chatglm3:6b
- **英文优化**：llama3.1 系列、mistral:7b
- **代码生成**：codellama:7b、qwen2.5-coder
- **多模态**：llava:7b、llava:13b

#### 6.4.2 模型更新和维护

**定期更新模型**：

```bash
#!/bin/bash
# update_models.sh - 模型更新脚本

echo "开始更新 Ollama 模型..."

# 获取已安装模型列表
MODELS=$(docker-compose exec ollama ollama list | grep -v NAME | awk '{print $1}')

# 更新每个模型
for model in $MODELS; do
    echo "更新模型: $model"
    docker-compose exec ollama ollama pull $model
done

echo "模型更新完成"
```

**清理未使用的模型**：

```bash
# 删除特定模型
docker-compose exec ollama ollama rm old-model:tag

# 查看模型占用空间
docker-compose exec ollama du -sh /root/.ollama/models/*
```

### 6.5 监控和故障排除

#### 6.5.1 服务监控

**健康检查配置**：

```yaml
# docker-compose.yml 中添加健康检查
ollama:
  image: ollama/ollama:latest
  container_name: coze-ollama
  restart: unless-stopped
  ports:
    - "11434:11434"
  volumes:
    - ollama_data:/root/.ollama
  environment:
    - OLLAMA_HOST=0.0.0.0
    - OLLAMA_ORIGINS=*
  networks:
    - coze-network
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 60s
```

**监控脚本**：

```bash
#!/bin/bash
# monitor_ollama.sh - Ollama 监控脚本

echo "=== Ollama 服务状态 ==="
docker-compose ps ollama

echo "\n=== 已安装模型 ==="
docker-compose exec ollama ollama list

echo "\n=== 资源使用情况 ==="
docker stats coze-ollama --no-stream

echo "\n=== 磁盘使用情况 ==="
docker-compose exec ollama du -sh /root/.ollama

echo "\n=== API 连通性测试 ==="
curl -s http://localhost:11434/api/tags | jq '.models[].name' || echo "API 连接失败"
```

#### 6.5.2 常见问题排除

**Ollama 特有问题**：

| 问题类型     | 快速解决方案                 | 详细排查             |
| ------------ | ---------------------------- | -------------------- |
| 模型加载失败 | `ollama rm && ollama pull`   | 检查模型文件完整性   |
| 内存不足     | 调整 `OLLAMA_NUM_PARALLEL=1` | 减少并发和缓存模型   |
| GPU 问题     | 检查 `nvidia-smi`            | 验证 Docker GPU 支持 |
| API 连接失败 | 检查端口和防火墙             | 参考 7.2.1 网络诊断  |

**性能优化参数**：

```bash
# 内存优化
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=1m

# GPU 优化
export OLLAMA_GPU_LAYERS=35  # 根据显存调整
```

通用故障排除方法请参考 7.2 章节。

### 6.6 安全配置

#### 6.6.1 网络安全

```bash
# 限制 Ollama 访问来源
export OLLAMA_ORIGINS="http://localhost:8888,http://coze-web:8888"

# 使用内部网络
# 在 docker-compose.yml 中不暴露外部端口，仅供内部服务访问
```

#### 6.6.2 数据安全

```bash
# 定期备份模型数据
docker run --rm -v coze-studio_ollama_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/ollama-models-$(date +%Y%m%d).tar.gz -C /data .

# 恢复模型数据
docker run --rm -v coze-studio_ollama_data:/data -v $(pwd)/backup:/backup alpine tar xzf /backup/ollama-models-20240101.tar.gz -C /data
```

---

## 7. 附录

### 7.1 常用命令参考

#### 7.1.1 服务管理命令

```bash
# 查看服务状态
docker-compose ps

# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 重启特定服务
docker-compose restart [service_name]

# 查看服务日志
docker-compose logs [service_name]     # 特定服务
docker-compose logs -f --tail=100      # 实时跟踪
docker-compose logs --since=1h          # 最近1小时
```

#### 7.1.2 数据库操作命令

```bash
# MySQL 连接测试
docker-compose exec mysql mysql -u root -p -e "SELECT 1"

# MySQL 数据备份
docker-compose exec mysql mysqldump -u root -p opencoze > backup_$(date +%Y%m%d_%H%M%S).sql
docker-compose exec mysql mysqldump -u root -p --all-databases > backup_all_$(date +%Y%m%d).sql

# Redis 连接测试
docker-compose exec redis redis-cli ping

# Redis 信息查看
docker-compose exec redis redis-cli info stats
```

#### 7.1.3 系统监控命令

```bash
# 查看容器资源使用
docker stats

# 查看磁盘使用情况
df -h

# 检查端口占用
netstat -tlnp | grep [port]

# 系统负载监控
top
htop
iostat -x 1
```

#### 7.1.4 清理和维护命令

```bash
# 清理 Docker 缓存
docker system prune -a

# 清理未使用的卷
docker volume prune

# 批量导出日志
mkdir -p logs && docker-compose config --services | \
xargs -I {} sh -c 'docker-compose logs --no-color {} > logs/{}.log'

# 数据目录备份
tar -czf coze_data_backup_$(date +%Y%m%d).tar.gz data/
```

### 7.2 官方文档链接

- [Coze Studio 官方文档](https://github.com/coze-dev/coze-studio)
- [Ollama 官方文档](https://ollama.ai/)
- [Docker 官方文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [MySQL 8.4 文档](https://dev.mysql.com/doc/refman/8.4/en/)
- [Redis 文档](https://redis.io/documentation)
- [Elasticsearch 文档](https://www.elastic.co/guide/)
- [MinIO 文档](https://min.io/docs/)
- [Milvus 文档](https://milvus.io/docs/)
- [NSQ 文档](https://nsq.io/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

---
