# JuiceFS 数据迁移指南（MySQL + MinIO）

## 1. 准备工作

### 1.1 停止所有写入

停止所有写入应用，确保无新的数据写入（保持 Mount Pod 和业务 Pod 运行）：  

```bash
# 验证当前无活跃写入（可选）
kubectl exec -it your-business-pod -- \
  lsof /mnt/juicefs | grep -v "READ"
```

> **注意**：我们保持 Mount Pod 和业务 Pod 继续运行，仅确保无新数据写入

### 1.2 备份元数据

备份 MySQL 数据库（使用 `mysqldump`）：  

```bash
mysqldump -h [mysql_host] -u [username] -p[password] [database_name] > juicefs_meta_backup_$(date +%F).sql
```

> 示例：`mysqldump -h [mysql_host] -u [username] -p'[password]' [database_name] > backup.sql`

### 1.3 验证新 MinIO 集群连通性

使用 MinIO 客户端工具直接测试新存储连通性：  

#### 1.3.1 使用 MinIO Client (mc)

```bash
# 安装 mc（如果未安装）
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# 配置新 MinIO 集群连接
mc alias set new-minio http://new-minio-endpoint:9000 NEW_MINIO_ACCESS_KEY NEW_MINIO_SECRET_KEY

# 验证连接
mc admin info new-minio

# 测试存储桶操作
mc mb new-minio/test-connectivity-bucket
mc cp /etc/hosts new-minio/test-connectivity-bucket/test-file
mc ls new-minio/test-connectivity-bucket/
mc cat new-minio/test-connectivity-bucket/test-file

# 清理测试资源
mc rm new-minio/test-connectivity-bucket/test-file
mc rb new-minio/test-connectivity-bucket
```

#### 1.3.2 使用 AWS CLI（S3 兼容）

```bash
# 配置 AWS CLI 连接新 MinIO
export AWS_ACCESS_KEY_ID=NEW_MINIO_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=NEW_MINIO_SECRET_KEY
export AWS_DEFAULT_REGION=us-east-1

# 测试连接
aws --endpoint-url http://new-minio-endpoint:9000 s3 ls

# 创建测试存储桶
aws --endpoint-url http://new-minio-endpoint:9000 s3 mb s3://test-connectivity-bucket

# 上传测试文件
echo "connectivity test" | aws --endpoint-url http://new-minio-endpoint:9000 s3 cp - s3://test-connectivity-bucket/test-file.txt

# 下载验证
aws --endpoint-url http://new-minio-endpoint:9000 s3 cp s3://test-connectivity-bucket/test-file.txt -

# 清理测试资源
aws --endpoint-url http://new-minio-endpoint:9000 s3 rm s3://test-connectivity-bucket/test-file.txt
aws --endpoint-url http://new-minio-endpoint:9000 s3 rb s3://test-connectivity-bucket
```

#### 1.3.3 使用 curl 直接测试

```bash
# 测试 MinIO API 健康状态
curl -I http://new-minio-endpoint:9000/minio/health/live

# 测试存储桶列表（需要认证）
curl -X GET \
  -H "Authorization: AWS4-HMAC-SHA256 ..." \
  http://new-minio-endpoint:9000/

# 或使用简单的连通性测试
curl -f http://new-minio-endpoint:9000/minio/health/ready
```

**验证成功标准**：

- ✅ MinIO 服务响应正常（HTTP 200）
- ✅ 能够成功创建和删除存储桶
- ✅ 能够正常上传和下载文件
- ✅ 网络延迟在可接受范围内（通常 < 100ms）

### 1.4 获取 JuiceFS 存储桶信息

#### 1.4.1 查询 MySQL 元数据库

查询当前 JuiceFS 使用的存储桶名称：  

```sql
USE juicefs_meta;
SELECT name, value FROM jfs_setting WHERE name IN ('storage', 'storage.bucket', 'storage.access', 'storage.secret');
```

#### 1.4.2 查看 Kubernetes CSI Secret（推荐）

如果使用 JuiceFS CSI，可以直接查看 Secret 配置：

```bash
# 查看 JuiceFS Secret
kubectl get secret juicefs-secret -o yaml

# 或者查看解码后的配置信息
kubectl get secret juicefs-secret -o jsonpath='{.data}' | jq -r 'to_entries[] | "\(.key): \(.value | @base64d)"'

# 查看特定字段
echo "存储类型: $(kubectl get secret juicefs-secret -o jsonpath='{.data.storage}' | base64 -d)"
echo "存储桶: $(kubectl get secret juicefs-secret -o jsonpath='{.data.bucket}' | base64 -d)"
echo "端点: $(kubectl get secret juicefs-secret -o jsonpath='{.data.endpoint}' | base64 -d)"
echo "访问密钥: $(kubectl get secret juicefs-secret -o jsonpath='{.data.access-key}' | base64 -d)"
```

> **注意**：Secret 名称可能因部署而异，常见名称包括 `juicefs-secret`、`juicefs-csi-secret` 等。可以通过 `kubectl get secrets | grep juicefs` 查找。

---

## 2. 数据迁移方案选择

### 2.1 方案 A：使用 JuiceFS Sync（推荐）

**优势**：

- 理解 JuiceFS 内部数据结构
- 自动处理元数据映射
- 支持增量同步和断点续传
- 内置数据校验机制

**适用场景**：跨云迁移、不同存储类型间迁移

#### 2.1.1 执行数据同步

```bash
# 获取旧 MinIO 配置信息
OLD_ENDPOINT="http://old-minio:9000"
OLD_BUCKET="juicefs-bucket"
OLD_ACCESS_KEY="old_access_key"
OLD_SECRET_KEY="old_secret_key"

# 新 MinIO 配置信息
NEW_ENDPOINT="http://new-minio:9000"
NEW_BUCKET="juicefs-bucket"
NEW_ACCESS_KEY="new_access_key"
NEW_SECRET_KEY="new_secret_key"

# 执行同步（支持断点续传）
juicefs sync \
  --src-endpoint $OLD_ENDPOINT \
  --dst-endpoint $NEW_ENDPOINT \
  s3://$OLD_BUCKET/ \
  s3://$NEW_BUCKET/ \
  --src-access-key $OLD_ACCESS_KEY \
  --src-secret-key $OLD_SECRET_KEY \
  --dst-access-key $NEW_ACCESS_KEY \
  --dst-secret-key $NEW_SECRET_KEY \
  --threads 32 \
  --list-threads 16 \
  --list-depth 2 \
  --no-https  # 如果使用 HTTP
```

#### 2.1.2 增量同步（可选）

如果迁移期间有少量写入：

```bash
juicefs sync \
  --src-endpoint $OLD_ENDPOINT \
  --dst-endpoint $NEW_ENDPOINT \
  s3://$OLD_BUCKET/ \
  s3://$NEW_BUCKET/ \
  --src-access-key $OLD_ACCESS_KEY \
  --src-secret-key $OLD_SECRET_KEY \
  --dst-access-key $NEW_ACCESS_KEY \
  --dst-secret-key $NEW_SECRET_KEY \
  --update \
  --threads 16
```

#### 2.1.3 数据一致性校验

```bash
juicefs sync \
  --src-endpoint $OLD_ENDPOINT \
  --dst-endpoint $NEW_ENDPOINT \
  s3://$OLD_BUCKET/ \
  s3://$NEW_BUCKET/ \
  --src-access-key $OLD_ACCESS_KEY \
  --src-secret-key $OLD_SECRET_KEY \
  --dst-access-key $NEW_ACCESS_KEY \
  --dst-secret-key $NEW_SECRET_KEY \
  --check-new \
  --check-all
```

### 2.2 方案 B：使用 MinIO Client (mc) 迁移

**优势**：

- 原生 MinIO 工具，性能优异
- 支持大规模并发传输
- 网络优化更好
- 支持服务端复制（如果新旧集群在同一网络）

**适用场景**：同构 MinIO 集群间迁移、大数据量迁移

#### 2.2.1 安装和配置 MinIO Client

```bash
# 安装 mc
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# 配置旧 MinIO 集群
mc alias set old-minio http://old-minio:9000 old_access_key old_secret_key

# 配置新 MinIO 集群
mc alias set new-minio http://new-minio:9000 new_access_key new_secret_key

# 验证连接
mc admin info old-minio
mc admin info new-minio
```

#### 2.2.2 创建目标存储桶

```bash
# 在新集群创建存储桶
mc mb new-minio/juicefs-bucket

# 复制存储桶策略（如果有）
mc policy get old-minio/juicefs-bucket > bucket-policy.json
mc policy set-json bucket-policy.json new-minio/juicefs-bucket
```

#### 2.2.3 执行数据迁移

```bash
# 方式1：使用 mc mirror（推荐）
mc mirror \
  --overwrite \
  --remove \
  --preserve \
  old-minio/juicefs-bucket/ \
  new-minio/juicefs-bucket/

# 方式2：使用 mc cp（适合大文件）
mc cp \
  --recursive \
  --preserve \
  old-minio/juicefs-bucket/ \
  new-minio/juicefs-bucket/
```

#### 2.2.4 数据校验

```bash
# 比较文件数量和大小
mc du old-minio/juicefs-bucket
mc du new-minio/juicefs-bucket

# 详细校验（可选，耗时较长）
mc mirror \
  --dry-run \
  --overwrite \
  old-minio/juicefs-bucket/ \
  new-minio/juicefs-bucket/
```

### 2.3 方案 C：混合迁移策略（大规模环境推荐）

适用于超大规模数据（TB级别以上）：

#### 2.3.1 第一阶段：使用 MinIO mc 批量迁移

利用 MinIO 原生工具的高性能完成主体数据迁移

#### 2.3.2 第二阶段：使用 JuiceFS sync 增量同步

处理迁移期间的增量数据和元数据一致性

---

## 3. 切换存储配置（关键步骤）

### 3.1 更新 MySQL 中的存储配置

使用 JuiceFS 命令修改元数据指向新 MinIO 集群：  

```bash
# 方式1：使用 juicefs config 命令
juicefs config "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]" \
  --storage s3 \
  --bucket http://new-minio-endpoint:9000/juicefs-bucket \
  --access-key NEW_MINIO_ACCESS_KEY \
  --secret-key NEW_MINIO_SECRET_KEY

# 方式2：如果需要指定 endpoint
juicefs config "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]" \
  --storage s3 \
  --bucket juicefs-bucket \
  --endpoint http://new-minio-endpoint:9000 \
  --access-key NEW_MINIO_ACCESS_KEY \
  --secret-key NEW_MINIO_SECRET_KEY
```

### 3.2 手动验证 MySQL 更新

连接 MySQL 确认配置更新：  

```sql
USE juicefs_meta;
SELECT name, value FROM jfs_setting WHERE name LIKE 'storage.%' ORDER BY name;
```

应看到更新后的值：

```text
+------------------+----------------------------------------+
| name             | value                                  |
+------------------+----------------------------------------+
| storage          | s3                                     |
| storage.bucket   | juicefs-bucket                         |
| storage.endpoint | http://new-minio-endpoint:9000         |
| storage.access   | NEW_MINIO_ACCESS_KEY                   |
| storage.secret   | NEW_MINIO_SECRET_KEY                   |
+------------------+----------------------------------------+
```

### 3.3 验证存储连接

测试新配置的连通性：

```bash
# 使用 juicefs status 检查存储状态
juicefs status "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]"

# 检查存储桶访问权限
juicefs info "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]"
```

---

## 4. 挂载测试

### 4.1 重新挂载文件系统

```bash
juicefs mount -d "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]" /mnt/juicefs
```

### 4.2 功能验证

```bash
# 测试新写入
echo "new-storage-test" > /mnt/juicefs/test.txt

# 验证历史文件
ls /mnt/juicefs/important_data

# 检查文件内容
tail /mnt/juicefs/existing-file.log
```

---

## 5. 恢复服务

### 5.1 批量更新客户端挂载命令

在所有客户端机器执行：  

```bash
juicefs mount -d "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]" /your/mountpoint
```

### 5.2 监控关键指标

```bash
# 实时监控
juicefs stats /mnt/juicefs --verbosity=1

# 检查错误日志
grep 'ERROR' /var/log/juicefs.log
```

### 5.3 JuiceFS CSI（Mount Pod 模式）配置更新

#### 5.3.1 更新 Secret 配置

使用 JuiceFS CSI Mount Pod 模式时，需要更新 Kubernetes 中的 Secret 配置：

```bash
# 查看当前 Secret（根据实际部署调整命名空间和名称）
kubectl get secret juicefs-sc-secret -n juicefs -o yaml

# 备份当前 Secret
kubectl get secret juicefs-sc-secret -n juicefs -o yaml > juicefs-secret-backup.yaml

# 方式1：直接编辑现有 Secret
kubectl edit secret juicefs-sc-secret -n juicefs

# 方式2：创建新的 Secret 配置文件
cat > juicefs-secret-updated.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: juicefs-sc-secret
  namespace: juicefs
  annotations:
    meta.helm.sh/release-name: juicefs
    meta.helm.sh/release-namespace: juicefs
  labels:
    app.kubernetes.io/instance: juicefs
    app.kubernetes.io/managed-by: Helm
    app.kubernetes.io/name: juicefs-csi-driver
    app.kubernetes.io/version: 0.22.0
    helm.sh/chart: juicefs-csi-driver-0.18.0
type: Opaque
stringData:
  # JuiceFS 文件系统名称
  name: "juicefstest"
  # MySQL 元数据库连接（需要更新为实际值）
  metaurl: "mysql://[username]:[password]@([mysql_host]:3306)/[database_name]"
  # 存储类型
  storage: "s3"
  # 需要更新的关键字段 - 新 MinIO 配置
  bucket: "NEW_MINIO_ENDPOINT:PORT/BUCKET_NAME"     # 新的MinIO端点和存储桶
  access-key: "NEW_MINIO_ACCESS_KEY"                # 新的访问密钥
  secret-key: "NEW_MINIO_SECRET_KEY"                # 新的密钥
EOF

# 应用新配置
kubectl apply -f juicefs-secret-updated.yaml
```

**重要说明**：

- **Secret 名称**：根据实际部署，可能是 `juicefs-sc-secret`、`juicefs-secret` 等
- **命名空间**：根据实际部署，可能是 `juicefs`、`kube-system`、`default` 等
- **bucket 字段格式**：当前格式为 `endpoint:port/bucket`，更新时保持相同格式
- **metaurl**：MySQL 连接字符串，通常不需要修改

#### 5.3.2 验证 Secret 更新

```bash
# 验证 Secret 更新是否成功
kubectl get secret juicefs-sc-secret -n juicefs -o jsonpath='{.data}' | jq -r 'to_entries[] | "\(.key): \(.value | @base64d)"'

# 或者查看特定字段
echo "存储类型: $(kubectl get secret juicefs-sc-secret -n juicefs -o jsonpath='{.data.storage}' | base64 -d)"
echo "存储桶: $(kubectl get secret juicefs-sc-secret -n juicefs -o jsonpath='{.data.bucket}' | base64 -d)"
echo "访问密钥: $(kubectl get secret juicefs-sc-secret -n juicefs -o jsonpath='{.data.access-key}' | base64 -d)"
```

如果使用 ConfigMap 存储非敏感配置：

```bash
# 查看当前 ConfigMap
kubectl get configmap juicefs-config -o yaml

# 创建新的 ConfigMap
cat > juicefs-configmap-new.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: juicefs-config
  namespace: default
data:
  storage: "s3"
  bucket: "juicefs-bucket"
  endpoint: "http://new-minio-endpoint:9000"
  # Mount Pod 特定配置
  mount-pod-cpu-limit: "2000m"
  mount-pod-memory-limit: "5Gi"
  mount-pod-cpu-request: "1000m"
  mount-pod-memory-request: "1Gi"
EOF

# 应用新配置
kubectl apply -f juicefs-configmap-new.yaml
```

#### 5.3.3 重启 JuiceFS CSI 组件和 Mount Pod

更新配置后，需要重启 CSI 组件并清理现有 Mount Pod：

```bash
# 查看当前 JuiceFS CSI 组件状态
kubectl get pods -n juicefs

# 重启 JuiceFS CSI 组件
# 重启 CSI Node DaemonSet
kubectl rollout restart daemonset/juicefs-csi-node -n juicefs

# 重启 CSI Controller StatefulSet
kubectl rollout restart statefulset/juicefs-csi-controller -n juicefs

# 查看现有的 Mount Pod（JuiceFS 应用 Pod）
kubectl get pods -n juicefs | grep juicefs-app

# 删除现有 Mount Pod 以强制重新创建（使用新配置）
# 注意：这会导致短暂的服务中断，请在维护窗口执行
kubectl delete pods -n juicefs -l app.kubernetes.io/name=juicefs-mount

# 等待 CSI 组件重启完成
kubectl rollout status daemonset/juicefs-csi-node -n juicefs
kubectl rollout status statefulset/juicefs-csi-controller -n juicefs

# 验证 CSI 组件状态
kubectl get pods -n juicefs
```

**重要提示**：

- CSI Controller 使用 StatefulSet 部署，不是 Deployment
- Mount Pod 删除会导致短暂的服务中断，建议在维护窗口执行
- 重启后 Mount Pod 会自动重新创建并使用新的存储配置

#### 5.3.4 验证 CSI 挂载状态

```bash
# 检查 PV 状态
kubectl get pv

# 检查 PVC 状态
kubectl get pvc

# 检查 Mount Pod 状态
kubectl get pods -A -l app.kubernetes.io/name=juicefs-mount -o wide
```

#### 5.3.5 验证业务 Pod

**操作步骤**：

```bash
# 测试现有业务 Pod 的写入功能
kubectl exec -it your-existing-pod -- \
  echo "Migration test - $(date)" > /mnt/juicefs/migration-test.txt

# 验证文件写入成功
kubectl exec -it your-existing-pod -- \
  cat /mnt/juicefs/migration-test.txt
```

**预期结果**：

- ✅ 现有 Pod 无需重启，能够正常写入数据
- ✅ 业务无中断，数据访问正常

#### 5.3.6 验证 StorageClass 和 PV

**检查现有资源**：

```bash
# 检查 StorageClass
kubectl get storageclass juicefs-sc -o yaml

# 检查现有 PV
kubectl get pv -l storage.kubernetes.io/provisioned-by=csi.juicefs.com

# 检查 PVC 状态
kubectl get pvc -A | grep juicefs
```

#### 5.3.7 验证新创建业务 Pod

**目的**：验证更新配置后，新创建的 Pod 能够正确使用新的 MinIO 存储配置。

##### 创建测试 PVC（动态供应）

```bash
# 创建测试 PVC 使用动态供应
cat > test-pvc-dynamic.yaml << EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: juicefs-test-pvc-dynamic
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
  storageClassName: juicefs-sc  # 触发动态供应
EOF

# 应用 PVC
kubectl apply -f test-pvc-dynamic.yaml

# 等待 PVC 绑定
kubectl wait --for=condition=Bound pvc/juicefs-test-pvc-dynamic --timeout=300s

# 验证动态创建的 PV
PV_NAME=$(kubectl get pvc juicefs-test-pvc-dynamic -o jsonpath='{.spec.volumeName}')
echo "动态创建的 PV 名称: $PV_NAME"
kubectl get pv $PV_NAME
```

##### 创建测试 Pod

```bash
# 创建使用动态供应 PVC 的测试 Pod
cat > test-pod-dynamic.yaml << EOF
apiVersion: v1
kind: Pod
metadata:
  name: juicefs-test-pod-dynamic
  namespace: default
spec:
  containers:
  - name: test-container
    image: busybox:latest
    command: ["/bin/sh"]
    args: ["-c", "while true; do sleep 3600; done"]
    volumeMounts:
    - name: juicefs-volume
      mountPath: /mnt/juicefs
  volumes:
  - name: juicefs-volume
    persistentVolumeClaim:
      claimName: juicefs-test-pvc-dynamic
  restartPolicy: Never
EOF

# 应用 Pod
kubectl apply -f test-pod-dynamic.yaml

# 等待 Pod 启动
kubectl wait --for=condition=Ready pod/juicefs-test-pod-dynamic --timeout=300s
```

##### 验证读写功能

```bash
# 检查 Pod 状态
kubectl get pod juicefs-test-pod-dynamic -o wide

# 验证挂载点
kubectl exec -it juicefs-test-pod-dynamic -- df -h | grep juicefs

# 测试写入功能
kubectl exec -it juicefs-test-pod-dynamic -- \
  echo "Dynamic provisioning test - New MinIO storage - $(date)" > /mnt/juicefs/test-file.txt

# 验证文件读取
kubectl exec -it juicefs-test-pod-dynamic -- \
  cat /mnt/juicefs/test-file.txt

# 测试文件列表
kubectl exec -it juicefs-test-pod-dynamic -- \
  ls -la /mnt/juicefs/
```

##### 清理测试资源

```bash
# 删除测试 Pod
kubectl delete pod juicefs-test-pod-dynamic

# 删除测试 PVC
kubectl delete pvc juicefs-test-pvc-dynamic

# 验证清理结果
kubectl get pod juicefs-test-pod-dynamic 2>/dev/null || echo "Pod 已删除"
kubectl get pvc juicefs-test-pvc-dynamic 2>/dev/null || echo "PVC 已删除"
```

**验证成功标准**：

- ✅ PVC 状态为 `Bound`，自动创建了对应的 PV
- ✅ Pod 状态为 `Running`，成功挂载 JuiceFS 存储
- ✅ 能够正常读写文件，数据持久化正常
- ✅ 清理资源成功

#### 5.3.8 CSI 故障排查

##### 常见问题检查

```bash
# 检查 CSI 驱动状态
kubectl get csidrivers

# 检查 CSI 节点信息
kubectl get csinodes

# 查看详细错误信息
kubectl describe pvc your-pvc-name
kubectl describe pv your-pv-name

# 检查事件
kubectl get events --sort-by=.metadata.creationTimestamp
```

##### 手动验证挂载

```bash
# 进入 Pod 验证挂载
kubectl exec -it your-pod-name -- df -h
kubectl exec -it your-pod-name -- ls -la /mnt/juicefs/

# 测试读写
kubectl exec -it your-pod-name -- touch /mnt/juicefs/test-file
kubectl exec -it your-pod-name -- echo "test content" > /mnt/juicefs/test-file
kubectl exec -it your-pod-name -- cat /mnt/juicefs/test-file
```

---

## 6. 清理与验证

### 6.1 删除测试文件系统

```bash
juicefs destroy "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]" temp-fs
```

### 6.2 最终一致性检查

运行全量校验（业务低峰期）：  

```bash
# 使用 JuiceFS 内置校验
juicefs gc "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]" --check

# 使用 MinIO mc 工具校验（可选）
mc diff old-minio/juicefs-bucket new-minio/juicefs-bucket
```

### 6.3 清理旧存储（谨慎操作）

确认迁移成功后，清理旧 MinIO 集群数据：

```bash
# 备份旧集群配置
mc admin config export old-minio > old-minio-config.json

# 删除旧存储桶（不可逆操作）
# mc rb --force old-minio/juicefs-bucket
```

---

## 7. MinIO 特定注意事项

### 7.1 网络配置

- **内网优化**：确保新旧 MinIO 集群间网络带宽充足（建议 10Gbps+）
- **DNS 解析**：使用内网域名时确保所有客户端能正确解析
- **负载均衡**：如果 MinIO 前端有负载均衡器，确保配置正确

### 7.2 存储桶策略和权限

```bash
# 导出旧集群的存储桶策略
mc policy get old-minio/juicefs-bucket > juicefs-bucket-policy.json

# 应用到新集群
mc policy set-json juicefs-bucket-policy.json new-minio/juicefs-bucket

# 检查权限
mc policy list new-minio/juicefs-bucket
```

### 7.3 MinIO 集群配置优化

```bash
# 检查新 MinIO 集群状态
mc admin info new-minio

# 优化配置（根据实际情况调整）
mc admin config set new-minio api requests_max=1000
mc admin config set new-minio api requests_deadline=10s

# 重启服务使配置生效
mc admin service restart new-minio
```

### 7.4 性能调优建议

#### 7.4.1 JuiceFS Sync 优化参数

```bash
# 大文件场景
--threads 64 --list-threads 32 --list-depth 3

# 小文件场景  
--threads 32 --list-threads 16 --list-depth 2

# 网络受限场景
--threads 16 --bwlimit 100  # 限制带宽为 100MB/s
```

#### 7.4.2 MinIO mc 优化参数

```bash
# 设置并发数
export MC_PARALLEL_UPLOADS=16

# 设置分片大小（适合大文件）
export MC_PART_SIZE=128MiB

# 使用 mirror 命令的高级选项
mc mirror --parallel 32 old-minio/bucket new-minio/bucket
```

---

## 8. MySQL 特定注意事项

### 8.1 连接字符串格式

必须使用标准格式：  
`"mysql://[username]:[password]@tcp([host]:[port])/[database_name]"`  

- 特殊字符需 URL 编码
- 确保客户端能访问 MySQL 端口

### 8.2 事务隔离

切换期间确保 MySQL 事务级别为 `REPEATABLE-READ`（JuiceFS 默认要求）

### 8.3 权限验证

确认 JuiceFS 进程有 MySQL 的 `SELECT/UPDATE/INSERT/DELETE` 权限

### 8.4 连接池优化

```sql
-- 检查当前连接数
SHOW PROCESSLIST;

-- 优化连接参数
SET GLOBAL max_connections = 1000;
SET GLOBAL wait_timeout = 28800;
```

---
