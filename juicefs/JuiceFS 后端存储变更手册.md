# JuiceFS 后端存储变更手册

## 1. 背景

我们使用了 JuiceFS 作为分布式存储解决方案，其架构如下：

- **Kubernetes 集群**
- **JuiceFS CSI Driver**
- **MySQL 数据库**（元数据存储）
- **MinIO 对象存储**（数据存储后端）

现在需要将 `MinIO` 进行升级（或者搬迁），由于某些原因，对外暴露的 IP、bucket、Access Key、Secret Key 等信息发生了变化（数据是完整保留的）。

### 1.1 核心需求

实现最小代价的存储切换，要求如下：

- **不重启业务 Pod**：保持应用服务的连续性
- **不重启 Mount Pod**：避免挂载点的中断

### 1.2 技术原理

JuiceFS 支持热配置切换的核心机制：

1. **元数据驱动**：JuiceFS 客户端通过 MySQL 元数据库获取存储配置
   - 存储后端信息存储在 `jfs_setting` 表中
   - 包括 bucket、access-key、secret-key 等关键配置

2. **配置更新机制**：
   - **主动更新**：通过 `juicefs config` 命令更新配置
   - **被动感知**：客户端在特定操作时重新读取配置
   - **缓存机制**：配置信息会在客户端缓存一定时间

**技术限制**：

- 配置生效可能存在延迟（通常 < 30秒）
- 大文件传输过程中不建议执行切换
- **需要确保新旧存储后端的数据一致性**

---

## 2. 核心切换流程（3步完成）

### 步骤1：备份配置

```bash
# 备份 MySQL 配置
mysqldump -h [mysql_host] -u [username] -p [database_name] jfs_setting > backup_$(date +%Y%m%d_%H%M%S).sql

# 备份 Kubernetes Secret
kubectl get secret juicefs-sc-secret -n juicefs -o yaml > secret-backup_$(date +%Y%m%d_%H%M%S).yaml
```

### 步骤2：更新存储配置（关键步骤）

使用 `juicefs config` 命令更新存储配置：

```bash
# 在 CSI Controller Pod 中执行（推荐方式）
kubectl exec -it -n juicefs juicefs-csi-controller-0 -c juicefs-plugin -- /bin/sh

# 更新 JuiceFS 元数据配置
juicefs config "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]" \
  --bucket [new-minio-endpoint:port/bucket] \
  --access-key [NEW_ACCESS_KEY] \
  --secret-key [NEW_SECRET_KEY]

# 验证配置更新
juicefs status "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]"
```

### 步骤3：更新 CSI Secret 并重启组件

通过 `kubectl patch` 命令更新 Secret 配置：

```bash
# 更新 Secret 配置
BUCKET=$(echo -n "[new-minio-endpoint:port/bucket]" | base64)
ACCESS_KEY=$(echo -n "[NEW_ACCESS_KEY]" | base64)
SECRET_KEY=$(echo -n "[NEW_SECRET_KEY]" | base64)

kubectl patch secret juicefs-sc-secret -n juicefs --type='json' \
  -p="[
    {\"op\": \"replace\", \"path\": \"/data/bucket\", \"value\": \"${BUCKET}\"},
    {\"op\": \"replace\", \"path\": \"/data/access-key\", \"value\": \"${ACCESS_KEY}\"},
    {\"op\": \"replace\", \"path\": \"/data/secret-key\", \"value\": \"${SECRET_KEY}\"}
  ]"

# 重启 CSI 组件
kubectl rollout restart statefulset/juicefs-csi-controller -n juicefs
kubectl rollout restart daemonset/juicefs-csi-node -n juicefs
```

---

## 3. 验证切换效果

### 3.1 快速验证

```bash
# 在 CSI Controller Pod 中测试挂载
kubectl exec -it -n juicefs juicefs-csi-controller-0 -c juicefs-plugin -- /bin/sh

mkdir -p /test_mount
juicefs mount -d "mysql://[username]:[password]@tcp([mysql_host]:3306)/[database_name]" /test_mount

# 测试读写
echo "switch-test-$(date)" > /test_mount/test.txt
cat /test_mount/test.txt
ls -la /test_mount/

# 清理
cd / && umount /test_mount && rm -rf /test_mount
```

### 3.2 业务 Pod 验证

```bash
# 老业务 Pod 无需重启，自动使用新存储
kubectl get pods -n juicefs | grep -E "(juicefs-.*-pvc|mount)"

# 在现有 Mount Pod 中测试
kubectl exec -it -n juicefs [mount-pod-name] -- /bin/sh
echo "new-storage-test-$(date)" > /jfs/[pvc-path]/test.txt
cat /jfs/[pvc-path]/test.txt
```

---

## 4. 故障排查与回滚

### 4.1 快速回滚

如果切换失败，可以快速回滚到原配置：

```bash
# 方式1：从备份恢复 MySQL 配置
mysql -h [mysql_host] -u [username] -p [database_name] < backup_[timestamp].sql

# 方式2：从备份恢复 Secret
kubectl apply -f secret-backup_[timestamp].yaml

# 重启 CSI 组件
kubectl rollout restart statefulset/juicefs-csi-controller -n juicefs
kubectl rollout restart daemonset/juicefs-csi-node -n juicefs
```

### 4.2 常见问题检查

```bash
# 检查 CSI 组件状态
kubectl get pods -n juicefs
kubectl get csidrivers
kubectl get csinodes

# 查看详细错误信息
kubectl describe pvc [pvc-name]
kubectl logs -n juicefs juicefs-csi-controller-0 -c juicefs-plugin

# 检查 Mount Pod 状态
kubectl get pods -n juicefs -l app.kubernetes.io/name=juicefs-mount
```

---

## 5. 数据迁移方案（当数据不一致时）

### 5.1 使用 JuiceFS Sync（推荐）

```bash
# 全量数据迁移
juicefs sync \
  --src-endpoint [OLD_ENDPOINT] \
  --dst-endpoint [NEW_ENDPOINT] \
  s3://[OLD_BUCKET]/ \
  s3://[NEW_BUCKET]/ \
  --src-access-key [OLD_ACCESS_KEY] \
  --src-secret-key [OLD_SECRET_KEY] \
  --dst-access-key [NEW_ACCESS_KEY] \
  --dst-secret-key [NEW_SECRET_KEY] \
  --threads 32

# 数据校验
juicefs sync \
  --src-endpoint [OLD_ENDPOINT] \
  --dst-endpoint [NEW_ENDPOINT] \
  s3://[OLD_BUCKET]/ \
  s3://[NEW_BUCKET]/ \
  --src-access-key [OLD_ACCESS_KEY] \
  --src-secret-key [OLD_SECRET_KEY] \
  --dst-access-key [NEW_ACCESS_KEY] \
  --dst-secret-key [NEW_SECRET_KEY] \
  --check-new \
  --check-all
```

### 5.2 使用 MinIO Client

```bash
# 配置新旧集群
mc alias set old-minio [OLD_ENDPOINT] [OLD_ACCESS_KEY] [OLD_SECRET_KEY]
mc alias set new-minio [NEW_ENDPOINT] [NEW_ACCESS_KEY] [NEW_SECRET_KEY]

# 创建目标存储桶
mc mb new-minio/[BUCKET_NAME]

# 执行数据迁移
mc mirror --overwrite --preserve old-minio/[BUCKET_NAME]/ new-minio/[BUCKET_NAME]/

# 数据校验
mc du old-minio/[BUCKET_NAME]
mc du new-minio/[BUCKET_NAME]
```

---
