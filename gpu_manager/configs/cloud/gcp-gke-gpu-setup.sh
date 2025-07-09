#!/bin/bash
# GCP GKE GPU集群部署脚本

set -e

# 配置变量
PROJECT_ID="your-project-id"
CLUSTER_NAME="gpu-gke-cluster"
ZONE="us-central1-a"
REGION="us-central1"
NODE_COUNT=2
MACHINE_TYPE="n1-standard-4"
ACCELERATOR_TYPE="nvidia-tesla-v100"
ACCELERATOR_COUNT=1
KUBERNETES_VERSION="1.28.5-gke.1217000"

echo "=== GCP GKE GPU集群部署开始 ==="

# 1. 设置项目
echo "设置GCP项目: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# 2. 启用必要的API
echo "启用必要的GCP API"
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com

# 3. 创建GPU集群
echo "创建GKE GPU集群: $CLUSTER_NAME"
gcloud container clusters create $CLUSTER_NAME \
  --zone $ZONE \
  --cluster-version $KUBERNETES_VERSION \
  --machine-type $MACHINE_TYPE \
  --accelerator type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT \
  --num-nodes $NODE_COUNT \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5 \
  --enable-autorepair \
  --enable-autoupgrade \
  --disk-size 100GB \
  --disk-type pd-ssd \
  --image-type COS_CONTAINERD \
  --enable-ip-alias \
  --network default \
  --subnetwork default \
  --enable-network-policy \
  --enable-cloud-logging \
  --enable-cloud-monitoring \
  --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
  --node-labels accelerator=nvidia-tesla-v100,node-type=gpu \
  --node-taints nvidia.com/gpu=true:NoSchedule

# 4. 获取集群凭据
echo "获取集群凭据"
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE

# 5. 安装NVIDIA GPU驱动（DaemonSet方式）
echo "安装NVIDIA GPU驱动"
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# 6. 等待驱动安装完成
echo "等待GPU驱动安装完成"
kubectl rollout status daemonset nvidia-driver-installer -n kube-system --timeout=600s

# 7. 安装NVIDIA Device Plugin
echo "安装NVIDIA Device Plugin"
kubectl apply -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/cluster/addons/device-plugins/nvidia-gpu/daemonset.yaml

# 8. 等待Device Plugin就绪
echo "等待Device Plugin就绪"
kubectl rollout status daemonset nvidia-gpu-device-plugin -n kube-system --timeout=300s

# 9. 验证GPU节点
echo "验证GPU节点状态"
kubectl get nodes -l accelerator=nvidia-tesla-v100
kubectl describe nodes -l accelerator=nvidia-tesla-v100

# 10. 检查GPU资源
echo "检查GPU资源分配"
kubectl get nodes -o json | jq '.items[] | select(.metadata.labels."accelerator"=="nvidia-tesla-v100") | {name: .metadata.name, gpu: .status.allocatable."nvidia.com/gpu"}'

# 11. 部署测试Pod
echo "部署GPU测试Pod"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test-gke
spec:
  containers:
  - name: gpu-test
    image: nvidia/cuda:12.4-runtime-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  nodeSelector:
    accelerator: nvidia-tesla-v100
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  restartPolicy: Never
EOF

# 12. 等待Pod完成并查看结果
echo "等待测试Pod完成"
kubectl wait --for=condition=Ready pod/gpu-test-gke --timeout=300s
kubectl logs gpu-test-gke

# 13. 创建GPU工作负载示例
echo "创建GPU工作负载示例"
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-workload-example
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gpu-workload
  template:
    metadata:
      labels:
        app: gpu-workload
    spec:
      containers:
      - name: gpu-container
        image: tensorflow/tensorflow:2.13.0-gpu
        command: ["python", "-c"]
        args:
          - |
            import tensorflow as tf
            print("TensorFlow version:", tf.__version__)
            print("GPU devices:", tf.config.list_physical_devices('GPU'))
            with tf.device('/GPU:0'):
              a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
              b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
              c = tf.matmul(a, b)
              print("Matrix multiplication result:", c.numpy())
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "2Gi"
            cpu: "1"
      nodeSelector:
        accelerator: nvidia-tesla-v100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
EOF

echo "=== GCP GKE GPU集群部署完成 ==="
echo "集群信息:"
echo "  项目ID: $PROJECT_ID"
echo "  集群名: $CLUSTER_NAME"
echo "  区域: $ZONE"
echo "  GPU类型: $ACCELERATOR_TYPE"
echo "  节点数: $NODE_COUNT"

# 清理测试Pod
kubectl delete pod gpu-test-gke

echo "部署完成！使用以下命令管理集群:"
echo "  查看节点: kubectl get nodes"
echo "  查看GPU资源: kubectl describe nodes -l accelerator=nvidia-tesla-v100"
echo "  查看工作负载: kubectl get deployment gpu-workload-example"
echo "  删除集群: gcloud container clusters delete $CLUSTER_NAME --zone $ZONE"