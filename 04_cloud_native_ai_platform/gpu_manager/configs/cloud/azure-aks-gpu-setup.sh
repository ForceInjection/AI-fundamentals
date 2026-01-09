#!/bin/bash
# Azure AKS GPU集群部署脚本

set -e

# 配置变量
RESOURCE_GROUP="gpu-cluster-rg"
CLUSTER_NAME="gpu-aks-cluster"
LOCATION="eastus"
NODE_COUNT=2
NODE_VM_SIZE="Standard_NC6s_v3"
KUBERNETES_VERSION="1.28.5"

echo "=== Azure AKS GPU集群部署开始 ==="

# 1. 创建资源组
echo "创建资源组: $RESOURCE_GROUP"
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# 2. 创建AKS集群（CPU节点池）
echo "创建AKS集群: $CLUSTER_NAME"
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --location $LOCATION \
  --kubernetes-version $KUBERNETES_VERSION \
  --node-count 1 \
  --node-vm-size Standard_DS2_v2 \
  --enable-addons monitoring \
  --enable-managed-identity \
  --generate-ssh-keys \
  --network-plugin azure \
  --network-policy azure \
  --load-balancer-sku standard \
  --vm-set-type VirtualMachineScaleSets \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 3

# 3. 添加GPU节点池
echo "添加GPU节点池"
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpunodepool \
  --node-count $NODE_COUNT \
  --node-vm-size $NODE_VM_SIZE \
  --kubernetes-version $KUBERNETES_VERSION \
  --node-taints nvidia.com/gpu=true:NoSchedule \
  --labels accelerator=nvidia-tesla-v100 node-type=gpu \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5 \
  --node-osdisk-size 128 \
  --max-pods 30

# 4. 获取集群凭据
echo "获取集群凭据"
az aks get-credentials \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --overwrite-existing

# 5. 安装NVIDIA GPU Operator
echo "安装NVIDIA GPU Operator"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set operator.defaultRuntime=containerd \
  --wait

# 6. 验证GPU节点
echo "验证GPU节点状态"
kubectl get nodes -l accelerator=nvidia-tesla-v100
kubectl describe nodes -l accelerator=nvidia-tesla-v100

# 7. 部署测试Pod
echo "部署GPU测试Pod"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
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

# 8. 等待Pod完成并查看结果
echo "等待测试Pod完成"
kubectl wait --for=condition=Ready pod/gpu-test --timeout=300s
kubectl logs gpu-test

echo "=== Azure AKS GPU集群部署完成 ==="
echo "集群信息:"
echo "  资源组: $RESOURCE_GROUP"
echo "  集群名: $CLUSTER_NAME"
echo "  位置: $LOCATION"
echo "  GPU节点池: gpunodepool"
echo "  GPU类型: $NODE_VM_SIZE"

# 清理测试Pod
kubectl delete pod gpu-test

echo "部署完成！使用以下命令管理集群:"
echo "  查看节点: kubectl get nodes"
echo "  查看GPU资源: kubectl describe nodes -l accelerator=nvidia-tesla-v100"
echo "  删除集群: az aks delete --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME"