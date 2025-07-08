# train_distributed.py - 应用程序代码无需修改
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    
    # 获取GPU设备（HAMi透明提供vGPU）
    device = torch.device('cuda')
    
    # 创建模型并移动到GPU
    model = MyModel().to(device)
    model = DDP(model)
    
    # 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            # HAMi自动进行资源控制和调度
            outputs = model(batch.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(784, 10)
    
    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    main()