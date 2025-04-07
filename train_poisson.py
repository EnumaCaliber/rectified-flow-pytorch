import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# ---- 加载 Poisson 数据集 ----
import numpy as np
from torch.utils.data import Dataset


class PoissonImageDataset(Dataset):

    def __init__(self, path, normalize=True):
        super().__init__()
        data = np.load(path)
        self.images = data["image"]  # shape: (N, H, W)
        self.normalize = normalize

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        tensor = torch.tensor(img, dtype=torch.float32)
        tensor_min, tensor_max = tensor.min(), tensor.max()
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
        tensor = tensor.unsqueeze(0)
        _, H, W = tensor.shape

        # 计算需要裁掉多少像素以满足能被 4 整除
        new_H = H - (H % 8)
        new_W = W - (W % 8)

        # 从中心区域裁剪
        tensor = tensor[:, :new_H, :new_W]

        return tensor


# ---- 模型与 Trainer ----
from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer

# 定义 Unet 模型
model = Unet(
    dim=64,
    channels = 1,  # Poisson 图像是单通道
)

# 构建 RectifiedFlow 实例
rectified_flow = RectifiedFlow(
    model,
)

# 加载数据集
poisson_dataset = PoissonImageDataset("poisson_dataset_5000.npz")

# 启动 Trainer
trainer = Trainer(
    rectified_flow=rectified_flow,
    dataset=poisson_dataset,
    num_train_steps=70000,
    results_folder="./results_poisson",
    checkpoints_folder="./checkpoints_poisson",
)

# 开始训练
trainer()
