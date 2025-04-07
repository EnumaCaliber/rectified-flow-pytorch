import torch
from torch.utils.data import DataLoader

# ---- 加载 Poisson 数据集 ----
import numpy as np
from torch.utils.data import Dataset


class PoissonImageDataset(Dataset):
    def __init__(self, path, normalize=True):
        super().__init__()
        data = np.load(path)
        self.images = data["image"]  # shape: (N, H, W)
        self.normalize = normalize
        self.global_min = float(np.min(self.images))
        self.global_max = float(np.max(self.images))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        tensor = torch.tensor(img, dtype=torch.float32)

        if self.normalize:
            # 高精度归一化到[0,1]区间
            tensor = (tensor - self.global_min) / (self.global_max - self.global_min + self.normalization_precision)

        tensor = tensor.unsqueeze(0)  # 添加通道维度
        _, H, W = tensor.shape

        # 计算需要裁掉多少像素以满足能被8整除
        new_H = H - (H % 8)
        new_W = W - (W % 8)

        # 裁剪图像
        tensor = tensor[:, :new_H, :new_W]

        return tensor


# ---- 模型与 Trainer ----
from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer

# 定义 Unet 模型
model = Unet(
    dim=64,
    channels=1,  # Poisson 图像是单通道
)

# 构建 RectifiedFlow 实例
rectified_flow = RectifiedFlow(
    model,
)

# 加载数据集
poisson_dataset = PoissonImageDataset(
        "poisson_dataset_5000.npz"
)

# 启动 Trainer
trainer = Trainer(
    rectified_flow=rectified_flow,
    dataset=poisson_dataset,
    num_train_steps=70000,
    batch_size=16,
    results_folder="./results_poisson",
    checkpoints_folder="./checkpoints_poisson",
    save_sample_every=100
)

# 开始训练
trainer()