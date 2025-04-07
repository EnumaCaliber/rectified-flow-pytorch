import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from datetime import datetime
import time
from scipy.ndimage import gaussian_filter
from einops import rearrange
from itertools import cycle

# 设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---- 改进的Poisson数据集类 ----
class PoissonImageDataset(Dataset):
    def __init__(self, path, normalize=True, normalization_precision=1e-10):
        super().__init__()
        data = np.load(path)
        self.images = data["image"]  # shape: (N, H, W)
        self.normalize = normalize
        self.normalization_precision = normalization_precision
        
        # 保存数据集统计信息，用于后期反归一化
        self.global_min = float(np.min(self.images))
        self.global_max = float(np.max(self.images))
        
        print(f"数据集形状: {self.images.shape}")
        print(f"数据范围: [{self.global_min}, {self.global_max}]")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx].copy()  # 使用copy防止修改原始数据
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

# ---- 自定义TensorBoard Trainer ----
from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer

class EnhancedTensorBoardTrainer(Trainer):
    def __init__(
        self,
        *args,
        log_dir='./runs',
        log_images_every=100,
        apply_post_processing=True,
        dataset_stats=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # 创建TensorBoard日志目录
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(log_dir, current_time)
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard 日志目录: {self.log_dir}")
        
        self.log_images_every = log_images_every
        self.start_time = time.time()
        self.apply_post_processing = apply_post_processing
        self.dataset_stats = dataset_stats  # 保存数据集统计信息，用于反归一化
        
        # 尝试记录模型结构
        try:
            dummy_input = torch.randn(1, 1, 96, 96).to(self.accelerator.device)
            dummy_t = torch.tensor([0.5]).to(self.accelerator.device)
            self.writer.add_graph(self.model.model, (dummy_input, dummy_t))
        except Exception as e:
            print(f"无法添加模型图: {e}")
    
    def log(self, *args, **kwargs):
        """重写log方法以添加TensorBoard支持"""
        if not self.is_main:
            return
            
        # 调用原始log方法
        result = super().log(*args, **kwargs)
        
        # 向TensorBoard添加标量数据
        step = kwargs.get('step', 0)
        
        # 记录字典中的所有值
        if args and isinstance(args[0], dict):
            for k, v in args[0].items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f'train/{k}', v, step)
                    
        return result
    
    def post_process_image(self, tensor):
        """对生成的图像应用后处理以减少噪声和增强对比度"""
        if not self.apply_post_processing:
            return tensor
        
        # 将张量转换到CPU并转为numpy以便处理
        if tensor.dim() == 3:  # 单个图像 [C,H,W]
            img_np = tensor.detach().cpu().numpy()
            if img_np.shape[0] == 1:  # 单通道
                # 1. 高斯滤波去噪
                filtered = gaussian_filter(img_np[0], sigma=0.5)
                
                # 2. 增强对比度
                p2, p98 = np.percentile(filtered, (2, 98))
                if p98 > p2:
                    enhanced = np.clip((filtered - p2) / (p98 - p2), 0, 1)
                    img_np[0] = enhanced
                
                return torch.tensor(img_np, dtype=tensor.dtype, device=tensor.device)
        
        # 处理批量图像
        batch_np = tensor.detach().cpu().numpy()
        for i in range(batch_np.shape[0]):
            # 高斯滤波去噪
            filtered = gaussian_filter(batch_np[i, 0], sigma=0.5)
            
            # 增强对比度
            p2, p98 = np.percentile(filtered, (2, 98))
            if p98 > p2:
                enhanced = np.clip((filtered - p2) / (p98 - p2), 0, 1)
                batch_np[i, 0] = enhanced
        
        return torch.tensor(batch_np, dtype=tensor.dtype, device=tensor.device)
        
    def sample(self, fname):
        """增强采样方法，提高生成质量并使用彩色热力图可视化"""
        eval_model = self.ema_model if self.use_ema and self.ema_model is not None else self.model
        dl = cycle(self.dl)
        mock_data = next(dl)
        data_shape = mock_data.shape[1:]

        with torch.no_grad():
            # 标准采样调用
            sampled = eval_model.sample(batch_size=self.num_samples, data_shape=data_shape)
        
        # 应用后处理，减少噪声，增强对比度
        processed_sampled = self.post_process_image(sampled)
        
        # 准备网格图像
        grid_sampled = rearrange(processed_sampled, '(row col) c h w -> c (row h) (col w)', row=self.num_sample_rows)
        grid_sampled.clamp_(0., 1.)
        
        # 保存标准图像
        save_image(grid_sampled, fname)
        
        # 额外保存彩色热力图版本
        colored_fname = str(fname).replace('.png', '_colored.png')
        self.save_colored_image(grid_sampled, colored_fname)
        
        return grid_sampled
    
    def save_colored_image(self, tensor, fname):
        """保存彩色热力图版本"""
        img = tensor.cpu().numpy()
        if img.shape[0] == 1:  # 单通道图像
            # 移除通道维度
            img = img.squeeze(0)
            
            # 使用viridis色彩映射生成热力图
            plt.figure(figsize=(10, 10))
            plt.imshow(img, cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(fname, bbox_inches='tight', dpi=300)
            plt.close()
    
    def log_images(self, images, step):
        """重写图像记录方法，增加彩色可视化支持"""
        if not self.is_main:
            return
            
        # 向TensorBoard添加标准图像
        self.writer.add_image('generated_samples', images, step)
        
        # 添加彩色热力图版本
        try:
            img_np = images.cpu().numpy()
            if img_np.shape[0] == 1:  # 单通道
                # 使用plt创建彩色图
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(img_np.squeeze(0), cmap='viridis')
                plt.colorbar(im)
                ax.axis('off')
                
                # 将plt图转换为张量
                self.writer.add_figure('generated_heatmap', fig, step)
                plt.close(fig)
        except Exception as e:
            print(f"无法添加彩色热力图: {e}")
    
    def forward(self):
        """重写训练循环，增强监控和可视化"""
        dl = self.get_dataloader()
        
        for ind in range(self.num_train_steps):
            step = ind + 1
            
            self.model.train()
            
            data = next(dl)
            
            # 记录原始数据样本
            if step == 1 or step % self.log_images_every == 0:
                if self.is_main:
                    # 可视化训练数据
                    grid = make_grid(data[:16].cpu(), nrow=4, normalize=True)
                    self.writer.add_image('training_samples', grid, step)
                    
                    # 添加彩色热力图
                    try:
                        if data.shape[1] == 1:  # 单通道
                            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                            for i in range(4):
                                row, col = i // 2, i % 2
                                im = ax[row, col].imshow(data[i, 0].cpu().numpy(), cmap='viridis')
                                plt.colorbar(im, ax=ax[row, col])
                                ax[row, col].set_title(f'Sample {i}')
                                ax[row, col].axis('off')
                            self.writer.add_figure('training_heatmap', fig, step)
                            plt.close(fig)
                    except Exception as e:
                        print(f"无法添加训练热力图: {e}")
            
            # 计算损失
            if self.return_loss_breakdown:
                loss, loss_breakdown = self.model(data, return_loss_breakdown=True)
                if self.is_main:
                    self.log(loss_breakdown._asdict(), step=step)
                    self.writer.add_scalar('train/total_loss', loss.item(), step)
            else:
                loss = self.model(data)
                if self.is_main:
                    self.writer.add_scalar('train/loss', loss.item(), step)
            
            # 打印训练进度
            if self.is_main:
                elapsed = time.time() - self.start_time
                iter_per_sec = step / elapsed if elapsed > 0 else 0
                remaining = (self.num_train_steps - step) / iter_per_sec if iter_per_sec > 0 else 0
                
                self.accelerator.print(
                    f'[{step}/{self.num_train_steps}] '
                    f'loss: {loss.item():.4f} | '
                    f'速度: {iter_per_sec:.2f}it/s | '
                    f'剩余: {int(remaining//60)}分{int(remaining%60)}秒'
                )
            
            # 反向传播和优化
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 更新EMA模型
            if getattr(self.model, 'use_consistency', False):
                self.model.ema_model.update()
            
            if self.is_main and self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()
            
            self.accelerator.wait_for_everyone()
            
            # 定期保存和可视化
            if self.is_main:
                # 记录学习率
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f'train/lr_group_{i}', param_group['lr'], step)
                
                # 定期保存结果和检查点
                if step % self.save_results_every == 0:
                    sampled = self.sample(fname=str(self.results_folder / f'results.{step}.png'))
                    self.log_images(sampled, step=step)
                
                if step % self.checkpoint_every == 0:
                    self.save(f'checkpoint.{step}.pt')
                    # 记录模型参数分布
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            self.writer.add_histogram(f"parameters/{name}", param.data, step)
                
            self.accelerator.wait_for_everyone()
        
        print('训练完成')
        if self.is_main:
            self.writer.close()
    
    def get_dataloader(self):
        """辅助方法：获取循环的dataloader"""
        return cycle(self.dl)

# ---- 主函数：初始化并训练模型 ----
def main():
    # 加载数据集
    poisson_dataset = PoissonImageDataset(
        "/data3/bx/2D-poisson-PDE-Solver/poisson_dataset_5000.npz",
        normalize=True,
        normalization_precision=1e-10  # 更高精度的归一化
    )
    
    # 定义Unet模型 - 删除不支持的参数
    model = Unet(
        dim=64,  # 增加基础通道数
        channels=1,  # Poisson图像是单通道
        dim_mults=(1, 2, 4, 8),  # 保持多尺度结构
    )
    
    # 构建RectifiedFlow实例
    rectified_flow = RectifiedFlow(
        model,
        predict="flow",  # 使用流预测
        loss_fn="mse",  # 如果huber不支持，回退到mse
        clip_values=(0., 1.),  # 值域裁剪
    )
    
    # 确保目录存在
    os.makedirs("./results_poisson_enhanced", exist_ok=True)
    os.makedirs("./checkpoints_poisson_enhanced", exist_ok=True)
    
    # 使用增强版Trainer
    trainer = EnhancedTensorBoardTrainer(
        rectified_flow=rectified_flow,
        dataset=poisson_dataset,
        num_train_steps=70000,  # 增加训练步数
        batch_size=64,
        results_folder="./results_poisson_enhanced",
        checkpoints_folder="./checkpoints_poisson_enhanced",
        learning_rate=5e-5,  # 使用更小的学习率提高稳定性
        save_results_every=100,  # 更频繁地保存结果
        checkpoint_every=500,  # 每500步保存一次模型
        num_samples=16,
        use_ema=True,
        apply_post_processing=True,  # 启用后处理
        dataset_stats={
            'min': poisson_dataset.global_min,
            'max': poisson_dataset.global_max
        },
        log_dir='./tensorboard_logs_enhanced',
        log_images_every=50,
        # 使用标准参数，避免API不兼容
        ema_kwargs={"beta": 0.995}  # 增加EMA强度
    )
    
    # 开始训练
    trainer()

if __name__ == "__main__":
    main()