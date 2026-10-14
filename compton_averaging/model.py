"""Архитектура нейросети DirectComptonModel."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaussian1d(nn.Module):
    def __init__(self, kernel_size=21, init_sigma=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor([init_sigma], dtype=torch.float32))
        self.register_buffer('x', torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32))

    def forward(self, x):
        sigma = torch.clamp(self.sigma, min=0.1)
        gaussian = torch.exp(-(self.x**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        weight = gaussian.view(1, 1, self.kernel_size)
        pad = self.kernel_size // 2
        return F.conv1d(x, weight, padding=pad)


class DirectComptonModel(nn.Module):
    def __init__(self, params_dim=3):
        super().__init__()
        
        # Блок 1: Проекция и расширение признаков (Dense)
        self.dense = nn.Sequential(
            nn.Linear(params_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            # Увеличиваем до размерности (C=64, Length=20)
            nn.Linear(256, 64 * 20),
            nn.GELU()
        )
        
        # # Блок 2: Формирование спектра (Conv1DTranspose)
        # self.conv_transpose = nn.Sequential(
        #     # Вход: (B, 64, 20) -> Выход: (B, 32, 100)
        #     nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5, padding=2, output_padding=4),
        #     nn.BatchNorm1d(32),
        #     nn.GELU(),
            
        #     # Вход: (B, 32, 100) -> Выход: (B, 16, 500)
        #     nn.ConvTranspose1d(32, 16, kernel_size=5, stride=5, padding=2, output_padding=4),
        #     nn.BatchNorm1d(16),
        #     nn.GELU(),
            
        #     # Вход: (B, 16, 500) -> Выход: (B, 8, 2500)
        #     nn.ConvTranspose1d(16, 8, kernel_size=5, stride=5, padding=2, output_padding=4),
        #     nn.BatchNorm1d(8),
        #     nn.GELU(),
            
        #     # Финальный слой для получения размерности спектра
        #     # Оставляем линейную активацию, так как таргет проходит Z-нормализацию
        #     nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1)
        # )

        # Блок 2: Формирование спектра (Upsample + Conv1d)
        self.conv_transpose = nn.Sequential(
            # Вход: (B, 64, 20) -> Выход: (B, 32, 100)
            nn.Upsample(scale_factor=5.0, mode='linear', align_corners=False),
            nn.Conv1d(64, 32, kernel_size=13, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            # Вход: (B, 32, 100) -> Выход: (B, 16, 500)
            nn.Upsample(scale_factor=5.0, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, kernel_size=13, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.GELU(),
            
            # Вход: (B, 16, 500) -> Выход: (B, 8, 2500)
            nn.Upsample(scale_factor=5.0, mode='linear', align_corners=False),
            nn.Conv1d(16, 8, kernel_size=13, stride=1, padding='same'),
            nn.BatchNorm1d(8),
            nn.GELU(),
            
            # Финальный слой для получения размерности спектра (1 канал)
            nn.Conv1d(8, 1, kernel_size=9, stride=1, padding='same')
        )

        # Блок 3: Гауссово размытие (Conv1D)
        # self.smooth = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        # with torch.no_grad():
        #     # Создаем гауссово ядро (необучаемое)
        #     kernel = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]).view(1, 1, -1)
        #     self.smooth.weight.copy_(kernel)
        # self.smooth.weight.requires_grad = False

        # kernel_size = 21  
        # sigma = 5.0       
        # padding = kernel_size // 2
        # self.smooth = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        # with torch.no_grad():
        #     x_coord = torch.arange(kernel_size) - (kernel_size - 1) / 2
        #     kernel = torch.exp(-(x_coord**2) / (2 * sigma**2))
        #     kernel = kernel / kernel.sum()
        #     self.smooth.weight.copy_(kernel.view(1, 1, -1))
        # self.smooth.weight.requires_grad = False

        self.smooth = LearnableGaussian1d(kernel_size=21, init_sigma=2.0)

    def forward(self, x_params):
        # 1. Проекция
        x = self.dense(x_params)
        # 2. Reshape в (Batch, Channels, Length)
        x = x.view(-1, 64, 20)
        # 3. Up-sampling свертками
        recon_spec = self.conv_transpose(x)
        # Сглаживание спектра
        recon_spec_smooth = self.smooth(recon_spec) # Финальная «полировка» спектра
        return recon_spec_smooth    
