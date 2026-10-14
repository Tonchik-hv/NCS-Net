"""Нормализация параметров и спектров."""
import torch


class DataNormalizer:
    def __init__(self):
        self.p_mean = None; self.p_std = None
        self.y_mean = None; self.y_std = None
        
    def fit(self, dataloader):
        p_list, y_list = [], []
        for p, y, _ in dataloader:
            p_list.append(p); y_list.append(y)
        P = torch.cat(p_list, dim=0)
        Y = torch.cat(y_list, dim=0)
        
        self.p_mean = P.mean(dim=0, keepdim=True)
        self.p_std = P.std(dim=0, keepdim=True) + 1e-8
        self.y_mean = Y.mean(dim=0, keepdim=True)
        # На угловом датасете часть бинов почти константная, std~0 вызывает взрыв при нормализации.
        self.y_std = torch.clamp(Y.std(dim=0, keepdim=True), min=1e-3)
        
    def to(self, device):
        if self.p_mean is not None:
            self.p_mean = self.p_mean.to(device)
            self.p_std = self.p_std.to(device)
            self.y_mean = self.y_mean.to(device)
            self.y_std = self.y_std.to(device)
    
    def cpu(self):
        if self.p_mean is not None:
            self.p_mean = self.p_mean.cpu()
            self.p_std = self.p_std.cpu()
            self.y_mean = self.y_mean.cpu()
            self.y_std = self.y_std.cpu()
        return self
        
    def norm_params(self, p): return (p - self.p_mean) / self.p_std
    def unnorm_params(self, p_norm): return p_norm * self.p_std + self.p_mean
    def norm_spec(self, y): return (y - self.y_mean) / self.y_std
    def unnorm_spec(self, y_norm): return y_norm * self.y_std + self.y_mean
