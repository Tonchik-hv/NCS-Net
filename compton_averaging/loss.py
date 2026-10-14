"""Физические компоненты функции потерь."""
import torch
import torch.nn as nn

from compton_averaging.normalizer import DataNormalizer


def soft_argmax_freq(intensities, freqs, beta=20.0):
    """
    Дифференцируемая аппроксимация argmax для поиска частоты пика.
    """
    # Нормируем интенсивности в батче от 0 до 1 для стабильности экспоненты
    max_vals = intensities.amax(dim=-1, keepdim=True) + 1e-8
    norm_intensities = intensities / max_vals
    
    # Применяем Softmax с температурой beta. 
    # Чем выше beta, тем ближе результат к строгому argmax.
    weights = torch.softmax(norm_intensities * beta, dim=-1)
    
    # Математическое ожидание: сумма (вероятность * частота)
    # Поскольку freqs одинаковы для всех, используем broadcasting
    peak_freqs = torch.sum(weights * freqs.unsqueeze(0), dim=-1) 
    return peak_freqs

def TV_loss(y_pred):
    """Штраф за негладкость (разность соседних бинов)."""
    return torch.mean(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))

def physics_loss(recon_spec_norm, params_unnorm, freqs, y_raw, normalizer):
    eps = 1e-8
    # Total Variance Loss для нормализованного спектра
    loss_tv = TV_loss(recon_spec_norm.squeeze(1)) 
    
    # Денормализуем и переводим из логарифма
    recon_spec_log = normalizer.unnorm_spec(recon_spec_norm).squeeze(1)
    recon_spec_log = torch.nan_to_num(recon_spec_log, nan=-9.0, posinf=8.0, neginf=-12.0)
    recon_spec_log = torch.clamp(recon_spec_log, min=-12.0, max=8.0)
    recon_spec_lin = torch.clamp(10.0 ** recon_spec_log - eps, min=0.0)

    y_log = torch.nan_to_num(y_raw.squeeze(1), nan=-9.0, posinf=8.0, neginf=-12.0)
    y_log = torch.clamp(y_log, min=-12.0, max=8.0)
    y_lin = torch.clamp(10.0 ** y_log - eps, min=0.0)
    
    # dw = freqs[:, 1] - freqs[:, 0] 
    freqs_1d = freqs[0] 
    dw = freqs_1d[1] - freqs_1d[0]
    
    # # 1. Площадь # Проверить это предположение
    # area_pred = (torch.sum(recon_spec_lin, dim=-1) * dw).view(-1) 
    # area_true = (torch.sum(y_lin, dim=-1) * dw).view(-1)
    # loss_area = torch.mean(((area_pred - area_true) / (area_true + 1e-8))**2)
    # # loss_area = nn.HuberLoss()(torch.log10(area_pred + eps), torch.log10(area_true + eps)) # Деление тоже дает нестабильный результат

    # # 2. Пик частоты
    # peak_pred = soft_argmax_freq(recon_spec_lin, freqs_1d, beta=20.0) # так как argmax не пропускает градиенты
    # # peak_true = 1.0 / (1.0 + a0**2)
    # peak_indices = torch.argmax(y_lin, dim=-1) 
    # peak_true = freqs_1d[peak_indices] 
    # loss_peak = torch.mean(((peak_pred.view(-1) - peak_true) / (peak_true + 1e-8))**2)
    # # loss_peak = nn.HuberLoss()(peak_pred.view(-1), peak_true.view(-1))

    # # 3. Подтягиваем пик по амплитуде
    # ampl_pred = recon_spec_lin.amax(dim=-1) # Размерность dim=-1 - критично, иначе ищется максимум по бачу
    # ampl_true = y_lin.amax(dim=-1)
    # loss_ampl = torch.mean(((ampl_pred - ampl_true) / (ampl_true + 1e-8))**2)
    # # loss_ampl = nn.HuberLoss()(ampl_pred.view(-1), ampl_true.view(-1))
    # return loss_area, loss_peak, loss_ampl, loss_tv

    # 1. Площадь
    area_pred = (torch.sum(recon_spec_lin, dim=-1) * dw).view(-1) 
    area_true = (torch.sum(y_lin, dim=-1) * dw).view(-1)
    # Нормируем на истинную площадь (detach отключает расчет градиентов по делителю)
    scale_area = torch.clamp(area_true.detach(), min=1e-3)
    loss_area = nn.HuberLoss()(area_pred / scale_area, area_true / scale_area)

    # 2. Пик частоты
    peak_pred = soft_argmax_freq(recon_spec_lin, freqs_1d, beta=100.0) 
    peak_indices = torch.argmax(y_lin, dim=-1) 
    peak_true = freqs_1d[peak_indices] 
    # Частота имеет порядок ~0.1 - 2.0, поэтому сильный взрыв тут реже, но защитим и ее
    scale_peak = torch.clamp(peak_true.detach(), min=0.1)
    loss_peak = nn.HuberLoss()(peak_pred.view(-1) / scale_peak, peak_true.view(-1) / scale_peak)

    # 3. Амплитуда
    ampl_pred_lin = recon_spec_lin.amax(dim=-1)
    ampl_true_lin = y_lin.amax(dim=-1)
    scale_ampl = torch.clamp(ampl_true_lin.detach(), min=1e-3)
    loss_ampl = nn.HuberLoss()(ampl_pred_lin / scale_ampl, ampl_true_lin / scale_ampl)
    
    return loss_area, loss_peak, loss_ampl, loss_tv
