"""Копия compton_averaging/training.py с путями к графикам и возвратом метрик (для CV)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from compton_averaging.loss import physics_loss
from compton_averaging.normalizer import DataNormalizer


def plot_history(train_loss, val_loss, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("История обучения (Total Loss)")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def train_model(
    model,
    train_loader,
    val_loader,
    normalizer,
    epochs=100,
    lr=3e-4,
    save_path="model_with_averaging.pth",
    history_plot_path=None,
    resume_from_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalizer.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    criterion_main = nn.HuberLoss()
    best_val_loss = float("inf")
    start_epoch = 0
    if resume_from_path:
        ckpt = torch.load(resume_from_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Продолжение обучения с эпохи {start_epoch + 1} (best_val_loss={best_val_loss:.4f})")

    LAMBDA_PHYSICS = 0.01
    LAMBDA_TV = 0.0001

    history = {"train": [], "val": []}

    print(f"Старт обучения на {device}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_accum = 0
        for p_raw, y_raw, freqs in train_loader:
            p_raw, y_raw, freqs = p_raw.to(device), y_raw.to(device), freqs.to(device)

            p_norm = normalizer.norm_params(p_raw)
            y_norm = normalizer.norm_spec(y_raw)

            recon_pred = model(p_norm)

            loss_main = criterion_main(recon_pred, y_norm)
            loss_area, loss_peak, loss_ampl, loss_tv = physics_loss(
                recon_pred, p_raw, freqs, y_raw, normalizer
            )

            if epoch < 20:
                current_lambda_physics = 0.0
            else:
                current_lambda_physics = LAMBDA_PHYSICS * min(1.0, (epoch - 20) / 10.0)

            loss = (
                loss_main
                + current_lambda_physics * (loss_area + loss_peak + loss_ampl)
                + LAMBDA_TV * loss_tv
            )

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.step()
            train_loss_accum += loss.item()

        avg_train = train_loss_accum / len(train_loader)

        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for p_raw, y_raw, freqs in val_loader:
                p_raw, y_raw, freqs = p_raw.to(device), y_raw.to(device), freqs.to(device)

                p_norm = normalizer.norm_params(p_raw)
                y_norm = normalizer.norm_spec(y_raw)

                recon_pred = model(p_norm)

                loss_main = criterion_main(recon_pred, y_norm)
                loss_area, loss_peak, loss_ampl, loss_tv = physics_loss(
                    recon_pred, p_raw, freqs, y_raw, normalizer
                )

                loss = (
                    loss_main
                    + current_lambda_physics * (loss_area + loss_peak + loss_ampl)
                    + LAMBDA_TV * loss_tv
                )
                val_loss_accum += loss.item()

        avg_val = val_loss_accum / len(val_loader)

        scheduler.step(avg_val)

        history["train"].append(avg_train)
        history["val"].append(avg_val)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {current_lr:.6e}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "norm_p_mean": normalizer.p_mean,
                    "norm_p_std": normalizer.p_std,
                    "norm_y_mean": normalizer.y_mean,
                    "norm_y_std": normalizer.y_std,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1,
                },
                save_path,
            )

    if history_plot_path:
        plot_history(history["train"], history["val"], history_plot_path)

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train"][-1] if history["train"] else None,
        "final_val_loss": history["val"][-1] if history["val"] else None,
        "history": history,
        "device": str(device),
    }
