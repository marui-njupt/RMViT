from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import loaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class MetricSums:
    sse: float = 0.0
    target_ss: float = 0.0
    n_elem: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        diff = pred - target
        self.sse += float(torch.sum(diff * diff).item())
        self.target_ss += float(torch.sum(target * target).item())
        self.n_elem += int(target.numel())

    def rmse(self) -> float:
        if self.n_elem == 0:
            return float("nan")
        return float(np.sqrt(self.sse / self.n_elem))

    def nmse(self) -> float:
        if self.target_ss <= 1e-12:
            return float("nan")
        return float(self.sse / self.target_ss)

    def nmse_db(self) -> float:
        v = self.nmse()
        if not np.isfinite(v) or v <= 0:
            return float("nan")
        return float(10.0 * np.log10(v))


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return torch.mean((pred - target) ** 2)


def sampled_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_rm: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    obs_mask = (torch.abs(x_rm) > 1e-8).float()

    err2 = (pred - target) ** 2
    num = torch.sum(obs_mask * err2, dim=(1, 2, 3))
    den = torch.sum(obs_mask, dim=(1, 2, 3)) + eps

    loss_per_sample = num / den
    return torch.mean(loss_per_sample)


def stage2_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_rm: torch.Tensor,
    lambda1: float,
    lambda2: float,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if normalize:
        pred_n = pred / 256.0
        target_n = target / 256.0
        x_rm_n = x_rm / 256.0
    else:
        pred_n = pred
        target_n = target
        x_rm_n = x_rm

    loss_mse = mse_loss(pred_n, target_n)
    loss_w = sampled_mse_loss(pred_n, target_n, x_rm_n)

    loss = lambda1 * loss_mse + lambda2 * loss_w
    return loss, loss_mse, loss_w


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


@torch.no_grad()
def run_stage1(
    stage1: nn.Module,
    x: torch.Tensor,
    rm_ch: int,
    bm_ch: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    x_rm = x[:, rm_ch:rm_ch + 1, :, :]
    x_bm = x[:, bm_ch:bm_ch + 1, :, :]

    coarse = stage1(x_rm, x_bm)
    return coarse, x_rm, x_bm


def train_one_epoch_stage2(
    stage1: nn.Module,
    stage2: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    rm_ch: int,
    bm_ch: int,
    lambda1: float,
    lambda2: float,
) -> Tuple[float, float, float]:
    stage1.eval()
    stage2.train()

    total_loss_sum = 0.0
    total_mse_sum = 0.0
    total_lw_sum = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            coarse, x_rm, x_bm = run_stage1(stage1, x, rm_ch, bm_ch)

        optimizer.zero_grad(set_to_none=True)

        pred = stage2(x_coarse=coarse, x_rm=x_rm, x_bm=x_bm)

        loss, loss_mse, loss_w = stage2_total_loss(
            pred=pred,
            target=y,
            x_rm=x_rm,
            lambda1=lambda1,
            lambda2=lambda2,
            normalize=True,
        )

        loss.backward()
        optimizer.step()

        bs = int(x.size(0))
        total_loss_sum += float(loss.item()) * bs
        total_mse_sum += float(loss_mse.item()) * bs
        total_lw_sum += float(loss_w.item()) * bs
        total_count += bs

    return (
        total_loss_sum / max(1, total_count),
        total_mse_sum / max(1, total_count),
        total_lw_sum / max(1, total_count),
    )


@torch.no_grad()
def validate_one_epoch_stage2(
    stage1: nn.Module,
    stage2: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rm_ch: int,
    bm_ch: int,
    lambda1: float,
    lambda2: float,
) -> Tuple[float, float, float]:
    stage1.eval()
    stage2.eval()

    total_loss_sum = 0.0
    total_mse_sum = 0.0
    total_lw_sum = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        coarse, x_rm, x_bm = run_stage1(stage1, x, rm_ch, bm_ch)
        pred = stage2(x_coarse=coarse, x_rm=x_rm, x_bm=x_bm)

        loss, loss_mse, loss_w = stage2_total_loss(
            pred=pred,
            target=y,
            x_rm=x_rm,
            lambda1=lambda1,
            lambda2=lambda2,
            normalize=True,
        )

        bs = int(x.size(0))
        total_loss_sum += float(loss.item()) * bs
        total_mse_sum += float(loss_mse.item()) * bs
        total_lw_sum += float(loss_w.item()) * bs
        total_count += bs

    return (
        total_loss_sum / max(1, total_count),
        total_mse_sum / max(1, total_count),
        total_lw_sum / max(1, total_count),
    )


@torch.no_grad()
def evaluate_metrics_stage2(
    stage1: nn.Module,
    stage2: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rm_ch: int,
    bm_ch: int,
) -> Tuple[float, float, float]:
    stage1.eval()
    stage2.eval()

    sums = MetricSums()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        coarse, x_rm, x_bm = run_stage1(stage1, x, rm_ch, bm_ch)
        pred = stage2(x_coarse=coarse, x_rm=x_rm, x_bm=x_bm)

        sums.update(pred, y)

    return sums.rmse(), sums.nmse(), sums.nmse_db()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lambda1", type=float, default=1.0)
    ap.add_argument("--lambda2", type=float, default=1.0)
    ap.add_argument("--stage1_ckpt", type=str)
    ap.add_argument("--out_dir", type=str)

    # Fixed architecture and data protocol shared by both stages.
    ap.set_defaults(
        thresh=0.2,
        train_samples_low=50,
        train_samples_high=300,
        test_points='50,100,150,200,250,300',
        rm_ch=2,
        bm_ch=0,
        num_workers=2,
        device='cuda',
        lr_drop_epoch=10,
        lr_drop_gamma=0.1,
        stage1_heads=8,
        stage1_enc_layers=12,
        stage1_dec_layers=4,
        stage1_dim_ff=512,
        stage1_dropout=0.1,
        stage2_heads=8,
        stage2_enc_layers=12,
        stage2_dec_layers=4,
        stage2_dim_ff=512,
        stage2_dropout=0.1,
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"[Info] device = {device}")


    from module_vit_128_skip import PatchTF_RM_BM_DecoderUse32Tokens_Half_Skip6_No1x1


    from module_vit_128_skip_refine import PatchTF_RefineRM_BM_Stage2_Half_Skip6_No1x1


    stage1 = PatchTF_RM_BM_DecoderUse32Tokens_Half_Skip6_No1x1(
        heads=args.stage1_heads,
        enc_layers=args.stage1_enc_layers,
        dec_layers=args.stage1_dec_layers,
        dim_ff=args.stage1_dim_ff,
        dropout=args.stage1_dropout,
    ).to(device)

    ckpt = torch.load(args.stage1_ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    stage1.load_state_dict(state_dict, strict=True)
    freeze_model(stage1)

    print("[Info] Stage-1 loaded and frozen.")
    print(f"[Info] Stage-1 checkpoint: {args.stage1_ckpt}")


    stage2 = PatchTF_RefineRM_BM_Stage2_Half_Skip6_No1x1(
        heads=args.stage2_heads,
        enc_layers=args.stage2_enc_layers,
        dec_layers=args.stage2_dec_layers,
        dim_ff=args.stage2_dim_ff,
        dropout=args.stage2_dropout,
    ).to(device)

    print("[Info] Stage-2 created.")
    print("[Info] Loss = lambda1 * global_MSE + lambda2 * sampled_MSE")
    print(f"[Info] lambda1 = {args.lambda1}, lambda2 = {args.lambda2}")

    tfm = transforms.ToTensor()


    train_ds = loaders.RadioUNet_s(
        phase="train",
        dir_dataset=os.path.join(args.data_root, ""),
        simulation="IRT2",
        carsSimul="no",
        carsInput="no",
        cityMap="complete",
        thresh=args.thresh,
        fix_samples=0,
        num_samples_low=args.train_samples_low,
        num_samples_high=args.train_samples_high,
        transform=tfm,
    )

    val_ds = loaders.RadioUNet_s(
        phase="val",
        dir_dataset=os.path.join(args.data_root, ""),
        simulation="IRT2",
        carsSimul="no",
        carsInput="no",
        cityMap="complete",
        thresh=args.thresh,
        fix_samples=0,
        num_samples_low=args.train_samples_low,
        num_samples_high=args.train_samples_high,
        transform=tfm,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )


    optimizer = torch.optim.Adam(stage2.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.lr_drop_epoch],
        gamma=args.lr_drop_gamma,
    )

    best_val = float("inf")
    best_ckpt = os.path.join(args.out_dir, "best_refine_stage2_128_skip.pt")

    print("\n[Train] Start Stage-2 training")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_mse, tr_lw = train_one_epoch_stage2(
            stage1=stage1,
            stage2=stage2,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            rm_ch=args.rm_ch,
            bm_ch=args.bm_ch,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
        )

        va_loss, va_mse, va_lw = validate_one_epoch_stage2(
            stage1=stage1,
            stage2=stage2,
            loader=val_loader,
            device=device,
            rm_ch=args.rm_ch,
            bm_ch=args.bm_ch,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {ep:03d}/{args.epochs} | lr={lr_now:.6g} | "
            f"train: total={tr_loss:.6f}, mse={tr_mse:.6f}, lw={tr_lw:.6f} | "
            f"val: total={va_loss:.6f}, mse={va_mse:.6f}, lw={va_lw:.6f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "stage2_model": stage2.state_dict(),
                    "stage1_ckpt": args.stage1_ckpt,
                    "args": vars(args),
                },
                best_ckpt
            )

        scheduler.step()

    print(f"\n[Train] Best val loss = {best_val:.6f}")
    print(f"[Train] Saved best checkpoint: {best_ckpt}")


    ckpt2 = torch.load(best_ckpt, map_location=device)
    stage2.load_state_dict(ckpt2["stage2_model"])
    stage2.to(device)
    stage2.eval()

    test_points = parse_int_list(args.test_points)
    out_txt = os.path.join(args.out_dir, "test_results_refine_stage2_128_skip.txt")

    print("\n[Test] Sweep fixed sampling points:", test_points)
    results = []

    for n in test_points:
        set_seed(args.seed + n)

        test_ds = loaders.RadioUNet_s(
            phase="test",
            dir_dataset=os.path.join(args.data_root, ""),
            simulation="IRT2",
            carsSimul="no",
            carsInput="no",
            cityMap="complete",
            thresh=args.thresh,
            fix_samples=n,
            num_samples_low=n,
            num_samples_high=n + 1,
            transform=tfm,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        rmse, nmse, nmse_db = evaluate_metrics_stage2(
            stage1=stage1,
            stage2=stage2,
            loader=test_loader,
            device=device,
            rm_ch=args.rm_ch,
            bm_ch=args.bm_ch,
        )

        results.append((n, rmse, nmse, nmse_db))
        print(f"  N={n:3d} | RMSE={rmse:.6f} | NMSE={nmse:.6e} | NMSE(dB)={nmse_db:.3f}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("FixedSamples\tRMSE\tNMSE\tNMSE_DB\n")
        for n, rmse, nmse, nmse_db in results:
            f.write(f"{n}\t{rmse:.8f}\t{nmse:.10e}\t{nmse_db:.6f}\n")

    print(f"\n[Done] Results saved to: {out_txt}")


if __name__ == "__main__":
    main()
