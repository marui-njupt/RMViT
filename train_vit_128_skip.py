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


def nmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    num = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    den = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    return torch.mean(num / den)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return torch.mean((pred - target) ** 2)


def gradient_loss_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = torch.mean(torch.abs(dx_pred - dx_target))
    loss_y = torch.mean(torch.abs(dy_pred - dy_target))
    return loss_x + loss_y


def total_loss(pred: torch.Tensor, target: torch.Tensor, lambda_grad: float = 0.1) -> torch.Tensor:

    pred_n = pred / 256.0
    target_n = target / 256.0

    loss_mse = mse_loss(pred_n, target_n)
    loss_grad = gradient_loss_l1(pred_n, target_n)

    return loss_mse + lambda_grad * loss_grad


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    rm_ch: int,
    bm_ch: int,
    lambda_grad: float = 0.1,
) -> float:
    model.train()
    total = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_rm = x[:, rm_ch:rm_ch + 1, :, :]
        x_bm = x[:, bm_ch:bm_ch + 1, :, :]

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_rm, x_bm)
        loss = total_loss(pred, y, lambda_grad=lambda_grad)
        loss.backward()
        optimizer.step()

        bs = int(x.size(0))
        total += float(loss.item()) * bs
        count += bs

    return total / max(1, count)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rm_ch: int,
    bm_ch: int,
    lambda_grad: float = 0.1,
) -> float:
    model.eval()
    total = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_rm = x[:, rm_ch:rm_ch + 1, :, :]
        x_bm = x[:, bm_ch:bm_ch + 1, :, :]

        pred = model(x_rm, x_bm)
        loss = total_loss(pred, y, lambda_grad=lambda_grad)

        bs = int(x.size(0))
        total += float(loss.item()) * bs
        count += bs

    return total / max(1, count)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rm_ch: int,
    bm_ch: int,
) -> Tuple[float, float, float]:
    model.eval()
    sums = MetricSums()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_rm = x[:, rm_ch:rm_ch + 1, :, :]
        x_bm = x[:, bm_ch:bm_ch + 1, :, :]

        pred = model(x_rm, x_bm)
        sums.update(pred, y)

    return sums.rmse(), sums.nmse(), sums.nmse_db()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lambda_grad", type=float)
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
        lr_drop_epoch=30,
        lr_drop_gamma=0.1,
        heads=8,
        enc_layers=12,
        dec_layers=4,
        dim_ff=512,
        dropout=0.1,
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"[Info] device = {device}")


    from module_vit_128_skip import PatchTF_RM_BM_DecoderUse32Tokens_Half_Skip6_No1x1  # noqa: E402

    model = PatchTF_RM_BM_DecoderUse32Tokens_Half_Skip6_No1x1(
        heads=args.heads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    print(
        f"[Info] model = PatchTF_RM_BM_DecoderUse32Tokens_Half_Skip4_No1x1("
        f"heads={args.heads}, enc_layers={args.enc_layers}, "
        f"dec_layers={args.dec_layers}, dim_ff={args.dim_ff}, dropout={args.dropout})"
    )

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.lr_drop_epoch],
        gamma=args.lr_drop_gamma,
    )

    best_val = float("inf")
    best_ckpt = os.path.join(args.out_dir, "best_tf_128_skip.pt")

    print("\n[Train] Start training (Loss = MSE + Grad)")
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, device,
            optimizer,
            args.rm_ch, args.bm_ch,
            lambda_grad=args.lambda_grad
        )
        va = validate_one_epoch(
            model, val_loader, device,
            args.rm_ch, args.bm_ch,
            lambda_grad=args.lambda_grad
        )
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:03d}/{args.epochs} | lr={lr_now:.6g} | train={tr:.6f} | val={va:.6f}")

        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_ckpt)

        scheduler.step()

    print(f"\n[Train] Best val loss = {best_val:.6f}")
    print(f"[Train] Saved best checkpoint: {best_ckpt}")


    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()


    test_points = parse_int_list(args.test_points)
    out_txt = os.path.join(args.out_dir, "test_results_fixed_samples.txt")

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

        rmse, nmse, nmse_db = evaluate_metrics(model, test_loader, device, args.rm_ch, args.bm_ch)
        results.append((n, rmse, nmse, nmse_db))
        print(f"  N={n:3d} | RMSE={rmse:.6f} | NMSE={nmse:.6e} | NMSE(dB)={nmse_db:.3f}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("FixedSamples\tRMSE\tNMSE\tNMSE_DB\n")
        for n, rmse, nmse, nmse_db in results:
            f.write(f"{n}\t{rmse:.8f}\t{nmse:.10e}\t{nmse_db:.6f}\n")

    print(f"\n[Done] Results saved to: {out_txt}")


if __name__ == "__main__":
    main()
