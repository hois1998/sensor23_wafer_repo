from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from wafer_repro.labels import PAPER_CLASSES
from wafer_repro.models import MODEL_SPECS, PAPER_MODEL_NAMES, count_parameters, create_model
from wafer_repro.utils import choose_device, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure model size, optional MACs, and throughput.")
    parser.add_argument("--models", nargs="+", default=["paper"], help="Model names or 'paper'.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="outputs/benchmark.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "directml", "dml", "cpu"])
    return parser.parse_args()


def expand_models(names: list[str]) -> list[str]:
    if len(names) == 1 and names[0] == "paper":
        return list(PAPER_MODEL_NAMES)
    return names


def maybe_profile_madds(model, dummy):
    try:
        from thop import profile
    except ImportError:
        return None, None
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    return float(macs / 1e6), float((2 * macs) / 1e6)


def measure_inference(model, dummy, warmup: int, iters: int, device) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return dummy.size(0) * iters / max(elapsed, 1e-9)


def measure_training(model, dummy, warmup: int, iters: int, device) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target = torch.zeros(dummy.size(0), dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(dummy), target)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(dummy), target)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return dummy.size(0) * iters / max(elapsed, 1e-9)


def main() -> None:
    args = parse_args()
    model_names = expand_models(args.models)
    out = Path(args.out)
    ensure_dir(out.parent)
    device_choice = choose_device(args.device)

    rows = []
    for name in model_names:
        model = create_model(name, num_classes=len(PAPER_CLASSES), pretrained=False).to(device_choice.device)
        dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device_choice.device)
        madds, mflops = maybe_profile_madds(model, dummy[:1])
        params = count_parameters(model)
        param_mb = params * 4 / (1024**2)
        inference_ips = measure_inference(model, dummy, args.warmup, args.iters, device_choice.device)
        training_ips = measure_training(model, dummy, max(1, args.warmup // 2), max(1, args.iters // 2), device_choice.device)
        spec = MODEL_SPECS.get(name)
        rows.append(
            {
                "model": name,
                "paper_name": spec.paper_name if spec else name,
                "params_m": params / 1e6,
                "param_size_mb_fp32": param_mb,
                "madds_m": madds,
                "mflops_m": mflops,
                "training_images_per_s": training_ips,
                "inference_images_per_s": inference_ips,
                "device": device_choice.backend,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
            }
        )
        print(rows[-1])

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

