from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    config_dict: dict,
    metrics: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config_dict,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model,
    device,
) -> dict:
    payload = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model"])
    return payload
