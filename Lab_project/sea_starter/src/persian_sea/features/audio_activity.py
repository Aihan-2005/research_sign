

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import numpy as np


def extract_audio_pcm(video: Path, sample_rate: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[:500]}")
    pcm = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    if pcm.size == 0:
        raise ValueError(f"No audio samples in {video}")
    return pcm


def compute_energy_activity(
    waveform: np.ndarray, sample_rate: int = 16000,
    hop_seconds: float = 0.04, relative_threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    frame = max(1, round(sample_rate * hop_seconds))
    length = (len(waveform) // frame) * frame
    if length == 0:
        raise ValueError("Audio shorter than one analysis frame")
    data = waveform[:length].astype(np.float32).reshape(-1, frame)
    energy = np.sqrt(np.mean(data * data, axis=1) + 1e-12)

    floor = float(np.quantile(energy, 0.20))
    peak = float(np.quantile(energy, 0.90))
    cutoff = floor + relative_threshold * max(0.0, peak - floor)
    active = energy > max(cutoff, 1e-4)
    centers = (np.arange(len(energy), dtype=np.float32) + 0.5) * (frame / sample_rate)
    return centers, energy.astype(np.float32), active.astype(np.bool_)


def save_audio_activity(video: Path, output: Path) -> Path:
    waveform = extract_audio_pcm(video)
    centers, energy, active = compute_energy_activity(waveform)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, times=centers, rms=energy, active=active)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract optional audio timing hints")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(save_audio_activity(args.video, args.out))


if __name__ == "__main__":
    main()
