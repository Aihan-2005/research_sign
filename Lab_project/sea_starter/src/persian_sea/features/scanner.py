from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from persian_sea.config import AppConfig, read_video_ids, segmentation_subdir
from persian_sea.features.crops import boxes_for_frame, crop_rgb
from persian_sea.features.dinov2 import DinoV2FeatureExtractor
from persian_sea.features.pose_reader import load_holistic_pose, normalized_pose_tensor
from persian_sea.features.sampling import sample_times, times_to_indices
from persian_sea.features.video_reader import IndexedVideoReader, find_video
from persian_sea.io.elan import read_sign_segments


class RGBPoseScanner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.rgb = DinoV2FeatureExtractor(
            model_name=cfg.scanner.rgb_model,
            device=cfg.scanner.device,
            amp=cfg.scanner.amp,
        )

    def scan_video(self, video_id: str) -> Path:
        cfg = self.cfg
        out_dir = cfg.paths.feature_cache
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / f"{video_id}.npz"
        out_json = out_dir / f"{video_id}.json"

        if out_npz.exists() and not cfg.scanner.overwrite:
            print(f"[scan] skip existing {out_npz}")
            return out_npz

        pose_path = cfg.paths.poses / f"{video_id}.pose"
        eaf_path = segmentation_subdir(cfg) / f"{video_id}.eaf"
        video_path = find_video(cfg.paths.videos, video_id)

        if not pose_path.exists():
            raise FileNotFoundError(pose_path)
        if not eaf_path.exists():
            raise FileNotFoundError(eaf_path)

        pose = load_holistic_pose(pose_path)
        video = IndexedVideoReader(video_path)
        segments, tier = read_sign_segments(eaf_path, cfg.elan.sign_tiers)

        if not segments:
            raise RuntimeError(f"No sign segments in {eaf_path}")

        views = cfg.scanner.resolved_views()
        K = int(cfg.scanner.samples_per_segment)

        all_pose = []
        all_rgb = []
        all_times = []
        all_pose_indices = []
        all_video_indices = []
        metadata = []

        for seg_idx, seg in enumerate(tqdm(segments, desc=f"scan {video_id}")):
            times = sample_times(seg.start, seg.end, K)
            pose_idx = times_to_indices(times, pose.fps, pose.length)
            video_idx = times_to_indices(times, video.fps, video.length)

            pose_tensor = normalized_pose_tensor(
                pose,
                pose_idx,
                shoulder_epsilon=cfg.scanner.pose_shoulder_epsilon,
                include_confidence=cfg.scanner.pose_include_confidence,
            )

            frames = video.get_batch(video_idx)
            crop_images: list[np.ndarray] = []
            crop_slots: list[tuple[int, int]] = []

            for k, (frame, pfi) in enumerate(zip(frames, pose_idx)):
                boxes = boxes_for_frame(
                    pose,
                    int(pfi),
                    video.width,
                    video.height,
                    hand_padding=cfg.scanner.hand_padding,
                    face_padding=cfg.scanner.face_padding,
                    signer_padding=cfg.scanner.signer_padding,
                )

                signer_crop = crop_rgb(frame, boxes.get("signer"))
                if signer_crop is None:
                    signer_crop = frame

                for vi, view in enumerate(views):
                    if view == "signer":
                        crop = signer_crop
                    else:
                        crop = crop_rgb(frame, boxes.get(view))
                        if crop is None and cfg.scanner.local_fallback_to_signer:
                            crop = signer_crop
                        elif crop is None:
                            crop = frame
                    crop_images.append(crop)
                    crop_slots.append((k, vi))

            encoded = self.rgb.encode(
                crop_images,
                batch_size=cfg.scanner.rgb_batch_size,
            )
            rgb_tensor = np.zeros(
                (K, len(views), self.rgb.hidden_size),
                dtype=np.float32,
            )
            for feat, (k, vi) in zip(encoded, crop_slots):
                rgb_tensor[k, vi] = feat

            all_pose.append(pose_tensor)
            all_rgb.append(rgb_tensor)
            all_times.append(times)
            all_pose_indices.append(pose_idx)
            all_video_indices.append(video_idx)

            metadata.append({
                "segment_index": seg_idx,
                "start": float(seg.start),
                "end": float(seg.end),
                "duration": float(seg.duration),
                "tier": tier,
                "text": seg.text,
            })

        pose_arr = np.stack(all_pose, axis=0).astype(np.float32)
        rgb_arr = np.stack(all_rgb, axis=0).astype(np.float32)
        times_arr = np.stack(all_times, axis=0).astype(np.float32)
        pose_idx_arr = np.stack(all_pose_indices, axis=0).astype(np.int32)
        video_idx_arr = np.stack(all_video_indices, axis=0).astype(np.int32)

        dtype = np.float16 if cfg.scanner.cache_dtype == "float16" else np.float32
        rgb_arr = rgb_arr.astype(dtype)
        pose_arr = pose_arr.astype(dtype)

        np.savez_compressed(
            out_npz,
            pose=pose_arr,
            rgb=rgb_arr,
            sample_times=times_arr,
            pose_indices=pose_idx_arr,
            video_indices=video_idx_arr,
            starts=np.asarray([s.start for s in segments], dtype=np.float32),
            ends=np.asarray([s.end for s in segments], dtype=np.float32),
            views=np.asarray(views),
            rgb_hidden=np.asarray([self.rgb.hidden_size], dtype=np.int32),
        )

        out_json.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "video": str(video_path),
                    "pose": str(pose_path),
                    "eaf": str(eaf_path),
                    "video_fps": video.fps,
                    "pose_fps": pose.fps,
                    "views": views,
                    "rgb_model": cfg.scanner.rgb_model,
                    "segments": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[scan] {video_id}: pose={pose_arr.shape}, rgb={rgb_arr.shape}, "
            f"saved={out_npz}"
        )
        return out_npz


def scan_all(cfg: AppConfig) -> None:
    scanner = RGBPoseScanner(cfg)
    for video_id in read_video_ids(cfg.paths.video_ids):
        scanner.scan_video(video_id)
