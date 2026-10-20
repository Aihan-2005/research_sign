from dataclasses import dataclass
from pathlib import Path
import importlib.util

from persian_sea.types import SignSegment


@dataclass(frozen=True)
class LegacyInference:
    sign_logits: object
    sentence_logits: object
    fps: float
    num_frames: int


class LegacyE4sAdapter:
 

    def __init__(self, model_name_or_path: str = "model_E4s-1.pth"):
        self.model_name_or_path = model_name_or_path

    @staticmethod
    def _imports():
        try:
            from pose_format import Pose
            from sign_language_segmentation.bin import (
                load_model,
                predict,
                process_pose,
            )
            from sign_language_segmentation.src.utils.probs_to_segments import (
                probs_to_segments,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install the SEA-compatible baseline dependency with:\n"
                'pip install "git+https://github.com/ZurichNLP/segmentation@bsl"'
            ) from exc

        return Pose, load_model, predict, process_pose, probs_to_segments

    def _resolve_model_path(self) -> str:
        model = Path(self.model_name_or_path)


        if model.is_absolute() or model.parent != Path("."):
            if not model.exists():
                raise FileNotFoundError(model)
            return str(model.resolve())


        spec = importlib.util.find_spec("sign_language_segmentation")
        if spec is None or spec.origin is None:
            raise RuntimeError("sign_language_segmentation is not installed")

        package_dir = Path(spec.origin).resolve().parent
        packaged = package_dir / "dist" / self.model_name_or_path
        if not packaged.exists():
            raise FileNotFoundError(
                f"Packaged checkpoint not found: {packaged}"
            )
        return str(packaged)

    def infer(self, pose_path: str | Path) -> LegacyInference:
        Pose, load_model, predict, process_pose, _ = self._imports()

        pose_path = Path(pose_path)
        with pose_path.open("rb") as f:
            pose = Pose.read(f.read())

        fps = float(pose.body.fps)
        num_frames = int(pose.body.data.shape[0])

        use_e4 = "E4" in Path(self.model_name_or_path).name
        processed = process_pose(
            pose,
            optical_flow=use_e4,
            hand_normalization=use_e4,
        )

        model = load_model(self._resolve_model_path())
        outputs = predict(model, processed)

        return LegacyInference(
            sign_logits=outputs["sign"].cpu(),
            sentence_logits=outputs["sentence"].cpu(),
            fps=fps,
            num_frames=num_frames,
        )

    def decode_signs(
        self,
        inference: LegacyInference,
        b_threshold: int,
        o_threshold: int,
    ) -> list[SignSegment]:
        _, _, _, _, probs_to_segments = self._imports()

        raw = probs_to_segments(
            inference.sign_logits,
            b_threshold=b_threshold,
            o_threshold=o_threshold,
        )

        segments: list[SignSegment] = []
        for item in raw:
            start_frame = int(item["start"])
            end_frame = int(item["end"])


            if end_frame <= start_frame:
                end_frame = start_frame + 1

            segments.append(
                SignSegment(
                    start=start_frame / inference.fps,
                    end=end_frame / inference.fps,
                )
            )

        return segments
