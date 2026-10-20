from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(1e-6, self.end - self.start)


_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})


def normalize_persian_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_ARABIC_TO_PERSIAN)

    text = text.replace("\u200c", " ")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def _vtt_time_to_seconds(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        h = 0
        m, s = parts
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError(f"Invalid timestamp: {value}")
    return int(h) * 3600 + int(m) * 60 + float(s)


def find_subtitle(directory: Path, video_id: str) -> Path | None:
    for ext in (".vtt", ".srt"):
        path = directory / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


def read_subtitles(path: str | Path) -> list[SubtitleCue]:
    path = Path(path)
    cues = []
    if path.suffix.lower() == ".vtt":
        try:
            import webvtt
        except ImportError:
            webvtt = None
        if webvtt is None:
            cues.extend(_read_standard_vtt(path))
        else:
            for item in webvtt.read(str(path)):
                cues.append(SubtitleCue(
                    _vtt_time_to_seconds(item.start),
                    _vtt_time_to_seconds(item.end),
                    normalize_persian_text(item.text),
                ))
    elif path.suffix.lower() == ".srt":
        try:
            import pysrt
        except ImportError:
            pysrt = None
        if pysrt is None:
            cues.extend(_read_standard_srt(path))
        else:
            for item in pysrt.open(str(path), encoding="utf-8"):
                cues.append(SubtitleCue(
                    item.start.ordinal / 1000.0,
                    item.end.ordinal / 1000.0,
                    normalize_persian_text(" ".join(item.text.splitlines())),
                ))
    else:
        raise ValueError(f"Unsupported subtitle extension: {path}")
    for item in cues:
        if item.start < 0 or item.end <= item.start:
            raise ValueError(f"Invalid subtitle timing in {path}: {item}")
    return cues


def _read_standard_vtt(path: Path) -> list[SubtitleCue]:
    """Dependency-free fallback for ordinary VTT cues; complex WebVTT: install webvtt-py."""
    contents = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if not contents.lstrip().startswith("WEBVTT"):
        raise ValueError(f"Missing WEBVTT header: {path}")
    result = []
    for block in re.split(r"\n\s*\n", contents.strip()):
        lines = [line.strip() for line in block.splitlines()]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timeline = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if timeline is None:
            continue
        left, right = lines[timeline].split("-->", 1)
        right = right.strip().split()[0]  
        
        result.append(SubtitleCue(_vtt_time_to_seconds(left.strip()),
            _vtt_time_to_seconds(right),
            normalize_persian_text(" ".join(lines[timeline + 1:]))))
    return result


def _read_standard_srt(path: Path) -> list[SubtitleCue]:
    contents = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    result = []
    for block in re.split(r"\n\s*\n", contents.strip()):
        lines = [line.strip() for line in block.splitlines()]
        timeline = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if timeline is None:
            continue
        left, right = lines[timeline].split("-->", 1)
        result.append(SubtitleCue(_vtt_time_to_seconds(left.strip()),
            _vtt_time_to_seconds(right.strip()),
            normalize_persian_text(" ".join(lines[timeline + 1:]))))
    return result



def pair_corrected_cues(
    rough: list[SubtitleCue], corrected: list[SubtitleCue] | None,
) -> list[SubtitleCue | None]:


    if not corrected:
        return [None] * len(rough)
    if len(rough) == len(corrected) and all(
        a.text == b.text for a, b in zip(rough, corrected)
    ):
        return list(corrected)

    from collections import defaultdict
    rough_counts = defaultdict(int)
    corrected_by_text = defaultdict(list)
    for item in rough:
        rough_counts[item.text] += 1
    for item in corrected:
        corrected_by_text[item.text].append(item)
    return [
        corrected_by_text[c.text][0]
        if rough_counts[c.text] == 1 and len(corrected_by_text[c.text]) == 1
        else None
        for c in rough
    ]


def shifted(cues: list[SubtitleCue], start_offset: float,
            end_offset: float) -> list[SubtitleCue]:

    result = []
    for c in cues:
        start = max(0.0, c.start + float(start_offset))
        end = c.end + float(end_offset)
        if end <= start:
            raise ValueError(f"Shift produces invalid subtitle time: {c}")
        result.append(SubtitleCue(start, end, c.text))
    return result


def seconds_to_vtt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    milliseconds = int(round(seconds * 1000))
    h, rest = divmod(milliseconds, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_vtt(cues: list[SubtitleCue], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for cue in cues:
        if cue.start < 0 or cue.end <= cue.start:
            raise ValueError(f"Invalid output subtitle: {cue}")
        lines.extend([
            f"{seconds_to_vtt(cue.start)} --> {seconds_to_vtt(cue.end)}",
            cue.text, "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
