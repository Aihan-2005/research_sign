from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from persian_sea.types import SignSegment
from persian_sea.io.subtitles import SubtitleCue


def available_tiers(eaf_path: str | Path) -> list[str]:
    root = ET.parse(eaf_path).getroot()
    return [t.get("TIER_ID") for t in root.findall("TIER") if t.get("TIER_ID")]


def read_sign_segments(
    eaf_path: str | Path, accepted_tiers: list[str],
) -> tuple[list[SignSegment], str]:
    eaf_path = Path(eaf_path)
    root = ET.parse(eaf_path).getroot()
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError(f"No TIME_ORDER in {eaf_path}")
    slots = {}
    for slot in time_order.findall("TIME_SLOT"):
        ref, value = slot.get("TIME_SLOT_ID"), slot.get("TIME_VALUE")
        if ref and value is not None:
            slots[ref] = float(value) / 1000.0
    selected = None
    selected_name = ""
    for requested in accepted_tiers:
        selected = root.find(f"./TIER[@TIER_ID='{requested}']")
        if selected is not None:
            selected_name = requested
            break
    if selected is None:
        raise ValueError(f"No tier from {accepted_tiers} in {eaf_path}; found {available_tiers(eaf_path)}")
    segments = []
    for annotation in selected.findall("ANNOTATION"):
        elem = next(iter(annotation), None)
        if elem is None:
            continue
        start = slots.get(elem.get("TIME_SLOT_REF1"))
        end = slots.get(elem.get("TIME_SLOT_REF2"))
        if start is None or end is None:
            continue
        value = elem.find("ANNOTATION_VALUE")
        text = value.text.strip() if value is not None and value.text else ""
        segment = SignSegment(start, end, text)
        try:
            segment.validate()
        except ValueError:
            continue
        segments.append(segment)
    segments.sort(key=lambda s: (s.start, s.end))
    return segments, selected_name


def write_aligned_subtitle_tier(
    source_eaf: str | Path,
    cues: list[SubtitleCue],
    output_eaf: str | Path,
    tier_id: str = "SUBTITLE_SHIFTED",
) -> None:


    source_eaf, output_eaf = Path(source_eaf), Path(output_eaf)
    tree = ET.parse(source_eaf)
    root = tree.getroot()
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError(f"No TIME_ORDER in {source_eaf}")
    for previous in list(root.findall("TIER")):
        if previous.get("TIER_ID") == tier_id:
            root.remove(previous)



    available_types = {
        x.get("LINGUISTIC_TYPE_ID") for x in root.findall("LINGUISTIC_TYPE")
    }
    lt_id = "PersianSEA-SubtitleLT"
    if lt_id not in available_types:
        new_type = ET.Element("LINGUISTIC_TYPE", {
            "LINGUISTIC_TYPE_ID": lt_id,
            "TIME_ALIGNABLE": "true",
            "GRAPHIC_REFERENCES": "false",
        })

        child_tags = [child.tag for child in root]
        after_types = [i for i, tag in enumerate(child_tags) if tag == "LINGUISTIC_TYPE"]
        after_tiers = [i for i, tag in enumerate(child_tags) if tag == "TIER"]
        insert_at = 1 + (after_types[-1] if after_types else after_tiers[-1] if after_tiers else 1)
        root.insert(insert_at, new_type)

    existing_ids = {
        x.attrib[key]
        for x in root.iter()
        for key in ("TIME_SLOT_ID", "ANNOTATION_ID")
        if key in x.attrib
    }
    serial = 1

    def unique(prefix: str) -> str:
        nonlocal serial
        while f"{prefix}{serial}" in existing_ids:
            serial += 1
        result = f"{prefix}{serial}"
        existing_ids.add(result)
        serial += 1
        return result

    tier = ET.Element("TIER", {
        "TIER_ID": tier_id, "LINGUISTIC_TYPE_REF": lt_id,
    })
    tier_positions = [i for i, child in enumerate(root) if child.tag == "TIER"]
    tier_insert_at = tier_positions[-1] + 1 if tier_positions else 2
    root.insert(tier_insert_at, tier)
    for cue in cues:
        if cue.start < 0 or cue.end <= cue.start:
            raise ValueError(f"Invalid subtitle to write to EAF: {cue}")
        start_ref, end_ref = unique("ts"), unique("ts")
        ET.SubElement(time_order, "TIME_SLOT", {
            "TIME_SLOT_ID": start_ref,
            "TIME_VALUE": str(int(round(cue.start * 1000))),
        })
        ET.SubElement(time_order, "TIME_SLOT", {
            "TIME_SLOT_ID": end_ref,
            "TIME_VALUE": str(int(round(cue.end * 1000))),
        })
        ann = ET.SubElement(tier, "ANNOTATION")
        alignable = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": unique("a"),
            "TIME_SLOT_REF1": start_ref,
            "TIME_SLOT_REF2": end_ref,
        })
        ET.SubElement(alignable, "ANNOTATION_VALUE").text = cue.text



    time_order[:] = sorted(
        time_order,
        key=lambda x: int(x.get("TIME_VALUE", "0")),
    )
    output_eaf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_eaf, encoding="utf-8", xml_declaration=True)
