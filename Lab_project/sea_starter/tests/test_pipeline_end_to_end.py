

from dataclasses import replace
from pathlib import Path
import json
import xml.etree.ElementTree as ET

from persian_sea.config import load_config, segmentation_subdir
from persian_sea.io.subtitles import SubtitleCue, read_subtitles, write_vtt
from persian_sea.io.elan import read_sign_segments
from persian_sea.alignment.pipeline import align_one


def _sign_eaf(path: Path, spans):
    root = ET.Element('ANNOTATION_DOCUMENT')
    timeorder = ET.SubElement(root, 'TIME_ORDER')
    tier = ET.SubElement(root, 'TIER', TIER_ID='SIGN', LINGUISTIC_TYPE_REF='default-lt')
    for i, (start, end) in enumerate(spans):
        a, b = f'ts{i*2+1}', f'ts{i*2+2}'
        ET.SubElement(timeorder, 'TIME_SLOT', TIME_SLOT_ID=a, TIME_VALUE=str(round(start*1000)))
        ET.SubElement(timeorder, 'TIME_SLOT', TIME_SLOT_ID=b, TIME_VALUE=str(round(end*1000)))
        annotation = ET.SubElement(tier, 'ANNOTATION')
        signed = ET.SubElement(annotation, 'ALIGNABLE_ANNOTATION',
                               ANNOTATION_ID=f'a{i+1}', TIME_SLOT_REF1=a,
                               TIME_SLOT_REF2=b)
        ET.SubElement(signed, 'ANNOTATION_VALUE').text = f'sign{i+1}'
    ET.SubElement(root, 'LINGUISTIC_TYPE', LINGUISTIC_TYPE_ID='default-lt', TIME_ALIGNABLE='true')
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def _config(tmp_path: Path, backend: str):
    cfg_path = Path(__file__).parent.parent / 'configs' / 'continuous_persian.yaml'
    cfg = load_config(cfg_path)
    paths = replace(cfg.paths,
                    subtitles_raw=tmp_path / 'subtitles',
                    segmentation_output=tmp_path / 'seg',
                    alignment_output=tmp_path / 'out')
    alignment = replace(cfg.alignment, backend=backend,
                        overwrite=True, write_updated_eaf=True,
                        auto_apply_small_shifts=False)
    return replace(cfg, paths=paths, alignment=alignment)


def test_pipeline_sea_proposes_but_preserves_original(tmp_path):
    cfg = _config(tmp_path, 'sea_temporal')
    cfg.paths.subtitles_raw.mkdir()
    write_vtt([SubtitleCue(10.2, 11.8, 'سلام'), SubtitleCue(15.2, 16.8, 'خوبی')],
              cfg.paths.subtitles_raw / 'v1.vtt')
    seg_eaf = segmentation_subdir(cfg) / 'v1.eaf'
    _sign_eaf(seg_eaf, [(0, 1), (10, 12), (15, 17)])

    out = align_one('v1', cfg)
    assert out.exists()
    assert (cfg.paths.alignment_output / 'v1.proposed.vtt').exists()
    assert (cfg.paths.alignment_output / 'v1.review.json').exists()
    assert (cfg.paths.alignment_output / 'v1.predicted.eaf').exists()
    applied = read_subtitles(out)
    assert [(x.start, x.end) for x in applied] == [(10.2, 11.8), (15.2, 16.8)]
    proposed = read_subtitles(cfg.paths.alignment_output / 'v1.proposed.vtt')
    assert [(x.start, x.end) for x in proposed] == [(10., 12.), (15., 17.)]
    report = json.loads((cfg.paths.alignment_output / 'v1.review.json').read_text())
    assert len(report['decisions']) == 2
    signs, tier = read_sign_segments(cfg.paths.alignment_output / 'v1.predicted.eaf', ['SIGN'])
    assert len(signs) == 3


def test_pipeline_original_needs_no_segmentation(tmp_path):
    cfg = _config(tmp_path, 'original')
    cfg.paths.subtitles_raw.mkdir()
    write_vtt([SubtitleCue(2, 3, 'سلام')], cfg.paths.subtitles_raw / 'v2.vtt')
    out = align_one('v2', cfg)
    assert read_subtitles(out)[0].start == 2
