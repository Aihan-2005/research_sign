from persian_sea.video_text.config import VideoTextConfig


def test_global_subtitle_offset_defaults_to_two_seconds():
    cfg = VideoTextConfig()
    assert cfg.get_subtitle_offset("any_video") == 2.0


def test_per_video_offset_overrides_global_default():
    cfg = VideoTextConfig(
        subtitle_to_sign_offset_seconds=2.0,
        subtitle_offsets_by_video={"news001": 3.25},
    )
    assert cfg.get_subtitle_offset("news001") == 3.25
    assert cfg.get_subtitle_offset("news002") == 2.0


def test_negative_offset_is_allowed_for_signs_before_subtitles():
    cfg = VideoTextConfig(subtitle_to_sign_offset_seconds=-1.5)
    assert cfg.get_subtitle_offset("news001") == -1.5
