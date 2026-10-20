from persian_sea.config import model_tag, segmentation_run_name


def test_model_tag():
    assert model_tag("model_E4s-1.pth") == "E4s-1"
    assert model_tag("/tmp/model_E4s-1.pth") == "E4s-1"


def test_run_name():
    assert segmentation_run_name("model_E4s-1.pth", 30, 50) == "E4s-1_30_50"
