import torch

from persian_sea.translation.data import collate


class TokenizerWithSpecials:
    all_special_ids = [0, 1, 101]

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        del padding, truncation, max_length, return_tensors
        ids = torch.tensor([[101, 10, 11, 1], [101, 20, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        return {"input_ids": ids, "attention_mask": mask}


def _row(text):
    return {
        "pose": torch.zeros(2, 2, 75, 4),
        "rgb": torch.zeros(2, 2, 1, 8),
        "starts": torch.tensor([0.0, 1.0]),
        "ends": torch.tensor([0.8, 1.8]),
        "text": text,
    }


def test_collate_provides_target_content_mask_for_ota():
    batch = collate(
        [_row("سلام"), _row("خوب")],
        TokenizerWithSpecials(),
        max_target_tokens=16,
    )
    assert batch["target_input_ids"].shape == (2, 4)
    assert batch["target_attention_mask"].shape == (2, 4)
    assert batch["target_content_mask"].tolist() == [
        [False, True, True, False],
        [False, True, False, False],
    ]
    assert batch["labels"][1, 2:].tolist() == [-100, -100]
