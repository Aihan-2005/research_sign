import torch

from persian_sea.translation.ot_alignment import PartialOTAlignmentLoss


def test_partial_ot_is_finite_and_backpropagates():
    torch.manual_seed(7)
    visual = torch.randn(2, 5, 12, requires_grad=True)
    text = torch.randn(2, 4, 12, requires_grad=True)
    visual_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]]
    )
    text_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )

    criterion = PartialOTAlignmentLoss(
        epsilon=0.1,
        iterations=25,
        dustbin_cost=0.4,
    )
    result = criterion(visual, text, visual_mask, text_mask)

    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["transport_cost"])
    assert 0.0 <= float(result["null_ratio"].detach()) <= 1.0
    result["loss"].backward()
    assert visual.grad is not None
    assert text.grad is not None
    assert torch.isfinite(visual.grad).all()
    assert torch.isfinite(text.grad).all()


def test_partial_ot_skips_samples_without_content_tokens():
    visual = torch.randn(1, 3, 8, requires_grad=True)
    text = torch.randn(1, 2, 8, requires_grad=True)
    visual_mask = torch.tensor([[True, True, True]])
    text_mask = torch.tensor([[False, False]])

    result = PartialOTAlignmentLoss()(visual, text, visual_mask, text_mask)
    assert result["loss"].item() == 0.0
    result["loss"].backward()
    assert visual.grad is not None
