from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(scores: torch.Tensor, video_ids: list[str],
                     text_keys: list[str], temperature: float = 0.07):
    batch_size = scores.shape[0]
    if (
        scores.shape != (batch_size, batch_size)
        or len(video_ids) != batch_size
        or len(text_keys) != batch_size
    ):
        raise ValueError("Mismatch between score matrix and sample identifiers")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    allowed = torch.ones(
        (batch_size, batch_size), dtype=torch.bool, device=scores.device
    )
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j and (
                video_ids[i] == video_ids[j] or text_keys[i] == text_keys[j]
            ):
                allowed[i, j] = False
    logits = scores.float() / temperature
    logits = logits.masked_fill(~allowed, -1e4)
    target = torch.arange(batch_size, device=scores.device)
    row_valid = allowed.sum(1) > 1
    col_valid = allowed.sum(0) > 1
    terms = []
    if row_valid.any():
        terms.append(F.cross_entropy(logits[row_valid], target[row_valid]))
    if col_valid.any():
        terms.append(F.cross_entropy(logits.T[col_valid], target[col_valid]))
    if not terms:
        return scores.sum() * 0.0, {
            "active_queries": 0,
            "active_videos": 0,
            "masked_pairs": int((~allowed).sum()),
        }
    return torch.stack(terms).mean(), {
        "active_queries": int(row_valid.sum()),
        "active_videos": int(col_valid.sum()),
        "masked_pairs": int((~allowed).sum()),
    }


def partial_ot_similarity_matrix(
    word_vectors: torch.Tensor,
    word_mask: torch.Tensor,
    span_vectors: torch.Tensor,
    span_mask: torch.Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 30,
    dustbin_score: float = 0.0,
) -> torch.Tensor:


    if word_vectors.ndim != 3 or span_vectors.ndim != 3:
        raise ValueError("word_vectors and span_vectors must be rank-3")
    if word_vectors.shape[-1] != span_vectors.shape[-1]:
        raise ValueError("Word and video embedding dimensions must match")
    if word_mask.shape != word_vectors.shape[:2]:
        raise ValueError("word_mask shape mismatch")
    if span_mask.shape != span_vectors.shape[:2]:
        raise ValueError("span_mask shape mismatch")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon must be positive and iterations >= 1")
    if not -1.0 <= dustbin_score <= 1.0:
        raise ValueError("dustbin_score must be in [-1, 1]")

    words = F.normalize(word_vectors.float(), dim=-1)
    spans = F.normalize(span_vectors.float(), dim=-1)
    word_mask = word_mask.bool()
    span_mask = span_mask.bool()
    query_count, max_words, _ = words.shape
    video_count, max_spans, _ = spans.shape
    if query_count < 1 or video_count < 1 or max_words < 1 or max_spans < 1:
        raise ValueError("Partial OT needs nonempty query/video dimensions")
    if not word_mask.any(dim=1).all():
        raise ValueError("Every word-level query must contain at least one word")
    if not span_mask.any(dim=1).all():
        raise ValueError("Every video bag must contain at least one valid span")

    cosine = torch.einsum("qwd,vsd->qvws", words, spans)
    valid_real = word_mask[:, None, :, None] & span_mask[None, :, None, :]
    log_kernel = cosine.new_full(
        (query_count, video_count, max_words + 1, max_spans + 1),
        float("-inf"),
    )
    log_kernel[:, :, :max_words, :max_spans] = (
        cosine / float(epsilon)
    ).masked_fill(~valid_real, float("-inf"))

    valid_words = word_mask[:, None, :].expand(query_count, video_count, max_words)
    valid_spans = span_mask[None, :, :].expand(query_count, video_count, max_spans)
    log_kernel[:, :, :max_words, max_spans] = torch.where(
        valid_words,
        cosine.new_full(valid_words.shape, float(dustbin_score) / float(epsilon)),
        cosine.new_full(valid_words.shape, float("-inf")),
    )
    log_kernel[:, :, max_words, :max_spans] = torch.where(
        valid_spans,
        cosine.new_full(valid_spans.shape, float(dustbin_score) / float(epsilon)),
        cosine.new_full(valid_spans.shape, float("-inf")),
    )
    log_kernel[:, :, max_words, max_spans] = 0.0

    word_count = word_mask.sum(dim=1).clamp_min(1).float()
    span_count = span_mask.sum(dim=1).clamp_min(1).float()
    a_real = word_mask.float() / word_count[:, None]
    b_real = span_mask.float() / span_count[:, None]
    a = torch.cat((a_real, torch.ones(query_count, 1, device=words.device)), dim=1)
    b = torch.cat((b_real, torch.ones(video_count, 1, device=spans.device)), dim=1)
    log_a = torch.where(a > 0, a.log(), torch.full_like(a, float("-inf")))
    log_b = torch.where(b > 0, b.log(), torch.full_like(b, float("-inf")))
    log_a = log_a[:, None, :].expand(query_count, video_count, max_words + 1)
    log_b = log_b[None, :, :].expand(query_count, video_count, max_spans + 1)

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iterations):
        row_lse = torch.logsumexp(log_kernel + v.unsqueeze(-2), dim=-1)
        row_lse = torch.where(
            torch.isfinite(row_lse), row_lse, torch.zeros_like(row_lse)
        )
        u = torch.where(
            torch.isfinite(log_a), log_a - row_lse,
            torch.full_like(log_a, float("-inf")),
        )
        col_lse = torch.logsumexp(log_kernel + u.unsqueeze(-1), dim=-2)
        col_lse = torch.where(
            torch.isfinite(col_lse), col_lse, torch.zeros_like(col_lse)
        )
        v = torch.where(
            torch.isfinite(log_b), log_b - col_lse,
            torch.full_like(log_b, float("-inf")),
        )

    plan = torch.exp(log_kernel + u.unsqueeze(-1) + v.unsqueeze(-2))
    real_plan = plan[:, :, :max_words, :max_spans]
    matched_mass = real_plan.sum(dim=(-1, -2)).clamp(0.0, 1.0)
    matched_score = (real_plan * cosine).sum(dim=(-1, -2))
    return matched_score + float(dustbin_score) * (1.0 - matched_mass)


@torch.no_grad()
def retrieval_diagnostic(scores: torch.Tensor, video_ids: list[str],
                         text_keys: list[str]):
    batch_size = scores.shape[0]
    if batch_size < 2:
        return {"r1": None, "r5": None, "queries": 0, "queries_at5": 0}
    mask = torch.ones(
        batch_size, batch_size, dtype=torch.bool, device=scores.device
    )
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j and (
                video_ids[i] == video_ids[j] or text_keys[i] == text_keys[j]
            ):
                mask[i, j] = False
    valid = mask.sum(1) > 1
    if not valid.any():
        return {"r1": None, "r5": None, "queries": 0, "queries_at5": 0}
    ranked = scores.masked_fill(~mask, -1e4).argsort(dim=1, descending=True)
    hits = ranked == torch.arange(batch_size, device=scores.device)[:, None]
    ranks = hits.float().argmax(1)[valid]
    valid5 = mask.sum(1) >= 5
    ranks5 = hits.float().argmax(1)[valid5]
    return {
        "r1": float((ranks < 1).float().mean()),
        "r5": float((ranks5 < 5).float().mean()) if valid5.any() else None,
        "queries": int(valid.sum()),
        "queries_at5": int(valid5.sum()),
    }
