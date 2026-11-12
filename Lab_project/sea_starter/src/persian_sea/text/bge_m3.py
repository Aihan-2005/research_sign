from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import unicodedata
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class BGEWordBatch:
    embeddings: np.ndarray
    word_embeddings: np.ndarray
    cue_offsets: np.ndarray
    word_texts: np.ndarray
    char_offsets: np.ndarray


class BGEM3DenseExtractor:
    def __init__(self, model_name="BAAI/bge-m3", device="auto", amp=True,
                 max_length=256, revision="main"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, use_fast=True
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("BGE word extraction requires a Fast Tokenizer with offset_mapping.")
        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    @staticmethod
    def _word_spans(text):
        spans, start = [], None
        for i, ch in enumerate(text):
            cat = unicodedata.category(ch)
            valid = cat[0] in ("L", "N", "M") or ch == "\u200c"
            if valid and start is None:
                start = i
            elif not valid and start is not None:
                word = text[start:i]
                if word.strip("\u200c"):
                    spans.append((word, start, i))
                start = None
        if start is not None:
            word = text[start:]
            if word.strip("\u200c"):
                spans.append((word, start, len(text)))
        return spans

    @staticmethod
    def _complete_word_token_indices(word_start, word_end, token_offsets):
        indices, covered = [], []
        for i, (a, b) in enumerate(token_offsets):
            if b > a and max(a, word_start) < min(b, word_end):
                indices.append(i)
                covered.append((max(a, word_start), min(b, word_end)))
        if not indices:
            return []
        covered.sort()
        cursor = word_start
        for a, b in covered:
            if a > cursor:
                return []
            cursor = max(cursor, b)
        return indices if cursor >= word_end else []

    def _amp_context(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.amp else nullcontext()

    @torch.inference_mode()
    def encode_with_words(self, texts, batch_size=16):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return BGEWordBatch(
                np.empty((0, self.hidden_size), np.float32),
                np.empty((0, self.hidden_size), np.float32),
                np.zeros(1, np.int64), np.empty(0, dtype=str),
                np.empty((0, 2), np.int32),
            )

        sentence_chunks, word_vectors, word_texts, char_offsets = [], [], [], []
        cue_offsets = [0]
        for base in range(0, len(texts), batch_size):
            batch_texts = texts[base:base + batch_size]
            encoded = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping").cpu().tolist()
            model_inputs = {k: v.to(self.device) for k, v in encoded.items()}
            with self._amp_context():
                hidden = self.model(**model_inputs).last_hidden_state.float()
                sentence = F.normalize(hidden[:, 0], p=2, dim=-1)
            sentence_chunks.append(sentence.cpu().numpy())
            attention = model_inputs["attention_mask"].bool().cpu().tolist()

            for row, text in enumerate(batch_texts):
                positions = [i for i, keep in enumerate(attention[row]) if keep]
                token_offsets = [
                    (int(offsets[row][i][0]), int(offsets[row][i][1]))
                    for i in positions
                ]
                row_vectors = hidden[row, positions]
                for word, ws, we in self._word_spans(text):
                    ids = self._complete_word_token_indices(ws, we, token_offsets)
                    if not ids:

                        
                        continue
                    pooled = row_vectors[ids].mean(dim=0)
                    pooled = F.normalize(pooled, p=2, dim=-1)
                    word_vectors.append(pooled.cpu().numpy())
                    word_texts.append(word)
                    char_offsets.append((ws, we))
                cue_offsets.append(len(word_vectors))

        return BGEWordBatch(
            embeddings=np.concatenate(sentence_chunks).astype(np.float32),
            word_embeddings=(
                np.stack(word_vectors).astype(np.float32) if word_vectors
                else np.empty((0, self.hidden_size), np.float32)
            ),
            cue_offsets=np.asarray(cue_offsets, np.int64),
            word_texts=np.asarray(word_texts, dtype=str),
            char_offsets=np.asarray(char_offsets, np.int32).reshape(-1, 2),
        )

    @torch.inference_mode()
    def encode(self, texts, batch_size=16):
        """Backward-compatible sentence-only API."""
        return self.encode_with_words(texts, batch_size).embeddings
