#!/usr/bin/env python3
"""Sense-conditioned relational attention for MAMS-ATSA.

The model uses one supervised objective: aspect sentiment cross-entropy.
WordNet/GlossBERT sense distributions and frozen gloss prototypes are used
only in the model forward pass. They provide pairwise semantic biases to
relation attention rather than an auxiliary prediction loss.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mams_joint_relations as common
import nltk
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

PERSISTENT_HOME = Path.home() / "private"
DEFAULT_STORAGE = (
    PERSISTENT_HOME if PERSISTENT_HOME.is_dir() else Path.home() / ".cache"
)
DEFAULT_ROOT = Path(
    os.environ.get("WSDAF_STORAGE", DEFAULT_STORAGE / "wsdaf-sense-relational")
)
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
POS_TO_WORDNET = {
    "NOUN": "n",
    "PROPN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}
METHODS = ("baseline", "sense", "relation", "full")
PAIR_FEATURE_MODES = ("raw", "orthogonal", "semantic-only")
ASPECT_REPRESENTATIONS = ("cls", "scope")
RELATION_ATTENTION_MODES = ("self", "cross", "relative", "cross-relative")
RELATION_OUTPUT_MODES = ("contextual", "residual-logit", "residual-context")


@dataclass(frozen=True)
class Candidate:
    synset: str
    definition: str
    score: float


@dataclass(frozen=True)
class AspectSense:
    target: str
    lemma: str
    pos: str
    target_start: int
    target_end: int
    candidates: tuple[Candidate, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, default="full")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_ROOT / "data" / "MAMS-ATSA" / "raw",
    )
    parser.add_argument(
        "--sense-cache",
        type=Path,
        default=DEFAULT_ROOT / "mams_aspect_senses.json.gz",
    )
    parser.add_argument(
        "--prototype-cache",
        type=Path,
        default=DEFAULT_ROOT / "mams_soft_prototypes_v2.npz",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "hf")
    parser.add_argument("--nltk-dir", type=Path, default=DEFAULT_ROOT / "nltk")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_ROOT / "runs" / "sense-relational",
    )
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--prototype-model", default="roberta-base")
    parser.add_argument("--wsd-model-name", default="kanishka/GlossBERT")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--wsd-max-length", type=int, default=192)
    parser.add_argument("--wsd-batch-size", type=int, default=128)
    parser.add_argument("--prototype-batch-size", type=int, default=128)
    parser.add_argument("--sense-temperature", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--relation-layers", type=int, default=1)
    parser.add_argument(
        "--relation-attention-mode",
        choices=RELATION_ATTENTION_MODES,
        default="self",
    )
    parser.add_argument(
        "--relation-output-mode",
        choices=RELATION_OUTPUT_MODES,
        default="contextual",
    )
    parser.add_argument("--max-relative-rank", type=int, default=4)
    parser.add_argument(
        "--aspect-representation",
        choices=ASPECT_REPRESENTATIONS,
        default="cls",
    )
    parser.add_argument(
        "--pair-feature-mode",
        choices=PAIR_FEATURE_MODES,
        default="orthogonal",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--rebuild-senses", action="store_true")
    parser.add_argument("--rebuild-prototypes", action="store_true")
    parser.add_argument("--prepare-senses-only", action="store_true")
    parser.add_argument("--train-on-validation", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--reuse-model", action="store_true")
    return parser.parse_args()


def ensure_wordnet(nltk_dir: Path) -> None:
    nltk_dir.mkdir(parents=True, exist_ok=True)
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    for resource in ("wordnet", "omw-1.4"):
        archive = nltk_dir / "corpora" / f"{resource}.zip"
        if archive.exists():
            continue
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=str(nltk_dir), quiet=True)


def choose_target(
    example: common.Example,
    doc,
) -> tuple[str, str, str, int, int] | None:
    span = doc.char_span(example.start, example.end, alignment_mode="expand")
    if span is None or len(span) == 0:
        return None

    content = [token for token in span if token.pos_ in CONTENT_POS]
    if not content:
        return None
    head_candidates = [
        token for token in content if token.head.i not in range(span.start, span.end)
    ]
    head = head_candidates[-1] if head_candidates else content[-1]
    phrase_lemma = "_".join(token.lemma_.lower() for token in content)
    if "_" in phrase_lemma:
        phrase_synsets = wordnet.synsets(
            phrase_lemma,
            pos=POS_TO_WORDNET[head.pos_],
        )
        if phrase_synsets:
            return (
                example.aspect,
                phrase_lemma,
                head.pos_,
                example.start,
                example.end,
            )
    return (
        head.text,
        head.lemma_.lower(),
        head.pos_,
        head.idx,
        head.idx + len(head.text),
    )


def mark_target(text: str, start: int, end: int) -> str:
    return text[:start] + '" ' + text[start:end] + ' "' + text[end:]


def serialize_senses(senses: dict[str, AspectSense]) -> dict[str, object]:
    return {
        uid: {
            "target": sense.target,
            "lemma": sense.lemma,
            "pos": sense.pos,
            "target_start": sense.target_start,
            "target_end": sense.target_end,
            "candidates": [asdict(candidate) for candidate in sense.candidates],
        }
        for uid, sense in senses.items()
    }


def deserialize_senses(payload: dict[str, object]) -> dict[str, AspectSense]:
    return {
        uid: AspectSense(
            target=raw["target"],
            lemma=raw["lemma"],
            pos=raw["pos"],
            target_start=raw["target_start"],
            target_end=raw["target_end"],
            candidates=tuple(Candidate(**candidate) for candidate in raw["candidates"]),
        )
        for uid, raw in payload.items()
    }


def build_sense_cache(
    examples: list[common.Example],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, AspectSense]:
    if args.sense_cache.exists() and not args.rebuild_senses:
        with gzip.open(args.sense_cache, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        senses = deserialize_senses(payload["senses"])
        if all(example.uid in senses for example in examples):
            print(f"Loaded sense cache: {args.sense_cache}", flush=True)
            return senses

    ensure_wordnet(args.nltk_dir)
    nlp = spacy.load(args.spacy_model, disable=["ner"])
    unique_texts = list(dict.fromkeys(example.text for example in examples))
    docs = {
        text: doc
        for text, doc in zip(
            unique_texts,
            nlp.pipe(unique_texts, batch_size=128),
            strict=True,
        )
    }

    metadata = {}
    pair_contexts = []
    pair_glosses = []
    pair_ids = []
    for example in examples:
        target = choose_target(example, docs[example.text])
        if target is None:
            metadata[example.uid] = ("", "", "", 0, 0)
            continue
        target_text, lemma, pos, start, end = target
        metadata[example.uid] = (target_text, lemma, pos, start, end)
        synsets = wordnet.synsets(lemma, pos=POS_TO_WORDNET[pos])
        marked_context = mark_target(example.text, start, end)
        for synset in synsets:
            pair_contexts.append(marked_context)
            pair_glosses.append(f"{target_text}: {synset.definition()}")
            pair_ids.append((example.uid, synset))

    tokenizer = AutoTokenizer.from_pretrained(
        args.wsd_model_name,
        cache_dir=args.cache_dir,
    )
    teacher = BertForSequenceClassification.from_pretrained(
        args.wsd_model_name,
        cache_dir=args.cache_dir,
    ).to(device)
    teacher.eval()
    candidate_scores = np.empty(len(pair_ids), dtype=np.float32)
    for start in range(0, len(pair_ids), args.wsd_batch_size):
        end = min(start + args.wsd_batch_size, len(pair_ids))
        encoded = tokenizer(
            pair_contexts[start:end],
            pair_glosses[start:end],
            padding=True,
            truncation=True,
            max_length=args.wsd_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ),
        ):
            logits = teacher(**encoded).logits
        candidate_scores[start:end] = logits.float().softmax(dim=-1)[:, 1].cpu().numpy()
        if end == len(pair_ids) or end % 20000 < args.wsd_batch_size:
            print(f"GlossBERT: {end}/{len(pair_ids)}", flush=True)

    grouped: dict[str, list[Candidate]] = {}
    for score, (uid, synset) in zip(candidate_scores, pair_ids, strict=True):
        grouped.setdefault(uid, []).append(
            Candidate(
                synset=synset.name(),
                definition=synset.definition(),
                score=float(score),
            )
        )

    senses = {}
    for example in examples:
        target, lemma, pos, start, end = metadata[example.uid]
        candidates = tuple(
            sorted(
                grouped.get(example.uid, []),
                key=lambda candidate: candidate.score,
                reverse=True,
            )
        )
        senses[example.uid] = AspectSense(
            target=target,
            lemma=lemma,
            pos=pos,
            target_start=start,
            target_end=end,
            candidates=candidates,
        )

    covered = sum(bool(sense.candidates) for sense in senses.values())
    ambiguous = sum(len(sense.candidates) > 1 for sense in senses.values())
    payload = {
        "statistics": {
            "instances": len(examples),
            "covered": covered,
            "ambiguous": ambiguous,
            "candidate_pairs": len(pair_ids),
        },
        "senses": serialize_senses(senses),
    }
    args.sense_cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.sense_cache, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    teacher.to("cpu")
    del teacher
    torch.cuda.empty_cache()
    print(json.dumps(payload["statistics"], indent=2), flush=True)
    return senses


@torch.inference_mode()
def encode_glosses(
    texts: list[str],
    tokenizer,
    encoder: nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = encoder(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        vectors.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
    return np.concatenate(vectors)


def build_prototype_cache(
    examples: list[common.Example],
    senses: dict[str, AspectSense],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float], int]:
    if args.prototype_cache.exists() and not args.rebuild_prototypes:
        payload = np.load(args.prototype_cache)
        uids = payload["uids"].tolist()
        vectors = payload["vectors"].astype(np.float32)
        confidences = payload["confidences"].astype(np.float32)
        return (
            dict(zip(uids, vectors, strict=True)),
            dict(zip(uids, confidences.tolist(), strict=True)),
            vectors.shape[1],
        )

    ensure_wordnet(args.nltk_dir)
    unique_synsets = sorted(
        {
            candidate.synset
            for sense in senses.values()
            for candidate in sense.candidates
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.prototype_model,
        cache_dir=args.cache_dir,
    )
    encoder_options = {}
    if args.prototype_model == "roberta-base":
        encoder_options["add_pooling_layer"] = False
    encoder = AutoModel.from_pretrained(
        args.prototype_model,
        cache_dir=args.cache_dir,
        **encoder_options,
    ).to(device)
    encoder.eval()
    gloss_texts = []
    for synset_name in unique_synsets:
        synset = wordnet.synset(synset_name)
        lemmas = ", ".join(name.replace("_", " ") for name in synset.lemma_names())
        gloss_texts.append(f"{lemmas}. {synset.definition()}")
    matrix = encode_glosses(
        gloss_texts,
        tokenizer,
        encoder,
        args.prototype_batch_size,
        device,
    )
    synset_to_row = {synset: row for row, synset in enumerate(unique_synsets)}
    hidden_size = matrix.shape[1]

    uids = []
    vectors = []
    confidences = []
    for example in examples:
        sense = senses[example.uid]
        if not sense.candidates:
            vector = np.zeros(hidden_size, dtype=np.float32)
            confidence = 0.0
        elif len(sense.candidates) == 1:
            vector = matrix[synset_to_row[sense.candidates[0].synset]].copy()
            confidence = 1.0
        else:
            scores = np.asarray(
                [candidate.score for candidate in sense.candidates],
                dtype=np.float32,
            )
            shifted = scores / args.sense_temperature
            weights = np.exp(shifted - shifted.max())
            weights /= weights.sum()
            rows = np.asarray(
                [synset_to_row[candidate.synset] for candidate in sense.candidates]
            )
            vector = weights @ matrix[rows]
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            entropy = -np.sum(weights * np.log(np.clip(weights, 1e-9, 1.0)))
            confidence = 1.0 - entropy / math.log(len(weights))
        uids.append(example.uid)
        vectors.append(vector)
        confidences.append(confidence)

    vector_array = np.asarray(vectors, dtype=np.float16)
    confidence_array = np.asarray(confidences, dtype=np.float32)
    args.prototype_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.prototype_cache,
        uids=np.asarray(uids),
        vectors=vector_array,
        confidences=confidence_array,
    )
    encoder.to("cpu")
    del encoder
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "prototype_instances": len(uids),
                "unique_synsets": len(unique_synsets),
                "mean_confidence": float(confidence_array.mean()),
                "nonzero_confidence": int((confidence_array > 0).sum()),
            },
            indent=2,
        ),
        flush=True,
    )
    return (
        dict(zip(uids, vector_array.astype(np.float32), strict=True)),
        dict(zip(uids, confidence_array.tolist(), strict=True)),
        hidden_size,
    )


class SentenceSenseDataset(Dataset):
    def __init__(
        self,
        groups: list[tuple[common.Example, ...]],
        repeats: int,
        tokenizer,
        prototypes: dict[str, np.ndarray],
        confidences: dict[str, float],
        max_length: int,
    ) -> None:
        self.groups = [group for group in groups for _ in range(repeats)]
        self.tokenizer = tokenizer
        self.prototypes = prototypes
        self.confidences = confidences
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, object]:
        group = self.groups[index]
        encodings = []
        sentence_masks = []
        aspect_masks = []
        for example in group:
            encoding = self.tokenizer(
                example.text,
                example.aspect,
                truncation=True,
                max_length=self.max_length,
            )
            sequence_ids = encoding.sequence_ids()
            encodings.append(dict(encoding))
            sentence_masks.append([sequence_id == 0 for sequence_id in sequence_ids])
            aspect_masks.append([sequence_id == 1 for sequence_id in sequence_ids])
        return {
            "encodings": encodings,
            "sentence_masks": sentence_masks,
            "aspect_masks": aspect_masks,
            "sense_vectors": [self.prototypes[example.uid] for example in group],
            "sense_confidences": [self.confidences[example.uid] for example in group],
            "labels": [example.label for example in group],
            "uids": [example.uid for example in group],
        }


class SentenceSenseCollator:
    def __init__(self, tokenizer) -> None:
        self.padding_side = tokenizer.padding_side
        self.padding = DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt",
        )

    def __call__(self, items: list[dict[str, object]]) -> dict[str, object]:
        encodings = [encoding for item in items for encoding in item["encodings"]]
        batch = dict(self.padding(encodings))
        max_length = batch["input_ids"].shape[1]
        for output_name, item_name in (
            ("sentence_mask", "sentence_masks"),
            ("aspect_mask", "aspect_masks"),
        ):
            masks = [mask for item in items for mask in item[item_name]]
            padded_masks = torch.zeros(
                len(masks),
                max_length,
                dtype=torch.bool,
            )
            for row, mask in enumerate(masks):
                mask_tensor = torch.tensor(mask, dtype=torch.bool)
                if self.padding_side == "right":
                    padded_masks[row, : len(mask)] = mask_tensor
                else:
                    padded_masks[row, -len(mask) :] = mask_tensor
            batch[output_name] = padded_masks
        batch["sense_vectors"] = torch.tensor(
            np.asarray([vector for item in items for vector in item["sense_vectors"]]),
            dtype=torch.float32,
        )
        batch["sense_confidences"] = torch.tensor(
            [confidence for item in items for confidence in item["sense_confidences"]],
            dtype=torch.float32,
        )
        batch["labels"] = torch.tensor(
            [label for item in items for label in item["labels"]],
            dtype=torch.long,
        )
        batch["group_sizes"] = torch.tensor(
            [len(item["labels"]) for item in items],
            dtype=torch.long,
        )
        batch["uids"] = [uid for item in items for uid in item["uids"]]
        return batch


class RelationBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout: float,
        max_relative_rank: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.max_relative_rank = max_relative_rank
        self.attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.sense_bias_weights = nn.Parameter(torch.zeros(heads, 2))
        self.relative_rank_bias = nn.Parameter(
            torch.zeros(heads, 2 * max_relative_rank + 1)
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
        sense_pair_features: torch.Tensor | None = None,
        use_relative_rank: bool = False,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        attention_bias = None
        key_padding_mask = padding_mask
        if sense_pair_features is not None:
            attention_bias = torch.einsum(
                "bijf,hf->bhij",
                sense_pair_features,
                self.sense_bias_weights,
            )
        if use_relative_rank:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            relative_rank = positions[None, :] - positions[:, None]
            relative_rank = relative_rank.clamp(
                -self.max_relative_rank,
                self.max_relative_rank,
            )
            relative_rank = relative_rank + self.max_relative_rank
            rank_bias = self.relative_rank_bias[:, relative_rank]
            rank_bias = rank_bias.unsqueeze(0).expand(hidden.shape[0], -1, -1, -1)
            attention_bias = (
                rank_bias if attention_bias is None else attention_bias + rank_bias
            )
        if exclude_self:
            if attention_bias is None:
                attention_bias = hidden.new_zeros(
                    hidden.shape[0],
                    self.heads,
                    hidden.shape[1],
                    hidden.shape[1],
                )
            diagonal = torch.eye(
                hidden.shape[1],
                dtype=torch.bool,
                device=hidden.device,
            )
            attention_bias = attention_bias.masked_fill(
                diagonal[None, None],
                torch.finfo(attention_bias.dtype).min,
            )
        attention_mask = None
        if attention_bias is not None:
            attention_bias = attention_bias.masked_fill(
                padding_mask[:, None, None, :],
                torch.finfo(attention_bias.dtype).min,
            )
            attention_mask = attention_bias.flatten(0, 1)
            key_padding_mask = None
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + attended)
        return self.ffn_norm(hidden + self.ffn(hidden))


class SenseRelationalModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        prototype_size: int,
        prototype_center: np.ndarray,
        semantic_center: float,
        semantic_scale: float,
    ) -> None:
        super().__init__()
        backbone_options = {}
        if args.model_name == "roberta-base":
            backbone_options["add_pooling_layer"] = False
        self.backbone = AutoModel.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            **backbone_options,
        )
        hidden_size = self.backbone.config.hidden_size
        self.method = args.method
        self.pair_feature_mode = args.pair_feature_mode
        self.aspect_representation = args.aspect_representation
        self.relation_attention_mode = args.relation_attention_mode
        self.relation_output_mode = args.relation_output_mode
        self.use_sense = args.method in {"sense", "full"}
        self.use_relation = args.method in {"relation", "full"}
        self.register_buffer(
            "prototype_center",
            torch.tensor(prototype_center, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "semantic_center",
            torch.tensor(semantic_center, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "semantic_scale",
            torch.tensor(semantic_scale, dtype=torch.float32),
            persistent=False,
        )
        self.input_dropout = nn.Dropout(args.dropout)
        self.sense_projection = nn.Sequential(
            nn.LayerNorm(prototype_size),
            nn.Linear(prototype_size, hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.sense_norm = nn.LayerNorm(hidden_size)
        self.relation_blocks = nn.ModuleList(
            RelationBlock(
                hidden_size,
                args.attention_heads,
                args.dropout,
                args.max_relative_rank,
            )
            for _ in range(args.relation_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, common.NUM_LABELS),
        )
        self.relation_delta_classifier = None
        if self.relation_output_mode.startswith("residual-"):
            self.relation_delta_classifier = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(hidden_size, common.NUM_LABELS, bias=False),
            )
            nn.init.zeros_(self.relation_delta_classifier[-1].weight)

    @staticmethod
    def pack(
        hidden: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = list(hidden.split(group_sizes.tolist()))
        padded = pad_sequence(groups, batch_first=True)
        lengths = torch.tensor(group_sizes.tolist(), device=hidden.device)
        positions = torch.arange(padded.shape[1], device=hidden.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        return padded, padding_mask

    def build_sense_pair_features(
        self,
        sense_vectors: torch.Tensor,
        sense_confidences: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> torch.Tensor:
        padded_senses, padding_mask = self.pack(sense_vectors, group_sizes)
        normalized_senses = F.normalize(
            padded_senses - self.prototype_center,
            dim=-1,
        )
        similarity = normalized_senses @ normalized_senses.transpose(1, 2)

        certainty = (2.0 * sense_confidences - 1.0).clamp_min(0.0)
        padded_certainty, _ = self.pack(certainty.unsqueeze(-1), group_sizes)
        reliability = padded_certainty * padded_certainty.transpose(1, 2)

        semantic_relation = reliability * similarity
        if self.pair_feature_mode != "raw":
            semantic_relation = (
                reliability * (similarity - self.semantic_center) / self.semantic_scale
            )
        reliability_feature = reliability
        if self.pair_feature_mode == "semantic-only":
            reliability_feature = torch.zeros_like(reliability)
        pair_features = torch.stack(
            (reliability_feature, semantic_relation),
            dim=-1,
        )
        pair_features.masked_fill_(
            padding_mask[:, :, None, None] | padding_mask[:, None, :, None],
            0.0,
        )
        diagonal = torch.eye(
            pair_features.shape[1],
            dtype=torch.bool,
            device=pair_features.device,
        )
        pair_features.masked_fill_(diagonal[None, :, :, None], 0.0)
        return pair_features

    def forward(
        self,
        group_sizes: torch.Tensor,
        sense_vectors: torch.Tensor,
        sense_confidences: torch.Tensor,
        sentence_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        **inputs: torch.Tensor,
    ) -> torch.Tensor:
        token_hidden = self.backbone(**inputs).last_hidden_state
        cls_hidden = token_hidden[:, 0]
        if self.aspect_representation == "scope":
            aspect_weights = aspect_mask.unsqueeze(-1).to(token_hidden.dtype)
            aspect_hidden = (token_hidden * aspect_weights).sum(dim=1)
            aspect_hidden = aspect_hidden / aspect_weights.sum(dim=1).clamp_min(1)
            scope_scores = torch.einsum(
                "bld,bd->bl",
                token_hidden,
                aspect_hidden,
            ) / math.sqrt(token_hidden.shape[-1])
            scope_scores = scope_scores.masked_fill(
                ~sentence_mask,
                torch.finfo(scope_scores.dtype).min,
            )
            scope_weights = scope_scores.softmax(dim=-1)
            evidence_hidden = torch.einsum(
                "bl,bld->bd",
                scope_weights,
                token_hidden,
            )
            base = F.layer_norm(
                cls_hidden + aspect_hidden + evidence_hidden,
                (token_hidden.shape[-1],),
            )
        else:
            base = cls_hidden
        base = self.input_dropout(base)
        hidden = base
        if self.method == "sense":
            sense_delta = self.sense_norm(self.sense_projection(sense_vectors))
            certainty_margin = (2.0 * sense_confidences - 1.0).clamp_min(0.0)
            hidden = base + certainty_margin.unsqueeze(-1) * sense_delta
        padded, padding_mask = self.pack(hidden, group_sizes)
        independent = padded
        if self.use_relation:
            pair_features = None
            if self.method == "full":
                pair_features = self.build_sense_pair_features(
                    sense_vectors,
                    sense_confidences,
                    group_sizes,
                )
            for block in self.relation_blocks:
                padded = block(
                    padded,
                    padding_mask,
                    pair_features,
                    use_relative_rank=(
                        self.relation_attention_mode in {"relative", "cross-relative"}
                    ),
                    exclude_self=(
                        self.relation_attention_mode in {"cross", "cross-relative"}
                    ),
                )
        final = padded[~padding_mask]
        if self.use_relation and self.relation_output_mode.startswith("residual-"):
            independent = independent[~padding_mask]
            independent_logits = self.classifier(self.output_norm(independent))
            relation_delta = (
                self.output_norm(final)
                if self.relation_output_mode == "residual-context"
                else final - independent
            )
            assert self.relation_delta_classifier is not None
            return independent_logits + self.relation_delta_classifier(relation_delta)
        return self.classifier(self.output_norm(final))


def compute_semantic_pair_statistics(
    groups: list[tuple[common.Example, ...]],
    prototypes: dict[str, np.ndarray],
    confidences: dict[str, float],
    prototype_center: np.ndarray,
) -> tuple[float, float]:
    pair_values = []
    pair_reliabilities = []
    for group in groups:
        vectors = np.stack([prototypes[example.uid] for example in group])
        vectors = vectors - prototype_center
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.divide(
            vectors,
            norms,
            out=np.zeros_like(vectors),
            where=norms > 0,
        )
        similarity = normalized @ normalized.T
        certainty = np.maximum(
            0.0,
            2.0 * np.asarray([confidences[example.uid] for example in group]) - 1.0,
        )
        reliability = np.outer(certainty, certainty)
        off_diagonal = ~np.eye(len(group), dtype=bool)
        valid = off_diagonal & (reliability > 0)
        if valid.any():
            pair_values.append(similarity[valid])
            pair_reliabilities.append(reliability[valid])

    similarities = np.concatenate(pair_values)
    reliabilities = np.concatenate(pair_reliabilities)
    squared_reliability = reliabilities**2
    semantic_center = float(
        np.sum(squared_reliability * similarities) / np.sum(squared_reliability)
    )
    residual = reliabilities * (similarities - semantic_center)
    semantic_scale = float(np.sqrt(np.mean(residual**2)))
    return semantic_center, max(semantic_scale, 1e-6)


def make_loader(
    groups: list[tuple[common.Example, ...]],
    repeats: int,
    tokenizer,
    prototypes: dict[str, np.ndarray],
    confidences: dict[str, float],
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = SentenceSenseDataset(
        groups,
        repeats,
        tokenizer,
        prototypes,
        confidences,
        args.max_length,
    )
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=SentenceSenseCollator(tokenizer),
        num_workers=2,
        pin_memory=True,
        generator=generator,
    )


def move_batch(
    batch: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    labels = batch["labels"].to(device, non_blocking=True)
    group_sizes = batch["group_sizes"].to(device, non_blocking=True)
    model_inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key not in {"labels", "group_sizes"}
    }
    return model_inputs, labels, group_sizes


@torch.inference_mode()
def predict(
    model: SenseRelationalModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    scores = []
    labels = []
    uids = []
    for batch in loader:
        inputs, batch_labels, group_sizes = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(group_sizes, **inputs)
        scores.append(logits.float().softmax(dim=-1).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
        uids.extend(batch["uids"])
    scores_array = np.concatenate(scores)
    labels_array = np.concatenate(labels)
    predictions = scores_array.argmax(axis=1)
    return {
        "scores": scores_array,
        "labels": labels_array,
        "uids": uids,
        "metrics": {
            "accuracy": float(accuracy_score(labels_array, predictions)),
            "macro_f1": float(f1_score(labels_array, predictions, average="macro")),
            "weighted_f1": float(
                f1_score(labels_array, predictions, average="weighted")
            ),
        },
    }


def train(
    model: SenseRelationalModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    class_weights: torch.Tensor | None,
    args: argparse.Namespace,
    checkpoint: Path,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    backbone_parameters = list(model.backbone.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": head_parameters,
                "lr": args.head_learning_rate,
                "weight_decay": args.weight_decay,
            },
        ]
    )
    updates = args.max_updates or (
        math.ceil(len(train_loader) / args.grad_accumulation) * args.epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(updates * args.warmup_ratio)),
        num_training_steps=updates,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    best_epoch = 0
    patience_used = 0
    updates_completed = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            inputs, labels, group_sizes = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(group_sizes, **inputs)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                    label_smoothing=args.label_smoothing,
                )
            (loss / args.grad_accumulation).backward()
            if step % args.grad_accumulation == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates_completed += 1
            total_loss += loss.item()
            if updates_completed >= updates:
                break

        record = {
            "epoch": epoch,
            "updates_completed": updates_completed,
            "seconds": time.time() - started,
            "train_sentiment_loss": total_loss / step,
        }
        if val_loader is None:
            torch.save(model.state_dict(), checkpoint)
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if updates_completed >= updates:
                break
            continue

        validation = predict(model, val_loader, device)
        record["validation"] = validation["metrics"]
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        current_f1 = validation["metrics"]["macro_f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            patience_used = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience_used += 1
            if patience_used >= args.patience:
                break
        if updates_completed >= updates:
            break

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    if val_loader is None:
        return history, {
            "fixed_epochs": args.epochs,
            "selection_metric": (
                "epoch count locked on official validation before "
                "train+validation refit"
            ),
        }
    return history, {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "selection_metric": "official validation macro_f1",
    }


def serializable_prediction(prediction: dict[str, object]) -> dict[str, object]:
    return {
        "scores": prediction["scores"].tolist(),
        "labels": prediction["labels"].tolist(),
        "uids": prediction["uids"],
        "metrics": prediction["metrics"],
    }


def load_model_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {name for name in missing if name.endswith("relative_rank_bias")}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.train_on_validation and not args.evaluate_test:
        raise ValueError("--train-on-validation requires --evaluate-test")
    if args.class_weight_power < 0:
        raise ValueError("--class-weight-power must be nonnegative")
    common.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    common.ensure_mams_files(args.data_dir)
    splits = {
        split: common.load_split(args.data_dir / f"{split}.xml")
        for split in ("train", "val", "test")
    }
    all_examples = splits["train"] + splits["val"] + splits["test"]
    senses = build_sense_cache(all_examples, args, device)
    prototypes, confidences, prototype_size = build_prototype_cache(
        all_examples,
        senses,
        args,
        device,
    )
    if args.prepare_senses_only:
        return

    groups = {
        split: common.group_examples(examples) for split, examples in splits.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    train_prototypes = np.stack(
        [prototypes[example.uid] for example in splits["train"]]
    )
    covered_train = np.linalg.norm(train_prototypes, axis=1) > 0
    prototype_center = train_prototypes[covered_train].mean(axis=0)
    semantic_center, semantic_scale = compute_semantic_pair_statistics(
        groups["train"],
        prototypes,
        confidences,
        prototype_center,
    )
    training_groups = groups["train"]
    if args.train_on_validation:
        training_groups = training_groups + groups["val"]
    label_counts = np.bincount(
        [example.label for group in training_groups for example in group],
        minlength=common.NUM_LABELS,
    ).astype(np.float64)
    class_weights = None
    class_weight_values = np.ones(common.NUM_LABELS, dtype=np.float64)
    if args.class_weight_power > 0:
        class_weight_values = (label_counts.mean() / label_counts) ** (
            args.class_weight_power
        )
        class_weight_values /= class_weight_values.mean()
        class_weights = torch.tensor(
            class_weight_values,
            dtype=torch.float32,
            device=device,
        )
    train_loader = make_loader(
        training_groups,
        args.train_repeats,
        tokenizer,
        prototypes,
        confidences,
        args,
        shuffle=True,
    )
    val_loader = make_loader(
        groups["val"],
        1,
        tokenizer,
        prototypes,
        confidences,
        args,
        shuffle=False,
    )

    model_tag = args.model_name.rsplit("/", 1)[-1]
    run_tag = f"-{args.run_tag}" if args.run_tag else ""
    if args.train_on_validation:
        budget_tag = (
            f"updates{args.max_updates}" if args.max_updates else f"epochs{args.epochs}"
        )
        run_name = (
            f"{args.method}{run_tag}-{model_tag}-trainval-{budget_tag}"
            f"-seed{args.seed}"
        )
    else:
        run_name = f"{args.method}{run_tag}-{model_tag}-seed{args.seed}"
    output_dir = args.run_dir / run_name
    checkpoint = output_dir / "best.pt"
    model = SenseRelationalModel(
        args,
        prototype_size,
        prototype_center,
        semantic_center,
        semantic_scale,
    ).to(device)
    if args.reuse_model and checkpoint.exists():
        load_model_checkpoint(model, checkpoint)
        model.to(device)
        history = []
        selection = {"reused_checkpoint": str(checkpoint)}
    else:
        history, selection = train(
            model,
            train_loader,
            None if args.train_on_validation else val_loader,
            class_weights,
            args,
            checkpoint,
            device,
        )

    validation = None
    if not args.train_on_validation:
        validation = predict(model, val_loader, device)
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    result = {
        "configuration": {
            **serialized_args,
            "objective": (
                "class-balanced sentiment cross-entropy only"
                if args.class_weight_power > 0
                else "sentiment cross-entropy only"
            ),
            "label_counts": label_counts.tolist(),
            "class_weights": class_weight_values.tolist(),
            "use_sense": model.use_sense,
            "use_relation": model.use_relation,
            "prototype_size": prototype_size,
            "semantic_center": semantic_center,
            "semantic_scale": semantic_scale,
        },
        "history": history,
        "selection": selection,
    }
    if validation is not None:
        result["validation"] = serializable_prediction(validation)
    if args.evaluate_test:
        test_loader = make_loader(
            groups["test"],
            1,
            tokenizer,
            prototypes,
            confidences,
            args,
            shuffle=False,
        )
        result["test"] = serializable_prediction(predict(model, test_loader, device))

    output = output_dir / "result.json"
    common.save_json(output, result)
    summary = {
        "run": run_name,
        "checkpoint": str(checkpoint),
    }
    if validation is not None:
        summary["validation"] = validation["metrics"]
    if "test" in result:
        summary["test"] = result["test"]["metrics"]
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
