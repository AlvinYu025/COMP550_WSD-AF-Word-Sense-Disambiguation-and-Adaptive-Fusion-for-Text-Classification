#!/usr/bin/env python3
"""Joint sentence-level multi-aspect learning for MAMS-ATSA.

MAMS guarantees that every sentence contains at least two aspects with
different sentiment labels. This runner models all aspects in a sentence
together and uses that dataset property during training and decoding.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve
from xml.etree import ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
NUM_LABELS = len(LABEL_TO_ID)
MAMS_RAW_URL = (
    "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/"
    "master/data/MAMS-ATSA/raw/{split}.xml"
)
DEFAULT_ROOT = Path.home() / ".cache" / "wsdaf" / "mams"


@dataclass(frozen=True)
class Example:
    uid: str
    sentence_id: str
    text: str
    aspect: str
    label: int
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_ROOT / "data" / "MAMS-ATSA" / "raw",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_ROOT / "hf",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_ROOT / "runs" / "joint-relations",
    )
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--relation-layers", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--diversity-weight", type=float, default=0.2)
    parser.add_argument("--pair-loss-weight", type=float, default=0.2)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-on-validation", action="store_true")
    parser.add_argument("--fixed-pair-decode-weight", type=float, default=0.1)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--reuse-model", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def ensure_mams_files(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        destination = data_dir / f"{split}.xml"
        if not destination.exists():
            urlretrieve(MAMS_RAW_URL.format(split=split), destination)


def load_split(path: Path) -> list[Example]:
    root = ET.parse(path).getroot()
    examples = []
    for sentence_index, sentence in enumerate(root.findall("./sentence")):
        sentence_id = sentence.attrib.get(
            "id",
            f"{path.stem}-{sentence_index:05d}",
        )
        text = sentence.findtext("text")
        if text is None:
            continue
        for aspect_index, aspect in enumerate(
            sentence.findall("./aspectTerms/aspectTerm")
        ):
            polarity = aspect.attrib["polarity"]
            if polarity not in LABEL_TO_ID:
                continue
            start = int(aspect.attrib["from"])
            end = int(aspect.attrib["to"])
            examples.append(
                Example(
                    uid=f"{path.stem}:{sentence_id}:{aspect_index}:{start}:{end}",
                    sentence_id=sentence_id,
                    text=text,
                    aspect=aspect.attrib["term"],
                    label=LABEL_TO_ID[polarity],
                    start=start,
                    end=end,
                )
            )
    return examples


def group_examples(examples: list[Example]) -> list[tuple[Example, ...]]:
    grouped: dict[str, list[Example]] = {}
    for example in examples:
        grouped.setdefault(example.sentence_id, []).append(example)
    return [
        tuple(sorted(group, key=lambda item: (item.start, item.end, item.uid)))
        for group in grouped.values()
    ]


class SentenceAspectDataset(Dataset):
    def __init__(
        self,
        groups: list[tuple[Example, ...]],
        repeats: int,
        tokenizer,
        max_length: int,
    ) -> None:
        self.groups = [group for group in groups for _ in range(repeats)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, object]:
        examples = self.groups[index]
        encodings = [
            self.tokenizer(
                example.text,
                example.aspect,
                truncation=True,
                max_length=self.max_length,
            )
            for example in examples
        ]
        return {
            "encodings": encodings,
            "labels": [example.label for example in examples],
            "uids": [example.uid for example in examples],
            "sentence_id": examples[0].sentence_id,
        }


class SentenceCollator:
    def __init__(self, tokenizer) -> None:
        self.padding = DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt",
        )

    def __call__(self, items: list[dict[str, object]]) -> dict[str, object]:
        encodings = [encoding for item in items for encoding in item["encodings"]]
        batch = dict(self.padding(encodings))
        batch["labels"] = torch.tensor(
            [label for item in items for label in item["labels"]],
            dtype=torch.long,
        )
        batch["group_sizes"] = torch.tensor(
            [len(item["labels"]) for item in items],
            dtype=torch.long,
        )
        batch["uids"] = [uid for item in items for uid in item["uids"]]
        batch["sentence_ids"] = [
            sentence_id
            for item in items
            for sentence_id in [item["sentence_id"]] * len(item["labels"])
        ]
        return batch


class RelationBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
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
    ) -> torch.Tensor:
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + attended)
        return self.ffn_norm(hidden + self.ffn(hidden))


class JointAspectModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
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
        self.input_dropout = nn.Dropout(args.dropout)
        self.relation_blocks = nn.ModuleList(
            RelationBlock(
                hidden_size,
                args.attention_heads,
                args.dropout,
            )
            for _ in range(args.relation_layers)
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, NUM_LABELS),
        )
        self.pair_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    @staticmethod
    def pack(
        hidden: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = group_sizes.tolist()
        groups = list(hidden.split(sizes))
        padded = pad_sequence(groups, batch_first=True)
        lengths = torch.tensor(sizes, device=hidden.device)
        positions = torch.arange(padded.shape[1], device=hidden.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        return padded, padding_mask

    def pair_logits_for_group(self, hidden: torch.Tensor) -> torch.Tensor:
        count = hidden.shape[0]
        if count < 2:
            return hidden.new_zeros((count, count))
        indices = torch.triu_indices(count, count, offset=1, device=hidden.device)
        left = hidden[indices[0]]
        right = hidden[indices[1]]
        features = torch.cat((torch.abs(left - right), left * right), dim=-1)
        values = self.pair_classifier(features).squeeze(-1)
        matrix = torch.zeros(
            (count, count),
            dtype=values.dtype,
            device=values.device,
        )
        matrix[indices[0], indices[1]] = values
        matrix[indices[1], indices[0]] = values
        return matrix

    def forward(
        self,
        group_sizes: torch.Tensor,
        **inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        encoded = self.backbone(**inputs).last_hidden_state[:, 0]
        base, padding_mask = self.pack(self.input_dropout(encoded), group_sizes)
        related = base
        for block in self.relation_blocks:
            related = block(related, padding_mask)
        gate = torch.sigmoid(self.gate(torch.cat((base, related), dim=-1)))
        fused = base + gate * (related - base)
        valid = ~padding_mask
        base_flat = base[valid]
        fused_flat = fused[valid]
        features = torch.cat(
            (
                base_flat,
                fused_flat,
                torch.abs(base_flat - fused_flat),
                base_flat * fused_flat,
            ),
            dim=-1,
        )
        logits = self.classifier(features)
        group_hidden = [
            group[:size]
            for group, size in zip(fused, group_sizes.tolist(), strict=True)
        ]
        pair_logits = [self.pair_logits_for_group(group) for group in group_hidden]
        return logits, pair_logits


def diversity_loss(
    logits: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    probabilities = logits.float().softmax(dim=-1)
    losses = []
    for group in probabilities.split(group_sizes.tolist()):
        all_same_probability = group.prod(dim=0).sum()
        losses.append(-torch.log1p(-all_same_probability.clamp(max=1.0 - 1e-6)))
    return torch.stack(losses).mean()


def pair_relation_loss(
    pair_logits: list[torch.Tensor],
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    losses = []
    offset = 0
    for matrix, size in zip(pair_logits, group_sizes.tolist(), strict=True):
        group_labels = labels[offset : offset + size]
        offset += size
        indices = torch.triu_indices(size, size, offset=1, device=labels.device)
        targets = (group_labels[indices[0]] == group_labels[indices[1]]).float()
        losses.append(
            F.binary_cross_entropy_with_logits(
                matrix[indices[0], indices[1]],
                targets,
            )
        )
    return torch.stack(losses).mean()


def make_loader(
    groups: list[tuple[Example, ...]],
    repeats: int,
    tokenizer,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = SentenceAspectDataset(
        groups,
        repeats,
        tokenizer,
        args.max_length,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=SentenceCollator(tokenizer),
        num_workers=2,
        pin_memory=True,
        generator=generator,
    )


def move_inputs(
    batch: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    labels = batch["labels"].to(device, non_blocking=True)
    group_sizes = batch["group_sizes"].to(device, non_blocking=True)
    inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key not in {"labels", "group_sizes"}
    }
    return inputs, labels, group_sizes


def metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
    }


def hard_diversity_decode(
    scores: np.ndarray,
    group_sizes: list[int],
) -> np.ndarray:
    predictions = scores.argmax(axis=1)
    offset = 0
    for size in group_sizes:
        group_scores = scores[offset : offset + size]
        group_predictions = predictions[offset : offset + size]
        if np.unique(group_predictions).size == 1:
            current = group_predictions[0]
            log_scores = np.log(np.clip(group_scores, 1e-12, 1.0))
            costs = log_scores[:, [current]] - log_scores
            costs[:, current] = np.inf
            aspect_index, label = np.unravel_index(
                np.argmin(costs),
                costs.shape,
            )
            group_predictions[aspect_index] = label
        offset += size
    return predictions


@lru_cache(maxsize=None)
def valid_assignments(count: int) -> np.ndarray:
    assignments = np.asarray(
        list(itertools.product(range(NUM_LABELS), repeat=count)),
        dtype=np.int8,
    )
    return assignments[np.any(assignments != assignments[:, :1], axis=1)]


def exact_relation_decode_group(
    scores: np.ndarray,
    pair_probabilities: np.ndarray,
    pair_weight: float,
) -> np.ndarray:
    count = scores.shape[0]
    assignments = valid_assignments(count)
    log_scores = np.log(np.clip(scores, 1e-12, 1.0))
    unary = log_scores[np.arange(count), assignments].sum(axis=1)
    if pair_weight == 0.0:
        return assignments[np.argmax(unary)]

    pair_score = np.zeros(assignments.shape[0], dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            same_probability = np.clip(
                pair_probabilities[left, right],
                1e-6,
                1.0 - 1e-6,
            )
            same = assignments[:, left] == assignments[:, right]
            pair_score += np.where(
                same,
                np.log(same_probability),
                np.log1p(-same_probability),
            )
    pair_score *= 2.0 / max(1, count - 1)
    return assignments[np.argmax(unary + pair_weight * pair_score)]


def relation_decode(
    scores: np.ndarray,
    pair_probabilities: list[np.ndarray],
    group_sizes: list[int],
    pair_weight: float,
) -> np.ndarray:
    predictions = []
    offset = 0
    for size, pair_matrix in zip(
        group_sizes,
        pair_probabilities,
        strict=True,
    ):
        predictions.append(
            exact_relation_decode_group(
                scores[offset : offset + size],
                pair_matrix,
                pair_weight,
            )
        )
        offset += size
    return np.concatenate(predictions)


@torch.inference_mode()
def predict(
    model: JointAspectModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    scores = []
    labels = []
    uids = []
    sentence_ids = []
    group_sizes = []
    pair_probabilities = []
    for batch in loader:
        inputs, batch_labels, batch_group_sizes = move_inputs(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, pair_logits = model(batch_group_sizes, **inputs)
        scores.append(logits.float().softmax(dim=-1).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
        uids.extend(batch["uids"])
        sentence_ids.extend(batch["sentence_ids"])
        group_sizes.extend(batch_group_sizes.cpu().tolist())
        pair_probabilities.extend(
            torch.sigmoid(matrix.float()).cpu().numpy() for matrix in pair_logits
        )
    return {
        "scores": np.concatenate(scores),
        "labels": np.concatenate(labels),
        "uids": uids,
        "sentence_ids": sentence_ids,
        "group_sizes": group_sizes,
        "pair_probabilities": pair_probabilities,
    }


def evaluate_predictions(
    prediction: dict[str, object],
    pair_weights: tuple[float, ...],
) -> dict[str, object]:
    scores = prediction["scores"]
    labels = prediction["labels"]
    group_sizes = prediction["group_sizes"]
    return {
        "independent": metrics(labels, scores.argmax(axis=1)),
        "hard_diversity": metrics(
            labels,
            hard_diversity_decode(scores, group_sizes),
        ),
        "relation_decode": {
            str(weight): metrics(
                labels,
                relation_decode(
                    scores,
                    prediction["pair_probabilities"],
                    group_sizes,
                    weight,
                ),
            )
            for weight in pair_weights
        },
    }


def train(
    model: JointAspectModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
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
    update_steps = math.ceil(len(train_loader) / args.grad_accumulation) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(update_steps * args.warmup_ratio)),
        num_training_steps=update_steps,
    )
    best_f1 = -1.0
    best_epoch = 0
    patience_used = 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "classification": 0.0, "diversity": 0.0, "pair": 0.0}
        started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            inputs, labels, group_sizes = move_inputs(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, pair_logits = model(group_sizes, **inputs)
                classification = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=args.label_smoothing,
                )
                diversity = diversity_loss(logits, group_sizes)
                pair = pair_relation_loss(pair_logits, labels, group_sizes)
                loss = (
                    classification
                    + args.diversity_weight * diversity
                    + args.pair_loss_weight * pair
                )
                scaled_loss = loss / args.grad_accumulation
            scaled_loss.backward()
            if step % args.grad_accumulation == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            totals["loss"] += loss.item()
            totals["classification"] += classification.item()
            totals["diversity"] += diversity.item()
            totals["pair"] += pair.item()

        record = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "train_loss": totals["loss"] / len(train_loader),
            "classification_loss": totals["classification"] / len(train_loader),
            "diversity_loss": totals["diversity"] / len(train_loader),
            "pair_loss": totals["pair"] / len(train_loader),
        }
        selection_f1 = None
        if val_loader is not None:
            val_prediction = predict(model, val_loader, device)
            val_results = evaluate_predictions(
                val_prediction,
                pair_weights=(0.0,),
            )
            record["validation"] = val_results
            selection_f1 = val_results["hard_diversity"]["macro_f1"]
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        if val_loader is None:
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        elif selection_f1 is not None and selection_f1 > best_f1:
            best_f1 = selection_f1
            best_epoch = epoch
            patience_used = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience_used += 1
            if patience_used >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    selection = {"best_epoch": best_epoch}
    if val_loader is None:
        selection["selection_metric"] = "fixed epochs after validation lock"
    else:
        selection.update(
            {
                "selection_metric": "validation hard-diversity macro_f1",
                "best_macro_f1": best_f1,
            }
        )
    return history, selection


def serializable_prediction(
    prediction: dict[str, object],
) -> dict[str, object]:
    return {
        "scores": prediction["scores"].tolist(),
        "labels": prediction["labels"].tolist(),
        "uids": prediction["uids"],
        "sentence_ids": prediction["sentence_ids"],
        "group_sizes": prediction["group_sizes"],
        "pair_probabilities": [
            matrix.tolist() for matrix in prediction["pair_probabilities"]
        ],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    ensure_mams_files(args.data_dir)
    splits = {
        split: load_split(args.data_dir / f"{split}.xml")
        for split in ("train", "val", "test")
    }
    groups = {split: group_examples(examples) for split, examples in splits.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )

    training_groups = groups["train"]
    if args.train_on_validation:
        training_groups = training_groups + groups["val"]
    train_loader = make_loader(
        training_groups,
        args.train_repeats,
        tokenizer,
        args,
        shuffle=True,
    )
    val_loader = None
    if not args.train_on_validation:
        val_loader = make_loader(
            groups["val"],
            1,
            tokenizer,
            args,
            shuffle=False,
        )

    model_tag = args.model_name.rsplit("/", maxsplit=1)[-1]
    final_tag = "-train-plus-val" if args.train_on_validation else ""
    run_name = f"joint-{model_tag}{final_tag}-seed{args.seed}"
    output_dir = args.run_dir / run_name
    checkpoint = output_dir / "best.pt"
    model = JointAspectModel(args).to(device)
    if args.reuse_model and checkpoint.exists():
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.to(device)
        history = []
        selection = {"reused_checkpoint": str(checkpoint)}
    else:
        history, selection = train(
            model,
            train_loader,
            val_loader,
            args,
            checkpoint,
            device,
        )

    val_evaluation = None
    val_prediction = None
    if args.train_on_validation:
        best_pair_weight = args.fixed_pair_decode_weight
    else:
        pair_weights = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
        val_prediction = predict(model, val_loader, device)
        val_evaluation = evaluate_predictions(val_prediction, pair_weights)
        best_pair_weight = max(
            pair_weights,
            key=lambda weight: val_evaluation["relation_decode"][str(weight)][
                "macro_f1"
            ],
        )

    result = {
        "configuration": {
            **vars(args),
            "data_dir": str(args.data_dir),
            "cache_dir": str(args.cache_dir),
            "run_dir": str(args.run_dir),
            "train_sentences": len(training_groups),
            "train_aspects": sum(len(group) for group in training_groups),
            "selection_split": (
                "fixed from prior official-validation run"
                if args.train_on_validation
                else "official validation"
            ),
        },
        "history": history,
        "selection": {
            **selection,
            "pair_decode_weight": best_pair_weight,
            "decode_selection": (
                "fixed from prior official-validation run"
                if args.train_on_validation
                else "official validation macro_f1"
            ),
        },
    }
    if val_evaluation is not None and val_prediction is not None:
        result["validation"] = val_evaluation
        result["validation_predictions"] = serializable_prediction(val_prediction)
    if args.evaluate_test:
        test_loader = make_loader(
            groups["test"],
            1,
            tokenizer,
            args,
            shuffle=False,
        )
        test_prediction = predict(model, test_loader, device)
        result["test"] = evaluate_predictions(
            test_prediction,
            (best_pair_weight,),
        )
        result["test_predictions"] = serializable_prediction(test_prediction)

    output = output_dir / "result.json"
    save_json(output, result)
    summary = {
        "run": run_name,
        "checkpoint": str(checkpoint),
        "selected_pair_weight": best_pair_weight,
    }
    if val_evaluation is not None:
        summary["validation"] = val_evaluation
    if "test" in result:
        summary["test"] = result["test"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
