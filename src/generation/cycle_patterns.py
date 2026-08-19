"""Discover recurring appliance-cycle classes from merged state sequences."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np


def sequence_edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Levenshtein distance for short discrete state sequences."""
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (int(a) != int(b)),
            ))
        previous = current
    return previous[-1]


def normalized_sequence_distance(left: Sequence[int], right: Sequence[int]) -> float:
    return float(sequence_edit_distance(left, right) / max(len(left), len(right), 1))


def _signature(blocks: Iterable[dict]) -> tuple:
    labels = []
    for block in blocks:
        label = int(block["state_label"])
        if not labels or labels[-1] != label:
            labels.append(label)
    return tuple(labels)


class CyclePatternClassifier:
    """Classify full activities by recurring primitive-state combinations.

    Frequent exact signatures form class anchors. Rare signatures join their
    nearest anchor only when the normalized edit distance is small; otherwise
    they remain outliers and are excluded from synthesis by default.
    """

    def __init__(self, min_support: int = 3, max_classes: int = 12,
                 rare_max_distance: float = 0.34,
                 min_pattern_blocks: int = 3, min_unique_states: int = 2):
        self.min_support = max(1, int(min_support))
        self.max_classes = max(1, int(max_classes))
        self.rare_max_distance = float(rare_max_distance)
        self.min_pattern_blocks = max(1, int(min_pattern_blocks))
        self.min_unique_states = max(1, int(min_unique_states))

    def fit(self, sequences: Dict[str, List[dict]]) -> dict:
        activities = {}
        signature_counts = Counter()
        for activity_id, blocks in sequences.items():
            if not blocks:
                continue
            signature = _signature(blocks)
            if not signature:
                continue
            normalized_blocks = [{
                "state_label": int(block["state_label"]),
                "length_samples": int(block.get("length_samples", 0)),
            } for block in blocks if int(block.get("length_samples", 0)) > 0]
            if not normalized_blocks:
                continue
            activities[str(activity_id)] = {
                "signature": list(signature),
                "blocks": normalized_blocks,
                "length_samples": int(sum(b["length_samples"] for b in normalized_blocks)),
            }
            if len(signature) < self.min_pattern_blocks:
                activities[str(activity_id)]["class_id"] = -1
                activities[str(activity_id)]["outlier_reason"] = "too_few_blocks"
            elif len(set(signature)) < self.min_unique_states:
                activities[str(activity_id)]["class_id"] = -1
                activities[str(activity_id)]["outlier_reason"] = "too_few_unique_states"
            else:
                signature_counts[signature] += 1
        if not activities:
            raise ValueError("no valid state sequences available for cycle classification")
        if not signature_counts:
            raise ValueError(
                "all state sequences failed cycle completeness constraints; "
                "lower min_pattern_blocks/min_unique_states if appropriate")

        ranked = sorted(signature_counts.items(), key=lambda item: (-item[1], item[0]))
        anchors = [signature for signature, count in ranked
                   if count >= self.min_support][:self.max_classes]
        if not anchors:
            anchors = [signature for signature, _ in ranked[:self.max_classes]]

        class_members = {class_id: [] for class_id in range(len(anchors))}
        outliers = []
        for activity_id, record in activities.items():
            if record.get("class_id") == -1:
                record["distance_to_representative"] = None
                outliers.append(activity_id)
                continue
            signature = tuple(record["signature"])
            distances = [normalized_sequence_distance(signature, anchor) for anchor in anchors]
            class_id = int(np.argmin(distances))
            distance = float(distances[class_id])
            if signature != anchors[class_id] and distance > self.rare_max_distance:
                record["class_id"] = -1
                record["distance_to_representative"] = distance
                record["outlier_reason"] = "sequence_distance"
                outliers.append(activity_id)
            else:
                record["class_id"] = class_id
                record["distance_to_representative"] = distance
                class_members[class_id].append(activity_id)

        classes = []
        for class_id, representative in enumerate(anchors):
            members = class_members[class_id]
            if not members:
                continue
            lengths = np.asarray([activities[m]["length_samples"] for m in members], dtype=np.int64)
            signatures = Counter(tuple(activities[m]["signature"]) for m in members)
            classes.append({
                "class_id": int(class_id),
                "representative_signature": list(representative),
                "support": int(len(members)),
                "member_ids": members,
                "unique_signatures": int(len(signatures)),
                "signature_counts": {
                    "->".join(map(str, key)): int(value)
                    for key, value in sorted(signatures.items())
                },
                "duration_samples": {
                    "min": int(lengths.min()),
                    "median": float(np.median(lengths)),
                    "max": int(lengths.max()),
                },
            })

        valid_ids = {entry["class_id"] for entry in classes}
        for record in activities.values():
            if record["class_id"] not in valid_ids:
                record["class_id"] = -1

        return {
            "version": 1,
            "method": "frequent_signature_edit_distance",
            "parameters": {
                "min_support": self.min_support,
                "max_classes": self.max_classes,
                "rare_max_distance": self.rare_max_distance,
                "min_pattern_blocks": self.min_pattern_blocks,
                "min_unique_states": self.min_unique_states,
            },
            "n_activities": int(len(activities)),
            "n_classes": int(len(classes)),
            "n_outliers": int(sum(r["class_id"] == -1 for r in activities.values())),
            "classes": classes,
            "outlier_ids": sorted(k for k, r in activities.items() if r["class_id"] == -1),
            "activities": activities,
        }


class CyclePatternCatalog:
    """Read and sample the serializable output of CyclePatternClassifier."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.activities = payload.get("activities", {})
        self.classes = {int(entry["class_id"]): entry
                        for entry in payload.get("classes", [])}
        if not self.classes:
            raise ValueError("cycle pattern catalog has no valid classes")

    @property
    def class_ids(self) -> List[int]:
        return sorted(self.classes)

    def resolve_classes(self, selector: str | int | None) -> List[int]:
        value = "all" if selector is None else str(selector).strip().lower()
        if value in ("", "all"):
            return self.class_ids
        if value in ("majority", "largest"):
            return [max(self.class_ids, key=lambda cid: self.classes[cid]["support"])]
        if value.startswith("class_"):
            value = value[len("class_"):]
        class_id = int(value)
        if class_id not in self.classes:
            raise ValueError(f"unknown cycle class {selector}; available: {self.class_ids}")
        return [class_id]

    def sequences_for_class(self, class_id: int) -> List[List[dict]]:
        return [self.activities[str(activity_id)]["blocks"]
                for activity_id in self.classes[int(class_id)]["member_ids"]]

    def mode_ids_for_class(self, class_id: int) -> List[int]:
        return sorted({
            int(self.activities[str(activity_id)].get("validation_mode_id", 0))
            for activity_id in self.classes[int(class_id)]["member_ids"]
        })

    def member_ids(self, class_id: int, mode_id: int | None = None) -> List[str]:
        members = [str(value) for value in self.classes[int(class_id)]["member_ids"]]
        if mode_id is None:
            return members
        return [activity_id for activity_id in members
                if int(self.activities[activity_id].get("validation_mode_id", 0))
                == int(mode_id)]

    def sequences_for_mode(self, class_id: int, mode_id: int) -> List[List[dict]]:
        return [self.activities[activity_id]["blocks"]
                for activity_id in self.member_ids(class_id, mode_id)]

    def sample_activity(self, class_id: int, rng: np.random.Generator,
                        mode_id: int | None = None) -> tuple[str, List[dict]]:
        members = self.member_ids(class_id, mode_id)
        if not members:
            raise ValueError(f"class {class_id} mode {mode_id} has no activities")
        activity_id = str(members[int(rng.integers(0, len(members)))])
        return activity_id, self.activities[activity_id]["blocks"]
