from __future__ import annotations

import copy
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


@dataclass
class DataConfig:
    data_path: Path = Path("data")
    random_sample_ref: bool = False
    random_sample_preceding: bool = False
    max_reference_frames: int | None = None
    max_preceding_frames: int | None = None


class MirageDataset(Dataset):
    """Lightweight dataset wrapper for simple sample files.

    The upstream project stores samples in LMDB shards, but this lightweight
    implementation supports a directory containing pickled or JSON samples.
    """

    def __init__(self, config: DataConfig, model_version: str):
        self.config = config
        self.data_path = Path(config.data_path)
        self.model_version = model_version
        self.samples: list[Any] = []
        self._load_samples()

    def _load_samples(self) -> None:
        if not self.data_path.exists():
            return

        for path in sorted(self.data_path.iterdir()):
            if path.is_file() and path.suffix in {".pkl", ".pickle"}:
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
                if isinstance(payload, list):
                    self.samples.extend(payload)
                else:
                    self.samples.append(payload)
            elif path.is_file() and path.suffix == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    self.samples.extend(payload)
                else:
                    self.samples.append(payload)
            elif path.is_file() and path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        if isinstance(sample, dict):
            normalized = copy.deepcopy(sample)
        else:
            normalized = {"value": sample}

        normalized.setdefault("idx", idx)
        normalized.setdefault("prompts", "")
        normalized.setdefault("target_latent", [])
        normalized.setdefault("target_scene_proj", [])
        normalized.setdefault("preceding_scene_proj", [])
        normalized.setdefault("preceding_latent", [])
        normalized.setdefault("reference_latent", [])
        normalized.setdefault("img", None)
        normalized.setdefault("meta", {})
        return normalized
