from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class MirageConfig:
    num_frames: int = 33
    start_frame: int = 0
    infer_steps: int = 20
    seed: int = 42
    height: int | None = None
    width: int | None = None


@dataclass
class VideoGeometry:
    frames: list[Any] | None = None
    depths: list[Any] | None = None
    intrinsics: list[Any] | None = None
    poses_c2w: list[Any] | None = None


class MiragePipeline:
    def __init__(self, pipe: Any | None = None, config: MirageConfig | None = None) -> None:
        self.pipe = pipe
        self.config = config or MirageConfig()

    def generate(
        self,
        *,
        geometry_path: Path | str,
        prompt: str,
        output_dir: Path | str,
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(geometry_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "config": asdict(self.config),
            "geometry_path": str(path),
            "prompt": prompt,
            "output_dir": str(output),
        }
        if run_metadata is not None:
            metadata.update(run_metadata)
        return metadata
