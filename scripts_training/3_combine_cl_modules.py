"""Combine the online scoring memory, recall entries, and feedback counts of CL modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def validate_inputs(module_dirs: list[Path]) -> None:
    if not module_dirs:
        raise ValueError("At least one input module directory is required")

    for module_dir in module_dirs:
        require_file(module_dir / "config.json")
        require_file(module_dir / "score_cl" / "config.json")
        require_file(module_dir / "score_cl" / "offline_data.npz")


def validate_same_offline_memory(module_dirs: list[Path]) -> None:
    reference_hash = file_sha256(module_dirs[0] / "score_cl" / "offline_data.npz")
    for module_dir in module_dirs[1:]:
        offline_path = module_dir / "score_cl" / "offline_data.npz"
        if file_sha256(offline_path) != reference_hash:
            raise ValueError(
                "Offline memories differ. Expected the same offline_data.npz, "
                f"but {offline_path} does not match {module_dirs[0] / 'score_cl' / 'offline_data.npz'}"
            )


def validate_same_configs(module_dirs: list[Path]) -> None:
    reference_root_config = read_json(module_dirs[0] / "config.json", {})
    reference_score_config = read_json(module_dirs[0] / "score_cl" / "config.json", {})

    for module_dir in module_dirs[1:]:
        root_config = read_json(module_dir / "config.json", {})
        score_config = read_json(module_dir / "score_cl" / "config.json", {})
        if root_config != reference_root_config:
            raise ValueError(f"Root config differs for {module_dir}")
        if score_config != reference_score_config:
            raise ValueError(f"score_cl config differs for {module_dir}")


def prepare_output_dir(output_dir: Path, overwrite: bool = False) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_shared_module_files(source_dir: Path, output_dir: Path) -> None:
    shutil.copy2(source_dir / "config.json", output_dir / "config.json")
    root_config = read_json(source_dir / "config.json", {})
    if root_config.get("encoder_location", "external") == "internal":
        encoder_path = Path(root_config["encoder_path"])
        if encoder_path.is_absolute() or ".." in encoder_path.parts:
            raise ValueError(f"Invalid internal encoder path: {encoder_path}")
        source_encoder = source_dir / encoder_path
        require_file(source_encoder)
        target_encoder = output_dir / encoder_path
        target_encoder.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_encoder, target_encoder)

    score_output_dir = output_dir / "score_cl"
    score_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "score_cl" / "config.json", score_output_dir / "config.json")
    shutil.copy2(source_dir / "score_cl" / "offline_data.npz", score_output_dir / "offline_data.npz")


def load_online_data(module_dir: Path, feature_dim: int | None):
    online_path = module_dir / "score_cl" / "online_data.npz"
    if not online_path.exists():
        return None

    score_config = read_json(module_dir / "score_cl" / "config.json", {})
    default_weight = float(score_config.get("online_scale_factor", 1.0))
    with np.load(online_path) as online_data:
        x = np.asarray(online_data["X"])
        y = np.asarray(online_data["y"])
        if "w" in online_data:
            w = np.asarray(online_data["w"])
        else:
            w = np.full(y.shape, default_weight, dtype=float)

    if x.shape[0] == 0:
        return None
    if y.shape[0] != x.shape[0] or w.shape[0] != x.shape[0]:
        raise ValueError(f"Inconsistent online data row counts in {online_path}")
    if feature_dim is not None and x.shape[1] != feature_dim:
        raise ValueError(f"Feature dimension mismatch in {online_path}: {x.shape[1]} != {feature_dim}")
    return x, y, w


def infer_feature_dim(module_dir: Path) -> int:
    with np.load(module_dir / "score_cl" / "offline_data.npz") as offline_data:
        return int(offline_data["X"].shape[1])


def fuse_online_memory(module_dirs: list[Path], output_dir: Path) -> int:
    feature_dim = infer_feature_dim(module_dirs[0])
    online_parts = []
    for module_dir in module_dirs:
        online_data = load_online_data(module_dir, feature_dim)
        if online_data is not None:
            online_parts.append(online_data)

    output_path = output_dir / "score_cl" / "online_data.npz"
    if not online_parts:
        if output_path.exists():
            output_path.unlink()
        return 0

    x = np.concatenate([part[0] for part in online_parts], axis=0)
    y = np.concatenate([part[1] for part in online_parts], axis=0)
    w = np.concatenate([part[2] for part in online_parts], axis=0)
    np.savez_compressed(output_path, X=x, y=y, w=w)
    return int(x.shape[0])


def copy_recall_entries(module_dirs: list[Path], output_dir: Path, recall_name: str) -> int:
    copied_count = 0
    output_recall_dir = output_dir / recall_name
    output_recall_dir.mkdir(parents=True, exist_ok=True)

    for source_root in module_dirs:
        source_recall_dir = source_root / recall_name
        if not source_recall_dir.exists():
            continue

        for entry_dir in sorted(path for path in source_recall_dir.iterdir() if path.is_dir()):
            target_dir = output_recall_dir / entry_dir.name
            if target_dir.exists():
                target_dir = output_recall_dir / f"{entry_dir.name}_{uuid4().hex}"
            shutil.copytree(entry_dir, target_dir)
            copied_count += 1

    return copied_count


def add_feedback_counts(total: dict, counts: dict) -> None:
    for object_type, object_counts in counts.items():
        if object_type not in total:
            total[object_type] = {}
        for count_name, value in object_counts.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feedback count must be numeric: {object_type}.{count_name}={value!r}")
            total[object_type][count_name] = total[object_type].get(count_name, 0) + value


def fuse_feedback_counts(module_dirs: list[Path], output_dir: Path) -> dict:
    combined_counts: dict = {}
    for module_dir in module_dirs:
        counts = read_json(module_dir / "feedback_counts.json", {})
        add_feedback_counts(combined_counts, counts)
    write_json(output_dir / "feedback_counts.json", combined_counts)
    return combined_counts


def combine_cl_modules(
    module_dirs: list[Path],
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    module_dirs = [Path(module_dir) for module_dir in module_dirs]
    output_dir = Path(output_dir)

    validate_inputs(module_dirs)
    validate_same_configs(module_dirs)
    validate_same_offline_memory(module_dirs)
    prepare_output_dir(output_dir, overwrite=overwrite)

    copy_shared_module_files(module_dirs[0], output_dir)
    online_count = fuse_online_memory(module_dirs, output_dir)
    recall_count = copy_recall_entries(module_dirs, output_dir, "recall")
    recall_patch_count = copy_recall_entries(module_dirs, output_dir, "recall_patches")
    feedback_counts = fuse_feedback_counts(module_dirs, output_dir)

    print(f"combined_module={output_dir}")
    print(f"online_datapoints={online_count}")
    print(f"recall_entries={recall_count}")
    print(f"recall_patch_entries={recall_patch_count}")
    print(f"feedback_counts={feedback_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-module-dirs",
        type=Path,
        nargs="+",
        required=True,
        metavar="PATH",
        help="CL module directories to combine",
    )
    parser.add_argument(
        "--output-module-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="Directory in which to write the combined CL module",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists",
    )
    args = parser.parse_args()
    combine_cl_modules(
        module_dirs=args.input_module_dirs,
        output_dir=args.output_module_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
