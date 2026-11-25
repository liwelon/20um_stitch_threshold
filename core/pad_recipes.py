# -*- coding: utf-8 -*-
"""按分辨率读写芯片 Pad 配方。"""
from __future__ import annotations

import json
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from core.first_pad import (
    DEFAULT_THRESH,
    PAD_H,
    PAD_MIN_NUM,
    PAD_RANGE,
    PAD_SEARCH_RANGE,
    PAD_W,
)
from core.stitching.params import app_base_dir

DEFAULT_CHIP_ID = "默认"
DEFAULT_PAD_TOP_TO_CMOS = 190

_PAD_RECIPES: dict[str, dict[str, "PadParams"]] = {}
_PAD_RECIPES_PATHS: dict[str, Path] = {}


@dataclass
class PadParams:
    pad_w: int = PAD_W
    pad_h: int = PAD_H
    pad_range: int = PAD_RANGE
    pad_search_range: int = PAD_SEARCH_RANGE
    pad_min_num: int = PAD_MIN_NUM
    pad_top_to_cmos_edge_pix: int = DEFAULT_PAD_TOP_TO_CMOS
    pad_gray_thresh: int = DEFAULT_THRESH

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict | None) -> "PadParams":
        if not data:
            return cls()
        defaults = cls()
        # 兼容旧字段 first_pad_binary_thresh
        thresh = data.get("pad_gray_thresh", data.get("first_pad_binary_thresh",
                                                     defaults.pad_gray_thresh))
        return cls(
            pad_w=int(data.get("pad_w", defaults.pad_w)),
            pad_h=int(data.get("pad_h", defaults.pad_h)),
            pad_range=int(data.get("pad_range", defaults.pad_range)),
            pad_search_range=int(
                data.get("pad_search_range", defaults.pad_search_range)
            ),
            pad_min_num=int(data.get("pad_min_num", defaults.pad_min_num)),
            pad_top_to_cmos_edge_pix=int(
                data.get(
                    "pad_top_to_cmos_edge_pix",
                    defaults.pad_top_to_cmos_edge_pix,
                )
            ),
            pad_gray_thresh=int(thresh),
        )


def default_pad_params(mode: str) -> PadParams:
    if mode == "20um":
        return PadParams()
    if mode == "5um":
        return PadParams(pad_w=66, pad_h=66)
    raise ValueError(f"unsupported resolution mode: {mode}")


def pad_recipes_json_path(mode: str) -> Path:
    if mode not in {"20um", "5um"}:
        raise ValueError(f"unsupported resolution mode: {mode}")
    if mode in _PAD_RECIPES_PATHS:
        return _PAD_RECIPES_PATHS[mode]

    file_name = f"pad_recipes_{mode}.json"
    candidates = [
        app_base_dir() / "config" / file_name,
        app_base_dir() / "resources" / file_name,
        Path(__file__).resolve().parents[1] / "config" / file_name,
        Path(__file__).resolve().parents[1] / "resources" / file_name,
    ]
    for path in candidates:
        if path.is_file():
            _PAD_RECIPES_PATHS[mode] = path
            return path

    _PAD_RECIPES_PATHS[mode] = candidates[0]
    return candidates[0]


def _ensure_default_file(path: Path, mode: str) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {DEFAULT_CHIP_ID: default_pad_params(mode).to_dict()}
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")


def _writable_path(path: Path, mode: str) -> Path:
    if path.parent.name == "resources":
        writable = app_base_dir() / "config" / path.name
        writable.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and not writable.is_file():
            writable.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        _PAD_RECIPES_PATHS[mode] = writable
        return writable
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_pad_recipes(mode: str, *, force: bool = False) -> dict[str, PadParams]:
    if mode in _PAD_RECIPES and not force:
        return _PAD_RECIPES[mode]

    path = pad_recipes_json_path(mode)
    _ensure_default_file(path, mode)
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid pad recipes json: {path}")

    recipes = {
        str(name): PadParams.from_dict(value if isinstance(value, dict) else None)
        for name, value in raw.items()
    }
    if not recipes:
        recipes = {DEFAULT_CHIP_ID: default_pad_params(mode)}
    _PAD_RECIPES[mode] = recipes
    return recipes


def list_chip_ids(mode: str) -> list[str]:
    return list(load_pad_recipes(mode).keys())


def get_pad_params(chip_id: str, mode: str) -> PadParams:
    recipes = load_pad_recipes(mode)
    if chip_id not in recipes:
        raise KeyError(f"unknown chip pad recipe: {chip_id}")
    return recipes[chip_id]


def get_default_chip_id(mode: str) -> str:
    recipes = load_pad_recipes(mode)
    if DEFAULT_CHIP_ID in recipes:
        return DEFAULT_CHIP_ID
    return next(iter(recipes))


def save_pad_recipe(
    chip_id: str,
    params: PadParams,
    mode: str,
    original_chip_id: str | None = None,
) -> None:
    path = _writable_path(pad_recipes_json_path(mode), mode)
    _ensure_default_file(path, mode)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        data = {}

    if original_chip_id != chip_id and chip_id in data:
        raise ValueError(f"芯片配方已存在: {chip_id}")
    if original_chip_id and original_chip_id != chip_id:
        if original_chip_id not in data:
            raise KeyError(f"unknown chip pad recipe: {original_chip_id}")
        del data[original_chip_id]
    data[chip_id] = params.to_dict()
    if path.is_file():
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")
    _PAD_RECIPES.pop(mode, None)


def delete_pad_recipe(chip_id: str, mode: str) -> None:
    path = _writable_path(pad_recipes_json_path(mode), mode)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or chip_id not in data:
        raise KeyError(f"unknown chip pad recipe: {chip_id}")
    if len(data) <= 1:
        raise ValueError("至少保留一个芯片配方")
    del data[chip_id]
    path.chmod(path.stat().st_mode | stat.S_IWRITE)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")
    _PAD_RECIPES.pop(mode, None)
