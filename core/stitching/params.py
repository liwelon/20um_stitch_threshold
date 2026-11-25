from __future__ import annotations

import json
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JointParams:
    h_dx: int
    h_dy: int
    v_dx: int
    v_dy: int
    rows: int = 2
    cols: int = 2
    layers: int = 2
    single_w: int = 4512
    single_h: int = 4096
    channel: int = 3

    @property
    def joint_count(self) -> int:
        return self.rows * self.cols * self.layers


DEFAULT_MODE = "20um"
DEFAULT_MACHINES = {
    "20um": "1#20um",
    "5um": "1#5um",
}

_MACHINE_PRESETS: dict[str, dict[str, JointParams]] = {}
_PARAMS_PATHS: dict[str, Path] = {}


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def params_json_path(mode: str = DEFAULT_MODE) -> Path:
    if mode in _PARAMS_PATHS:
        return _PARAMS_PATHS[mode]

    file_name = f"params_{mode}.json"

    candidates = [app_base_dir() / "config" / file_name]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "resources" / file_name)
    candidates.extend(
        [
            app_base_dir() / "resources" / file_name,
            Path(__file__).resolve().parents[2] / "resources" / file_name,
        ]
    )

    for path in candidates:
        if path.is_file():
            _PARAMS_PATHS[mode] = path
            return path

    _PARAMS_PATHS[mode] = candidates[0]
    return _PARAMS_PATHS[mode]


def _parse_machine(raw: dict) -> JointParams:
    return JointParams(
        h_dx=int(raw["hDx"]),
        h_dy=int(raw["hDy"]),
        v_dx=int(raw["vDx"]),
        v_dy=int(raw["vDy"]),
        rows=int(raw.get("rows", 2)),
        cols=int(raw.get("cols", 2)),
        layers=int(raw.get("layers", 2)),
        single_w=int(raw.get("singleW", 4512)),
        single_h=int(raw.get("singleH", 4096)),
        channel=int(raw.get("channel", 3)),
    )


def _serialize_machine(params: JointParams) -> dict[str, int]:
    return {
        "hDx": params.h_dx,
        "hDy": params.h_dy,
        "vDx": params.v_dx,
        "vDy": params.v_dy,
        "rows": params.rows,
        "cols": params.cols,
        "layers": params.layers,
        "singleW": params.single_w,
        "singleH": params.single_h,
        "channel": params.channel,
    }


def load_machine_presets(
    mode: str = DEFAULT_MODE,
    force_reload: bool = False,
) -> dict[str, JointParams]:
    if mode in _MACHINE_PRESETS and not force_reload:
        return _MACHINE_PRESETS[mode]

    path = params_json_path(mode)
    if not path.is_file():
        raise FileNotFoundError(f"stitch params json not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"invalid stitch params json: {path}")

    presets: dict[str, JointParams] = {}
    for machine_id, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid machine entry: {machine_id}")
        params = _parse_machine(raw)
        if mode == "5um" and (
            params.rows < 1 or params.cols < 1 or params.layers != 1
        ):
            raise ValueError(
                f"5um machine {machine_id} must use positive rows and cols, layers=1"
            )
        presets[str(machine_id)] = params

    _MACHINE_PRESETS[mode] = presets
    return presets


def list_machine_ids(mode: str = DEFAULT_MODE) -> list[str]:
    return list(load_machine_presets(mode).keys())


def get_default_machine(mode: str = DEFAULT_MODE) -> str:
    presets = load_machine_presets(mode)
    default_machine = DEFAULT_MACHINES.get(mode, "")
    if default_machine in presets:
        return default_machine
    return next(iter(presets))


def get_joint_params(machine_id: str, mode: str = DEFAULT_MODE) -> JointParams:
    presets = load_machine_presets(mode)
    if machine_id not in presets:
        raise KeyError(f"unknown machine preset: {machine_id}")
    return presets[machine_id]


# Lazy-compatible alias used by UI
def get_machine_presets(mode: str = DEFAULT_MODE) -> dict[str, JointParams]:
    return load_machine_presets(mode)


def save_machine_preset(
    mode: str,
    machine_id: str,
    params: JointParams,
    original_machine_id: str | None = None,
) -> None:
    path = params_json_path(mode)
    if path.parent.name == "resources":
        writable_path = app_base_dir() / "config" / path.name
        writable_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, writable_path)
        path = writable_path
        _PARAMS_PATHS[mode] = path
    if path.is_file():
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"invalid stitch params json: {path}")
    else:
        data = {}

    if original_machine_id != machine_id and machine_id in data:
        raise ValueError(f"machine preset already exists: {machine_id}")
    if original_machine_id and original_machine_id != machine_id:
        if original_machine_id not in data:
            raise KeyError(f"unknown machine preset: {original_machine_id}")
        del data[original_machine_id]
    data[machine_id] = _serialize_machine(params)
    if path.is_file():
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")
    _MACHINE_PRESETS.pop(mode, None)


def delete_machine_preset(mode: str, machine_id: str) -> None:
    path = params_json_path(mode)
    if path.parent.name == "resources":
        writable_path = app_base_dir() / "config" / path.name
        writable_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, writable_path)
        path = writable_path
        _PARAMS_PATHS[mode] = path
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or machine_id not in data:
        raise KeyError(f"unknown machine preset: {machine_id}")
    del data[machine_id]
    path.chmod(path.stat().st_mode | stat.S_IWRITE)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")
    _MACHINE_PRESETS.pop(mode, None)
