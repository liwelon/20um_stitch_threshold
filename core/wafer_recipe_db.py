from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str


DEFAULT_DATABASE_CONFIG = DatabaseConfig(
    host="localhost",
    port=3306,
    database="szx",
    username="root",
    password=os.environ.get("WAFER_DB_PASSWORD", ""),
)


@dataclass(frozen=True)
class WaferRecipe:
    recipe_id: int | None
    recipe_name: str
    chip_model: str
    params: dict[str, Any]


def _merge_params(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_params(current, value)
        else:
            merged[key] = value
    return merged


def _parse_params_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def classify_recipe_mode(params: dict[str, Any]) -> str | None:
    if isinstance(params.get("DetectionParams"), dict):
        return "5um"
    keys = {str(key).lower() for key in params}
    if {"detectparams", "splitparams"} & keys:
        return "20um"
    return None


def load_exported_wafer_recipes(
    file_path: str,
    resolution_mode: str,
) -> list[WaferRecipe]:
    with Path(file_path).open("r", encoding="utf-8-sig") as file:
        exported = json.load(file)

    if isinstance(exported, dict):
        rows = [exported]
    elif isinstance(exported, list):
        rows = exported
    else:
        raise ValueError("导出文件根节点必须是配方对象或配方数组。")

    recipes: list[WaferRecipe] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {index} 项配方不是对象。")
        if "params_json" not in row:
            raise ValueError(f"第 {index} 项配方缺少 params_json。")

        raw_params = row["params_json"]
        try:
            params = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {index} 项配方的 params_json 格式错误。") from exc
        if not isinstance(params, dict):
            raise ValueError(f"第 {index} 项配方的 params_json 必须是对象。")
        if classify_recipe_mode(params) != resolution_mode:
            continue

        recipe_name = str(row.get("recipe_name") or "").strip()
        if not recipe_name:
            raise ValueError(f"第 {index} 项配方缺少 recipe_name。")
        recipe_id = row.get("recipe_id")
        try:
            parsed_recipe_id = int(recipe_id) if recipe_id is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {index} 项配方的 recipe_id 无效。") from exc

        recipes.append(
            WaferRecipe(
                recipe_id=parsed_recipe_id,
                recipe_name=recipe_name,
                chip_model=str(row.get("chip_model") or ""),
                params=params,
            )
        )

    if not recipes:
        raise ValueError(f"文件中没有匹配 {resolution_mode} 结构的配方。")
    return recipes


def load_wafer_recipes(
    config: DatabaseConfig,
    resolution_mode: str,
) -> list[WaferRecipe]:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise RuntimeError("缺少 PyMySQL，请先安装 requirements.txt 依赖。") from exc

    recipe_table = "20um_recipe_info" if resolution_mode == "20um" else "recipe_info"
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.username,
        password=config.password,
        database=config.database,
        charset="utf8mb4",
        cursorclass=DictCursor,
        connect_timeout=5,
        read_timeout=10,
        write_timeout=10,
        autocommit=True,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT r.recipe_id, r.recipe_name, r.chip_model, "
                "r.params_json AS recipe_params_json, "
                "c.params_json AS chip_params_json "
                f"FROM `{recipe_table}` AS r "
                "LEFT JOIN chip_info AS c ON c.chip_model = r.chip_model "
                "ORDER BY r.last_modified DESC, r.recipe_id DESC"
            )
            rows = cursor.fetchall()
    finally:
        connection.close()

    recipes: list[WaferRecipe] = []
    for row in rows:
        recipe_params = _parse_params_json(row.get("recipe_params_json"))
        chip_params = _parse_params_json(row.get("chip_params_json"))
        params = _merge_params(chip_params, recipe_params)
        if not params:
            continue
        if classify_recipe_mode(params) != resolution_mode:
            continue
        recipes.append(
            WaferRecipe(
                recipe_id=(
                    int(row["recipe_id"])
                    if row.get("recipe_id") is not None
                    else None
                ),
                recipe_name=str(row.get("recipe_name") or ""),
                chip_model=str(row.get("chip_model") or ""),
                params=params,
            )
        )
    return recipes
