"""Train a binary classifier from the reviewed 5um crop images and masks."""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet18


SOURCE_PATTERN = re.compile(r"^\d+__(?:blind|circuit)__(.+?\.(?:jpg|png|bmp))_", re.I)
CLASS_NAMES = ("无目标", "有目标")


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"无法编码图像：{path}")
    encoded.tofile(str(path))


def tile_starts(length: int, size: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def patch_at(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    patch = image[y:y + size, x:x + size]
    bottom = size - patch.shape[0]
    right = size - patch.shape[1]
    if bottom or right:
        patch = cv2.copyMakeBorder(
            patch, 0, bottom, 0, right, cv2.BORDER_REFLECT_101
        )
    return patch


def prepare_samples(
    data_root: Path, output_dir: Path, size: int, negatives_per_image: int, seed: int
) -> list[dict]:
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    reviews_dir = data_root / "reviews"
    if not all(path.is_dir() for path in (images_dir, masks_dir, reviews_dir)):
        raise FileNotFoundError("数据目录必须包含 images、masks 和 reviews 子目录。")

    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"没有找到裁剪图：{images_dir}")

    samples = []
    rng = random.Random(seed)
    for image_path in image_paths:
        mask_path = masks_dir / image_path.name
        review_path = reviews_dir / f"{image_path.stem}.json"
        if not mask_path.is_file() or not review_path.is_file():
            raise FileNotFoundError(f"缺少同名掩膜或审核文件：{image_path.name}")
        match = SOURCE_PATTERN.match(image_path.name)
        if match is None:
            raise ValueError(f"无法从文件名提取原图标识：{image_path.name}")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("state") != "approved" or review.get("wholeImageChecked") is not True:
            raise ValueError(f"图片尚未完整审核：{image_path.name}")
        if review.get("ignoredPixels") != 0:
            raise ValueError(f"掩膜含忽略区域，不能直接生成二分类标签：{image_path.name}")

        mask = read_image(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim != 2 or not np.isin(np.unique(mask), (0, 1, 255)).all():
            raise ValueError(f"掩膜必须为单通道 0/1 或 0/255：{mask_path}")
        image = read_image(image_path, cv2.IMREAD_GRAYSCALE)
        if image.shape != mask.shape:
            raise ValueError(f"图片与掩膜尺寸不一致：{image_path.name}")
        foreground = (mask != 0).astype(np.uint8)
        if int(foreground.sum()) != review.get("defectPixels"):
            raise ValueError(f"掩膜目标像素数与审核记录不一致：{image_path.name}")

        height, width = mask.shape
        coordinates: dict[tuple[int, int], int] = {}
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            x = max(0, min(bx + bw // 2 - size // 2, width - size))
            y = max(0, min(by + bh // 2 - size // 2, height - size))
            coordinates[(x, y)] = 1

        negative_coordinates = [
            (x, y)
            for y in tile_starts(height, size)
            for x in tile_starts(width, size)
            if not foreground[y:y + size, x:x + size].any()
        ]
        rng.shuffle(negative_coordinates)
        for position in negative_coordinates[:negatives_per_image]:
            coordinates[position] = 0

        image_id = hashlib.sha256(image_path.name.encode("utf-8")).hexdigest()[:16]
        for (x, y), label in sorted(coordinates.items()):
            patch_path = patch_dir / f"{image_id}_{x}_{y}.png"
            write_image(patch_path, patch_at(image, x, y, size))
            samples.append({
                "path": str(patch_path),
                "source": match.group(1).lower(),
                "label": label,
                "image": image_path.name,
                "x": x,
                "y": y,
            })

    (output_dir / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return samples


def split_samples(samples: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, bool] = {}
    for sample in samples:
        source = sample["source"]
        groups[source] = groups.get(source, False) or bool(sample["label"])
    rng = random.Random(seed)
    validation_groups = set()
    for positive in (False, True):
        names = [name for name, label in groups.items() if label is positive]
        rng.shuffle(names)
        if len(names) < 2:
            raise ValueError("正负原图组均至少需要两个，才能分组划分训练集与验证集。")
        validation_groups.update(names[:max(1, min(len(names) - 1, round(len(names) * fraction)))])
    train = [sample for sample in samples if sample["source"] not in validation_groups]
    validation = [sample for sample in samples if sample["source"] in validation_groups]
    for name, subset in (("训练集", train), ("验证集", validation)):
        if {sample["label"] for sample in subset} != {0, 1}:
            raise ValueError(f"{name}缺少正样本或负样本。")
    return train, validation


class PatchDataset(Dataset):
    def __init__(self, samples: list[dict], augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = read_image(Path(sample["path"]), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            image = np.rot90(image, random.randrange(4))
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image[None, :, :])
        return (tensor - 0.5) / 0.5, sample["label"]


def make_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            tp += int(((predictions == 1) & (labels == 1)).sum())
            fp += int(((predictions == 1) & (labels == 0)).sum())
            tn += int(((predictions == 0) & (labels == 0)).sum())
            fn += int(((predictions == 0) & (labels == 1)).sum())
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("classifier_runs/5um_binary"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--negatives-per-image", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.tile_size < 32
            or args.negatives_per_image < 1 or not 0 < args.validation_fraction < 1
            or args.learning_rate <= 0):
        parser.error("训练参数超出有效范围。")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = prepare_samples(
        args.data_root, args.output_dir, args.tile_size, args.negatives_per_image, args.seed
    )
    train_samples, validation_samples = split_samples(
        samples, args.validation_fraction, args.seed
    )
    counts = [sum(sample["label"] == label for sample in train_samples) for label in (0, 1)]
    weights = [1.0 / counts[sample["label"]] for sample in train_samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)
    train_loader = DataLoader(
        PatchDataset(train_samples, augment=True), batch_size=args.batch_size, sampler=sampler
    )
    validation_loader = DataLoader(
        PatchDataset(validation_samples, augment=False), batch_size=args.batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print(
        f"设备：{device}；训练：{len(train_samples)} 块；验证：{len(validation_samples)} 块；"
        f"训练负/正：{counts[0]}/{counts[1]}"
    )

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
        metrics = evaluate(model, validation_loader, device)
        print(
            f"第 {epoch}/{args.epochs} 轮：loss={loss_sum / len(train_samples):.4f}，"
            f"召回率={metrics['recall']:.4f}，特异度={metrics['specificity']:.4f}，"
            f"误检块={metrics['fp']}，漏检块={metrics['fn']}"
        )
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            torch.save({
                "model": model.state_dict(),
                "class_names": CLASS_NAMES,
                "tile_size": args.tile_size,
                "grayscale": True,
                "epoch": epoch,
                "metrics": metrics,
            }, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
