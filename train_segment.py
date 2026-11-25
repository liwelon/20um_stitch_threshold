"""Train a 5um binary segmentation model from reviewed crop images and masks."""

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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet18


SOURCE_PATTERN = re.compile(r"^\d+__(?:blind|circuit)__(.+?\.(?:jpg|png|bmp))_", re.I)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def log(message: str) -> None:
    print(message, flush=True)


def read_image(path: Path, flags: int, expected_hash: str | None = None) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"文件与审核记录的 SHA256 不一致：{path}")
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"无法编码图像：{path}")
    data.tofile(str(path))


def tile_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def crop_patch(array: np.ndarray, x: int, y: int, size: int, is_mask: bool) -> np.ndarray:
    patch = array[y:y + size, x:x + size]
    bottom = size - patch.shape[0]
    right = size - patch.shape[1]
    if bottom or right:
        border = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
        patch = cv2.copyMakeBorder(patch, 0, bottom, 0, right, border, value=0)
    return patch


def collect_records(data_root: Path) -> list[dict]:
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    reviews_dir = data_root / "reviews"
    if not all(path.is_dir() for path in (images_dir, masks_dir, reviews_dir)):
        raise FileNotFoundError("数据目录必须包含 images、masks、reviews 子目录。")
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"没有找到裁剪图：{images_dir}")
    records = []
    for image_path in image_paths:
        mask_path = masks_dir / image_path.name
        review_path = reviews_dir / f"{image_path.stem}.json"
        if not mask_path.is_file() or not review_path.is_file():
            raise FileNotFoundError(f"缺少同名掩膜或审核文件：{image_path.name}")
        match = SOURCE_PATTERN.match(image_path.name)
        if match is None:
            raise ValueError(f"无法提取原图标识：{image_path.name}")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("state") != "approved" or review.get("wholeImageChecked") is not True:
            raise ValueError(f"图片尚未完整审核：{image_path.name}")
        if review.get("ignoredPixels") != 0:
            raise ValueError(f"掩膜含忽略区域：{image_path.name}")
        records.append({
            "image": str(image_path), "mask": str(mask_path),
            "source": match.group(1).lower(), "name": image_path.name,
            "positive": int(review["defectPixels"]) > 0, "review": review,
        })
    return records


def split_records(records: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, bool] = {}
    for record in records:
        source = record["source"]
        groups[source] = groups.get(source, False) or record["positive"]
    rng = random.Random(seed)
    validation_groups = set()
    for positive in (False, True):
        names = [name for name, label in groups.items() if label is positive]
        rng.shuffle(names)
        if len(names) < 2:
            raise ValueError("正负原图组均至少需要两个。")
        count = max(1, min(len(names) - 1, round(len(names) * fraction)))
        validation_groups.update(names[:count])
    train = [record for record in records if record["source"] not in validation_groups]
    validation = [record for record in records if record["source"] in validation_groups]
    for name, subset in (("训练集", train), ("验证集", validation)):
        if {record["positive"] for record in subset} != {False, True}:
            raise ValueError(f"{name}缺少正样本或负样本。")
    return train, validation


def read_record(record: dict) -> tuple[np.ndarray, np.ndarray]:
    review = record["review"]
    image = read_image(Path(record["image"]), cv2.IMREAD_COLOR, review.get("imageSha256"))
    mask = read_image(Path(record["mask"]), cv2.IMREAD_UNCHANGED, review.get("maskSha256"))
    if mask.ndim != 2 or not np.isin(np.unique(mask), (0, 1)).all():
        raise ValueError(f"掩膜必须为单通道 0/1：{record['mask']}")
    if image.shape[:2] != mask.shape or int(mask.sum()) != review["defectPixels"]:
        raise ValueError(f"图片、掩膜和审核记录不一致：{record['name']}")
    return image, mask


def prepare_patches(
    records: list[dict], output_dir: Path, size: int, negatives_per_image: int, seed: int
) -> list[dict]:
    image_dir = output_dir / "patches" / "images"
    mask_dir = output_dir / "patches" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    samples = []
    log(f"准备训练小块：0/{len(records)} 张")
    for index, record in enumerate(records, 1):
        image, mask = read_record(record)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = mask.shape
        coordinates: dict[tuple[int, int], bool] = {}
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            x = max(0, min(bx + bw // 2 - size // 2, width - size))
            y = max(0, min(by + bh // 2 - size // 2, height - size))
            coordinates[(x, y)] = True
        negative_positions = [
            (x, y)
            for y in tile_starts(height, size, size)
            for x in tile_starts(width, size, size)
            if not mask[y:y + size, x:x + size].any()
        ]
        rng.shuffle(negative_positions)
        for position in negative_positions[:negatives_per_image]:
            coordinates[position] = False

        image_id = hashlib.sha256(record["name"].encode("utf-8")).hexdigest()[:16]
        for (x, y), positive in sorted(coordinates.items()):
            filename = f"{image_id}_{x}_{y}.png"
            patch_mask = crop_patch(mask, x, y, size, is_mask=True)
            if bool(patch_mask.any()) != positive:
                raise ValueError(f"小块标签与掩膜不一致：{record['name']} {x},{y}")
            write_png(image_dir / filename, crop_patch(gray, x, y, size, is_mask=False))
            write_png(mask_dir / filename, patch_mask)
            samples.append({
                "image": str(image_dir / filename), "mask": str(mask_dir / filename),
                "positive": positive, "foreground_pixels": int(patch_mask.sum()),
                "source": record["source"], "name": record["name"], "x": x, "y": y,
            })
        if index % 10 == 0 or index == len(records):
            log(f"准备训练小块：{index}/{len(records)} 张，已生成 {len(samples)} 块")
    if {sample["positive"] for sample in samples} != {False, True}:
        raise ValueError("训练小块缺少正样本或负样本。")
    return samples


class PatchDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = read_image(Path(sample["image"]), cv2.IMREAD_GRAYSCALE)
        mask = read_image(Path(sample["mask"]), cv2.IMREAD_GRAYSCALE)
        turns = random.randrange(4)
        image = np.rot90(image, turns)
        mask = np.rot90(mask, turns)
        rgb = np.repeat(image[:, :, None], 3, axis=2).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1))
        target = np.ascontiguousarray(mask[None, :, :], dtype=np.float32)
        return torch.from_numpy(tensor), torch.from_numpy(target)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((image, skip), dim=1))


class ResNet18UNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.decode3 = DecoderBlock(512 + 256, 256)
        self.decode2 = DecoderBlock(256 + 128, 128)
        self.decode1 = DecoderBlock(128 + 64, 64)
        self.decode0 = DecoderBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, 1, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward_logits(self, image: torch.Tensor) -> torch.Tensor:
        rgb = image * self.std + self.mean
        gray = (rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114)
        gray = (gray - 0.5) / 0.5
        stem = self.stem(gray)
        x1 = self.layer1(self.pool(stem))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        decoded = self.decode3(x4, x3)
        decoded = self.decode2(decoded, x2)
        decoded = self.decode1(decoded, x1)
        decoded = self.decode0(decoded, stem)
        decoded = F.interpolate(decoded, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(decoded)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(image))


def predict_roi(
    model: nn.Module, image: np.ndarray, device: torch.device, size: int, stride: int,
    batch_size: int,
) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rotated = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    height, width = rotated.shape[:2]
    padded = cv2.copyMakeBorder(
        rotated, 0, max(0, size - height), 0, max(0, size - width), cv2.BORDER_REFLECT_101
    )
    padded_height, padded_width = padded.shape[:2]
    positions = [
        (x, y)
        for y in tile_starts(padded_height, size, stride)
        for x in tile_starts(padded_width, size, stride)
    ]
    axis = np.maximum(np.hanning(size), 0.05).astype(np.float32)
    window = axis[:, None] * axis[None, :]
    accumulated = np.zeros((padded_height, padded_width), dtype=np.float32)
    weights = np.zeros_like(accumulated)
    with torch.no_grad():
        for offset in range(0, len(positions), batch_size):
            batch = positions[offset:offset + batch_size]
            tiles = np.stack([padded[y:y + size, x:x + size] for x, y in batch])
            normalized = ((tiles.astype(np.float32) / 255.0 - MEAN) / STD)
            tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))).to(device)
            probabilities = model(tensor).cpu().numpy()[:, 0]
            for (x, y), probability in zip(batch, probabilities):
                accumulated[y:y + size, x:x + size] += probability * window
                weights[y:y + size, x:x + size] += window
    probability = (accumulated / weights)[:height, :width]
    return cv2.rotate(probability, cv2.ROTATE_90_COUNTERCLOCKWISE)


def evaluate(
    model: nn.Module, records: list[dict], device: torch.device, size: int,
    stride: int, batch_size: int, threshold: float,
) -> dict:
    model.eval()
    positive_dice = []
    positive_found = 0
    negative_count = 0
    negative_false_alarms = 0
    negative_predicted_areas = []
    negative_components = 0
    log(f"验证整张裁剪图：0/{len(records)} 张")
    for index, record in enumerate(records, 1):
        image, mask = read_record(record)
        probability = predict_roi(model, image, device, size, stride, batch_size)
        predicted = probability >= threshold
        actual = mask != 0
        if actual.any():
            intersection = int(np.logical_and(predicted, actual).sum())
            positive_dice.append(2 * intersection / max(1, int(predicted.sum()) + int(actual.sum())))
            positive_found += intersection > 0
        else:
            negative_count += 1
            negative_predicted_areas.append(float(predicted.mean()))
            _, _, stats, _ = cv2.connectedComponentsWithStats(
                predicted.astype(np.uint8), connectivity=8
            )
            component_count = int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= 30))
            negative_components += component_count
            negative_false_alarms += component_count > 0
        if index % 10 == 0 or index == len(records):
            log(f"验证整张裁剪图：{index}/{len(records)} 张")
    dice = float(np.mean(positive_dice))
    negative_predicted_area = float(np.mean(negative_predicted_areas))
    score = dice * (1 - negative_predicted_area)
    return {
        "positive_dice": dice,
        "positive_found": positive_found,
        "positive_images": len(positive_dice),
        "negative_false_alarms": negative_false_alarms,
        "negative_images": negative_count,
        "negative_predicted_area": negative_predicted_area,
        "negative_components": negative_components,
        "score": score,
    }


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("segment_runs/5um_1111_v2"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--negatives-per-image", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-interval", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.tile_size < 32
            or args.stride < 1 or args.stride > args.tile_size
            or args.negatives_per_image < 1 or args.validation_interval < 1
            or not 0 < args.validation_fraction < 1 or not 0 < args.threshold < 1
            or args.learning_rate <= 0):
        parser.error("训练参数超出有效范围。")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备：{device}")
    records = collect_records(args.data_root)
    train_records, validation_records = split_records(records, args.validation_fraction, args.seed)
    log(
        f"数据：共 {len(records)} 张，训练 {len(train_records)} 张，验证 {len(validation_records)} 张；"
        f"正样本 {sum(r['positive'] for r in records)} 张"
    )
    samples = prepare_patches(
        train_records, args.output_dir, args.tile_size, args.negatives_per_image, args.seed
    )
    (args.output_dir / "split.json").write_text(json.dumps({
        "train": [record["name"] for record in train_records],
        "validation": [record["name"] for record in validation_records],
        "patches": samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = [sum(sample["positive"] == label for sample in samples) for label in (False, True)]
    weights = [1.0 / counts[int(sample["positive"])] for sample in samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
    loader = DataLoader(PatchDataset(samples), batch_size=args.batch_size, sampler=sampler)
    model = ResNet18UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    log(
        f"设备：{device}；小块负/正：{counts[0]}/{counts[1]}；"
        f"每轮 {len(loader)} 步"
    )

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, masks) in enumerate(loader, 1):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model.forward_logits(images)
                probability = torch.sigmoid(logits)
                pixel_loss = bce(logits, masks)
                intersection = (probability * masks).sum(dim=(1, 2, 3))
                dice_loss = 1 - ((2 * intersection + 1) / (
                    probability.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1
                )).mean()
                loss = pixel_loss + dice_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            if step % max(1, len(loader) // 10) == 0 or step == len(loader):
                log(
                    f"第 {epoch}/{args.epochs} 轮，"
                    f"步骤 {step}/{len(loader)}，loss={running_loss / step:.4f}"
                )
        scheduler.step()
        metrics = None
        if epoch == 1 or epoch % args.validation_interval == 0 or epoch == args.epochs:
            metrics = evaluate(
                model, validation_records, device, args.tile_size, args.stride,
                args.batch_size, args.threshold,
            )
            log(
                f"验证：正图 Dice={metrics['positive_dice']:.4f}，"
                f"检出 {metrics['positive_found']}/{metrics['positive_images']}，"
                f"负图预测面积占比 {metrics['negative_predicted_area']:.4%}，"
                f"负图面积≥30 的连通域 {metrics['negative_components']} 个，"
                f"误检 {metrics['negative_false_alarms']}/{metrics['negative_images']}"
            )
            if metrics["score"] > best_score:
                best_score = metrics["score"]
                save_checkpoint(args.output_dir / "best.pt", {
                    "model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                    "tile_size": args.tile_size, "threshold": args.threshold,
                    "architecture": "ResNet18UNet_gray_from_rgb",
                })
                log(f"保存最佳模型：第 {epoch} 轮，得分 {best_score:.4f}")
        save_checkpoint(args.output_dir / "last.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "metrics": metrics, "best_score": best_score,
            "tile_size": args.tile_size, "threshold": args.threshold,
            "architecture": "ResNet18UNet_gray_from_rgb",
        })
    log(f"训练完成，最佳模型：{args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
