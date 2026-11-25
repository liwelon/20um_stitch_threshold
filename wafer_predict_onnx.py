"""ONNX deployment for the blind/circuit joint UNet.

Dependencies: numpy, opencv-python.
Input: already cropped ROI at its original resolution, uint8 BGR or grayscale.
One predictor instance is for serial use; reuse it across images.
The ONNX output must be sigmoid probabilities named probability.
"""
from pathlib import Path
import json
import time

import cv2
import numpy as np

MODEL_PATH = Path(r'D:\WaferStudio\model\wafer_joint_20261022.onnx')
IMAGE_FOLDER = Path(r'E:\data\5umdb\mangyuan')
BATCH_SIZE = 8
THRESHOLD = 0.35
MIN_AREA = 30
SAVE_PROBABILITY = False


def read_image(path):
    """Support Windows paths containing Chinese characters."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def tile_starts(length, patch=512, stride=384):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


class WaferPredictor:
    """Runs the ONNX model with OpenCV DNN."""
    def __init__(self, onnx_path, batch_size=8, threshold=0.35, min_area=30):
        if not 0 < threshold < 1 or min_area < 1:
            raise ValueError('Require 0 < threshold < 1 and min_area >= 1.')
        if batch_size < 1:
            raise ValueError('Require batch >= 1.')
        onnx_path = Path(onnx_path)
        if onnx_path.suffix.lower() != '.onnx' or not onnx_path.is_file():
            raise FileNotFoundError(f'ONNX model not found: {onnx_path}')
        self.batch_size, self.threshold, self.min_area = batch_size, threshold, min_area
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        self.mean = np.array([.485, .456, .406], dtype=np.float32)
        self.std = np.array([.229, .224, .225], dtype=np.float32)
        axis = np.maximum(np.hanning(512), .05).astype(np.float32)
        self.window = axis[:, None] * axis[None, :]

    def _infer_tiles(self, rgb_tiles):
        data = np.stack(rgb_tiles).astype(np.float32) / np.float32(255)
        data = np.ascontiguousarray(((data - self.mean) / self.std).transpose(0, 3, 1, 2))
        probabilities = []
        for tile in data:
            self.net.setInput(tile[np.newaxis], 'image')
            probability = np.asarray(self.net.forward('probability'))
            if probability.shape != (1, 1, 512, 512):
                raise ValueError(f'Unexpected ONNX output shape: {probability.shape}')
            if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
                raise ValueError('ONNX output must contain finite sigmoid probabilities in [0, 1].')
            probabilities.append(probability[0, 0])
        return np.stack(probabilities)

    def predict(self, image):
        """Return float32 probability, uint8 mask (0/255), and ROI-local defects."""
        if image is None or image.size == 0 or image.dtype != np.uint8:
            raise ValueError('Expected a nonempty uint8 image.')
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError('Expected grayscale HxW or BGR HxWx3; convert other formats explicitly.')
        start = time.perf_counter()
        h, w = rgb.shape[:2]
        padded = cv2.copyMakeBorder(rgb, 0, max(0, 512-h), 0, max(0, 512-w), cv2.BORDER_REFLECT_101)
        ph, pw = padded.shape[:2]
        positions = [(x, y) for y in tile_starts(ph) for x in tile_starts(pw)]
        accumulated = np.zeros((ph, pw), dtype=np.float32)
        weights = np.zeros((ph, pw), dtype=np.float32)
        for offset in range(0, len(positions), self.batch_size):
            batch = positions[offset:offset+self.batch_size]
            probabilities = self._infer_tiles([padded[y:y+512, x:x+512] for x, y in batch])
            for (x, y), probability in zip(batch, probabilities):
                accumulated[y:y+512, x:x+512] += probability * self.window
                weights[y:y+512, x:x+512] += self.window
        probability = (accumulated / weights)[:h, :w].copy()
        if not np.isfinite(probability).all():
            raise RuntimeError('ONNX model returned non-finite probabilities.')
        # Filter once, after merging probabilities and cropping back to the ROI.
        binary = (probability >= self.threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= self.min_area
        keep[0] = False
        mask = keep[labels].astype(np.uint8) * 255
        defects = []
        for label in range(1, count):
            if not keep[label]:
                continue
            x, y, width, height, area = map(int, stats[label])
            local = labels[y:y+height, x:x+width] == label
            values = probability[y:y+height, x:x+width][local]
            defects.append(dict(id=len(defects)+1, bbox=[x, y, width, height], area=area,
                                mean_probability=float(values.mean()), max_probability=float(values.max())))
        return dict(probability=probability, mask=mask, defects=defects, tiles=len(positions),
                    elapsed_ms=(time.perf_counter()-start)*1000)


def main():
    if not IMAGE_FOLDER.is_dir():
        raise FileNotFoundError(f'Image folder not found: {IMAGE_FOLDER}')
    image_paths = sorted(
        path for path in IMAGE_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        and not path.stem.endswith(('_mask', '_overlay'))
    )
    if not image_paths:
        raise ValueError(f'No input images in: {IMAGE_FOLDER}')
    cv2.setNumThreads(2)
    predictor = WaferPredictor(MODEL_PATH, BATCH_SIZE, THRESHOLD, MIN_AREA)
    for index, image_path in enumerate(image_paths, start=1):
        original = read_image(image_path)
        result = predictor.predict(original)
        stem = image_path.stem

        write_png(IMAGE_FOLDER / f'{stem}_mask.png', result['mask'])
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if original.ndim == 2 else original.copy()
        contours, _ = cv2.findContours(result['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
        for defect in result['defects']:
            x, y, width, height = defect['bbox']
            cv2.rectangle(overlay, (x, y), (x+width-1, y+height-1), (0, 255, 255), 1)
        write_png(IMAGE_FOLDER / f'{stem}_overlay.png', overlay)
        if SAVE_PROBABILITY:
            np.save(IMAGE_FOLDER / f'{stem}_probability.npy', result['probability'])
        summary = dict(image=str(image_path.resolve()), onnx=str(MODEL_PATH.resolve()),
                       image_size=[original.shape[1], original.shape[0]], bbox_format='x,y,width,height in input ROI pixels',
                       threshold=THRESHOLD, min_area=MIN_AREA, batch=BATCH_SIZE,
                       count=len(result['defects']), defects=result['defects'], tiles=result['tiles'], elapsed_ms=result['elapsed_ms'])
        (IMAGE_FOLDER / f'{stem}_result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{index}/{len(image_paths)} {image_path.name}: {len(result["defects"])} defects', flush=True)


if __name__ == '__main__':
    main()
