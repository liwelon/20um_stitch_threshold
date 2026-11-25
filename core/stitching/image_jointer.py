from __future__ import annotations

import numpy as np

from core.stitching.params import JointParams


class ImageJointer:
    """Grid stitch with known mechanical offsets (ported from cpp/ImageJointer)."""

    def __init__(self, params: JointParams) -> None:
        self.h_dx = params.h_dx
        self.h_dy = params.h_dy
        self.v_dx = params.v_dx
        self.v_dy = params.v_dy
        self.rows = params.rows
        self.cols = params.cols
        self.layers = params.layers
        self.single_w = params.single_w
        self.single_h = params.single_h
        self.channel = params.channel
        self.joint_count = params.joint_count

        self.id_array = self._calc_id_array()
        self._calc_jointed_size()

    def _calc_id_array(self) -> list[list[list[int]]]:
        if self.layers == 1:
            single_layer = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            for row in range(self.rows):
                if row % 2 == 0:
                    start = row * self.cols
                    for col in range(self.cols):
                        single_layer[row][col] = start + col
                else:
                    start = (row + 1) * self.cols - 1
                    for col in range(self.cols):
                        single_layer[row][col] = start - col
            return [single_layer]

        up_ids: list[int] = []
        down_ids: list[int] = []
        image_id = 0
        for index in range(self.rows * self.cols):
            if index % 2 == 0:
                up_ids.append(image_id)
                image_id += 1
                down_ids.append(image_id)
                image_id += 1
            else:
                down_ids.append(image_id)
                image_id += 1
                up_ids.append(image_id)
                image_id += 1

        id_array = [
            [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            for _ in range(2)
        ]
        for row in range(self.rows):
            if row % 2 == 0:
                start = row * self.cols
                for col in range(self.cols):
                    id_array[0][row][col] = up_ids[start]
                    id_array[1][row][col] = down_ids[start]
                    start += 1
            else:
                start = (row + 1) * self.cols - 1
                for col in range(self.cols):
                    id_array[0][row][col] = up_ids[start]
                    id_array[1][row][col] = down_ids[start]
                    start -= 1
        return id_array

    @staticmethod
    def _gen_horizontal_mask(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        if width <= 0 or height <= 0:
            raise ValueError("invalid horizontal overlap mask size")
        xs = np.linspace(1.0, 0.0, width, dtype=np.float32)
        mask1 = np.broadcast_to(xs.reshape(1, width), (height, width))
        mask2 = 1.0 - mask1
        return mask1, mask2

    @staticmethod
    def _gen_vertical_mask(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        if width <= 0 or height <= 0:
            raise ValueError("invalid vertical overlap mask size")
        ys = np.linspace(1.0, 0.0, height, dtype=np.float32)
        mask1 = np.broadcast_to(ys.reshape(height, 1), (height, width))
        mask2 = 1.0 - mask1
        return mask1, mask2

    def _calc_jointed_size(self) -> None:
        self.h_overlap = self.single_w - abs(self.h_dx)
        self.h_jointed_w = self.single_w * self.cols - (self.cols - 1) * self.h_overlap
        self.h_jointed_h = self.single_h - (self.cols - 1) * abs(self.h_dy)
        if self.cols > 1:
            self.h_overlap_mask_1, self.h_overlap_mask_2 = self._gen_horizontal_mask(
                self.h_overlap,
                self.h_jointed_h,
            )

        self.v_overlap = self.single_h - self.v_dy - (self.cols - 1) * abs(self.h_dy)
        self.v_jointed_h = self.h_jointed_h * self.rows - (self.rows - 1) * self.v_overlap
        self.v_jointed_w = self.h_jointed_w - (self.rows - 1) * abs(self.v_dx)
        if self.rows > 1:
            self.v_overlap_mask_1, self.v_overlap_mask_2 = self._gen_vertical_mask(
                self.v_jointed_w,
                self.v_overlap,
            )

    def horizontal_joint(self, images: list[np.ndarray]) -> np.ndarray:
        if len(images) != self.cols:
            raise ValueError(f"horizontal joint needs {self.cols} images, got {len(images)}")
        if self.cols == 1:
            return images[0].copy()

        end_id = self.cols - 1
        if self.h_dy > 0:
            clipped = [
                images[i][(end_id - i) * self.h_dy : self.single_h - i * self.h_dy, :, :]
                for i in range(self.cols)
            ]
        elif self.h_dy < 0:
            clipped = [
                images[i][-i * self.h_dy : self.single_h + (end_id - i) * self.h_dy, :, :]
                for i in range(self.cols)
            ]
        else:
            clipped = images

        jointed = np.zeros((self.h_jointed_h, self.h_jointed_w, self.channel), dtype=np.uint8)
        for i in range(self.cols - 1):
            img1 = clipped[i]
            img2 = clipped[i + 1]
            overlap1 = img1[:, -self.h_overlap :, :]
            overlap2 = img2[:, : self.h_overlap, :]
            overlap = (
                self.h_overlap_mask_1[:, :, None] * overlap1.astype(np.float32)
                + self.h_overlap_mask_2[:, :, None] * overlap2.astype(np.float32)
            ).astype(np.uint8)

            if i == 0:
                jointed[:, : self.h_dx, :] = img1[:, : self.h_dx, :]
                jointed[:, self.h_dx : self.single_w, :] = overlap
            else:
                x0 = self.single_w + (i - 1) * self.h_dx
                x1 = x0 + self.h_dx - self.h_overlap
                x2 = x0 + self.h_dx
                jointed[:, x0:x1, :] = img1[:, self.h_overlap : -self.h_overlap, :]
                jointed[:, x1:x2, :] = overlap

            if i == self.cols - 2:
                x0 = (i + 1) * self.h_dx + self.h_overlap
                jointed[:, x0 : self.h_jointed_w, :] = img2[:, self.h_overlap :, :]
        return jointed

    def vertical_joint(self, images: list[np.ndarray]) -> np.ndarray:
        if len(images) != self.rows:
            raise ValueError(f"vertical joint needs {self.rows} images, got {len(images)}")
        if self.rows == 1:
            return images[0].copy()

        end_id = self.rows - 1
        if self.v_dx > 0:
            clipped = [
                images[i][:, (end_id - i) * self.v_dx : self.h_jointed_w - i * self.v_dx, :]
                for i in range(self.rows)
            ]
        elif self.v_dx < 0:
            clipped = [
                images[i][
                    :,
                    -i * self.v_dx : self.h_jointed_w + (end_id - i) * self.v_dx,
                    :,
                ]
                for i in range(self.rows)
            ]
        else:
            clipped = images

        jointed = np.zeros((self.v_jointed_h, self.v_jointed_w, self.channel), dtype=np.uint8)
        for i in range(self.rows - 1):
            img1 = clipped[i]
            img2 = clipped[i + 1]
            overlap1 = img1[-self.v_overlap :, :, :]
            overlap2 = img2[: self.v_overlap, :, :]
            overlap = (
                self.v_overlap_mask_1[:, :, None] * overlap1.astype(np.float32)
                + self.v_overlap_mask_2[:, :, None] * overlap2.astype(np.float32)
            ).astype(np.uint8)

            if i == 0:
                jointed[: self.v_dy, :, :] = img1[: self.v_dy, :, :]
                jointed[self.v_dy : self.h_jointed_h, :, :] = overlap
            else:
                y0 = self.h_jointed_h + (i - 1) * self.v_dy
                y1 = y0 + self.v_dy - self.v_overlap
                y2 = y0 + self.v_dy
                jointed[y0:y1, :, :] = img1[self.v_overlap : -self.v_overlap, :, :]
                jointed[y1:y2, :, :] = overlap

            if i == self.rows - 2:
                y0 = (i + 1) * self.v_dy + self.v_overlap
                jointed[y0 : self.v_jointed_h, :, :] = img2[self.v_overlap :, :, :]
        return jointed

    def _joint_layer(self, images: list[np.ndarray], layer: int) -> np.ndarray:
        line_jointed: list[np.ndarray] = []
        for row in range(self.rows):
            line_imgs = [images[self.id_array[layer][row][col]] for col in range(self.cols)]
            line_jointed.append(self.horizontal_joint(line_imgs))
        return self.vertical_joint(line_jointed)

    def joint_up_down(self, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self.layers != 2:
            raise ValueError(f"joint_up_down needs 2 layers, got {self.layers}")
        self._validate_images(images)
        up_img = self._joint_layer(images, 0)
        down_img = self._joint_layer(images, 1)
        return up_img, down_img

    def joint_single_layer(self, images: list[np.ndarray]) -> np.ndarray:
        if self.layers != 1:
            raise ValueError(f"joint_single_layer needs 1 layer, got {self.layers}")
        self._validate_images(images)
        return self._joint_layer(images, 0)

    def _validate_images(self, images: list[np.ndarray]) -> None:
        if len(images) != self.joint_count:
            raise ValueError(f"need {self.joint_count} images, got {len(images)}")
        for index, image in enumerate(images):
            if image is None or image.size == 0:
                raise ValueError(f"image {index} is empty")
            height, width = image.shape[:2]
            if width != self.single_w or height != self.single_h:
                raise ValueError(
                    f"image {index} size is {width}x{height}, "
                    f"expected {self.single_w}x{self.single_h}"
                )
            if image.ndim == 2:
                images[index] = np.repeat(image[:, :, None], 3, axis=2)
            elif image.shape[2] == 4:
                images[index] = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(f"unsupported channel count for image {index}: {image.shape}")
