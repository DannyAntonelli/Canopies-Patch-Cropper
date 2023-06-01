import argparse
import cv2

from math import sqrt
import random
import os

import copy

import numpy as np


WINDOW_NAME = "Canopies Patch Cropper"

REGENERATE_ALL_KEY = ord(" ")
REGENERATE_PARTIAL_KEY = ord("r")
SAVE_KEY = ord("s")
QUIT_KEY = ord("q")


class MoveState:
    def __init__(
        self, current_position: tuple[int, int] = (-1, -1), patch_moving: int = -1
    ) -> None:
        self.current_position = current_position
        self.patch_moving = patch_moving


def generate_patches(
    patch_size: int,
    num_patches: int,
    min_x: int,
    max_x: int,
    x_mean: float,
    x_std: float,
    min_y: int,
    max_y: int,
    y_mean: float,
    y_std: float,
) -> list[tuple[int, int]]:
    return [
        (
            max(
                min_x,
                min(
                    int(random.gauss(x_mean, x_std) - patch_size / 2),
                    max_x,
                ),
            ),
            max(
                min_y,
                min(
                    int(random.gauss(y_mean, y_std) - patch_size / 2),
                    max_y,
                ),
            ),
        )
        for _ in range(num_patches)
    ]


def draw_patches(
    image: np.ndarray[int, np.dtype[np.generic]],
    patches: list[tuple[int, int]],
    patch_size: int,
) -> None:
    for patch in patches:
        cv2.rectangle(
            image,
            (patch[0], patch[1]),
            (patch[0] + patch_size, patch[1] + patch_size),
            (0, 255, 0),
            3,
        )


def save_patches(
    image: np.ndarray[int, np.dtype[np.generic]],
    patches: list[tuple[int, int]],
    patch_size: int,
    output_path: str,
) -> None:
    for patch in patches:
        cv2.imwrite(
            output_path
            + f"_{patch[0]}_{patch[1]}_{patch[0] + patch_size}_{patch[1] + patch_size}.jpg",
            image[patch[1] : patch[1] + patch_size, patch[0] : patch[0] + patch_size],
        )


def find_patch(patches: list[tuple[int, int]], patch_size: int, x: int, y: int) -> int:
    for i, patch in enumerate(patches):
        if (
            patch[0] <= x <= patch[0] + patch_size
            and patch[1] <= y <= patch[1] + patch_size
        ):
            return i
    return -1


def move_patches(
    event: int,
    x: int,
    y: int,
    _flags: int,
    _param: object,
    image: np.ndarray[int, np.dtype[np.generic]],
    patches: list[tuple[int, int]],
    patch_size: int,
    move_state: MoveState,
) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        move_state.patch_moving = find_patch(patches, patch_size, x, y)
        if move_state.patch_moving != -1:
            move_state.current_position = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        idx = find_patch(patches, patch_size, x, y)
        if idx != -1:
            patches.pop(idx)
    elif event == cv2.EVENT_LBUTTONUP:
        move_state.current_position = (-1, -1)

    if move_state.current_position == (-1, -1):
        return

    delta_pos = (x - move_state.current_position[0], y - move_state.current_position[1])
    patches[move_state.patch_moving] = (
        max(
            0,
            min(
                patches[move_state.patch_moving][0] + delta_pos[0],
                image.shape[1] - patch_size,
            ),
        ),
        max(
            0,
            min(
                patches[move_state.patch_moving][1] + delta_pos[1],
                image.shape[0] - patch_size,
            ),
        ),
    )
    move_state.current_position = (x, y)


def crop_patches(
    images_path: str,
    output_path: str,
    patch_size: int,
    num_patches: int,
) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for image_path in os.listdir(images_path):
        initial_image = cv2.imread(os.path.join(images_path, image_path))
        patches = []
        rows, cols, _ = initial_image.shape
        move_state = MoveState()

        key = REGENERATE_ALL_KEY
        while key == REGENERATE_ALL_KEY or key == REGENERATE_PARTIAL_KEY or key == -1:
            image = copy.deepcopy(initial_image)

            if key == REGENERATE_ALL_KEY:
                patches = generate_patches(
                    patch_size=patch_size,
                    num_patches=num_patches,
                    min_x=0,
                    max_x=cols - patch_size,
                    x_mean=cols / 2,
                    x_std=cols / 6,
                    min_y=0,
                    max_y=rows - patch_size,
                    y_mean=rows / 3,
                    y_std=rows / 4,
                )
            elif key == REGENERATE_PARTIAL_KEY:
                patches.extend(
                    generate_patches(
                        patch_size=patch_size,
                        num_patches=num_patches - len(patches),
                        min_x=0,
                        max_x=cols - patch_size,
                        x_mean=cols / 2,
                        x_std=cols / 6,
                        min_y=0,
                        max_y=rows - patch_size,
                        y_mean=rows / 3,
                        y_std=rows / 4,
                    )
                )
            draw_patches(image, patches, patch_size)

            cv2.imshow(WINDOW_NAME, image)
            cv2.resizeWindow(WINDOW_NAME, int(cols / 2.5), int(rows / 2.5))

            cv2.setMouseCallback(
                WINDOW_NAME,
                lambda event, x, y, flags, param, image=image, patches=patches: move_patches(
                    event, x, y, flags, param, image, patches, patch_size, move_state
                ),
            )

            key = cv2.waitKey(10)

        if key == QUIT_KEY:
            break

        if key == SAVE_KEY:
            save_patches(
                initial_image,
                patches,
                patch_size,
                os.path.join(output_path, image_path[:-4]),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-path", type=str, default=os.path.join("data", "images")
    )
    parser.add_argument(
        "--output-path", type=str, default=os.path.join("data", "patches")
    )
    parser.add_argument("--patch-size", type=int, default=300)
    parser.add_argument("--num-patches", type=int, default=5)
    args = parser.parse_args()

    crop_patches(
        images_path=args.images_path,
        output_path=args.output_path,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
    )


if __name__ == "__main__":
    main()
