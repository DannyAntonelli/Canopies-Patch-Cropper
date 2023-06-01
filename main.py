import argparse
import cv2

from math import sqrt
import random
import os

import copy

import numpy as np


WINDOW_NAME = "Canopies Patch Cropper"


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


def crop_patches(
    images_path: str,
    output_path: str,
    patch_size: int,
    num_patches: int,
) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for image_path in os.listdir(images_path):
        image = cv2.imread(os.path.join(images_path, image_path))

        rows, cols, _ = image.shape

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
        initial_image = copy.deepcopy(image)
        draw_patches(image, patches, patch_size)

        cv2.imshow(WINDOW_NAME, image)
        cv2.resizeWindow(WINDOW_NAME, int(cols / 2.5), int(rows / 2.5))
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

        if key == ord("s"):
            save_patches(
                initial_image,
                patches,
                patch_size,
                os.path.join(output_path, image_path[:-4]),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_path", type=str, default=os.path.join("data", "images\\")
    )
    parser.add_argument(
        "--output_path", type=str, default=os.path.join("data", "patches\\")
    )
    parser.add_argument("--patch_size", type=int, default=300)
    parser.add_argument("--num_patches", type=int, default=5)
    args = parser.parse_args()

    crop_patches(
        images_path=args.images_path,
        output_path=args.output_path,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
    )


if __name__ == "__main__":
    main()
