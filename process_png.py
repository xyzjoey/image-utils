from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np


class InputArgs:
    def __init__(self):
        self.args = self.__get_args()

    def __get_args(self):
        parser = ArgumentParser(description="file dialog will pop up to ask for input image(s)")
        parser.add_argument("-r", "--replace", action="store_true", default=False, help="overwrite result to input path. if not set, output path will be generated")
        parser.add_argument("steps", nargs="*", type=ImageUtils.Process, help=f"run steps in order. available steps: {[e.value for e in ImageUtils.Process]}")
        parser.add_argument("-g", "--gamma", default=2.2, type=float, help="default 2.2")
        parser.add_argument("-d", "--dtype", default=None, type=ImageUtils.DataType, help=f"output data type. available: {[e.value for e in ImageUtils.DataType]}")
        parser.add_argument("-c", "--compress", default=9, type=int, help="compression level. default 9 (smallest file size)")

        return parser.parse_args()

    def __ask_image_paths(self):
        print("asking for input image file(s)...")

        root = Tk()
        root.withdraw()

        return filedialog.askopenfilenames(filetypes=[("PNG", ".png")])

    def __generate_dst_path(self, path, ext=None):
        p = Path(path)
        prefix = path[:-len(p.suffix)]

        if ext is None:
            ext = p.suffix[1:]

        i = 0

        while True:
            new_path = f"{prefix}_processed.{ext}" if i == 0 else f"{prefix}_processed_{i}.{ext}"
            i += 1

            if not Path(new_path).exists():
                return new_path

    def get_image_paths(self):
        src_paths = self.__ask_image_paths()
        get_dst_path = lambda path: self.__generate_dst_path(path, "png") if not self.args.replace else path

        return [(src_path, get_dst_path(src_path)) for src_path in src_paths]


class Debug:
    @staticmethod
    def draw(img):
        print(type(img), img.dtype)
        print(img)

        cv2.imshow("debug", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def print_img_info(img):
        print(f"image info: shape={img.shape} dtype={img.dtype}")


class ImageUtils:
    class Process(Enum):
        REVERT_GAMMA = "revert_gamma"
        CORRECT_GAMMA = "correct_gamma"
        UNMULTIPLY_ALPHA = "unmultiply_alpha"
        CROP_TRANSPARENT = "crop_transparent"

    class DataType(Enum):
        UINT8 = "uint8"
        UINT16 = "uint16"

    @classmethod
    def process(cls, img, input_args):
        print("processing ...")

        output_dtype = img.dtype if input_args.args.dtype is None else getattr(np, input_args.args.dtype.value)

        img = cls.normalize(img)

        for step in input_args.args.steps:
            if step == cls.Process.REVERT_GAMMA:
                img = cls.revert_gamma(img, gamma=input_args.args.gamma)
            elif step == cls.Process.CORRECT_GAMMA:
                img = cls.correct_gamma(img, gamma=input_args.args.gamma)
            elif step == cls.Process.UNMULTIPLY_ALPHA:
                img = cls.unmultiply_alpha(img)
            elif step == cls.Process.CROP_TRANSPARENT:
                img = cls.crop_transparent(img)

        img = cls.denormalize(img, output_dtype)

        return img

    @staticmethod
    def normalize(img):
        max_value = np.iinfo(img.dtype).max
        return img / max_value

    @staticmethod
    def denormalize(img, dtype):
        max_value = np.iinfo(dtype).max
        return (img * max_value).astype(dtype)

    @staticmethod
    def clamp(img):
        new_img = np.copy(img)
        new_img[new_img < 0] = 0
        new_img[new_img > 1] = 1
        return new_img

    @staticmethod
    def revert_gamma(img, gamma):
        new_img = np.copy(img)
        new_img[:,:,0:3] = new_img[:,:,0:3] ** gamma
        return new_img

    @classmethod
    def correct_gamma(cls, img, gamma):
        return cls.revert_gamma(img, 1 / gamma)

    @classmethod
    def unmultiply_alpha(cls, img):
        # img = cls.denormalize(img, np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_mRGBA2RGBA)  # only support uint8
        # return cls.normalize(img)

        new_img = np.copy(img)
        y, x = img[:,:,3].nonzero()
        new_img[y,x,0:3] = new_img[y,x,0:3] / new_img[y,x,3,None]
        return cls.clamp(new_img)

    @staticmethod
    def crop_transparent(img):
        y, x = img[:,:,3].nonzero()
        return img[np.min(y):np.max(y), np.min(x):np.max(x)]


def main():
    input_args = InputArgs()
    path_pairs = input_args.get_image_paths()

    print("start processing...")

    for i, (src_path, dst_path) in enumerate(path_pairs):
        print(f"({i + 1} / {len(path_pairs)})")

        print(f"reading from {src_path}")
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)

        Debug.print_img_info(img)

        img = ImageUtils.process(img, input_args)

        # Debug.draw(img)

        print(f"writing to {dst_path}")
        cv2.imwrite(dst_path, img, [cv2.IMWRITE_PNG_COMPRESSION, input_args.args.compress])

    print("done!")


if __name__ == "__main__":
    main()
