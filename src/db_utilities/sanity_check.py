import os
import argparse
import multiprocessing
import imghdr
import cv2
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc", type=str, nargs="+", required=True)

    args = parser.parse_args()

    for loc in args.data_loc:
        if not os.path.exists(loc):
            parser.error(f"Invalid --data_loc: {loc}")

    return args


def check_pil(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except:
        return False


def check_opencv(file_path):
    try:
        img = cv2.imread(file_path)
        return img is not None
    except:
        return False


def check_imghdr(file_path):
    image_type = imghdr.what(file_path)
    return image_type is not None


def check_validity(file_path):
    if file_path.endswith((".png", ".jpg", ".jpeg")):
        if not (
            (
                check_pil(file_path) and
                check_opencv(file_path) and
                check_imghdr(file_path)
            )
        ):
            print(f"{file_path}")


def main():
    args = parse_arguments()

    files = []
    for loc in args.data_loc:
        files.extend([os.path.join(loc, filename) for filename in os.listdir(loc)])
        print(f"Checking files in {loc}")
        print(f"{len(files)} files")

    pool = multiprocessing.Pool()
    pool.map(check_validity, files)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
