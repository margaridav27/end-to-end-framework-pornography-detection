import os
import argparse
from typing import List
import cv2
import numpy as np

def _parse_arguments():
    parser = argparse.ArgumentParser(description="Training a pytorch model to classify pornographic content")
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--info_loc", type=str, required=True, help="Location of the directory containing the bounding boxes or the mask files.")
    parser.add_argument("--save_loc", type=str, required=True)
    
    args = parser.parse_args()

    if not os.path.exists(args.data_loc): parser.error("Invalid --data_loc.")
    if not os.path.exists(args.info_loc): parser.error("Invalid --info_loc.")

    os.makedirs(args.save_loc, exist_ok=True)

    return args


def _blur_area_by_bounding_box(image: np.ndarray, bounding_box_coordinates: List[int]) -> np.ndarray:
    image_height, image_width, _ = image.shape

    # Convert bounding box coordinates to points
    points = np.array(bounding_box_coordinates).reshape(-1, 2)

    # Get the minimum area rectangle that encloses the points
    rect = cv2.minAreaRect(points)

    # Get the box points of the rotated rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Create a mask for the rotated rectangle
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [box], (255))

    # Extract the region to blur from the image using the rotated rectangle mask
    x, y, w, h = cv2.boundingRect(box)
    region_to_blur = image[y:y+h, x:x+w]

    # Apply blurring only if the region to blur is not empty
    if region_to_blur.size != 0:
        blurred = cv2.GaussianBlur(region_to_blur, (51, 51), 0)
        image[y:y+h, x:x+w] = np.where(mask[y:y+h, x:x+w, None], blurred, image[y:y+h, x:x+w])

    return image


def _blur_area_by_mask(image : np.ndarray, mask : np.ndarray):
  if mask.shape != image.shape[:2]:
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
  
  mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]
  blurred = cv2.GaussianBlur(image, (51, 51), 0)
  return np.where(mask[..., None] > 0, blurred, image)


def main():
  args = _parse_arguments()
  
  for filename in os.listdir(args.data_loc):
    if filename.startswith("."): continue 

    filename_no_ext, ext = os.path.splitext(filename)
    if ext and ext != ".jpg": continue
    
    print(f"Blurring logo(s) of {filename}...")
    image = cv2.imread(os.path.join(args.data_loc, filename))
    mask = cv2.imread(os.path.join(args.info_loc, f"res_{filename_no_ext}_mask.jpg"), cv2.IMREAD_GRAYSCALE)
    
    bounding_boxes = os.path.join(args.info_loc, f"res_{filename_no_ext}.txt")
    with open(bounding_boxes, 'r') as file:
      for line in file:
        coordinates = list(map(int, line.strip().split(',')))
        image = _blur_area_by_bounding_box(image, coordinates)

    image = _blur_area_by_mask(image, mask[:, :mask.shape[1]//2])  
    cv2.imwrite(os.path.join(args.save_loc, filename), image)


if __name__ == "__main__":
  main()
