import argparse

from frame_extractors import MiddleFrameExtractor

parser = argparse.ArgumentParser(description="Extracting frames from dataset")

parser.add_argument("--data_loc", type=str)
parser.add_argument("--save_loc", type=str)
parser.add_argument("--n_frames", type=int)

args = parser.parse_args()

extractor = MiddleFrameExtractor(args.data_loc, args.save_loc, args.n_frames)

extractor()