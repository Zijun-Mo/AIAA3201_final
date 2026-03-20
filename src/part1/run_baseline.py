#!/usr/bin/env python3
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Part 1 baseline pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--dataset", default="all", help="Dataset split or name")
    return parser.parse_args()


def main():
    args = parse_args()
    print("[Part1] Baseline pipeline entrypoint")
    print(f"config={args.config}, dataset={args.dataset}")
    print("TODO: implement YOLO/Mask R-CNN + flow filtering + cv2.inpaint")


if __name__ == "__main__":
    main()
