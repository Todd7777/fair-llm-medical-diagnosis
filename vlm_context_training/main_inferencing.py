import os
import argparse
from model import Model
import torch

"""
If image_dir_1 contains:

img1.jpg
img2.jpg
img3.jpg

Then ground_truth.txt should look like:

0.0,1.0,0.0,1.0,etc...
0.0,1.0,1.0,0.0,etc...
0.0,0.0,1.0,0.0,etc...
"""


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(
        description="A simulation of inference with context windows(chat history) using two phase multi-image inference"
    )
    parser.add_argument("--model_name", required=True, help='"qwen" or "llava"')
    parser.add_argument(
        "--image_dir_1", required=True, help="Dir for first set of images"
    )
    parser.add_argument(
        "--image_dir_2", required=True, help="Dir for second set of images"
    )
    parser.add_argument(
        "--ground_truth",
        help="Text file with ground truth annotations",
    )
    args = parser.parse_args()

    model = Model(args.model_name)

    training_prompt = (
        "Analyze the images with these boundaries, with the metrics: "
        "Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,"
        "Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices,No Finding"
        "\nAnd give predictions with series of 13 booleans(emitting support devices) represented by floating point numbers, like this: "
        "0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0"
    )

    testing_prompt = (
        "\nBased on the training set and your answers, as well as the ground truth, "
        "do inference on these new testing images."
    )

    supported_exts = (".jpg", ".jpeg", ".png")

    # Load and sort training images
    training_images = sorted(
        [
            os.path.join(args.image_dir_1, f)
            for f in os.listdir(args.image_dir_1)
            if f.lower().endswith(supported_exts)
        ]
    )
    if not training_images:
        raise Exception(f"No images found in {args.image_dir_1}")

    # Load ground truth lines
    if not os.path.exists(args.ground_truth):
        raise Exception(f"No ground truth file found - {args.ground_truth}")
    with open(args.ground_truth, "r") as f:
        ground_truth_lines = [line.strip() for line in f.readlines()]

    # Check for mismatch
    if len(ground_truth_lines) != len(training_images):
        raise Exception(
            "Mismatch between number of training images and ground truth lines."
        )

    # Combine image names with ground truths
    ground_truth_text = "\n".join(
        f"{os.path.basename(img)}: {gt}"
        for img, gt in zip(training_images, ground_truth_lines)
    )

    print("Phase 1: Analyzing training images")
    resp1 = model.infer(prompt=training_prompt, image_paths=training_images)
    print("Response (phase 1):", resp1)

    # Load and sort testing images
    testing_images = sorted(
        [
            os.path.join(args.image_dir_2, f)
            for f in os.listdir(args.image_dir_2)
            if f.lower().endswith(supported_exts)
        ]
    )
    if not testing_images:
        raise Exception(f"No images found in {args.image_dir_2}")

    print("\nPhase 2: Analyzing new images with context")
    combined_prompt = (
        f'First, you were asked: "{training_prompt}" about images {training_images}. '
        f'You answered: "{resp1}".\n\n'
        f"The ground truth, with the last column of each being support devices(ignore them when comparing to yours):\n{ground_truth_text}\n\n"
        f"Here are the new test images: {testing_images}. {testing_prompt}"
    )

    resp2 = model.infer(
        prompt=combined_prompt, image_paths=training_images + testing_images
    )
    print("Response (phase 2):", resp2)


if __name__ == "__main__":
    main()
