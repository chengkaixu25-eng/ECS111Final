from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FOOD_CSV = DATA_DIR / "food101_labels_final.csv"
NUTRITION_CSV = DATA_DIR / "nutrition5k" / "nutrition5k_labels.csv"

FOOD_CLASSES = ["burger", "pasta", "pizza", "salad", "sushi"]
PORTION_CLASSES = ["small", "medium", "large"]
CALORIE_CLASSES = ["low", "medium", "high"]
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

calorie_map = {
    ("salad", "small"): "low",
    ("salad", "medium"): "low",
    ("salad", "large"): "medium",
    ("sushi", "small"): "low",
    ("sushi", "medium"): "medium",
    ("sushi", "large"): "high",
    ("pizza", "small"): "medium",
    ("pizza", "medium"): "high",
    ("pizza", "large"): "high",
    ("burger", "small"): "medium",
    ("burger", "medium"): "high",
    ("burger", "large"): "high",
    ("pasta", "small"): "medium",
    ("pasta", "medium"): "high",
    ("pasta", "large"): "high",
}

eval_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def resolve_image_path(path_value: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path

    parts = list(path.parts)
    if parts and parts[0].lower() == "data":
        path = Path("Data", *parts[1:])

    return PROJECT_ROOT / path


def extract_picture_number(image_path: Path) -> str:
    stem = image_path.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[-1]
    return stem


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {arch}")


def load_checkpoint_model(checkpoint_name: str) -> tuple[nn.Module, list[str]]:
    checkpoint_path = OUTPUT_DIR / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_model(checkpoint["arch"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()
    return model, checkpoint["classes"]


@torch.no_grad()
def predict_single_image(
    model: nn.Module,
    image_path: Path,
    classes: list[str],
) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred_idx = int(probs.argmax().item())
    return classes[pred_idx], float(probs[pred_idx].item())


def choose_random_image() -> Path:
    if FOOD_CSV.exists():
        food_df = pd.read_csv(FOOD_CSV)
        labeled_df = food_df[food_df["portion_label"].notna()].copy()
        if "split" in labeled_df.columns:
            test_df = labeled_df[labeled_df["split"] == "test"].copy()
            if not test_df.empty:
                row = test_df.sample(n=1).iloc[0]
                return resolve_image_path(row["image_path"])
        if not labeled_df.empty:
            row = labeled_df.sample(n=1).iloc[0]
            return resolve_image_path(row["image_path"])

    candidates = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No images found under {DATA_DIR}")
    return random.choice(candidates)


def lookup_actual_labels(image_path: Path) -> dict[str, str | None]:
    resolved_image_path = image_path.resolve()

    if FOOD_CSV.exists():
        food_df = pd.read_csv(FOOD_CSV)
        food_df["resolved_image_path"] = food_df["image_path"].apply(
            lambda value: str(resolve_image_path(value).resolve())
        )
        matches = food_df[food_df["resolved_image_path"] == str(resolved_image_path)]
        if not matches.empty:
            row = matches.iloc[0]
            actual_food = row.get("food_label")
            actual_portion = row.get("portion_label")
            actual_calorie = None
            if isinstance(actual_portion, str) and actual_portion in PORTION_CLASSES:
                actual_calorie = calorie_map.get((actual_food, actual_portion))
            return {
                "actual_food": actual_food if isinstance(actual_food, str) else None,
                "actual_portion": actual_portion if isinstance(actual_portion, str) else None,
                "actual_calorie_level": actual_calorie,
            }

    if NUTRITION_CSV.exists():
        nutrition_df = pd.read_csv(NUTRITION_CSV)
        nutrition_df["resolved_image_path"] = nutrition_df["image_path"].apply(
            lambda value: str(resolve_image_path(value).resolve())
        )
        matches = nutrition_df[nutrition_df["resolved_image_path"] == str(resolved_image_path)]
        if not matches.empty:
            row = matches.iloc[0]
            calorie_level = row.get("calorie_level")
            return {
                "actual_food": None,
                "actual_portion": None,
                "actual_calorie_level": calorie_level if isinstance(calorie_level, str) else None,
            }

    return {
        "actual_food": None,
        "actual_portion": None,
        "actual_calorie_level": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a random or user-specified image through the saved final models."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path. If omitted, the script picks a random dataset image.",
    )
    args = parser.parse_args()

    image_path = resolve_image_path(args.image) if args.image else choose_random_image()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    picture_number = extract_picture_number(image_path)
    actual_labels = lookup_actual_labels(image_path)

    food_model, food_classes = load_checkpoint_model("food_model.pth")
    portion_model, portion_classes = load_checkpoint_model("portion_model.pth")
    calorie_model, calorie_classes = load_checkpoint_model("nutrition5k_balanced_baseline_model.pth")

    predicted_food, food_conf = predict_single_image(food_model, image_path, food_classes)
    predicted_portion, portion_conf = predict_single_image(portion_model, image_path, portion_classes)
    pipeline_calorie = calorie_map[(predicted_food, predicted_portion)]
    predicted_calorie, calorie_conf = predict_single_image(calorie_model, image_path, calorie_classes)

    print(f"Using device: {DEVICE}")
    print(f"Image: {image_path}")
    print(f"Picture number: {picture_number}")
    print(
        "Actual calorie level: "
        f"{actual_labels['actual_calorie_level'] if actual_labels['actual_calorie_level'] else 'unavailable'}"
    )
    print()
    print("Final pipeline prediction")
    print(f"  Food: {predicted_food} ({food_conf * 100:.2f}% confidence)")
    print(f"  Portion: {predicted_portion} ({portion_conf * 100:.2f}% confidence)")
    print(f"  Calorie level (rule-based from final pipeline): {pipeline_calorie}")
    print()
    print("Direct calorie model prediction")
    print(f"  Calorie level: {predicted_calorie} ({calorie_conf * 100:.2f}% confidence)")


if __name__ == "__main__":
    main()
