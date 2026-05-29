# ECS111 Final Project

This project builds an image-based food analysis pipeline using PyTorch. The main notebook trains and evaluates models for:

- food classification (`burger`, `pasta`, `pizza`, `salad`, `sushi`)
- portion-size classification (`small`, `medium`, `large`)
- calorie-level classification on Nutrition5k (`low`, `medium`, `high`)

The final pipeline predicts food type and portion size from an input image, then maps those predictions to a calorie level with a rule-based step.

## Project Structure

```text
ECS111Final/
|-- Data/
|   |-- food101_subset/
|   |-- nutrition5k/
|   |-- food101_labels_final.csv
|   |-- portion_labeling_1500_clean.csv
|-- outputs/
|   |-- *.pth
|   |-- *_metrics.csv
|   |-- *_history.csv
|   |-- *_confusion_matrix.png
|   |-- final_pipeline_examples.csv
|   |-- results_report.txt
|-- Main.ipynb
|-- .gitignore
```

## Main Files

- `Main.ipynb`: main training, evaluation, and inference notebook
- `Data/food101_labels_final.csv`: labeled Food-101 subset metadata
- `Data/portion_labeling_1500_clean.csv`: cleaned portion labels used for portion-model training
- `Data/nutrition5k/nutrition5k_labels.csv`: Nutrition5k calorie labels and metadata
- `outputs/results_report.txt`: short written summary of the final results

## Models Saved in `outputs/`

- `outputs/food_model.pth`: EfficientNet-B0 food classifier
- `outputs/portion_model.pth`: EfficientNet-B0 portion classifier
- `outputs/nutrition5k_baseline_model.pth`: EfficientNet-B0 calorie-level classifier
- `outputs/food_resnet18_baseline_model.pth`: ResNet-18 food baseline

Each checkpoint stores:

- model architecture name
- class labels
- trained model weights
- best validation accuracy
- image size and normalization values

## Results Summary

Based on the saved metrics:

- Food classification performed best.
  - EfficientNet-B0: `96.8%` validation accuracy, `96.4%` test accuracy
  - ResNet-18 baseline: `95.4%` validation accuracy, `94.6%` test accuracy
- Portion classification was much harder.
  - EfficientNet-B0: `46.7%` validation accuracy, `33.3%` test accuracy
- Nutrition5k calorie classification had moderate performance.
  - EfficientNet-B0: `33.3%` validation accuracy, `50.0%` test accuracy

See:

- `outputs/food_metrics.csv`
- `outputs/food_resnet18_baseline_metrics.csv`
- `outputs/portion_metrics.csv`
- `outputs/nutrition5k_metrics.csv`

## Environment

This project uses Python in `.venv` and PyTorch with CUDA support.

To verify GPU availability:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## How to Run

Open `Main.ipynb` in PyCharm or Jupyter and run the cells in order.

The notebook will:

1. load data from `Data/`
2. build dataloaders
3. train food, portion, and calorie models
4. save checkpoints and metrics into `outputs/`
5. generate pipeline example predictions

## Predict on a Single Image

Use the saved `.pth` checkpoints to run inference on one image.

Example workflow:

1. load a checkpoint from `outputs/`
2. rebuild the model architecture
3. apply the evaluation transform
4. run the image through the model
5. read the predicted class and confidence

You can use:

- `outputs/food_model.pth` for food prediction
- `outputs/portion_model.pth` for portion prediction
- `outputs/nutrition5k_baseline_model.pth` for direct calorie-level prediction

## Notes

- The notebook expects the dataset folder to be named `Data/`
- CSV image paths are resolved correctly even when they use `data/...`
- `outputs/` contains generated artifacts and is ignored for future Git changes by default in `.gitignore`

## Future Improvements

- improve portion-size labeling quality and dataset size
- add more balanced calorie-level training data
- replace or augment the rule-based calorie step with a learned model
- add a standalone inference script outside the notebook
