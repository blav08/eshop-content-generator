# E-shop Content Generator

An AI-powered tool that automatically classifies fashion product images and generates marketing-ready descriptions for e-commerce use.

## What it does

Given a product photo, the system:
1. **Classifies** the item into one of 11 fashion categories (e.g. sneakers, jacket, handbag) using a fine-tuned ResNet18 model
2. **Generates** a natural language description of the product using the BLIP image captioning model
3. **Outputs** a marketing-ready text combining both, along with a confidence score
4. **Flags** low-confidence results for manual review via a built-in governance layer

## Categories

| Main Category | Items |
|---|---|
| Clothing | dress, jeans, jacket, t-shirt |
| Shoes | sneakers, high heels, running shoes |
| Accessories | handbag, sunglasses, hat, watch |

## Example output

```json
{
  "image": "product.jpg",
  "category": "Shoes - sneakers",
  "confidence": 0.91,
  "needs_review": false,
  "technical_desc": "a pair of white sneakers on a white background",
  "marketing_desc": "Discover our new sneakers. A pair of white sneakers on a white background. A perfect choice for your style!",
  "model_used": "ResNet18_Detailed_v2"
}
```

## Tech stack

- **PyTorch** – model training and inference
- **ResNet18** – fine-tuned image classifier (custom fashion dataset)
- **HuggingFace Transformers / BLIP** – image captioning
- **Selenium + BeautifulSoup** – dataset preparation
- **CSV logging** – governance and audit trail

## Project structure

```
├── main.py                 # Main pipeline (classify + caption + generate)
├── train_model.py          # ResNet18 fine-tuning
├── prepare_kaggle_data.py  # Dataset preparation
├── student_model.pth       # Trained model weights
├── test_images/            # Place your product images here
├── result/                 # Output JSON + governance log
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.8+. GPU recommended but not required (CPU fallback included).

## Usage

Place your product images in the `test_images/` folder, then run:

```bash
python main.py
```

Results are saved to `result/final_output.json`. Low-confidence predictions are flagged in `result/governance_log.csv`.

## Author

[Vítek Blažek](https://github.com/blav08)