# Street Name Matcher using BERT

A deep learning-based solution for matching street names across different languages and formats using multilingual BERT. This system can compare two street names and determine if they refer to the same street, along with a confidence score.

## Features

- Multilingual support (104 languages via BERT)
- Handles different street name formats and abbreviations
- Returns both confidence score (0-1) and boolean match status
- Support for various writing systems and scripts
- Handles common variations like "Street" vs "St", "Avenue" vs "Ave", etc.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

- `street_matcher_bert.py`: Main implementation of the StreetMatcherBERT model and inference functions
- `train_street_matcher_bert.py`: Training script and dataset handling
- `requirements.txt`: Required Python packages
- `street_matcher.pth`: Trained model weights (generated after training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/street-matcher-bert.git
cd street-matcher-bert
```

3. (Optional) Download pre-trained weights if available:
```bash
# Place street_matcher.pth in the project root directory
```

## Usage

### Basic Comparison

```python
from street_matcher_bert import StreetMatcherBERT, compare_streets

# Initialize model
model = StreetMatcherBERT()

# Optional: Load trained weights
# model.load_state_dict(torch.load('street_matcher.pth'))

# Compare two street names
street1 = "Main Street"
street2 = "Main St"
confidence, is_match = compare_streets(street1, street2, model)

print(f"Confidence: {confidence:.2f}")
print(f"Match: {is_match}")
```

### Training Custom Model

```python
from train_street_matcher_bert import train_model, StreetPairDataset
from torch.utils.data import DataLoader

# Prepare training data
street_pairs = [
    ("Main Street", "Main St"),
    ("Broadway Ave", "Broadway Avenue"),
    # Add more pairs...
]

labels = [
    (0.9, 1),  # (confidence, is_match)
    (0.95, 1),
    # Add more labels...
]

# Create dataset and train
dataset = StreetPairDataset(street_pairs, labels)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
model = StreetMatcherBERT()
train_model(model, train_loader)
```

## Multilingual Capabilities

The model supports 104 languages through BERT's multilingual model. Examples:

```python
test_pairs = [
    ("Main Street", "Main St"),            # English
    ("Hauptstraße", "Hauptstrasse"),       # German
    ("Rue de la Paix", "Rue de la Paix"),  # French
    ("新宿通り", "しんじゅくどおり"),        # Japanese
    ("Calle Mayor", "C/ Mayor"),           # Spanish
]
```

## Model Details

### Architecture
- Base Model: bert-base-multilingual-cased
- Additional Layers:
  - Linear classifier layer (BERT hidden size → 2)
  - Sigmoid activation for confidence and match prediction

### Input Processing
- Maximum sequence length: 128 tokens
- Special tokens added automatically
- Handles two street names as a pair
- Automatic padding and truncation

### Outputs
- Confidence Score: Float between 0 and 1
- Match Status: Boolean (True/False)

## Training Information

### Dataset Preparation
Recommended training data should include:
- Street pairs with various formats
- Multiple languages and scripts
- Common abbreviations
- Misspellings and variations
- Regional naming conventions

### Training Parameters
- Learning Rate: 2e-5
- Optimizer: AdamW
- Loss Functions:
  - MSE Loss for confidence
  - BCE Loss for match prediction
- Batch Size: 2 (adjustable)
- Epochs: 10 (adjustable)

### Best Practices
1. Balance the dataset across languages
2. Include both matching and non-matching pairs
3. Use real-world street names when possible
4. Include edge cases and common variations
