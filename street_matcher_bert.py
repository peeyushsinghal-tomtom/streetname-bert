from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class StreetMatcherBERT(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        super(StreetMatcherBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add a classifier layer on top of BERT's CLS token
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 2 outputs: confidence and match/nomatch
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, street1, street2):
        # Prepare input text
        encoded = self.tokenizer(
            text=street1,
            text_pair=street2,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
        
        # Get CLS token representation
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Get classifier outputs
        logits = self.classifier(cls_token)
        
        # Split outputs into confidence and match prediction
        confidence = self.sigmoid(logits[:, 0])  # Scale to 0-1
        match_prob = self.sigmoid(logits[:, 1])  # Scale to 0-1
        
        return confidence, match_prob > 0.5  # Return confidence and boolean match

def compare_streets(street1, street2, model=None, threshold=0.5):
    """
    Compare two street names and return confidence score and match status
    
    Args:
        street1 (str): First street name
        street2 (str): Second street name
        model (StreetMatcherBERT): Pre-trained model instance
        threshold (float): Threshold for considering a match (0-1)
        
    Returns:
        tuple: (confidence_score, is_match)
    """
    if model is None:
        model = StreetMatcherBERT()
        # Note: In production, you should load pre-trained weights here
    
    model.eval()
    with torch.no_grad():
        confidence, is_match = model(street1, street2)
        return confidence.item(), bool(is_match.item())

# Example usage
if __name__ == "__main__":
    # Test with different languages
    test_pairs = [
        ("Main Street", "Main St"),  # English
        ("Hauptstraße", "Hauptstrasse"),  # German
        ("Rue de la Paix", "Rue de la Paix"),  # French
        ("新宿通り", "しんじゅくどおり"),  # Japanese
        ("Calle Mayor", "C/ Mayor"),  # Spanish
    ]
    
    model = StreetMatcherBERT()
    
    for street1, street2 in test_pairs:
        confidence, is_match = compare_streets(street1, street2, model)
        print(f"\nComparing: '{street1}' vs '{street2}'")
        print(f"Confidence: {confidence:.2f}")
        print(f"Match: {is_match}")