

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch

def process_text_with_xlmr(text, max_length=128):
    # Initialize the XLM-R tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")

    # Return the tokenized input
    return inputs

def classify_text_with_xlmr(text):
    # Initialize the XLM-R model for sequence classification
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base")

    # Tokenize the input text
    inputs = process_text_with_xlmr(text)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class (assumes single-label classification)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()

    return predicted_class

# Example usage:
text = "Как дела?"  # Example Russian text
inputs = process_text_with_xlmr(text)
print("Tokenized Inputs:", inputs)

predicted_class = classify_text_with_xlmr(text)
print("Predicted Class:", predicted_class)
