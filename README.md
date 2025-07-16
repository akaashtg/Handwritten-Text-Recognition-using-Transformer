# Handwritten-Text-Recognition-using-Transformer

---

## 📚 Abstract

This project presents a deep learning-based handwritten text recognition system using the **TrOCR model**. The pipeline consists of three major phases:

1. **Data Preprocessing**:
   - Uses IAM dataset (35,391 word-image/label pairs)
   - RGB conversion using PIL
   - Tokenization and normalization using `TrOCRProcessor`

2. **Model Fine-Tuning**:
   - Model: `microsoft/trocr-base-handwritten`
   - Tools: Hugging Face `Trainer`, `Seq2SeqTrainingArguments`
   - Data augmentation: rotation, blur, contrast

3. **Inference & Evaluation**:
   - Decoding using beam search
   - Metrics: CER and WER via `evaluate` and `jiwer`

---

## 🖼️ Sample Prediction

| Input Image     | Ground Truth     | Predicted Text             |
|-----------------|------------------|-----------------------------|
| word_001.png    | handwritten text | handwritten text            |
| word_002.png    | machine learning | machine learning            |

---

## 🛠️ Technology Stack

- **Python 3.10**
- **PyTorch**
- **Hugging Face Transformers & Datasets**
- **TrOCRProcessor**
- **PIL / OpenCV** for image handling
- **evaluate, jiwer** for WER/CER computation

---

## ✅ Requirements

```txt
transformers
datasets
evaluate
jiwer
torch
opencv-python
Pillow
