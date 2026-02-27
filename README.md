# AI-Driven Analysis of Human Emotions & Psychological Health

**Complete End-to-End Machine Learning Pipeline for Emotion Classification, Psychological State Estimation, and Empathetic Response Generation**

---

**Author:** Nipun Chugh | **Institution:** Chandigarh University | **Date:** February 2026  

---

## Overview

This project addresses a fundamental gap in current AI systems: the inability to understand human emotions with genuine psychological depth. Most existing approaches treat emotional understanding as simple sentiment classification (positive/negative). This work builds a system capable of recognizing 28 fine-grained emotions, estimating psychological health indicators (depression, anxiety, stress), detecting crisis signals, tracking emotional trajectories over time, and generating empathetic responses grounded in the user's emotional state.

The system is designed as augmentative technology — it is not a replacement for professional therapy, but a tool to make AI interactions meaningfully more human-aware.

---

## Pipeline Architecture

```
Raw Text
    |
    v
[Feature Extraction]
    |
    v
[DistilBERT Transformer Encoder]
    |
    +---> Emotion Classification    (28 classes)
    +---> Psychological Estimation  (Depression / Anxiety / Stress)
    +---> Crisis Detection          (Binary: safe / at-risk)
    |
    v
[Temporal LSTM Tracker]   <-- Tracks emotional trajectory across conversation turns
    |
    v
[Empathetic Response Generator]
    |
    v
[Safety & Ethics Filter]
    |
    v
Output Response
```

---

## Key Features

- **28-Class Emotion Classification** using GoEmotions taxonomy (fine-grained beyond basic happy/sad/angry)
- **Psychological State Estimation** — continuous scoring for Depression, Anxiety, and Stress levels
- **Crisis Detection Module** with high-recall binary classification; false negative rate target below 5%
- **Temporal LSTM Tracker** — maintains a rolling window of conversation history to detect emotional deterioration or improvement
- **Empathetic Response Generator** with integrated safety filter to prevent harmful or medically prescriptive outputs
- **20 Psychologically-Validated Linguistic Features** including first-person pronoun density, absolutist language, lexical diversity, and isolation word usage
- **Multi-Task Learning Architecture** with weighted loss function (crisis detection prioritized at 2x weight)

---

## Datasets

The system is trained on a multi-source corpus of approximately 230,000 entries across four datasets:

| # | Dataset | Size | Purpose |
|---|---------|------|---------|
| 1 | GoEmotions (Google) | ~58,000 Reddit comments | Fine-grained emotion classification (27 categories) |
| 2 | Empathetic Dialogues (Facebook Research) | ~25,000 conversations | Empathetic response training |
| 3 | DailyDialog | ~13,000 dialogues | Cross-dataset emotion validation in everyday conversation |
| 4 | TweetEval | ~45,000 tweets | Social media emotion and sentiment |

---

## Model Architecture

### Core Encoder
- **DistilBERT** (`distilbert-base-uncased`) — a lightweight, efficient transformer encoder fine-tuned for multi-task learning

### Output Heads

**Emotion Head**  
`Linear(768+features → 512) → ReLU → Dropout(0.3) → Linear(512 → 28)`  
Multi-label sigmoid classification across 28 emotion categories.

**Psychological State Head**  
`Linear(768+features → 256) → ReLU → Dropout(0.3) → Linear(256 → 3)`  
Continuous regression output for Depression, Anxiety, and Stress scores.

**Crisis Detection Head**  
`Linear(768+features → 128) → ReLU → Linear(128 → 2)`  
Binary softmax classifier. Weighted at γ=2.0 in the composite loss function.

### Temporal Tracker
A two-layer LSTM with a rolling window of 10 messages tracks psychological score sequences to detect trends (improving, deteriorating, volatile) across conversation turns.

### Loss Function
```
Total_Loss = α · EmotionLoss + β · PsychLoss + γ · CrisisLoss
```
where γ = 2.0 to prioritize crisis recall over precision.

---

## Psychological Feature Engineering

20 psychologically-validated features are extracted from each input text and concatenated with the transformer's [CLS] token representation:

| Feature Category | Examples |
|-----------------|---------|
| Sentiment | VADER compound score, positive/negative word counts |
| Self-focus | First-person pronoun density (correlates with depression) |
| Absolutist language | "always", "never", "completely" (common in anxiety/depression) |
| Lexical diversity | Unique word ratio (low diversity signals cognitive narrowing) |
| Isolation markers | "alone", "lonely", "nobody" |
| Communication style | Message length, question/exclamation density, ellipsis usage |
| Negation | Negation word frequency |

---

## Project Structure

```
AI_Emotions_Psychological_Health_Project_Version2.ipynb
|
|-- 1.  Environment Setup
|-- 2.  Multi-Source Data Loading (GoEmotions, Empathetic Dialogues, DailyDialog, TweetEval)
|-- 3.  Exploratory Data Analysis
|-- 4.  Psychological Feature Engineering
|-- 5.  GoEmotions Multi-Task Preparation
|-- 6.  Linguistic Feature Extraction (Full Dataset)
|-- 7.  Feature Correlation with Psychological States
|-- 8.  PyTorch Dataset & DataLoader
|-- 9.  Multi-Task Model Architecture
|-- 10. Multi-Task Training
|-- 11. Training Progress Visualization
|-- 12. Comprehensive Test Set Evaluation
|-- 13. Detailed Per-Emotion Analysis
|-- 14. Psychological Score Detailed Analysis
|-- 15. Crisis Detection Safety Analysis
|-- 16. Temporal State Tracker (LSTM)
|-- 17. Empathetic Response Generator + Safety Filter
|-- 18. Complete End-to-End Pipeline
|-- 19. Interactive Conversation Demo
|-- 20. Conversation Trajectory Visualization
|-- 21. Empathetic Dialogues Response Training Analysis
|-- 22. DailyDialog Cross-Dataset Validation
|-- 23. Save Model & Artifacts
|-- 24. Final Results Report
|-- 25. Future Work & Recommendations
```

---

## Evaluation Methodology

The system is validated across multiple dimensions:

**Technical Metrics (Emotion Classification)**
- Accuracy, F1-Macro, F1-Micro, F1-Weighted
- Per-class Precision and Recall
- Confusion Matrix analysis

**Psychological State Assessment**
- Pearson and Spearman correlation between predicted scores and reference clinical scores
- Target: r > 0.75 (strong correlation)

**Crisis Detection**
- Sensitivity (Recall) — must exceed 95%
- False Negative Rate — must remain below 5%
- Precision-Recall tradeoff explicitly favors recall (missing a crisis is more dangerous than a false alarm)

**Cross-Dataset Validation**
- DailyDialog used as an independent held-out validation set
- Evaluates generalization beyond the training distribution

**Planned Human Evaluation (Future Work)**
- User study: N=50 participants, 7-day usage period
- Therapist blind evaluation comparing AI vs human responses
- Empathy, Appropriateness, and Therapeutic Value rubric (1-5 scale)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Nipunchugh10/ai-emotions-psychological-health
cd ai-emotions-psychological-health

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets scikit-learn pandas numpy matplotlib seaborn
pip install nltk textblob scipy tqdm wordcloud

# For GPU support, replace the torch URL with:
# --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

Open the notebook in Jupyter or Google Colab and run cells sequentially:

```bash
jupyter notebook AI_Emotions_Psychological_Health_Project_Version2.ipynb
```

The notebook is self-contained. Each section builds on the previous one, from raw data loading through to the interactive conversation demo in Section 19.

For a quick end-to-end test, run Section 18 (Complete End-to-End Pipeline) after training is complete. The interactive demo in Section 19 accepts multi-turn conversation input and returns emotion labels, psychological scores, crisis risk, and an empathetic response.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP | DistilBERT, NLTK, TextBlob, VADER |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Temporal Modeling | PyTorch LSTM |
| Datasets | HuggingFace Datasets Hub, CSV upload |
| Experiment Tracking | Manual logging (Weights & Biases planned) |

---

## Ethical Considerations

This system is designed as a research prototype and augmentative tool only.

- It does not provide medical diagnoses or clinical assessments
- It does not replace licensed therapists or mental health professionals
- Crisis detection is designed to flag for human review, not to act autonomously
- All training data used consists of publicly available, consent-based datasets
- The safety filter actively suppresses prescriptive medical advice in generated responses

Any real-world deployment of this system would require institutional review board approval, clinical validation, and ongoing human oversight.

---

## Future Work

**Model Improvements**
- Upgrade encoder to `mental/mental-bert-base-uncased` or `roberta-large`
- Train for 10+ epochs on GPU with learning rate warmup scheduler
- Integrate DAIC-WOZ clinical interview data for real PHQ-9/GAD-7 ground truth

**Response Generation**
- Fine-tune LLaMA-2-7B or Mistral-7B on the 19k empathetic dialogue pairs
- Implement Retrieval-Augmented Generation (RAG) with a curated therapeutic response database

**Safety & Fairness**
- Red-team crisis detection for false negatives across demographic groups
- Bias audits across age, gender, and cultural subgroups
- Real-time human escalation integration for high-risk detections

**Evaluation**
- Conduct N=50 user study with 7-day longitudinal tracking
- Blind therapist evaluation of AI vs human response quality
- Multimodal extension incorporating voice prosody and physiological signals

---





## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

*This project is part of final year B.Tech research work. It is intended strictly for academic and research purposes.*
