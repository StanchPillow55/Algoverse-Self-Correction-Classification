# Error + Confidence Classifier Architecture

## Overview

The Error + Confidence Classifier is a multi-head neural network designed to predict:
1. **Failure mode classification** (6 classes): anchored, overcorrected, corrected, unchanged_correct, wavering, perfectionism
2. **Confidence score prediction** (regression): Delta accuracy score (-1, 0, +1)

## Architecture Design

### 1. Input Processing
- **Text Input Format**: `"Initial: {initial_answer} [SEP] Revised: {revised_answer} [SEP] Prompt: {reprompt_id}"`
- **Optional Logits Features**: Numerical features derived from model confidence/uncertainty
- **Tokenization**: Uses lightweight transformer tokenizer (sentence-transformers/all-MiniLM-L6-v2)

### 2. Model Components

#### A. Text Encoder
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - Lightweight transformer (22M parameters)
  - Fast inference, good performance on sentence similarity tasks
  - Pre-trained on diverse text data
- **Output**: [CLS] token representation (384-dim vector)

#### B. Optional Logits Feature Encoder (if available)
```python
self.logits_encoder = nn.Sequential(
    nn.Linear(logits_feature_dim, hidden_dim),      # e.g., 64 -> 256
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2)         # 256 -> 128
)
```

#### C. Feature Fusion Layer
```python
fusion_input_dim = text_encoder_dim + logits_output_dim  # 384 + 128 = 512
self.fusion_layer = nn.Sequential(
    nn.Linear(fusion_input_dim, hidden_dim),       # 512 -> 256
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim)              # 256 -> 256
)
```

#### D. Prediction Heads

**Failure Mode Classifier Head:**
```python
self.failure_mode_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),        # 256 -> 128
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, num_failure_modes)  # 128 -> 6
)
```

**Confidence Score Regression Head:**
```python
self.confidence_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),        # 256 -> 128
    nn.ReLU(), 
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, 1)                  # 128 -> 1
)
```

### 3. Loss Function
- **Multi-task Loss**: `total_loss = classification_loss + 0.5 * regression_loss`
- **Classification Loss**: CrossEntropyLoss with class balancing weights
- **Regression Loss**: MSELoss for confidence score prediction

## Training Configuration

```python
@dataclass
class TrainingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    hidden_dim: int = 256
    dropout: float = 0.1
    logits_feature_dim: int = 64
    weight_decay: float = 0.01
    use_class_weighting: bool = True
```

## Data Format

### Input CSV Columns
- `initial_answer`: Original answer from the model
- `revised_answer`: Revised answer after reprompting  
- `reprompt_id`: ID of the reprompt template used
- `failure_mode`: Target failure mode label
- `delta_accuracy`: Change in accuracy (-1, 0, +1)
- `logits_features` (optional): JSON array of logits-derived features

### Example Data
```csv
initial_answer,revised_answer,reprompt_id,failure_mode,delta_accuracy
London,Paris,neutral_verification,corrected,1
Paris,Lyon,adversarial_challenge,overcorrected,-1
Madrid,Madrid,critique_revise_specific,anchored,0
Paris,Paris,concise_confirmation,unchanged_correct,0
```

## Feature Engineering

### Text Features
- **Sequence Pair Processing**: Compares initial vs revised answers
- **Prompt Context**: Includes reprompt ID for style-aware prediction
- **Semantic Representations**: Uses pre-trained embeddings

### Optional Logits Features
- **Confidence Scores**: Raw model confidence in answers
- **Entropy Measures**: Uncertainty quantification
- **Top-K Probabilities**: Distribution over candidate answers
- **Token-level Statistics**: Per-token confidence measures

## Training Process

### 1. Data Preparation
```python
# Load and split data
df = pd.read_csv(data_path)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=failure_modes)

# Compute class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels)
```

### 2. Training Loop
```python
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_ids, attention_mask, logits_features)
    
    # Multi-task loss
    failure_loss = CrossEntropyLoss(outputs['failure_mode_logits'], labels)
    confidence_loss = MSELoss(outputs['confidence_score'], delta_accuracy)
    total_loss = failure_loss + 0.5 * confidence_loss
    
    # Optimization
    total_loss.backward()
    optimizer.step()
```

### 3. Evaluation Metrics
- **Classification**: Macro F1-score, per-class precision/recall
- **Regression**: Mean Squared Error for confidence prediction
- **Combined**: Weighted loss combining both objectives

## Usage Examples

### Training
```python
# Initialize
config = TrainingConfig(batch_size=16, num_epochs=10)
trainer = ErrorConfidenceTrainer(config)

# Train
train_loader, val_loader = trainer.prepare_data("training_data.csv")
metrics = trainer.train(train_loader, val_loader)

# Save
trainer.save_model("model.pt", "vectorizer.pkl")
```

### Inference
```python
# Load model
trainer.load_model("model.pt", "vectorizer.pkl")

# Predict
prediction = trainer.predict(
    initial_answer="London",
    revised_answer="Paris", 
    reprompt_id="neutral_verification"
)

print(f"Failure mode: {prediction['failure_mode']}")
print(f"Confidence: {prediction['confidence_score']:.3f}")
```

## Performance Considerations

### Model Size
- **Base transformer**: ~22M parameters (lightweight)
- **Additional layers**: ~1M parameters  
- **Total**: ~23M parameters (manageable for CPU inference)

### Inference Speed
- **Text encoding**: ~5ms per example (CPU)
- **Feature fusion**: ~1ms per example
- **Total**: ~6ms per prediction (suitable for real-time use)

### Memory Usage
- **Training**: ~2GB GPU memory (batch_size=16)
- **Inference**: ~500MB RAM for model loading

## Extension Points

### 1. Additional Features
- **Turn History**: Include previous turns in multi-pass correction
- **Question Context**: Add original question for better understanding
- **Prompt Embeddings**: Learn representations for reprompt templates

### 2. Architecture Variations
- **Attention Fusion**: Replace concatenation with attention mechanisms
- **Task-Specific Encoders**: Separate encoders for different tasks
- **Hierarchical Classification**: Multi-level failure mode taxonomy

### 3. Training Improvements
- **Curriculum Learning**: Start with easier examples
- **Active Learning**: Select most informative examples for labeling
- **Multi-Task Balancing**: Dynamic loss weighting during training

## Integration with RTS Policy

The classifier outputs are designed to integrate directly with the RTS (Reprompt Template Selection) policy:

```python
# Get predictions
failure_mode, confidence = classifier.predict(initial, revised, prompt_id)

# Use in RTS policy
if failure_mode == "anchored" and confidence > 0.6:
    selected_prompt = rts_policy.select_adversarial_prompt()
elif failure_mode == "overcorrected":
    selected_prompt = rts_policy.select_supportive_prompt()
# ... etc
```

This enables the system to make informed decisions about when and how to reprompt based on detected failure patterns.
