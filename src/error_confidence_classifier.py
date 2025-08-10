"""
Error + Confidence Classifier for Self-Correction Failure Mode Detection

This module implements a multi-head classifier that predicts:
1. Failure mode classification (anchored, overcorrected, corrected, etc.)
2. Confidence score prediction (regression)

Architecture: Lightweight transformer encoder + MLP heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for classifier training"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight transformer
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    hidden_dim: int = 256
    dropout: float = 0.1
    logits_feature_dim: int = 64  # Dimension for logits features if available
    weight_decay: float = 0.01
    use_class_weighting: bool = True

class SelfCorrectionDataset(Dataset):
    """Dataset for self-correction failure mode classification"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 tokenizer,
                 label_encoder: LabelEncoder,
                 max_length: int = 512,
                 logits_features: Optional[np.ndarray] = None):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.logits_features = logits_features
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Create input text by combining initial and revised answers
        initial_answer = str(row['initial_answer'])
        revised_answer = str(row['revised_answer'])
        reprompt_id = str(row.get('reprompt_id', ''))
        
        # Format input text
        input_text = f"Initial: {initial_answer} [SEP] Revised: {revised_answer} [SEP] Prompt: {reprompt_id}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get labels
        failure_mode = self.label_encoder.transform([row['failure_mode']])[0]
        delta_accuracy = float(row['delta_accuracy'])
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'failure_mode': torch.tensor(failure_mode, dtype=torch.long),
            'delta_accuracy': torch.tensor(delta_accuracy, dtype=torch.float),
        }
        
        # Add logits features if available
        if self.logits_features is not None:
            item['logits_features'] = torch.tensor(
                self.logits_features[idx], dtype=torch.float
            )
        
        return item

class ErrorConfidenceClassifier(nn.Module):
    """
    Multi-head classifier for failure mode detection and confidence prediction
    
    Architecture:
    1. Text encoder (lightweight transformer)
    2. Optional logits feature encoder
    3. Feature fusion layer
    4. Two prediction heads:
       - Failure mode classifier (multi-class)
       - Confidence score predictor (regression)
    """
    
    def __init__(self, 
                 config: TrainingConfig,
                 num_failure_modes: int,
                 logits_feature_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_failure_modes = num_failure_modes
        
        # Text encoder (frozen lightweight transformer)
        self.text_encoder = AutoModel.from_pretrained(config.model_name)
        self.text_encoder_dim = self.text_encoder.config.hidden_size
        
        # Optional logits feature encoder
        self.use_logits_features = logits_feature_dim is not None
        if self.use_logits_features:
            self.logits_encoder = nn.Sequential(
                nn.Linear(logits_feature_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2)
            )
            logits_output_dim = config.hidden_dim // 2
        else:
            logits_output_dim = 0
        
        # Feature fusion layer
        fusion_input_dim = self.text_encoder_dim + logits_output_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Prediction heads
        self.failure_mode_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, num_failure_modes)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                logits_features: Optional[torch.Tensor] = None):
        
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Process logits features if available
        if self.use_logits_features and logits_features is not None:
            logits_encoded = self.logits_encoder(logits_features)
            # Concatenate text and logits features
            fused_features = torch.cat([text_features, logits_encoded], dim=1)
        else:
            fused_features = text_features
        
        # Feature fusion
        fused_features = self.fusion_layer(fused_features)
        
        # Prediction heads
        failure_mode_logits = self.failure_mode_head(fused_features)
        confidence_score = self.confidence_head(fused_features).squeeze(-1)
        
        return {
            'failure_mode_logits': failure_mode_logits,
            'confidence_score': confidence_score
        }

class ErrorConfidenceTrainer:
    """Trainer for the error and confidence classifier"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_weights = None
        
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare training/validation data
        
        Args:
            data_path: Path to CSV file with columns:
                      initial_answer, revised_answer, reprompt_id, failure_mode, delta_accuracy
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        # Encode failure modes
        failure_modes = df['failure_mode'].values
        self.label_encoder.fit(failure_modes)
        num_classes = len(self.label_encoder.classes_)
        logger.info(f"Found {num_classes} failure mode classes: {self.label_encoder.classes_}")
        
        # Compute class weights for balanced loss
        if self.config.use_class_weighting:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.arange(num_classes),
                y=self.label_encoder.transform(failure_modes)
            )
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            logger.info(f"Computed class weights: {self.class_weights}")
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=failure_modes
        )
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation")
        
        # Check for logits features (optional)
        logits_features_train = None
        logits_features_val = None
        if 'logits_features' in df.columns:
            # Assume logits features are stored as JSON strings or arrays
            logits_features_train = self._parse_logits_features(train_df['logits_features'])
            logits_features_val = self._parse_logits_features(val_df['logits_features'])
            logger.info(f"Loaded logits features with dimension {logits_features_train.shape[1]}")
        
        # Create datasets
        train_dataset = SelfCorrectionDataset(
            train_df.reset_index(drop=True),
            self.tokenizer,
            self.label_encoder,
            self.config.max_length,
            logits_features_train
        )
        
        val_dataset = SelfCorrectionDataset(
            val_df.reset_index(drop=True),
            self.tokenizer,
            self.label_encoder,
            self.config.max_length,
            logits_features_val
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Initialize model
        logits_dim = logits_features_train.shape[1] if logits_features_train is not None else None
        self.model = ErrorConfidenceClassifier(
            self.config,
            num_classes,
            logits_dim
        ).to(self.device)
        
        return train_loader, val_loader
    
    def _parse_logits_features(self, logits_series) -> np.ndarray:
        """Parse logits features from DataFrame column"""
        features = []
        for item in logits_series:
            if isinstance(item, str):
                # Parse JSON string
                feature_vec = json.loads(item)
            else:
                # Assume it's already a list/array
                feature_vec = item
            features.append(feature_vec)
        return np.array(features)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train the model
        
        Returns:
            Dictionary with training metrics
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Loss functions
        if self.class_weights is not None:
            failure_mode_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            failure_mode_criterion = nn.CrossEntropyLoss()
        
        confidence_criterion = nn.MSELoss()
        
        # Training loop
        train_metrics = {'epochs': [], 'train_loss': [], 'val_loss': [], 'val_f1': []}\n        \n        for epoch in range(self.config.num_epochs):\n            # Training\n            self.model.train()\n            total_train_loss = 0\n            \n            for batch in train_loader:\n                optimizer.zero_grad()\n                \n                # Move batch to device\n                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v \n                        for k, v in batch.items()}\n                \n                # Forward pass\n                outputs = self.model(\n                    input_ids=batch['input_ids'],\n                    attention_mask=batch['attention_mask'],\n                    logits_features=batch.get('logits_features')\n                )\n                \n                # Compute losses\n                failure_mode_loss = failure_mode_criterion(\n                    outputs['failure_mode_logits'], batch['failure_mode']\n                )\n                confidence_loss = confidence_criterion(\n                    outputs['confidence_score'], batch['delta_accuracy']\n                )\n                \n                # Combined loss (weighted)\n                total_loss = failure_mode_loss + 0.5 * confidence_loss\n                \n                # Backward pass\n                total_loss.backward()\n                optimizer.step()\n                \n                total_train_loss += total_loss.item()\n            \n            # Validation\n            val_metrics = self.evaluate(val_loader)\n            \n            avg_train_loss = total_train_loss / len(train_loader)\n            \n            logger.info(\n                f\"Epoch {epoch+1}/{self.config.num_epochs}: \"\n                f\"Train Loss: {avg_train_loss:.4f}, \"\n                f\"Val Loss: {val_metrics['total_loss']:.4f}, \"\n                f\"Val F1: {val_metrics['macro_f1']:.4f}\"\n            )\n            \n            # Store metrics\n            train_metrics['epochs'].append(epoch + 1)\n            train_metrics['train_loss'].append(avg_train_loss)\n            train_metrics['val_loss'].append(val_metrics['total_loss'])\n            train_metrics['val_f1'].append(val_metrics['macro_f1'])\n        \n        return train_metrics\n    \n    def evaluate(self, data_loader: DataLoader) -> Dict:\n        \"\"\"Evaluate model on validation set\"\"\"\n        self.model.eval()\n        total_loss = 0\n        all_failure_preds = []\n        all_failure_labels = []\n        all_confidence_preds = []\n        all_confidence_labels = []\n        \n        failure_mode_criterion = nn.CrossEntropyLoss()\n        confidence_criterion = nn.MSELoss()\n        \n        with torch.no_grad():\n            for batch in data_loader:\n                # Move batch to device\n                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v \n                        for k, v in batch.items()}\n                \n                # Forward pass\n                outputs = self.model(\n                    input_ids=batch['input_ids'],\n                    attention_mask=batch['attention_mask'],\n                    logits_features=batch.get('logits_features')\n                )\n                \n                # Compute losses\n                failure_mode_loss = failure_mode_criterion(\n                    outputs['failure_mode_logits'], batch['failure_mode']\n                )\n                confidence_loss = confidence_criterion(\n                    outputs['confidence_score'], batch['delta_accuracy']\n                )\n                total_loss += (failure_mode_loss + 0.5 * confidence_loss).item()\n                \n                # Collect predictions\n                failure_preds = torch.argmax(outputs['failure_mode_logits'], dim=1)\n                all_failure_preds.extend(failure_preds.cpu().numpy())\n                all_failure_labels.extend(batch['failure_mode'].cpu().numpy())\n                all_confidence_preds.extend(outputs['confidence_score'].cpu().numpy())\n                all_confidence_labels.extend(batch['delta_accuracy'].cpu().numpy())\n        \n        # Compute metrics\n        from sklearn.metrics import f1_score\n        macro_f1 = f1_score(all_failure_labels, all_failure_preds, average='macro')\n        confidence_mse = mean_squared_error(all_confidence_labels, all_confidence_preds)\n        \n        return {\n            'total_loss': total_loss / len(data_loader),\n            'macro_f1': macro_f1,\n            'confidence_mse': confidence_mse\n        }\n    \n    def save_model(self, model_path: str, vectorizer_path: str):\n        \"\"\"Save trained model and label encoder\"\"\"\n        # Save model state dict\n        torch.save({\n            'model_state_dict': self.model.state_dict(),\n            'config': self.config,\n            'num_classes': len(self.label_encoder.classes_)\n        }, model_path)\n        \n        # Save label encoder and tokenizer info\n        with open(vectorizer_path, 'wb') as f:\n            pickle.dump({\n                'label_encoder': self.label_encoder,\n                'tokenizer_name': self.config.model_name,\n                'max_length': self.config.max_length\n            }, f)\n        \n        logger.info(f\"Model saved to {model_path}\")\n        logger.info(f\"Vectorizer saved to {vectorizer_path}\")\n    \n    def load_model(self, model_path: str, vectorizer_path: str):\n        \"\"\"Load trained model and vectorizer\"\"\"\n        # Load vectorizer\n        with open(vectorizer_path, 'rb') as f:\n            vectorizer_data = pickle.load(f)\n            self.label_encoder = vectorizer_data['label_encoder']\n            self.tokenizer = AutoTokenizer.from_pretrained(vectorizer_data['tokenizer_name'])\n        \n        # Load model\n        checkpoint = torch.load(model_path, map_location=self.device)\n        self.config = checkpoint['config']\n        \n        self.model = ErrorConfidenceClassifier(\n            self.config,\n            checkpoint['num_classes']\n        ).to(self.device)\n        self.model.load_state_dict(checkpoint['model_state_dict'])\n        \n        logger.info(f\"Model loaded from {model_path}\")\n    \n    def predict(self, \n                initial_answer: str,\n                revised_answer: str,\n                reprompt_id: str,\n                logits_features: Optional[np.ndarray] = None) -> Dict:\n        \"\"\"Make prediction for a single example\"\"\"\n        self.model.eval()\n        \n        # Prepare input\n        input_text = f\"Initial: {initial_answer} [SEP] Revised: {revised_answer} [SEP] Prompt: {reprompt_id}\"\n        \n        encoding = self.tokenizer(\n            input_text,\n            truncation=True,\n            padding='max_length',\n            max_length=self.config.max_length,\n            return_tensors='pt'\n        )\n        \n        input_ids = encoding['input_ids'].to(self.device)\n        attention_mask = encoding['attention_mask'].to(self.device)\n        \n        logits_tensor = None\n        if logits_features is not None:\n            logits_tensor = torch.tensor(logits_features, dtype=torch.float).unsqueeze(0).to(self.device)\n        \n        with torch.no_grad():\n            outputs = self.model(\n                input_ids=input_ids,\n                attention_mask=attention_mask,\n                logits_features=logits_tensor\n            )\n        \n        # Get predictions\n        failure_mode_probs = F.softmax(outputs['failure_mode_logits'], dim=1)\n        failure_mode_pred = torch.argmax(failure_mode_probs, dim=1).item()\n        failure_mode_label = self.label_encoder.inverse_transform([failure_mode_pred])[0]\n        \n        confidence_score = outputs['confidence_score'].item()\n        \n        return {\n            'failure_mode': failure_mode_label,\n            'failure_mode_confidence': failure_mode_probs[0][failure_mode_pred].item(),\n            'confidence_score': confidence_score,\n            'all_failure_mode_probs': {\n                label: prob.item() \n                for label, prob in zip(self.label_encoder.classes_, failure_mode_probs[0])\n            }\n        }

# Example usage and training script
def create_sample_training_data(output_path: str = \"sample_training_data.csv\"):\n    \"\"\"Create sample training data for demonstration\"\"\"\n    \n    sample_data = [\n        {\n            'initial_answer': 'London',\n            'revised_answer': 'Paris', \n            'reprompt_id': 'neutral_verification',\n            'failure_mode': 'corrected',\n            'delta_accuracy': 1\n        },\n        {\n            'initial_answer': 'Paris',\n            'revised_answer': 'Lyon',\n            'reprompt_id': 'adversarial_challenge',\n            'failure_mode': 'overcorrected', \n            'delta_accuracy': -1\n        },\n        {\n            'initial_answer': 'Madrid',\n            'revised_answer': 'Madrid',\n            'reprompt_id': 'critique_revise_specific',\n            'failure_mode': 'anchored',\n            'delta_accuracy': 0\n        },\n        {\n            'initial_answer': 'Paris',\n            'revised_answer': 'Paris',\n            'reprompt_id': 'concise_confirmation',\n            'failure_mode': 'unchanged_correct',\n            'delta_accuracy': 0\n        },\n        {\n            'initial_answer': 'Berlin',\n            'revised_answer': 'Munich', \n            'reprompt_id': 'step_by_step_break_down',\n            'failure_mode': 'wavering',\n            'delta_accuracy': 0\n        },\n        {\n            'initial_answer': 'Paris',\n            'revised_answer': 'Paris, the capital city',\n            'reprompt_id': 'exhaustive_explanation',\n            'failure_mode': 'perfectionism',\n            'delta_accuracy': 0\n        }\n    ]\n    \n    # Replicate to create more training samples\n    expanded_data = []\n    for _ in range(100):  # Create 600 samples total\n        for item in sample_data:\n            expanded_data.append(item.copy())\n    \n    df = pd.DataFrame(expanded_data)\n    df.to_csv(output_path, index=False)\n    logger.info(f\"Created sample training data with {len(df)} samples at {output_path}\")\n    return output_path

if __name__ == \"__main__\":\n    # Setup logging\n    logging.basicConfig(level=logging.INFO)\n    \n    # Create sample data\n    data_path = create_sample_training_data()\n    \n    # Initialize trainer\n    config = TrainingConfig(\n        batch_size=8,  # Small batch for demo\n        num_epochs=3,  # Few epochs for demo\n        learning_rate=5e-5\n    )\n    \n    trainer = ErrorConfidenceTrainer(config)\n    \n    # Prepare data and train\n    train_loader, val_loader = trainer.prepare_data(data_path)\n    metrics = trainer.train(train_loader, val_loader)\n    \n    # Save model\n    trainer.save_model(\"error_confidence_model.pt\", \"vectorizer.pkl\")\n    \n    # Test prediction\n    prediction = trainer.predict(\n        initial_answer=\"London\",\n        revised_answer=\"Paris\",\n        reprompt_id=\"neutral_verification\"\n    )\n    print(\"Sample prediction:\", prediction)\n
