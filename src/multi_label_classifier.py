"""
Multi-Label Error Classifier for Self-Correction Failure Mode Detection

This module implements a multi-label classifier that outputs probabilities for
multiple simultaneous error types, rather than a single argmax prediction.

Key Changes from Original:
- Multi-label classification (multiple errors can co-occur)
- Output: probabilities for each error type
- Threshold-based error selection (e.g., >25% probability)
- Direct mapping from error types to specific reprompts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MultiLabelTrainingConfig:
    """Configuration for multi-label classifier training"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    hidden_dim: int = 256
    dropout: float = 0.1
    logits_feature_dim: int = 64
    weight_decay: float = 0.01
    threshold: float = 0.25  # Probability threshold for error selection

class MultiLabelSelfCorrectionDataset(Dataset):
    """Dataset for multi-label error classification"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 tokenizer,
                 label_binarizer: MultiLabelBinarizer,
                 max_length: int = 512,
                 logits_features: Optional[np.ndarray] = None):
        self.data = data
        self.tokenizer = tokenizer
        self.label_binarizer = label_binarizer
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
        
        # Get multi-label targets
        # Expect 'error_labels' column with list of error types or comma-separated string
        if 'error_labels' in row:
            if isinstance(row['error_labels'], str):
                error_labels = [label.strip() for label in row['error_labels'].split(',') if label.strip()]
            else:
                error_labels = row['error_labels'] if isinstance(row['error_labels'], list) else [row['error_labels']]
        else:
            # Fallback to single label
            error_labels = [row.get('failure_mode', 'unknown')]
        
        # Convert to binary vector
        label_vector = self.label_binarizer.transform([error_labels])[0]
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_vector, dtype=torch.float),
        }
        
        # Add logits features if available
        if self.logits_features is not None:
            item['logits_features'] = torch.tensor(
                self.logits_features[idx], dtype=torch.float
            )
        
        return item

class MultiLabelErrorClassifier(nn.Module):
    """
    Multi-label classifier for error detection with probability outputs.
    
    Architecture:
    1. Text encoder (lightweight transformer)
    2. Optional logits feature encoder
    3. Feature fusion layer
    4. Multi-label prediction head with sigmoid activation
    """
    
    def __init__(self, 
                 config: MultiLabelTrainingConfig,
                 num_error_types: int,
                 logits_feature_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_error_types = num_error_types
        
        # Text encoder
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
        
        # Multi-label prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, num_error_types)
            # Note: No sigmoid here - will be applied in forward pass
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
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Process logits features if available
        if self.use_logits_features and logits_features is not None:
            logits_encoded = self.logits_encoder(logits_features)
            fused_features = torch.cat([text_features, logits_encoded], dim=1)
        else:
            fused_features = text_features
        
        # Feature fusion
        fused_features = self.fusion_layer(fused_features)
        
        # Multi-label prediction with sigmoid
        logits = self.prediction_head(fused_features)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities
        }

class MultiLabelErrorTrainer:
    """Trainer for the multi-label error classifier"""
    
    def __init__(self, config: MultiLabelTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.label_binarizer = MultiLabelBinarizer()
        self.model = None
        
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare training/validation data for multi-label classification
        
        Args:
            data_path: Path to CSV file with columns:
                      initial_answer, revised_answer, reprompt_id, error_labels
                      
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        # Parse error labels
        all_error_labels = []
        for _, row in df.iterrows():
            if 'error_labels' in row:
                if isinstance(row['error_labels'], str):
                    labels = [label.strip() for label in row['error_labels'].split(',') if label.strip()]
                else:
                    labels = row['error_labels'] if isinstance(row['error_labels'], list) else [row['error_labels']]
            else:
                # Fallback to single label
                labels = [row.get('failure_mode', 'unknown')]
            all_error_labels.append(labels)
        
        # Fit label binarizer
        self.label_binarizer.fit(all_error_labels)
        num_classes = len(self.label_binarizer.classes_)
        logger.info(f"Found {num_classes} error types: {list(self.label_binarizer.classes_)}")
        
        # Add parsed labels to dataframe for easier handling
        df['parsed_labels'] = all_error_labels
        
        # Split data (stratification is complex for multi-label, so we'll do simple split)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation")
        
        # Check for logits features (optional)
        logits_features_train = None
        logits_features_val = None
        if 'logits_features' in df.columns:
            logits_features_train = self._parse_logits_features(train_df['logits_features'])
            logits_features_val = self._parse_logits_features(val_df['logits_features'])
            logger.info(f"Loaded logits features with dimension {logits_features_train.shape[1]}")
        
        # Create datasets
        train_dataset = MultiLabelSelfCorrectionDataset(
            train_df.reset_index(drop=True),
            self.tokenizer,
            self.label_binarizer,
            self.config.max_length,
            logits_features_train
        )
        
        val_dataset = MultiLabelSelfCorrectionDataset(
            val_df.reset_index(drop=True),
            self.tokenizer,
            self.label_binarizer,
            self.config.max_length,
            logits_features_val
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        logits_dim = logits_features_train.shape[1] if logits_features_train is not None else None
        self.model = MultiLabelErrorClassifier(
            self.config,
            num_classes,
            logits_dim
        ).to(self.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        return train_loader, val_loader
    
    def _parse_logits_features(self, logits_series) -> np.ndarray:
        """Parse logits features from DataFrame column"""
        features = []
        for item in logits_series:
            if isinstance(item, str):
                feature_vec = json.loads(item)
            else:
                feature_vec = item
            features.append(feature_vec)
        return np.array(features)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train the multi-label model
        
        Returns:
            Dictionary with training metrics
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Multi-label loss (Binary Cross Entropy)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        train_metrics = {'epochs': [], 'train_loss': [], 'val_loss': [], 'val_jaccard': []}
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    logits_features=batch.get('logits_features')
                )
                
                # Compute loss
                loss = criterion(outputs['logits'], batch['labels'])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Jaccard: {val_metrics['jaccard']:.4f}"
            )
            
            # Store metrics
            train_metrics['epochs'].append(epoch + 1)
            train_metrics['train_loss'].append(avg_train_loss)
            train_metrics['val_loss'].append(val_metrics['loss'])
            train_metrics['val_jaccard'].append(val_metrics['jaccard'])
        
        return train_metrics
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    logits_features=batch.get('logits_features')
                )
                
                # Compute loss
                loss = criterion(outputs['logits'], batch['labels'])
                total_loss += loss.item()
                
                # Collect predictions (threshold at 0.5 for metrics)
                predictions = (outputs['probabilities'] > 0.5).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        jaccard = jaccard_score(all_labels, all_predictions, average='samples', zero_division=0)
        hamming = hamming_loss(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(data_loader),
            'jaccard': jaccard,
            'hamming_loss': hamming
        }
    
    def predict(self, 
                initial_answer: str,
                revised_answer: str,
                reprompt_id: str,
                threshold: Optional[float] = None,
                logits_features: Optional[np.ndarray] = None) -> Dict:
        """
        Make multi-label prediction for a single example
        
        Args:
            initial_answer: Original answer
            revised_answer: Revised answer
            reprompt_id: Reprompt template used
            threshold: Probability threshold (default: config.threshold)
            logits_features: Optional logits features
            
        Returns:
            Dictionary with error probabilities and errors above threshold
        """
        if threshold is None:
            threshold = self.config.threshold
            
        self.model.eval()
        
        # Prepare input
        input_text = f"Initial: {initial_answer} [SEP] Revised: {revised_answer} [SEP] Prompt: {reprompt_id}"
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        logits_tensor = None
        if logits_features is not None:
            logits_tensor = torch.tensor(logits_features, dtype=torch.float).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_features=logits_tensor
            )
        
        # Get probabilities
        probabilities = outputs['probabilities'].cpu().numpy()[0]
        
        # Create probability dictionary
        error_probabilities = {}
        errors_above_threshold = []
        
        for i, error_type in enumerate(self.label_binarizer.classes_):
            prob = float(probabilities[i])
            error_probabilities[error_type] = prob
            
            if prob >= threshold:
                errors_above_threshold.append((error_type, prob))
        
        # Sort errors above threshold by probability (highest first)
        errors_above_threshold.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'error_probabilities': error_probabilities,
            'errors_above_threshold': errors_above_threshold,
            'threshold_used': threshold
        }
    
    def save_model(self, model_path: str, metadata_path: str):
        """Save trained model and metadata"""
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'num_classes': len(self.label_binarizer.classes_)
        }, model_path)
        
        # Save label binarizer and tokenizer info
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'label_binarizer': self.label_binarizer,
                'tokenizer_name': self.config.model_name,
                'max_length': self.config.max_length,
                'error_types': list(self.label_binarizer.classes_)
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str, metadata_path: str):
        """Load trained model and metadata"""
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.label_binarizer = metadata['label_binarizer']
            self.tokenizer = AutoTokenizer.from_pretrained(metadata['tokenizer_name'])
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        self.model = MultiLabelErrorClassifier(
            self.config,
            checkpoint['num_classes']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")

def create_sample_multi_label_data(output_path: str = "multi_label_training_data.csv"):
    """Create sample multi-label training data"""
    
    sample_data = [
        {
            'initial_answer': 'London',
            'revised_answer': 'Paris', 
            'reprompt_id': 'neutral_verification',
            'error_labels': 'corrected'  # Single label
        },
        {
            'initial_answer': 'Paris',
            'revised_answer': 'Lyon',
            'reprompt_id': 'adversarial_challenge',
            'error_labels': 'overcorrected,cognitive_overload'  # Multiple labels
        },
        {
            'initial_answer': 'Madrid',
            'revised_answer': 'Madrid',
            'reprompt_id': 'critique_revise_specific',
            'error_labels': 'anchored'
        },
        {
            'initial_answer': 'Berlin',
            'revised_answer': 'Munich',
            'reprompt_id': 'step_by_step_break_down',
            'error_labels': 'wavering,perfectionism'
        },
        {
            'initial_answer': 'Rome',
            'revised_answer': 'Rome, the eternal city',
            'reprompt_id': 'exhaustive_explanation',
            'error_labels': 'perfectionism,overcorrected'
        },
    ]
    
    # Replicate to create more training samples
    expanded_data = []
    for _ in range(200):  # Create 1000 samples total
        for item in sample_data:
            expanded_data.append(item.copy())
    
    df = pd.DataFrame(expanded_data)
    df.to_csv(output_path, index=False)
    logger.info(f"Created multi-label training data with {len(df)} samples at {output_path}")
    return output_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    data_path = create_sample_multi_label_data()
    
    # Initialize trainer
    config = MultiLabelTrainingConfig(
        batch_size=8,
        num_epochs=3,
        learning_rate=5e-5,
        threshold=0.25
    )
    
    trainer = MultiLabelErrorTrainer(config)
    
    # Prepare data and train
    train_loader, val_loader = trainer.prepare_data(data_path)
    metrics = trainer.train(train_loader, val_loader)
    
    # Save model
    trainer.save_model("models/multi_label_error_model.pt", "models/multi_label_metadata.pkl")
    
    # Test prediction
    prediction = trainer.predict(
        initial_answer="London",
        revised_answer="Paris",
        reprompt_id="neutral_verification",
        threshold=0.25
    )
    
    print("Sample prediction:")
    print(f"Error probabilities: {prediction['error_probabilities']}")
    print(f"Errors above threshold: {prediction['errors_above_threshold']}")
