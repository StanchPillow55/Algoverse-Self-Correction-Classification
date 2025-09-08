"""
Cost Tracking System for Scaling Study

Tracks API costs and token usage across experiments for the scaling study.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CostRecord:
    """Record of cost and token usage for a single API call."""
    timestamp: float
    model_name: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_per_1k_tokens: float
    total_cost: float
    experiment_id: str
    dataset_name: str
    sample_id: str
    turn_number: int

class CostTracker:
    """Tracks costs and token usage across experiments."""
    
    def __init__(self, output_dir: str = "outputs/cost_tracking"):
        """Initialize cost tracker."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.records: List[CostRecord] = []
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Model cost mapping (from configs/scaling_models.json)
        self.model_costs = {
            "gpt-4o-mini": 0.00015,
            "claude-haiku": 0.00025,
            "gpt-4o": 0.0025,
            "claude-sonnet": 0.003,
            "llama-70b": 0.0007,
            "gpt-4": 0.03,
            "claude-opus": 0.015
        }
    
    def record_usage(self, model_name: str, provider: str, 
                    input_tokens: int, output_tokens: int,
                    experiment_id: str, dataset_name: str, 
                    sample_id: str, turn_number: int = 0):
        """Record token usage and calculate cost."""
        total_tokens = input_tokens + output_tokens
        cost_per_1k = self.model_costs.get(model_name, 0.001)  # Default fallback
        total_cost = (total_tokens / 1000) * cost_per_1k
        
        record = CostRecord(
            timestamp=time.time(),
            model_name=model_name,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_per_1k_tokens=cost_per_1k,
            total_cost=total_cost,
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            sample_id=sample_id,
            turn_number=turn_number
        )
        
        self.records.append(record)
        self.total_cost += total_cost
        self.total_tokens += total_tokens
        
        logger.debug(f"Cost recorded: {model_name} - {total_tokens} tokens - ${total_cost:.4f}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary by model and experiment."""
        if not self.records:
            return {"total_cost": 0.0, "total_tokens": 0, "by_model": {}, "by_experiment": {}}
        
        summary = {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "by_model": {},
            "by_experiment": {},
            "by_dataset": {}
        }
        
        # Group by model
        for record in self.records:
            model = record.model_name
            if model not in summary["by_model"]:
                summary["by_model"][model] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "calls": 0,
                    "provider": record.provider
                }
            
            summary["by_model"][model]["total_cost"] += record.total_cost
            summary["by_model"][model]["total_tokens"] += record.total_tokens
            summary["by_model"][model]["calls"] += 1
        
        # Group by experiment
        for record in self.records:
            exp_id = record.experiment_id
            if exp_id not in summary["by_experiment"]:
                summary["by_experiment"][exp_id] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "calls": 0,
                    "dataset": record.dataset_name
                }
            
            summary["by_experiment"][exp_id]["total_cost"] += record.total_cost
            summary["by_experiment"][exp_id]["total_tokens"] += record.total_tokens
            summary["by_experiment"][exp_id]["calls"] += 1
        
        # Group by dataset
        for record in self.records:
            dataset = record.dataset_name
            if dataset not in summary["by_dataset"]:
                summary["by_dataset"][dataset] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "calls": 0
                }
            
            summary["by_dataset"][dataset]["total_cost"] += record.total_cost
            summary["by_dataset"][dataset]["total_tokens"] += record.total_tokens
            summary["by_dataset"][dataset]["calls"] += 1
        
        return summary
    
    def save_records(self, filename: str = None):
        """Save all records to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"cost_records_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        records_data = [asdict(record) for record in self.records]
        
        with open(filepath, 'w') as f:
            json.dump({
                "records": records_data,
                "summary": self.get_summary(),
                "metadata": {
                    "total_records": len(self.records),
                    "export_time": time.time(),
                    "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, f, indent=2)
        
        logger.info(f"Cost records saved to: {filepath}")
        return filepath
    
    def load_records(self, filename: str):
        """Load records from JSON file."""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            logger.error(f"Cost records file not found: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing records
        self.records = []
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Load records
        for record_data in data.get("records", []):
            record = CostRecord(**record_data)
            self.records.append(record)
            self.total_cost += record.total_cost
            self.total_tokens += record.total_tokens
        
        logger.info(f"Loaded {len(self.records)} cost records from {filepath}")
        return True
    
    def estimate_experiment_cost(self, models: List[str], datasets: List[str], 
                                sample_sizes: List[int]) -> Dict[str, Any]:
        """Estimate total cost for scaling experiments."""
        total_cost = 0.0
        breakdown = {}
        
        # Estimate tokens per sample (conservative estimate)
        tokens_per_sample = 200  # Input + output tokens per sample
        turns_per_sample = 3     # Average self-correction turns
        
        for model_name in models:
            model_cost = 0.0
            cost_per_1k = self.model_costs.get(model_name, 0.001)
            
            for dataset in datasets:
                for sample_size in sample_sizes:
                    total_tokens = sample_size * tokens_per_sample * turns_per_sample
                    cost = (total_tokens / 1000) * cost_per_1k
                    model_cost += cost
            
            breakdown[model_name] = model_cost
            total_cost += model_cost
        
        return {
            "total_cost": total_cost,
            "breakdown": breakdown,
            "models": models,
            "datasets": datasets,
            "sample_sizes": sample_sizes,
            "tokens_per_sample": tokens_per_sample,
            "turns_per_sample": turns_per_sample
        }
    
    def print_summary(self):
        """Print cost summary to console."""
        summary = self.get_summary()
        
        print("\n💰 Cost Tracking Summary")
        print("=" * 40)
        print(f"Total cost: ${summary['total_cost']:.2f}")
        print(f"Total tokens: {summary['total_tokens']:,}")
        print(f"Total calls: {len(self.records)}")
        
        print("\nBy Model:")
        for model, data in summary['by_model'].items():
            print(f"  {model:15} | ${data['total_cost']:6.2f} | {data['total_tokens']:6,} tokens | {data['calls']:3} calls")
        
        print("\nBy Experiment:")
        for exp_id, data in summary['by_experiment'].items():
            print(f"  {exp_id:20} | ${data['total_cost']:6.2f} | {data['total_tokens']:6,} tokens | {data['calls']:3} calls")
        
        print("\nBy Dataset:")
        for dataset, data in summary['by_dataset'].items():
            print(f"  {dataset:15} | ${data['total_cost']:6.2f} | {data['total_tokens']:6,} tokens | {data['calls']:3} calls")

# Global cost tracker instance
_global_cost_tracker = None

def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker()
    return _global_cost_tracker

def record_cost(model_name: str, provider: str, input_tokens: int, output_tokens: int,
                experiment_id: str, dataset_name: str, sample_id: str, turn_number: int = 0):
    """Convenience function to record cost using global tracker."""
    tracker = get_cost_tracker()
    tracker.record_usage(model_name, provider, input_tokens, output_tokens,
                        experiment_id, dataset_name, sample_id, turn_number)
