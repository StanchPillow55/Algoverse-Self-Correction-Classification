# Dataset Reproducibility Analysis: Fixing Random Sampling Issues

## 🚨 **Problem Identified**

The current system uses **random sampling** for dataset subsets, which means ensemble experiments and multi-turn experiments on the same model/dataset combination **do not use the same problems**, violating experimental validity.

---

## 🔍 **Current Issues**

### **Issue 1: Random Sampling in Dataset Loading**

**File: `src/data/scaling_datasets.py` (Line 512-514)**
```python
if sample_size and sample_size < len(samples):
    # Simple random sampling (in practice, you might want stratified sampling)
    import random
    samples = random.sample(samples, sample_size)  # ❌ RANDOM!
```

**Problem**: Each run gets different problems, making comparison invalid.

### **Issue 2: Subset Selection Without Seeding**

**HumanEval Loader: `src/data/humaneval_loader.py` (Lines 65-68)**
```python
if subset == "subset_20":
    return data[:20]  # ✅ Deterministic (first 20)
if subset == "subset_100": 
    return data[:100]  # ✅ Deterministic (first 100)
```

**GSM8K Subset Generation: `src/data/scaling_datasets.py` (Line 557)**
```python
subset_data["samples"] = samples[:size]  # ✅ Deterministic (first N)
```

**Analysis**: 
- ✅ **HumanEval and pre-generated subsets** are deterministic 
- ❌ **Dynamic sampling** during experiments is random

### **Issue 3: No Experiment-to-Experiment Consistency**

**Current behavior:**
```bash
# These two commands get DIFFERENT subsets!
python run_ensemble_experiments.py --dataset gsm8k --subset subset_100  # Random 100 problems
python src/main.py run --dataset gsm8k --subset subset_100              # Different random 100 problems
```

---

## 🎯 **Solution: Deterministic Dataset Handling**

### **Approach 1: Fix Random Sampling with Seeds**

**Replace in `src/data/scaling_datasets.py`:**
```python
def load_dataset(self, dataset_name: str, sample_size: int = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Load a dataset with optional deterministic sampling."""
    dataset_path = self.data_dir / f"{dataset_name}.json"
    
    if not dataset_path.exists():
        logger.error(f"Dataset {dataset_name} not found. Run download first.")
        return []
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    
    if sample_size and sample_size < len(samples):
        # FIXED: Deterministic sampling with seed
        import random
        random.seed(seed)  # 🔧 Fixed: Use deterministic seed
        samples = random.sample(samples, sample_size)
        # Restore random state to avoid affecting other code
        random.seed()
    
    return samples
```

### **Approach 2: Pre-generate All Required Subsets** (Recommended)

**Enhanced subset generation:**
```python
def create_deterministic_subsets(self, dataset_name: str, sizes: List[int] = None) -> Dict[str, str]:
    """Create deterministic subsets that are consistent across experiments."""
    if sizes is None:
        sizes = [20, 50, 100, 500, 1000]
    
    dataset_path = self.data_dir / f"{dataset_name}.json"
    
    if not dataset_path.exists():
        logger.error(f"Dataset {dataset_name} not found")
        return {}
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    created_files = {}
    
    # Create deterministic subsets using FIRST N samples (not random)
    for size in sizes:
        if size <= len(samples):
            subset_path = self.data_dir / f"{dataset_name}_deterministic_{size}.json"
            
            subset_data = data.copy()
            # DETERMINISTIC: Always use first N samples
            subset_data["samples"] = samples[:size]
            subset_data["sample_size"] = size
            subset_data["sampling_method"] = "deterministic_first_n"
            subset_data["created_date"] = datetime.now().isoformat()
            
            with open(subset_path, 'w') as f:
                json.dump(subset_data, f, indent=2)
            
            created_files[f"subset_{size}"] = str(subset_path)
            logger.info(f"Created deterministic {dataset_name} subset with {size} samples")
    
    return created_files
```

### **Approach 3: Content-Based Hashing for Subset Selection**

**For even more robustness:**
```python
def create_content_hash_subsets(self, dataset_name: str, sizes: List[int] = None) -> Dict[str, str]:
    """Create subsets based on content hash for ultimate reproducibility."""
    import hashlib
    
    # Sort samples by content hash to ensure deterministic ordering
    samples_with_hash = []
    for sample in samples:
        content = json.dumps(sample, sort_keys=True)
        hash_val = hashlib.md5(content.encode()).hexdigest()
        samples_with_hash.append((hash_val, sample))
    
    # Sort by hash for deterministic ordering
    samples_with_hash.sort(key=lambda x: x[0])
    sorted_samples = [sample for _, sample in samples_with_hash]
    
    # Create subsets from sorted samples
    for size in sizes:
        if size <= len(sorted_samples):
            subset_samples = sorted_samples[:size]
            # ... save subset
```

---

## 📊 **Verification Commands**

### **Test Current System:**
```bash
# Run this multiple times - you'll get different results each time!
python -c "
from src.data.scaling_datasets import ScalingDatasetManager
dm = ScalingDatasetManager()
data1 = dm.load_dataset('gsm8k', sample_size=10)
data2 = dm.load_dataset('gsm8k', sample_size=10) 
print('Same samples:', [d1['id'] for d1 in data1] == [d2['id'] for d2 in data2])
print('Sample 1 IDs:', [d['id'] for d in data1[:5]])
print('Sample 2 IDs:', [d['id'] for d in data2[:5]])
"
```

### **Test Fixed System:**
```bash
# With seed fix - should be identical every time
python -c "
from src.data.scaling_datasets import ScalingDatasetManager
dm = ScalingDatasetManager()
data1 = dm.load_dataset('gsm8k', sample_size=10, seed=42)
data2 = dm.load_dataset('gsm8k', sample_size=10, seed=42)
print('Same samples:', [d1['id'] for d1 in data1] == [d2['id'] for d2 in data2])
print('Sample 1 IDs:', [d['id'] for d in data1[:5]])
print('Sample 2 IDs:', [d['id'] for d in data2[:5]])
"
```

---

## 🔧 **Implementation Plan**

### **Step 1: Immediate Fix (Random Seed)**
1. Add `seed` parameter to `load_dataset()` methods
2. Set consistent seed (e.g., `42`) for all experimental runs
3. Update both ensemble and multi-turn runners to use the same seed

### **Step 2: Long-term Solution (Pre-generated Subsets)**
1. Create deterministic subset files using `subset_data["samples"] = samples[:size]` 
2. Update subset naming: `gsm8k_deterministic_100.json` instead of `gsm8k_sample_100.json`
3. Modify loaders to prefer pre-generated deterministic subsets

### **Step 3: Documentation and Validation**
1. Document subset generation methodology
2. Add checksum verification for subset consistency
3. Create validation scripts to verify ensemble/multi-turn use identical data

---

## ✅ **Expected Results After Fix**

### **Before (Current):**
```bash
# Ensemble run on GSM8K subset_100
Experiment ID: ensemble_gsm8k_gpt4o_20241217
Problems: [gsm8k_42, gsm8k_17, gsm8k_91, ...]  # Random selection

# Multi-turn run on GSM8K subset_100  
Experiment ID: multiturn_gsm8k_gpt4o_20241217
Problems: [gsm8k_8, gsm8k_156, gsm8k_73, ...]  # Different random selection
```

### **After (Fixed):**
```bash
# Ensemble run on GSM8K subset_100
Experiment ID: ensemble_gsm8k_gpt4o_20241217  
Problems: [gsm8k_0, gsm8k_1, gsm8k_2, ..., gsm8k_99]  # First 100, deterministic

# Multi-turn run on GSM8K subset_100
Experiment ID: multiturn_gsm8k_gpt4o_20241217
Problems: [gsm8k_0, gsm8k_1, gsm8k_2, ..., gsm8k_99]  # SAME first 100, deterministic
```

---

## 🧪 **Experimental Validity Restored**

With deterministic subsets, researchers can confidently compare:

- **Ensemble vs Multi-turn**: Same problems, different approaches
- **Model A vs Model B**: Same problems, different models  
- **Reproduction studies**: Exact same experimental conditions
- **Ablation studies**: Controlled variable changes

The fix ensures **scientific rigor** and **reproducible results** across all experimental conditions.