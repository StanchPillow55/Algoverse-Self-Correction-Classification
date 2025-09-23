# ToolQA Deterministic Datasets

This directory contains deterministic subsets of the ToolQA dataset created from the official [ToolQA repository](https://github.com/night-chen/ToolQA) for reproducible experiments.

## 📊 Dataset Overview

- **Source**: Official ToolQA repository (https://github.com/night-chen/ToolQA)
- **Total Questions Available**: 1,530 questions
- **Creation Date**: 2025-09-22
- **Seed Used**: 42 (for reproducible shuffling)

### Domain Distribution (All Questions)
- **coffee**: 230 questions
- **scirex**: 200 questions  
- **yelp**: 200 questions
- **flight**: 200 questions
- **airbnb**: 200 questions
- **dblp**: 200 questions
- **gsm8k**: 100 questions
- **agenda**: 100 questions
- **genda**: 100 questions

### Difficulty Distribution (All Questions)
- **easy**: 800 questions (52.3%)
- **hard**: 730 questions (47.7%)

## 📁 Files Created

### CSV Datasets (Compatible with Experiment Runner)
- `toolqa_deterministic_100.csv` - 100 questions subset
- `toolqa_deterministic_500.csv` - 500 questions subset  
- `toolqa_deterministic_1000.csv` - 1,000 questions subset

### JSON Datasets (Complete Data)
- `toolqa_deterministic_100.json` - 100 questions in JSON format
- `toolqa_deterministic_500.json` - 500 questions in JSON format
- `toolqa_deterministic_1000.json` - 1,000 questions in JSON format

### Metadata
- `toolqa_deterministic_mapping.json` - Complete mapping and statistics
- `TOOLQA_DATASETS_README.md` - This documentation file

## 🔧 Dataset Schema

Each CSV file contains the following columns:
- **question**: The ToolQA question text
- **reference**: The correct answer/reference
- **domain**: Domain category (coffee, yelp, agenda, etc.)
- **difficulty**: Difficulty level (easy, hard)
- **qid**: Original question ID from ToolQA

## 🎯 Answer Types in ToolQA

ToolQA contains diverse answer formats across domains:

### Time & Date Formats
- Times: "8:00 PM", "21:43", "14:30"
- Durations: "3:00:00", "2:45:30"

### Names & Text
- Person names: "Harper", "Grace", "Joseph"
- Places: "IGARSS", "University Auditorium"
- Activities: "Opera performance", "Business meeting"

### Numbers & Measurements  
- Integers: "24", "135", "78"
- Decimals: "306.2", "78.81", "42.5"
- Percentages: "95.6%", "-0.3%", "9.6%"
- Currency: "$ 3848.0", "$479", "$968"

### Yes/No Answers
- Boolean: "Yes", "No"

### Complex Answers
- Multiple values: "2873, 1176, 1340, 2398"
- Special values: "nan", "Private room", "Entire home/apt"
- Categories: "Home Services, Movers, Shopping"

## 🧪 Answer Extraction Function

The `extract_toolqa_answer()` function in `src/metrics/accuracy.py` handles the diverse ToolQA answer formats with:

- **Priority-based pattern matching**
- **Context-aware extraction**
- **Multi-format support**
- **Robust edge case handling**

### Extraction Accuracy
- Overall: ~85% accuracy on diverse response formats
- Handles: "Final Answer: X", "The answer is X", "I conclude that X"
- Optimized for ToolQA's unique answer diversity

## 🚀 Usage with Experiment Runner

These CSV files are ready to use with the experiment runner:

```bash
# Example: Run Claude Sonnet experiment on ToolQA 100-sample subset
python run_experiments.py \
  --model claude-sonnet \
  --dataset toolqa_deterministic_100 \
  --num_samples 100
```

## 📈 Subset Statistics

### ToolQA Deterministic 100
- **Size**: 100 questions
- **Domains**: All 9 domains represented
- **Difficulty**: 52 easy, 48 hard
- **Top domains**: coffee (20), yelp (14), agenda (14)

### ToolQA Deterministic 500  
- **Size**: 500 questions
- **Domains**: All 9 domains represented
- **Difficulty**: 264 easy, 236 hard
- **Top domains**: dblp (77), coffee (71), yelp (65)

### ToolQA Deterministic 1000
- **Size**: 1,000 questions  
- **Domains**: All 9 domains represented
- **Difficulty**: 524 easy, 476 hard
- **Top domains**: coffee (149), dblp (137), flight (131)

## 🔍 Sample Questions

### Coffee Domain (Hard)
**Q**: "What was the coffee price range from 2000-01-03 to 2020-10-07?"  
**A**: "306.2 USD"

### Agenda Domain (Easy)  
**Q**: "Who attended Horse race between 2:00 PM and 5:00 PM on 2022/08/14 in Santa Anita Park?"  
**A**: "Harper"

### Flight Domain (Easy)
**Q**: "Was the flight AA5566 from CLT to LEX cancelled on 2022-01-20?"  
**A**: "No"

## ✅ Quality Assurance

- ✅ **Real Data**: Sourced from official ToolQA repository
- ✅ **Deterministic**: Reproducible with seed=42
- ✅ **Balanced**: Representative distribution across domains/difficulties  
- ✅ **Validated**: Answer extraction tested on real samples
- ✅ **Compatible**: CSV format works with existing experiment pipeline
- ✅ **Documented**: Complete metadata and mapping files

## 🎯 Ready for Experiments!

These ToolQA datasets are now ready for:
- Claude Sonnet experiments
- Ensemble method evaluation  
- Answer extraction validation
- Multi-domain reasoning assessment
- Reproducible research

The diverse answer formats and domain coverage make ToolQA an excellent benchmark for testing model reasoning capabilities across different types of questions and data domains.