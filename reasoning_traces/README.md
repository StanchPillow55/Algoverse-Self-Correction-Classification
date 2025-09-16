# Reasoning Traces for Good Runs

This directory contains reasoning traces for the 'good' experimental runs.

## Structure

Each model-dataset combination has its own directory:
- `{model}_{dataset}/` - Contains individual problem files
- `problem_{qid}.txt` - Individual reasoning trace for each problem

## Format

Each problem file contains:
- Question text
- Reference answer (if available)
- Turn-by-turn reasoning traces
- Teacher feedback and bias information
- Template usage information

## Models and Datasets Processed

### llama-3-70b
- gsm8k
- superglue
- toolqa
- mathbench

### gpt-4
- gsm8k
- superglue
- toolqa
- mathbench

### gpt-4o
- gsm8k
- superglue
- toolqa
- mathbench

### gpt-4o-mini
- gsm8k
- superglue
- toolqa
- mathbench

### claude-haiku
- gsm8k
- superglue
- toolqa
- mathbench

### claude-sonnet
- gsm8k
- superglue
- toolqa
- mathbench

