#!/usr/bin/env python3
"""
Document model specifications, size bins, and availability checks
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_model_availability(model_name: str, provider: str) -> Dict[str, Any]:
    """Check if a model is available and working."""
    try:
        from src.agents.learner import LearnerBot
        
        # Create learner with the model
        learner = LearnerBot(provider=provider, model=model_name)
        
        # Test with a simple prompt
        test_prompt = "What is 2+2?"
        test_history = []
        
        # Try to get an answer
        answer, confidence = learner.answer(
            test_prompt, test_history, 
            experiment_id="availability_test", 
            dataset_name="test", 
            sample_id="test", 
            turn_number=0
        )
        
        return {
            "available": True,
            "test_answer": answer,
            "confidence": confidence,
            "error": None
        }
        
    except Exception as e:
        return {
            "available": False,
            "test_answer": None,
            "confidence": 0.0,
            "error": str(e)
        }

def get_model_specifications() -> Dict[str, Any]:
    """Get comprehensive model specifications."""
    
    # Load model configurations
    config_path = project_root / "configs" / "scaling_models.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    models = config.get("models", [])
    
    # Model size bins
    size_bins = {
        "small": {"min_params": 0, "max_params": 7, "description": "Small models (1-7B parameters)"},
        "medium": {"min_params": 8, "max_params": 70, "description": "Medium models (8-70B parameters)"},
        "large": {"min_params": 71, "max_params": 1000, "description": "Large models (70B+ parameters)"}
    }
    
    # Provider information
    providers = {
        "openai": {
            "name": "OpenAI",
            "website": "https://openai.com",
            "api_docs": "https://platform.openai.com/docs",
            "rate_limits": "60 RPM, 10,000 TPM",
            "authentication": "API Key"
        },
        "anthropic": {
            "name": "Anthropic",
            "website": "https://anthropic.com",
            "api_docs": "https://docs.anthropic.com",
            "rate_limits": "5 RPM, 10,000 TPM",
            "authentication": "API Key"
        },
        "replicate": {
            "name": "Replicate",
            "website": "https://replicate.com",
            "api_docs": "https://replicate.com/docs",
            "rate_limits": "Variable by model",
            "authentication": "API Token"
        }
    }
    
    # Enhanced model specifications
    enhanced_models = []
    
    for model in models:
        # Determine size bin
        params = model.get("params", 0)
        if params <= 7:
            size_bin = "small"
        elif params <= 70:
            size_bin = "medium"
        else:
            size_bin = "large"
        
        # Add additional specifications
        enhanced_model = {
            **model,
            "size_bin": size_bin,
            "size_bin_description": size_bins[size_bin]["description"],
            "provider_info": providers.get(model.get("provider", "unknown"), {}),
            "estimated_monthly_cost_1k_samples": model.get("cost_per_1k_tokens", 0) * 200 * 1000,  # Rough estimate
            "recommended_use_cases": get_recommended_use_cases(model.get("name", ""), size_bin),
            "limitations": get_model_limitations(model.get("name", ""), size_bin)
        }
        
        enhanced_models.append(enhanced_model)
    
    return {
        "models": enhanced_models,
        "size_bins": size_bins,
        "providers": providers,
        "total_models": len(enhanced_models),
        "models_by_size": {
            "small": [m for m in enhanced_models if m["size_bin"] == "small"],
            "medium": [m for m in enhanced_models if m["size_bin"] == "medium"],
            "large": [m for m in enhanced_models if m["size_bin"] == "large"]
        }
    }

def get_recommended_use_cases(model_name: str, size_bin: str) -> List[str]:
    """Get recommended use cases for a model."""
    use_cases = {
        "small": [
            "Quick prototyping and testing",
            "Cost-sensitive applications",
            "Simple reasoning tasks",
            "Educational purposes"
        ],
        "medium": [
            "Production applications",
            "Complex reasoning tasks",
            "Multi-step problem solving",
            "Balanced performance and cost"
        ],
        "large": [
            "Research and experimentation",
            "High-stakes applications",
            "Complex multi-modal tasks",
            "Maximum performance requirements"
        ]
    }
    
    # Model-specific recommendations
    model_specific = {
        "gpt-4o-mini": ["Rapid iteration", "Cost-effective development"],
        "claude-haiku": ["Fast responses", "Simple tasks"],
        "gpt-4o": ["Balanced performance", "General purpose"],
        "claude-sonnet": ["Complex reasoning", "Code generation"],
        "llama-70b": ["Open source alternative", "Custom fine-tuning"],
        "gpt-4": ["Maximum capability", "Research applications"],
        "claude-opus": ["Advanced reasoning", "Complex analysis"]
    }
    
    base_cases = use_cases.get(size_bin, [])
    specific_cases = model_specific.get(model_name, [])
    
    return base_cases + specific_cases

def get_model_limitations(model_name: str, size_bin: str) -> List[str]:
    """Get limitations for a model."""
    general_limitations = {
        "small": [
            "Limited reasoning capability",
            "May struggle with complex tasks",
            "Lower accuracy on specialized tasks"
        ],
        "medium": [
            "Moderate cost for high-volume usage",
            "May not handle all edge cases",
            "Balanced but not optimal performance"
        ],
        "large": [
            "High cost for production use",
            "Slower response times",
            "May be overkill for simple tasks"
        ]
    }
    
    # Model-specific limitations
    model_specific = {
        "gpt-4o-mini": ["Limited context window", "Basic reasoning only"],
        "claude-haiku": ["Fast but less accurate", "Limited complex reasoning"],
        "gpt-4o": ["Newer model, less tested", "May have unknown limitations"],
        "claude-sonnet": ["Higher cost than smaller models", "May be slow for simple tasks"],
        "llama-70b": ["Requires Replicate API", "May have availability issues"],
        "gpt-4": ["Very high cost", "Rate limits", "May be overkill"],
        "claude-opus": ["Highest cost", "May be slow", "Complex setup"]
    }
    
    base_limitations = general_limitations.get(size_bin, [])
    specific_limitations = model_specific.get(model_name, [])
    
    return base_limitations + specific_limitations

def check_all_models_availability() -> Dict[str, Any]:
    """Check availability of all configured models."""
    config_path = project_root / "configs" / "scaling_models.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    models = config.get("models", [])
    availability_results = {}
    
    print("🔍 Checking model availability...")
    
    for model in models:
        model_name = model.get("name", "unknown")
        provider = model.get("provider", "unknown")
        
        print(f"  Checking {model_name} ({provider})...")
        
        availability = check_model_availability(model_name, provider)
        availability_results[model_name] = {
            "provider": provider,
            "availability": availability
        }
        
        status = "✅" if availability["available"] else "❌"
        print(f"    {status} {model_name}: {availability['error'] or 'Available'}")
    
    return availability_results

def generate_model_documentation() -> Dict[str, Any]:
    """Generate comprehensive model documentation."""
    print("📚 Generating Model Documentation")
    print("=" * 40)
    
    # Get model specifications
    specs = get_model_specifications()
    print(f"✓ Loaded {specs['total_models']} model specifications")
    
    # Check availability
    availability = check_all_models_availability()
    print("✓ Checked model availability")
    
    # Combine specifications and availability
    documentation = {
        "metadata": {
            "generated_at": "2024-09-08T20:00:00Z",
            "total_models": specs["total_models"],
            "available_models": sum(1 for a in availability.values() if a["availability"]["available"]),
            "unavailable_models": sum(1 for a in availability.values() if not a["availability"]["available"])
        },
        "model_specifications": specs,
        "availability_status": availability,
        "recommendations": generate_model_recommendations(specs, availability)
    }
    
    return documentation

def generate_model_recommendations(specs: Dict, availability: Dict) -> Dict[str, Any]:
    """Generate model recommendations based on specifications and availability."""
    
    available_models = [name for name, info in availability.items() if info["availability"]["available"]]
    
    recommendations = {
        "by_use_case": {
            "cost_sensitive": [m for m in available_models if specs["models"][m]["size_bin"] == "small"],
            "balanced": [m for m in available_models if specs["models"][m]["size_bin"] == "medium"],
            "maximum_performance": [m for m in available_models if specs["models"][m]["size_bin"] == "large"]
        },
        "by_budget": {
            "low": [m for m in available_models if specs["models"][m]["cost_per_1k_tokens"] < 0.001],
            "medium": [m for m in available_models if 0.001 <= specs["models"][m]["cost_per_1k_tokens"] < 0.01],
            "high": [m for m in available_models if specs["models"][m]["cost_per_1k_tokens"] >= 0.01]
        },
        "by_provider": {
            "openai": [m for m in available_models if specs["models"][m]["provider"] == "openai"],
            "anthropic": [m for m in available_models if specs["models"][m]["provider"] == "anthropic"],
            "replicate": [m for m in available_models if specs["models"][m]["provider"] == "replicate"]
        }
    }
    
    return recommendations

def main():
    """Main function to generate model documentation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive model documentation')
    parser.add_argument('--output-file', default='outputs/model_documentation.json',
                       help='Output file for documentation')
    parser.add_argument('--check-availability', action='store_true',
                       help='Check model availability (requires API keys)')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json',
                       help='Output format')
    
    args = parser.parse_args()
    
    try:
        # Generate documentation
        if args.check_availability:
            documentation = generate_model_documentation()
        else:
            # Skip availability check
            specs = get_model_specifications()
            documentation = {
                "metadata": {
                    "generated_at": "2024-09-08T20:00:00Z",
                    "total_models": specs["total_models"],
                    "availability_checked": False
                },
                "model_specifications": specs,
                "recommendations": generate_model_recommendations(specs, {})
            }
        
        # Save documentation
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'json':
            with open(output_path, 'w') as f:
                json.dump(documentation, f, indent=2, default=str)
            print(f"✓ Documentation saved to: {output_path}")
        elif args.format == 'markdown':
            # Convert to markdown (simplified)
            md_content = f"""# Model Documentation

## Summary
- Total models: {documentation['metadata']['total_models']}
- Available models: {documentation['metadata'].get('available_models', 'Not checked')}
- Generated: {documentation['metadata']['generated_at']}

## Models by Size

### Small Models (1-7B parameters)
{chr(10).join(f"- {m['name']}: {m['description']}" for m in documentation['model_specifications']['models_by_size']['small'])}

### Medium Models (8-70B parameters)
{chr(10).join(f"- {m['name']}: {m['description']}" for m in documentation['model_specifications']['models_by_size']['medium'])}

### Large Models (70B+ parameters)
{chr(10).join(f"- {m['name']}: {m['description']}" for m in documentation['model_specifications']['models_by_size']['large'])}

## Recommendations

### By Use Case
- Cost Sensitive: {', '.join(documentation['recommendations']['by_use_case']['cost_sensitive'])}
- Balanced: {', '.join(documentation['recommendations']['by_use_case']['balanced'])}
- Maximum Performance: {', '.join(documentation['recommendations']['by_use_case']['maximum_performance'])}

### By Budget
- Low: {', '.join(documentation['recommendations']['by_budget']['low'])}
- Medium: {', '.join(documentation['recommendations']['by_budget']['medium'])}
- High: {', '.join(documentation['recommendations']['by_budget']['high'])}
"""
            with open(output_path, 'w') as f:
                f.write(md_content)
            print(f"✓ Markdown documentation saved to: {output_path}")
        
        print("✅ Model documentation complete!")
        
    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
