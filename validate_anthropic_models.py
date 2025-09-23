#!/usr/bin/env python3
"""
Anthropic Model Validation Utility

This script validates that all Anthropic model names in configurations are correct
and prevents experiments from running with invalid model names.

Usage:
    python3 validate_anthropic_models.py
    python3 validate_anthropic_models.py --test-api
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set

def load_model_config() -> Dict:
    """Load the definitive Anthropic model configuration."""
    config_path = Path("configs/anthropic_models.json")
    if not config_path.exists():
        raise FileNotFoundError("Anthropic model config not found: configs/anthropic_models.json")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_valid_model_names() -> Set[str]:
    """Get set of all valid Anthropic model names."""
    config = load_model_config()
    valid_names = set()
    
    for model_info in config["models"].values():
        valid_names.add(model_info["model_id"])
    
    return valid_names

def get_deprecated_models() -> Dict[str, str]:
    """Get mapping of deprecated/invalid model names to correct ones."""
    config = load_model_config()
    return config.get("deprecation_warnings", {})

def validate_config_files() -> Dict[str, List[str]]:
    """Validate all configuration files for correct Anthropic model names."""
    valid_names = get_valid_model_names()
    deprecated = get_deprecated_models()
    
    issues = {}
    config_files = list(Path("configs").rglob("*.json")) + list(Path("configs").rglob("*.yaml"))
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            file_issues = []
            
            # Check for deprecated/invalid model names
            for bad_name, suggestion in deprecated.items():
                if bad_name in content:
                    file_issues.append(f"Found invalid model name '{bad_name}' - {suggestion}")
            
            # Check for any model names that look like Claude models but aren't valid
            import re
            claude_pattern = r'claude-[^"\\s,}\\]]*'
            claude_matches = re.findall(claude_pattern, content)
            
            for match in claude_matches:
                if match not in valid_names and match not in deprecated:
                    file_issues.append(f"Unrecognized Claude model name: '{match}'")
            
            if file_issues:
                issues[str(config_file)] = file_issues
                
        except Exception as e:
            issues[str(config_file)] = [f"Error reading file: {e}"]
    
    return issues

def test_api_connectivity(model_names: List[str] = None) -> Dict[str, str]:
    """Test API connectivity for specified models."""
    if model_names is None:
        model_names = list(get_valid_model_names())
    
    results = {}
    
    try:
        import anthropic
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not found in environment"}
        
        client = anthropic.Anthropic(api_key=api_key)
        
        for model in model_names:
            try:
                # Test with minimal request
                response = client.messages.create(
                    model=model,
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                results[model] = "✅ WORKING"
                
            except anthropic.NotFoundError:
                results[model] = "❌ NOT_FOUND"
            except anthropic.AuthenticationError:
                results[model] = "🔑 AUTH_ERROR"
            except Exception as e:
                results[model] = f"⚠️ ERROR: {str(e)[:50]}"
                
    except ImportError:
        results["error"] = "anthropic package not installed"
    except Exception as e:
        results["error"] = f"API test failed: {e}"
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Validate Anthropic model configurations")
    parser.add_argument("--test-api", action="store_true", help="Test API connectivity")
    parser.add_argument("--fix-configs", action="store_true", help="Automatically fix config files")
    
    args = parser.parse_args()
    
    print("🔍 ANTHROPIC MODEL VALIDATION")
    print("=" * 50)
    
    try:
        # Load model configuration
        config = load_model_config()
        valid_models = get_valid_model_names()
        
        print(f"✅ Loaded model config with {len(valid_models)} valid models:")
        for model in sorted(valid_models):
            model_info = next(m for m in config["models"].values() if m["model_id"] == model)
            print(f"  - {model} ({model_info['name']})")
        
        print()
        
        # Validate configuration files
        print("🔍 Validating configuration files...")
        issues = validate_config_files()
        
        if issues:
            print(f"❌ Found issues in {len(issues)} files:")
            for file_path, file_issues in issues.items():
                print(f"\\n📄 {file_path}:")
                for issue in file_issues:
                    print(f"  • {issue}")
        else:
            print("✅ All configuration files are valid!")
        
        # Test API connectivity if requested
        if args.test_api:
            print("\\n🌐 Testing API connectivity...")
            api_results = test_api_connectivity()
            
            if "error" in api_results:
                print(f"❌ API test failed: {api_results['error']}")
            else:
                print("API Test Results:")
                for model, status in api_results.items():
                    print(f"  {status} {model}")
        
        # Summary
        print("\\n" + "=" * 50)
        if issues:
            print(f"❌ Validation failed: {len(issues)} files have issues")
            print("\\n🔧 To fix automatically, run:")
            print("find configs/ -name '*.json' -o -name '*.yaml' | xargs sed -i '' 's/claude-3-5-sonnet-20241210/claude-3-5-sonnet-20241022/g'")
            return 1
        else:
            print("✅ All Anthropic model configurations are valid!")
            return 0
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())