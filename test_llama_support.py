#!/usr/bin/env python3
"""
Comprehensive Test for Llama Model Support

This script validates that the repository has proper support for Llama models
according to the scaling results table, including:
1. Model registry configuration
2. Provider implementation (Replicate API)
3. LearnerBot integration
4. Cost estimation
5. Missing model variants verification
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scaling.model_registry import MODEL_REGISTRY, get_model_config, get_models_by_category
from agents.learner import LearnerBot

class LlamaModelValidator:
    def __init__(self):
        self.results = {}
        self.issues = []
        self.test_timestamp = datetime.now().isoformat()
        
    def validate_table_requirements(self):
        """Validate models according to the scaling results table"""
        print("="*80)
        print("VALIDATING LLAMA SUPPORT ACCORDING TO SCALING TABLE")
        print("="*80)
        
        # Expected models from the table
        expected_models = {
            "Small (1–7B)": ["GPT-4o-mini", "Claude Haiku"],
            "Medium (8–70B)": ["GPT-4o", "Claude Sonnet", "Llama-70B"],
            "Large (100B+)": ["GPT-4", "Claude Opus"]
        }
        
        print("\nTable 1 — Scaling Results by Model Size Category")
        print("Size Category          Models                           Avg Δ        Cost per Sample  Cost-Benefit Ratio")
        print("Small (1–7B)          GPT-4o-mini, Claude Haiku       0.12 ± 0.03  $0.0003          400")
        print("Medium (8–70B)        GPT-4o, Claude Sonnet, Llama-70B 0.18 ± 0.04  $0.002           90")
        print("Large (100B+)         GPT-4, Claude Opus              0.22 ± 0.05  $0.015           15")
        
        # Check current registry
        current_registry = {}
        for category in ["Small", "Medium", "Large"]:
            models = get_models_by_category(category)
            current_registry[category] = [model.name for model in models]
            
        print(f"\n\nCURRENT REGISTRY STATUS:")
        print("-"*50)
        
        all_valid = True
        for category, expected in expected_models.items():
            registry_category = category.split(" ")[0]  # Extract "Small", "Medium", "Large"
            current = current_registry.get(registry_category, [])
            
            print(f"\n{category}:")
            print(f"  Expected: {', '.join(expected)}")
            print(f"  Current:  {', '.join(current)}")
            
            missing = []
            for model in expected:
                if not any(model in curr for curr in current):
                    missing.append(model)
            
            if missing:
                print(f"  ❌ MISSING: {', '.join(missing)}")
                self.issues.append(f"Missing models in {category}: {', '.join(missing)}")
                all_valid = False
            else:
                print(f"  ✅ COMPLETE")
                
        self.results['table_validation'] = {
            'valid': all_valid,
            'expected_models': expected_models,
            'current_registry': current_registry,
            'issues': [issue for issue in self.issues if 'Missing models' in issue]
        }
        
        return all_valid

    def validate_llama_registry_entries(self):
        """Validate Llama model registry entries"""
        print("\n\nVALIDATING LLAMA REGISTRY ENTRIES")
        print("-"*50)
        
        llama_models = []
        for key, config in MODEL_REGISTRY.items():
            if 'llama' in config.name.lower() or 'llama' in key.lower():
                llama_models.append((key, config))
        
        print(f"Found {len(llama_models)} Llama models in registry:")
        
        valid_entries = True
        for key, config in llama_models:
            print(f"\n{key}:")
            print(f"  Name: {config.name}")
            print(f"  Provider: {config.provider}")
            print(f"  Parameter Count: {config.parameter_count_b}B")
            print(f"  API Model: {config.api_model_name}")
            print(f"  Cost per 1K tokens: ${config.cost_per_1k_tokens}")
            print(f"  Size Category: {config.size_category}")
            
            # Validate entry
            if config.provider != "replicate":
                print(f"  ⚠️  WARNING: Provider is '{config.provider}', expected 'replicate'")
                self.issues.append(f"Llama model {key} uses provider '{config.provider}' instead of 'replicate'")
            else:
                print(f"  ✅ Provider: replicate")
                
            if not config.api_model_name.startswith("meta/llama"):
                print(f"  ⚠️  WARNING: API model name doesn't follow meta/llama pattern")
                self.issues.append(f"Llama model {key} API name doesn't follow expected pattern")
            else:
                print(f"  ✅ API Model: Valid pattern")
        
        if not llama_models:
            print("❌ No Llama models found in registry!")
            self.issues.append("No Llama models found in registry")
            valid_entries = False
            
        self.results['llama_registry'] = {
            'valid': valid_entries,
            'llama_models': [(key, config.name, config.provider, config.api_model_name) for key, config in llama_models],
            'count': len(llama_models)
        }
        
        return valid_entries

    def validate_provider_implementation(self):
        """Validate Replicate provider implementation in LearnerBot"""
        print("\n\nVALIDATING REPLICATE PROVIDER IMPLEMENTATION")
        print("-"*50)
        
        try:
            # Test LearnerBot initialization with replicate provider
            learner = LearnerBot(provider="replicate", model="meta/llama-2-70b-chat")
            print("✅ LearnerBot initialization with replicate provider: SUCCESS")
            
            # Check if replicate provider is handled in the answer method
            import inspect
            answer_source = inspect.getsource(learner.answer)
            
            if 'replicate' in answer_source:
                print("✅ Replicate provider handling in answer method: FOUND")
            else:
                print("❌ Replicate provider handling in answer method: NOT FOUND")
                self.issues.append("Replicate provider not handled in LearnerBot.answer method")
                return False
                
            # Check if _call_replicate method exists
            if hasattr(learner, '_call_replicate'):
                print("✅ _call_replicate method: EXISTS")
                
                # Get source of _call_replicate method
                replicate_source = inspect.getsource(learner._call_replicate)
                
                required_components = [
                    ('import replicate', 'Replicate library import'),
                    ('replicate.run', 'Replicate API call'),
                    ('self.model', 'Model specification'),
                    ('prompt', 'Prompt handling'),
                    ('max_tokens', 'Token limit configuration'),
                    ('temperature', 'Temperature configuration')
                ]
                
                for component, description in required_components:
                    if component in replicate_source:
                        print(f"  ✅ {description}: FOUND")
                    else:
                        print(f"  ❌ {description}: MISSING")
                        self.issues.append(f"Replicate implementation missing: {description}")
                        return False
                        
            else:
                print("❌ _call_replicate method: NOT FOUND")
                self.issues.append("_call_replicate method not implemented")
                return False
                
        except Exception as e:
            print(f"❌ Error testing LearnerBot with replicate: {e}")
            self.issues.append(f"LearnerBot replicate initialization failed: {e}")
            return False
        
        self.results['provider_implementation'] = {
            'valid': len([issue for issue in self.issues if 'Replicate' in issue or 'replicate' in issue]) == 0,
            'learner_init': True,
            'method_exists': True,
            'components_valid': True
        }
        
        return True

    def validate_api_dependencies(self):
        """Validate that required API dependencies are available"""
        print("\n\nVALIDATING API DEPENDENCIES")
        print("-"*50)
        
        dependencies = [
            ('replicate', 'Replicate API client'),
            ('anthropic', 'Anthropic API client (for comparison)'),
            ('openai', 'OpenAI API client (for comparison)')
        ]
        
        available_deps = []
        missing_deps = []
        
        for dep, description in dependencies:
            try:
                __import__(dep)
                print(f"✅ {description}: AVAILABLE")
                available_deps.append(dep)
            except ImportError:
                print(f"❌ {description}: MISSING")
                missing_deps.append(dep)
        
        if 'replicate' in missing_deps:
            print("\n⚠️  WARNING: Replicate library is required for Llama model support")
            print("   Install with: pip install replicate")
            self.issues.append("Replicate library not installed")
        
        self.results['api_dependencies'] = {
            'available': available_deps,
            'missing': missing_deps,
            'replicate_available': 'replicate' in available_deps
        }
        
        return 'replicate' in available_deps

    def test_model_instantiation(self):
        """Test actual model instantiation"""
        print("\n\nTESTING MODEL INSTANTIATION")
        print("-"*50)
        
        test_results = {}
        
        # Test Llama-70B configuration
        llama_config = get_model_config("llama-70b")
        if llama_config:
            print(f"Testing {llama_config.name} ({llama_config.api_model_name})...")
            
            try:
                learner = LearnerBot(provider=llama_config.provider, model=llama_config.api_model_name)
                print(f"  ✅ Instantiation: SUCCESS")
                print(f"  ✅ Provider: {learner.provider}")
                print(f"  ✅ Model: {learner.model}")
                
                test_results['llama-70b'] = {
                    'instantiation': True,
                    'provider': learner.provider,
                    'model': learner.model
                }
                
            except Exception as e:
                print(f"  ❌ Instantiation failed: {e}")
                test_results['llama-70b'] = {
                    'instantiation': False,
                    'error': str(e)
                }
                self.issues.append(f"Llama-70B instantiation failed: {e}")
                
        else:
            print("❌ Llama-70B config not found in registry")
            test_results['llama-70b'] = {'instantiation': False, 'error': 'Config not found'}
            
        self.results['model_instantiation'] = test_results
        
        return len([r for r in test_results.values() if r.get('instantiation', False)]) > 0

    def check_missing_llama_variants(self):
        """Check for missing Llama variants that might be needed"""
        print("\n\nCHECKING FOR MISSING LLAMA VARIANTS")
        print("-"*50)
        
        # Current Llama models in registry
        current_llama = []
        for key, config in MODEL_REGISTRY.items():
            if 'llama' in config.name.lower():
                current_llama.append((key, config.name, config.parameter_count_b))
        
        # Common Llama variants that might be useful
        suggested_variants = [
            ("llama-7b", "Llama-7B", 7.0, "Small", "meta/llama-2-7b-chat"),
            ("llama-13b", "Llama-13B", 13.0, "Medium", "meta/llama-2-13b-chat"),
            ("llama-3-8b", "Llama-3-8B", 8.0, "Medium", "meta/llama-3-8b-instruct"),
            ("llama-3-70b", "Llama-3-70B", 70.0, "Medium", "meta/llama-3-70b-instruct"),
        ]
        
        print("Current Llama models:")
        for key, name, params in current_llama:
            print(f"  {key}: {name} ({params}B params)")
        
        print("\nSuggested additional variants:")
        for key, name, params, category, api_name in suggested_variants:
            if not any(existing_key == key for existing_key, _, _ in current_llama):
                print(f"  {key}: {name} ({params}B params) - {category} - {api_name}")
        
        # Check if we need small Llama models to meet table requirements
        has_small_llama = any(params < 8.0 for _, _, params in current_llama)
        if not has_small_llama:
            print("\n⚠️  RECOMMENDATION: Consider adding a small Llama model (7B) to the Small category")
            print("   This would provide better coverage of the scaling spectrum")
            
        self.results['llama_variants'] = {
            'current': current_llama,
            'suggested': suggested_variants,
            'has_small_llama': has_small_llama
        }

    def validate_cost_estimation(self):
        """Validate cost estimation for Llama models"""
        print("\n\nVALIDATING COST ESTIMATION")
        print("-"*50)
        
        from scaling.model_registry import estimate_experiment_cost
        
        try:
            cost_info = estimate_experiment_cost("llama-70b", 100, 2000, 3)
            print("✅ Cost estimation for Llama-70B:")
            print(f"  Model: {cost_info['model']}")
            print(f"  Parameter Count: {cost_info['parameter_count_b']}B")
            print(f"  Total Cost: ${cost_info['total_cost_usd']:.4f}")
            print(f"  Cost per Sample: ${cost_info['cost_per_sample']:.6f}")
            
            # Validate against table expectations
            expected_cost_per_sample = 0.002  # From the table
            actual_cost = cost_info['cost_per_sample']
            
            if abs(actual_cost - expected_cost_per_sample) < 0.001:
                print(f"  ✅ Cost matches table expectation (~${expected_cost_per_sample})")
            else:
                print(f"  ⚠️  Cost difference from table: expected ~${expected_cost_per_sample}, got ${actual_cost:.6f}")
            
            self.results['cost_estimation'] = {
                'valid': True,
                'cost_info': cost_info,
                'matches_table': abs(actual_cost - expected_cost_per_sample) < 0.001
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Cost estimation failed: {e}")
            self.issues.append(f"Cost estimation failed for Llama-70B: {e}")
            self.results['cost_estimation'] = {'valid': False, 'error': str(e)}
            return False

    def run_comprehensive_validation(self):
        """Run all validation tests"""
        print("COMPREHENSIVE LLAMA MODEL SUPPORT VALIDATION")
        print("Started:", self.test_timestamp)
        print("="*80)
        
        tests = [
            ("Table Requirements", self.validate_table_requirements),
            ("Llama Registry Entries", self.validate_llama_registry_entries),
            ("Provider Implementation", self.validate_provider_implementation),
            ("API Dependencies", self.validate_api_dependencies),
            ("Model Instantiation", self.test_model_instantiation),
            ("Cost Estimation", self.validate_cost_estimation),
            ("Missing Variants Check", self.check_missing_llama_variants),
        ]
        
        results = {}
        all_passed = True
        
        for test_name, test_func in tests:
            try:
                if test_name == "Missing Variants Check":
                    test_func()  # This test doesn't return a boolean
                    results[test_name] = True
                else:
                    result = test_func()
                    results[test_name] = result
                    if not result:
                        all_passed = False
            except Exception as e:
                print(f"\n❌ Test '{test_name}' failed with exception: {e}")
                results[test_name] = False
                self.issues.append(f"Test '{test_name}' failed: {e}")
                all_passed = False
        
        # Summary
        print("\n\nVALIDATION SUMMARY")
        print("="*80)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        print(f"Tests Passed: {passed}/{total}")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name:<25} {status}")
        
        if self.issues:
            print(f"\nIssues Found ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        print(f"\nOverall Status: {'✅ PASS' if all_passed else '❌ FAIL'}")
        
        # Save results
        self.results['validation_summary'] = {
            'overall_pass': all_passed,
            'tests_passed': passed,
            'tests_total': total,
            'issues_count': len(self.issues),
            'issues': self.issues,
            'timestamp': self.test_timestamp
        }
        
        return all_passed

    def save_results(self, output_file: str = "llama_validation_results.json"):
        """Save validation results to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    validator = LlamaModelValidator()
    success = validator.run_comprehensive_validation()
    validator.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)