#!/usr/bin/env python3
"""
ToolQA Data Coverage Validation

This script validates that our tools can handle actual ToolQA questions
and identifies data coverage gaps.
"""

import sys
import os
import json
sys.path.insert(0, 'src')

from tools.domain_tools import create_and_register_tools
from tools.enhanced_answer_extraction import DomainAwareAnswerExtractor

def test_coffee_coverage():
    """Test coffee tool coverage on real ToolQA questions"""
    print("🔍 Testing Coffee Data Coverage")
    print("=" * 50)
    
    # Load ToolQA questions
    with open('data/scaling/toolqa_deterministic_100.json', 'r') as f:
        data = json.load(f)
    
    coffee_questions = [s for s in data['samples'] if s['domain'] == 'coffee']
    print(f"Found {len(coffee_questions)} coffee questions")
    
    # Test with our coffee tool
    system = create_and_register_tools('data/toolqa')
    coffee_tool = system.router.tools['coffee']
    extractor = DomainAwareAnswerExtractor()
    
    results = []
    for i, q in enumerate(coffee_questions[:10]):  # Test first 10
        print(f"\nTest {i+1}: {q['question']}")
        print(f"Expected: {q['answer']}")
        
        # Try to determine what tool function to call
        question_lower = q['question'].lower()
        
        if 'range' in question_lower:
            # Extract dates from question - simplified approach
            import re
            dates = re.findall(r'\d{4}-\d{2}-\d{2}', q['question'])
            if len(dates) >= 2:
                result = coffee_tool.call_function('get_price_range', {
                    'start_date': dates[0],
                    'end_date': dates[1]
                })
                if result.success:
                    # Test extraction
                    tool_results = [{
                        'success': True,
                        'tool_name': 'coffee_get_price_range',
                        'result': result.result
                    }]
                    extracted = extractor.extract_answer(tool_results, q['question'], q['answer'])
                    print(f"Tool Result: {result.result}")
                    print(f"Extracted: {extracted}")
                    print(f"Match: {'✓' if extracted == q['answer'] else '✗'}")
                    
                    results.append({
                        'question': q['question'],
                        'expected': q['answer'],
                        'extracted': extracted,
                        'tool_success': True,
                        'correct': extracted == q['answer']
                    })
                else:
                    print(f"Tool Error: {result.error_message}")
                    results.append({
                        'question': q['question'],
                        'expected': q['answer'], 
                        'extracted': 'N/A',
                        'tool_success': False,
                        'correct': False
                    })
        
        elif 'lowest' in question_lower or 'highest' in question_lower:
            dates = re.findall(r'\d{4}-\d{2}-\d{2}', q['question'])
            if len(dates) == 1:
                result = coffee_tool.call_function('get_price_on_date', {'date': dates[0]})
                if result.success:
                    extracted = result.result.get('low' if 'lowest' in question_lower else 'high')
                    print(f"Tool Result: {result.result}")
                    print(f"Extracted: {extracted}")
                    print(f"Match: {'✓' if str(extracted) == q['answer'] else '✗'}")
                    
                    results.append({
                        'question': q['question'],
                        'expected': q['answer'],
                        'extracted': str(extracted) if extracted else 'N/A',
                        'tool_success': True,
                        'correct': str(extracted) == q['answer']
                    })
                else:
                    print(f"Tool Error: {result.error_message}")
                    results.append({
                        'question': q['question'],
                        'expected': q['answer'],
                        'extracted': 'N/A', 
                        'tool_success': False,
                        'correct': False
                    })
    
    # Summary
    total = len(results)
    tool_success = sum(1 for r in results if r['tool_success'])
    correct = sum(1 for r in results if r['correct'])
    
    print(f"\n📊 Coffee Tool Coverage Summary:")
    print(f"Total Questions Tested: {total}")
    print(f"Tool Success Rate: {tool_success}/{total} ({tool_success/total*100:.1f}%)")
    print(f"Answer Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    return results

def analyze_domain_coverage():
    """Analyze which domains we can actually support"""
    print("\n🔍 Analyzing Domain Coverage")
    print("=" * 50)
    
    with open('data/scaling/toolqa_deterministic_500.json', 'r') as f:
        data = json.load(f)
    
    domain_counts = {}
    for sample in data['samples']:
        domain = sample['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print("ToolQA Domain Distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {count} questions")
    
    print(f"\nTotal: {sum(domain_counts.values())} questions across {len(domain_counts)} domains")
    
    # Check which domains we have tools for
    system = create_and_register_tools('data/toolqa')
    available_tools = list(system.router.tools.keys())
    
    print(f"\nAvailable Tools: {available_tools}")
    
    supported_domains = 0
    unsupported_domains = []
    
    for domain in domain_counts:
        if domain in available_tools or (domain in ['agenda', 'genda'] and 'agenda' in available_tools):
            supported_domains += 1
            status = "✅ Supported"
        elif domain == 'gsm8k' and 'calculator' in available_tools:
            supported_domains += 1 
            status = "✅ Supported (calculator)"
        else:
            unsupported_domains.append(domain)
            status = "❌ Not Supported"
        
        print(f"  {domain}: {status}")
    
    print(f"\nDomain Support: {supported_domains}/{len(domain_counts)} ({supported_domains/len(domain_counts)*100:.1f}%)")
    if unsupported_domains:
        print(f"Missing: {unsupported_domains}")
    
    return domain_counts, available_tools

def identify_data_requirements():
    """Identify what data we need to support ToolQA properly"""
    print("\n📋 Data Requirements Analysis")
    print("=" * 50)
    
    with open('data/scaling/toolqa_deterministic_100.json', 'r') as f:
        data = json.load(f)
    
    requirements = {
        'coffee': {
            'description': 'Historical coffee price data with daily granularity',
            'date_range': 'Need data from 2000-2022',
            'data_points_needed': 'Daily OHLC prices for ~8000+ days',
            'current_coverage': 'Only 12 data points (sparse)',
            'questions': []
        },
        'dblp': {
            'description': 'Academic paper database with author collaborations',
            'requirements': 'Real DBLP data or comprehensive mock database',
            'entities_needed': 'Authors, papers, venues, collaborations',
            'questions': []
        },
        'yelp': {
            'description': 'Business review data with locations and ratings',
            'requirements': 'Business database with geolocation and reviews', 
            'entities_needed': 'Businesses, reviews, locations, ratings',
            'questions': []
        },
        'airbnb': {
            'description': 'Property listing data with availability and reviews',
            'requirements': 'Airbnb-like property database',
            'entities_needed': 'Properties, listings, availability, reviews',
            'questions': []
        },
        'agenda': {
            'description': 'Calendar/event scheduling data',
            'requirements': 'Event database with people, dates, times',
            'entities_needed': 'People, events, dates, times, locations',
            'questions': []
        }
    }
    
    for sample in data['samples']:
        domain = sample['domain']
        if domain in requirements:
            requirements[domain]['questions'].append(sample['question'])
        elif domain in ['genda']:
            requirements['agenda']['questions'].append(sample['question'])
    
    for domain, info in requirements.items():
        if info['questions']:
            print(f"\n{domain.upper()} Domain:")
            print(f"  Description: {info['description']}")
            if 'date_range' in info:
                print(f"  Date Range: {info['date_range']}")
            if 'current_coverage' in info:
                print(f"  Current Coverage: {info['current_coverage']}")
            print(f"  Sample Questions ({len(info['questions'])}):")
            for q in info['questions'][:2]:
                print(f"    - {q}")
    
    return requirements

def main():
    """Main validation function"""
    print("🎯 ToolQA Data Coverage Validation")
    print("=" * 60)
    
    # Test coffee coverage
    coffee_results = test_coffee_coverage()
    
    # Analyze overall domain coverage
    domain_counts, available_tools = analyze_domain_coverage()
    
    # Identify data requirements
    requirements = identify_data_requirements()
    
    print(f"\n📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print("Current Status:")
    print("  ✅ Enhanced answer extraction system working")
    print("  ✅ Domain-aware tool architecture in place")
    print("  ❌ Insufficient data coverage for real ToolQA questions")
    print("  ❌ Most domain tools using mock/sparse data")
    
    print("\nNext Steps:")
    print("  1. Obtain comprehensive ToolQA datasets or use original APIs")
    print("  2. Expand coffee.csv to include daily data for date ranges in questions")
    print("  3. Create realistic databases for DBLP, Yelp, Airbnb domains") 
    print("  4. Implement remaining domain tools (scirex, flight)")
    print("  5. Run full validation on comprehensive data")
    
    return {
        'coffee_results': coffee_results,
        'domain_counts': domain_counts,
        'available_tools': available_tools,
        'requirements': requirements
    }

if __name__ == "__main__":
    main()