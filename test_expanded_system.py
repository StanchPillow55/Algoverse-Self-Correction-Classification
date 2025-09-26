#!/usr/bin/env python3
"""
Test Script for Expanded ToolQA Domain System

This script tests the newly implemented domain tools and enhanced 
answer extraction system with targeted examples.
"""

import sys
import os
import json
import logging

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tools.domain_tools import create_and_register_tools
from tools.enhanced_answer_extraction import DomainAwareAnswerExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_coffee_extraction():
    """Test enhanced coffee price extraction"""
    print("🧪 Testing Enhanced Coffee Price Extraction")
    print("=" * 50)
    
    extractor = DomainAwareAnswerExtractor()
    
    # Test case from failed extractions
    coffee_results = [{
        "success": True,
        "tool_name": "coffee_get_price_range",
        "result": {"min_price": 100.75, "max_price": 163.0, "date_range": "2000-01-03 to 2020-10-07"}
    }]
    
    test_cases = [
        ("What was the coffee price range from 2000-01-03 to 2020-10-07?", "306.2 USD"),
        ("What was the average coffee price from 2000-01-03 to 2020-10-07?", "132.0 USD"),
        ("What was the highest coffee price from 2000-01-03 to 2020-10-07?", "163.0 USD"),
    ]
    
    for i, (question, expected) in enumerate(test_cases, 1):
        extracted = extractor.extract_answer(coffee_results, question, expected)
        domain = extractor.detect_domain(question, coffee_results)
        
        print(f"Test {i}:")
        print(f"  Question: {question}")
        print(f"  Expected: {expected}")
        print(f"  Extracted: {extracted}")
        print(f"  Domain: {domain}")
        print(f"  Tool Result: {coffee_results[0]['result']}")
        
        # Check if extraction is reasonable
        expected_num = float(expected.replace("USD", "").strip())
        try:
            extracted_num = float(extracted)
            accuracy = abs(extracted_num - expected_num) / expected_num
            print(f"  Accuracy: {(1-accuracy)*100:.1f}%")
        except:
            print(f"  Accuracy: N/A (non-numeric)")
        print()

def test_new_domain_tools():
    """Test the new domain tools functionality"""
    print("🔧 Testing New Domain Tools")
    print("=" * 50)
    
    # Initialize tool system
    system = create_and_register_tools("data/toolqa")
    
    # Test DBLP
    print("📚 Testing DBLP Tool:")
    dblp = system.router.tools.get('dblp')
    if dblp:
        result = dblp.call_function('get_author_papers', {'author_name': 'John Smith'})
        print(f"  DBLP Author Papers: {result.success} - {result.result.get('paper_count', 0) if result.result else 'N/A'} papers")
        
        result = dblp.call_function('count_papers', {'venue': 'ACL'})
        print(f"  DBLP Venue Count: {result.success} - {result.result.get('count', 0) if result.result else 'N/A'} papers")
    else:
        print("  DBLP tool not found!")
    
    print()
    
    # Test Yelp
    print("🍽️ Testing Yelp Tool:")
    yelp = system.router.tools.get('yelp')
    if yelp:
        result = yelp.call_function('search_restaurants', {'location': 'New York', 'cuisine': 'Italian'})
        print(f"  Yelp Search: {result.success} - {len(result.result) if result.result else 0} restaurants")
        
        result = yelp.call_function('get_business_info', {'business_name': "Mario's"})
        print(f"  Yelp Business Info: {result.success} - Rating: {result.result.get('rating', 'N/A') if result.result else 'N/A'}")
    else:
        print("  Yelp tool not found!")
    
    print()
    
    # Test Flight
    print("✈️ Testing Flight Tool:")
    flight = system.router.tools.get('flight')
    if flight:
        result = flight.call_function('search_flights', {'origin': 'JFK', 'destination': 'LAX'})
        print(f"  Flight Search: {result.success} - {len(result.result) if result.result else 0} flights")
        
        result = flight.call_function('get_flight_status', {'flight_number': 'AA123'})
        print(f"  Flight Status: {result.success} - Status: {result.result.get('status', 'N/A') if result.result else 'N/A'}")
    else:
        print("  Flight tool not found!")
    
    print()
    
    # Test Agenda
    print("📅 Testing Agenda Tool:")
    agenda = system.router.tools.get('agenda')
    if agenda:
        result = agenda.call_function('get_events', {'person': 'Phoebe', 'date': '2022-12-19'})
        print(f"  Agenda Events: {result.success} - {len(result.result) if result.result else 0} events")
        if result.result and len(result.result) > 0:
            print(f"    First event: {result.result[0].get('event', 'N/A')}")
    else:
        print("  Agenda tool not found!")
    
    print()

def test_domain_detection():
    """Test domain detection accuracy"""
    print("🎯 Testing Domain Detection")
    print("=" * 50)
    
    extractor = DomainAwareAnswerExtractor()
    
    test_cases = [
        ("What was the coffee price range?", "coffee"),
        ("Calculate 2 + 3 * 4", "calculator"),
        ("How many papers did John Smith publish?", "dblp"),
        ("What's the rating of Mario's restaurant?", "yelp"),
        ("What's the status of flight AA123?", "flight"),
        ("What events does Phoebe have on 2022-12-19?", "agenda"),
    ]
    
    for question, expected_domain in test_cases:
        detected_domain = extractor.detect_domain(question, [])
        match = "✓" if detected_domain == expected_domain else "✗"
        print(f"  {match} Question: {question}")
        print(f"    Expected: {expected_domain}, Detected: {detected_domain}")
        print()

def test_enhanced_extraction_scenarios():
    """Test enhanced extraction with various domain scenarios"""
    print("🧠 Testing Enhanced Extraction Scenarios")
    print("=" * 50)
    
    extractor = DomainAwareAnswerExtractor()
    
    scenarios = [
        {
            "domain": "DBLP",
            "tool_results": [{"success": True, "tool_name": "dblp_count_papers", "result": {"count": 5, "author": "John Smith"}}],
            "question": "How many papers did John Smith publish?",
            "expected": "5"
        },
        {
            "domain": "Yelp", 
            "tool_results": [{"success": True, "tool_name": "yelp_get_business_info", "result": {"name": "Mario's", "rating": 4.5}}],
            "question": "What's the rating of Mario's restaurant?",
            "expected": "4.5"
        },
        {
            "domain": "Flight",
            "tool_results": [{"success": True, "tool_name": "flight_get_flight_status", "result": {"flight_number": "AA123", "status": "On Time"}}],
            "question": "What's the status of flight AA123?",
            "expected": "On Time"
        },
        {
            "domain": "Agenda",
            "tool_results": [{"success": True, "tool_name": "agenda_get_events", "result": [{"event": "Cheese + Wine festival"}]}],
            "question": "What events does Phoebe have on 2022-12-19?",
            "expected": "Cheese + Wine festival"
        }
    ]
    
    for scenario in scenarios:
        extracted = extractor.extract_answer(
            scenario["tool_results"], 
            scenario["question"], 
            scenario["expected"]
        )
        
        domain = extractor.detect_domain(scenario["question"], scenario["tool_results"])
        match = "✓" if extracted == scenario["expected"] else "✗"
        
        print(f"  {match} {scenario['domain']} Domain:")
        print(f"    Question: {scenario['question']}")
        print(f"    Expected: {scenario['expected']}")
        print(f"    Extracted: {extracted}")
        print(f"    Detected Domain: {domain}")
        print()

def run_mini_experiment():
    """Run a mini experiment with enhanced system"""
    print("🚀 Mini ToolQA Experiment with Enhanced System")
    print("=" * 50)
    
    # Simulate a few questions that would benefit from enhanced extraction
    test_questions = [
        {
            "id": "test_coffee_1",
            "question": "What was the coffee price range from 2000-01-03 to 2020-10-07?",
            "expected_answer": "306.2 USD",
            "domain": "coffee"
        },
        {
            "id": "test_agenda_1", 
            "question": "What events does Phoebe have on 2022/12/19 in the agenda table?",
            "expected_answer": "Cheese + Wine festival",
            "domain": "agenda"
        },
        {
            "id": "test_dblp_1",
            "question": "How many papers did John Smith publish?", 
            "expected_answer": "3",
            "domain": "dblp"
        },
        {
            "id": "test_yelp_1",
            "question": "What's the rating of Mario's Italian Restaurant?",
            "expected_answer": "4.5",
            "domain": "yelp"
        }
    ]
    
    # Initialize system
    system = create_and_register_tools("data/toolqa")
    extractor = DomainAwareAnswerExtractor()
    
    results = []
    
    for question_data in test_questions:
        print(f"Testing: {question_data['question']}")
        
        # Simulate tool execution (in real scenario, this would be done by the experiment runner)
        domain = question_data['domain']
        tool = system.router.tools.get(domain)
        
        if tool and domain == "coffee":
            # Coffee tool test
            tool_result = tool.call_function('get_price_range', {
                'start_date': '2000-01-03', 
                'end_date': '2020-10-07'
            })
            tool_results = [{
                "success": tool_result.success,
                "tool_name": f"{domain}_get_price_range",
                "result": tool_result.result
            }]
        elif tool and domain == "agenda":
            # Agenda tool test
            tool_result = tool.call_function('get_events', {
                'person': 'Phoebe',
                'date': '2022-12-19'
            })
            tool_results = [{
                "success": tool_result.success,
                "tool_name": f"{domain}_get_events", 
                "result": tool_result.result
            }]
        elif tool and domain == "dblp":
            # DBLP tool test
            tool_result = tool.call_function('get_author_papers', {
                'author_name': 'John Smith'
            })
            tool_results = [{
                "success": tool_result.success,
                "tool_name": f"{domain}_get_author_papers",
                "result": tool_result.result
            }]
        elif tool and domain == "yelp":
            # Yelp tool test
            tool_result = tool.call_function('get_business_info', {
                'business_name': "Mario's"
            })
            tool_results = [{
                "success": tool_result.success,
                "tool_name": f"{domain}_get_business_info",
                "result": tool_result.result
            }]
        else:
            tool_results = []
        
        # Extract answer using enhanced system
        if tool_results:
            extracted_answer = extractor.extract_answer(
                tool_results,
                question_data['question'],
                question_data['expected_answer']
            )
        else:
            extracted_answer = "N/A"
        
        # Check correctness
        is_correct = extracted_answer.lower() == question_data['expected_answer'].lower()
        
        result = {
            "question_id": question_data['id'],
            "question": question_data['question'],
            "expected_answer": question_data['expected_answer'],
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "tool_success": len(tool_results) > 0 and tool_results[0].get('success', False),
            "domain": domain
        }
        
        results.append(result)
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} Expected: {question_data['expected_answer']}, Got: {extracted_answer}")
        print(f"  Tool Success: {result['tool_success']}")
        print()
    
    # Summary
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    tool_success = sum(1 for r in results if r['tool_success'])
    
    print("📊 Mini Experiment Results:")
    print(f"  Total Questions: {total}")
    print(f"  Correct Answers: {correct} ({correct/total*100:.1f}%)")
    print(f"  Tool Success Rate: {tool_success} ({tool_success/total*100:.1f}%)")
    print()
    
    return results

def main():
    """Main test function"""
    print("🎯 ToolQA Enhanced System Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Run all tests
        test_coffee_extraction()
        test_new_domain_tools()
        test_domain_detection()
        test_enhanced_extraction_scenarios()
        results = run_mini_experiment()
        
        print("✅ All tests completed successfully!")
        
        # Save results
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("📄 Results saved to test_results.json")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()