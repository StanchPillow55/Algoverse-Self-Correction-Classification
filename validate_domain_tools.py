#!/usr/bin/env python3
"""
Validate Domain Tools Against ToolQA Questions

Test script to validate that our domain tools can properly handle
real ToolQA questions and produce correct answers.
"""

import sys
import os
sys.path.append('src')

from tools.domain_tools import create_and_register_tools
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_yelp_domain():
    """Test Yelp domain with real ToolQA questions"""
    print("\n🧪 Testing Yelp Domain")
    print("=" * 50)
    
    system = create_and_register_tools()
    yelp_tool = system.router.tools.get('yelp')
    
    if not yelp_tool:
        print("❌ Yelp tool not found")
        return False
    
    # Test cases based on real ToolQA questions
    test_cases = [
        {
            "function": "get_address",
            "params": {"business_name": "Snip Philadelphia", "postal_code": "19130"},
            "expected": "2052 Fairmount Ave",
            "description": "Get address of Snip Philadelphia in postal code 19130"
        },
        {
            "function": "get_postal_code", 
            "params": {"business_name": "Smilies", "city": "Edmonton", "state": "AB"},
            "expected": "T5V 1H9",
            "description": "Get postal code of Smilies in Edmonton, AB"
        },
        {
            "function": "get_coordinates",
            "params": {"business_name": "Snip Philadelphia"},
            "expected": "39.9652, -75.1734",
            "description": "Get coordinates of Snip Philadelphia"
        },
        {
            "function": "get_hours",
            "params": {"business_name": "Smilies"},
            "expected": "Monday: 11:0-22:0, Tuesday: 11:0-22:0, Wednesday: 11:0-22:0, Thursday: 11:0-22:0, Friday: 11:0-23:0, Saturday: 11:0-23:0, Sunday: 12:0-21:0",
            "description": "Get hours of Smilies"
        },
        {
            "function": "check_appointment_required",
            "params": {"business_name": "Snip Philadelphia"},
            "expected": "Yes",
            "description": "Check if Snip Philadelphia requires appointments"
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test {i+1}: {test_case['description']}")
        
        try:
            result = yelp_tool.call_function(test_case["function"], test_case["params"])
            
            if result.success:
                actual = str(result.result)
                expected = test_case["expected"]
                
                if actual == expected:
                    print(f"✅ PASS: Got '{actual}'")
                    success_count += 1
                else:
                    print(f"❌ FAIL: Expected '{expected}', got '{actual}'")
            else:
                print(f"❌ FAIL: Tool execution failed - {result.error_message}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📊 Yelp Results: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)

def test_airbnb_domain():
    """Test Airbnb domain with real ToolQA questions"""
    print("\n🧪 Testing Airbnb Domain")
    print("=" * 50)
    
    system = create_and_register_tools()
    airbnb_tool = system.router.tools.get('airbnb')
    
    if not airbnb_tool:
        print("❌ Airbnb tool not found")
        return False
    
    # Test cases based on real ToolQA questions
    test_cases = [
        {
            "function": "get_host_info",
            "params": {"listing_name": "Amazing One Bedroom Apartment in Prime Brooklyn"},
            "expected_host": "Alan",
            "description": "Get host name for Amazing One Bedroom Apartment"
        },
        {
            "function": "get_availability_info", 
            "params": {"listing_id": 23456},
            "expected_nights": 347,
            "description": "Get availability for Cozy 2 bedroom listing"
        },
        {
            "function": "search_listings",
            "params": {"neighbourhood": "Bushwick"},
            "expected_count": 1,
            "description": "Search listings in Bushwick"
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test {i+1}: {test_case['description']}")
        
        try:
            result = airbnb_tool.call_function(test_case["function"], test_case["params"])
            
            if result.success:
                if "expected_host" in test_case:
                    actual_host = result.result.get("host_name", "")
                    expected = test_case["expected_host"]
                    
                    if actual_host == expected:
                        print(f"✅ PASS: Got host '{actual_host}'")
                        success_count += 1
                    else:
                        print(f"❌ FAIL: Expected host '{expected}', got '{actual_host}'")
                        
                elif "expected_nights" in test_case:
                    actual_nights = result.result.get("availability_365", 0)
                    expected = test_case["expected_nights"]
                    
                    if actual_nights == expected:
                        print(f"✅ PASS: Got availability {actual_nights}")
                        success_count += 1
                    else:
                        print(f"❌ FAIL: Expected availability {expected}, got {actual_nights}")
                        
                elif "expected_count" in test_case:
                    actual_count = len(result.result) if isinstance(result.result, list) else 0
                    expected = test_case["expected_count"]
                    
                    if actual_count == expected:
                        print(f"✅ PASS: Found {actual_count} listing(s)")
                        success_count += 1
                    else:
                        print(f"❌ FAIL: Expected {expected} listings, got {actual_count}")
                        
            else:
                print(f"❌ FAIL: Tool execution failed - {result.error_message}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📊 Airbnb Results: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)

def test_coffee_domain():
    """Test Coffee domain (should already work)"""
    print("\n🧪 Testing Coffee Domain")
    print("=" * 50)
    
    system = create_and_register_tools()
    coffee_tool = system.router.tools.get('coffee')
    
    if not coffee_tool:
        print("❌ Coffee tool not found")
        return False
    
    # Quick test to verify coffee tool works
    test_cases = [
        {
            "function": "get_price_range",
            "params": {"start_date": "2021-01-01", "end_date": "2021-01-31"},
            "description": "Get coffee price range for January 2021"
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test {i+1}: {test_case['description']}")
        
        try:
            result = coffee_tool.call_function(test_case["function"], test_case["params"])
            
            if result.success:
                print(f"✅ PASS: Tool executed successfully")
                print(f"   Result: {result.result}")
                success_count += 1
            else:
                print(f"❌ FAIL: Tool execution failed - {result.error_message}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📊 Coffee Results: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)

def test_tool_system():
    """Test overall tool system functionality"""
    print("\n🧪 Testing Tool System")
    print("=" * 50)
    
    try:
        system = create_and_register_tools()
        registered_tools = list(system.router.tools.keys())
        
        print(f"📋 Registered tools: {registered_tools}")
        
        expected_tools = ['calculator', 'coffee', 'dblp', 'yelp', 'flight', 'airbnb', 'agenda']
        missing_tools = [tool for tool in expected_tools if tool not in registered_tools]
        
        if missing_tools:
            print(f"❌ Missing tools: {missing_tools}")
            return False
        else:
            print(f"✅ All expected tools registered")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: Tool system initialization failed - {e}")
        return False

def main():
    """Run all validation tests"""
    print("🔍 ToolQA Domain Tools Validation")
    print("=" * 60)
    
    results = {
        "Tool System": test_tool_system(),
        "Coffee Domain": test_coffee_domain(),
        "Yelp Domain": test_yelp_domain(),
        "Airbnb Domain": test_airbnb_domain()
    }
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} domain tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 SUCCESS: All domain validations passed!")
        return True
    else:
        print(f"\n⚠️  WARNING: {total_tests - passed_tests} domain validation(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)