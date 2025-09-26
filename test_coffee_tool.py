#!/usr/bin/env python3
"""
Simple test of coffee tool with comprehensive data
"""

import sys
sys.path.insert(0, 'src')

from tools.domain_tools import CoffeeTool

def test_coffee_tool():
    print("🎯 Testing Coffee Tool Functions")
    print("=" * 50)
    
    coffee_tool = CoffeeTool('data/toolqa')
    
    # Test 1: Price range
    print("Test 1: Price Range")
    result = coffee_tool.call_function('get_price_range', {
        'start_date': '2000-01-03',
        'end_date': '2020-10-07'
    })
    if result.success:
        print(f"  ✅ Success: {result.result}")
        min_price = result.result.get('min_price', 0)
        max_price = result.result.get('max_price', 0)
        total = min_price + max_price
        print(f"  Min + Max = {total:.2f} (Expected: 306.2)")
    else:
        print(f"  ❌ Error: {result.error_message}")
    print()
    
    # Test 2: Single date price
    print("Test 2: Single Date Price")
    result = coffee_tool.call_function('get_price_on_date', {
        'date': '2000-09-13'
    })
    if result.success:
        print(f"  ✅ Success: {result.result}")
        low_price = result.result.get('low', 0)
        print(f"  Low price = {low_price} (Expected: 79.0)")
    else:
        print(f"  ❌ Error: {result.error_message}")
    print()
    
    # Test 3: Check data coverage
    print("Test 3: Data Coverage Check")
    result = coffee_tool.call_function('get_price_on_date', {
        'date': '2022-08-16'
    })
    if result.success:
        print(f"  ✅ Has recent data: {result.result}")
    else:
        print(f"  ❌ Missing recent data: {result.error_message}")
    print()
    
    # Test available functions
    print("Available functions:")
    for func in coffee_tool.get_available_functions():
        print(f"  - {func}")

if __name__ == "__main__":
    test_coffee_tool()