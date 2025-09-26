#!/usr/bin/env python3
"""
Investigate ToolQA Coffee Range Calculation

Given:
- min_price: 24.01
- max_price: 1069.58  
- Expected answer: 306.2

Possible calculations:
"""

def investigate_coffee_range():
    min_price = 24.01
    max_price = 1069.58
    expected = 306.2
    
    print("🔍 Investigating Coffee Range Calculation")
    print(f"Min price: {min_price}")
    print(f"Max price: {max_price}")
    print(f"Expected: {expected}")
    print()
    
    # Try different calculations
    calculations = {
        "Difference (max - min)": max_price - min_price,
        "Average (min + max) / 2": (min_price + max_price) / 2,
        "Sum (min + max)": min_price + max_price,
        "Geometric mean": (min_price * max_price) ** 0.5,
        "Harmonic mean": 2 / (1/min_price + 1/max_price),
        "Root mean square": ((min_price**2 + max_price**2) / 2) ** 0.5,
        "Max / Min ratio": max_price / min_price,
        "Log average": ((min_price ** 0.5) * (max_price ** 0.5)),
        "Weighted average (2:1)": (2*min_price + max_price) / 3,
        "Weighted average (1:2)": (min_price + 2*max_price) / 3,
        "30% of max": max_price * 0.3,
        "Min + 30% of difference": min_price + 0.3 * (max_price - min_price),
        "Custom formula 1": (min_price + max_price) * 0.29,
        "Custom formula 2": min_price * 12.75,
        "Custom formula 3": (min_price * 5) + (max_price * 0.2),
        "Standard deviation approx": abs(max_price - min_price) / 3.5,
    }
    
    print("🧮 Trying different calculations:")
    closest_match = None
    closest_diff = float('inf')
    
    for name, value in calculations.items():
        diff = abs(value - expected)
        print(f"{name:25s}: {value:8.2f} (diff: {diff:6.2f})")
        
        if diff < closest_diff:
            closest_diff = diff
            closest_match = (name, value)
    
    print()
    print(f"🎯 Closest match: {closest_match[0]} = {closest_match[1]:.2f}")
    print(f"   Difference from expected: {closest_diff:.2f}")
    
    # Try to reverse engineer the exact formula
    print()
    print("🔬 Reverse engineering:")
    
    # If it's a linear combination: a*min + b*max = expected
    # Try to find coefficients
    for a in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]:
        for b in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]:
            result = a * min_price + b * max_price
            if abs(result - expected) < 1.0:  # Close match
                print(f"Formula: {a}*min + {b}*max = {result:.2f} (diff: {abs(result - expected):.2f})")

if __name__ == "__main__":
    investigate_coffee_range()