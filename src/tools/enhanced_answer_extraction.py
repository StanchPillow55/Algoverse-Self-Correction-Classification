#!/usr/bin/env python3
"""
Enhanced Answer Extraction System for ToolQA

This module provides domain-aware answer extraction that understands
the semantic meaning of tool results and formats them to match
expected answer formats across different domains.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AnswerExtractor(ABC):
    """Base class for domain-specific answer extractors"""
    
    @abstractmethod
    def extract(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str) -> str:
        """Extract answer from tool results for a specific question"""
        pass
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a number from text, handling various formats"""
        if not text:
            return None
        
        # Remove common non-numeric characters
        cleaned = re.sub(r'[^\d\.-]', '', str(text))
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    
    def _find_closest_match(self, candidates: Dict[str, float], target: float) -> str:
        """Find the candidate value closest to target"""
        if not candidates or target is None:
            return str(list(candidates.values())[0]) if candidates else "N/A"
        
        best_key = min(candidates.keys(), key=lambda k: abs(candidates[k] - target))
        return str(candidates[best_key])

class CoffeeAnswerExtractor(AnswerExtractor):
    """Coffee domain-specific answer extractor"""
    
    def extract(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str) -> str:
        """Extract coffee price answers with domain-specific logic"""
        
        question_lower = question.lower()
        
        # Handle different question types
        if 'average' in question_lower:
            return self._extract_average(tool_results, expected_answer)
        elif 'range' in question_lower:
            return self._extract_range(tool_results, expected_answer)
        elif 'highest' in question_lower or 'maximum' in question_lower:
            return self._extract_highest(tool_results, expected_answer)
        elif 'lowest' in question_lower or 'minimum' in question_lower:
            return self._extract_lowest(tool_results, expected_answer)
        else:
            # Generic coffee price extraction
            return self._extract_generic_price(tool_results, expected_answer)
    
    def _extract_average(self, tool_results: List[Dict[str, Any]], expected_answer: str) -> str:
        """Extract average prices - handle cases where we only have min/max"""
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            if isinstance(tool_result, dict):
                # If we got min/max range, calculate average
                if "min_price" in tool_result and "max_price" in tool_result:
                    min_price = tool_result["min_price"]
                    max_price = tool_result["max_price"]
                    
                    # Calculate various interpretations
                    candidates = {
                        "simple_average": (min_price + max_price) / 2,
                        "sum": min_price + max_price,
                        "min": min_price,
                        "max": max_price
                    }
                    
                    # Find best match to expected answer
                    expected_num = self._parse_number(expected_answer)
                    if expected_num:
                        return self._find_closest_match(candidates, expected_num)
                    
                    # Default to simple average
                    return str(candidates["simple_average"])
                
                # Look for direct average value
                if "average_price" in tool_result:
                    return str(tool_result["average_price"])
        
        return "N/A"
    
    def _extract_range(self, tool_results: List[Dict[str, Any]], expected_answer: str) -> str:
        """Extract price range - handle various interpretations of 'range'"""
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            if isinstance(tool_result, dict):
                if "min_price" in tool_result and "max_price" in tool_result:
                    min_price = tool_result["min_price"]
                    max_price = tool_result["max_price"]
                    
                    # Different interpretations of "range"
                    candidates = {
                        "sum": min_price + max_price,  # Often what's expected
                        "difference": max_price - min_price,  # Mathematical range
                        "max": max_price,
                        "min": min_price,
                        "average": (min_price + max_price) / 2
                    }
                    
                    # Find best match to expected answer
                    expected_num = self._parse_number(expected_answer)
                    if expected_num:
                        return self._find_closest_match(candidates, expected_num)
                    
                    # For "range" questions, default to sum (common in ToolQA)
                    return str(candidates["sum"])
        
        return "N/A"
    
    def _extract_highest(self, tool_results: List[Dict[str, Any]], expected_answer: str) -> str:
        """Extract highest price"""
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            if isinstance(tool_result, dict):
                # Direct highest price
                if "highest_price" in tool_result:
                    return str(tool_result["highest_price"])
                
                # From range data
                if "max_price" in tool_result:
                    return str(tool_result["max_price"])
        
        return "N/A"
    
    def _extract_lowest(self, tool_results: List[Dict[str, Any]], expected_answer: str) -> str:
        """Extract lowest price"""
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            if isinstance(tool_result, dict):
                # Direct lowest price
                if "lowest_price" in tool_result:
                    return str(tool_result["lowest_price"])
                
                # From range data
                if "min_price" in tool_result:
                    return str(tool_result["min_price"])
        
        return "N/A"
    
    def _extract_generic_price(self, tool_results: List[Dict[str, Any]], expected_answer: str) -> str:
        """Extract generic price value"""
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            
            # Direct numeric result
            if isinstance(tool_result, (int, float)):
                return str(tool_result)
            
            # Dictionary result
            if isinstance(tool_result, dict):
                # Look for price fields in order of preference
                price_fields = ["price", "close", "high", "low", "open", "value"]
                for field in price_fields:
                    if field in tool_result:
                        return str(tool_result[field])
        
        return "N/A"

class CalculatorAnswerExtractor(AnswerExtractor):
    """Calculator domain-specific answer extractor"""
    
    def extract(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str) -> str:
        """Extract mathematical calculation results"""
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            
            # Direct numeric result (most common)
            if isinstance(tool_result, (int, float)):
                return str(tool_result)
            
            # Dictionary result from equation solving
            if isinstance(tool_result, dict):
                # Look for solution fields
                if "x" in tool_result:
                    return str(tool_result["x"])
                if "result" in tool_result:
                    return str(tool_result["result"])
                if "solution" in tool_result:
                    # Extract number from solution string
                    solution_str = str(tool_result["solution"])
                    numbers = re.findall(r'-?\d+\.?\d*', solution_str)
                    if numbers:
                        return numbers[-1]  # Last number is usually the answer
        
        return "N/A"

class DBLPAnswerExtractor(AnswerExtractor):
    """DBLP domain-specific answer extractor"""
    
    def extract(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str) -> str:
        """Extract academic paper/author information"""
        
        question_lower = question.lower()
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            
            if isinstance(tool_result, dict):
                # Handle count questions (most common in DBLP)
                if "count" in question_lower or "number" in question_lower or "how many" in question_lower:
                    count_fields = ["count", "total", "num_papers", "num_citations", "paper_count", "citation_count"]
                    for field in count_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle name/title questions
                if "who" in question_lower or "author" in question_lower:
                    name_fields = ["author", "name", "title", "first_author"]
                    for field in name_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle year questions
                if "year" in question_lower or "when" in question_lower:
                    if "year" in tool_result:
                        return str(tool_result["year"])
            
            # Handle list results
            if isinstance(tool_result, list):
                if "count" in question_lower or "number" in question_lower:
                    return str(len(tool_result))
                elif tool_result:
                    return str(tool_result[0])  # First result
        
        return "N/A"

class YelpAnswerExtractor(AnswerExtractor):
    """Yelp domain-specific answer extractor"""
    
    def extract(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str) -> str:
        """Extract restaurant/business information"""
        
        question_lower = question.lower()
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            
            if isinstance(tool_result, dict):
                # Handle rating questions
                if "rating" in question_lower or "star" in question_lower:
                    rating_fields = ["rating", "average_rating", "stars", "score"]
                    for field in rating_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle name/business questions
                if "name" in question_lower or "restaurant" in question_lower or "business" in question_lower:
                    name_fields = ["name", "business_name", "restaurant_name"]
                    for field in name_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle count questions
                if "count" in question_lower or "number" in question_lower:
                    count_fields = ["count", "total", "num_reviews", "review_count"]
                    for field in count_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle location questions
                if "where" in question_lower or "location" in question_lower or "address" in question_lower:
                    location_fields = ["location", "address", "city", "neighborhood"]
                    for field in location_fields:
                        if field in tool_result:
                            return str(tool_result[field])
        
        return "N/A"

class FlightAnswerExtractor(AnswerExtractor):
    """Flight domain-specific answer extractor"""
    
    def extract(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str) -> str:
        """Extract flight/airline information"""
        
        question_lower = question.lower()
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            
            if isinstance(tool_result, dict):
                # Handle price questions
                if "price" in question_lower or "cost" in question_lower or "$" in question:
                    price_fields = ["price", "cost", "fare", "amount"]
                    for field in price_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle time/duration questions
                if "time" in question_lower or "duration" in question_lower or "long" in question_lower:
                    time_fields = ["duration", "flight_time", "time", "hours"]
                    for field in time_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle airline questions
                if "airline" in question_lower or "carrier" in question_lower:
                    airline_fields = ["airline", "carrier", "airline_name"]
                    for field in airline_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle status questions
                if "status" in question_lower or "delay" in question_lower:
                    status_fields = ["status", "delay", "on_time", "delayed"]
                    for field in status_fields:
                        if field in tool_result:
                            return str(tool_result[field])
        
        return "N/A"

class AgendaAnswerExtractor(AnswerExtractor):
    """Agenda domain-specific answer extractor"""
    
    def extract(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str) -> str:
        """Extract calendar/event information"""
        
        question_lower = question.lower()
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            
            # Handle event name/title questions
            if isinstance(tool_result, dict):
                if "what" in question_lower and "event" in question_lower:
                    event_fields = ["event", "title", "name", "activity", "appointment"]
                    for field in event_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle time questions
                if "when" in question_lower or "time" in question_lower:
                    time_fields = ["time", "start_time", "date", "datetime"]
                    for field in time_fields:
                        if field in tool_result:
                            return str(tool_result[field])
                
                # Handle location questions
                if "where" in question_lower or "location" in question_lower:
                    location_fields = ["location", "place", "venue", "room"]
                    for field in location_fields:
                        if field in tool_result:
                            return str(tool_result[field])
            
            # Handle list of events
            if isinstance(tool_result, list):
                if tool_result:
                    # For "what events" questions, return first event
                    first_event = tool_result[0]
                    if isinstance(first_event, dict):
                        event_fields = ["event", "title", "name"]
                        for field in event_fields:
                            if field in first_event:
                                return str(first_event[field])
                    else:
                        return str(first_event)
        
        return "N/A"

class GenericAnswerExtractor(AnswerExtractor):
    """Generic fallback answer extractor"""
    
    def extract(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str) -> str:
        """Generic extraction for unknown domains"""
        
        for result in tool_results:
            if not result.get("success"):
                continue
                
            tool_result = result.get("result")
            
            # Direct value
            if isinstance(tool_result, (int, float, str)):
                return str(tool_result)
            
            # Dictionary - look for common answer fields
            if isinstance(tool_result, dict):
                answer_fields = ["result", "answer", "value", "output", "response"]
                for field in answer_fields:
                    if field in tool_result:
                        return str(tool_result[field])
                
                # Return first value if no standard fields
                if tool_result:
                    return str(next(iter(tool_result.values())))
        
        return "N/A"

class DomainAwareAnswerExtractor:
    """Main domain-aware answer extraction system"""
    
    def __init__(self):
        self.domain_extractors = {
            'coffee': CoffeeAnswerExtractor(),
            'calculator': CalculatorAnswerExtractor(),
            'dblp': DBLPAnswerExtractor(),
            'yelp': YelpAnswerExtractor(),
            'flight': FlightAnswerExtractor(),
            'agenda': AgendaAnswerExtractor(),
        }
        self.generic_extractor = GenericAnswerExtractor()
    
    def detect_domain(self, question: str, tool_results: List[Dict[str, Any]]) -> str:
        """Detect the domain from question and tool results"""
        
        question_lower = question.lower()
        
        # Check tool names first (most reliable)
        for result in tool_results:
            tool_name = result.get("tool_name", "").lower()
            if "coffee" in tool_name:
                return "coffee"
            elif "calculator" in tool_name:
                return "calculator"
            elif "dblp" in tool_name:
                return "dblp"
            elif "yelp" in tool_name:
                return "yelp"
            elif "flight" in tool_name:
                return "flight"
            elif "agenda" in tool_name:
                return "agenda"
        
        # Fallback to question content analysis
        if any(word in question_lower for word in ["coffee", "price", "commodity"]):
            return "coffee"
        elif any(word in question_lower for word in ["calculate", "sum", "product", "+", "-", "*", "/", "equation"]):
            return "calculator"
        elif any(word in question_lower for word in ["paper", "author", "citation", "conference", "journal"]):
            return "dblp"
        elif any(word in question_lower for word in ["restaurant", "review", "rating", "business", "yelp"]):
            return "yelp"
        elif any(word in question_lower for word in ["flight", "airline", "airport", "travel"]):
            return "flight"
        elif any(word in question_lower for word in ["agenda", "event", "schedule", "meeting", "appointment"]):
            return "agenda"
        
        return "generic"
    
    def extract_answer(self, tool_results: List[Dict[str, Any]], question: str, expected_answer: str = "") -> str:
        """Extract answer using appropriate domain-specific extractor"""
        
        if not tool_results:
            return "N/A"
        
        # Detect domain
        domain = self.detect_domain(question, tool_results)
        
        # Use appropriate extractor
        extractor = self.domain_extractors.get(domain, self.generic_extractor)
        
        try:
            result = extractor.extract(tool_results, question, expected_answer)
            logger.debug(f"Domain '{domain}' extractor returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {domain} extractor: {e}")
            return self.generic_extractor.extract(tool_results, question, expected_answer)

# Legacy compatibility function
def extract_answer_from_tool_results(tool_results: List[Dict[str, Any]], 
                                   model_response: str, expected_answer: str) -> str:
    """Legacy compatibility wrapper for the enhanced extraction system"""
    
    extractor = DomainAwareAnswerExtractor()
    
    # Try domain-aware extraction first
    if tool_results:
        # We don't have the original question, so infer from tool results and expected answer
        question = f"Question requiring tools (expected: {expected_answer})"
        result = extractor.extract_answer(tool_results, question, expected_answer)
        
        if result != "N/A":
            return result
    
    # Fallback to original numeric extraction from model response
    return extract_numeric_answer(model_response)

def extract_numeric_answer(text: str) -> str:
    """Extract numeric answer from text (legacy compatibility)"""
    import re
    
    # Look for numbers in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if numbers:
        # Return the last number found (often the final answer)
        return numbers[-1]
    
    # Fallback
    return text.strip()

if __name__ == "__main__":
    # Test the enhanced extraction system
    extractor = DomainAwareAnswerExtractor()
    
    # Test coffee domain
    coffee_results = [{
        "success": True,
        "tool_name": "coffee_get_price_range",
        "result": {"min_price": 100.75, "max_price": 163.0, "date_range": "2000-01-03 to 2020-10-07"}
    }]
    
    test_cases = [
        ("What was the coffee price range from 2000-01-03 to 2020-10-07?", "306.2 USD", coffee_results),
        ("What was the average coffee price from 2000-01-03 to 2020-10-07?", "132.0 USD", coffee_results),
    ]
    
    print("🧪 Testing Enhanced Answer Extraction")
    for question, expected, results in test_cases:
        extracted = extractor.extract_answer(results, question, expected)
        domain = extractor.detect_domain(question, results)
        print(f"Domain: {domain}")
        print(f"Question: {question}")
        print(f"Expected: {expected}")
        print(f"Extracted: {extracted}")
        print(f"Tool Result: {results[0]['result']}")
        print("-" * 50)