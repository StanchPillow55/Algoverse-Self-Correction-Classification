#!/usr/bin/env python3
"""
ToolQA Domain-Specific Tool Implementations

This module implements the actual tools for each ToolQA domain, providing
concrete functionality for data queries, calculations, and information retrieval.

Tools implemented:
- Calculator: For GSM8K math problems  
- Coffee: Commodity price analysis
- DBLP: Citation network analysis
- Yelp: Business review analysis
- Flight: Flight status checking
- Airbnb: Property listing analysis
- Agenda: Event scheduling analysis
"""

import json
import logging
import time
import math
import re
import csv
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass

from .unified_tool_system import ToolInterface, ToolResult

logger = logging.getLogger(__name__)

class CalculatorTool(ToolInterface):
    """Calculator tool for GSM8K mathematical problems"""
    
    def get_name(self) -> str:
        return "calculator"
    
    def get_functions(self) -> List[str]:
        return ["calculate", "solve_equation", "evaluate_expression"]
    
    def get_function_schema(self, function: str) -> Dict[str, Any]:
        schemas = {
            "calculate": {
                "description": "Perform mathematical calculations and arithmetic operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', '(10 - 5) / 2')"
                        }
                    },
                    "required": ["expression"]
                }
            },
            "solve_equation": {
                "description": "Solve simple algebraic equations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equation": {
                            "type": "string", 
                            "description": "Equation to solve (e.g., 'x + 5 = 10', '2*x = 8')"
                        }
                    },
                    "required": ["equation"]
                }
            },
            "evaluate_expression": {
                "description": "Evaluate complex mathematical expressions with variables",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Expression to evaluate"
                        },
                        "variables": {
                            "type": "object",
                            "description": "Variable values as key-value pairs"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
        return schemas.get(function, {})
    
    def call_function(self, function: str, parameters: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            if function == "calculate":
                result = self._calculate(parameters["expression"])
            elif function == "solve_equation":
                result = self._solve_equation(parameters["equation"])
            elif function == "evaluate_expression":
                result = self._evaluate_expression(
                    parameters["expression"],
                    parameters.get("variables", {})
                )
            else:
                raise ValueError(f"Unknown function: {function}")
                
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _calculate(self, expression: str) -> Union[int, float]:
        """Safely evaluate mathematical expressions"""
        # Clean up expression
        expression = expression.strip()
        
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
        
        # Replace common math functions
        expression = expression.replace("^", "**")  # Power operator
        
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return result
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")
    
    def _solve_equation(self, equation: str) -> Dict[str, Union[float, str]]:
        """Solve simple linear equations"""
        equation = equation.strip().replace(" ", "")
        
        # Handle simple linear equations like x + 5 = 10, 2*x = 8
        if "=" not in equation:
            raise ValueError("Equation must contain '=' sign")
        
        left, right = equation.split("=")
        
        # Simple x + a = b or a + x = b
        if "+" in left:
            parts = left.split("+")
            if "x" in parts[0] and parts[0] == "x":
                a = float(parts[1])
                b = float(right)
                x = b - a
            elif "x" in parts[1] and parts[1] == "x":
                a = float(parts[0])
                b = float(right)
                x = b - a
            else:
                raise ValueError("Cannot solve this equation format")
        # Simple x - a = b
        elif "-" in left:
            parts = left.split("-")
            if "x" in parts[0] and parts[0] == "x":
                a = float(parts[1])
                b = float(right)
                x = b + a
            else:
                raise ValueError("Cannot solve this equation format")
        # Simple a*x = b or x*a = b
        elif "*" in left:
            parts = left.split("*")
            if "x" in parts[0] and parts[0] == "x":
                a = float(parts[1])
                b = float(right)
                x = b / a
            elif "x" in parts[1] and parts[1] == "x":
                a = float(parts[0])
                b = float(right)
                x = b / a
            else:
                raise ValueError("Cannot solve this equation format")
        # Simple x = b
        elif left == "x":
            x = float(right)
        else:
            raise ValueError("Cannot solve this equation format")
        
        return {"x": x, "equation": equation, "solution": f"x = {x}"}
    
    def _evaluate_expression(self, expression: str, variables: Dict[str, float]) -> float:
        """Evaluate expression with variable substitution"""
        # Substitute variables
        for var, val in variables.items():
            expression = expression.replace(var, str(val))
        
        # Use calculate function
        return self._calculate(expression)

class CoffeeTool(ToolInterface):
    """Tool for coffee commodity price analysis"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.price_data = self._load_coffee_data()
    
    def _load_coffee_data(self) -> List[Dict[str, Any]]:
        """Load coffee price data from CSV"""
        coffee_file = self.data_dir / "coffee.csv"
        if not coffee_file.exists():
            logger.warning(f"Coffee data file not found: {coffee_file}")
            return []
        
        data = []
        try:
            with open(coffee_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert date string to date object
                    row['Date'] = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                    row['Open'] = float(row['Open'])
                    row['High'] = float(row['High']) 
                    row['Low'] = float(row['Low'])
                    row['Close'] = float(row['Close'])
                    data.append(row)
        except Exception as e:
            logger.error(f"Failed to load coffee data: {e}")
            
        return data
    
    def get_name(self) -> str:
        return "coffee"
    
    def get_functions(self) -> List[str]:
        return ["get_price_range", "get_highest_price", "get_lowest_price", "get_price_on_date", "analyze_trend"]
    
    def get_function_schema(self, function: str) -> Dict[str, Any]:
        schemas = {
            "get_price_range": {
                "description": "Get coffee price range (min/max) for a date range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string", 
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            "get_highest_price": {
                "description": "Get the highest coffee price in a date range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            "get_lowest_price": {
                "description": "Get the lowest coffee price in a date range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            "get_price_on_date": {
                "description": "Get coffee price on a specific date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["date"]
                }
            },
            "analyze_trend": {
                "description": "Analyze if trend is bullish or bearish in date range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            }
        }
        return schemas.get(function, {})
    
    def call_function(self, function: str, parameters: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            if function == "get_price_range":
                result = self._get_price_range(parameters["start_date"], parameters["end_date"])
            elif function == "get_highest_price":
                result = self._get_highest_price(parameters["start_date"], parameters["end_date"])
            elif function == "get_lowest_price":
                result = self._get_lowest_price(parameters["start_date"], parameters["end_date"])
            elif function == "get_price_on_date":
                result = self._get_price_on_date(parameters["date"])
            elif function == "analyze_trend":
                result = self._analyze_trend(parameters["start_date"], parameters["end_date"])
            else:
                raise ValueError(f"Unknown function: {function}")
                
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object"""
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    
    def _filter_data_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Filter price data by date range"""
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)
        
        filtered_data = []
        for record in self.price_data:
            if start_dt <= record['Date'] <= end_dt:
                filtered_data.append(record)
        
        return filtered_data
    
    def _get_price_range(self, start_date: str, end_date: str) -> Dict[str, float]:
        """Get price range in date range"""
        data = self._filter_data_by_date_range(start_date, end_date)
        
        if not data:
            raise ValueError(f"No data found for date range {start_date} to {end_date}")
        
        high_prices = [record['High'] for record in data]
        low_prices = [record['Low'] for record in data]
        
        return {
            "min_price": min(low_prices),
            "max_price": max(high_prices),
            "date_range": f"{start_date} to {end_date}",
            "data_points": len(data)
        }
    
    def _get_highest_price(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get highest price in date range"""
        data = self._filter_data_by_date_range(start_date, end_date)
        
        if not data:
            raise ValueError(f"No data found for date range {start_date} to {end_date}")
        
        highest_record = max(data, key=lambda x: x['High'])
        
        return {
            "highest_price": highest_record['High'],
            "date": highest_record['Date'].strftime('%Y-%m-%d'),
            "open": highest_record['Open'],
            "close": highest_record['Close']
        }
    
    def _get_lowest_price(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get lowest price in date range"""
        data = self._filter_data_by_date_range(start_date, end_date)
        
        if not data:
            raise ValueError(f"No data found for date range {start_date} to {end_date}")
        
        lowest_record = min(data, key=lambda x: x['Low'])
        
        return {
            "lowest_price": lowest_record['Low'],
            "date": lowest_record['Date'].strftime('%Y-%m-%d'),
            "open": lowest_record['Open'],
            "close": lowest_record['Close']
        }
    
    def _get_price_on_date(self, date_str: str) -> Dict[str, float]:
        """Get price on specific date"""
        target_date = self._parse_date(date_str)
        
        for record in self.price_data:
            if record['Date'] == target_date:
                return {
                    "date": date_str,
                    "open": record['Open'],
                    "high": record['High'],
                    "low": record['Low'],
                    "close": record['Close']
                }
        
        raise ValueError(f"No data found for date {date_str}")
    
    def _analyze_trend(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze if trend is bullish or bearish"""
        data = self._filter_data_by_date_range(start_date, end_date)
        
        if not data:
            raise ValueError(f"No data found for date range {start_date} to {end_date}")
        
        # Sort by date
        data.sort(key=lambda x: x['Date'])
        
        start_price = data[0]['Close']
        end_price = data[-1]['Close']
        
        price_change = end_price - start_price
        percent_change = (price_change / start_price) * 100
        
        if percent_change > 5:
            trend = "bullish"
        elif percent_change < -5:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "trend": trend,
            "start_price": start_price,
            "end_price": end_price,
            "price_change": price_change,
            "percent_change": round(percent_change, 2),
            "period": f"{start_date} to {end_date}"
        }

class DBLPTool(ToolInterface):
    """Tool for DBLP academic paper and author analysis"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.papers_data = self._load_papers_data()
    
    def _load_papers_data(self) -> List[Dict[str, Any]]:
        """Load papers data from JSON"""
        papers_file = self.data_dir / "dblp" / "papers.json"
        if not papers_file.exists():
            logger.warning(f"DBLP papers file not found: {papers_file}")
            return []
        
        try:
            with open(papers_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load DBLP data: {e}")
            return []
    
    def get_name(self) -> str:
        return "dblp"
    
    def get_functions(self) -> List[str]:
        return ["search_papers", "get_author_papers", "get_citation_count", "get_paper_info", "count_papers"]
    
    def get_function_schema(self, function: str) -> Dict[str, Any]:
        schemas = {
            "search_papers": {
                "description": "Search for papers by title keywords, author, or venue",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keywords to search for in titles"},
                        "author": {"type": "string", "description": "Author name to search for"},
                        "venue": {"type": "string", "description": "Conference or journal venue"}
                    }
                }
            },
            "get_author_papers": {
                "description": "Get all papers by a specific author",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "author_name": {"type": "string", "description": "Name of the author"}
                    },
                    "required": ["author_name"]
                }
            },
            "get_citation_count": {
                "description": "Get total citation count for an author or paper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "author": {"type": "string", "description": "Author name"},
                        "paper_title": {"type": "string", "description": "Paper title"}
                    }
                }
            },
            "count_papers": {
                "description": "Count papers by author, venue, or year",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "author": {"type": "string", "description": "Author name"},
                        "venue": {"type": "string", "description": "Conference or journal venue"},
                        "year": {"type": "integer", "description": "Publication year"}
                    }
                }
            }
        }
        return schemas.get(function, {})
    
    def call_function(self, function: str, parameters: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            if function == "search_papers":
                result = self._search_papers(
                    parameters.get("query", ""),
                    parameters.get("author", ""),
                    parameters.get("venue", "")
                )
            elif function == "get_author_papers":
                result = self._get_author_papers(parameters["author_name"])
            elif function == "get_citation_count":
                result = self._get_citation_count(
                    parameters.get("author", ""),
                    parameters.get("paper_title", "")
                )
            elif function == "count_papers":
                result = self._count_papers(
                    parameters.get("author", ""),
                    parameters.get("venue", ""),
                    parameters.get("year")
                )
            else:
                raise ValueError(f"Unknown function: {function}")
                
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _search_papers(self, query: str, author: str, venue: str) -> List[Dict[str, Any]]:
        """Search papers by various criteria"""
        results = []
        
        for paper in self.papers_data:
            match = True
            
            if query and query.lower() not in paper["title"].lower():
                match = False
            
            if author and not any(author.lower() in a.lower() for a in paper["authors"]):
                match = False
            
            if venue and venue.lower() != paper["venue"].lower():
                match = False
            
            if match:
                results.append(paper)
        
        return results
    
    def _get_author_papers(self, author_name: str) -> Dict[str, Any]:
        """Get papers by specific author"""
        papers = []
        total_citations = 0
        
        for paper in self.papers_data:
            if any(author_name.lower() in author.lower() for author in paper["authors"]):
                papers.append(paper)
                total_citations += paper["citations"]
        
        return {
            "author": author_name,
            "papers": papers,
            "paper_count": len(papers),
            "total_citations": total_citations
        }
    
    def _get_citation_count(self, author: str, paper_title: str) -> Dict[str, Any]:
        """Get citation count for author or paper"""
        if paper_title:
            # Search for specific paper
            for paper in self.papers_data:
                if paper_title.lower() in paper["title"].lower():
                    return {
                        "paper_title": paper["title"],
                        "citations": paper["citations"],
                        "authors": paper["authors"]
                    }
            raise ValueError(f"Paper not found: {paper_title}")
        
        elif author:
            # Get total citations for author
            total_citations = 0
            paper_count = 0
            
            for paper in self.papers_data:
                if any(author.lower() in a.lower() for a in paper["authors"]):
                    total_citations += paper["citations"]
                    paper_count += 1
            
            if paper_count == 0:
                raise ValueError(f"Author not found: {author}")
            
            return {
                "author": author,
                "total_citations": total_citations,
                "paper_count": paper_count
            }
        
        else:
            raise ValueError("Either author or paper_title must be provided")
    
    def _count_papers(self, author: str, venue: str, year: Optional[int]) -> Dict[str, Any]:
        """Count papers by criteria"""
        count = 0
        matching_papers = []
        
        for paper in self.papers_data:
            match = True
            
            if author and not any(author.lower() in a.lower() for a in paper["authors"]):
                match = False
            
            if venue and venue.lower() != paper["venue"].lower():
                match = False
            
            if year and paper["year"] != year:
                match = False
            
            if match:
                count += 1
                matching_papers.append(paper["title"])
        
        return {
            "count": count,
            "criteria": {"author": author, "venue": venue, "year": year},
            "papers": matching_papers
        }

class YelpTool(ToolInterface):
    """Tool for Yelp business and review analysis"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.business_data = self._load_business_data()
    
    def _load_business_data(self) -> List[Dict[str, Any]]:
        """Load business data from JSON"""
        business_file = self.data_dir / "yelp" / "businesses.json"
        if not business_file.exists():
            logger.warning(f"Yelp business file not found: {business_file}")
            return []
        
        try:
            with open(business_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Yelp data: {e}")
            return []
    
    def get_name(self) -> str:
        return "yelp"
    
    def get_functions(self) -> List[str]:
        return ["search_restaurants", "get_business_info", "get_ratings", "compare_restaurants", "get_address", "get_postal_code", "get_coordinates", "get_hours", "check_appointment_required"]
    
    def get_function_schema(self, function: str) -> Dict[str, Any]:
        schemas = {
            "search_restaurants": {
                "description": "Search for restaurants by location, cuisine, or name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City or location"},
                        "cuisine": {"type": "string", "description": "Cuisine type"},
                        "name": {"type": "string", "description": "Restaurant name"}
                    }
                }
            },
            "get_business_info": {
                "description": "Get detailed information about a specific business",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {"type": "string", "description": "Name of the business"}
                    },
                    "required": ["business_name"]
                }
            },
            "get_ratings": {
                "description": "Get ratings for businesses matching criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City or location"},
                        "cuisine": {"type": "string", "description": "Cuisine type"}
                    }
                }
            },
            "get_address": {
                "description": "Get the address of a business",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {"type": "string", "description": "Name of the business"},
                        "postal_code": {"type": "string", "description": "Postal code area"}
                    },
                    "required": ["business_name"]
                }
            },
            "get_postal_code": {
                "description": "Get the postal code of a business",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {"type": "string", "description": "Name of the business"},
                        "city": {"type": "string", "description": "City name"},
                        "state": {"type": "string", "description": "State code"}
                    },
                    "required": ["business_name"]
                }
            },
            "get_coordinates": {
                "description": "Get GPS coordinates of a business",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {"type": "string", "description": "Name of the business"},
                        "postal_code": {"type": "string", "description": "Postal code area"}
                    },
                    "required": ["business_name"]
                }
            },
            "get_hours": {
                "description": "Get operating hours of a business",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {"type": "string", "description": "Name of the business"},
                        "postal_code": {"type": "string", "description": "Postal code area"}
                    },
                    "required": ["business_name"]
                }
            },
            "check_appointment_required": {
                "description": "Check if a business requires appointments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {"type": "string", "description": "Name of the business"},
                        "postal_code": {"type": "string", "description": "Postal code area"}
                    },
                    "required": ["business_name"]
                }
            }
        }
        return schemas.get(function, {})
    
    def call_function(self, function: str, parameters: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            if function == "search_restaurants":
                result = self._search_restaurants(
                    parameters.get("location", ""),
                    parameters.get("cuisine", ""),
                    parameters.get("name", "")
                )
            elif function == "get_business_info":
                result = self._get_business_info(parameters["business_name"])
            elif function == "get_ratings":
                result = self._get_ratings(
                    parameters.get("location", ""),
                    parameters.get("cuisine", "")
                )
            elif function == "get_address":
                result = self._get_address(
                    parameters["business_name"],
                    parameters.get("postal_code", "")
                )
            elif function == "get_postal_code":
                result = self._get_postal_code(
                    parameters["business_name"],
                    parameters.get("city", ""),
                    parameters.get("state", "")
                )
            elif function == "get_coordinates":
                result = self._get_coordinates(
                    parameters["business_name"],
                    parameters.get("postal_code", "")
                )
            elif function == "get_hours":
                result = self._get_hours(
                    parameters["business_name"],
                    parameters.get("postal_code", "")
                )
            elif function == "check_appointment_required":
                result = self._check_appointment_required(
                    parameters["business_name"],
                    parameters.get("postal_code", "")
                )
            else:
                raise ValueError(f"Unknown function: {function}")
                
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _search_restaurants(self, location: str, cuisine: str, name: str) -> List[Dict[str, Any]]:
        """Search restaurants by criteria"""
        results = []
        
        for business in self.business_data:
            match = True
            
            if location and location.lower() not in business["location"].lower():
                match = False
            
            if cuisine and cuisine.lower() != business["cuisine"].lower():
                match = False
            
            if name and name.lower() not in business["name"].lower():
                match = False
            
            if match:
                results.append(business)
        
        return results
    
    def _get_business_info(self, business_name: str) -> Dict[str, Any]:
        """Get specific business information"""
        for business in self.business_data:
            if business_name.lower() in business["name"].lower():
                return business
        
        raise ValueError(f"Business not found: {business_name}")
    
    def _get_ratings(self, location: str, cuisine: str) -> Dict[str, Any]:
        """Get ratings for businesses"""
        matching_businesses = self._search_restaurants(location, cuisine, "")
        
        if not matching_businesses:
            raise ValueError(f"No businesses found for location: {location}, cuisine: {cuisine}")
        
        ratings = [b["rating"] for b in matching_businesses]
        
        return {
            "businesses": matching_businesses,
            "average_rating": sum(ratings) / len(ratings),
            "highest_rating": max(ratings),
            "lowest_rating": min(ratings),
            "count": len(matching_businesses)
        }
    
    def _get_address(self, business_name: str, postal_code: str = "") -> str:
        """Get address of a business"""
        for business in self.business_data:
            name_match = business_name.lower() in business["name"].lower()
            postal_match = not postal_code or business["postal_code"] == postal_code
            
            if name_match and postal_match:
                return business["address"]
        
        raise ValueError(f"Business not found: {business_name}")
    
    def _get_postal_code(self, business_name: str, city: str = "", state: str = "") -> str:
        """Get postal code of a business"""
        for business in self.business_data:
            name_match = business_name.lower() in business["name"].lower()
            city_match = not city or city.lower() in business["city"].lower()
            state_match = not state or state.upper() == business["state"].upper()
            
            if name_match and city_match and state_match:
                return business["postal_code"]
        
        raise ValueError(f"Business not found: {business_name}")
    
    def _get_coordinates(self, business_name: str, postal_code: str = "") -> str:
        """Get coordinates of a business"""
        for business in self.business_data:
            name_match = business_name.lower() in business["name"].lower()
            postal_match = not postal_code or business["postal_code"] == postal_code
            
            if name_match and postal_match:
                return f"{business['latitude']}, {business['longitude']}"
        
        raise ValueError(f"Business not found: {business_name}")
    
    def _get_hours(self, business_name: str, postal_code: str = "") -> str:
        """Get operating hours of a business"""
        for business in self.business_data:
            name_match = business_name.lower() in business["name"].lower()
            postal_match = not postal_code or business["postal_code"] == postal_code
            
            if name_match and postal_match:
                hours = business.get("hours", {})
                if not hours:
                    return "Hours not available"
                
                formatted_hours = []
                for day, time in hours.items():
                    formatted_hours.append(f"{day}: {time}")
                return ", ".join(formatted_hours)
        
        raise ValueError(f"Business not found: {business_name}")
    
    def _check_appointment_required(self, business_name: str, postal_code: str = "") -> str:
        """Check if business requires appointments"""
        for business in self.business_data:
            name_match = business_name.lower() in business["name"].lower()
            postal_match = not postal_code or business["postal_code"] == postal_code
            
            if name_match and postal_match:
                attributes = business.get("attributes", {})
                appointment_only = attributes.get("ByAppointmentOnly", "False")
                return "Yes" if appointment_only == "True" else "No"
        
        raise ValueError(f"Business not found: {business_name}")

class FlightTool(ToolInterface):
    """Tool for flight information and airline data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.flight_data = self._load_flight_data()
    
    def _load_flight_data(self) -> List[Dict[str, Any]]:
        """Load flight data from JSON"""
        flight_file = self.data_dir / "flights" / "flights.json"
        if not flight_file.exists():
            logger.warning(f"Flight data file not found: {flight_file}")
            return []
        
        try:
            with open(flight_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load flight data: {e}")
            return []
    
    def get_name(self) -> str:
        return "flight"
    
    def get_functions(self) -> List[str]:
        return ["search_flights", "get_flight_status", "get_flight_info", "compare_prices"]
    
    def get_function_schema(self, function: str) -> Dict[str, Any]:
        schemas = {
            "search_flights": {
                "description": "Search for flights by origin, destination, or airline",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Origin airport code"},
                        "destination": {"type": "string", "description": "Destination airport code"},
                        "airline": {"type": "string", "description": "Airline name"}
                    }
                }
            },
            "get_flight_status": {
                "description": "Get status of a specific flight",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_number": {"type": "string", "description": "Flight number"}
                    },
                    "required": ["flight_number"]
                }
            },
            "compare_prices": {
                "description": "Compare flight prices for a route",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Origin airport code"},
                        "destination": {"type": "string", "description": "Destination airport code"}
                    },
                    "required": ["origin", "destination"]
                }
            }
        }
        return schemas.get(function, {})
    
    def call_function(self, function: str, parameters: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            if function == "search_flights":
                result = self._search_flights(
                    parameters.get("origin", ""),
                    parameters.get("destination", ""),
                    parameters.get("airline", "")
                )
            elif function == "get_flight_status":
                result = self._get_flight_status(parameters["flight_number"])
            elif function == "compare_prices":
                result = self._compare_prices(parameters["origin"], parameters["destination"])
            else:
                raise ValueError(f"Unknown function: {function}")
                
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _search_flights(self, origin: str, destination: str, airline: str) -> List[Dict[str, Any]]:
        """Search flights by criteria"""
        results = []
        
        for flight in self.flight_data:
            match = True
            
            if origin and origin.upper() != flight["origin"]:
                match = False
            
            if destination and destination.upper() != flight["destination"]:
                match = False
            
            if airline and airline.lower() not in flight["airline"].lower():
                match = False
            
            if match:
                results.append(flight)
        
        return results
    
    def _get_flight_status(self, flight_number: str) -> Dict[str, Any]:
        """Get flight status"""
        for flight in self.flight_data:
            if flight["flight_number"].upper() == flight_number.upper():
                return {
                    "flight_number": flight["flight_number"],
                    "status": flight["status"],
                    "airline": flight["airline"],
                    "origin": flight["origin"],
                    "destination": flight["destination"],
                    "departure_time": flight["departure_time"]
                }
        
        raise ValueError(f"Flight not found: {flight_number}")
    
    def _compare_prices(self, origin: str, destination: str) -> Dict[str, Any]:
        """Compare prices for a route"""
        route_flights = self._search_flights(origin, destination, "")
        
        if not route_flights:
            raise ValueError(f"No flights found for route {origin} to {destination}")
        
        prices = [f["price"] for f in route_flights]
        
        return {
            "route": f"{origin} to {destination}",
            "flights": route_flights,
            "lowest_price": min(prices),
            "highest_price": max(prices),
            "average_price": sum(prices) / len(prices),
            "price_range": max(prices) - min(prices)
        }

class AirbnbTool(ToolInterface):
    """Tool for Airbnb listing analysis"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.listing_data = self._load_listing_data()
    
    def _load_listing_data(self) -> List[Dict[str, Any]]:
        """Load Airbnb listing data from JSON"""
        listing_file = self.data_dir / "airbnb" / "listings.json"
        if not listing_file.exists():
            logger.warning(f"Airbnb listing file not found: {listing_file}")
            return []
        
        try:
            with open(listing_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Airbnb data: {e}")
            return []
    
    def get_name(self) -> str:
        return "airbnb"
    
    def get_functions(self) -> List[str]:
        return ["search_listings", "get_host_info", "get_pricing_info", "get_availability_info"]
    
    def get_function_schema(self, function: str) -> Dict[str, Any]:
        schemas = {
            "search_listings": {
                "description": "Search for Airbnb listings by name, neighborhood, or host",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Listing name"},
                        "neighbourhood": {"type": "string", "description": "Neighborhood name"},
                        "host_name": {"type": "string", "description": "Host name"}
                    }
                }
            },
            "get_host_info": {
                "description": "Get host information for a specific listing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "listing_name": {"type": "string", "description": "Name of the listing"}
                    },
                    "required": ["listing_name"]
                }
            },
            "get_pricing_info": {
                "description": "Get pricing information for listings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighbourhood": {"type": "string", "description": "Neighborhood name"}
                    }
                }
            },
            "get_availability_info": {
                "description": "Get availability information for listings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "listing_id": {"type": "integer", "description": "Listing ID"}
                    }
                }
            }
        }
        return schemas.get(function, {})
    
    def call_function(self, function: str, parameters: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            if function == "search_listings":
                result = self._search_listings(
                    parameters.get("name", ""),
                    parameters.get("neighbourhood", ""),
                    parameters.get("host_name", "")
                )
            elif function == "get_host_info":
                result = self._get_host_info(parameters["listing_name"])
            elif function == "get_pricing_info":
                result = self._get_pricing_info(parameters.get("neighbourhood", ""))
            elif function == "get_availability_info":
                result = self._get_availability_info(parameters.get("listing_id"))
            else:
                raise ValueError(f"Unknown function: {function}")
                
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _search_listings(self, name: str, neighbourhood: str, host_name: str) -> List[Dict[str, Any]]:
        """Search listings by criteria"""
        results = []
        
        for listing in self.listing_data:
            match = True
            
            if name and name.lower() not in listing["name"].lower():
                match = False
            
            if neighbourhood and neighbourhood.lower() != listing["neighbourhood"].lower():
                match = False
            
            if host_name and host_name.lower() not in listing["host_name"].lower():
                match = False
            
            if match:
                results.append(listing)
        
        return results
    
    def _get_host_info(self, listing_name: str) -> Dict[str, Any]:
        """Get host information for a listing"""
        for listing in self.listing_data:
            if listing_name.lower() in listing["name"].lower():
                return {
                    "host_name": listing["host_name"],
                    "listing_name": listing["name"],
                    "neighbourhood": listing["neighbourhood"],
                    "host_listings_count": listing["calculated_host_listings_count"]
                }
        
        raise ValueError(f"Listing not found: {listing_name}")
    
    def _get_pricing_info(self, neighbourhood: str) -> Dict[str, Any]:
        """Get pricing information for a neighbourhood"""
        matching_listings = []
        
        for listing in self.listing_data:
            if not neighbourhood or neighbourhood.lower() == listing["neighbourhood"].lower():
                matching_listings.append(listing)
        
        if not matching_listings:
            raise ValueError(f"No listings found for neighbourhood: {neighbourhood}")
        
        prices = [l["price"] for l in matching_listings]
        
        return {
            "neighbourhood": neighbourhood,
            "average_price": sum(prices) / len(prices),
            "min_price": min(prices),
            "max_price": max(prices),
            "listings": matching_listings
        }
    
    def _get_availability_info(self, listing_id: int) -> Dict[str, Any]:
        """Get availability information for a listing"""
        for listing in self.listing_data:
            if listing["id"] == listing_id:
                return {
                    "id": listing["id"],
                    "name": listing["name"],
                    "availability_365": listing["availability_365"],
                    "minimum_nights": listing["minimum_nights"]
                }
        
        raise ValueError(f"Listing not found: {listing_id}")

class AgendaTool(ToolInterface):
    """Tool for calendar and agenda analysis"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.event_data = self._load_event_data()
    
    def _load_event_data(self) -> List[Dict[str, Any]]:
        """Load event data from JSON"""
        event_file = self.data_dir / "agenda" / "events.json"
        if not event_file.exists():
            logger.warning(f"Agenda data file not found: {event_file}")
            return []
        
        try:
            with open(event_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load agenda data: {e}")
            return []
    
    def get_name(self) -> str:
        return "agenda"
    
    def get_functions(self) -> List[str]:
        return ["get_events", "search_events", "get_person_schedule"]
    
    def get_function_schema(self, function: str) -> Dict[str, Any]:
        schemas = {
            "get_events": {
                "description": "Get events for a person on a specific date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {"type": "string", "description": "Person's name"},
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                    },
                    "required": ["person", "date"]
                }
            },
            "search_events": {
                "description": "Search events by keyword or date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Event name or keyword"},
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                    }
                }
            },
            "get_person_schedule": {
                "description": "Get all events for a person",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {"type": "string", "description": "Person's name"}
                    },
                    "required": ["person"]
                }
            }
        }
        return schemas.get(function, {})
    
    def call_function(self, function: str, parameters: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            if function == "get_events":
                result = self._get_events(parameters["person"], parameters["date"])
            elif function == "search_events":
                result = self._search_events(
                    parameters.get("query", ""),
                    parameters.get("date", "")
                )
            elif function == "get_person_schedule":
                result = self._get_person_schedule(parameters["person"])
            else:
                raise ValueError(f"Unknown function: {function}")
                
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.get_name(),
                function=function,
                parameters=parameters,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _get_events(self, person: str, date: str) -> List[Dict[str, Any]]:
        """Get events for person on specific date"""
        events = []
        
        for event in self.event_data:
            if event["person"].lower() == person.lower() and event["date"] == date:
                events.append(event)
        
        return events
    
    def _search_events(self, query: str, date: str) -> List[Dict[str, Any]]:
        """Search events by query and/or date"""
        events = []
        
        for event in self.event_data:
            match = True
            
            if query and query.lower() not in event["event"].lower():
                match = False
            
            if date and event["date"] != date:
                match = False
            
            if match:
                events.append(event)
        
        return events
    
    def _get_person_schedule(self, person: str) -> Dict[str, Any]:
        """Get all events for a person"""
        person_events = []
        
        for event in self.event_data:
            if event["person"].lower() == person.lower():
                person_events.append(event)
        
        # Sort by date
        person_events.sort(key=lambda x: x["date"])
        
        return {
            "person": person,
            "events": person_events,
            "event_count": len(person_events)
        }

# Factory function to create and register tools
def create_and_register_tools(data_dir: str = "data/toolqa"):
    """Create all ToolQA tools and register them with the unified system"""
    from .unified_tool_system import get_unified_tool_system
    
    system = get_unified_tool_system()
    
    # Create tools
    calculator = CalculatorTool()
    coffee = CoffeeTool(data_dir)
    dblp = DBLPTool(data_dir)
    yelp = YelpTool(data_dir)
    flight = FlightTool(data_dir)
    airbnb = AirbnbTool(data_dir)
    agenda = AgendaTool(data_dir)
    
    # Register tools
    system.router.register_tool(calculator)
    system.router.register_tool(coffee)
    system.router.register_tool(dblp)
    system.router.register_tool(yelp)
    system.router.register_tool(flight)
    system.router.register_tool(airbnb)
    system.router.register_tool(agenda)
    
    logger.info("Created and registered ToolQA tools")
    logger.info(f"Registered tools: {list(system.router.tools.keys())}")
    return system

if __name__ == "__main__":
    # Test calculator tool
    calc = CalculatorTool()
    
    print("🧮 Calculator Tool Test:")
    test_expressions = [
        "2 + 3 * 4",
        "(10 - 5) / 2", 
        "7 * 8 - 3"
    ]
    
    for expr in test_expressions:
        result = calc.call_function("calculate", {"expression": expr})
        print(f"   {expr} = {result.result}")
    
    # Test equation solving
    equation_tests = ["x + 5 = 10", "2*x = 8", "x - 3 = 7"]
    for eq in equation_tests:
        result = calc.call_function("solve_equation", {"equation": eq})
        if result.success:
            print(f"   {eq} → x = {result.result['x']}")
        else:
            print(f"   {eq} → Error: {result.error_message}")
    
    print("\n☕ Coffee Tool Test:")
    # Note: This will only work if coffee.csv exists in data/toolqa/
    coffee = CoffeeTool("data/toolqa")
    print(f"   Loaded {len(coffee.price_data)} coffee price records")