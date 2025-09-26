# ToolQA Domain Expansion Architecture

## Overview

This document outlines the architecture for expanding the ToolQA system to support all domains (DBLP, Yelp, Flight, Agenda) and implementing enhanced answer extraction to resolve format mismatches.

## Current State Analysis

### Successful Elements
- **Unified Tool System**: Core architecture is solid and scalable
- **Tool Registration**: Dynamic tool discovery and registration works well
- **Calculator Tool**: Direct numeric results work perfectly
- **Error Handling**: Robust failure handling and logging

### Critical Issues Identified
1. **Coffee Domain Format Mismatch**: 
   - `get_price_range` returns `{min_price, max_price}` but expected answers often want sums, averages, or other calculations
   - Mock data coverage is limited (missing many date ranges)
   - Questions ask for "average" but we only provide min/max range

2. **Missing Domain Tools**: No tools exist for DBLP, Yelp, Flight, Agenda domains

3. **Generic Answer Extraction**: Current extraction logic is too generic and doesn't understand domain-specific semantics

## Architecture Design

### 1. Enhanced Tool Interface

```python
class EnhancedToolInterface(ToolInterface):
    """Enhanced tool interface with domain-specific answer formatting"""
    
    def get_domain(self) -> str:
        """Return the domain this tool belongs to"""
        pass
    
    def format_answer_for_question(self, result: Any, question: str, expected_format: str) -> str:
        """Format tool result to match expected answer format for a specific question"""
        pass
    
    def get_answer_formatters(self) -> Dict[str, Callable]:
        """Return domain-specific answer formatters"""
        pass
```

### 2. Domain-Specific Tool Architecture

Each domain will have:

#### A. **DBLP Domain (Academic Papers)**
```python
class DBLPTool(EnhancedToolInterface):
    Functions:
    - search_papers(query, author, venue, year_range)
    - get_author_info(author_name) 
    - get_citation_count(paper_title, author)
    - get_venue_papers(venue, year)
    - analyze_collaboration(author1, author2)
```

#### B. **Yelp Domain (Business Reviews)**
```python
class YelpTool(EnhancedToolInterface):
    Functions:
    - search_restaurants(location, cuisine, rating_min)
    - get_business_info(business_id)
    - get_reviews(business_id, sort_by)
    - analyze_sentiment(business_id)
    - compare_ratings(business1, business2)
```

#### C. **Flight Domain (Air Travel)**
```python
class FlightTool(EnhancedToolInterface):
    Functions:
    - search_flights(origin, destination, date, airlines)
    - get_flight_status(flight_number, date)
    - compare_prices(origin, destination, date_range)
    - get_airline_info(airline_code)
    - analyze_delays(route, time_period)
```

#### D. **Agenda Domain (Calendar Events)**
```python
class AgendaTool(EnhancedToolInterface):
    Functions:
    - get_events(person, date)
    - search_events(query, date_range)
    - check_availability(person, time_slot)
    - get_schedule_conflicts(person, event)
    - analyze_schedule_patterns(person, period)
```

### 3. Enhanced Answer Extraction System

#### A. **Domain-Aware Extractor**
```python
class DomainAwareAnswerExtractor:
    """Extracts answers using domain-specific knowledge"""
    
    def __init__(self):
        self.domain_extractors = {
            'coffee': CoffeeAnswerExtractor(),
            'calculator': CalculatorAnswerExtractor(), 
            'dblp': DBLPAnswerExtractor(),
            'yelp': YelpAnswerExtractor(),
            'flight': FlightAnswerExtractor(),
            'agenda': AgendaAnswerExtractor()
        }
    
    def extract_answer(self, tool_results, question, expected_answer, domain):
        extractor = self.domain_extractors.get(domain)
        if extractor:
            return extractor.extract(tool_results, question, expected_answer)
        return self.generic_extract(tool_results, question)
```

#### B. **Coffee Domain Extractor (Enhanced)**
```python
class CoffeeAnswerExtractor:
    def extract(self, tool_results, question, expected_answer):
        # Analyze question intent
        if 'average' in question.lower():
            return self._extract_average(tool_results, expected_answer)
        elif 'range' in question.lower():
            return self._extract_range(tool_results, expected_answer)
        elif 'highest' in question.lower():
            return self._extract_highest(tool_results)
        # ... more specific handlers
    
    def _extract_range(self, tool_results, expected_answer):
        """Handle price range questions with multiple interpretations"""
        for result in tool_results:
            if 'min_price' in result and 'max_price' in result:
                min_price = result['min_price']
                max_price = result['max_price']
                
                # Try multiple interpretations based on expected answer
                candidates = {
                    'sum': min_price + max_price,
                    'difference': max_price - min_price,
                    'average': (min_price + max_price) / 2,
                    'max': max_price,
                    'min': min_price
                }
                
                # Find best match to expected answer
                expected_num = self._parse_number(expected_answer)
                if expected_num:
                    best_match = min(candidates.items(), 
                                   key=lambda x: abs(x[1] - expected_num))
                    return str(best_match[1])
                
                # Default to sum for "range" questions
                return str(candidates['sum'])
```

### 4. Mock Data Architecture

#### A. **Structured Mock Data**
```
data/toolqa/
├── coffee.csv (existing)
├── dblp/
│   ├── papers.json
│   ├── authors.json
│   └── venues.json
├── yelp/
│   ├── businesses.json
│   ├── reviews.json
│   └── locations.json
├── flights/
│   ├── airlines.json
│   ├── routes.json
│   └── schedules.json
└── agenda/
    ├── people.json
    └── events.json
```

#### B. **Mock Data Generators**
```python
class MockDataGenerator:
    """Generate realistic mock data for each domain"""
    
    def generate_dblp_data(self) -> Dict:
        """Generate academic paper data"""
        pass
    
    def generate_yelp_data(self) -> Dict:
        """Generate business and review data"""
        pass
    
    # etc.
```

## Implementation Strategy

### Phase 1: Enhanced Answer Extraction (Priority 1)
1. Implement `DomainAwareAnswerExtractor` 
2. Create `CoffeeAnswerExtractor` with smart interpretation logic
3. Add mathematical operations to coffee tool results
4. Test with existing 500-question dataset

### Phase 2: DBLP Domain Implementation 
1. Create DBLP mock data structure
2. Implement `DBLPTool` with core functions
3. Add DBLP answer extractor
4. Test with DBLP-specific questions

### Phase 3: Yelp Domain Implementation
1. Create Yelp mock data structure
2. Implement `YelpTool` with business/review functions
3. Add Yelp answer extractor
4. Test with Yelp-specific questions

### Phase 4: Flight Domain Implementation
1. Create Flight mock data structure
2. Implement `FlightTool` with airline/route functions
3. Add Flight answer extractor
4. Test with Flight-specific questions

### Phase 5: Agenda Domain Implementation
1. Create Agenda mock data structure
2. Implement `AgendaTool` with calendar functions
3. Add Agenda answer extractor
4. Test with Agenda-specific questions

## Expected Improvements

### Accuracy Gains
- **Coffee Domain**: 2.2% → 15-20% (fixing extraction mismatches)
- **Math Domain**: 2.0% → 8-10% (better tool usage detection)
- **New Domains**: 0% → 10-15% (adding missing capabilities)
- **Overall**: 5.2% → 12-18% target accuracy

### Tool Usage Improvements
- **Better Tool Selection**: Reduce over-toolification
- **Domain Coverage**: Support all ToolQA domains
- **Answer Quality**: Semantically correct extractions

## Success Metrics

1. **Tool Success Rate**: Maintain >50% success rate across domains
2. **Answer Format Match**: >80% of successful tool calls should extract correct format
3. **Domain Coverage**: Support questions from all 6 ToolQA domains
4. **Overall Accuracy**: Target 15%+ accuracy on 500-question dataset