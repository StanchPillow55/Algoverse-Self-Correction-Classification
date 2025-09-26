#!/usr/bin/env python3
"""
ToolQA Tool Specifications

Defines the tools and data sources needed for each ToolQA domain based on the 
official ToolQA repository structure and requirements.

Domains:
1. Coffee - Time series tabular data (daily coffee prices)
2. DBLP - Citation network graph data (academic collaborations)
3. Yelp - Spatial tabular data (business reviews and locations)
4. Flight - Temporal tabular data (flight operations and delays)
5. Airbnb - Tabular data (rental listings and availability)
6. Agenda - Text corpus (personal scheduling events)
7. GSM8K - Mathematical computation (word problems)
8. SciREX - Text corpus (research papers and metrics)
9. Genda - Event scheduling data (similar to agenda)
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

class ToolType(Enum):
    """Types of tools available for ToolQA"""
    SQL_DATABASE = "sql_database"
    GRAPH_DATABASE = "graph_database" 
    TEXT_RETRIEVER = "text_retriever"
    CALCULATOR = "calculator"
    TABLE_PROCESSOR = "table_processor"
    TIME_SERIES_ANALYZER = "time_series_analyzer"

@dataclass
class ToolSpec:
    """Specification for a ToolQA tool"""
    name: str
    tool_type: ToolType
    description: str
    required_data_files: List[str]
    functions: List[str]  # Available functions for this tool
    input_format: str
    output_format: str

class ToolQADomainSpecs:
    """Complete specifications for all ToolQA domains"""
    
    @staticmethod
    def get_domain_specs() -> Dict[str, Dict[str, Any]]:
        """Get comprehensive tool specifications for each ToolQA domain"""
        
        return {
            "coffee": {
                "description": "Daily coffee price time series data (2000-2022)",
                "data_source": "Kaggle daily coffee prices",
                "knowledge_format": "Tabular Database",
                "knowledge_size": 5746,
                "tools": [
                    ToolSpec(
                        name="coffee_price_analyzer",
                        tool_type=ToolType.TIME_SERIES_ANALYZER,
                        description="Analyze coffee price trends, ranges, changes over time",
                        required_data_files=["coffee.csv"],
                        functions=[
                            "get_price_range",
                            "get_average_price", 
                            "get_highest_price",
                            "get_lowest_price",
                            "get_price_change",
                            "get_bullish_bearish_days",
                            "count_price_increases"
                        ],
                        input_format="date_range, price_type",
                        output_format="numeric_value, percentage, or classification"
                    )
                ],
                "sample_queries": [
                    "What was the coffee price range from 2000-01-03 to 2020-10-07?",
                    "Was 2013-07-26 a bearish or bullish day for coffee price?"
                ]
            },
            
            "dblp": {
                "description": "Academic citation network and collaboration data",
                "data_source": "DBLP Citation Network V14",
                "knowledge_format": "Graph",
                "knowledge_size": 553320,
                "tools": [
                    ToolSpec(
                        name="dblp_graph_analyzer",
                        tool_type=ToolType.GRAPH_DATABASE,
                        description="Query DBLP citation network for collaborations, venues, papers",
                        required_data_files=["dblp_citations.json", "dblp_papers.json"],
                        functions=[
                            "get_collaboration_venue",
                            "count_common_collaborators",
                            "get_paper_count",
                            "get_paper_pages",
                            "get_paper_citations",
                            "get_author_organization"
                        ],
                        input_format="author_names, paper_titles",
                        output_format="venue_name, count, organization"
                    )
                ],
                "sample_queries": [
                    "What venue did Eric F. Vermote and J.-C. Roger collaborate most in?",
                    "How many common collaborators does Deli Zhao have with Jiapeng Zhu?"
                ]
            },
            
            "yelp": {
                "description": "Business reviews, ratings, and location data",
                "data_source": "Yelp Academic Dataset",
                "knowledge_format": "Tabular Database",
                "knowledge_size": 150346,
                "tools": [
                    ToolSpec(
                        name="yelp_business_analyzer",
                        tool_type=ToolType.SQL_DATABASE,
                        description="Query Yelp business data for ratings, reviews, locations",
                        required_data_files=["yelp_business.json"],
                        functions=[
                            "get_average_review_count_radius",
                            "get_average_star_rating",
                            "get_highest_rated_business",
                            "get_most_reviews_business", 
                            "get_business_categories",
                            "get_business_hours",
                            "get_postal_code_businesses",
                            "get_business_location"
                        ],
                        input_format="business_name, location, category, radius",
                        output_format="rating, count, business_name, categories"
                    )
                ],
                "sample_queries": [
                    "What is the average review counts of businesses within a 5-mile radius from Paper Source?",
                    "Which Movers business has the highest review count in Cherry Hill, NJ?"
                ]
            },
            
            "flight": {
                "description": "Flight operations, delays, and cancellations (2022)",
                "data_source": "Kaggle Flight Delay Dataset 2018-2022", 
                "knowledge_format": "Tabular Database",
                "knowledge_size": 4078318,
                "tools": [
                    ToolSpec(
                        name="flight_operations_analyzer",
                        tool_type=ToolType.SQL_DATABASE,
                        description="Query flight data for operations, delays, cancellations",
                        required_data_files=["Combined_Flights_2022.csv"],
                        functions=[
                            "check_flight_cancelled",
                            "get_total_flights_by_airline",
                            "get_delay_percentage",
                            "count_flights_by_distance",
                            "get_taxi_time"
                        ],
                        input_format="flight_number, airline, date, origin, destination",
                        output_format="boolean, count, percentage, minutes"
                    )
                ],
                "sample_queries": [
                    "Was the flight AA5566 from CLT to LEX cancelled on 2022-01-20?",
                    "What is the total number of flights operated by Allegiant Air on 2022-04-12?"
                ]
            },
            
            "airbnb": {
                "description": "Airbnb rental listings, availability, and pricing",
                "data_source": "Airbnb Open Data",
                "knowledge_format": "Tabular Database", 
                "knowledge_size": 102599,
                "tools": [
                    ToolSpec(
                        name="airbnb_listing_analyzer",
                        tool_type=ToolType.SQL_DATABASE,
                        description="Query Airbnb listings for availability, pricing, reviews",
                        required_data_files=["Airbnb_Open_data.csv"],
                        functions=[
                            "get_listing_review_rate",
                            "get_availability_days",
                            "get_lowest_price_listing",
                            "get_most_expensive_listing",
                            "get_average_reviews_per_month",
                            "get_host_name",
                            "get_total_price_nights"
                        ],
                        input_format="listing_id, location, room_type, nights",
                        output_format="rate, days, listing_name, price, host_name"
                    )
                ],
                "sample_queries": [
                    "What is the review rate number of Trendy Chelsea 1BR w/ Balcony?",
                    "How many days are Large BR in Spacious Artist Loft available during a year?"
                ]
            },
            
            "agenda": {
                "description": "Personal scheduling and event attendance data", 
                "data_source": "Generated agenda events",
                "knowledge_format": "Pure-Text Corpus",
                "knowledge_size": 10000,
                "tools": [
                    ToolSpec(
                        name="agenda_event_retriever",
                        tool_type=ToolType.TEXT_RETRIEVER,
                        description="Retrieve personal agenda events and attendance info",
                        required_data_files=["agenda_events.jsonl"],
                        functions=[
                            "get_person_activity",
                            "get_event_attendee", 
                            "get_event_time",
                            "get_event_location",
                            "get_event_duration"
                        ],
                        input_format="person_name, event_name, date, time_range",
                        output_format="activity, person, time, location, duration"
                    )
                ],
                "sample_queries": [
                    "Who attended Horse race between 2:00 PM and 5:00 PM on 2022/08/14?",
                    "What did Elizabeth do from 7:30 PM to 10:00 PM on 2022/06/29?"
                ]
            },
            
            "genda": {
                "description": "Event scheduling and counting data (similar to agenda)",
                "data_source": "Generated scheduling events", 
                "knowledge_format": "Pure-Text Corpus",
                "knowledge_size": 10000,
                "tools": [
                    ToolSpec(
                        name="genda_event_counter",
                        tool_type=ToolType.TEXT_RETRIEVER,
                        description="Count and analyze scheduling events by date and person",
                        required_data_files=["agenda_events.jsonl"],
                        functions=[
                            "count_events_by_date",
                            "count_person_scheduled_dates",
                            "get_person_events",
                            "suggest_meeting_times"
                        ],
                        input_format="date, person_name, time_range",
                        output_format="count, dates, events, time_slots"
                    )
                ],
                "sample_queries": [
                    "How many events happen on 2022/03/09 in the agenda table?",
                    "How many dates in the agenda table have Christopher scheduled?"
                ]
            },
            
            "gsm8k": {
                "description": "Grade school math word problems requiring computation",
                "data_source": "GSM8K dataset",
                "knowledge_format": "Professional Ability",
                "knowledge_size": None,
                "tools": [
                    ToolSpec(
                        name="math_calculator",
                        tool_type=ToolType.CALCULATOR,
                        description="Perform mathematical calculations for word problems",
                        required_data_files=[],
                        functions=[
                            "calculate_basic_arithmetic",
                            "solve_algebra",
                            "calculate_percentages", 
                            "solve_word_problem",
                            "calculate_compound_interest",
                            "solve_optimization"
                        ],
                        input_format="mathematical_expression, word_problem",
                        output_format="numeric_result"
                    )
                ],
                "sample_queries": [
                    "A farmer extracts 5 liters of milk a day from a cow...",
                    "Aiden and 12 of his friends are going to see a film..."
                ]
            },
            
            "scirex": {
                "description": "Scientific research papers and experimental results",
                "data_source": "SciREX Dataset", 
                "knowledge_format": "Pure-Text Corpus",
                "knowledge_size": 438,
                "tools": [
                    ToolSpec(
                        name="scirex_paper_retriever",
                        tool_type=ToolType.TEXT_RETRIEVER,
                        description="Retrieve scientific paper results, metrics, and evaluations",
                        required_data_files=["train.jsonl", "val.jsonl", "test.jsonl"],
                        functions=[
                            "get_method_score",
                            "get_method_metrics",
                            "get_method_datasets",
                            "get_evaluation_results"
                        ],
                        input_format="method_name, dataset_name, task_type, metric_type",
                        output_format="score, metrics_list, datasets_list"
                    )
                ],
                "sample_queries": [
                    "What is the corresponding Top_5_Accuracy score of the ResNeXt-101 method?",
                    "On what metrics is the LapSRN method evaluated on Urban100?"
                ]
            }
        }

# Sample usage and validation
if __name__ == "__main__":
    specs = ToolQADomainSpecs.get_domain_specs()
    
    print("🛠️ ToolQA Domain Specifications Summary:")
    print("=" * 60)
    
    for domain, spec in specs.items():
        print(f"\n📊 {domain.upper()}")
        print(f"   Description: {spec['description']}")
        print(f"   Knowledge Size: {spec['knowledge_size']}")
        print(f"   Tools: {len(spec['tools'])}")
        for tool in spec['tools']:
            print(f"     - {tool.name} ({len(tool.functions)} functions)")
    
    print(f"\n✅ Total domains: {len(specs)}")
    total_tools = sum(len(spec['tools']) for spec in specs.values())
    print(f"✅ Total tools: {total_tools}")