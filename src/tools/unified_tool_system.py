#!/usr/bin/env python3
"""
Unified Tool Integration Architecture for ToolQA

This module provides a unified interface for tool-augmented reasoning that works
across all model providers (Anthropic, OpenAI, etc.) and both ensemble and 
multi-turn pipelines.

Key Features:
- Provider-agnostic tool calling (function calling for OpenAI, tools for Anthropic)
- Automatic tool routing based on question domain detection
- Tool result caching for efficiency
- Error handling and fallback mechanisms
- Integration with existing multi-turn and ensemble pipelines
"""

import json
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from pathlib import Path

# Import tool specifications
from .toolqa_specifications import ToolQADomainSpecs, ToolType

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"

@dataclass
class ToolCall:
    """Represents a tool call request"""
    tool_name: str
    function: str
    parameters: Dict[str, Any]
    domain: str

@dataclass 
class ToolResult:
    """Represents a tool call result"""
    tool_name: str
    function: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

class ToolInterface(ABC):
    """Abstract interface for ToolQA tools"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the tool name"""
        pass
        
    @abstractmethod
    def get_functions(self) -> List[str]:
        """Return list of available functions"""
        pass
        
    @abstractmethod
    def call_function(self, function: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a function with given parameters"""
        pass
        
    @abstractmethod
    def get_function_schema(self, function: str) -> Dict[str, Any]:
        """Return OpenAI/Anthropic function schema for the function"""
        pass

class DomainDetector:
    """Detects the ToolQA domain from a question"""
    
    def __init__(self):
        self.domain_keywords = {
            "coffee": ["coffee", "price", "bullish", "bearish", "commodity"],
            "dblp": ["paper", "author", "collaborate", "citation", "venue", "research"],
            "yelp": ["business", "review", "rating", "restaurant", "postal code", "star rating"],
            "flight": ["flight", "airline", "cancelled", "delayed", "airport", "taxi"],
            "airbnb": ["airbnb", "listing", "host", "apartment", "bedroom", "rental"],
            "agenda": ["attend", "activity", "event", "meeting", "schedule"],
            "genda": ["events happen", "scheduled", "agenda table"],
            "gsm8k": ["farmer", "students", "money", "cost", "calculate", "math"],
            "scirex": ["method", "dataset", "accuracy", "score", "evaluation", "metrics"]
        }
    
    def detect_domain(self, question: str) -> str:
        """Detect the most likely domain for a question"""
        question_lower = question.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            # Fallback to gsm8k for math-like questions
            if any(word in question_lower for word in ["how many", "calculate", "find", "what is"]):
                return "gsm8k"
            return "unknown"
        
        return max(domain_scores.items(), key=lambda x: x[1])[0]

class ToolCallRouter:
    """Routes tool calls to appropriate implementations"""
    
    def __init__(self):
        self.tools: Dict[str, ToolInterface] = {}
        self.domain_detector = DomainDetector()
        self.domain_specs = ToolQADomainSpecs.get_domain_specs()
        self.cache: Dict[str, ToolResult] = {}
    
    def register_tool(self, tool: ToolInterface):
        """Register a tool implementation"""
        self.tools[tool.get_name()] = tool
        logger.info(f"Registered tool: {tool.get_name()}")
    
    def get_available_tools_for_domain(self, domain: str) -> List[str]:
        """Get available tools for a specific domain"""
        if domain not in self.domain_specs:
            return []
        
        # Map domain to actual tool implementations
        domain_tool_mapping = {
            "coffee": ["coffee"],
            "gsm8k": ["calculator"],
            "dblp": ["dblp"],
            "yelp": ["yelp"],
            "flight": ["flight"],
            "airbnb": ["airbnb"],
            "agenda": ["agenda"],
            "genda": ["genda"],
            "scirex": ["scirex"]
        }
        
        available_tools = []
        if domain in domain_tool_mapping:
            for tool_name in domain_tool_mapping[domain]:
                if tool_name in self.tools:
                    available_tools.append(tool_name)
        
        return available_tools
    
    def detect_required_tools(self, question: str) -> List[str]:
        """Detect which tools are needed for a question"""
        domain = self.domain_detector.detect_domain(question)
        return self.get_available_tools_for_domain(domain)
    
    def _get_cache_key(self, tool_call: ToolCall) -> str:
        """Generate cache key for tool call"""
        data = f"{tool_call.tool_name}:{tool_call.function}:{json.dumps(tool_call.parameters, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def execute_tool_call(self, tool_call: ToolCall, use_cache: bool = True) -> ToolResult:
        """Execute a tool call with caching"""
        
        # Check cache first
        cache_key = self._get_cache_key(tool_call)
        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for tool call: {tool_call.tool_name}.{tool_call.function}")
            return self.cache[cache_key]
        
        # Execute tool call
        if tool_call.tool_name not in self.tools:
            result = ToolResult(
                tool_name=tool_call.tool_name,
                function=tool_call.function,
                parameters=tool_call.parameters,
                result=None,
                success=False,
                error_message=f"Tool {tool_call.tool_name} not found"
            )
        else:
            try:
                tool = self.tools[tool_call.tool_name]
                result = tool.call_function(tool_call.function, tool_call.parameters)
            except Exception as e:
                logger.error(f"Error executing tool call: {e}")
                result = ToolResult(
                    tool_name=tool_call.tool_name,
                    function=tool_call.function,
                    parameters=tool_call.parameters,
                    result=None,
                    success=False,
                    error_message=str(e)
                )
        
        # Cache result
        if use_cache:
            self.cache[cache_key] = result
        
        return result

class ModelProviderAdapter(ABC):
    """Abstract adapter for different model providers"""
    
    @abstractmethod
    def format_tools_for_model(self, tools: List[ToolInterface]) -> List[Dict[str, Any]]:
        """Format tools for the specific model provider"""
        pass
    
    @abstractmethod
    def parse_tool_calls_from_response(self, response: Any) -> List[ToolCall]:
        """Parse tool calls from model response"""
        pass
    
    @abstractmethod
    def format_tool_results_for_model(self, results: List[ToolResult]) -> str:
        """Format tool results for model consumption"""
        pass

class AnthropicAdapter(ModelProviderAdapter):
    """Adapter for Anthropic Claude models"""
    
    def format_tools_for_model(self, tools: List[ToolInterface]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic's tools API"""
        formatted_tools = []
        
        for tool in tools:
            for function_name in tool.get_functions():
                schema = tool.get_function_schema(function_name)
                input_schema = schema.get("parameters", {})
                
                # Ensure input_schema has required 'type' field for Anthropic
                if "type" not in input_schema:
                    input_schema["type"] = "object"
                
                formatted_tools.append({
                    "name": f"{tool.get_name()}_{function_name}",
                    "description": schema.get("description", ""),
                    "input_schema": input_schema
                })
        
        return formatted_tools
    
    def parse_tool_calls_from_response(self, response: Any) -> List[ToolCall]:
        """Parse tool calls from Claude response"""
        tool_calls = []
        
        if hasattr(response, 'content'):
            for content_block in response.content:
                if content_block.type == "tool_use":
                    # Parse tool name and function
                    full_name = content_block.name
                    if '_' in full_name:
                        tool_name, function = full_name.rsplit('_', 1)
                    else:
                        tool_name, function = full_name, "default"
                    
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        function=function,
                        parameters=content_block.input,
                        domain="auto-detected"
                    ))
        
        return tool_calls
    
    def format_tool_results_for_model(self, results: List[ToolResult]) -> str:
        """Format tool results for Claude"""
        formatted_results = []
        for result in results:
            if result.success:
                formatted_results.append(f"Tool {result.tool_name}.{result.function} returned: {result.result}")
            else:
                formatted_results.append(f"Tool {result.tool_name}.{result.function} failed: {result.error_message}")
        
        return "\n".join(formatted_results)

class OpenAIAdapter(ModelProviderAdapter):
    """Adapter for OpenAI models with function calling"""
    
    def format_tools_for_model(self, tools: List[ToolInterface]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI's function calling API"""
        formatted_tools = []
        
        for tool in tools:
            for function_name in tool.get_functions():
                schema = tool.get_function_schema(function_name)
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": f"{tool.get_name()}_{function_name}",
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", {})
                    }
                })
        
        return formatted_tools
    
    def parse_tool_calls_from_response(self, response: Any) -> List[ToolCall]:
        """Parse tool calls from OpenAI response"""
        tool_calls = []
        
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    # Parse tool name and function
                    full_name = tool_call.function.name
                    if '_' in full_name:
                        tool_name, function = full_name.rsplit('_', 1)
                    else:
                        tool_name, function = full_name, "default"
                    
                    parameters = json.loads(tool_call.function.arguments)
                    
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        function=function,
                        parameters=parameters,
                        domain="auto-detected"
                    ))
        
        return tool_calls
    
    def format_tool_results_for_model(self, results: List[ToolResult]) -> str:
        """Format tool results for OpenAI"""
        formatted_results = []
        for result in results:
            if result.success:
                formatted_results.append(f"Function {result.tool_name}.{result.function} returned: {result.result}")
            else:
                formatted_results.append(f"Function {result.tool_name}.{result.function} failed: {result.error_message}")
        
        return "\n".join(formatted_results)

class UnifiedToolSystem:
    """Main unified tool system for ToolQA integration"""
    
    def __init__(self, data_dir: str = "data/toolqa"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.router = ToolCallRouter()
        self.adapters = {
            ModelProvider.ANTHROPIC: AnthropicAdapter(),
            ModelProvider.OPENAI: OpenAIAdapter()
        }
        
        # Tool registry will be populated by tool implementations
        self.initialized = False
    
    def initialize_tools(self):
        """Initialize all ToolQA tools (will be implemented in next step)"""
        # This will be called by the tool implementations
        self.initialized = True
        logger.info("ToolQA unified tool system initialized")
    
    def get_tools_for_question(self, question: str) -> List[str]:
        """Get required tools for a question"""
        return self.router.detect_required_tools(question)
    
    def get_formatted_tools_for_provider(self, provider: ModelProvider, question: str) -> List[Dict[str, Any]]:
        """Get tools formatted for a specific model provider"""
        if provider not in self.adapters:
            raise ValueError(f"Unsupported provider: {provider}")
        
        required_tool_names = self.get_tools_for_question(question)
        required_tools = [self.router.tools[name] for name in required_tool_names if name in self.router.tools]
        
        return self.adapters[provider].format_tools_for_model(required_tools)
    
    def execute_tool_calls_from_response(self, provider: ModelProvider, response: Any) -> List[ToolResult]:
        """Execute tool calls parsed from model response"""
        if provider not in self.adapters:
            raise ValueError(f"Unsupported provider: {provider}")
        
        adapter = self.adapters[provider]
        tool_calls = adapter.parse_tool_calls_from_response(response)
        
        results = []
        for tool_call in tool_calls:
            result = self.router.execute_tool_call(tool_call)
            results.append(result)
        
        return results
    
    def format_tool_results_for_provider(self, provider: ModelProvider, results: List[ToolResult]) -> str:
        """Format tool results for a specific model provider"""
        if provider not in self.adapters:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return self.adapters[provider].format_tool_results_for_model(results)
    
    def is_tool_augmented_question(self, question: str) -> bool:
        """Check if a question requires tool augmentation"""
        required_tools = self.get_tools_for_question(question)
        return len(required_tools) > 0
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "initialized": self.initialized,
            "registered_tools": len(self.router.tools),
            "cached_results": len(self.router.cache),
            "supported_providers": list(self.adapters.keys()),
            "data_directory": str(self.data_dir)
        }

# Global instance
_unified_tool_system = None

def get_unified_tool_system() -> UnifiedToolSystem:
    """Get the global unified tool system instance"""
    global _unified_tool_system
    if _unified_tool_system is None:
        _unified_tool_system = UnifiedToolSystem()
    return _unified_tool_system

# Example usage and testing
if __name__ == "__main__":
    # Test domain detection
    detector = DomainDetector()
    test_questions = [
        "What was the coffee price range from 2000-01-03 to 2020-10-07?",
        "What venue did Eric F. Vermote and J.-C. Roger collaborate most in the DBLP citation network?",
        "Which Movers business has the highest review count in Cherry Hill, NJ?",
        "Was the flight AA5566 from CLT to LEX cancelled on 2022-01-20?",
        "A farmer extracts 5 liters of milk a day from a cow..."
    ]
    
    print("🔍 Domain Detection Test:")
    for question in test_questions:
        domain = detector.detect_domain(question)
        print(f"   '{question[:50]}...' → {domain}")
    
    # Test tool system
    system = get_unified_tool_system()
    print(f"\n🛠️ Tool System Stats:")
    stats = system.get_system_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")