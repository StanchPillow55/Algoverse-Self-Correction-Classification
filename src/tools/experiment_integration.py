#!/usr/bin/env python3
"""
Tool Integration for ToolQA Experiments

This module integrates the unified tool system with existing experiment runners
to enable tool-augmented reasoning for ToolQA benchmark evaluation.

Key features:
- Integration with multi-turn and ensemble pipelines  
- Tool call execution for Anthropic and OpenAI models
- Enhanced answer extraction with tool results
- Tool usage metrics and logging
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from .unified_tool_system import (
    UnifiedToolSystem, 
    get_unified_tool_system, 
    ModelProvider, 
    ToolResult
)
from .domain_tools import create_and_register_tools

logger = logging.getLogger(__name__)

class ToolAugmentedExperimentRunner:
    """Enhanced experiment runner with tool augmentation capabilities"""
    
    def __init__(self, data_dir: str = "data/toolqa"):
        self.tool_system = get_unified_tool_system()
        self.data_dir = Path(data_dir)
        
        # Initialize tools
        create_and_register_tools(str(self.data_dir))
        self.tool_system.initialize_tools()
        
        # Metrics tracking
        self.tool_usage_stats = {
            "questions_with_tools": 0,
            "total_tool_calls": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "tools_used": {},
            "domains_processed": {}
        }
        
        logger.info("ToolAugmentedExperimentRunner initialized")
    
    def is_toolqa_question(self, question: str) -> bool:
        """Check if a question requires ToolQA tools"""
        return self.tool_system.is_tool_augmented_question(question)
    
    def get_tool_prompt_enhancement(self, question: str, provider: ModelProvider) -> Tuple[str, List[Dict[str, Any]]]:
        """Get enhanced prompt and tools for a question"""
        
        if not self.is_toolqa_question(question):
            return question, []
        
        # Get tools for this question
        available_tools = self.tool_system.get_formatted_tools_for_provider(provider, question)
        
        # Enhance prompt with tool instructions
        if provider == ModelProvider.ANTHROPIC:
            enhanced_prompt = self._get_anthropic_tool_prompt(question)
        elif provider == ModelProvider.OPENAI:
            enhanced_prompt = self._get_openai_tool_prompt(question)
        else:
            enhanced_prompt = question
        
        return enhanced_prompt, available_tools
    
    def _get_anthropic_tool_prompt(self, question: str, expected_answer: str = "") -> str:
        """Get Anthropic-specific tool-enhanced prompt"""
        
        # Determine the expected answer format to guide the model
        format_hint = ""  
        if expected_answer:
            expected_str = str(expected_answer)
            if expected_str.replace('.', '').replace('-', '').isdigit():
                format_hint = "\n\nIMPORTANT: Your final answer should be a precise number. If it's a decimal, include decimal places."
            elif any(unit in expected_str.upper() for unit in ['USD', '$', '%']):
                format_hint = "\n\nIMPORTANT: Include appropriate units in your final answer (USD, $, %, etc.)."
            elif expected_str.replace('.', '').isdigit() and '.' in expected_str:
                format_hint = "\n\nIMPORTANT: Provide your answer as a decimal number (e.g., 2.0 not 2)."
        
        return f"""You are a helpful AI assistant with access to specialized tools for data analysis and computation.

Question: {question}

Instructions:
1. ALWAYS use the available tools when they can help answer the question
2. Analyze the tool results carefully and extract the exact information needed
3. Show your complete reasoning process
4. End with your final answer in this exact format:

FINAL ANSWER: [Your precise answer here]{format_hint}

Remember: Use tools first, then reason about the results to provide your final answer."""
    
    def _get_openai_tool_prompt(self, question: str, expected_answer: str = "") -> str:
        """Get OpenAI-specific tool-enhanced prompt"""
        
        # Determine the expected answer format to guide the model
        format_hint = ""  
        if expected_answer:
            expected_str = str(expected_answer)
            if expected_str.replace('.', '').replace('-', '').isdigit():
                format_hint = "\n\nIMPORTANT: Your final answer should be a precise number. If it's a decimal, include decimal places."
            elif any(unit in expected_str.upper() for unit in ['USD', '$', '%']):
                format_hint = "\n\nIMPORTANT: Include appropriate units in your final answer (USD, $, %, etc.)."
            elif expected_str.replace('.', '').isdigit() and '.' in expected_str:
                format_hint = "\n\nIMPORTANT: Provide your answer as a decimal number (e.g., 2.0 not 2)."
        
        return f"""You are a helpful AI assistant with access to specialized functions for data analysis and computation.

Question: {question}

Instructions:
1. ALWAYS use the available functions when they can help answer the question
2. Analyze the function results carefully and extract the exact information needed
3. Show your complete reasoning process
4. End with your final answer in this exact format:

FINAL ANSWER: [Your precise answer here]{format_hint}

Remember: Use functions first, then reason about the results to provide your final answer."""
    
    def execute_tool_calls_from_response(self, response: Any, provider: ModelProvider) -> Tuple[List[ToolResult], str]:
        """Execute tool calls from model response and format results"""
        
        tool_results = self.tool_system.execute_tool_calls_from_response(provider, response)
        
        # Update metrics
        self.tool_usage_stats["total_tool_calls"] += len(tool_results)
        
        for result in tool_results:
            if result.success:
                self.tool_usage_stats["successful_tool_calls"] += 1
            else:
                self.tool_usage_stats["failed_tool_calls"] += 1
            
            # Track tool usage
            tool_name = result.tool_name
            if tool_name not in self.tool_usage_stats["tools_used"]:
                self.tool_usage_stats["tools_used"][tool_name] = 0
            self.tool_usage_stats["tools_used"][tool_name] += 1
        
        # Format results for model
        formatted_results = self.tool_system.format_tool_results_for_provider(provider, tool_results)
        
        return tool_results, formatted_results
    
    def enhance_multi_turn_conversation(self, question: str, provider: ModelProvider, 
                                      conversation_history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Enhance multi-turn conversation with tool capabilities"""
        
        if not self.is_toolqa_question(question):
            return conversation_history, []
        
        # Get tools and enhanced prompt
        enhanced_prompt, available_tools = self.get_tool_prompt_enhancement(question, provider)
        
        # Update first message with enhanced prompt
        if conversation_history and conversation_history[0].get("role") == "user":
            conversation_history[0]["content"] = enhanced_prompt
        
        # Track that this question uses tools
        self.tool_usage_stats["questions_with_tools"] += 1
        
        return conversation_history, available_tools
    
    def process_model_response_with_tools(self, response: Any, provider: ModelProvider, 
                                        question: str) -> Dict[str, Any]:
        """Process model response and execute any tool calls"""
        
        result = {
            "original_response": response,
            "tool_calls_executed": [],
            "tool_results": [],
            "final_response": None,
            "used_tools": False
        }
        
        if not self.is_toolqa_question(question):
            result["final_response"] = response
            return result
        
        # Execute tool calls if present
        tool_results, formatted_results = self.execute_tool_calls_from_response(response, provider)
        
        if tool_results:
            result["used_tools"] = True
            result["tool_calls_executed"] = [
                {
                    "tool": r.tool_name,
                    "function": r.function,
                    "parameters": r.parameters,
                    "success": r.success,
                    "result": r.result if r.success else r.error_message
                }
                for r in tool_results
            ]
            result["tool_results"] = tool_results
            
            # For now, return the formatted results as the final response
            # In a real implementation, you might want to send this back to the model
            result["final_response"] = formatted_results
        else:
            result["final_response"] = response
        
        return result
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        stats = self.tool_usage_stats.copy()
        stats["success_rate"] = (
            stats["successful_tool_calls"] / max(1, stats["total_tool_calls"]) * 100
        )
        return stats
    
    def log_tool_usage_summary(self):
        """Log summary of tool usage"""
        stats = self.get_usage_statistics()
        
        logger.info("=== Tool Usage Summary ===")
        logger.info(f"Questions with tools: {stats['questions_with_tools']}")
        logger.info(f"Total tool calls: {stats['total_tool_calls']}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"Tools used: {stats['tools_used']}")
        logger.info(f"Domains processed: {stats['domains_processed']}")

class AnthropicToolIntegration:
    """Anthropic-specific tool integration"""
    
    def __init__(self, runner: ToolAugmentedExperimentRunner):
        self.runner = runner
    
    def enhance_anthropic_call(self, messages: List[Dict[str, str]], question: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Enhance Anthropic API call with tools"""
        enhanced_messages, tools = self.runner.enhance_multi_turn_conversation(
            question, ModelProvider.ANTHROPIC, messages
        )
        return enhanced_messages, tools
    
    def process_anthropic_response(self, response: Any, question: str) -> Dict[str, Any]:
        """Process Anthropic response with tool execution"""
        return self.runner.process_model_response_with_tools(
            response, ModelProvider.ANTHROPIC, question
        )

class OpenAIToolIntegration:
    """OpenAI-specific tool integration"""
    
    def __init__(self, runner: ToolAugmentedExperimentRunner):
        self.runner = runner
    
    def enhance_openai_call(self, messages: List[Dict[str, str]], question: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Enhance OpenAI API call with tools"""
        enhanced_messages, tools = self.runner.enhance_multi_turn_conversation(
            question, ModelProvider.OPENAI, messages
        )
        return enhanced_messages, tools
    
    def process_openai_response(self, response: Any, question: str) -> Dict[str, Any]:
        """Process OpenAI response with tool execution"""
        return self.runner.process_model_response_with_tools(
            response, ModelProvider.OPENAI, question
        )

def create_tool_augmented_runners() -> Tuple[AnthropicToolIntegration, OpenAIToolIntegration]:
    """Create tool-augmented runners for both Anthropic and OpenAI"""
    
    base_runner = ToolAugmentedExperimentRunner()
    
    anthropic_integration = AnthropicToolIntegration(base_runner)
    openai_integration = OpenAIToolIntegration(base_runner)
    
    return anthropic_integration, openai_integration

# Enhanced answer extraction with tool context
def extract_answer_with_tool_context(response_text: str, tool_results: List[ToolResult] = None) -> str:
    """Extract answer considering tool execution results"""
    
    if not tool_results:
        # Fallback to standard answer extraction
        return extract_numeric_answer(response_text)
    
    # Look for numerical results in tool outputs first
    for result in tool_results:
        if result.success and result.result is not None:
            if isinstance(result.result, (int, float)):
                return str(result.result)
            elif isinstance(result.result, dict):
                # Look for common answer keys
                for key in ['result', 'answer', 'value', 'solution', 'x']:
                    if key in result.result:
                        return str(result.result[key])
    
    # Fallback to text extraction
    return extract_numeric_answer(response_text)

def extract_numeric_answer(text: str) -> str:
    """Extract numeric answer from text"""
    import re
    
    # Look for numbers in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if numbers:
        # Return the last number found (often the final answer)
        return numbers[-1]
    
    # Fallback
    return text.strip()

# Global instance
_tool_runner = None

def get_tool_augmented_runner() -> ToolAugmentedExperimentRunner:
    """Get global tool-augmented runner instance"""
    global _tool_runner
    if _tool_runner is None:
        _tool_runner = ToolAugmentedExperimentRunner()
    return _tool_runner

if __name__ == "__main__":
    # Test tool integration
    runner = ToolAugmentedExperimentRunner()
    
    test_questions = [
        "What was the coffee price range from 2000-01-03 to 2020-10-07?",
        "A farmer extracts 5 liters of milk a day from a cow. How many liters in a week?",
        "What is 25 * 4 + 17?"
    ]
    
    print("🔧 Tool Integration Test:")
    for question in test_questions:
        is_tool_question = runner.is_toolqa_question(question)
        enhanced_prompt, tools = runner.get_tool_prompt_enhancement(question, ModelProvider.ANTHROPIC)
        
        print(f"\n   Question: {question[:50]}...")
        print(f"   Uses tools: {is_tool_question}")
        print(f"   Available tools: {len(tools)}")
        if tools:
            tool_names = [tool.get('name', 'unknown') for tool in tools]
            print(f"   Tool names: {tool_names[:3]}...")  # First 3 tools
    
    print(f"\n📊 System Stats: {runner.tool_system.get_system_stats()}")