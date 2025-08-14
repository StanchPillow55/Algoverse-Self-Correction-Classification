"""
Large-Scale Multi-Pass Self-Correction Pipeline

This module orchestrates the full pipeline:
1. Stream questions from datasets
2. Generate multi-pass correction traces using classifier + RTS policy
3. Log all interactions with proper schema
4. Periodically update policy with collected rewards
5. Support for real LLM APIs (OpenAI, Anthropic, etc.)
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import aiohttp
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

# Import our components
from train_classifier import ErrorConfidenceTrainer, TrainingConfig
from rts_policy_enhanced import RTSPolicyEnhanced, RTSContext, RTSAction
from trace_annotator import TraceAnnotatorEngine

logger = logging.getLogger(__name__)

@dataclass 
class TurnRecord:
    """Single turn in a multi-pass trace (matches proposal schema)"""
    turn_id: int
    question: str
    answer: str
    prompt_id: Optional[str]
    prompt_text: Optional[str]
    error_mode: str
    confidence_score: float
    delta_accuracy: int
    token_count: int
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class TraceRecord:
    """Complete multi-pass trace record"""
    trace_id: str
    dataset_name: str
    question_id: str
    reference_answer: str
    turns: List[TurnRecord]
    final_accuracy: int
    total_tokens: int
    total_turns: int
    pipeline_version: str
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'turns': [turn.to_dict() for turn in self.turns]
        }

class LLMProvider(ABC):
    """Abstract base for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
        """Generate response and return (text, token_count)"""
        pass

class SentenceTransformerLLM(LLMProvider):
    """Placeholder LLM using simple text responses (for testing)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info("Using mock LLM for testing (sentence-transformers not required)")
        
    async def generate(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
        """Simulate LLM response using sentence similarity"""
        # Simulate response based on prompt type
        responses = {
            "what": "The answer depends on the specific context and requirements.",
            "how": "Here's a step-by-step approach to solve this problem.",
            "why": "This occurs because of several interconnected factors.",
            "capital": "Paris",  # For geography questions
            "math": "42",       # For math questions
        }
        
        # Simple keyword matching for simulation
        prompt_lower = prompt.lower()
        response = "I need to think about this carefully."
        
        for keyword, answer in responses.items():
            if keyword in prompt_lower:
                response = answer
                break
                
        # Simulate token count
        token_count = len(response.split()) + 5  # Add overhead
        
        return response, token_count

class OpenAIProvider(LLMProvider):
    """OpenAI API provider (for production use)"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    async def generate(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
        """Generate using OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    message = result["choices"][0]["message"]["content"]
                    tokens = result["usage"]["total_tokens"]
                    return message.strip(), tokens
                else:
                    raise Exception(f"OpenAI API error: {response.status}")

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider (for production use)"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def generate(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
        """Generate using Anthropic API"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    message = result["content"][0]["text"]
                    # Estimate tokens (Anthropic doesn't always return usage)
                    tokens = len(message.split()) * 1.3  # Rough estimate
                    return message.strip(), int(tokens)
                else:
                    raise Exception(f"Anthropic API error: {response.status}")

class DatasetStreamer:
    """Streams questions from various datasets"""
    
    def __init__(self, dataset_name: str, data_path: Optional[str] = None):
        self.dataset_name = dataset_name
        self.data_path = data_path
        
    def stream_questions(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Stream questions from dataset"""
        
        if self.dataset_name == "gsm8k":
            yield from self._stream_gsm8k(limit)
        elif self.dataset_name == "hotpotqa":
            yield from self._stream_hotpotqa(limit)
        elif self.dataset_name == "boolq":
            yield from self._stream_boolq(limit)
        elif self.dataset_name.endswith(".csv"):
            yield from self._stream_csv(limit)
        else:
            yield from self._stream_mock_data(limit)
    
    def _stream_gsm8k(self, limit: Optional[int]) -> Iterator[Dict]:
        """Stream GSM8K math word problems"""
        # In production, this would load from HuggingFace datasets
        mock_questions = [
            {
                "id": f"gsm8k_{i}",
                "question": f"If Sarah has {3+i} apples and gives away {1+i//2}, how many does she have left?",
                "reference_answer": str(3+i-(1+i//2))
            }
            for i in range(limit or 100)
        ]
        yield from mock_questions
    
    def _stream_hotpotqa(self, limit: Optional[int]) -> Iterator[Dict]:
        """Stream HotpotQA questions"""
        mock_questions = [
            {
                "id": f"hotpot_{i}",
                "question": f"What is the capital of the country where the {i}th largest city is located?",
                "reference_answer": f"Capital_{i}"
            }
            for i in range(limit or 100)
        ]
        yield from mock_questions
    
    def _stream_boolq(self, limit: Optional[int]) -> Iterator[Dict]:
        """Stream BoolQ yes/no questions"""
        mock_questions = [
            {
                "id": f"boolq_{i}",
                "question": f"Is it true that the number {i} is greater than 50?",
                "reference_answer": "yes" if i > 50 else "no"
            }
            for i in range(limit or 100)
        ]
        yield from mock_questions
    
    def _stream_csv(self, limit: Optional[int]) -> Iterator[Dict]:
        """Stream from CSV file"""
        if not self.data_path:
            raise ValueError("CSV path required for CSV dataset")
            
        df = pd.read_csv(self.data_path)
        if limit:
            df = df.head(limit)
            
        for _, row in df.iterrows():
            yield {
                "id": str(row.get("id", hash(str(row)))),
                "question": row["question"],
                "reference_answer": row["reference_answer"]
            }
    
    def _stream_mock_data(self, limit: Optional[int]) -> Iterator[Dict]:
        """Stream mock data for testing"""
        mock_questions = [
            {
                "id": f"mock_{i}",
                "question": f"What is the answer to question number {i}?",
                "reference_answer": f"Answer {i}"
            }
            for i in range(limit or 10)
        ]
        yield from mock_questions

class ScalePipeline:
    """
    Main pipeline orchestrator for large-scale trace generation.
    
    Coordinates:
    - Dataset streaming
    - LLM interaction
    - Error classification
    - RTS policy decisions
    - Trace logging
    - Policy updates
    """
    
    def __init__(self,
                 llm_provider: LLMProvider,
                 classifier_model_path: str,
                 classifier_vectorizer_path: str,
                 rts_algorithm: str = "thompson_sampling",
                 max_turns: int = 5,
                 policy_update_interval: int = 100,
                 log_dir: str = "pipeline_logs"):
        
        self.llm_provider = llm_provider
        self.max_turns = max_turns
        self.policy_update_interval = policy_update_interval
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize classifier
        self.classifier = ErrorConfidenceTrainer(TrainingConfig())
        try:
            self.classifier.load_model(classifier_model_path, classifier_vectorizer_path)
            logger.info("Loaded trained classifier")
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}. Using mock classification.")
            self.classifier = None
        
        # Initialize RTS policy
        self.policy = RTSPolicyEnhanced(algorithm=rts_algorithm)
        
        # Initialize trace annotator
        self.annotator = TraceAnnotatorEngine()
        
        # Statistics tracking
        self.stats = {
            "traces_processed": 0,
            "total_turns": 0,
            "policy_updates": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # RTS templates for prompt text lookup
        self.rts_templates = self._load_rts_templates()
        
        logger.info(f"Pipeline initialized with {rts_algorithm} algorithm")
    
    def _load_rts_templates(self) -> Dict[str, Dict]:
        """Load RTS templates for prompt text lookup"""
        try:
            with open("rts_templates.json", 'r') as f:
                templates_list = json.load(f)
            return {t['id']: t for t in templates_list}
        except FileNotFoundError:
            logger.warning("RTS templates not found, using defaults")
            return {
                "p_try_again": {"text": "Please try again."},
                "p_are_you_sure": {"text": "Are you sure about <ANSWER>?"}
            }
    
    async def process_dataset(self,
                            dataset_name: str,
                            limit: Optional[int] = None,
                            data_path: Optional[str] = None,
                            batch_size: int = 10) -> Dict:
        """
        Main pipeline processing function.
        
        Args:
            dataset_name: Name of dataset to process
            limit: Maximum number of questions to process
            data_path: Path to dataset file (if needed)
            batch_size: Number of traces to process before policy update
            
        Returns:
            Summary statistics
        """
        logger.info(f"Starting pipeline on {dataset_name} (limit: {limit})")
        
        # Initialize dataset streamer
        streamer = DatasetStreamer(dataset_name, data_path)
        
        # Process questions in batches
        batch_traces = []
        
        async for question_data in self._async_stream(streamer.stream_questions(limit)):
            try:
                # Generate multi-pass trace
                trace = await self.generate_trace(
                    question_data["id"],
                    question_data["question"],
                    question_data["reference_answer"],
                    dataset_name
                )
                
                batch_traces.append(trace)
                self.stats["traces_processed"] += 1
                self.stats["total_turns"] += trace.total_turns
                
                # Log trace
                await self._log_trace(trace)
                
                # Update policy periodically
                if len(batch_traces) >= batch_size:
                    await self._update_policy_batch(batch_traces)
                    batch_traces = []
                    self.stats["policy_updates"] += 1
                
                if self.stats["traces_processed"] % 50 == 0:
                    logger.info(f"Processed {self.stats['traces_processed']} traces, "
                              f"{self.stats['total_turns']} total turns")
                              
            except Exception as e:
                logger.error(f"Error processing question {question_data['id']}: {e}")
                continue
        
        # Process remaining batch
        if batch_traces:
            await self._update_policy_batch(batch_traces)
            self.stats["policy_updates"] += 1
        
        # Save final policy state
        self.policy.save_policy(self.log_dir / "final_policy.pkl")
        
        # Generate summary
        summary = self._generate_summary()
        await self._save_summary(summary)
        
        logger.info("Pipeline processing completed")
        return summary
    
    async def generate_trace(self,
                           question_id: str,
                           question: str,
                           reference_answer: str,
                           dataset_name: str) -> TraceRecord:
        """
        Generate a complete multi-pass self-correction trace.
        
        This is the core logic that implements the proposal schema:
        1. Get initial answer from LLM
        2. Classify error + confidence
        3. Query RTS policy for reprompt decision
        4. Continue until policy says stop or max turns reached
        """
        trace_id = f"{dataset_name}_{question_id}_{int(time.time())}"
        turns = []
        
        # Get initial answer
        current_answer, tokens = await self.llm_provider.generate(question)
        
        for turn_idx in range(self.max_turns):
            # Classify error and confidence
            if self.classifier:
                try:
                    prediction = self.classifier.predict(
                        initial_answer=turns[0].answer if turns else current_answer,
                        revised_answer=current_answer,
                        reprompt_id="none"
                    )
                    error_mode = prediction['failure_mode']
                    confidence_score = prediction['confidence_score']
                except Exception as e:
                    logger.warning(f"Classifier error: {e}, using defaults")
                    error_mode = "unknown"
                    confidence_score = 0.5
            else:
                # Mock classification for testing
                error_mode = np.random.choice(["anchored", "overcorrected", "corrected"])
                confidence_score = np.random.uniform(0.3, 0.9)
            
            # Calculate delta accuracy for this turn
            delta_accuracy = self._calculate_delta_accuracy(
                current_answer, reference_answer, turns
            )
            
            # Create turn record
            turn = TurnRecord(
                turn_id=turn_idx,
                question=question,
                answer=current_answer,
                prompt_id=None,  # Will be set if we reprompt
                prompt_text=None,
                error_mode=error_mode,
                confidence_score=confidence_score,
                delta_accuracy=delta_accuracy,
                token_count=tokens,
                timestamp=datetime.now().isoformat()
            )
            turns.append(turn)
            
            # Query RTS policy for reprompt decision
            context = RTSContext(
                detected_error=error_mode,
                confidence=confidence_score,
                last_prompt_id=turns[-2].prompt_id if len(turns) > 1 else None,
                turn_index=turn_idx
            )
            
            action = self.policy.select_prompt(context)
            
            # If policy says don't reprompt, break
            if not action.reprompt:
                break
                
            # Update policy with reward from this turn
            self.policy.update_policy(context, action, delta_accuracy, tokens)
            
            # Prepare reprompt
            prompt_template = self.rts_templates.get(action.prompt_id, {})
            prompt_text = prompt_template.get('text', 'Please reconsider your answer.')
            
            # Replace <ANSWER> placeholder if present
            if '<ANSWER>' in prompt_text:
                prompt_text = prompt_text.replace('<ANSWER>', current_answer)
            
            # Create reprompt
            reprompt = f"{question}\n\nYour previous answer: {current_answer}\n\n{prompt_text}"
            
            # Update turn record with reprompt info
            turn.prompt_id = action.prompt_id
            turn.prompt_text = prompt_text
            
            # Get new answer
            try:
                current_answer, tokens = await self.llm_provider.generate(reprompt)
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                break
        
        # Calculate final accuracy
        final_accuracy = 1 if self._is_correct(current_answer, reference_answer) else 0
        total_tokens = sum(turn.token_count for turn in turns)
        
        return TraceRecord(
            trace_id=trace_id,
            dataset_name=dataset_name,
            question_id=question_id,
            reference_answer=reference_answer,
            turns=turns,
            final_accuracy=final_accuracy,
            total_tokens=total_tokens,
            total_turns=len(turns),
            pipeline_version="1.0",
            created_at=datetime.now().isoformat()
        )
    
    async def _async_stream(self, iterator):
        """Convert sync iterator to async"""
        for item in iterator:
            yield item
            await asyncio.sleep(0)  # Allow other tasks to run
    
    def _calculate_delta_accuracy(self, 
                                current_answer: str, 
                                reference_answer: str, 
                                previous_turns: List[TurnRecord]) -> int:
        """Calculate accuracy change for this turn"""
        current_correct = self._is_correct(current_answer, reference_answer)
        
        if not previous_turns:
            return 1 if current_correct else 0
            
        previous_correct = self._is_correct(previous_turns[-1].answer, reference_answer)
        
        if current_correct and not previous_correct:
            return 1   # Improvement
        elif not current_correct and previous_correct:
            return -1  # Degradation
        else:
            return 0   # No change
    
    def _is_correct(self, answer: str, reference: str) -> bool:
        """Simple correctness evaluation"""
        # This is a simplified version - would need task-specific evaluation
        answer_clean = answer.strip().lower()
        reference_clean = reference.strip().lower()
        
        # Check exact match
        if answer_clean == reference_clean:
            return True
            
        # Check if reference is contained in answer
        if reference_clean in answer_clean:
            return True
            
        # For numeric answers, try to extract and compare numbers
        import re
        answer_nums = re.findall(r'-?\d+\.?\d*', answer)
        ref_nums = re.findall(r'-?\d+\.?\d*', reference)
        
        if answer_nums and ref_nums:
            try:
                return float(answer_nums[-1]) == float(ref_nums[-1])
            except ValueError:
                pass
                
        return False
    
    async def _log_trace(self, trace: TraceRecord):
        """Log trace to file"""
        log_file = self.log_dir / f"traces_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(trace.to_dict()) + '\n')
    
    async def _update_policy_batch(self, traces: List[TraceRecord]):
        """Update policy with batch of traces"""
        logger.info(f"Updating policy with {len(traces)} traces")
        
        # Policy is already updated during trace generation,
        # but we could do additional batch processing here
        
        # For example, recalibrate based on overall performance
        total_reward = 0
        total_turns = 0
        
        for trace in traces:
            for turn in trace.turns:
                if turn.prompt_id:  # If a reprompt was used
                    reward = turn.delta_accuracy - (0.001 * turn.token_count)
                    total_reward += reward
                    total_turns += 1
        
        if total_turns > 0:
            avg_reward = total_reward / total_turns
            logger.info(f"Batch average reward: {avg_reward:.3f}")
    
    def _generate_summary(self) -> Dict:
        """Generate pipeline run summary"""
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.stats["start_time"])
        duration = (end_time - start_time).total_seconds()
        
        return {
            "pipeline_stats": self.stats,
            "policy_stats": self.policy.get_action_statistics(),
            "duration_seconds": duration,
            "traces_per_second": self.stats["traces_processed"] / duration if duration > 0 else 0,
            "avg_turns_per_trace": self.stats["total_turns"] / max(1, self.stats["traces_processed"]),
            "completed_at": end_time.isoformat()
        }
    
    async def _save_summary(self, summary: Dict):
        """Save run summary"""
        summary_file = self.log_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

# Configuration and runner
async def run_pipeline_example():
    """Example pipeline run"""
    
    # Initialize LLM provider (placeholder for testing)
    # In production, replace with:
    # llm_provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
    # llm_provider = AnthropicProvider(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
    llm_provider = SentenceTransformerLLM()
    
    # Initialize pipeline
    pipeline = ScalePipeline(
        llm_provider=llm_provider,
        classifier_model_path="models/error_confidence_model.pt",
        classifier_vectorizer_path="models/vectorizer.pkl",
        rts_algorithm="thompson_sampling",
        max_turns=3,  # Small for demo
        policy_update_interval=5
    )
    
    # Process dataset
    summary = await pipeline.process_dataset(
        dataset_name="mock",
        limit=20,  # Small for demo
        batch_size=5
    )
    
    print("Pipeline Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_pipeline_example())
