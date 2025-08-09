#!/usr/bin/env python3
"""
Core Algorithms Implementation - Phase 1 Month 2
Task Decomposition and Routing Algorithms

Implements:
1. Task Decomposition Algorithm (Simplified Equation 14)
2. Simple Classifier-based Router 
3. Adaptive Routing with contextual feedback
"""

import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from pathlib import Path
import time
import logging

# Import our existing components
from minimum_viable_benchmark import BenchmarkTask, BenchmarkResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"  
    COMPLEX = "complex"

class ModelType(Enum):
    """Model types for routing."""
    SLM = "slm"
    LLM = "llm"
    HYBRID = "hybrid"

@dataclass
class DecomposedTask:
    """A decomposed subtask."""
    subtask_id: str
    parent_task_id: str
    content: str
    complexity: TaskComplexity
    dependencies: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    priority: int = 1

@dataclass
class RoutingDecision:
    """Routing decision for a task."""
    task_id: str
    chosen_model: ModelType
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_latency: float

@dataclass
class TaskFeatures:
    """Extracted features for task routing."""
    length: int
    domain: str  
    complexity_score: float
    has_code: bool
    has_math: bool
    has_reasoning: bool
    keyword_density: Dict[str, float] = field(default_factory=dict)

class TaskDecomposer:
    """
    Implements simplified Task Decomposition Algorithm based on Equation 14:
    
    D*(t) = argmin_D [ Σ C_model(t_i) + λ|D| ]
    
    Where:
    - D is the decomposition
    - C_model(t_i) is the cost of processing subtask t_i
    - λ is the decomposition penalty
    """
    
    def __init__(self, decomposition_penalty: float = 0.1):
        self.decomposition_penalty = decomposition_penalty  # λ in equation
        self.slm_cost_per_token = 0.00001  # $0.01 per 1K tokens
        self.llm_cost_per_token = 0.00015  # $0.15 per 1K tokens
        
    def should_decompose(self, task: BenchmarkTask) -> bool:
        """Determine if a task should be decomposed."""
        
        # Rule-based decomposition criteria from implementation plan
        content_length = len(task.input_text.split())
        
        # Decompose if:
        # 1. Task is too long (>500 tokens as per plan)
        if content_length > 500:
            return True
            
        # 2. Complexity score is high
        complexity_score = self._calculate_complexity_score(task)
        if complexity_score > 0.7:
            return True
            
        # 3. Multiple subtasks detected
        if self._has_multiple_subtasks(task):
            return True
            
        return False
    
    def decompose_task(self, task: BenchmarkTask) -> List[DecomposedTask]:
        """
        Decompose a task into optimal subtasks.
        
        Implementation of simplified Equation 14:
        Find decomposition D* that minimizes total cost + penalty
        """
        
        if not self.should_decompose(task):
            # Return original task as single subtask
            return [DecomposedTask(
                subtask_id=f"{task.task_id}_0",
                parent_task_id=task.task_id,
                content=task.input_text,
                complexity=TaskComplexity(task.complexity)
            )]
        
        # Identify natural breakpoints using dependency parsing
        breakpoints = self._find_breakpoints(task.input_text)
        
        # Generate possible decompositions
        possible_decompositions = self._generate_decompositions(task, breakpoints)
        
        # Find optimal decomposition using Equation 14
        best_decomposition = self._optimize_decomposition(possible_decompositions)
        
        logger.info(f"Decomposed task {task.task_id} into {len(best_decomposition)} subtasks")
        
        return best_decomposition
    
    def _calculate_complexity_score(self, task: BenchmarkTask) -> float:
        """Calculate complexity score for a task."""
        content = task.input_text.lower()
        score = 0.0
        
        # Domain-specific complexity indicators
        complex_patterns = {
            "code_generation": ["algorithm", "recursive", "dynamic programming", "optimization"],
            "document_qa": ["analyze", "compare", "why", "how", "explain"],
            "structured_data": ["calculate", "aggregate", "join", "transform"]
        }
        
        domain_patterns = complex_patterns.get(task.domain, [])
        for pattern in domain_patterns:
            if pattern in content:
                score += 0.2
        
        # General complexity indicators
        if any(word in content for word in ["multiple", "complex", "several", "various"]):
            score += 0.3
        
        # Length-based complexity
        word_count = len(task.input_text.split())
        if word_count > 100:
            score += 0.2
        if word_count > 300:
            score += 0.3
            
        return min(score, 1.0)  # Cap at 1.0
    
    def _has_multiple_subtasks(self, task: BenchmarkTask) -> bool:
        """Detect if task has multiple subtasks."""
        content = task.input_text
        
        # Look for enumeration patterns
        enum_patterns = [
            r'\d+\.',  # 1. 2. 3.
            r'\([a-z]\)',  # (a) (b) (c)
            r'first.*second.*third',  # sequence words
            r'step \d+',  # step 1, step 2
        ]
        
        for pattern in enum_patterns:
            if len(re.findall(pattern, content, re.IGNORECASE)) >= 2:
                return True
                
        # Look for coordination patterns
        coord_patterns = ["and then", "after that", "next", "also", "in addition"]
        coord_count = sum(1 for pattern in coord_patterns if pattern in content.lower())
        
        return coord_count >= 2
    
    def _find_breakpoints(self, text: str) -> List[int]:
        """Find natural breakpoints in text for decomposition."""
        breakpoints = [0]  # Always start at 0
        
        sentences = text.split('.')
        current_pos = 0
        
        for sentence in sentences:
            current_pos += len(sentence) + 1  # +1 for the period
            # Add breakpoint if sentence is substantial
            if len(sentence.strip()) > 20:
                breakpoints.append(current_pos)
        
        breakpoints.append(len(text))  # Always end at text length
        return sorted(list(set(breakpoints)))
    
    def _generate_decompositions(self, task: BenchmarkTask, breakpoints: List[int]) -> List[List[DecomposedTask]]:
        """Generate possible decompositions based on breakpoints."""
        decompositions = []
        text = task.input_text
        
        # Single task (no decomposition)
        decompositions.append([DecomposedTask(
            subtask_id=f"{task.task_id}_0",
            parent_task_id=task.task_id,
            content=text,
            complexity=TaskComplexity(task.complexity)
        )])
        
        # Binary split at midpoint
        if len(breakpoints) > 2:
            mid_idx = len(breakpoints) // 2
            mid_pos = breakpoints[mid_idx]
            
            subtask1 = DecomposedTask(
                subtask_id=f"{task.task_id}_0",
                parent_task_id=task.task_id,
                content=text[:mid_pos],
                complexity=TaskComplexity.SIMPLE
            )
            
            subtask2 = DecomposedTask(
                subtask_id=f"{task.task_id}_1", 
                parent_task_id=task.task_id,
                content=text[mid_pos:],
                complexity=TaskComplexity.SIMPLE,
                dependencies=[f"{task.task_id}_0"]
            )
            
            decompositions.append([subtask1, subtask2])
        
        # Tri-split for complex tasks
        if len(breakpoints) > 4:
            third = len(breakpoints) // 3
            two_thirds = 2 * third
            
            subtasks = []
            for i, (start_idx, end_idx) in enumerate([(0, third), (third, two_thirds), (two_thirds, len(breakpoints))]):
                start_pos = breakpoints[start_idx]
                end_pos = breakpoints[end_idx - 1] if end_idx < len(breakpoints) else len(text)
                
                deps = [f"{task.task_id}_{j}" for j in range(i)] if i > 0 else []
                
                subtask = DecomposedTask(
                    subtask_id=f"{task.task_id}_{i}",
                    parent_task_id=task.task_id,
                    content=text[start_pos:end_pos],
                    complexity=TaskComplexity.SIMPLE,
                    dependencies=deps
                )
                subtasks.append(subtask)
            
            decompositions.append(subtasks)
        
        return decompositions
    
    def _optimize_decomposition(self, decompositions: List[List[DecomposedTask]]) -> List[DecomposedTask]:
        """
        Find optimal decomposition using Equation 14:
        D* = argmin_D [ Σ C_model(t_i) + λ|D| ]
        """
        best_decomposition = decompositions[0]
        best_cost = float('inf')
        
        for decomposition in decompositions:
            # Calculate total processing cost
            processing_cost = sum(self._estimate_subtask_cost(subtask) for subtask in decomposition)
            
            # Add decomposition penalty (λ|D|)
            penalty_cost = self.decomposition_penalty * len(decomposition)
            
            total_cost = processing_cost + penalty_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_decomposition = decomposition
                
        # Update cost estimates
        for subtask in best_decomposition:
            subtask.estimated_cost = self._estimate_subtask_cost(subtask)
            
        return best_decomposition
    
    def _estimate_subtask_cost(self, subtask: DecomposedTask) -> float:
        """Estimate cost of processing a subtask."""
        token_count = len(subtask.content.split())
        
        # Use SLM cost for simple tasks, LLM cost for complex
        if subtask.complexity == TaskComplexity.SIMPLE:
            return token_count * self.slm_cost_per_token
        else:
            return token_count * self.llm_cost_per_token

class TaskRouter:
    """
    Simple Classifier-based Router as specified in implementation plan.
    
    Routes tasks between SLMs and LLMs based on:
    - Task complexity score
    - Domain-specific patterns  
    - Length and feature analysis
    """
    
    def __init__(self):
        self.complexity_threshold_slm = 0.3  # From config: below this -> SLM
        self.complexity_threshold_llm = 0.7  # From config: above this -> LLM
        self.default_model = ModelType.SLM
        self.fallback_model = ModelType.LLM
        
        # Track routing decisions for analysis
        self.routing_history = []
        
    def extract_features(self, task: BenchmarkTask) -> TaskFeatures:
        """Extract features from task for routing decision."""
        content = task.input_text.lower()
        
        # Basic features
        length = len(task.input_text.split())
        
        # Pattern detection
        has_code = any(pattern in content for pattern in ['def ', 'function', 'return', '()', 'import'])
        has_math = any(pattern in content for pattern in ['calculate', 'equation', '+', '-', '*', '/', '='])
        has_reasoning = any(pattern in content for pattern in ['why', 'how', 'explain', 'analyze', 'because'])
        
        # Complexity score
        complexity_score = self._calculate_routing_complexity(task)
        
        # Domain-specific keywords
        keyword_density = self._calculate_keyword_density(content, task.domain)
        
        return TaskFeatures(
            length=length,
            domain=task.domain,
            complexity_score=complexity_score,
            has_code=has_code,
            has_math=has_math,
            has_reasoning=has_reasoning,
            keyword_density=keyword_density
        )
    
    def route_task(self, task: BenchmarkTask) -> RoutingDecision:
        """
        Route task to appropriate model based on classifier logic.
        
        As per implementation plan:
        - If complexity < 0.3 → SLM
        - If complexity > 0.7 → LLM  
        - Otherwise → Hybrid approach (start with SLM, fallback to LLM)
        """
        
        features = self.extract_features(task)
        
        # Main routing logic
        if features.complexity_score < self.complexity_threshold_slm:
            chosen_model = ModelType.SLM
            confidence = 0.9
            reasoning = f"Low complexity ({features.complexity_score:.2f}) -> SLM"
            
        elif features.complexity_score > self.complexity_threshold_llm:
            chosen_model = ModelType.LLM
            confidence = 0.8
            reasoning = f"High complexity ({features.complexity_score:.2f}) -> LLM"
            
        else:
            # Hybrid decision - use additional features
            chosen_model = self._hybrid_decision(features)
            confidence = 0.6
            reasoning = f"Moderate complexity ({features.complexity_score:.2f}) -> {chosen_model.value}"
        
        # Estimate costs
        est_cost, est_latency = self._estimate_execution_cost(features, chosen_model)
        
        decision = RoutingDecision(
            task_id=task.task_id,
            chosen_model=chosen_model,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=est_cost,
            estimated_latency=est_latency
        )
        
        # Record decision for analysis
        self.routing_history.append(decision)
        
        return decision
    
    def _calculate_routing_complexity(self, task: BenchmarkTask) -> float:
        """Calculate complexity score for routing decisions."""
        content = task.input_text.lower()
        score = 0.0
        
        # Length-based complexity
        word_count = len(task.input_text.split())
        if word_count > 50:
            score += 0.1
        if word_count > 100:
            score += 0.2
        if word_count > 200:
            score += 0.3
        
        # Domain-specific complexity patterns
        domain_complexity = {
            "code_generation": {
                "simple": ["print", "return", "if", "else"],
                "moderate": ["for", "while", "function", "class"],
                "complex": ["algorithm", "recursive", "optimization", "dynamic programming"]
            },
            "document_qa": {
                "simple": ["what", "when", "where", "who"],
                "moderate": ["how", "why", "list", "describe"],
                "complex": ["analyze", "compare", "evaluate", "synthesize"]
            },
            "structured_data": {
                "simple": ["find", "get", "select"],
                "moderate": ["calculate", "count", "sum"],
                "complex": ["join", "aggregate", "transform", "pivot"]
            }
        }
        
        if task.domain in domain_complexity:
            patterns = domain_complexity[task.domain]
            
            for level, keywords in patterns.items():
                matches = sum(1 for keyword in keywords if keyword in content)
                if level == "simple":
                    score += matches * 0.05
                elif level == "moderate":  
                    score += matches * 0.15
                else:  # complex
                    score += matches * 0.25
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_keyword_density(self, content: str, domain: str) -> Dict[str, float]:
        """Calculate keyword density for domain-specific routing."""
        total_words = len(content.split())
        if total_words == 0:
            return {}
        
        domain_keywords = {
            "code_generation": ["function", "return", "variable", "loop", "condition"],
            "document_qa": ["question", "answer", "context", "information", "fact"],
            "structured_data": ["table", "column", "row", "data", "field"]
        }
        
        keywords = domain_keywords.get(domain, [])
        densities = {}
        
        for keyword in keywords:
            count = content.count(keyword)
            densities[keyword] = count / total_words
            
        return densities
    
    def _hybrid_decision(self, features: TaskFeatures) -> ModelType:
        """Make hybrid routing decision based on multiple features."""
        
        # Start with SLM preference for efficiency
        slm_score = 0.6
        
        # Adjust based on features
        if features.length < 50:
            slm_score += 0.2
        elif features.length > 150:
            slm_score -= 0.3
            
        if features.has_code and features.domain == "code_generation":
            slm_score += 0.1  # SLMs can handle simple code
            
        if features.has_reasoning:
            slm_score -= 0.2  # LLMs better at reasoning
            
        if features.has_math:
            slm_score -= 0.1  # Slight preference for LLM
        
        # Domain-specific adjustments
        if features.domain == "structured_data":
            slm_score += 0.1  # SLMs good at structured tasks
        elif features.domain == "document_qa" and features.has_reasoning:
            slm_score -= 0.2  # LLMs better at complex QA
            
        return ModelType.SLM if slm_score > 0.5 else ModelType.LLM
    
    def _estimate_execution_cost(self, features: TaskFeatures, model_type: ModelType) -> Tuple[float, float]:
        """Estimate execution cost and latency."""
        token_count = features.length
        
        if model_type == ModelType.SLM:
            cost = token_count * 0.00001  # $0.01 per 1K tokens
            latency = 50 + (token_count * 0.1)  # Base 50ms + processing time
        else:  # LLM
            cost = token_count * 0.00015  # $0.15 per 1K tokens  
            latency = 200 + (token_count * 0.5)  # Base 200ms + processing time
            
        return cost, latency
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics for analysis."""
        if not self.routing_history:
            return {"total_decisions": 0}
        
        total = len(self.routing_history)
        slm_count = sum(1 for d in self.routing_history if d.chosen_model == ModelType.SLM)
        llm_count = sum(1 for d in self.routing_history if d.chosen_model == ModelType.LLM)
        
        avg_confidence = sum(d.confidence for d in self.routing_history) / total
        avg_complexity = sum(float(d.reasoning.split('(')[1].split(')')[0]) 
                           for d in self.routing_history 
                           if '(' in d.reasoning and ')' in d.reasoning) / total
        
        return {
            "total_decisions": total,
            "slm_usage_rate": slm_count / total,
            "llm_usage_rate": llm_count / total,
            "average_confidence": avg_confidence,
            "average_complexity": avg_complexity,
            "target_slm_coverage": 0.7,  # From implementation plan
            "achieved_target": slm_count / total >= 0.7
        }

class CoreAlgorithmManager:
    """Manages task decomposition and routing algorithms."""
    
    def __init__(self):
        self.decomposer = TaskDecomposer()
        self.router = TaskRouter()
        
    async def process_task_with_algorithms(self, task: BenchmarkTask) -> Dict[str, Any]:
        """Process a task through decomposition and routing algorithms."""
        
        start_time = time.time()
        
        # Step 1: Task Decomposition
        decomposition_start = time.time()
        subtasks = self.decomposer.decompose_task(task)
        decomposition_time = time.time() - decomposition_start
        
        # Step 2: Route each subtask
        routing_decisions = []
        for subtask in subtasks:
            # Create a BenchmarkTask for routing
            routing_task = BenchmarkTask(
                domain=task.domain,
                task_id=subtask.subtask_id,
                input_text=subtask.content,
                expected_output="",  # Not needed for routing
                complexity=subtask.complexity.value
            )
            
            decision = self.router.route_task(routing_task)
            routing_decisions.append(decision)
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        total_cost = sum(d.estimated_cost for d in routing_decisions)
        avg_latency = sum(d.estimated_latency for d in routing_decisions) / len(routing_decisions)
        slm_subtasks = sum(1 for d in routing_decisions if d.chosen_model == ModelType.SLM)
        
        return {
            "original_task": task.task_id,
            "decomposition": {
                "subtask_count": len(subtasks),
                "subtasks": [{"id": st.subtask_id, "complexity": st.complexity.value, 
                            "cost": st.estimated_cost} for st in subtasks],
                "decomposition_time_ms": decomposition_time * 1000
            },
            "routing": {
                "decisions": [{"subtask_id": d.task_id, "model": d.chosen_model.value,
                              "confidence": d.confidence, "reasoning": d.reasoning}
                             for d in routing_decisions],
                "slm_usage_rate": slm_subtasks / len(routing_decisions),
                "total_estimated_cost": total_cost,
                "avg_estimated_latency": avg_latency
            },
            "performance": {
                "total_processing_time_ms": total_time * 1000,
                "meets_slm_target": slm_subtasks / len(routing_decisions) >= 0.7
            }
        }

# Example usage and testing
async def test_core_algorithms():
    """Test the core algorithms implementation."""
    
    print("🧪 TESTING CORE ALGORITHMS - PHASE 1 MONTH 2")
    print("="*60)
    
    manager = CoreAlgorithmManager()
    
    # Test tasks of varying complexity
    test_tasks = [
        BenchmarkTask(
            domain="code_generation",
            task_id="test_simple",
            input_text="Write a function that adds two numbers",
            expected_output="def add(a, b): return a + b",
            complexity="simple"
        ),
        BenchmarkTask(
            domain="document_qa", 
            task_id="test_moderate",
            input_text="Given this context about machine learning, explain the difference between supervised and unsupervised learning, provide examples of each, and discuss when you would use each approach.",
            expected_output="Supervised learning uses labeled data...",
            complexity="moderate"
        ),
        BenchmarkTask(
            domain="structured_data",
            task_id="test_complex", 
            input_text="Analyze the following sales data table, calculate quarterly growth rates, identify seasonal patterns, create summary statistics for each product category, and recommend optimization strategies based on the trends observed.",
            expected_output="Analysis shows...",
            complexity="complex"
        )
    ]
    
    print(f"\\n📊 Testing {len(test_tasks)} tasks...")
    
    results = []
    for task in test_tasks:
        print(f"\\n🔍 Processing {task.task_id} ({task.complexity})...")
        result = await manager.process_task_with_algorithms(task)
        results.append(result)
        
        # Print summary
        dec = result["decomposition"]
        rout = result["routing"]
        perf = result["performance"]
        
        print(f"  ✓ Decomposed into {dec['subtask_count']} subtasks")
        print(f"  ✓ SLM usage: {rout['slm_usage_rate']:.1%}")
        print(f"  ✓ Est. cost: ${rout['total_estimated_cost']:.6f}")
        print(f"  ✓ Target met: {perf['meets_slm_target']}")
    
    # Overall statistics
    print(f"\\n📈 ALGORITHM PERFORMANCE SUMMARY:")
    print("="*40)
    
    router_stats = manager.router.get_routing_stats()
    
    print(f"Total routing decisions: {router_stats['total_decisions']}")
    print(f"SLM usage rate: {router_stats['slm_usage_rate']:.1%}")
    print(f"Average confidence: {router_stats['average_confidence']:.1%}")
    print(f"Target achieved: {router_stats['achieved_target']}")
    
    total_subtasks = sum(r["decomposition"]["subtask_count"] for r in results)
    avg_decomp_time = sum(r["decomposition"]["decomposition_time_ms"] for r in results) / len(results)
    
    print(f"Total subtasks generated: {total_subtasks}")
    print(f"Average decomposition time: {avg_decomp_time:.1f}ms")
    
    print("\\n✅ Core algorithms implementation complete!")
    print("🎯 Ready for Phase 1 Month 3: Initial Results")

if __name__ == "__main__":
    asyncio.run(test_core_algorithms())