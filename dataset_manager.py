#!/usr/bin/env python3
"""
Community Benchmark Dataset Manager
Fetches and manages standard datasets used by researchers for comparison
"""

import os
import json
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datasets import load_dataset
import asyncio
from dataclasses import dataclass

# Add HuggingFace datasets library to requirements if not present
# pip install datasets

@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""
    name: str
    domain: str
    description: str
    paper_reference: str
    huggingface_name: str = None
    github_url: str = None
    download_url: str = None
    local_path: str = None
    size_mb: int = None
    task_count: int = None

class CommunityDatasetManager:
    """Manages downloading and processing of community benchmark datasets."""
    
    def __init__(self, data_dir: str = "benchmarks/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define community benchmark datasets for our domains
        self.datasets = {
            # Code Generation Domain
            "humaneval": DatasetInfo(
                name="HumanEval",
                domain="code_generation", 
                description="Hand-written programming problems for code generation evaluation",
                paper_reference="Chen et al. (2021) - Evaluating Large Language Models Trained on Code",
                huggingface_name="openai_humaneval",
                task_count=164,
                size_mb=1,
                local_path="code_generation/humaneval"
            ),
            
            "mbpp": DatasetInfo(
                name="MBPP", 
                domain="code_generation",
                description="Mostly Basic Python Problems - crowd-sourced Python programming problems",
                paper_reference="Austin et al. (2021) - Program Synthesis with Large Language Models", 
                huggingface_name="mbpp",
                task_count=974,
                size_mb=2,
                local_path="code_generation/mbpp"
            ),
            
            "apps": DatasetInfo(
                name="APPS",
                domain="code_generation",
                description="Automated Programming Assessment System - competitive programming problems",
                paper_reference="Hendrycks et al. (2021) - Measuring Coding Challenge Competence",
                huggingface_name="codeparrot/apps", 
                task_count=10000,
                size_mb=50,
                local_path="code_generation/apps"
            ),
            
            # Document QA Domain
            "squad_v1": DatasetInfo(
                name="SQuAD 1.1",
                domain="document_qa",
                description="Stanford Question Answering Dataset - reading comprehension",
                paper_reference="Rajpurkar et al. (2016) - SQuAD: 100,000+ Questions for Machine Reading Comprehension",
                huggingface_name="squad",
                task_count=107785,
                size_mb=35,
                local_path="document_qa/squad_v1"
            ),
            
            "squad_v2": DatasetInfo(
                name="SQuAD 2.0", 
                domain="document_qa",
                description="SQuAD 2.0 combines answerable questions with unanswerable ones",
                paper_reference="Rajpurkar et al. (2018) - Know What You Don't Know: Unanswerable Questions for SQuAD",
                huggingface_name="squad_v2",
                task_count=150000,
                size_mb=45,
                local_path="document_qa/squad_v2"
            ),
            
            "natural_questions": DatasetInfo(
                name="Natural Questions",
                domain="document_qa", 
                description="Real questions from Google Search with Wikipedia answers",
                paper_reference="Kwiatkowski et al. (2019) - Natural Questions: a Benchmark for Question Answering Research",
                huggingface_name="natural_questions",
                task_count=323000,
                size_mb=200,
                local_path="document_qa/natural_questions"
            ),
            
            "ms_marco": DatasetInfo(
                name="MS MARCO QA",
                domain="document_qa",
                description="Microsoft Machine Reading Comprehension Dataset",
                paper_reference="Nguyen et al. (2016) - MS MARCO: A Human Generated MAchine Reading COmprehension Dataset", 
                huggingface_name="ms_marco",
                task_count=100000,
                size_mb=120,
                local_path="document_qa/ms_marco"
            ),
            
            # Structured Data Processing Domain  
            "wikitableqs": DatasetInfo(
                name="WikiTableQuestions",
                domain="structured_data",
                description="Question answering over semi-structured Wikipedia tables",
                paper_reference="Pasupat & Liang (2015) - Compositional Semantic Parsing on Semi-Structured Tables",
                huggingface_name="wikitablequestions", 
                task_count=22033,
                size_mb=75,
                local_path="structured_data/wikitableqs"
            ),
            
            "tabfact": DatasetInfo(
                name="TabFact",
                domain="structured_data", 
                description="Table-based fact verification dataset",
                paper_reference="Chen et al. (2019) - TabFact: A Large-scale Dataset for Table-based Fact Verification",
                huggingface_name="tab_fact",
                task_count=183000,
                size_mb=95,
                local_path="structured_data/tabfact"
            ),
            
            "spider": DatasetInfo(
                name="Spider",
                domain="structured_data",
                description="Cross-domain text-to-SQL semantic parsing dataset", 
                paper_reference="Yu et al. (2018) - Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL",
                huggingface_name="spider",
                task_count=10181,
                size_mb=30,
                local_path="structured_data/spider"
            ),
            
            # Additional Multi-domain Benchmarks
            "mmlu": DatasetInfo(
                name="MMLU", 
                domain="multi_domain",
                description="Massive Multitask Language Understanding - knowledge across 57 subjects",
                paper_reference="Hendrycks et al. (2021) - Measuring Massive Multitask Language Understanding",
                huggingface_name="cais/mmlu",
                task_count=15908,
                size_mb=75,
                local_path="multi_domain/mmlu"
            ),
            
            "big_bench": DatasetInfo(
                name="BIG-bench",
                domain="multi_domain", 
                description="Beyond the Imitation Game collaborative benchmark",
                paper_reference="Srivastava et al. (2022) - Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models",
                huggingface_name="bigbench",
                task_count=200000,
                size_mb=500,
                local_path="multi_domain/big_bench"
            )
        }
    
    def list_available_datasets(self) -> None:
        """Print available community datasets."""
        print("📊 COMMUNITY BENCHMARK DATASETS")
        print("="*70)
        
        by_domain = {}
        for dataset_id, info in self.datasets.items():
            domain = info.domain
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append((dataset_id, info))
        
        for domain, datasets in by_domain.items():
            print(f"\\n🔍 {domain.replace('_', ' ').title()}:")
            print("-" * 50)
            
            for dataset_id, info in datasets:
                status = "✅ Downloaded" if self.is_downloaded(dataset_id) else "⬇️  Available"
                size_info = f"({info.size_mb}MB, {info.task_count:,} tasks)" if info.size_mb else ""
                print(f"  {status} {info.name} {size_info}")
                print(f"      {info.description}")
                print(f"      Paper: {info.paper_reference}")
                print()
    
    def is_downloaded(self, dataset_id: str) -> bool:
        """Check if dataset is already downloaded."""
        info = self.datasets.get(dataset_id)
        if not info:
            return False
        
        dataset_path = self.data_dir / info.local_path
        return dataset_path.exists() and any(dataset_path.iterdir())
    
    async def download_dataset(self, dataset_id: str, force: bool = False) -> bool:
        """Download a specific dataset."""
        info = self.datasets.get(dataset_id)
        if not info:
            print(f"❌ Unknown dataset: {dataset_id}")
            return False
        
        dataset_path = self.data_dir / info.local_path
        
        if self.is_downloaded(dataset_id) and not force:
            print(f"✅ {info.name} already downloaded to {dataset_path}")
            return True
        
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        print(f"⬇️  Downloading {info.name} ({info.size_mb}MB, {info.task_count:,} tasks)...")
        print(f"    Path: {dataset_path}")
        
        try:
            if info.huggingface_name:
                # Download from HuggingFace
                dataset = load_dataset(info.huggingface_name)
                
                # Save as JSON files
                for split_name, split_data in dataset.items():
                    output_file = dataset_path / f"{split_name}.json"
                    
                    # Convert to pandas for easier handling
                    df = split_data.to_pandas()
                    
                    # Save first 1000 examples for initial testing (full dataset for production)
                    sample_size = min(1000, len(df)) if dataset_id != "humaneval" else len(df)
                    sample_df = df.head(sample_size)
                    
                    sample_df.to_json(output_file, orient='records', indent=2)
                    print(f"    ✓ Saved {len(sample_df)} examples to {split_name}.json")
                
                # Save metadata
                metadata = {
                    "name": info.name,
                    "domain": info.domain,
                    "description": info.description,
                    "paper_reference": info.paper_reference,
                    "huggingface_name": info.huggingface_name,
                    "download_date": pd.Timestamp.now().isoformat(),
                    "total_examples": info.task_count,
                    "sample_size": sample_size,
                    "splits": list(dataset.keys())
                }
                
                with open(dataset_path / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"    ✅ {info.name} downloaded successfully!")
                return True
                
            else:
                print(f"    ⚠️  Manual download required for {info.name}")
                print(f"        No automated download available")
                return False
                
        except Exception as e:
            print(f"    ❌ Failed to download {info.name}: {str(e)}")
            return False
    
    async def download_priority_datasets(self) -> Dict[str, bool]:
        """Download priority datasets for our 3 core domains."""
        
        # Priority datasets for Phase 1 evaluation
        priority_datasets = [
            "humaneval",    # Code generation - small, high-quality
            "mbpp",         # Code generation - more examples  
            "squad_v1",     # Document QA - standard benchmark
            "wikitableqs",  # Structured data - table QA
            "mmlu"          # Multi-domain - general capabilities
        ]
        
        print("🎯 Downloading Priority Datasets for Phase 1")
        print("="*50)
        
        results = {}
        
        for dataset_id in priority_datasets:
            success = await self.download_dataset(dataset_id)
            results[dataset_id] = success
            
            if success:
                print()
            else:
                print(f"    ⚠️  Will need manual download for {dataset_id}\\n")
        
        return results
    
    def get_manual_download_instructions(self) -> Dict[str, str]:
        """Get manual download instructions for datasets that need it."""
        
        instructions = {}
        
        # Datasets requiring manual download
        manual_datasets = [
            ("apps", "https://github.com/hendrycks/apps", "git clone and extract to benchmarks/datasets/code_generation/apps/"),
            ("natural_questions", "https://ai.google.com/research/NaturalQuestions", "Download from Google AI and extract to benchmarks/datasets/document_qa/natural_questions/"),
            ("spider", "https://yale-lily.github.io/spider", "Download from Yale and extract to benchmarks/datasets/structured_data/spider/"),
            ("big_bench", "https://github.com/google/BIG-bench", "git clone and extract to benchmarks/datasets/multi_domain/big_bench/")
        ]
        
        for dataset_id, url, instruction in manual_datasets:
            if dataset_id in self.datasets:
                info = self.datasets[dataset_id]
                instructions[dataset_id] = {
                    "name": info.name,
                    "url": url, 
                    "instruction": instruction,
                    "target_directory": str(self.data_dir / info.local_path),
                    "paper": info.paper_reference
                }
        
        return instructions
    
    def create_benchmark_integration(self, dataset_id: str) -> str:
        """Create integration code for a specific dataset."""
        
        info = self.datasets.get(dataset_id)
        if not info:
            return ""
        
        integration_code = f'''
def load_{dataset_id}_data(self) -> List[BenchmarkTask]:
    """Load {info.name} dataset for evaluation."""
    
    dataset_path = Path("benchmarks/datasets/{info.local_path}")
    tasks = []
    
    # Load the dataset
    if (dataset_path / "train.json").exists():
        with open(dataset_path / "train.json", 'r') as f:
            data = json.load(f)
        
        for i, item in enumerate(data[:300]):  # Limit for Phase 1
            task = BenchmarkTask(
                domain="{info.domain}",
                task_id=f"{dataset_id}_{{i}}",
                input_text=self._extract_input_{dataset_id}(item),
                expected_output=self._extract_expected_{dataset_id}(item), 
                complexity=self._assess_complexity_{dataset_id}(item),
                metadata={{"dataset": "{info.name}", "original_index": i}}
            )
            tasks.append(task)
    
    return tasks

def _extract_input_{dataset_id}(self, item: Dict) -> str:
    """Extract input text from {info.name} item."""
    # TODO: Implement based on dataset structure
    pass

def _extract_expected_{dataset_id}(self, item: Dict) -> str:
    """Extract expected output from {info.name} item."""  
    # TODO: Implement based on dataset structure
    pass

def _assess_complexity_{dataset_id}(self, item: Dict) -> str:
    """Assess complexity of {info.name} item."""
    # TODO: Implement complexity assessment
    return "simple"
'''
        
        return integration_code

async def main():
    """Main function to manage dataset downloads."""
    
    manager = CommunityDatasetManager()
    
    print("🗂️  SLM Agentic Orchestration - Community Dataset Manager")
    print("="*65)
    
    # List available datasets
    manager.list_available_datasets()
    
    # Download priority datasets
    print("\\n" + "="*65)
    results = await manager.download_priority_datasets()
    
    # Show manual download instructions
    manual_instructions = manager.get_manual_download_instructions()
    
    if manual_instructions:
        print("\\n📋 MANUAL DOWNLOAD REQUIRED")
        print("="*40)
        
        for dataset_id, info in manual_instructions.items():
            if dataset_id not in results or not results[dataset_id]:
                print(f"\\n📊 {info['name']}:")
                print(f"   URL: {info['url']}")
                print(f"   Directory: {info['target_directory']}")
                print(f"   Instructions: {info['instruction']}")
                print(f"   Paper: {info['paper']}")
    
    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\\n📈 DOWNLOAD SUMMARY")
    print("="*25)
    print(f"✅ Automated downloads: {successful}/{total}")
    print(f"📋 Manual downloads needed: {len(manual_instructions)}")
    print("\\n🎯 Priority datasets for Phase 1 evaluation ready!")
    print("\\nNext steps:")
    print("1. Review downloaded datasets in benchmarks/datasets/")
    print("2. Download any manual datasets needed")
    print("3. Run: python enhanced_benchmark.py")

if __name__ == "__main__":
    asyncio.run(main())