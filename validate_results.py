# File: validate_results.py
"""
Validation script to ensure P0 results meet publication requirements
"""
import json
import pandas as pd
from pathlib import Path

def validate_p0_results(results_dir: str = "./results"):
    """Validate that P0 results meet minimum requirements."""
    
    results_path = Path(results_dir)
    
    # Load results
    csv_path = results_path / "evaluation_results.csv"
    if not csv_path.exists():
        print("❌ No results found. Run experiment first.")
        return False
    
    df = pd.read_csv(csv_path)
    
    # Calculate key metrics
    slm_models = ['gemma-2b', 'phi-3-mini', 'llama-3.2-3b']
    slm_results = df[df['model_name'].isin(slm_models)]
    llm_results = df[~df['model_name'].isin(slm_models)]
    
    # Validation criteria
    checks = {
        "Minimum tasks": len(df) >= 100,
        "SLM coverage > 65%": (len(slm_results) / len(df)) >= 0.65,
        "Cost reduction > 5x": (llm_results['cost_usd'].mean() / slm_results['cost_usd'].mean()) >= 5,
        "Success rate > 80%": slm_results['success'].mean() >= 0.8,
        "Latency improvement > 2x": (llm_results['latency_ms'].mean() / slm_results['latency_ms'].mean()) >= 2
    }
    
    print("P0 Validation Results:")
    print("=" * 40)
    
    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}: {passed}")
        all_passed = all_passed and passed
    
    print("=" * 40)
    if all_passed:
        print("✅ P0 implementation meets publication requirements!")
    else:
        print("❌ Some requirements not met. Continue optimization.")
    
    return all_passed

if __name__ == "__main__":
    validate_p0_results()