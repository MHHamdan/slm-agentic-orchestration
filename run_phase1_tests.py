#!/usr/bin/env python3
"""
Phase 1 Test Runner - Validate P0 Implementation
Executes proof-of-concept and minimum viable benchmark
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def run_proof_of_concept():
    """Run the proof-of-concept demonstration."""
    print("🎭 RUNNING PROOF-OF-CONCEPT DEMO")
    print("="*50)
    
    try:
        # Import and run proof of concept
        from proof_of_concept_demo import main as poc_main
        await poc_main()
        return True
    except Exception as e:
        print(f"❌ Proof-of-concept failed: {e}")
        return False

async def run_minimum_viable_benchmark():
    """Run the minimum viable benchmark."""
    print("\\n\\n📊 RUNNING MINIMUM VIABLE BENCHMARK")
    print("="*50)
    
    try:
        # Import and run benchmark
        from minimum_viable_benchmark import main as benchmark_main
        await benchmark_main()
        return True
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("⚙️  TESTING CONFIGURATION")
    print("="*30)
    
    try:
        from configs.config import config
        
        print(f"✓ Project root: {config.PROJECT_ROOT}")
        print(f"✓ SLM models: {len(config.slm_models) if hasattr(config, 'slm_models') else 0}")
        print(f"✓ LLM models: {len(config.llm_models) if hasattr(config, 'llm_models') else 0}")
        print(f"✓ API keys configured: {bool(config.OPENAI_API_KEY and config.ANTHROPIC_API_KEY)}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def main():
    """Run all Phase 1 tests."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                  PHASE 1 TEST RUNNER                     ║
║            Priority P0 - Critical Foundation            ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    results = []
    
    # Test 1: Configuration
    print("Test 1/3: Configuration Loading")
    config_success = test_configuration()
    results.append(("Configuration", config_success))
    
    if not config_success:
        print("⚠️  Skipping further tests due to configuration failure")
        return
    
    # Test 2: Proof of Concept
    print("\\nTest 2/3: Proof-of-Concept Demo")
    poc_success = await run_proof_of_concept()
    results.append(("Proof-of-Concept", poc_success))
    
    # Test 3: Minimum Viable Benchmark (only if PoC succeeds)
    if poc_success:
        print("\\nTest 3/3: Minimum Viable Benchmark")
        benchmark_success = await run_minimum_viable_benchmark()
        results.append(("Benchmark", benchmark_success))
    else:
        print("\\n⚠️  Skipping benchmark due to PoC failure")
        results.append(("Benchmark", False))
    
    # Summary
    total_time = time.time() - start_time
    
    print("\\n" + "="*60)
    print("📋 PHASE 1 TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<20} {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\\n🎯 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total runtime: {total_time:.1f} seconds")
    
    if passed == total:
        print("\\n🎉 Phase 1 P0 implementation READY!")
        print("Next steps:")
        print("  1. Review results in poc_results.json")
        print("  2. Check benchmark_results/ directory")
        print("  3. Proceed to Phase 1 Month 2: Core Algorithm Implementation")
    else:
        print("\\n⚠️  Some tests failed. Review errors above.")
        print("  Fix issues before proceeding to next phase.")

if __name__ == "__main__":
    asyncio.run(main())