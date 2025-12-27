#!/usr/bin/env python3
"""
Quick script to check if backend can start and verify configuration.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to load .env
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
load_dotenv()

def check_env():
    """Check if .env is configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not set in .env file")
        print(f"   .env path: {env_path}")
        return False
    print(f"✓ GEMINI_API_KEY found (length: {len(api_key)})")
    return True

def check_imports():
    """Check if all imports work."""
    try:
        print("Checking imports...")
        from pipeline1_hyde import HyDEPipeline
        from pipeline2_semantic import SemanticHeuristicPipeline
        from final_selection import FinalSelector
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def check_initialization():
    """Check if pipelines can initialize."""
    try:
        print("Initializing pipelines...")
        from pipeline1_hyde import HyDEPipeline
        from pipeline2_semantic import SemanticHeuristicPipeline
        from final_selection import FinalSelector
        
        hyde = HyDEPipeline()
        semantic = SemanticHeuristicPipeline()
        selector = FinalSelector()
        print("✓ All pipelines initialized")
        return True
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Backend Configuration Check")
    print("=" * 60)
    print()
    
    if not check_env():
        print("\n❌ Please set GEMINI_API_KEY in .env file")
        return 1
    
    if not check_imports():
        print("\n❌ Please install dependencies: pip install -r requirements.txt")
        return 1
    
    if not check_initialization():
        print("\n❌ Pipeline initialization failed. Check error above.")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ All checks passed! Backend is ready to start.")
    print("=" * 60)
    print("\nTo start the backend:")
    print("  cd backend")
    print("  python main.py")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

