#!/usr/bin/env python3
"""
Verification script to check if everything is set up correctly.
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has API key."""
    print("🔍 Checking .env file...")
    
    # Check in parent directory (project root)
    env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        print("❌ .env file not found!")
        print(f"   Expected location: {env_path}")
        print("   Please create .env file from .env.example and add your GEMINI_API_KEY")
        return False
    
    print(f"✓ .env file found at: {env_path}")
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv(env_path)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not set or still has placeholder value!")
        print("   Please set GEMINI_API_KEY in .env file")
        return False
    
    print(f"✓ GEMINI_API_KEY is set (length: {len(api_key)} chars)")
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n🔍 Checking Python dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'google.generativeai',
        'sentence_transformers',
        'networkx',
        'numpy',
        'sklearn',
        'spacy',
        'dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_spacy_model():
    """Check if spaCy model is installed."""
    print("\n🔍 Checking spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_md")
        print("✓ en_core_web_md model is installed")
        return True
    except OSError:
        print("❌ en_core_web_md model is NOT installed")
        print("   Run: python -m spacy download en_core_web_md")
        return False

def check_backend_files():
    """Check if all backend files exist."""
    print("\n🔍 Checking backend files...")
    
    backend_dir = Path(__file__).parent
    required_files = [
        'main.py',
        'pipeline1_hyde.py',
        'pipeline2_semantic.py',
        'final_selection.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for file in required_files:
        file_path = backend_dir / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_api_connection():
    """Test if Gemini API key works."""
    print("\n🔍 Testing Gemini API connection...")
    
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Cannot test: GEMINI_API_KEY not set")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Simple test
        response = model.generate_content("Say 'test'")
        if response and response.text:
            print("✓ Gemini API connection successful")
            return True
        else:
            print("❌ Gemini API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False

def main():
    """Run all checks."""
    print("=" * 70)
    print("KNIT-LLM Setup Verification")
    print("=" * 70)
    print()
    
    checks = [
        ("Environment File", check_env_file),
        ("Backend Files", check_backend_files),
        ("Python Dependencies", check_dependencies),
        ("spaCy Model", check_spacy_model),
        ("Gemini API Connection", check_api_connection),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ All checks passed! You're ready to start the backend.")
        print("\nTo start the backend:")
        print("  cd backend")
        print("  python main.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

