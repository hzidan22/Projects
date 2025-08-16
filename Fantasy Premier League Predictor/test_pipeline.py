

import sys
import os

def test_imports():
    
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas not found - run: pip install pandas")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError:
        print("❌ numpy not found - run: pip install numpy")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError:
        print("❌ scikit-learn not found - run: pip install scikit-learn")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError:
        print("❌ requests not found - run: pip install requests")
        return False
    
    optional_packages = ['xgboost', 'lightgbm', 'matplotlib', 'seaborn']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"⚠️  {package} not found (optional) - run: pip install {package}")
    
    return True

def test_modules():
    
    print("\n🔧 Testing custom modules...")
    
    modules = [
        'data_gathering', 'data_preprocessing', 'model_training', 
        'team_selection', 'config', 'utils'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ {module} failed to import: {e}")
            return False
    
    return True

def test_config():
    
    print("\n⚙️  Testing configuration...")
    
    try:
        from config import Config
        
        assert Config.BUDGET == 100.0, "Budget should be 100.0"
        assert Config.MAX_PLAYERS_PER_TEAM == 3, "Max players per team should be 3"
        assert Config.SQUAD_SIZE == 15, "Squad size should be 15"
        
        Config.create_directories()
        
        directories = [Config.DATA_DIR, Config.MODEL_DIR, Config.OUTPUT_DIR]
        for directory in directories:
            if os.path.exists(directory):
                print(f"✅ Directory {directory} exists")
            else:
                print(f"❌ Directory {directory} not created")
                return False
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_api_connection():
    
    print("\n🌐 Testing FPL API connection...")
    
    try:
        import requests
        from config import Config
        
        response = requests.get(f"{Config.FPL_BASE_URL}bootstrap-static/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            player_count = len(data.get('elements', []))
            team_count = len(data.get('teams', []))
            
            print(f"✅ FPL API connected successfully")
            print(f"   📊 Found {player_count} players and {team_count} teams")
            return True
        else:
            print(f"❌ FPL API returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ FPL API connection failed: {e}")
        return False

def run_all_tests():
    
    print("🚀 Starting FPL Predictor Tests")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_modules),
        ("Configuration", test_config),
        ("API Connection", test_api_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready to run the FPL predictor!")
        print("\nTo run the full pipeline:")
        print("python main.py")
        return True
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
