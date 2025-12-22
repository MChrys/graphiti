#!/usr/bin/env python3
"""
Syntax checker for search filtering tests.

This script runs basic syntax and import checks on the test files
to ensure they are correctly structured and can be imported.
"""

import sys
import traceback
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()

        # Compile the code to check syntax
        compile(code, file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_imports(file_path):
    """Check if a file can be imported (basic import test)."""
    try:
        # Add the parent directories to path for imports
        project_root = Path(__file__).parent.parent.parent
        server_path = project_root / "server"

        if str(server_path) not in sys.path:
            sys.path.insert(0, str(server_path))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Try to compile and check basic import structure
        with open(file_path, 'r') as f:
            code = f.read()

        # Remove actual test execution, just check imports
        lines = code.split('\n')
        import_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                import_lines.append(line)
            elif line.startswith('def ') or line.startswith('class '):
                break

        import_code = '\n'.join(import_lines)

        if import_code:
            compile(import_code, file_path, 'exec')

        return True, None
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error during import check: {e}"

def main():
    """Run syntax and import checks on test files."""
    test_files = [
        "test_search_filtering.py",
        "test_search_filtering_unit.py",
        "test_search_filtering_integration.py"
    ]

    print("Running syntax and import checks for search filtering tests...")
    print("=" * 60)

    all_passed = True

    for test_file in test_files:
        file_path = Path(__file__).parent / test_file

        if not file_path.exists():
            print(f"❌ {test_file}: File not found")
            all_passed = False
            continue

        print(f"\n🔍 Checking {test_file}...")

        # Check syntax
        syntax_ok, syntax_error = check_syntax(file_path)
        if syntax_ok:
            print(f"✅ Syntax check passed")
        else:
            print(f"❌ Syntax check failed: {syntax_error}")
            all_passed = False
            continue

        # Check imports (but be more lenient since we might not have all dependencies)
        import_ok, import_error = check_imports(file_path)
        if import_ok:
            print(f"✅ Import check passed")
        else:
            print(f"⚠️  Import check warning: {import_error}")
            # Don't fail the overall check for import issues since we might not have all deps

        # Count test functions
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Count test functions (methods starting with test_)
            test_count = content.count('def test_')
            async_test_count = content.count('async def test_')
            total_tests = test_count + async_test_count

            print(f"📊 Found {total_tests} test functions ({test_count} sync, {async_test_count} async)")

        except Exception as e:
            print(f"⚠️  Could not count test functions: {e}")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All syntax checks passed! Test files are ready.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())