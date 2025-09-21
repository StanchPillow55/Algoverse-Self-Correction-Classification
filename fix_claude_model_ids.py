#!/usr/bin/env python3
"""
Fix all instances of the incorrect Claude 3.5 Sonnet model ID.
Changes claude-3-5-sonnet-20241210 to claude-3-5-sonnet-20241210
"""

import os
import re
from pathlib import Path

# The incorrect model ID that needs to be replaced
OLD_MODEL_ID = "claude-3-5-sonnet-20241210"
NEW_MODEL_ID = "claude-3-5-sonnet-20241210"

def fix_file(file_path: Path) -> bool:
    """Fix model ID in a single file."""
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the old model ID exists in the file
        if OLD_MODEL_ID not in content:
            return False
        
        # Replace the model ID
        updated_content = content.replace(OLD_MODEL_ID, NEW_MODEL_ID)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ Fixed: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def find_and_fix_files(root_dir: Path = None) -> int:
    """Find and fix all files containing the old model ID."""
    if root_dir is None:
        root_dir = Path(".")
    
    # File extensions to check
    extensions = ['.py', '.json', '.yaml', '.yml', '.md', '.txt', '.sh']
    
    # Directories to skip
    skip_dirs = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache'}
    
    fixed_count = 0
    
    for file_path in root_dir.rglob('*'):
        # Skip directories
        if file_path.is_dir():
            continue
        
        # Skip files in excluded directories
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            continue
        
        # Only check relevant file extensions
        if file_path.suffix not in extensions:
            continue
        
        # Try to fix the file
        if fix_file(file_path):
            fixed_count += 1
    
    return fixed_count

def main():
    print("🔍 Searching for files with old Claude 3.5 Sonnet model ID...")
    print(f"Replacing: {OLD_MODEL_ID}")
    print(f"With:      {NEW_MODEL_ID}")
    print("-" * 60)
    
    # Find and fix files
    fixed_count = find_and_fix_files()
    
    print("-" * 60)
    print(f"✅ Fixed {fixed_count} files!")
    
    if fixed_count > 0:
        print("\n💡 Next steps:")
        print("1. Review the changes in the fixed files")
        print("2. Re-run any failed Claude Sonnet experiments")
        print("3. Test API connection with new model ID")
    else:
        print("\n✅ All files are already up to date!")

if __name__ == "__main__":
    main()