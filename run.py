#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src/physical_os to python path so 'import vision_engine' works (legacy support)
root_dir = Path(__file__).parent.absolute()
package_dir = root_dir / "src" / "physical_os"
sys.path.insert(0, str(package_dir))

# Also add src so 'from physical_os import main' works if needed generally
src_dir = root_dir / "src"
sys.path.insert(0, str(src_dir))

# Import main (now findable because package_dir is in path)
import main

if __name__ == "__main__":
    # Ensure CWD is root so data folders are found
    os.chdir(root_dir)
    
    # Run the main function
    main.main()
