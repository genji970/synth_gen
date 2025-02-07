import glob
import os
import subprocess

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Find all Python files in the current directory
python_files = glob.glob(os.path.join(current_directory, "*.py"))

# Execute each Python file
for file in python_files:
    if file != os.path.abspath(__file__):  # Avoid executing the current file
        subprocess.run(["python", file], check=True)
