import subprocess
import sys
from pathlib import Path

repo_dir = Path("MUSC_violin")
if repo_dir.exists():
    print("Repository already exists at", repo_dir)
else:
    subprocess.run(["git", "clone", "https://github.com/MTG/violin-transcription.git", str(repo_dir)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(repo_dir / "requirements.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(repo_dir / "requirements_dev.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)], check=True)
