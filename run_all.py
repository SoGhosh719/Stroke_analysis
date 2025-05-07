# run_all.py

import subprocess
import os

def run_script(script_path):
    print(f"\n🚀 Running: {script_path}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print("📄 Output:\n", result.stdout)
    if result.stderr:
        print("⚠️ Errors:\n", result.stderr)

if __name__ == "__main__":
    print("🧠 Stroke Analysis - Pipeline Runner\n")

    # Paths to scripts
    scripts = [
        "src/eda.py",
        "src/train_model.py",
        "src/evaluate_model.py"
    ]

    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"❌ Script not found: {script}")

    print("\n✅ All scripts executed.")

