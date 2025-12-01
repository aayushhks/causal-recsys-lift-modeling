import os
import shutil


def reorganize():
    # Define the target structure based on your roadmap
    structure = {
        "notebooks": [],
        "data": ["raw", "processed"],
        "src": ["pipeline", "ab_testing", "causal", "dashboard"],
        "docs": [],
        "tests": []
    }

    print("🚀 Reorganizing project into FAANG-standard structure...")

    # 1. Create Directories
    for folder, subfolders in structure.items():
        os.makedirs(folder, exist_ok=True)
        for sub in subfolders:
            os.makedirs(os.path.join(folder, sub), exist_ok=True)

    # 2. Move Files (Smart Move)
    moves = {
        # Current Path -> New Path
        "src/data_pipeline.py": "src/pipeline/data_pipeline.py",
        "src/ab_test.py": "src/ab_testing/bayesian_engine.py",  # Renaming for clarity
        "src/causal_model.py": "src/causal/inference_engine.py",
        "dashboard/app.py": "src/dashboard/app.py",
        "run_ab_simulation.py": "notebooks/00_quick_start.py"  # Moving scripts to notebooks/scripts
    }

    for src, dst in moves.items():
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"✅ Moved: {src} -> {dst}")
        else:
            print(f"⚠️ Skipped (not found): {src}")

    # 3. Create placeholder README if not exists
    if not os.path.exists("README.md"):
        with open("README.md", "w") as f:
            f.write("# Project Placeholder")

    print("\n🎉 Project structure is now clean.")


if __name__ == "__main__":
    reorganize()