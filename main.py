# Save this as: setup_project.py
import os


def create_structure():
    structure = {
        "data": ["raw", "processed", "features"],
        "notebooks": [],
        "src": [],
        "models": ["ranking", "uplift"],
        "dashboard": [],
        "sql": []
    }

    files = [
        "README.md",
        "requirements.txt",
        "src/__init__.py",
        "src/data_pipeline.py",
        "src/feature_engineering.py",
        "src/models.py",
        "src/inference.py",
        "dashboard/app.py"
    ]

    print("Initializing Project Structure...")

    # Create Directories
    for folder, subfolders in structure.items():
        os.makedirs(folder, exist_ok=True)
        print(f" Created: {folder}/")
        for sub in subfolders:
            path = os.path.join(folder, sub)
            os.makedirs(path, exist_ok=True)
            print(f" Created: {path}/")

    # Create Files
    for file in files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                pass  # Create empty file
            print(f" Created: {file}")
        else:
            print(f" Exists: {file}")

    print("\n Project structure ready!")


if __name__ == "__main__":
    create_structure()