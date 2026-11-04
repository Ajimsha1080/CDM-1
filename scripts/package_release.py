import os
import zipfile
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOWNLOADS_DIR = r"C:\Users\91730\Downloads"

EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".idea", ".vscode", "venv", ".venv"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".sqlite", ".sqlite3", ".db"}

def build_zip(zip_path, include_weights=False):
    print(f"Building {zip_path} (include_weights={include_weights})...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in EXCLUDE_EXTS:
                    continue
                if file.endswith(".zip"):
                    continue
                if ext == ".safetensors" and not include_weights:
                    continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ROOT_DIR)
                zf.write(full_path, arcname=os.path.join("CDM-1", rel_path))
    
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Completed {zip_path}: {size_mb:.2f} MB")

if __name__ == "__main__":
    proj_zip = os.path.join(ROOT_DIR, "CDM-1-Project.zip")
    weights_zip = os.path.join(ROOT_DIR, "CDM-1-Complete-With-Weights.zip")
    
    build_zip(proj_zip, include_weights=False)
    build_zip(weights_zip, include_weights=True)
    
    # Copy to user Downloads
    shutil.copy2(proj_zip, os.path.join(DOWNLOADS_DIR, "CDM-1-Project.zip"))
    shutil.copy2(weights_zip, os.path.join(DOWNLOADS_DIR, "CDM-1-Complete-With-Weights.zip"))
    print("Copied zip archives to Downloads folder.")
