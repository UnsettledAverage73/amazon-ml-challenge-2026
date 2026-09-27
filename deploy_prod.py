import os
import gzip
import base64
import subprocess
import sys

def main():
    print("=== Step 1: Installing dependencies on Colab prod ===")
    dep_cmd = "import subprocess; print(subprocess.check_output(['pip', 'install', '-q', 'unidecode', 'gdown', 'lightgbm', 'rapidfuzz'], text=True)); print('DEP_OK')"
    res = subprocess.run(['colab', 'exec', '-s', 'prod'], input=dep_cmd, text=True, capture_output=True)
    print("Deps output:", res.stdout.strip())

    print("\n=== Step 2: Downloading test dataset files on Colab prod ===")
    dl_script = """
import gdown, os

os.makedirs("/content/dataset/test", exist_ok=True)
files = {
    "/content/dataset/test/test_source1.tsv": "1YCHIkodt19YOs5i3JBp4nm274sQ7CMZg",
    "/content/dataset/test/test_source2.tsv": "1HWw3ltqt-0qpuHadd9p8ITZw77CtbD4f",
    "/content/dataset/test/test_source3.tsv": "1DM3rXJiAlmGFgQhpV8cU2WkC5vCRzBbF"
}

for path, file_id in files.items():
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        print(f"Downloading {path}...")
        gdown.download(id=file_id, output=path, quiet=True)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Verified: {path} ({size_mb:.2f} MB)")

print("DATA_READY")
"""
    res = subprocess.run(['colab', 'exec', '-s', 'prod'], input=dl_script, text=True, capture_output=True)
    print("DL output:", res.stdout.strip())

    print("\n=== Step 3: Packing and uploading source code & models ===")
    files_to_pack = {
        '/content/src/normalization.py': 'amazon_ml_challenge/src/normalization.py',
        '/content/src/blocking.py': 'amazon_ml_challenge/src/blocking.py',
        '/content/src/features.py': 'amazon_ml_challenge/src/features.py',
        '/content/src/resolution.py': 'amazon_ml_challenge/src/resolution.py',
        '/content/models/top_tier_lightgbm.txt': 'amazon_ml_challenge/models/top_tier_lightgbm.txt',
        '/content/models/top_tier_idf.json': 'amazon_ml_challenge/models/top_tier_idf.json',
        '/content/generate_top_tier_submission.py': 'amazon_ml_challenge/generate_top_tier_submission.py',
        '/content/validate_submission.py': 'amazon_ml_challenge/validate_submission.py',
    }

    manifest = {}
    for remote_path, local_path in files_to_pack.items():
        with open(local_path, 'rb') as f:
            data = gzip.compress(f.read())
            manifest[remote_path] = base64.b64encode(data).decode('ascii')

    unpack_script = ['import os, gzip, base64']
    for r_path, b64_str in manifest.items():
        unpack_script.append(f'os.makedirs(os.path.dirname("{r_path}"), exist_ok=True)')
        unpack_script.append(f'with open("{r_path}", "wb") as f:')
        unpack_script.append(f'    f.write(gzip.decompress(base64.b64decode("{b64_str}")))')
    unpack_script.append('print("UNPACK_COMPLETE")')

    res = subprocess.run(['colab', 'exec', '-s', 'prod'], input='\n'.join(unpack_script), text=True, capture_output=True)
    print("Unpack output:", res.stdout.strip())

    print("\n=== Step 4: Launching accelerated 1-hour submission pipeline in background ===")
    launch_script = """
import subprocess
proc = subprocess.Popen(
    "python3 -u /content/generate_top_tier_submission.py > /content/submission.log 2>&1 &",
    shell=True
)
print("LAUNCHED_PID")
"""
    res = subprocess.run(['colab', 'exec', '-s', 'prod'], input=launch_script, text=True, capture_output=True)
    print("Launch output:", res.stdout.strip())

if __name__ == '__main__':
    main()
