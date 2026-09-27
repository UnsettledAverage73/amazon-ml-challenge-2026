import gdown, os

os.makedirs('/content/dataset/train', exist_ok=True)
os.makedirs('/content/dataset/test', exist_ok=True)
os.makedirs('/content/utils', exist_ok=True)

files = {
    '/content/dataset/train/train_ground_truth.tsv': '1Z9L0FVONtvOO1dE9ndgsbmF7BYwD7nnL',
    '/content/dataset/train/train_source1.tsv': '1_o5geQ9zCdblr_4eJFY85KAN_FNUjM8P',
    '/content/dataset/train/train_source2.tsv': '1VBAOMewHthrKcB7CB3S6T8il3-jeSJ9Z',
    '/content/dataset/train/train_source3.tsv': '1-W_5KjuH810ybpkzb_RjP9CXNxjDMfKr',
    '/content/dataset/test/test_source1.tsv': '1YCHIkodt19YOs5i3JBp4nm274sQ7CMZg',
    '/content/dataset/test/test_source2.tsv': '1HWw3ltqt-0qpuHadd9p8ITZw77CtbD4f',
    '/content/dataset/test/test_source3.tsv': '1DM3rXJiAlmGFgQhpV8cU2WkC5vCRzBbF'
}

for path, file_id in files.items():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        gdown.download(id=file_id, output=path, quiet=True)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Verified: {path} ({size_mb:.2f} MB)")

print("All dataset files successfully downloaded and verified on Colab!")
