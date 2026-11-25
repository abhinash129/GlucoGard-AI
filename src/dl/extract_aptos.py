import zipfile
from tqdm import tqdm
from src.config import DATA_RAW

folder = DATA_RAW / "aptos"
zip_path = folder / "aptos2019-blindness-detection.zip"

bad = []
with zipfile.ZipFile(zip_path) as zf:
    members = [m for m in zf.infolist() if not m.is_dir()]
    print("files in zip:", len(members))
    for m in tqdm(members, desc="extracting"):
        try:
            zf.extract(m, folder)          # also verifies the CRC checksum
        except Exception as e:
            bad.append(m.filename)
            (folder / m.filename).unlink(missing_ok=True)   # remove partial file

print(f"\nunreadable files: {len(bad)}")
for name in bad[:30]:
    print(" -", name)
print("bad files in train_images (these matter):",
      sum(n.startswith("train_images/") for n in bad))