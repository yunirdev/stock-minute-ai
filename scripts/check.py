"""Quick sanity-check: list all locally cached bar files and their date ranges."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.data_cache import list_cached_files

files = list_cached_files()
if not files:
    print("No cached bar files found in data/bars/")
    sys.exit(0)

print(f"{'File':<35} {'Rows':>7}  {'Start':>10}  {'End':>10}  {'Updated':>16}  {'KB':>7}")
print("-" * 95)
for f in files:
    print(
        f"{f['文件']:<35} {str(f['行数']):>7}  "
        f"{str(f['起始']):>10}  {str(f['截止']):>10}  "
        f"{str(f['更新时间']):>16}  {str(f['大小(KB)']):>7}"
    )
