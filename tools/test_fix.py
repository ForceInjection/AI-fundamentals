import os
import urllib.parse
from auto_fix_links import build_file_map

base_dir = "/Users/wangtianqing/Project/study/AI-fundermentals"
file_map = build_file_map(base_dir)
print("mooncake_architecture.md in map:", file_map.get("mooncake_architecture.md"))
