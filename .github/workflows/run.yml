import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: python dispatch.py <node_id>")
    sys.exit(1)

node_id = int(sys.argv[1])  # 1 to 20
total_nodes = 20

# 10^20 から 10^27 の領域を20台で均等分割 (128-bit長整数の演算)
SPACE_START = 10**20
SPACE_END = 10**27
CHUNK_SIZE = (SPACE_END - SPACE_START) // total_nodes

y_start = SPACE_START + CHUNK_SIZE * (node_id - 1)
y_end = y_start + CHUNK_SIZE - 1

if node_id == total_nodes:
    y_end = SPACE_END

print(f"🚀 Node {node_id} / {total_nodes} 起動")
print(f"📦 担当領域: y = {y_start} 〜 {y_end}")

# ネイティブバイナリへ巨大整数を文字列として渡す
subprocess.run([
    "./deep_space", 
    str(y_start), 
    str(y_end)
])
