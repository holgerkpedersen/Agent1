"""Scratch probe: does flush+print of many lines crash? (deleted after)"""
for i in range(400):
    print(f"line {i}", flush=True)
print("done", flush=True)
