import importlib.util

spec = importlib.util.spec_from_file_location("seed_kg", "scripts/seed_kg.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("SEED sections:", sorted(mod.SEED))
print("count_seed:", mod.count_seed())
merged, added = mod.merge(mod.SEED, {})
print("merged sections:", sorted(merged))
print("added count:", len(added), "seed count:", mod.count_seed())
