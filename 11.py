import os

folder = "."

for filename in os.listdir(folder):
    if filename.endswith(",json"):
        new_name = filename.replace(",json", ".json")
        os.rename(filename, new_name)
        print(f"Renamed: {filename} -> {new_name}")

print("Correction terminée.")