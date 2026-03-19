import os
import json
import re

# Directory paths
BLIP2_DIR = 'BLIP2'
NORMALIZED_DIR = 'BLIP2_normalized'

# Ensure normalized directory exists
os.makedirs(NORMALIZED_DIR, exist_ok=True)

def flatten_text(obj):
    """Flatten nested objects/arrays to a single string."""
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        return ' '.join(str(item) for item in obj if item)
    elif isinstance(obj, dict):
        return ' '.join(f"{k}: {flatten_text(v)}" for k, v in obj.items())
    else:
        return str(obj)

def normalize_blip2_file(filepath):
    """Normalize a single BLIP2 JSON file to common schema."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    normalized = {
        "name": "",
        "scientific_name": "",
        "causal_agent": "",
        "hosts": [],
        "symptoms": "",
        "description": "",
        "management": "",
        "prevention": "",
        "sources": []
    }

    # Map existing keys to normalized fields
    if 'disease' in data:
        normalized['name'] = data['disease']
    elif 'disease_name' in data:
        normalized['name'] = data['disease_name']
    elif 'pest' in data:
        normalized['name'] = data['pest']

    if 'scientific_name' in data:
        normalized['scientific_name'] = data['scientific_name']
    elif 'synonym' in data:
        normalized['scientific_name'] = data['synonym']

    if 'causal_agent' in data:
        normalized['causal_agent'] = data['causal_agent']
    elif 'other_agents' in data:
        normalized['causal_agent'] = ', '.join(data['other_agents'])

    if 'hosts' in data:
        if isinstance(data['hosts'], list):
            normalized['hosts'] = data['hosts']
        elif isinstance(data['hosts'], dict):
            # Flatten primary/secondary hosts
            hosts = []
            for key, value in data['hosts'].items():
                if isinstance(value, list):
                    hosts.extend(value)
                else:
                    hosts.append(value)
            normalized['hosts'] = hosts

    if 'symptoms' in data:
        normalized['symptoms'] = flatten_text(data['symptoms'])
    elif 'symptoms_and_damage' in data:
        normalized['symptoms'] = flatten_text(data['symptoms_and_damage'])

    if 'description' in data:
        normalized['description'] = data['description']

    if 'management' in data:
        normalized['management'] = flatten_text(data['management'])
    elif 'cultural_control' in data:
        normalized['management'] = flatten_text(data['cultural_control'])
    elif 'biological_control' in data:
        normalized['management'] = flatten_text(data['biological_control'])
    elif 'chemical_control' in data:
        normalized['management'] = flatten_text(data['chemical_control'])

    if 'prevention' in data:
        normalized['prevention'] = flatten_text(data['prevention'])

    if 'sources' in data:
        if isinstance(data['sources'], list):
            normalized['sources'] = data['sources']
        else:
            normalized['sources'] = [data['sources']]
    elif 'references' in data:
        if isinstance(data['references'], list):
            normalized['sources'] = data['references']
        else:
            normalized['sources'] = [data['references']]

    return normalized

def main():
    for filename in os.listdir(BLIP2_DIR):
        if filename.endswith('.json') or not '.' in filename:  # Handle files without extension
            filepath = os.path.join(BLIP2_DIR, filename)
            try:
                normalized_data = normalize_blip2_file(filepath)
                # Ensure output has .json extension
                if not filename.endswith('.json'):
                    filename += '.json'
                output_path = os.path.join(NORMALIZED_DIR, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(normalized_data, f, indent=2, ensure_ascii=False)
                print(f"Normalized: {filename}")
            except Exception as e:
                print(f"Error normalizing {filename}: {e}")

if __name__ == "__main__":
    main()