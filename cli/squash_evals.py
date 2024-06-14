import os, json

from cli.utils.constants import RESULTS_DIR


def update_json(orig, new_json):
    for k in orig.keys():
        if k in new_json.keys():
            orig[k].update(new_json[k])
            del new_json[k]
    orig.update(new_json)


def combine_json_files(folder_path):
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    
    if not json_files:
        print("No JSON files found in the folder.")
        return
    print(f'Found files: {json_files}')
    
    combined_data = {}

    for filename in json_files:
        with open(os.path.join(folder_path, filename), 'r') as f:
            # for k, v in json.load(f).items():
            #     combined_data[k] = v
            update_json(combined_data, json.load(f))
    
    combined_file_path = os.path.join(RESULTS_DIR, json_files[0].split('.')[0] + '_combined.json')
    with open(combined_file_path, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined JSON file saved as: {combined_file_path}")


if __name__ == "__main__":
    folder_path = '4-9-code-llama-13b-all-subsets'
    combine_json_files(os.path.join(RESULTS_DIR, folder_path))
