import ijson
import json
import os
from decimal import Decimal

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def filter_json_stream(input_path, output_path):
    """
    Filters a large JSON file based on specific criteria using streaming.
    It keeps only the chats where the name is "Silvia" or "Meri".
    """
    # Create the directory for the output path if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'rb') as f_in, open(output_path, 'w') as f_out:
        # This assumes a structure like {"chats": {"list": [...]}}
        # We will reconstruct the JSON structure around the filtered list.
        
        # Find the top-level keys to reconstruct the structure
        # This is a simplified approach. A more robust solution might be needed
        # if the structure is more complex or varies.
        
        f_out.write('{\n  "chats": {\n    "list": [\n')
        
        first_item = True
        # Use ijson to iteratively parse the 'list' array
        chats = ijson.items(f_in, 'chats.list.item')
        
        for chat in chats:
            if chat.get('name') in ["Silvia", "Meri"]:
                if not first_item:
                    f_out.write(',\n')
                json.dump(chat, f_out, indent=4, ensure_ascii=False, default=decimal_default)
                first_item = False
        
        f_out.write('\n    ]\n  }\n}\n')

if __name__ == '__main__':
    # Using absolute paths for clarity
    input_file = '/Users/eddavtyan/Documents/Passed Courses/IU/Semester 2/Thesis/CausalWhatNot/result.json'
    output_file = '/Users/eddavtyan/Documents/Passed Courses/IU/Semester 2/Thesis/CausalWhatNot/result.filtered.json'
    
    print(f"Starting to filter {input_file}...")
    filter_json_stream(input_file, output_file)
    print(f"Filtered JSON saved to {output_file}")

