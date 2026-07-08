#!/usr/bin/env python3
"""Extract information from wildreceipt dataset.

Usage:
    python extract_wildreceipt.py --data_dir <path_to_wildreceipt> [options]

Examples:
    # Process both train and test with pseudo paths
    python extract_wildreceipt.py --data_dir . --pseudo_path

    # Process only train set
    python extract_wildreceipt.py --data_dir . --split train --pseudo_path

Arguments:
    --data_dir       Directory containing wildreceipt data (required)
    --split          Data split: train, test (default: all splits)
    --output_file    Output JSONL file path (default: extracted_wildreceipt.jsonl)
    --pseudo_path    Use pseudo image paths (path/to/image.jpg)

Output Format:
    Each line is a JSON object with:
    - image_info: List of image references with matched_text_index and image_url
    - text_info: List of text entries with "mask" (OCR placeholder) and "no_mask" (extracted data as JSON string)

Field Mapping (from class_list.txt):
    - Store_name_value (1) / Store_name_key (2) → store_name
    - Store_addr_value (3) / Store_addr_key (4) → store_addr
    - Tel_value (5) / Tel_key (6) → tel
    - Date_value (7) / Date_key (8) → date
    - Time_value (9) / Time_key (10) → time
    - Prod_item_value (11) / Prod_item_key (12) → products
    - Prod_quantity_value (13) / Prod_quantity_key (14) → quantities
    - Prod_price_value (15) / Prod_price_key (16) → prices
    - Subtotal_value (17) / Subtotal_key (18) → subtotal
    - Tax_value (19) / Tax_key (20) → tax
    - Tips_value (21) / Tips_key (22) → tips
    - Total_value (23) / Total_key (24) → total
    - Others (25) → others
    - Ignore (0) → ignored
"""

import json
import os
import argparse


# Class ID to field name mapping (from class_list.txt)
CLASS_MAP = {
    0: None,           # Ignore
    1: "store_name",   # Store_name_value
    2: "store_name",   # Store_name_key
    3: "store_addr",   # Store_addr_value
    4: "store_addr",   # Store_addr_key
    5: "tel",          # Tel_value
    6: "tel",          # Tel_key
    7: "date",         # Date_value
    8: "date",         # Date_key
    9: "time",         # Time_value
    10: "time",        # Time_key
    11: "products",    # Prod_item_value
    12: "products",    # Prod_item_key
    13: "quantities",  # Prod_quantity_value
    14: "quantities",  # Prod_quantity_key
    15: "prices",      # Prod_price_value
    16: "prices",      # Prod_price_key
    17: "subtotal",    # Subtotal_value
    18: "subtotal",    # Subtotal_key
    19: "tax",         # Tax_value
    20: "tax",         # Tax_key
    21: "tips",        # Tips_value
    22: "tips",        # Tips_key
    23: "total",       # Total_value
    24: "total",       # Total_key
    25: "others",      # Others
}


def extracted_data_to_string(extracted_data):
    """Convert extracted_data dict to JSON string format."""
    return json.dumps(extracted_data, ensure_ascii=False)


def get_image_path(file_name, data_dir, use_pseudo_path):
    """Get image path based on configuration."""
    if use_pseudo_path:
        # Extract just the filename from the path
        filename = os.path.basename(file_name)
        return f"path/to/{filename}"
    else:
        return os.path.join(data_dir, file_name)


def extract_annotations(annotations):
    """Extract and group annotations by field type."""
    result = {}
    
    for ann in annotations:
        label = ann["label"]
        text = ann["text"]
        
        # Skip ignored labels
        if label == 0 or CLASS_MAP.get(label) is None:
            continue
        
        field_name = CLASS_MAP[label]
        
        if field_name not in result:
            result[field_name] = []
        
        # Add text if not empty
        if text.strip():
            result[field_name].append(text)
    
    # Concatenate multiple values for same field
    for field_name in result:
        if len(result[field_name]) == 1:
            result[field_name] = result[field_name][0]
        else:
            result[field_name] = "".join(result[field_name])
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract information from wildreceipt dataset")
    parser.add_argument("--data_dir", required=True, help="Directory containing wildreceipt data")
    parser.add_argument("--split", default=None, help="Data split (train, test)")
    parser.add_argument("--output_file", default="extracted_wildreceipt.jsonl", help="Output JSONL file path")
    parser.add_argument("--pseudo_path", action="store_true", help="Use pseudo image paths")
    
    args = parser.parse_args()
    
    # Find splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = []
        if os.path.exists(os.path.join(args.data_dir, "train.txt")):
            splits.append("train")
        if os.path.exists(os.path.join(args.data_dir, "test.txt")):
            splits.append("test")
    
    print(f"Processing splits: {splits}")
    
    total_records = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for split in splits:
            split_file = os.path.join(args.data_dir, f"{split}.txt")
            
            if not os.path.exists(split_file):
                print(f"Split file not found: {split_file}")
                continue
            
            print(f"Processing {split}.txt...")
            
            with open(split_file, "r", encoding="utf-8") as sf:
                for line_num, line in enumerate(sf, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  JSON decode error at line {line_num}: {e}")
                        continue
                    
                    file_name = data["file_name"]
                    annotations = data.get("annotations", [])
                    
                    # Extract and group annotations
                    extracted_data = extract_annotations(annotations)
                    
                    image_url = get_image_path(file_name, args.data_dir, args.pseudo_path)
                    
                    record = {
                        "image_info": [
                            {"matched_text_index": 0, "image_url": image_url}
                        ],
                        "text_info": [
                            {"text": "OCR:{}", "tag": "mask"},
                            {"text": extracted_data_to_string(extracted_data), "tag": "no_mask"}
                        ]
                    }
                    
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_records += 1
    
    print(f"Done! Output written to {args.output_file}")
    print(f"Total records: {total_records}")


if __name__ == "__main__":
    main()
