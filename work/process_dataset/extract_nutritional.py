#!/usr/bin/env python3
"""Extract information from Nutritional Data POIE-1 dataset.

Usage:
    python extract_nutritional.py --data_dir <path_to_data_dir> [options]

Examples:
    # Process all splits with pseudo paths
    python extract_nutritional.py --data_dir . --pseudo_path

    # Process only train set
    python extract_nutritional.py --data_dir . --split train --pseudo_path

Arguments:
    --data_dir       Directory containing train/test/valid folders (required)
    --split          Data split: train, test, valid (default: all splits)
    --output_file    Output JSONL file path (default: extracted_nutritional.jsonl)
    --pseudo_path    Use pseudo image paths (path/to/image.jpg)
                     If not set, real image paths will be used

Output Format:
    Each line is a JSON object with:
    - image_info: List of image references with matched_text_index and image_url
    - text_info: List of text entries with "mask" (OCR placeholder) and "no_mask" (extracted data as JSON string)
"""

import json
import os
import argparse


def extracted_data_to_string(extracted_data):
    """Convert extracted_data dict to JSON string format."""
    return json.dumps(extracted_data, ensure_ascii=False)


def extract_ground_truth(ground_truth):
    """Extract structured data from ground_truth."""
    gt_parse = ground_truth.get("gt_parse", [])
    
    extracted = {}
    
    # Handle both list and dict formats
    if isinstance(gt_parse, list):
        nutrients = gt_parse
    elif isinstance(gt_parse, dict):
        nutrients = [gt_parse]
    else:
        nutrients = []
    
    # Extract nutrient information
    for nutrient_entry in nutrients:
        if isinstance(nutrient_entry, dict):
            nutrient_name = nutrient_entry.get("nutrient", "")
            unit = nutrient_entry.get("unit", "")
            value = nutrient_entry.get("value", "")
            
            if nutrient_name:
                # Create a clean key from nutrient name
                key = nutrient_name.strip()
                if unit:
                    extracted[key] = f"{value} {unit}".strip()
                else:
                    extracted[key] = value
    
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract information from Nutritional Data POIE-1 dataset")
    parser.add_argument("--data_dir", required=True, help="Directory containing train/test/valid folders")
    parser.add_argument("--split", default=None, help="Data split (train, test, valid)")
    parser.add_argument("--output_file", default="extracted_nutritional.jsonl", help="Output JSONL file path")
    parser.add_argument("--pseudo_path", action="store_true", help="Use pseudo image paths")
    
    args = parser.parse_args()
    
    # Find splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = []
        for split_name in ["train", "test", "valid"]:
            if os.path.exists(os.path.join(args.data_dir, split_name)):
                splits.append(split_name)
    
    print(f"Processing splits: {splits}")
    
    total_records = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for split in splits:
            annotations_file = os.path.join(args.data_dir, split, "annotations.json")
            
            if not os.path.exists(annotations_file):
                print(f"Annotations file not found: {annotations_file}")
                continue
            
            with open(annotations_file, "r", encoding="utf-8") as af:
                data = json.load(af)
            
            print(f"Found {len(data)} records in {split}")
            
            for record in data:
                doc_id = record.get("id", total_records)
                image_path = record.get("image_path", "")
                ground_truth = record.get("ground_truth", {})
                
                # Extract structured data
                extracted_data = extract_ground_truth(ground_truth)
                
                # Get image path
                if args.pseudo_path:
                    img_filename = os.path.basename(image_path)
                    image_url = f"path/to/{img_filename}"
                else:
                    image_url = os.path.join(args.data_dir, split, image_path)
                
                output_record = {
                    "image_info": [
                        {"matched_text_index": 0, "image_url": image_url}
                    ],
                    "text_info": [
                        {"text": "OCR:{}", "tag": "mask"},
                        {"text": extracted_data_to_string(extracted_data), "tag": "no_mask"}
                    ]
                }
                
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                total_records += 1
    
    print(f"Done! Output written to {args.output_file}")
    print(f"Total records: {total_records}")


if __name__ == "__main__":
    main()
