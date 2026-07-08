#!/usr/bin/env python3
"""Extract information from CORD-v2 dataset.

Usage:
    python extract_cord.py --data_dir <path_to_data_dir> [options]

Examples:
    # Process all splits with pseudo paths
    python extract_cord.py --data_dir . --pseudo_path

    # Process only train set
    python extract_cord.py --data_dir . --split train --pseudo_path

Arguments:
    --data_dir       Directory containing train/test/valid folders (required)
    --split          Data split: train, test, valid (default: all splits)
    --output_file    Output JSONL file path (default: extracted_cord.jsonl)
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
    gt_parse = ground_truth.get("gt_parse", {})
    
    extracted = {}
    
    # Extract menu items
    menu_data = gt_parse.get("menu", [])
    if menu_data:
        extracted["menu"] = []
        # Handle both list and dict formats
        if isinstance(menu_data, dict):
            menu_items = [menu_data]
        else:
            menu_items = menu_data
        
        for item in menu_items:
            if isinstance(item, dict):
                menu_entry = {}
                if "nm" in item:
                    menu_entry["name"] = item["nm"]
                if "cnt" in item:
                    menu_entry["quantity"] = item["cnt"]
                if "price" in item:
                    menu_entry["price"] = item["price"]
                if "unitprice" in item:
                    menu_entry["unit_price"] = item["unitprice"]
                if "sub" in item:
                    menu_entry["sub"] = item["sub"]
                extracted["menu"].append(menu_entry)
    
    # Extract sub_total
    sub_total_data = gt_parse.get("sub_total", {})
    if sub_total_data:
        # Handle both dict and list formats
        if isinstance(sub_total_data, dict):
            extracted["sub_total"] = {}
            for key, value in sub_total_data.items():
                if value:
                    extracted["sub_total"][key] = value
        elif isinstance(sub_total_data, list):
            extracted["sub_total"] = []
            for item in sub_total_data:
                if isinstance(item, dict):
                    entry = {}
                    for key, value in item.items():
                        if value:
                            entry[key] = value
                    extracted["sub_total"].append(entry)
    
    # Extract total
    total_data = gt_parse.get("total", {})
    if total_data:
        # Handle both dict and list formats
        if isinstance(total_data, dict):
            extracted["total"] = {}
            for key, value in total_data.items():
                if value:
                    extracted["total"][key] = value
        elif isinstance(total_data, list):
            extracted["total"] = []
            for item in total_data:
                if isinstance(item, dict):
                    entry = {}
                    for key, value in item.items():
                        if value:
                            entry[key] = value
                    extracted["total"].append(entry)
    
    # Extract voided menu if exists
    voided_menu = gt_parse.get("voided_menu", [])
    if voided_menu:
        extracted["voided_menu"] = []
        for item in voided_menu:
            menu_entry = {}
            if "nm" in item:
                menu_entry["name"] = item["nm"]
            if "cnt" in item:
                menu_entry["quantity"] = item["cnt"]
            if "price" in item:
                menu_entry["price"] = item["price"]
            extracted["voided_menu"].append(menu_entry)
    
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract information from CORD-v2 dataset")
    parser.add_argument("--data_dir", required=True, help="Directory containing train/test/valid folders")
    parser.add_argument("--split", default=None, help="Data split (train, test, valid)")
    parser.add_argument("--output_file", default="extracted_cord.jsonl", help="Output JSONL file path")
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
