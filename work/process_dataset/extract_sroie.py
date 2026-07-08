#!/usr/bin/env python3
"""Extract information from SROIE 2019 dataset.

Usage:
    python extract_sroie.py --data_dir <path_to_data_dir> [options]

Examples:
    # Process both train and test with pseudo paths
    python extract_sroie.py --data_dir . --pseudo_path

    # Process only train set
    python extract_sroie.py --data_dir . --split train --pseudo_path

    # Process with real image paths
    python extract_sroie.py --data_dir .

Arguments:
    --data_dir       Directory containing train/test folders (required)
    --split          Data split: train, test (default: all splits)
    --output_file    Output JSONL file path (default: extracted_sroie.jsonl)
    --pseudo_path    Use pseudo image paths (path/to/image.jpg)
                     If not set, real image paths will be used

Output Format:
    Each line is a JSON object with:
    - image_info: List of image references with matched_text_index and image_url
    - text_info: List of text entries with "mask" (OCR placeholder) and "no_mask" (extracted data as JSON string)
"""

import json
import os
import glob
import argparse


def extracted_data_to_string(extracted_data):
    """Convert extracted_data dict to JSON string format."""
    return json.dumps(extracted_data, ensure_ascii=False)


def get_image_path(doc_id, data_dir, split, use_pseudo_path):
    """Get image path based on configuration."""
    image_filename = doc_id + ".jpg"
    
    if use_pseudo_path:
        return f"path/to/{image_filename}"
    else:
        return os.path.join(data_dir, split, "img", image_filename)


def main():
    parser = argparse.ArgumentParser(description="Extract information from SROIE 2019 dataset")
    parser.add_argument("--data_dir", required=True, help="Directory containing train/test folders")
    parser.add_argument("--split", default=None, help="Data split (train, test)")
    parser.add_argument("--output_file", default="extracted_sroie.jsonl", help="Output JSONL file path")
    parser.add_argument("--pseudo_path", action="store_true", help="Use pseudo image paths")
    
    args = parser.parse_args()
    
    # Find splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = []
        if os.path.exists(os.path.join(args.data_dir, "train")):
            splits.append("train")
        if os.path.exists(os.path.join(args.data_dir, "test")):
            splits.append("test")
    
    print(f"Processing splits: {splits}")
    
    total_records = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for split in splits:
            entities_dir = os.path.join(args.data_dir, split, "entities")
            
            if not os.path.exists(entities_dir):
                print(f"Entities directory not found: {entities_dir}")
                continue
            
            entity_files = sorted(glob.glob(os.path.join(entities_dir, "*.txt")))
            print(f"Found {len(entity_files)} entity files in {split}")
            
            for entity_file in entity_files:
                doc_id = os.path.basename(entity_file).replace(".txt", "")
                
                with open(entity_file, "r", encoding="utf-8") as ef:
                    extracted_data = json.load(ef)
                
                image_url = get_image_path(doc_id, args.data_dir, split, args.pseudo_path)
                
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
