#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process extracted JSONL dataset and convert it to JSONL format.

This script traverses a parent directory to find files ending with
'.extracted.jsonl', reads each line, and generates output entries.
Each input JSONL file produces one output JSONL file.

For each record in the input:
- document_id maps to an image file (e.g., it_val_18 -> it.val/it_val_18.jpg)
- tag "mask" text is "OCR:{}"
- tag "no_mask" text is the extracted_data field as JSON string
"""

import os
import json
import argparse
from pathlib import Path


def process_extracted_dataset(parent_dir, output_dir=None, image_root=None):
    """
    Process extracted JSONL dataset and generate output JSONL files.

    Args:
        parent_dir (str): Parent directory containing *.*.extracted.jsonl files
        output_dir (str, optional): Output directory for generated JSONL files.
                                     If not provided, output files are placed in
                                     the same directory as input files.
        image_root (str, optional): Root directory for image URLs. If provided,
                                     it will be prepended to the relative image path.
                                     If not provided, the parent_dir is used as root.
    """
    parent_path = Path(parent_dir)

    if not parent_path.exists():
        print(f"Error: Parent directory '{parent_dir}' does not exist.")
        return

    # Set default image_root to parent_dir if not provided
    if image_root is None:
        image_root = str(parent_path)

    # Find all files matching *.extracted.jsonl pattern
    extracted_files = list(parent_path.rglob('*.extracted.jsonl'))

    if not extracted_files:
        print(f"No *.extracted.jsonl files found under '{parent_dir}'")
        return

    print(f"Found {len(extracted_files)} extracted JSONL file(s)")

    for jsonl_file in extracted_files:
        print(f"\nProcessing file: {jsonl_file.name}")

        # Determine output file path
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            output_file = out_path / jsonl_file.name.replace('.extracted.jsonl', '.jsonl')
        else:
            output_file = jsonl_file.parent / jsonl_file.name.replace('.extracted.jsonl', '.jsonl')

        # Derive the subdirectory name (e.g., "it.val") from the filename
        # Filename pattern: {lang}.{split}.extracted.jsonl -> subdir is {lang}.{split}
        # e.g., "it.val.extracted.jsonl" -> "it.val"
        subdir_name = jsonl_file.name.replace('.extracted.jsonl', '')  # e.g., "it.val"

        output_data = []
        line_count = 0

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Error decoding JSON at line {line_count + 1}: {e}")
                    line_count += 1
                    continue

                line_count += 1

                # Extract document_id and build image path
                document_id = record.get('document_id', '')
                if not document_id:
                    print(f"  Warning: Missing document_id at line {line_count}, skipping")
                    continue

                # Build image URL: subdir_name/document_id.jpg
                # e.g., "it.val/it_val_18.jpg"
                image_relative = f"{subdir_name}/{document_id}.jpg"

                if image_root:
                    image_url = os.path.join(image_root, image_relative).replace(os.sep, '/')
                else:
                    image_url = image_relative

                # Extract extracted_data field
                extracted_data = record.get('extracted_data', {})

                # Build output item
                output_item = {
                    "image_info": [
                        {
                            "matched_text_index": 0,
                            "image_url": image_url
                        }
                    ],
                    "text_info": [
                        {
                            "text": "OCR:{}",
                            "tag": "mask"
                        },
                        {
                            "text": json.dumps(extracted_data, ensure_ascii=False),
                            "tag": "no_mask"
                        }
                    ]
                }

                output_data.append(output_item)

        # Write output JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"  Processed {len(output_data)} records. Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Process extracted JSONL dataset and convert to JSONL format'
    )
    parser.add_argument(
        'parent_dir',
        help='Parent directory containing *.*.extracted.jsonl files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for generated JSONL files (default: same as input)'
    )
    parser.add_argument(
        '-r', '--image-root',
        help='Root directory for image URLs (default: parent_dir)'
    )

    args = parser.parse_args()

    process_extracted_dataset(args.parent_dir, args.output_dir, args.image_root)


if __name__ == '__main__':
    main()
