#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge multiple JSONL files into a single JSONL file.

When the --extended option is used, each record generates additional entries
by randomly selecting a subset of fields from the no_mask text, similar to
process_ner_dataset.py's random field selection behavior.

- First entry: original record (mask text "OCR:{}", no_mask text with all fields)
- Subsequent entries: randomly selected fields with type-aware mask placeholders
"""

import os
import json
import argparse
import random
from pathlib import Path


def _randomly_select_fields(data_field):
    """
    Randomly select fields from the data.

    Args:
        data_field (dict): Data field to select from

    Returns:
        list: List of selected top-level keys
    """
    all_keys = list(data_field.keys())
    num_fields = random.randint(1, len(all_keys))
    selected_keys = random.sample(all_keys, num_fields)
    return selected_keys


def _generate_mask_text(selected_keys, data_field=None):
    """
    Generate mask text from selected keys in JSON format.

    Uses different placeholders based on the value type:
    - Single values (str, int, float, bool, etc.) -> ""
    - list -> []
    - dict (nested JSON) -> {}

    Args:
        selected_keys (list): List of selected top-level keys
        data_field (dict, optional): Original data field to inspect value types

    Returns:
        str: Mask text in JSON format
    """
    mask_dict = {}

    for key in selected_keys:
        if data_field and key in data_field:
            value = data_field[key]
            if isinstance(value, dict):
                mask_dict[key] = {}
            elif isinstance(value, list):
                mask_dict[key] = []
            else:
                mask_dict[key] = ""
        else:
            mask_dict[key] = ""

    json_str = json.dumps(mask_dict, ensure_ascii=False)
    return f"OCR:{json_str}"


def _generate_no_mask_text(selected_keys, data_field):
    """
    Generate no-mask text from selected keys and data in JSON format.

    Args:
        selected_keys (list): List of selected top-level keys
        data_field (dict): Original data field

    Returns:
        str: No-mask text (JSON string with actual values)
    """
    result_dict = {}
    for key in selected_keys:
        if key in data_field:
            result_dict[key] = data_field[key]
    return json.dumps(result_dict, ensure_ascii=False)


def merge_jsonl_files(input_dir, output_file, extended=0, split=None):
    """
    Merge multiple JSONL files into a single JSONL file.

    Args:
        input_dir (str): Directory containing JSONL files to merge
        output_file (str): Output JSONL file path
        extended (int): Number of additional entries to generate per record
                        using random field selection. 0 means no extension.
        split (str, optional): Which split to include. 'train' for *.train.jsonl,
                               'val' for *.val.jsonl, None for all files.
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Find JSONL files based on split filter
    if split == 'train':
        jsonl_files = sorted(input_path.glob('*.train.jsonl'))
    elif split == 'val':
        jsonl_files = sorted(input_path.glob('*.val.jsonl'))
    else:
        jsonl_files = sorted(input_path.glob('*.jsonl'))

    if not jsonl_files:
        print(f"No .jsonl files found in '{input_dir}'")
        return

    print(f"Found {len(jsonl_files)} JSONL file(s) to merge")

    total_records = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for jsonl_file in jsonl_files:
            print(f"  Processing: {jsonl_file.name}")
            file_records = 0

            with open(jsonl_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"    Error decoding JSON: {e}")
                        continue

                    # Write the original record as-is (first entry)
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    file_records += 1

                    # Generate extended entries if requested
                    if extended > 0:
                        image_url = record['image_info'][0]['image_url']

                        # Parse the no_mask text to get the data field
                        no_mask_text = None
                        for text_item in record['text_info']:
                            if text_item.get('tag') == 'no_mask':
                                no_mask_text = text_item['text']
                                break

                        if no_mask_text:
                            try:
                                data_field = json.loads(no_mask_text)
                            except json.JSONDecodeError:
                                print(f"    Warning: Could not parse no_mask text, skipping extension")
                                continue

                            if not isinstance(data_field, dict) or not data_field:
                                continue

                            for _ in range(extended):
                                selected_fields = _randomly_select_fields(data_field)
                                mask_text = _generate_mask_text(selected_fields, data_field)
                                no_mask_text_ext = _generate_no_mask_text(selected_fields, data_field)

                                ext_item = {
                                    "image_info": [
                                        {
                                            "matched_text_index": 0,
                                            "image_url": image_url
                                        }
                                    ],
                                    "text_info": [
                                        {
                                            "text": mask_text,
                                            "tag": "mask"
                                        },
                                        {
                                            "text": no_mask_text_ext,
                                            "tag": "no_mask"
                                        }
                                    ]
                                }

                                out_f.write(json.dumps(ext_item, ensure_ascii=False) + '\n')
                                file_records += 1

            total_records += file_records
            print(f"    {file_records} records written")

    print(f"\nTotal: {total_records} records written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple JSONL files into a single JSONL file'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing JSONL files to merge'
    )
    parser.add_argument(
        '-o', '--output',
        default='merged.jsonl',
        help='Output JSONL file path (default: merged.jsonl)'
    )
    parser.add_argument(
        '-e', '--extended',
        type=int,
        default=0,
        help='Number of additional entries per record via random field selection (default: 0)'
    )
    parser.add_argument(
        '-s', '--split',
        choices=['train', 'val'],
        help='Which split to include: train (*.train.jsonl) or val (*.val.jsonl). Default: all'
    )

    args = parser.parse_args()

    merge_jsonl_files(args.input_dir, args.output, args.extended, args.split)


if __name__ == '__main__':
    main()
