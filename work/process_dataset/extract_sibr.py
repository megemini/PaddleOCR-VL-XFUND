#!/usr/bin/env python3
"""Extract information from SIBR dataset.

Usage:
    python extract_sibr.py --data_dir <path_to_data_dir> [options]

Examples:
    # Process both train and test with pseudo paths
    python extract_sibr.py --data_dir . --pseudo_path

    # Process only train set
    python extract_sibr.py --data_dir . --split train --pseudo_path

Arguments:
    --data_dir       Directory containing label/, images/, train.txt, test.txt (required)
    --split          Data split: train, test (default: all splits)
    --output_file    Output JSONL file path (default: extracted_sibr.jsonl)
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
import re


# Language detection from filename prefix and header translation
HEADER_TRANSLATIONS = {
    'eng': 'header',
    'hot': 'header',
    'med': 'header'
}


def extracted_data_to_string(extracted_data):
    """Convert extracted_data dict to JSON string format."""
    return json.dumps(extracted_data, ensure_ascii=False)


def detect_language(label_filename):
    """Detect language from label filename prefix."""
    match = re.match(r'^([a-z]+)-', label_filename)
    if match:
        return match.group(1)
    return 'eng'


def contains_chinese(text):
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def extract_entities(label_data):
    """Extract structured data from label entities using linking field."""
    # Handle different data structures
    if isinstance(label_data, list):
        entities = label_data
    elif isinstance(label_data, dict):
        entities = label_data.get("form", label_data.get("entities", []))
    else:
        entities = []
    
    # Build entity lookup by ID
    entities_by_id = {}
    for entity in entities:
        entities_by_id[entity['id']] = entity
    
    # Collect headers
    headers = []
    for entity in entities:
        if entity.get('label') == 'header':
            headers.append(entity['text'])
    
    # Build QA pairs from linking
    qa_pairs = {}
    processed_questions = set()
    
    for entity in entities:
        if entity.get('label') == 'question':
            q_id = entity['id']
            q_text = entity['text']
            
            # Skip if already processed
            if q_id in processed_questions:
                continue
            
            # Find linked answers
            answers = []
            for link in entity.get('linking', []):
                link_type = link.get('type', '')
                link_ids = link.get('id', [])
                
                if link_type == 'inter' and len(link_ids) == 2:
                    linked_q_id, linked_a_id = link_ids
                    if linked_q_id == q_id and linked_a_id in entities_by_id:
                        linked_entity = entities_by_id[linked_a_id]
                        if linked_entity.get('label') == 'answer':
                            answers.append(linked_entity['text'])
            
            # Store QA pair
            if answers:
                if len(answers) == 1:
                    qa_pairs[q_text] = answers[0]
                else:
                    qa_pairs[q_text] = answers
            else:
                qa_pairs[q_text] = ""
            
            processed_questions.add(q_id)
    
    return {
        'header': headers,
        'question': qa_pairs
    }


def main():
    parser = argparse.ArgumentParser(description="Extract information from SIBR dataset")
    parser.add_argument("--data_dir", required=True, help="Directory containing label/, images/, train.txt, test.txt")
    parser.add_argument("--split", default=None, help="Data split (train, test)")
    parser.add_argument("--output_file", default="extracted_sibr.jsonl", help="Output JSONL file path")
    parser.add_argument("--pseudo_path", action="store_true", help="Use pseudo image paths")
    
    args = parser.parse_args()
    
    # Find splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = []
        for split_name in ["train", "test"]:
            if os.path.exists(os.path.join(args.data_dir, f"{split_name}.txt")):
                splits.append(split_name)
    
    print(f"Processing splits: {splits}")
    
    total_records = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for split in splits:
            split_file = os.path.join(args.data_dir, f"{split}.txt")
            
            if not os.path.exists(split_file):
                print(f"Split file not found: {split_file}")
                continue
            
            with open(split_file, "r", encoding="utf-8") as sf:
                label_files = [line.strip() for line in sf if line.strip()]
            
            print(f"Found {len(label_files)} records in {split}")
            
            for label_path in label_files:
                # Get full path to label file
                full_label_path = os.path.join(args.data_dir, label_path)
                
                if not os.path.exists(full_label_path):
                    continue
                
                with open(full_label_path, "r", encoding="utf-8") as lf:
                    label_data = json.load(lf)
                
                # Extract structured data
                extracted = extract_entities(label_data)

                # Get image path from label path
                # label/eng-0000.json -> images/eng-0000.jpg
                label_filename = os.path.basename(label_path)
                img_filename = label_filename.replace('.json', '.jpg')

                # Detect language and build extracted_data
                lang = detect_language(label_filename)
                extracted_data = {}

                # Add translated header
                if extracted['header']:
                    header_key = HEADER_TRANSLATIONS.get(lang, 'header')
                    extracted_data[header_key] = extracted['header']
                    extracted_data['header'] = extracted['header']

                # Flatten question pairs to top level
                if extracted['question']:
                    for k, v in extracted['question'].items():
                        extracted_data[k] = v

                # Check if Chinese characters exist and add 标题 field
                has_chinese = False
                for k, v in extracted_data.items():
                    if contains_chinese(k) or contains_chinese(v):
                        has_chinese = True
                        break

                if has_chinese and 'header' in extracted_data:
                    extracted_data['标题'] = extracted_data['header']
                
                if args.pseudo_path:
                    image_url = f"path/to/{img_filename}"
                else:
                    image_url = os.path.join(args.data_dir, "images", img_filename)
                
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
