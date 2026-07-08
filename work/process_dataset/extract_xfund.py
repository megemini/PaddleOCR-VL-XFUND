#!/usr/bin/env python3
"""Extract information from xfund dataset images and annotations.

Usage:
    python extract_xfund.py --dataset_dir <path_to_dataset> [options]

Examples:
    # Process Chinese validation set with pseudo paths
    python extract_xfund.py --dataset_dir dataset --lang zh --split val --pseudo_path

    # Process all languages training set
    python extract_xfund.py --dataset_dir dataset --split train --pseudo_path

    # Process specific language with real paths
    python extract_xfund.py --dataset_dir dataset --lang zh --split val

Arguments:
    --dataset_dir     Directory containing dataset files (required)
    --lang            Language code: zh, de, es, fr, it, ja, pt (default: all languages)
    --split           Data split: train, val (default: all splits)
    --output_file     Output JSONL file path (default: extracted_xfund.jsonl)
    --pseudo_path     Use pseudo image paths (path/to/image.jpg)
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


def extract_qa_pairs(document):
    """Extract question-answer pairs from document entities using linking field.
    
    Handles:
    1. Multi-level fields: when a question links to sub-questions
    2. Multiple values: concatenate values with same key
    """
    # Build entity lookup by ID
    entities_by_id = {}
    for entity in document:
        entities_by_id[entity['id']] = entity
    
    # Collect headers
    headers = []
    for entity in document:
        if entity['label'] == 'header':
            headers.append(entity['text'])
    
    # Collect other texts
    others = []
    for entity in document:
        if entity['label'] == 'other':
            others.append(entity['text'])
    
    # Build relationships from linking
    # parent_id -> [child_id, ...] (questions linked from this question)
    parent_to_children = {}
    # entity_id -> [linked_ids...] (all linked entities)
    entity_links = {}
    
    for entity in document:
        if entity['label'] == 'question' and entity['linking']:
            q_id = entity['id']
            entity_links[q_id] = []
            for link in entity['linking']:
                first_id, second_id = link
                entity_links[q_id].append((first_id, second_id))
                # If this question is the first in the link, it's the parent
                if first_id == q_id and second_id != q_id:
                    if q_id not in parent_to_children:
                        parent_to_children[q_id] = []
                    if second_id not in parent_to_children[q_id]:
                        parent_to_children[q_id].append(second_id)
    
    # Find all child question IDs (questions that are second in a link where first is also a question)
    child_question_ids = set()
    for q_id, children in parent_to_children.items():
        for child_id in children:
            if child_id in entities_by_id and entities_by_id[child_id]['label'] == 'question':
                child_question_ids.add(child_id)
    
    # Build result
    qa_pairs = {}
    
    for entity in document:
        if entity['label'] != 'question':
            continue
        
        q_id = entity['id']
        q_text = entity['text']
        
        # Skip if this is a child question (will be handled by parent)
        if q_id in child_question_ids:
            continue
        
        # Get all linked IDs from this question
        linked_ids = []
        if q_id in entity_links:
            for first_id, second_id in entity_links[q_id]:
                if first_id == q_id:
                    linked_ids.append(second_id)
        
        # Separate into answer IDs and child question IDs
        answer_ids = []
        child_q_ids = []
        for linked_id in linked_ids:
            if linked_id in entities_by_id:
                linked_entity = entities_by_id[linked_id]
                if linked_entity['label'] == 'answer':
                    answer_ids.append(linked_id)
                elif linked_entity['label'] == 'question':
                    child_q_ids.append(linked_id)
        
        # Get answer texts
        answers = []
        for a_id in answer_ids:
            if a_id in entities_by_id:
                answers.append(entities_by_id[a_id]['text'])
        
        if child_q_ids:
            # Has child questions -> create nested structure
            nested = {}
            for child_q_id in child_q_ids:
                if child_q_id in entities_by_id:
                    child_q_text = entities_by_id[child_q_id]['text']
                    # Get answers for this child question
                    child_answer_ids = []
                    if child_q_id in entity_links:
                        for first_id, second_id in entity_links[child_q_id]:
                            if first_id == child_q_id:
                                child_answer_ids.append(second_id)
                    
                    child_answers = []
                    for a_id in child_answer_ids:
                        if a_id in entities_by_id and entities_by_id[a_id]['label'] == 'answer':
                            child_answers.append(entities_by_id[a_id]['text'])
                    
                    # Concatenate multiple answers
                    if len(child_answers) == 1:
                        nested[child_q_text] = child_answers[0]
                    elif len(child_answers) > 1:
                        nested[child_q_text] = "".join(child_answers)
                    else:
                        nested[child_q_text] = ""
            
            # Also add direct answers if any
            if answers:
                nested["_value"] = "".join(answers)
            
            qa_pairs[q_text] = nested
        elif answers:
            # No child questions, just answers -> concatenate
            qa_pairs[q_text] = "".join(answers)
        else:
            qa_pairs[q_text] = ""
    
    return {
        'header': headers,
        'other': others,
        'question': qa_pairs
    }


# Language to header translation mapping
HEADER_TRANSLATIONS = {
    'zh': '标题',
    'de': 'Kopfzeile',
    'es': 'Encabezado',
    'fr': 'En-tête',
    'it': 'Intestazione',
    'ja': 'ヘッダー',
    'pt': 'Cabeçalho'
}


def extracted_data_to_string(extracted_data):
    """Convert extracted_data dict to JSON string format."""
    return json.dumps(extracted_data, ensure_ascii=False)


def get_image_path(doc_id, dataset_dir, lang, split, use_pseudo_path):
    """Get image path based on configuration."""
    image_filename = doc_id + ".jpg"
    
    if use_pseudo_path:
        return f"path/to/{image_filename}"
    else:
        return os.path.join(dataset_dir, f"{lang}.{split}", image_filename)


def main():
    parser = argparse.ArgumentParser(description="Extract information from xfund dataset")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing dataset files")
    parser.add_argument("--lang", default=None, help="Language code (zh, de, es, fr, it, ja, pt)")
    parser.add_argument("--split", default=None, help="Data split (train, val)")
    parser.add_argument("--output_file", default="extracted_xfund.jsonl", help="Output JSONL file path")
    parser.add_argument("--pseudo_path", action="store_true", help="Use pseudo image paths")
    
    args = parser.parse_args()
    
    # Find all JSON files
    if args.lang and args.split:
        json_files = [os.path.join(args.dataset_dir, f"{args.lang}.{args.split}.json")]
    elif args.lang:
        json_files = glob.glob(os.path.join(args.dataset_dir, f"{args.lang}.*.json"))
    elif args.split:
        json_files = glob.glob(os.path.join(args.dataset_dir, f"*.{args.split}.json"))
    else:
        json_files = glob.glob(os.path.join(args.dataset_dir, "*.json"))
    
    json_files = sorted(json_files)
    print(f"Found {len(json_files)} JSON files")
    
    total_records = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for json_file in json_files:
            # Parse filename to get lang and split
            basename = os.path.basename(json_file)
            parts = basename.replace(".json", "").split(".")
            lang = parts[0]
            split = parts[1]
            
            print(f"Processing {basename}...")
            
            with open(json_file, "r", encoding="utf-8") as af:
                data = json.load(af)
            
            for doc in data['documents']:
                doc_id = doc['id']
                extracted = extract_qa_pairs(doc['document'])
                
                # Build extracted_data - flatten to top level
                extracted_data = {}
                if extracted['header']:
                    # Add translated header field
                    header_key = HEADER_TRANSLATIONS.get(lang, 'header')
                    extracted_data[header_key] = extracted['header']
                    extracted_data['header'] = extracted['header']
                if extracted['other']:
                    extracted_data['other'] = extracted['other']
                if extracted['question']:
                    # Merge question pairs directly into top level
                    for k, v in extracted['question'].items():
                        extracted_data[k] = v
                
                image_url = get_image_path(doc_id, args.dataset_dir, lang, split, args.pseudo_path)
                
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
