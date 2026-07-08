#!/usr/bin/env python3
"""Extract information from invoice images and annotations.

Usage:
    python extract_fatura.py --images_dir <path_to_images> [options]

Examples:
    # Use pseudo paths (path/to/image.jpg)
    python extract_fatura.py --images_dir invoices_dataset_final/images --pseudo_path

    # Use real image paths
    python extract_fatura.py --images_dir invoices_dataset_final/images

    # Custom annotation directory and output file
    python extract_fatura.py \
        --images_dir invoices_dataset_final/images \
        --annotations_dir invoices_dataset_final/Annotations/Original_Format \
        --output_file my_output.jsonl \
        --pseudo_path

Arguments:
    --images_dir       Directory containing invoice images (required)
    --annotations_dir  Directory containing annotation JSON files
                       (default: invoices_dataset_final/Annotations/Original_Format)
    --output_file      Output JSONL file path
                       (default: extracted_invoices.jsonl)
    --pseudo_path      Use pseudo image paths (path/to/image.jpg)
                       If not set, real image paths will be used

Output Format:
    Each line is a JSON object with:
    - image_info: List of image references with matched_text_index and image_url
    - text_info: List of text entries with "mask" (OCR placeholder) and "no_mask" (extracted data as JSON string)
"""

import json
import os
import glob
import re
import argparse


def split_key_value(text):
    """Split text into (key, value) if it has clear 'key: value' format."""
    match = re.match(r'^([^:]+?)\s*:\s*(.+)$', text, re.DOTALL)
    if match:
        key = match.group(1).strip()
        value = match.group(2).strip()
        if len(key) < 50 and '\n' not in key:
            return key, value
    return None, text


def parse_buyer_info(text):
    """Parse buyer info into structured fields."""
    result = {}
    lines = text.strip().split("\n")
    first_line = lines[0].strip()

    found_prefix = False
    for prefix in ["Bill to:", "BILL_TO:", "Bill To:", "Buyer :", "Buyer:"]:
        if first_line.startswith(prefix):
            result["Name"] = first_line[len(prefix):].strip()
            lines = lines[1:]
            found_prefix = True
            break

    if not found_prefix:
        if first_line in ["Bill to", "BILL_TO", "Buyer"]:
            if len(lines) > 1:
                result["Name"] = lines[1].strip()
                lines = lines[2:]
        else:
            key, value = split_key_value(first_line)
            if key and key.lower() in ["buyer", "bill to", "bill_to"]:
                result["Name"] = value
                lines = lines[1:]
            else:
                result["Name"] = first_line

    address_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("Tel:"):
            result["Phone"] = line[4:].strip()
        elif line.startswith("Email:"):
            result["Email"] = line[6:].strip()
        elif line.startswith("Site:"):
            result["Website"] = line[5:].strip()
        elif line:
            address_lines.append(line)

    if address_lines:
        result["Address"] = "\n".join(address_lines)

    return result


def extract_fields(annotation):
    """Extract fields from annotation data, preserving original keys."""
    extracted = {}

    # Title
    if "TITLE" in annotation:
        extracted["Title"] = annotation["TITLE"].get("text", "")

    # Date - preserve original key
    if "DATE" in annotation:
        date_text = annotation["DATE"].get("text", "")
        key, value = split_key_value(date_text)
        if key:
            extracted[key] = value
        else:
            extracted["Date"] = date_text

    # Due Date - preserve original key
    if "DUE_DATE" in annotation:
        due_text = annotation["DUE_DATE"].get("text", "")
        key, value = split_key_value(due_text)
        if key:
            extracted[key] = value
        else:
            extracted["Due Date"] = due_text

    # Invoice Number - preserve original key
    if "NUMBER" in annotation:
        num_text = annotation["NUMBER"].get("text", "")
        match = re.match(r'^([A-Za-z\s#]+)\s+(.+)$', num_text)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            extracted[key] = value
        else:
            extracted["Invoice Number"] = num_text

    # PO Number - preserve original key
    if "PO_NUMBER" in annotation:
        po_text = annotation["PO_NUMBER"].get("text", "")
        key, value = split_key_value(po_text)
        if key:
            extracted[key] = value
        else:
            extracted["PO Number"] = po_text

    # Seller info
    if "SELLER_NAME" in annotation:
        extracted["Seller Name"] = annotation["SELLER_NAME"].get("text", "")

    # Seller Address - preserve original key
    if "SELLER_ADDRESS" in annotation:
        addr_text = annotation["SELLER_ADDRESS"].get("text", "")
        key, value = split_key_value(addr_text)
        if key:
            extracted[key] = value
        else:
            extracted["Seller Address"] = addr_text

    # Seller Email - preserve original key
    if "SELLER_EMAIL" in annotation:
        email_text = annotation["SELLER_EMAIL"].get("text", "")
        key, value = split_key_value(email_text)
        if key:
            extracted[key] = value
        else:
            extracted["Seller Email"] = email_text

    # Seller Site
    if "SELLER_SITE" in annotation:
        extracted["Seller Website"] = annotation["SELLER_SITE"].get("text", "")

    # Buyer info - flatten to top-level fields
    buyer_key = "BUYER" if "BUYER" in annotation else "BILL_TO"
    if buyer_key in annotation:
        buyer_text = annotation[buyer_key].get("text", "")
        buyer_info = parse_buyer_info(buyer_text)
        for k, v in buyer_info.items():
            extracted["Buyer " + k] = v

    # GSTIN
    if "GSTIN_BUYER" in annotation:
        gstin_text = annotation["GSTIN_BUYER"].get("text", "")
        gstin_text = gstin_text.lstrip("(")
        key, value = split_key_value(gstin_text)
        if key and "GSTIN" in key.upper():
            extracted["Buyer GSTIN"] = value
        elif key:
            extracted["Buyer " + key] = value
        else:
            extracted["Buyer GSTIN"] = gstin_text

    if "GSTIN_SELLER" in annotation:
        gstin_text = annotation["GSTIN_SELLER"].get("text", "")
        gstin_text = gstin_text.lstrip("(")
        key, value = split_key_value(gstin_text)
        if key and "GSTIN" in key.upper():
            extracted["Seller GSTIN"] = value
        elif key:
            extracted["Seller " + key] = value
        else:
            extracted["Seller GSTIN"] = gstin_text

    # SUB_TOTAL - preserve original key
    if "SUB_TOTAL" in annotation:
        sub_text = annotation["SUB_TOTAL"].get("text", "")
        key, value = split_key_value(sub_text)
        if key:
            extracted[key] = value
        else:
            extracted["SUB_TOTAL"] = sub_text

    # DISCOUNT - strip "DISCOUNT" prefix from value
    if "DISCOUNT" in annotation:
        discount_text = annotation["DISCOUNT"].get("text", "")
        if discount_text.startswith("DISCOUNT"):
            discount_text = discount_text[8:].strip()
        extracted["DISCOUNT"] = discount_text

    # TAX - strip "TAX:" prefix from value
    if "TAX" in annotation:
        tax_text = annotation["TAX"].get("text", "")
        if tax_text.startswith("TAX:"):
            tax_text = tax_text[4:].strip()
        elif tax_text.startswith("TAX :"):
            tax_text = tax_text[5:].strip()
        extracted["TAX"] = tax_text

    # TOTAL - preserve original key
    if "TOTAL" in annotation:
        total_text = annotation["TOTAL"].get("text", "")
        key, value = split_key_value(total_text)
        if key:
            extracted[key] = value
        else:
            extracted["TOTAL"] = total_text

    # TOTAL_WORDS - preserve original key
    if "TOTAL_WORDS" in annotation:
        words_text = annotation["TOTAL_WORDS"].get("text", "")
        key, value = split_key_value(words_text)
        if key:
            extracted[key] = value
        else:
            extracted["Total in words"] = words_text

    # Payment details - keep as text
    if "PAYMENT_DETAILS" in annotation:
        payment_text = annotation["PAYMENT_DETAILS"].get("text", "")
        extracted["Payment Details"] = payment_text

    # Note - preserve original key
    if "NOTE" in annotation:
        note_text = annotation["NOTE"].get("text", "")
        key, value = split_key_value(note_text)
        if key:
            extracted[key] = value
        else:
            extracted["Note"] = note_text

    # Conditions
    if "CONDITIONS" in annotation:
        extracted["Terms and Conditions"] = annotation["CONDITIONS"].get("text", "")

    return extracted


def extracted_data_to_string(extracted_data):
    """Convert extracted_data dict to JSON string format."""
    return json.dumps(extracted_data, ensure_ascii=False)


def get_image_path(doc_id, images_dir, use_pseudo_path):
    """Get image path based on configuration."""
    # doc_id is like "Template10_Instance0"
    image_filename = doc_id + ".jpg"
    
    if use_pseudo_path:
        return f"path/to/{image_filename}"
    else:
        return os.path.join(images_dir, image_filename)


def main():
    parser = argparse.ArgumentParser(description="Extract information from invoice images")
    parser.add_argument("--images_dir", required=True, help="Directory containing invoice images")
    parser.add_argument("--annotations_dir", default="invoices_dataset_final/Annotations/Original_Format",
                        help="Directory containing annotation JSON files")
    parser.add_argument("--output_file", default="extracted_invoices.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--pseudo_path", action="store_true",
                        help="Use pseudo image paths (path/to/image.jpg)")
    
    args = parser.parse_args()
    
    annotation_files = sorted(glob.glob(os.path.join(args.annotations_dir, "*.json")))
    print(f"Found {len(annotation_files)} annotation files")
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        for i, filepath in enumerate(annotation_files):
            filename = os.path.basename(filepath)
            doc_id = filename.replace(".json", "")
            
            with open(filepath, "r", encoding="utf-8") as af:
                annotation = json.load(af)
            
            extracted_data = extract_fields(annotation)
            image_url = get_image_path(doc_id, args.images_dir, args.pseudo_path)
            
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
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(annotation_files)} files")
    
    print(f"Done! Output written to {args.output_file}")
    print(f"Total records: {len(annotation_files)}")


if __name__ == "__main__":
    main()
