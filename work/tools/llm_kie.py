# -*- coding: utf-8 -*-
"""
LLM-based Key Information Extraction (KIE) Script
Usage: 
  Single document: python llm_kie.py <document_id> [dataset_name]
  Batch processing: python llm_kie.py --batch <dataset_name>
Example: 
  python llm_kie.py zh_val_0 zh.val
  python llm_kie.py --batch zh.val
"""

import os
import sys
import json
import base64
import time
import io
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from json_repair import repair_json
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")
MODEL = os.getenv("MODEL", "ernie-4.5-turbo-vl")

# Dataset paths
DATASET_ROOT = Path("/home/shun/workspace/Projects/megemini/erniekit_paddleocr_vl_kie/dataset/xfund")


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def compress_image(image_path: str, quality: int = 85) -> str:
    """Compress image by reducing JPEG quality.
    
    Args:
        image_path: Path to the image file
        quality: JPEG quality (1-100)
    
    Returns:
        str: Base64 encoded compressed image
    """
    img = Image.open(image_path)
    buffer = io.BytesIO()
    
    # Convert to RGB if necessary (remove alpha channel)
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    
    # Save as baseline JPEG (not progressive)
    img.save(buffer, format='JPEG', quality=quality, progressive=False)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode("utf-8")


def resize_image(image_path: str, max_size: int = 2000) -> str:
    """Resize image while maintaining aspect ratio.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height)
    
    Returns:
        str: Base64 encoded resized image
    """
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    
    # Calculate new size maintaining aspect ratio
    width, height = img.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85, progressive=False)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode("utf-8")


def get_image_base64_with_fallback(image_path: str) -> tuple:
    """Get image base64 with fallback strategies for large or problematic images.
    
    Returns:
        tuple: (base64_string, strategy_used)
    """
    # Try original image first
    original_base64 = image_to_base64(image_path)
    original_size = len(original_base64)
    
    # Check if image is progressive JPEG or too large
    img = Image.open(image_path)
    is_progressive = img.info.get('progressive', 0) or img.info.get('progression', 0)
    
    # If size is acceptable and not progressive, return original
    if original_size < 2 * 1024 * 1024 and not is_progressive:
        return original_base64, "original"
    
    if is_progressive:
        print(f"  Progressive JPEG detected, converting to baseline...")
    elif original_size >= 2 * 1024 * 1024:
        print(f"  Image too large ({original_size / 1024 / 1024:.2f} MB), trying fallback strategies...")
    
    # Strategy 1: Convert to baseline JPEG and compress slightly
    try:
        compressed = compress_image(image_path, quality=85)
        if len(compressed) < 2 * 1024 * 1024:
            print(f"  ✓ Strategy 1: Baseline JPEG compression ({len(compressed) / 1024 / 1024:.2f} MB)")
            return compressed, "baseline_compressed"
    except Exception as e:
        print(f"  ✗ Strategy 1 failed: {e}")
    
    # Strategy 2: More aggressive compression
    try:
        compressed = compress_image(image_path, quality=70)
        if len(compressed) < 2 * 1024 * 1024:
            print(f"  ✓ Strategy 2: Aggressive compression ({len(compressed) / 1024 / 1024:.2f} MB)")
            return compressed, "aggressive_compression"
    except Exception as e:
        print(f"  ✗ Strategy 2 failed: {e}")
    
    # Strategy 3: Resize to max 2000px
    try:
        resized = resize_image(image_path, max_size=2000)
        if len(resized) < 2 * 1024 * 1024:
            print(f"  ✓ Strategy 3: Resize to 2000px ({len(resized) / 1024 / 1024:.2f} MB)")
            return resized, "resize_2000"
    except Exception as e:
        print(f"  ✗ Strategy 3 failed: {e}")
    
    # Strategy 4: Resize to max 1500px with compression
    try:
        img = Image.open(image_path)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        width, height = img.size
        max_size = 1500
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70, progressive=False)
        buffer.seek(0)
        final = base64.b64encode(buffer.read()).decode("utf-8")
        
        print(f"  ✓ Strategy 4: Resize to 1500px + compression ({len(final) / 1024 / 1024:.2f} MB)")
        return final, "resize_1500_compressed"
    except Exception as e:
        print(f"  ✗ Strategy 4 failed: {e}")
    
    # If all strategies fail, return the most compressed version
    print(f"  ⚠ All strategies failed, using best effort (resize 1500px)")
    return resize_image(image_path, max_size=1500), "fallback"


def load_key_value_pairs(document_id: str, kv_json_path: Path) -> list:
    """Load key-value pairs for a specific document from JSON file."""
    with open(kv_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Find the document with matching ID
    for doc in data["documents"]:
        if doc["id"] == document_id:
            return doc["key_value_pairs"]
    
    raise ValueError(f"Document ID '{document_id}' not found in {kv_json_path}")


def format_kv_pairs_for_prompt(kv_pairs: list) -> str:
    """Format key-value pairs as a readable string for the prompt."""
    formatted = []
    for pair in kv_pairs:
        for key, value in pair.items():
            formatted.append(f"{key} {value}")
    return "\n".join(formatted)


def extract_key_info_with_retry(document_id: str, dataset_name: str = "zh.val", max_retries: int = 5, retry_delay: int = 2):
    """Extract key information from document using LLM with retry mechanism.
    
    Args:
        document_id: Document ID (e.g., zh_val_0)
        dataset_name: Dataset name (e.g., zh.val, en.val)
        max_retries: Maximum number of retries on failure
        retry_delay: Base delay between retries in seconds (exponential backoff)
    
    Returns:
        dict: Extracted data or None if failed
    """
    # Validate API key
    if not API_KEY:
        raise ValueError("API_KEY not found in .env file. Please set API_KEY in your .env file.")
    
    # Construct paths based on dataset name
    image_dir = DATASET_ROOT / dataset_name
    kv_json_path = DATASET_ROOT / f"{dataset_name}.kv.json"
    
    # Construct image path
    image_path = image_dir / f"{document_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check if KV JSON exists
    if not kv_json_path.exists():
        raise FileNotFoundError(f"KV JSON file not found: {kv_json_path}")
    
    # Load key-value pairs
    kv_pairs = load_key_value_pairs(document_id, kv_json_path)
    kv_text = format_kv_pairs_for_prompt(kv_pairs)
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    # Construct prompt
    prompt = f"""请从以上图片中抽取键值对，注意，不要有图片中不存在的信息，层级关系尽量保持简单。

参考的键值对信息如下：
{kv_text}

请根据图片内容验证并提取准确的键值对信息。
注意：
1. 只返回json格式的数据
2. 不要遗漏参考的键值对的`键`，可以进行有效的值的合并操作
"""

    # Try with original image first
    image_base64 = image_to_base64(str(image_path))
    fallback_used = False
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            # Call LLM API (non-streaming for better performance)
            chat_completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                stream=False,  # Disable streaming for better performance
                extra_body={
                    "penalty_score": 1,
                    "enable_thinking": False
                },
                max_completion_tokens=12000,
                temperature=0.,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            # Get response content
            response_content = chat_completion.choices[0].message.content
            
            # Try to repair and validate JSON
            try:
                repaired_json = repair_json(response_content)
                extracted_data = json.loads(repaired_json)
                
                return {
                    "document_id": document_id,
                    "extracted_data": extracted_data,
                    "raw_response": response_content,
                    "status": "success",
                    "fallback_used": fallback_used
                }
            except Exception as e:
                return {
                    "document_id": document_id,
                    "raw_response": response_content,
                    "error": f"JSON validation failed: {e}",
                    "status": "json_error",
                    "fallback_used": fallback_used
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"  ⚠ Attempt {attempt + 1}/{max_retries} failed for {document_id}: {error_msg}")
            
            # Check if it's an image format/size error
            is_image_error = "400" in error_msg and ("image" in error_msg.lower() or "format" in error_msg.lower())
            
            # If image error and haven't used fallback yet, try fallback strategies
            if is_image_error and not fallback_used:
                print(f"  Image error detected, trying fallback strategies...")
                image_base64, strategy = get_image_base64_with_fallback(str(image_path))
                fallback_used = True
                print(f"  Using fallback strategy: {strategy}")
                continue
            
            # Check if it's a rate limit error (429)
            if "429" in error_msg or "rate" in error_msg.lower():
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"  Rate limit detected. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
    
    # All retries failed
    return {
        "document_id": document_id,
        "error": f"Failed after {max_retries} attempts",
        "status": "failed",
        "fallback_used": fallback_used
    }


def extract_key_info(document_id: str, dataset_name: str = "zh.val"):
    """Extract key information from document using LLM (single document mode).
    
    Args:
        document_id: Document ID (e.g., zh_val_0)
        dataset_name: Dataset name (e.g., zh.val, en.val)
    """
    result = extract_key_info_with_retry(document_id, dataset_name)
    
    print(f"\n{'='*60}")
    print(f"Document ID: {document_id}")
    print(f"{'='*60}\n")
    
    if result["status"] == "success":
        print("Raw LLM Response:")
        print(result["raw_response"])
        print("\n")
        print("Validated JSON Data:")
        print(json.dumps(result["extracted_data"], ensure_ascii=False, indent=2))
        print("\n")
        
        # Save to JSON file
        output_path = DATASET_ROOT / f"{document_id}_extracted.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Results saved to: {output_path}")
        
    elif result["status"] == "json_error":
        print("Raw LLM Response:")
        print(result["raw_response"])
        print("\n")
        print(f"⚠ {result['error']}")
        
        # Save raw response
        output_path = DATASET_ROOT / f"{document_id}_extracted_raw.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["raw_response"])
        
        print(f"✓ Raw response saved to: {output_path}")
        
    else:
        print(f"✗ {result['error']}")


def load_existing_results(output_path: Path) -> dict:
    """Load existing results from JSONL file.
    
    Returns:
        dict: Dictionary with document_id as key and result as value
    """
    existing_results = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    result = json.loads(line)
                    doc_id = result.get("document_id")
                    if doc_id:
                        existing_results[doc_id] = result
    return existing_results


def batch_extract(dataset_name: str = "zh.val"):
    """Batch extract key information from all documents in a dataset.
    
    Args:
        dataset_name: Dataset name (e.g., zh.val, en.val)
    """
    # Construct paths
    image_dir = DATASET_ROOT / dataset_name
    kv_json_path = DATASET_ROOT / f"{dataset_name}.kv.json"
    
    # Check if paths exist
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not kv_json_path.exists():
        raise FileNotFoundError(f"KV JSON file not found: {kv_json_path}")
    
    # Get all image files
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    # Prepare output JSONL file
    output_path = DATASET_ROOT / f"{dataset_name}.extracted.jsonl"
    
    # Load existing results
    existing_results = load_existing_results(output_path)
    
    print(f"\n{'='*60}")
    print(f"Batch Processing: {dataset_name}")
    print(f"Total documents: {len(image_files)}")
    print(f"Previously processed: {len(existing_results)}")
    print(f"{'='*60}\n")
    
    # Process each document
    new_results = []
    skipped_count = 0
    success_count = 0
    error_count = 0
    
    for idx, image_file in enumerate(image_files, 1):
        document_id = image_file.stem  # Get filename without extension
        
        # Check if already successfully processed
        if document_id in existing_results:
            existing_result = existing_results[document_id]
            if existing_result.get("status") == "success":
                print(f"[{idx}/{len(image_files)}] Skipping {document_id} (already processed)")
                skipped_count += 1
                continue
            else:
                print(f"[{idx}/{len(image_files)}] Retrying {document_id} (previously failed)...")
        else:
            print(f"[{idx}/{len(image_files)}] Processing {document_id}...")
        
        result = extract_key_info_with_retry(document_id, dataset_name)
        
        if result["status"] == "success":
            new_results.append(result)
            success_count += 1
            print(f"  ✓ Success")
        else:
            error_count += 1
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Add a small delay between requests to avoid rate limiting
        if idx < len(image_files):
            time.sleep(1)
    
    # Save all results to JSONL file (only successful ones)
    # First, write existing successful results
    with open(output_path, "w", encoding="utf-8") as f:
        # Write existing successful results
        for doc_id, result in existing_results.items():
            if result.get("status") == "success":
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # Write new successful results
        for result in new_results:
            if result.get("status") == "success":
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"{'='*60}")
    print(f"Total: {len(image_files)}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"New success: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single document: python llm_kie.py <document_id> [dataset_name]")
        print("  Batch processing: python llm_kie.py --batch <dataset_name>")
        print("\nExample:")
        print("  python llm_kie.py zh_val_0 zh.val")
        print("  python llm_kie.py --batch zh.val")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        # Batch processing mode
        dataset_name = sys.argv[2] if len(sys.argv) > 2 else "zh.val"
        try:
            batch_extract(dataset_name)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Single document mode
        document_id = sys.argv[1]
        dataset_name = sys.argv[2] if len(sys.argv) > 2 else "zh.val"
        
        try:
            extract_key_info(document_id, dataset_name)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
