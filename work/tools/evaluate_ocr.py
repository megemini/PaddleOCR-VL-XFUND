#!/usr/bin/env python3
"""
OCR Model Evaluation Script

This script evaluates different OCR models (PaddleOCR-VL, HunyuanOCR, GLM-OCR) 
on JSONL datasets containing image and text information.
"""

import json
import os
import sys
import warnings
import base64
from typing import Dict, List, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
import json_repair
from io import BytesIO

# Add PaddleOCR-VL path
sys.path.insert(0, '/home/shun/workspace/Projects/megemini/PaddleOCR-VL-REC')


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def resize_image(image_path: str, resize_ratio: float = None, output_path: str = None) -> str:
    """
    Resize an image by a ratio while maintaining aspect ratio.
    
    Args:
        image_path: Path to the original image
        resize_ratio: Resize ratio (e.g., 0.5 for half size). If None, no resizing.
        output_path: Path to save resized image. If None, creates temp file.
        
    Returns:
        Path to the resized image (or original if no resize needed)
    """
    if resize_ratio is None or resize_ratio >= 1.0:
        return image_path
    
    from PIL import Image
    
    # Open image
    img = Image.open(image_path)
    width, height = img.size
    
    # Calculate new dimensions
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    
    # Ensure minimum size of 1 pixel
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    # Resize image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Save resized image
    if output_path is None:
        # Create temporary file
        import tempfile
        import os
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(tempfile.gettempdir(), f"{name}_resized{ext}")
    
    resized_img.save(output_path)
    return output_path


class OCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    def __init__(self, resize_ratio: Optional[float] = None, stream: bool = False):
        """
        Initialize OCR engine with optional image resizing.
        
        Args:
            resize_ratio: Image resize ratio (e.g., 0.5 for half size). If None, no resizing.
            stream: Whether to use streaming mode for output (for debugging)
        """
        self.resize_ratio = resize_ratio
        self.stream = stream
    
    @abstractmethod
    def recognize(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Perform OCR recognition on an image with a given prompt.
        
        Args:
            image_path: Path to the image file
            prompt: The prompt to guide OCR extraction
            
        Returns:
            Dictionary containing the extracted information
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the OCR model"""
        pass
    
    def _prepare_image(self, image_path: str) -> str:
        """
        Prepare image for processing (resize if needed).
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Path to the prepared image
        """
        return resize_image(image_path, self.resize_ratio)


class PaddleOCRVL(OCREngine):
    """PaddleOCR-VL implementation supporting multiple backends"""
    
    def __init__(
        self,
        backend: str = "paddleocrvl_rec",
        model_path: str = "PaddleOCR-VL-0.9B",
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        genai_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        resize_ratio: Optional[float] = None,
        stream: bool = False,
    ):
        """
        Initialize PaddleOCR-VL engine with specified backend.
        
        Args:
            backend: Backend to use - "vllm", "transformers", or "paddleocrvl_rec"
            model_path: Model path or name
            model_dir: Path to the model directory (for paddleocrvl_rec)
            device: Device to run on (e.g., 'cpu', 'gpu:0')
            batch_size: Batch size for inference
            genai_config: GenAI configuration (for paddleocrvl_rec)
            temperature: Sampling temperature (for vllm)
            max_tokens: Maximum tokens to generate
            resize_ratio: Image resize ratio (e.g., 0.5 for half size)
            stream: Whether to use streaming mode for output (for debugging)
        """
        super().__init__(resize_ratio, stream)
        self.backend = backend
        self.model_path = model_path
        
        if backend == "paddleocrvl_rec":
            from paddleocr_vl_rec import PaddleOCRVLRec
            self.model = PaddleOCRVLRec(
                model_name=model_path,
                model_dir=model_dir,
                device=device,
                batch_size=batch_size,
                genai_config=genai_config,
            )
        elif backend == "vllm":
            from vllm import LLM, SamplingParams
            from modelscope import AutoProcessor
            from PIL import Image
            
            self.llm = LLM(model=model_path, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            self.Image = Image
        elif backend == "transformers":
            from modelscope import AutoProcessor, AutoModelForImageTextToText
            from PIL import Image
            import torch
            
            self.processor = AutoProcessor.from_pretrained(model_path)
            device_map = "auto" if device is None else device
            self.model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype="auto",
                device_map=device_map,
            )
            self.max_new_tokens = max_tokens
            self.Image = Image
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: vllm, transformers, paddleocrvl_rec")
    
    def get_model_name(self) -> str:
        return f"PaddleOCR-VL ({self.backend})"
    
    def recognize(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Perform OCR recognition using PaddleOCR-VL.
        
        Args:
            image_path: Path to the image file
            prompt: The prompt in format 'OCR:{"xxx":""}'
            
        Returns:
            Dictionary containing the extracted information
        """
        # Prepare image (resize if needed)
        prepared_image_path = self._prepare_image(image_path)
        
        if self.backend == "paddleocrvl_rec":
            return self._recognize_paddleocrvl_rec(prepared_image_path, prompt)
        elif self.backend == "vllm":
            return self._recognize_vllm(prepared_image_path, prompt)
        elif self.backend == "transformers":
            return self._recognize_transformers(prepared_image_path, prompt)
    
    def _recognize_paddleocrvl_rec(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Recognize using PaddleOCRVLRec backend"""
        # Extract the JSON part from the prompt (remove "OCR:" prefix)
        if prompt.startswith("OCR:"):
            query_str = prompt[4:]
        else:
            query_str = prompt
        
        try:
            query_dict = json.loads(query_str)
        except json.JSONDecodeError:
            query_dict = {}
        
        result = self.model.predict(
            image=image_path,
            prompt_label="ocr",
            query=query_dict,
            return_json=True,
            max_new_tokens=4096
        )
        
        if isinstance(result, str):
            try:
                result = json_repair.loads(result)
            except Exception:
                result = {}
        
        return result if isinstance(result, dict) else {}
    
    def _recognize_vllm(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Recognize using vLLM backend"""
        # Use shared vLLM generation function
        result_text = vllm_generate(
            self.llm,
            self.processor,
            self.sampling_params,
            image_path,
            prompt,
            self.Image
        )
        
        try:
            result = json_repair.loads(result_text)
        except Exception as e:
            warnings.warn(f"Failed to parse JSON result: {e}")
            result = {}
        
        return result if isinstance(result, dict) else {}
    
    def _recognize_transformers(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Recognize using transformers backend"""
        # Convert image to base64
        image_base64 = image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
        
        # Use shared generation function
        result_text = transformers_generate_with_stream(
            self.model,
            self.processor,
            inputs,
            self.max_new_tokens,
            self.stream
        )
        
        try:
            result = json_repair.loads(result_text)
        except Exception as e:
            warnings.warn(f"Failed to parse JSON result: {e}")
            result = {}
        
        return result if isinstance(result, dict) else {}
    
    def close(self):
        """Close the model and release resources"""
        if self.backend == "paddleocrvl_rec" and hasattr(self, 'model'):
            self.model.close()


class HunyuanOCR(OCREngine):
    """HunyuanOCR implementation supporting multiple backends"""
    
    def __init__(
        self,
        backend: str = "vllm",
        model_path: str = "Tencent-Hunyuan/HunyuanOCR",
        device: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 16384,
        resize_ratio: Optional[float] = None,
        stream: bool = False,
    ):
        """
        Initialize HunyuanOCR engine with specified backend.
        
        Args:
            backend: Backend to use - "vllm" or "transformers"
            model_path: Path to the HunyuanOCR model
            device: Device to run on
            temperature: Sampling temperature (for vllm)
            max_tokens: Maximum tokens to generate
            resize_ratio: Image resize ratio (e.g., 0.5 for half size)
            stream: Whether to use streaming mode for output (for debugging)
        """
        super().__init__(resize_ratio, stream)
        self.backend = backend
        self.model_path = model_path
        
        if backend == "vllm":
            from vllm import LLM, SamplingParams
            from modelscope import AutoProcessor
            from PIL import Image
            
            self.llm = LLM(model=model_path, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            self.Image = Image
        elif backend == "transformers":
            from modelscope import AutoProcessor, AutoModelForImageTextToText
            from PIL import Image
            import torch
            
            self.processor = AutoProcessor.from_pretrained(model_path)
            device_map = "auto" if device is None else device
            self.model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype="auto",
                device_map=device_map,
            )
            self.max_new_tokens = max_tokens
            self.Image = Image
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: vllm, transformers")
    
    def get_model_name(self) -> str:
        return f"HunyuanOCR ({self.backend})"
    
    def recognize(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Perform OCR recognition using HunyuanOCR.
        
        Args:
            image_path: Path to the image file
            prompt: The prompt in format '从图片中提取字段内容: ['key1','key2', ...] 并以 JSON 格式返回。'
            
        Returns:
            Dictionary containing the extracted information
        """
        # Prepare image (resize if needed)
        prepared_image_path = self._prepare_image(image_path)
        
        if self.backend == "vllm":
            return self._recognize_vllm(prepared_image_path, prompt)
        elif self.backend == "transformers":
            return self._recognize_transformers(prepared_image_path, prompt)
    
    def _recognize_vllm(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Recognize using vLLM backend"""
        # Use shared vLLM generation function
        result_text = vllm_generate(
            self.llm,
            self.processor,
            self.sampling_params,
            image_path,
            prompt,
            self.Image
        )
        
        try:
            result = json_repair.loads(result_text)
        except Exception as e:
            warnings.warn(f"Failed to parse JSON result: {e}")
            result = {}
        
        return result if isinstance(result, dict) else {}
    
    def _recognize_transformers(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Recognize using transformers backend"""
        # Convert image to base64
        image_base64 = image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
        
        # Use shared generation function
        result_text = transformers_generate_with_stream(
            self.model,
            self.processor,
            inputs,
            self.max_new_tokens,
            self.stream
        )
        
        try:
            result = json_repair.loads(result_text)
        except Exception as e:
            warnings.warn(f"Failed to parse JSON result: {e}")
            result = {}
        
        return result if isinstance(result, dict) else {}


class GLMOCR(OCREngine):
    """GLM-OCR implementation supporting multiple backends"""
    
    def __init__(
        self,
        backend: str = "transformers",
        model_path: str = "ZhipuAI/GLM-OCR",
        device: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 8192,
        resize_ratio: Optional[float] = None,
        stream: bool = False,
    ):
        """
        Initialize GLM-OCR engine with specified backend.
        
        Args:
            backend: Backend to use - "vllm" or "transformers"
            model_path: Path to the GLM-OCR model
            device: Device to run on
            temperature: Sampling temperature (for vllm)
            max_tokens: Maximum tokens to generate
            resize_ratio: Image resize ratio (e.g., 0.5 for half size)
            stream: Whether to use streaming mode for output (for debugging)
        """
        super().__init__(resize_ratio, stream)
        self.backend = backend
        self.model_path = model_path
        
        if backend == "vllm":
            from vllm import LLM, SamplingParams
            from modelscope import AutoProcessor
            from PIL import Image
            
            self.llm = LLM(model=model_path, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            self.Image = Image
        elif backend == "transformers":
            from modelscope import AutoProcessor, AutoModelForImageTextToText
            from PIL import Image
            import torch
            
            self.processor = AutoProcessor.from_pretrained(model_path)
            device_map = "auto" if device is None else device
            self.model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype="auto",
                device_map=device_map,
            )
            self.max_new_tokens = max_tokens
            self.Image = Image
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: vllm, transformers")
    
    def get_model_name(self) -> str:
        return f"GLM-OCR ({self.backend})"
    
    def recognize(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Perform OCR recognition using GLM-OCR.
        
        Args:
            image_path: Path to the image file
            prompt: The prompt in format '请按下列JSON格式输出图中信息:{"xxx": "",...}'
            
        Returns:
            Dictionary containing the extracted information
        """
        # Prepare image (resize if needed)
        prepared_image_path = self._prepare_image(image_path)
        
        if self.backend == "vllm":
            return self._recognize_vllm(prepared_image_path, prompt)
        elif self.backend == "transformers":
            return self._recognize_transformers(prepared_image_path, prompt)
    
    def _recognize_vllm(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Recognize using vLLM backend"""
        # Use shared vLLM generation function
        result_text = vllm_generate(
            self.llm,
            self.processor,
            self.sampling_params,
            image_path,
            prompt,
            self.Image
        )
        
        try:
            result = json_repair.loads(result_text)
        except Exception as e:
            warnings.warn(f"Failed to parse JSON result: {e}")
            result = {}
        
        return result if isinstance(result, dict) else {}
    
    def _recognize_transformers(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Recognize using transformers backend"""
        # Convert image to base64
        image_base64 = image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
        
        # Use shared generation function
        result_text = transformers_generate_with_stream(
            self.model,
            self.processor,
            inputs,
            self.max_new_tokens,
            self.stream
        )
        
        try:
            result = json_repair.loads(result_text)
        except Exception as e:
            warnings.warn(f"Failed to parse JSON result: {e}")
            result = {}
        
        return result if isinstance(result, dict) else {}


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_no_mask_text(data_item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Extract text with tag 'no_mask' from a data item.
    
    Args:
        data_item: A single data item from JSONL
        
    Returns:
        Tuple of (text, tag) if found, None otherwise
    """
    text_info = data_item.get('text_info', [])
    for item in text_info:
        if item.get('tag') == 'no_mask':
            return item.get('text', ''), item.get('tag')
    return None


def extract_image_path(data_item: Dict[str, Any]) -> Optional[str]:
    """
    Extract image path from a data item.
    
    Args:
        data_item: A single data item from JSONL
        
    Returns:
        Image path if found, None otherwise
    """
    image_info = data_item.get('image_info', [])
    if image_info and len(image_info) > 0:
        return image_info[0].get('image_url')
    return None


def parse_json_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON text, handling potential errors.
    
    Args:
        text: JSON string to parse
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON text: {e}")
        return None


def construct_paddleocr_prompt(json_data: Dict[str, Any]) -> str:
    """
    Construct prompt for PaddleOCR-VL.
    Format: OCR:{"xxx":""}
    
    Args:
        json_data: Dictionary containing the keys to extract
        
    Returns:
        Formatted prompt string
    """
    def build_prompt_dict(data: Any) -> Any:
        """Recursively build prompt dict based on value types."""
        if isinstance(data, dict):
            # For dict, recursively process each value
            result = {}
            for key, value in data.items():
                result[key] = build_prompt_dict(value)
            return result
        elif isinstance(data, list):
            # For list, use empty list
            return []
        else:
            # For string/number, use empty string
            return ""
    
    prompt_dict = build_prompt_dict(json_data)
    return f"OCR:{json.dumps(prompt_dict, ensure_ascii=False)}"


def construct_hunyuan_prompt(json_data: Dict[str, Any]) -> str:
    """
    Construct prompt for HunyuanOCR.
    Format: 从图片中提取字段内容: ['key1','key2', ...] 并以 JSON 格式返回。
    
    Args:
        json_data: Dictionary containing the keys to extract
        
    Returns:
        Formatted prompt string
    """
    keys = list(json_data.keys())
    keys_str = str(keys)
    return f"从图片中提取字段内容: {keys_str} 并以 JSON 格式返回。"


def construct_glm_prompt(json_data: Dict[str, Any]) -> str:
    """
    Construct prompt for GLM-OCR.
    Format: 请按下列JSON格式输出图中信息:{"xxx": "",...}
    
    Args:
        json_data: Dictionary containing the keys to extract
        
    Returns:
        Formatted prompt string
    """
    def build_prompt_dict(data: Any) -> Any:
        """Recursively build prompt dict based on value types."""
        if isinstance(data, dict):
            # For dict, recursively process each value
            result = {}
            for key, value in data.items():
                result[key] = build_prompt_dict(value)
            return result
        else:
            # For list, string, or number, use empty string
            return ""
    
    prompt_dict = build_prompt_dict(json_data)
    return f"请按下列JSON格式输出图中信息:{json.dumps(prompt_dict, ensure_ascii=False)}"


def compare_results(ground_truth: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare ground truth with prediction and calculate accuracy.
    
    Args:
        ground_truth: The expected result
        prediction: The OCR model's prediction
        
    Returns:
        Dictionary containing comparison results and accuracy metrics
    """
    results = {
        'total_fields': 0,
        'correct_fields': 0,
        'field_results': {},
        'accuracy': 0.0
    }
    
    for key, expected_value in ground_truth.items():
        results['total_fields'] += 1
        predicted_value = prediction.get(key)
        
        # Handle nested dictionaries
        if isinstance(expected_value, dict) and isinstance(predicted_value, dict):
            nested_result = compare_results(expected_value, predicted_value)
            is_correct = nested_result['accuracy'] == 1.0
        elif isinstance(expected_value, list) and isinstance(predicted_value, list):
            # For lists, sort and compare
            is_correct = compare_sorted_values(expected_value, predicted_value)
        else:
            is_correct = str(expected_value) == str(predicted_value)
        
        results['field_results'][key] = {
            'expected': expected_value,
            'predicted': predicted_value,
            'correct': is_correct
        }
        
        if is_correct:
            results['correct_fields'] += 1
    
    if results['total_fields'] > 0:
        results['accuracy'] = results['correct_fields'] / results['total_fields']
    
    return results


def compare_sorted_values(expected: Any, predicted: Any) -> bool:
    """
    Compare two values after sorting (for lists and dicts).
    
    Args:
        expected: Expected value
        predicted: Predicted value
        
    Returns:
        True if values match after sorting, False otherwise
    """
    if isinstance(expected, list) and isinstance(predicted, list):
        # Sort lists for comparison
        try:
            # Try to sort if elements are comparable
            sorted_expected = sorted(expected, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
            sorted_predicted = sorted(predicted, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
            return sorted_expected == sorted_predicted
        except (TypeError, ValueError):
            # If sorting fails, compare directly
            return expected == predicted
    elif isinstance(expected, dict) and isinstance(predicted, dict):
        # For dicts, compare recursively
        if set(expected.keys()) != set(predicted.keys()):
            return False
        for key in expected.keys():
            if not compare_sorted_values(expected[key], predicted[key]):
                return False
        return True
    else:
        # For other types, compare directly
        return str(expected) == str(predicted)


def transformers_generate_with_stream(
    model,
    processor,
    inputs: Dict[str, Any],
    max_new_tokens: int,
    stream: bool = False
) -> str:
    """
    Generate text using transformers model with optional streaming support.
    
    Args:
        model: The transformers model
        processor: The processor/tokenizer
        inputs: Input dictionary for the model
        max_new_tokens: Maximum number of new tokens to generate
        stream: Whether to use streaming mode for output
        
    Returns:
        Generated text string
    """
    # Stream mode for debugging
    if stream:
        print("\n[STREAM MODE] Generating output character by character:")
        print("-" * 50)
        
        from transformers import TextIteratorStreamer
        import threading
        
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False
        )
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
        )
        
        # Run generation in a separate thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Print output character by character
        result_text = ""
        for char in streamer:
            print(char, end='', flush=True)
            result_text += char
        
        print("\n" + "-" * 50)
        thread.join()
    else:
        # Normal mode
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        result_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
    
    return result_text


def vllm_generate(
    llm,
    processor,
    sampling_params,
    image_path: str,
    prompt: str,
    Image_class
) -> str:
    """
    Generate text using vLLM backend.
    
    Args:
        llm: The vLLM model instance
        processor: The processor/tokenizer
        sampling_params: vLLM sampling parameters
        image_path: Path to the image file
        prompt: The text prompt
        Image_class: PIL Image class for loading images
        
    Returns:
        Generated text string
    """
    # Load image
    img = Image_class.open(image_path)
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Apply chat template
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Prepare inputs
    inputs = {
        "prompt": prompt_text,
        "multi_modal_data": {"image": [img]}
    }
    
    # Generate
    output = llm.generate([inputs], sampling_params)[0]
    result_text = output.outputs[0].text
    
    return result_text


def evaluate_dataset(
    jsonl_path: str,
    ocr_engine: OCREngine,
    prompt_type: str = 'paddleocr',
    predictions_path: Optional[str] = None,
    log_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate an OCR model on a JSONL dataset.
    
    Args:
        jsonl_path: Path to the JSONL file
        ocr_engine: OCR engine instance
        prompt_type: Type of prompt to construct ('paddleocr', 'hunyuan', 'glm')
        predictions_path: Optional path to save predictions in JSONL format
        log_path: Optional path to save log entries
        
    Returns:
        Dictionary containing evaluation results
    """
    # Load data
    data = load_jsonl(jsonl_path)
    
    # Select prompt constructor
    prompt_constructors = {
        'paddleocr': construct_paddleocr_prompt,
        'hunyuan': construct_hunyuan_prompt,
        'glm': construct_glm_prompt
    }
    
    if prompt_type not in prompt_constructors:
        raise ValueError(f"Unknown prompt type: {prompt_type}. "
                        f"Supported types: {list(prompt_constructors.keys())}")
    
    construct_prompt = prompt_constructors[prompt_type]
    
    # Evaluation results
    evaluation_results = {
        'model_name': ocr_engine.get_model_name(),
        'prompt_type': prompt_type,
        'total_samples': 0,
        'processed_samples': 0,
        'failed_samples': 0,
        'overall_accuracy': 0.0,
        'sample_results': []
    }
    
    # Storage for predictions and logs
    predictions = []
    log_entries = []
    
    total_accuracy = 0.0
    
    for idx, item in enumerate(data):
        evaluation_results['total_samples'] += 1
        
        # Extract no_mask text
        text_tag = extract_no_mask_text(item)
        if not text_tag:
            msg = f"Sample {idx}: No 'no_mask' text found, skipping"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            continue
        
        text, tag = text_tag
        
        # Extract image path
        image_path = extract_image_path(item)
        if not image_path:
            msg = f"Sample {idx}: No image path found, skipping"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            continue
        
        # Check if image exists
        if not os.path.exists(image_path):
            msg = f"Sample {idx}: Image not found: {image_path}, skipping"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            continue
        
        # Parse JSON text
        json_data = parse_json_text(text)
        if not json_data:
            msg = f"Sample {idx}: Failed to parse JSON text, skipping"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            continue
        
        # Construct prompt
        prompt = construct_prompt(json_data)
        
        # Perform OCR
        try:
            prediction = ocr_engine.recognize(image_path, prompt)
            
            # Compare results
            comparison = compare_results(json_data, prediction)
            
            sample_result = {
                'sample_id': idx,
                'image_path': image_path,
                'prompt': prompt,
                'ground_truth': json_data,
                'prediction': prediction,
                'accuracy': comparison['accuracy'],
                'field_results': comparison['field_results']
            }
            
            evaluation_results['sample_results'].append(sample_result)
            total_accuracy += comparison['accuracy']
            evaluation_results['processed_samples'] += 1
            
            # Add to predictions list
            predictions.append({
                'sample_id': idx,
                'image_path': image_path,
                'ground_truth': json_data,
                'prediction': prediction,
                'accuracy': comparison['accuracy']
            })
            
            # Add log entry
            msg = f"Sample {idx}: Accuracy = {comparison['accuracy']:.2%}"
            print(msg)
            log_entries.append(msg)
            
        except Exception as e:
            msg = f"Sample {idx}: Error during OCR processing: {e}"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            continue
    
    # Calculate overall accuracy
    if evaluation_results['processed_samples'] > 0:
        evaluation_results['overall_accuracy'] = (
            total_accuracy / evaluation_results['processed_samples']
        )
    
    # Save predictions if path provided
    if predictions_path:
        save_predictions(predictions, predictions_path)
    
    # Save log if path provided
    if log_path:
        save_log(log_entries, log_path)
    
    return evaluation_results


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save the results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")


def save_predictions(predictions: List[Dict[str, Any]], output_path: str):
    """
    Save prediction results to a JSONL file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the predictions
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f"Predictions saved to: {output_path}")


def save_log(log_entries: List[str], log_path: str):
    """
    Save log entries to a text file.
    
    Args:
        log_entries: List of log entry strings
        log_path: Path to save the log
    """
    with open(log_path, 'w', encoding='utf-8') as f:
        for entry in log_entries:
            f.write(entry + '\n')
    print(f"Log saved to: {log_path}")


def print_summary(results: Dict[str, Any]):
    """
    Print a summary of evaluation results.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {results['model_name']}")
    print(f"Prompt Type: {results['prompt_type']}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Processed Samples: {results['processed_samples']}")
    print(f"Failed Samples: {results['failed_samples']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    print("="*60)


def main():
    """Main function to run the evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate OCR models on JSONL datasets')
    parser.add_argument('--jsonl', type=str, required=True,
                       help='Path to the JSONL file')
    parser.add_argument('--model', type=str, required=True,
                       choices=['paddleocr', 'hunyuan', 'glm'],
                       help='OCR model to use')
    parser.add_argument('--backend', type=str, default=None,
                       help='Backend to use: "vllm", "transformers", or "paddleocrvl_rec" (for PaddleOCR-VL only). '
                            'Default: "paddleocrvl_rec" for PaddleOCR-VL, "vllm" for HunyuanOCR, "transformers" for GLM-OCR')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for all results (default: output)')
    
    # Model path arguments
    parser.add_argument('--model_path', type=str, default=None,
                       help='Model path (overrides default for each model)')
    
    # PaddleOCR-VL specific arguments
    parser.add_argument('--paddleocr_model_dir', type=str, default=None,
                       help='PaddleOCR-VL model directory (for paddleocrvl_rec backend)')
    
    # Common arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (e.g., "cpu", "gpu:0")')
    parser.add_argument('--temperature', type=float, default=0,
                       help='Sampling temperature (for vllm backend, default: 0)')
    parser.add_argument('--max_tokens', type=int, default=8192,
                       help='Maximum tokens to generate (default: 8192)')
    parser.add_argument('--resize_ratio', type=float, default=None,
                       help='Image resize ratio (e.g., 0.5 for half size). If not specified, no resizing.')
    parser.add_argument('--stream', action='store_true',
                       help='Enable streaming mode to print output character by character (for debugging)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output file paths
    output_results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    output_predictions_path = os.path.join(args.output_dir, 'predictions.jsonl')
    output_log_path = os.path.join(args.output_dir, 'evaluation.log')
    
    # Set default backends for each model
    default_backends = {
        'paddleocr': 'paddleocrvl_rec',
        'hunyuan': 'vllm',
        'glm': 'transformers'
    }
    
    backend = args.backend if args.backend else default_backends[args.model]
    
    # Set default model paths
    default_model_paths = {
        'paddleocr': 'PaddleOCR-VL-0.9B',
        'hunyuan': 'Tencent-Hunyuan/HunyuanOCR',
        'glm': 'ZhipuAI/GLM-OCR'
    }
    
    model_path = args.model_path if args.model_path else default_model_paths[args.model]
    
    # Create OCR engine based on model type
    if args.model == 'paddleocr':
        if backend not in ['vllm', 'transformers', 'paddleocrvl_rec']:
            raise ValueError(f"Invalid backend '{backend}' for PaddleOCR-VL. Supported: vllm, transformers, paddleocrvl_rec")
        
        ocr_engine = PaddleOCRVL(
            backend=backend,
            model_path=model_path,
            model_dir=args.paddleocr_model_dir,
            device=args.device,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resize_ratio=args.resize_ratio,
            stream=args.stream
        )
    elif args.model == 'hunyuan':
        if backend not in ['vllm', 'transformers']:
            raise ValueError(f"Invalid backend '{backend}' for HunyuanOCR. Supported: vllm, transformers")
        
        ocr_engine = HunyuanOCR(
            backend=backend,
            model_path=model_path,
            device=args.device,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resize_ratio=args.resize_ratio,
            stream=args.stream
        )
    elif args.model == 'glm':
        if backend not in ['vllm', 'transformers']:
            raise ValueError(f"Invalid backend '{backend}' for GLM-OCR. Supported: vllm, transformers")
        
        ocr_engine = GLMOCR(
            backend=backend,
            model_path=model_path,
            device=args.device,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resize_ratio=args.resize_ratio,
            stream=args.stream
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    try:
        # Run evaluation
        print(f"Starting evaluation with {ocr_engine.get_model_name()}...")
        print(f"Output directory: {args.output_dir}")
        if args.resize_ratio:
            print(f"Image resizing enabled: ratio = {args.resize_ratio}")
        
        results = evaluate_dataset(
            args.jsonl, 
            ocr_engine, 
            args.model,
            predictions_path=output_predictions_path,
            log_path=output_log_path
        )
        
        # Print and save results
        print_summary(results)
        save_results(results, output_results_path)
        
        print(f"\nAll results saved to: {args.output_dir}")
        print(f"  - Evaluation results: {output_results_path}")
        print(f"  - Predictions: {output_predictions_path}")
        print(f"  - Log: {output_log_path}")
    finally:
        # Clean up resources
        if hasattr(ocr_engine, 'close'):
            ocr_engine.close()


if __name__ == '__main__':
    main()
