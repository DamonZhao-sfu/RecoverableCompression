from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image
import argparse
import requests
import json
import io
import base64
from transformers import AutoTokenizer, AutoProcessor
import time
import torch
import io
import base64
import requests
import json
import os
from vllm import LLM, SamplingParams
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, MapType, LongType, BooleanType, DoubleType
from pyspark.sql.functions import col, to_json, explode, collect_list, when, size, element_at, slice, array_sort, concat_ws, lit, struct
import traceback
import sys
import base64
import json
import os
import re
import time
import asyncio
import aiohttp
from typing import Dict, List, Union
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime
import csv
import json
import time
import requests
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import udf, pandas_udf

from RCEncoder import CLIPVisionTower,CLIPTextEncoder  # Assuming first file is saved as clipEncoder.py
vision_tower_name = "/data/models/clip-vit-p14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"  # Default CLIP model

# Initialize CLIPVisionTower with advanced pruning
class MockArgs:
    def __init__(self):
        self.mm_vision_select_layer = -2
        self.mm_vision_select_feature = 'patch'

mock_args = MockArgs()
vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
vision_tower = vision_tower.to("cpu")
# Initialize vLLM model globally
vllm_model = None

def initialize_vllm_model(model_path):
    """Initialize vLLM model once"""
    global vllm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if vllm_model is None:
        print(f"Initializing vLLM model: {model_path}")
        vllm_model = LLM(model=model_path, gpu_memory_utilization=0.8, tensor_parallel_size=2)
        print("vLLM model initialized successfully")
    return vllm_model

def call_vllm_generate_with_embeds(image_embedding, question="What's in this image?", model_path="llava-hf/llava-1.5-7b-hf"):
    """
    Call vLLM generate API with image embeddings
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings
        question: Question to ask about the image
        model_path: Model path for vLLM
        
    Returns:
        Generated text response
    """
    global vllm_model
    
    try:
        # Initialize vLLM model if not already done
        if vllm_model is None:
            vllm_model = initialize_vllm_model(model_path)
        
        # Format prompt according to LLaVA format
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            stop_token_ids=None
        )
        
        image_embedding= image_embedding.detach().cpu()
        print(image_embedding.shape)
        # Prepare input for vLLM
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_embedding},
        }
        
        print(f"Generating response with vLLM...")
        
        # Generate response
        outputs = vllm_model.generate(
            inputs,
            sampling_params=sampling_params
        )
        
        # Extract generated text
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return {
                "choices": [
                    {
                        "message": {
                            "content": generated_text
                        }
                    }
                ]
            }
        else:
            print("No output generated from vLLM")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM generate: {e}")
        import traceback
        traceback.print_exc()
        return None


# Wrapper function to choose between embed and image path methods
def call_vllm_generate(image_embedding=None, image_path=None, question="What's in this image?", model_path="llava-hf/llava-1.5-7b-hf", use_image_path=False):
    """
    Call vLLM generate API with either image embedding or direct image path
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings (for embed method)
        image_path: Path to image file (for direct path method)
        question: Question to ask about the image
        model_path: Model path for vLLM
        use_image_path: If True, use direct image path method; if False, use embedding method
        
    Returns:
        Generated text response
    """

    if image_embedding is None:
        raise ValueError("image_embedding must be provided when use_image_path=False")
    return call_vllm_generate_with_embeds(image_embedding, question, model_path)

def divprune(model, processor, image_path, text_input=None, device="cuda", prune=True):
    """
    Enhanced divprune function with text-guided token recycling
    
    Args:
        model: The vision-language model
        processor: The processor for image and text
        image_path: Path to the image file
        text_input: Text input for guided pruning (if None, uses default text)
        device: Device to run on
        prune: Whether to apply pruning or not
    """
    global vision_tower
    
    if not prune:
        preprocess_start = time.time()
        # Load and process image
        image = Image.open(image_path)
        inputs = processor.image_processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
                
        encode_start = time.time()                
        with torch.no_grad():
            # Get the vision tower from the model
            vision_tower = model.vision_tower.to(device=device)

            # Extract features
            image_features = vision_tower(pixel_values, output_hidden_states=False)
                        
            # Handle different output formats
            if hasattr(image_features, 'last_hidden_state'):
                visual_tokens = image_features.last_hidden_state
            else:
                visual_tokens = image_features
            
            encode_end = time.time()
            encode_time = encode_end - encode_start
            project_start = time.time()
            visual_tokens = model.multi_modal_projector(visual_tokens.to(torch.float16))
            project_end = time.time()
            project_time = project_end - project_start
        prune_time = 0
        return visual_tokens.cpu(), visual_tokens.shape[1], preprocess_time, encode_time, project_time, prune_time
     
    else:
        preprocess_start = time.time()
        image = Image.open(image_path)
        inputs = vision_tower.image_processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
        
        # Text processing for guided pruning
        text_start = time.time()


        # Create text encoder if not available in model
        # You'll need to pass the appropriate arguments for your CLIPTextEncoder
        import argparse
        args = argparse.Namespace()
        args.mm_vision_select_layer = -1  # Adjust as needed
        args.mm_text_select_feature = 'patch'  # or 'cls_patch'
        args.max_length = 64  # Adjust as needed
        
        text_encoder = CLIPTextEncoder(
            text_encoder="/data/models/clip-vit-p14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",  # or your specific model path
            args=args,
            delay_load=False
        )

        # Ensure text encoder is loaded
        if not text_encoder.is_loaded:
            text_encoder.load_model()

        # Convert text_input string to token IDs using the CLIP tokenizer
        text_ids = text_encoder.text_processor(
            text=[text_input], 
            max_length=text_encoder.max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )

        # Extract text features using the text encoder
        with torch.no_grad():
            text_forward_out = text_encoder.text_encoder(text_ids["input_ids"].to(text_encoder.device))
            text_forward_out = text_forward_out['last_hidden_state']
            text_features = text_forward_out[:, -1, :]  # Get the last token's features
            text_features = text_encoder.text_encoder.text_projection(text_features)

        text_end = time.time()
        text_time = text_end - text_start
        text_features = (text_features, text_input)
        text_score = vision_tower.get_score_by_text(pixel_values, text_features)

        # Apply text-guided token recycling
        with torch.no_grad():
            pruned_features = vision_tower.token_recycling_with_text_in_vision_and_clustring(
                pixel_values, 
                text_score, 
                reduction_ratio=0.05
            ).to(device)
            original_visualToken = 576  # Standard ViT patch count for 336x336 image
            
        prune_start = time.time()
        # Additional processing if needed
        prune_end = time.time()
        prune_time = prune_end - prune_start + text_time
        encode_time = 0
                
        project_start = time.time()
        model.multi_modal_projector = model.multi_modal_projector.to(device)
        pruned_features = model.multi_modal_projector(pruned_features.to(torch.float16))
        project_end = time.time()
        project_time = project_end - project_start
        
        return pruned_features.cpu(), original_visualToken, preprocess_time, encode_time, project_time, prune_time

def main():
    MODEL_PATH = "/data/models/llava-1.5-7b-hf"
    IMAGE_PATH = "/home/haikai/AI_UDF/sparkai/examples/python/35b31d9b4f723f806fd32662ef29edf7.jpg"
    
    print("Loading LLaVA model...")
    
    # Initialize vLLM model
    initialize_vllm_model(MODEL_PATH)
    
    # Only load local model and processor if using embedding method
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto",
        attn_implementation="eager"
    )
    
    processor = LlavaProcessor.from_pretrained(MODEL_PATH, patch_size=14)
        
        # # Initialize vision tower (add this to fix the vision_tower error)
        # vision_tower_name = "/data/models/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
        
        # class MockArgs:
        #     def __init__(self):
        #         self.mm_vision_select_layer = -2
        #         self.mm_vision_select_feature = 'patch'
        
        # mock_args = MockArgs()
        # vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
        # vision_tower = vision_tower.to("cuda")
   
    print("Comparing different token pruning methods...")
    
    # Initialize results storage
    results_data = []
    
    # Get sample images for testing
    spark = SparkSession.builder.appName("AudioVisualQAProcessor") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    # Load CSV data with questions and hints
    df = spark.read.parquet("/home/haikai/MMbench/dev-00000-of-00001.parquet")
    df.show()

    print("Extracting sample data for testing...")
    sample_data = df.select(
        col("index"),
        col("question"), 
        col("hint"), 
        col("answer"),
        col("A"),
        col("B"), 
        col("C"),
        col("D"),
    ).collect()
    
    print(f"Testing {len(sample_data)} questions...")
    print("-" * 60)

    # Run tests with CSV logging
    for i, row in enumerate(sample_data, 1):
        question = row['question']  if row['question'] else ""
        hint = row['hint'] if row['hint'] else ""
        correct_answer = row['answer']  if row['answer']  else ""
        option_a = row['A'] if row['A'] else ""
        option_b = row['B'] if row['B'] else ""
        option_c = row['C'] if row['C'] else ""
        option_d = row['D'] if row['D'] else ""
        image_path = "/home/haikai/MMbench/extracted_images/" + str(i-1) + ".jpg" if row['index'] else ""
        
        # Format the complete question with options
        formatted_question = f"Question: {question}\n\nHint: {hint}\n\nOptions:\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nPlease analyze the image and answer the question."

        print(f"Test {i}/{len(sample_data)}: {image_path}")
        print(f"Question: {question}")
        print(f"Correct Answer: {correct_answer}")
        
        # Initialize result record for this iteration
        result_record = {
            'test_number': i,
            'total_tests': len(sample_data),
            'sample_image_path': image_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'embed_time': 0,
            'api_call_time': 0,
            'total_time': 0,
            'api_success': False,
            'generated_text': '',
            'predicted_answer': '',
            'is_correct': False,
            'error_message': '',
            'full_response': '',
            'model_path': MODEL_PATH,
            'original_token': 0,
            'pruned_token': 0,
            'method': 'direct_image_path' if USE_IMAGE_PATH else 'embeddings'
        }
        
        try:
            if USE_IMAGE_PATH:
                # Direct image path method - no local processing needed
                embed_time = 0
                result_record['embed_time'] = embed_time
                result_record['original_token'] = 0  # Not applicable for direct method
                result_record['pruned_token'] = 0    # Not applicable for direct method
                result_record['preprocess_time'] = 0  # Not applicable for direct method
                result_record['encode_time'] = 0      # Not applicable for direct method
                result_record['project_time'] = 0     # Not applicable for direct method
                result_record['prune_time'] = 0       # Not applicable for direct method
                
                api_time_begin = time.time()
                response = call_vllm_generate(
                    image_path=image_path,
                    question=formatted_question,
                    model_path=MODEL_PATH,
                    use_image_path=True
                )
                api_time_end = time.time()
                
            else:
                # Embedding method - use existing pruning logic
                prune_time_begin = time.time()
                reduced_tokens, originaTokenNum, preprocess_time, encode_time, project_time, pruneTime = divprune(model, processor, image_path, text_input=formatted_question, device="cuda", prune=True)
                prune_time_end = time.time()

                embed_time = prune_time_end - prune_time_begin
                result_record['original_token'] = originaTokenNum
                result_record['pruned_token'] = reduced_tokens.shape[1]
                result_record['preprocess_time'] = preprocess_time
                result_record['encode_time'] = encode_time
                result_record['project_time'] = project_time
                result_record['prune_time'] = pruneTime

                api_time_begin = time.time()
                response = call_vllm_generate(
                    image_embedding=reduced_tokens.to(torch.float16),
                    question=formatted_question,
                    model_path=MODEL_PATH,
                    use_image_path=False
                )
                api_time_end = time.time()
            
            api_call_time = api_time_end - api_time_begin
            result_record['embed_time'] = embed_time
            result_record['api_call_time'] = api_call_time
            result_record['total_time'] = embed_time + api_call_time
            
            if response:
                result_record['api_success'] = True
                print(f"embed time: {embed_time:.2f} seconds")
                print(f"generation time: {api_call_time:.2f} seconds")
                print("=" * 60)
                print("vLLM RESPONSE:")
                print("=" * 60)
                
                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']
                    result_record['generated_text'] = content
                    
                    # Extract predicted answer (A, B, C, or D)
                    predicted_answer = ""
                    content_upper = content.upper().strip()
                    if content_upper in ['A', 'B', 'C', 'D']:
                        predicted_answer = content_upper
                    elif 'A)' in content_upper or content_upper.startswith('A'):
                        predicted_answer = 'A'
                    elif 'B)' in content_upper or content_upper.startswith('B'):
                        predicted_answer = 'B'
                    elif 'C)' in content_upper or content_upper.startswith('C'):
                        predicted_answer = 'C'
                    elif 'D)' in content_upper or content_upper.startswith('D'):
                        predicted_answer = 'D'
                    
                    result_record['predicted_answer'] = predicted_answer
                    result_record['is_correct'] = (predicted_answer == correct_answer.upper())
                    
                    print(f"Generated text: {content}")
                    print(f"Predicted answer: {predicted_answer}")
                    print(f"Correct answer: {correct_answer}")
                    print(f"Is correct: {result_record['is_correct']}")
                else:
                    result_record['full_response'] = json.dumps(response, indent=2)
                    print(f"Full response: {json.dumps(response, indent=2)}")
            else:
                result_record['api_success'] = False
                result_record['error_message'] = "Failed to get response from vLLM"
                print("Failed to get response from vLLM")
                
        except Exception as e:
            result_record['api_success'] = False
            result_record['error_message'] = str(e)
            print(f"Error processing test {i}: {e}")
            import traceback
            traceback.print_exc()
        
        # Add the result to our data list
        results_data.append(result_record)
        print()

    # Calculate accuracy
    successful_tests = [r for r in results_data if r['api_success'] and r['predicted_answer']]
    if successful_tests:
        accuracy = sum(1 for r in successful_tests if r['is_correct']) / len(successful_tests)
        print(f"\nOverall Accuracy: {accuracy:.2%} ({sum(1 for r in successful_tests if r['is_correct'])}/{len(successful_tests)})")

    print(f"\nToken pruning with vLLM direct API completed successfully using {'direct image path' if USE_IMAGE_PATH else 'embedding'} method!")
    
    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    method_suffix = "direct_path" if USE_IMAGE_PATH else "prune_1_4"
    results_csv_path = f"llava_eval_results_{method_suffix}.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved detailed results to {results_csv_path}")

    # Calculate summary stats
    successful = results_df[results_df['api_success'] & results_df['predicted_answer'].astype(bool)]

    summary = {}

    if not successful.empty:
        summary['accuracy'] = successful['is_correct'].mean()
        summary['accuracy_count'] = f"{successful['is_correct'].sum()}/{len(successful)}"
        summary['method'] = 'direct_image_path' if USE_IMAGE_PATH else 'embeddings'

        if not USE_IMAGE_PATH:
            # Only calculate these stats for embedding method
            summary['project_time_avg'] = successful['project_time'].mean()
            summary['project_time_min'] = successful['project_time'].min()
            summary['project_time_max'] = successful['project_time'].max()

            summary['preprocess_time_avg'] = successful['preprocess_time'].mean()
            summary['preprocess_time_min'] = successful['preprocess_time'].min()
            summary['preprocess_time_max'] = successful['preprocess_time'].max()

            summary['encode_time_avg'] = successful['encode_time'].mean()
            summary['encode_time_min'] = successful['encode_time'].min()
            summary['encode_time_max'] = successful['encode_time'].max()

            summary['prune_time_avg'] = successful['prune_time'].mean()
            summary['prune_time_min'] = successful['prune_time'].min()
            summary['prune_time_max'] = successful['prune_time'].max()

            summary['token_original_avg'] = successful['original_token'].mean()
            summary['token_pruned_avg'] = successful['pruned_token'].mean()

        summary['api_call_time_avg'] = successful['api_call_time'].mean()
        summary['api_call_time_min'] = successful['api_call_time'].min()
        summary['api_call_time_max'] = successful['api_call_time'].max()

        summary['total_time_avg'] = successful['total_time'].mean()
        summary['total_time_min'] = successful['total_time'].min()
        summary['total_time_max'] = successful['total_time'].max()
        summary['total_time_sum'] = successful['total_time'].sum()

        print(f"\n=== Summary Statistics ({summary['method']}) ===")
        print(f"Accuracy:       {summary['accuracy']:.2%} ({summary['accuracy_count']})")
        
        if not USE_IMAGE_PATH:
            print(f"PreProcess Time:     avg={summary['preprocess_time_avg']:.2f}s, min={summary['preprocess_time_min']:.2f}s, max={summary['preprocess_time_max']:.2f}s")
            print(f"Encode Time:     avg={summary['encode_time_avg']:.2f}s, min={summary['encode_time_min']:.2f}s, max={summary['encode_time_max']:.2f}s")
            print(f"Project Time:     avg={summary['project_time_avg']:.2f}s, min={summary['project_time_min']:.2f}s, max={summary['project_time_max']:.2f}s")
            print(f"Prune Time:     avg={summary['prune_time_avg']:.2f}s, min={summary['prune_time_min']:.2f}s, max={summary['prune_time_max']:.2f}s")
            print(f"Tokens:         avg original={summary['token_original_avg']:.1f}, avg pruned={summary['token_pruned_avg']:.1f}")
        
        print(f"Generation Time:  avg={summary['api_call_time_avg']:.2f}s, min={summary['api_call_time_min']:.2f}s, max={summary['api_call_time_max']:.2f}s")
        print(f"Total Time:     avg={summary['total_time_avg']:.2f}s, min={summary['total_time_min']:.2f}s, max={summary['total_time_max']:.2f}s, sum={summary['total_time_sum']:.2f}s")

        # Save summary stats
        summary_df = pd.DataFrame([summary])
        summary_csv_path = f"llava_summary_stats_{method_suffix}.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary statistics to {summary_csv_path}")

    else:
        print("No successful responses to compute summary statistics.")

if __name__ == '__main__':
    main()