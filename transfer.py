#!/usr/bin/env python
# coding: utf-8

"""
Neuron Attribution for Efficient Arithmetic Knowledge Transfer across LLaMA Models

This implementation applies the Comparative Neuron Analysis (CNA) method to identify 
critical neurons responsible for arithmetic capabilities in LLaMA-13B, and transfers
these neurons to LLaMA-7B to enhance its arithmetic reasoning abilities.

The approach includes:
1. Creating arithmetic datasets for various operations
2. Identifying critical neurons in the teacher model using CNA
3. Transferring neuron parameters from teacher to student model
4. Applying minimal calibration to stabilize the transferred knowledge
5. Evaluating performance across different operations and complexity levels
"""

import os
import json
import math
import random
import copy
import time
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

###############################################################################
# 1. Arithmetic Dataset with Multiple Operations and Complexity Levels
###############################################################################

class ArithmeticDataset(Dataset):
    """
    Creates arithmetic problems with varying complexity levels and operations.
    Supports addition, subtraction, multiplication, and division.
    """
    def __init__(self, size=1000, digits_range=(1, 2), operations=('+',), seed=42):
        """
        Initialize the dataset with customizable parameters.
        
        Args:
            size: Number of examples to generate
            digits_range: Tuple of (min_digits, max_digits) for operands
            operations: List of operations to include ('+', '-', '*', '/')
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.size = size
        self.min_digits, self.max_digits = digits_range
        self.operations = operations
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        self.samples = self._build_samples()

    def _build_samples(self):
        """Generate arithmetic problems with solutions."""
        samples = []
        operation_names = {
            '+': "sum",
            '-': "difference",
            '*': "product",
            '/': "quotient"
        }
        
        operation_words = {
            '+': "plus",
            '-': "minus",
            '*': "times",
            '/': "divided by"
        }
        
        for _ in range(self.size):
            # Select random digit complexity
            num_digits = random.randint(self.min_digits, self.max_digits)
            
            # Select random operation
            operation = random.choice(self.operations)
            
            # Generate operands based on digit complexity
            min_val = 10 ** (num_digits - 1)
            max_val = 10 ** num_digits - 1
            
            a = random.randint(min_val, max_val)
            
            # For division, ensure clean division (no remainders)
            if operation == '/':
                b = random.randint(1, min(20, a))  # Smaller divisor to keep answers reasonable
                # Make a divisible by b
                a = a - (a % b)
            else:
                b = random.randint(min_val, max_val)
            
            # Compute result
            if operation == '+':
                result = a + b
            elif operation == '-':
                # Ensure positive result
                if a < b:
                    a, b = b, a
                result = a - b
            elif operation == '*':
                result = a * b
            elif operation == '/':
                result = a // b
            
            # Create different prompt formats
            formats = [
                f"The {operation_names[operation]} of {a} and {b} is",
                f"Q: What is {a} {operation} {b}? A:",
                f"{a} {operation_words[operation]} {b} is",
                f"{a} {operation} {b} ="
            ]
            prompt_format = random.choice(formats)
            
            samples.append({
                "prompt": prompt_format,
                "answer": str(result),
                "operation": operation,
                "digits": num_digits,
                "a": a,
                "b": b
            })
        
        return samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloader(dataset, tokenizer, batch_size=8, max_length=64):
    """Create a dataloader for the arithmetic dataset."""
    
    def _collate_fn(batch):
        """Combine prompt and answer for each example."""
        texts = []
        for item in batch:
            combined = item["prompt"] + " " + item["answer"]
            texts.append(combined)
        
        # Tokenize all texts together
        encodings = tokenizer(
            texts, 
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the same as input_ids
        labels = encodings["input_ids"].clone()
        
        # Create prompt mask to compute loss only on answer tokens
        prompt_masks = []
        for item in batch:
            prompt_tokens = tokenizer(item["prompt"], return_tensors="pt")
            prompt_length = prompt_tokens["input_ids"].size(1)
            
            # Create mask: -100 for prompt tokens (ignored in loss), actual token ids for answer
            mask = torch.ones(max_length, dtype=torch.long) * -100
            mask[prompt_length:] = labels[0, prompt_length:]
            prompt_masks.append(mask)
        
        labels = torch.stack(prompt_masks)
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn
    )


###############################################################################
# 2. Model Loading and Analysis Tools
###############################################################################

class LLaMAModelManager:
    """
    Manages loading and handling of LLaMA models for neuron transfer experiments.
    """
    
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
    
    def load_model(self, model_name, is_8bit=False, device=device):
        """
        Load a LLaMA model and tokenizer.
        
        Args:
            model_name: HuggingFace model name or path
            is_8bit: Whether to load in 8-bit precision to save memory
            device: Device to load the model on
            
        Returns:
            model, tokenizer
        """
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=False,
            cache_dir=self.cache_dir
        )
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with memory optimizations if needed
        load_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float16,  # Load in half precision
            "device_map": "auto" if is_8bit else None,  # Needed for 8-bit loading
        }
        
        if is_8bit:
            load_kwargs["load_in_8bit"] = True
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Apply model optimizations
        if not is_8bit:
            model = model.to(device)
        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        # Return loaded model and tokenizer
        return model, tokenizer
    
    def get_model_size_info(self, model, tokenizer):
        """Get detailed information about model size and structure."""
        info = {
            "model_name": model.config._name_or_path,
            "parameter_count": sum(p.numel() for p in model.parameters()) / 1_000_000,
            "layer_count": len(model.model.layers),
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.model.layers[0].mlp.gate_proj.weight.shape[0],
            "num_attention_heads": model.config.num_attention_heads,
            "head_dim": model.config.hidden_size // model.config.num_attention_heads,
            "vocab_size": len(tokenizer),
        }
        
        return info


###############################################################################
# 3. Comparative Neuron Analysis (CNA) Implementation
###############################################################################

class LLaMANeuronAnalyzer:
    """
    Implements the Comparative Neuron Analysis method for identifying
    critical neurons in LLaMA models for arithmetic operations.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Extract model structure information
        self.num_layers = len(model.model.layers)
        self.hidden_size = model.config.hidden_size
        try:
            self.intermediate_size = model.model.layers[0].mlp.gate_proj.weight.shape[0]
        except:
            self.intermediate_size = model.model.layers[0].mlp.up_proj.weight.shape[0]
    
    def compute_output_probability(self, text, target_token=None, intervened_neurons=None):
        """
        Compute the probability of the target token given the input text,
        with optional neuron intervention.
        
        Args:
            text: Input text prompt
            target_token: Target token to compute probability for
            intervened_neurons: List of (layer_idx, neuron_idx) tuples to zero out
            
        Returns:
            Probability of the target token
        """
        # Encode input text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # Determine target token ID
        if target_token is None:
            # Use the last token of input as target
            target_id = input_ids[0, -1].item()
        elif isinstance(target_token, str):
            # Encode the target token string
            target_encoding = self.tokenizer(target_token, add_special_tokens=False)
            target_id = target_encoding["input_ids"][0]
        else:
            # Assume target_token is already a token ID
            target_id = target_token
        
        # Save original state if we'll be doing intervention
        if intervened_neurons and len(intervened_neurons) > 0:
            # Instead of saving the entire state dict (which is memory intensive),
            # we'll just save the affected neurons' parameters
            original_params = {}
            for layer_idx, neuron_idx in intervened_neurons:
                # Save gate, up, and down projection parameters for this neuron
                mlp = self.model.model.layers[layer_idx].mlp
                
                # Gate projection (up-projection in some LLaMA variants)
                if hasattr(mlp, 'gate_proj'):
                    original_params[(layer_idx, neuron_idx, 'gate')] = {
                        'weight': mlp.gate_proj.weight[neuron_idx, :].clone(),
                    }
                    if hasattr(mlp.gate_proj, 'bias') and mlp.gate_proj.bias is not None:
                        original_params[(layer_idx, neuron_idx, 'gate')]['bias'] = mlp.gate_proj.bias[neuron_idx].clone()
                
                # Up projection
                original_params[(layer_idx, neuron_idx, 'up')] = {
                    'weight': mlp.up_proj.weight[neuron_idx, :].clone(),
                }
                if hasattr(mlp.up_proj, 'bias') and mlp.up_proj.bias is not None:
                    original_params[(layer_idx, neuron_idx, 'up')]['bias'] = mlp.up_proj.bias[neuron_idx].clone()
                
                # Down projection
                col_tensor = mlp.down_proj.weight[:, neuron_idx].clone()
                original_params[(layer_idx, neuron_idx, 'down')] = {
                    'weight': col_tensor,
                }
        
        # Apply neuron intervention if specified
        if intervened_neurons and len(intervened_neurons) > 0:
            for layer_idx, neuron_idx in intervened_neurons:
                self._zero_out_neuron(layer_idx, neuron_idx)
        
        # Forward pass to get logits
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for the last position
            probs = F.softmax(logits, dim=-1)
            target_prob = probs[target_id].item()
        
        # Restore original parameters if we did intervention
        if intervened_neurons and len(intervened_neurons) > 0:
            for (layer_idx, neuron_idx, proj_type), params in original_params.items():
                mlp = self.model.model.layers[layer_idx].mlp
                
                if proj_type == 'gate' and hasattr(mlp, 'gate_proj'):
                    mlp.gate_proj.weight[neuron_idx, :] = params['weight']
                    if 'bias' in params and hasattr(mlp.gate_proj, 'bias') and mlp.gate_proj.bias is not None:
                        mlp.gate_proj.bias[neuron_idx] = params['bias']
                
                elif proj_type == 'up':
                    mlp.up_proj.weight[neuron_idx, :] = params['weight']
                    if 'bias' in params and hasattr(mlp.up_proj, 'bias') and mlp.up_proj.bias is not None:
                        mlp.up_proj.bias[neuron_idx] = params['bias']
                
                elif proj_type == 'down':
                    mlp.down_proj.weight[:, neuron_idx] = params['weight']
        
        return target_prob
    
    def _zero_out_neuron(self, layer_idx, neuron_idx):
        """Zero out a neuron's parameters in the model."""
        with torch.no_grad():
            mlp = self.model.model.layers[layer_idx].mlp
            
            # Zero out gate projection (if exists)
            if hasattr(mlp, 'gate_proj'):
                mlp.gate_proj.weight[neuron_idx, :] = 0
                if hasattr(mlp.gate_proj, 'bias') and mlp.gate_proj.bias is not None:
                    mlp.gate_proj.bias[neuron_idx] = 0
            
            # Zero out up projection
            mlp.up_proj.weight[neuron_idx, :] = 0
            if hasattr(mlp.up_proj, 'bias') and mlp.up_proj.bias is not None:
                mlp.up_proj.bias[neuron_idx] = 0
            
            # Zero out down projection
            mlp.down_proj.weight[:, neuron_idx] = 0
    
    def compute_neuron_importance(self, text, target_token=None, layer_range=None, top_k=100, batch_size=50):
        """
        Compute importance scores for neurons using the CNA method.
        
        Args:
            text: Input text prompt
            target_token: Target token to compute probability for
            layer_range: Tuple of (start_layer, end_layer) to analyze
            top_k: Number of top neurons to return
            batch_size: Number of neurons to test at once (for efficiency)
            
        Returns:
            List of (layer_idx, neuron_idx, importance_score) tuples for top neurons
        """
        # Set default layer range if not provided
        if layer_range is None:
            layer_range = (0, min(8, self.num_layers))  # Default to first 8 layers
        
        # Get baseline probability without intervention
        baseline_prob = self.compute_output_probability(text, target_token)
        logger.info(f"Baseline probability: {baseline_prob:.6f}")
        
        # Store neuron importance scores
        neuron_scores = []
        
        # Analyze each layer
        for layer_idx in tqdm(range(layer_range[0], layer_range[1]), desc="Analyzing layers"):
            # First do batch screening to find potentially important regions
            batch_indices = list(range(0, self.intermediate_size, batch_size))
            for start_idx in tqdm(batch_indices, desc=f"Screening layer {layer_idx}", leave=False):
                end_idx = min(start_idx + batch_size, self.intermediate_size)
                
                # Create a batch of neurons to intervene
                batch_neurons = [(layer_idx, i) for i in range(start_idx, end_idx)]
                
                # Compute probability with this batch zeroed out
                batch_prob = self.compute_output_probability(text, target_token, batch_neurons)
                
                # If significant drop, test individual neurons
                batch_importance = baseline_prob - batch_prob
                
                if batch_importance > 0.01:
                    # Test each neuron in the batch individually
                    for neuron_idx in range(start_idx, end_idx):
                        # Test a single neuron
                        neuron_prob = self.compute_output_probability(
                            text, target_token, [(layer_idx, neuron_idx)]
                        )
                        
                        # Compute importance score (probability drop)
                        importance = baseline_prob - neuron_prob
                        
                        # Store if positive importance
                        if importance > 0:
                            neuron_scores.append((layer_idx, neuron_idx, importance))
        
        # Sort neurons by importance (highest first)
        neuron_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k neurons
        return neuron_scores[:top_k]
    
    def analyze_arithmetic_neurons(self, operations=('+',), digits_range=(1, 1), 
                                  samples_per_op=3, top_k=100, layer_range=None):
        """
        Identify critical neurons for arithmetic operations.
        
        Args:
            operations: List of operations to analyze
            digits_range: Range of digit complexity to test
            samples_per_op: Number of examples to test per operation
            top_k: Number of top neurons to return
            layer_range: Range of layers to analyze
            
        Returns:
            List of critical neurons across all tested examples
        """
        # Create small dataset for analysis
        dataset = ArithmeticDataset(
            size=samples_per_op * len(operations), 
            digits_range=digits_range,
            operations=operations
        )
        
        # Collect critical neurons across all samples
        all_critical_neurons = []
        
        for item in tqdm(dataset.samples, desc="Analyzing arithmetic problems"):
            prompt = item["prompt"]
            answer = item["answer"]
            
            logger.info(f"\nAnalyzing: {prompt} {answer} (Operation: {item['operation']})")
            
            # Compute neuron importance for this example
            critical_neurons = self.compute_neuron_importance(
                prompt + " ", answer, layer_range=layer_range, 
                top_k=min(top_k // len(dataset.samples), 20)
            )
            
            all_critical_neurons.extend(critical_neurons)
            
            # Print top neurons for this example
            logger.info(f"Top 5 neurons for this example:")
            for i, (layer, neuron, importance) in enumerate(critical_neurons[:5]):
                logger.info(f"  {i+1}. Layer {layer}, Neuron {neuron}: {importance:.6f}")
        
        # Deduplicate and aggregate neuron scores
        neuron_dict = {}
        for layer, neuron, importance in all_critical_neurons:
            key = (layer, neuron)
            neuron_dict[key] = neuron_dict.get(key, 0) + importance
        
        # Convert back to list and sort by importance
        critical_neurons = [(layer, neuron, importance) 
                           for (layer, neuron), importance in neuron_dict.items()]
        critical_neurons.sort(key=lambda x: x[2], reverse=True)
        
        return critical_neurons[:top_k]

    def evaluate_arithmetic_accuracy(self, dataset, max_batch_size=4):
        """
        Evaluate the model's accuracy on arithmetic problems.
        
        Args:
            dataset: ArithmeticDataset instance
            max_batch_size: Maximum batch size for evaluation
            
        Returns:
            Dictionary with accuracy metrics
        """
        self.model.eval()
        
        correct = 0
        total = 0
        
        operation_correct = {}
        operation_total = {}
        
        digits_correct = {}
        digits_total = {}
        
        # Process examples in batches
        loader = DataLoader(
            dataset, 
            batch_size=max_batch_size, 
            shuffle=False,
            collate_fn=lambda batch: [{
                "prompt": item["prompt"],
                "answer": item["answer"],
                "operation": item["operation"],
                "digits": item["digits"]
            } for item in batch]
        )
        
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            for item in batch:
                prompt = item["prompt"]
                expected = item["answer"]
                operation = item["operation"]
                digits = item["digits"]
                
                # Update operation counters
                if operation not in operation_total:
                    operation_total[operation] = 0
                    operation_correct[operation] = 0
                operation_total[operation] += 1
                
                # Update digits counters
                if digits not in digits_total:
                    digits_total[digits] = 0
                    digits_correct[digits] = 0
                digits_total[digits] += 1
                
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate prediction
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs.input_ids,
                        max_new_tokens=len(expected) + 2,
                        num_beams=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        temperature=0.1
                    )
                
                # Extract generated text after the prompt
                generated = outputs[0, inputs.input_ids.shape[1]:]
                prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                
                # Check if prediction is correct
                # We're flexible about whitespace and leading characters
                is_correct = expected in prediction
                
                if is_correct:
                    correct += 1
                    operation_correct[operation] += 1
                    digits_correct[digits] += 1
                
                total += 1
        
        # Calculate overall accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-operation accuracy
        operation_accuracy = {
            op: operation_correct.get(op, 0) / cnt 
            for op, cnt in operation_total.items()
        }
        
        # Calculate per-digits accuracy
        digits_accuracy = {
            d: digits_correct.get(d, 0) / cnt 
            for d, cnt in digits_total.items()
        }
        
        return {
            "overall_accuracy": accuracy,
            "operation_accuracy": operation_accuracy,
            "digits_accuracy": digits_accuracy,
            "total_examples": total,
            "total_correct": correct
        }
        
###############################################################################
# 4. Neuron-Level Knowledge Transfer
###############################################################################

class LLaMANeuronTransfer:
    """
    Handles the transfer of neuron parameters from a teacher to student LLaMA model.
    """
    
    def __init__(self, teacher_model, student_model, device="cuda"):
        """
        Initialize with teacher and student models.
        
        Args:
            teacher_model: The source LLaMA model (e.g., 13B)
            student_model: The target LLaMA model (e.g., 7B)
            device: Compute device
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        
        # Extract model structure information
        self.teacher_layers = len(teacher_model.model.layers)
        self.student_layers = len(student_model.model.layers)
        
        try:
            self.teacher_intermediate_size = teacher_model.model.layers[0].mlp.gate_proj.weight.shape[0]
            self.student_intermediate_size = student_model.model.layers[0].mlp.gate_proj.weight.shape[0]
        except:
            self.teacher_intermediate_size = teacher_model.model.layers[0].mlp.up_proj.weight.shape[0]
            self.student_intermediate_size = student_model.model.layers[0].mlp.up_proj.weight.shape[0]
        
        # Create layer mapping between teacher and student
        self.layer_mapping = self._create_layer_mapping()
        
        logger.info(f"Teacher model: {self.teacher_layers} layers, {self.teacher_intermediate_size} neurons per layer")
        logger.info(f"Student model: {self.student_layers} layers, {self.student_intermediate_size} neurons per layer")
    
    def _create_layer_mapping(self):
        """Create a mapping from teacher layers to student layers."""
        mapping = {}
        # Create a proportional mapping
        for teacher_layer in range(self.teacher_layers):
            student_layer = min(int(teacher_layer * self.student_layers / self.teacher_layers), 
                               self.student_layers - 1)
            mapping[teacher_layer] = student_layer
        return mapping
    
    def transfer_neurons(self, critical_neurons):
        """
        Transfer critical neurons from teacher to student model.
        
        Args:
            critical_neurons: List of (layer_idx, neuron_idx, importance) tuples
            
        Returns:
            Modified student model and list of successfully transferred neurons
        """
        # Create a copy of the student model
        modified_student = copy.deepcopy(self.student_model)
        
        # Track successfully transferred neurons
        transferred_neurons = []
        
        logger.info(f"Transferring {len(critical_neurons)} neurons from teacher to student...")
        
        # Process each critical neuron
        for i, (layer_idx, neuron_idx, importance) in enumerate(tqdm(critical_neurons)):
            # Get corresponding student layer
            student_layer_idx = self.layer_mapping.get(layer_idx)
            
            if student_layer_idx is None:
                continue
            
            # For neuron index, we need to check if it's within bounds of student model
            if neuron_idx >= self.student_intermediate_size:
                # Skip if neuron index is out of bounds
                continue
            
            # Get teacher and student MLP modules
            teacher_mlp = self.teacher_model.model.layers[layer_idx].mlp
            student_mlp = modified_student.model.layers[student_layer_idx].mlp
            
            # Transfer gate projection weights/biases (if it exists)
            if hasattr(teacher_mlp, 'gate_proj'):
                with torch.no_grad():
                    # Copy gate projection weight
                    student_mlp.gate_proj.weight[neuron_idx, :] = teacher_mlp.gate_proj.weight[neuron_idx, :]
                    
                    # Copy gate projection bias if exists
                    if (hasattr(teacher_mlp.gate_proj, 'bias') and 
                        teacher_mlp.gate_proj.bias is not None and
                        hasattr(student_mlp.gate_proj, 'bias') and
                        student_mlp.gate_proj.bias is not None):
                        student_mlp.gate_proj.bias[neuron_idx] = teacher_mlp.gate_proj.bias[neuron_idx]
            
            # Transfer up projection weights/biases
            with torch.no_grad():
                # Copy up projection weight
                student_mlp.up_proj.weight[neuron_idx, :] = teacher_mlp.up_proj.weight[neuron_idx, :]
                
                # Copy up projection bias if exists
                if (hasattr(teacher_mlp.up_proj, 'bias') and 
                    teacher_mlp.up_proj.bias is not None and
                    hasattr(student_mlp.up_proj, 'bias') and
                    student_mlp.up_proj.bias is not None):
                    student_mlp.up_proj.bias[neuron_idx] = teacher_mlp.up_proj.bias[neuron_idx]
            
            # Transfer down projection weights
            with torch.no_grad():
                # Down projection is a bit trickier since it's the other dimension
                # We're copying column neuron_idx
                student_mlp.down_proj.weight[:, neuron_idx] = teacher_mlp.down_proj.weight[:, neuron_idx]
            
            # Record this neuron as successfully transferred
            transferred_neurons.append((student_layer_idx, neuron_idx, importance))
        
        logger.info(f"Successfully transferred {len(transferred_neurons)} neurons")
        return modified_student, transferred_neurons

    def neuron_activation_profile(self, model, tokenizer, text, layer_indices, neuron_indices):
        """
        Generate activation profile for specified neurons.
        
        Args:
            model: The model to analyze
            tokenizer: Tokenizer for the model
            text: Input text prompt
            layer_indices: List of layer indices to profile
            neuron_indices: List of neuron indices to profile
            
        Returns:
            Dictionary mapping (layer, neuron) to activation value
        """
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        
        # Set up hooks to capture activations
        activations = {}
        handles = []
        
        def get_activation_hook(layer_idx, neuron_idx):
            def hook(module, input, output):
                # Capture the gate_proj and up_proj activations
                # In LLaMA, gate_proj and up_proj outputs are multiplied (SwiGLU)
                gate_output = None
                up_output = None
                
                if hasattr(module, 'gate_proj'):
                    gate_output = torch.nn.functional.silu(module.gate_proj(input[0]))
                
                up_output = module.up_proj(input[0])
                
                # SwiGLU activation
                if gate_output is not None:
                    neuron_value = (gate_output * up_output)[0, -1, neuron_idx].item()
                else:
                    neuron_value = up_output[0, -1, neuron_idx].item()
                
                activations[(layer_idx, neuron_idx)] = neuron_value
                
            return hook
        
        # Register hooks for each (layer, neuron) pair to monitor
        for layer_idx in layer_indices:
            for neuron_idx in neuron_indices:
                if layer_idx < len(model.model.layers):
                    mlp = model.model.layers[layer_idx].mlp
                    handle = mlp.register_forward_hook(get_activation_hook(layer_idx, neuron_idx))
                    handles.append(handle)
        
        # Forward pass to get activations
        with torch.no_grad():
            _ = model(**inputs)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return activations


###############################################################################
# 5. Minimal Calibration Process
###############################################################################

class LLaMACalibration:
    """
    Implements the minimal calibration process for stabilizing transferred neurons.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        """
        Initialize with model to calibrate.
        
        Args:
            model: The LLaMA model to calibrate
            tokenizer: Tokenizer for the model
            device: Compute device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def calibrate(self, dataset, learning_rate=1e-5, num_epochs=1, batch_size=4):
        """
        Perform minimal calibration on the model using the provided dataset.
        
        Args:
            dataset: ArithmeticDataset for calibration
            learning_rate: Learning rate for calibration (should be very small)
            num_epochs: Number of epochs for calibration (typically just 1)
            batch_size: Batch size for calibration
            
        Returns:
            Calibrated model
        """
        logger.info(f"Starting minimal calibration with lr={learning_rate}, epochs={num_epochs}")
        
        # Create data loader
        loader = create_dataloader(dataset, self.tokenizer, batch_size=batch_size)
        
        # Set up optimizer with very low learning rate
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        total_loss = 0
        total_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in tqdm(loader, desc=f"Calibration epoch {epoch+1}/{num_epochs}"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                total_batches += 1
            
            avg_epoch_loss = epoch_loss / len(loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average loss: {avg_epoch_loss:.4f}")
            total_loss += epoch_loss
        
        avg_loss = total_loss / (total_batches * num_epochs)
        logger.info(f"Calibration complete. Average loss: {avg_loss:.4f}")
        
        return self.model


###############################################################################
# 6. Evaluation Framework
###############################################################################

class ArithmeticEvaluator:
    """
    Comprehensive evaluation framework for arithmetic capabilities.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        """
        Initialize with model to evaluate.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            device: Compute device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.analyzer = LLaMANeuronAnalyzer(model, tokenizer, device)
    
    def evaluate_comprehensive(self, operations=('+', '-', '*', '/'), digits_range=(1, 2), samples_per_op=50):
        """
        Run a comprehensive evaluation across different operations and complexity levels.
        
        Args:
            operations: List of operations to evaluate
            digits_range: Range of digit complexity to test
            samples_per_op: Number of examples to test per operation
            
        Returns:
            Dictionary with detailed evaluation results
        """
        logger.info(f"Running comprehensive evaluation...")
        
        # Create dataset with all operations
        dataset = ArithmeticDataset(
            size=samples_per_op * len(operations),
            digits_range=digits_range,
            operations=operations
        )
        
        # Evaluate on dataset
        results = self.analyzer.evaluate_arithmetic_accuracy(dataset)
        
        # Log results
        logger.info(f"Overall accuracy: {results['overall_accuracy']:.4f}")
        logger.info("Accuracy by operation:")
        for op, acc in results['operation_accuracy'].items():
            logger.info(f"  {op}: {acc:.4f}")
        logger.info("Accuracy by digits:")
        for digits, acc in results['digits_accuracy'].items():
            logger.info(f"  {digits}-digit: {acc:.4f}")
        
        return results
    
    def visualize_results(self, results, title="Arithmetic Evaluation Results", save_path=None):
        """
        Create visualizations for evaluation results.
        
        Args:
            results: Results dictionary from evaluate_comprehensive
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(12, 10))
        
        # 1. Overall accuracy
        plt.subplot(2, 2, 1)
        plt.bar(['Overall'], [results['overall_accuracy']], color='navy')
        plt.ylim(0, 1)
        plt.title("Overall Accuracy")
        plt.ylabel("Accuracy")
        
        # 2. Accuracy by operation
        plt.subplot(2, 2, 2)
        ops = list(results['operation_accuracy'].keys())
        accs = [results['operation_accuracy'][op] for op in ops]
        plt.bar(ops, accs, color='lightblue')
        plt.ylim(0, 1)
        plt.title("Accuracy by Operation")
        plt.ylabel("Accuracy")
        
        # 3. Accuracy by digit complexity
        plt.subplot(2, 2, 3)
        digits = sorted(list(results['digits_accuracy'].keys()))
        accs = [results['digits_accuracy'][d] for d in digits]
        plt.bar([f"{d}-digit" for d in digits], accs, color='lightgreen')
        plt.ylim(0, 1)
        plt.title("Accuracy by Digit Complexity")
        plt.ylabel("Accuracy")
        
        # 4. Sample count and correct count
        plt.subplot(2, 2, 4)
        plt.bar(['Total', 'Correct'], [results['total_examples'], results['total_correct']], 
               color=['gray', 'lightcoral'])
        plt.title("Sample Counts")
        plt.ylabel("Count")
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Visualization saved to {save_path}")
            return save_path
        else:
            plt.show()
        
        return None


###############################################################################
# 7. Complete Experiment Pipeline
###############################################################################

class ArithmeticTransferExperiment:
    """
    End-to-end pipeline for arithmetic knowledge transfer experiments.
    """
    
    def __init__(self, 
                teacher_model_name="meta-llama/Llama-13b-hf", 
                student_model_name="meta-llama/Llama-7b-hf",
                output_dir=None, 
                cache_dir=None,
                device="cuda"):
        """
        Initialize experiment.
        
        Args:
            teacher_model_name: HuggingFace model name for teacher
            student_model_name: HuggingFace model name for student
            output_dir: Directory to save results
            cache_dir: Directory to cache models
            device: Compute device
        """
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.device = device
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"arithmetic_transfer_experiment_{timestamp}"
        else:
            self.output_dir = output_dir
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging to file
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "experiment.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        
        # Initialize manager for loading models
        self.model_manager = LLaMAModelManager(cache_dir=cache_dir)
        
        # Models will be loaded when needed
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.student_model = None
        self.student_tokenizer = None
        self.modified_student = None
        self.calibrated_student = None
    
    def run(self, 
           operations=('+', '-'),
           digits_range=(1, 2),
           samples_per_op=50,
           neuron_count=100,
           calibration_lr=1e-5,
           calibration_epochs=1):
        """
        Run the complete experiment pipeline.
        
        Args:
            operations: Arithmetic operations to test
            digits_range: Range of digit complexity
            samples_per_op: Number of examples per operation
            neuron_count: Number of neurons to transfer
            calibration_lr: Learning rate for calibration
            calibration_epochs: Number of epochs for calibration
            
        Returns:
            Dictionary with experiment results
        """
        # Start timing
        start_time = time.time()
        
        # 1. Save experiment configuration
        config = {
            "teacher_model": self.teacher_model_name,
            "student_model": self.student_model_name,
            "operations": operations,
            "digits_range": digits_range,
            "samples_per_op": samples_per_op,
            "neuron_count": neuron_count,
            "calibration_lr": calibration_lr,
            "calibration_epochs": calibration_epochs,
            "start_time": start_time,
            "device": str(self.device)
        }
        
        with open(os.path.join(self.output_dir, "experiment_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Starting experiment with config: {config}")
        
        try:
            # 2. Load teacher model
            logger.info(f"Loading teacher model: {self.teacher_model_name}")
            self.teacher_model, self.teacher_tokenizer = self.model_manager.load_model(
                self.teacher_model_name, is_8bit=True, device=self.device
            )
            
            # 3. Load student model
            logger.info(f"Loading student model: {self.student_model_name}")
            self.student_model, self.student_tokenizer = self.model_manager.load_model(
                self.student_model_name, is_8bit=False, device=self.device
            )
            
            # Get model info
            teacher_info = self.model_manager.get_model_size_info(self.teacher_model, self.teacher_tokenizer)
            student_info = self.model_manager.get_model_size_info(self.student_model, self.student_tokenizer)
            
            logger.info(f"Teacher model info: {teacher_info}")
            logger.info(f"Student model info: {student_info}")
            
            with open(os.path.join(self.output_dir, "model_info.json"), "w") as f:
                json.dump({"teacher": teacher_info, "student": student_info}, f, indent=2)
            
            # 4. Evaluate baseline student performance
            logger.info("Evaluating baseline student performance...")
            evaluator = ArithmeticEvaluator(self.student_model, self.student_tokenizer, self.device)
            baseline_results = evaluator.evaluate_comprehensive(
                operations=operations,
                digits_range=digits_range,
                samples_per_op=samples_per_op
            )
            
            # Save baseline results
            with open(os.path.join(self.output_dir, "baseline_results.json"), "w") as f:
                json.dump(baseline_results, f, indent=2)
            
            # Visualize baseline results
            evaluator.visualize_results(
                baseline_results,
                title="Baseline Student Model Performance",
                save_path=os.path.join(self.output_dir, "baseline_results.png")
            )
            
            # 5. Identify critical neurons in teacher model
            logger.info("Identifying critical neurons in teacher model...")
            analyzer = LLaMANeuronAnalyzer(self.teacher_model, self.teacher_tokenizer, self.device)
            
            critical_neurons = analyzer.analyze_arithmetic_neurons(
                operations=operations,
                digits_range=digits_range,
                samples_per_op=min(3, samples_per_op),  # Limit to 3 examples per op for speed
                top_k=neuron_count,
                layer_range=(0, 8)  # Focus on early layers
            )
            
            # Save critical neurons
            critical_neurons_data = [
                {"layer": int(l), "neuron": int(n), "importance": float(i)}
                for l, n, i in critical_neurons
            ]
            
            with open(os.path.join(self.output_dir, "critical_neurons.json"), "w") as f:
                json.dump(critical_neurons_data, f, indent=2)
            
            # 6. Transfer neurons from teacher to student
            logger.info("Transferring neurons from teacher to student...")
            transfer_manager = LLaMANeuronTransfer(self.teacher_model, self.student_model, self.device)
            self.modified_student, transferred_neurons = transfer_manager.transfer_neurons(critical_neurons)
            
            # Save transferred neurons
            transferred_neurons_data = [
                {"layer": int(l), "neuron": int(n), "importance": float(i)}
                for l, n, i in transferred_neurons
            ]
            
            with open(os.path.join(self.output_dir, "transferred_neurons.json"), "w") as f:
                json.dump(transferred_neurons_data, f, indent=2)
            
            # 7. Evaluate modified student (before calibration)
            logger.info("Evaluating modified student (before calibration)...")
            evaluator = ArithmeticEvaluator(self.modified_student, self.student_tokenizer, self.device)
            pre_calibration_results = evaluator.evaluate_comprehensive(
                operations=operations,
                digits_range=digits_range,
                samples_per_op=samples_per_op
            )
            
            # Save pre-calibration results
            with open(os.path.join(self.output_dir, "pre_calibration_results.json"), "w") as f:
                json.dump(pre_calibration_results, f, indent=2)
            
            # Visualize pre-calibration results
            evaluator.visualize_results(
                pre_calibration_results,
                title="Modified Student Model Performance (Before Calibration)",
                save_path=os.path.join(self.output_dir, "pre_calibration_results.png")
            )
            
            # 8. Create calibration dataset
            logger.info("Creating calibration dataset...")
            calibration_dataset = ArithmeticDataset(
                size=samples_per_op * len(operations),
                digits_range=digits_range,
                operations=operations,
                seed=42  # Use fixed seed for reproducibility
            )
            
            # 9. Calibrate the modified student model
            logger.info("Calibrating modified student model...")
            calibrator = LLaMACalibration(self.modified_student, self.student_tokenizer, self.device)
            self.calibrated_student = calibrator.calibrate(
                calibration_dataset,
                learning_rate=calibration_lr,
                num_epochs=calibration_epochs,
                batch_size=4
            )
            
            # 10. Evaluate calibrated student
            logger.info("Evaluating calibrated student...")
            evaluator = ArithmeticEvaluator(self.calibrated_student, self.student_tokenizer, self.device)
            post_calibration_results = evaluator.evaluate_comprehensive(
                operations=operations,
                digits_range=digits_range,
                samples_per_op=samples_per_op
            )
            
            # Save post-calibration results
            with open(os.path.join(self.output_dir, "post_calibration_results.json"), "w") as f:
                json.dump(post_calibration_results, f, indent=2)
            
            # Visualize post-calibration results
            evaluator.visualize_results(
                post_calibration_results,
                title="Calibrated Student Model Performance",
                save_path=os.path.join(self.output_dir, "post_calibration_results.png")
            )
            
            # 11. Compare results and generate summary
            summary = {
                "baseline_accuracy": baseline_results["overall_accuracy"],
                "pre_calibration_accuracy": pre_calibration_results["overall_accuracy"],
                "post_calibration_accuracy": post_calibration_results["overall_accuracy"],
                "neurons_analyzed": len(critical_neurons),
                "neurons_transferred": len(transferred_neurons),
                "absolute_improvement": post_calibration_results["overall_accuracy"] - baseline_results["overall_accuracy"],
                "relative_improvement": (post_calibration_results["overall_accuracy"] / baseline_results["overall_accuracy"]) - 1,
                "runtime_seconds": time.time() - start_time
            }
            
            with open(os.path.join(self.output_dir, "experiment_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            
            # 12. Save calibrated model (optional)
            # Only save if there was improvement and user wants to
            if summary["absolute_improvement"] > 0:
                model_save_path = os.path.join(self.output_dir, "calibrated_model")
                logger.info(f"Saving calibrated model to {model_save_path}")
                
                # Save in safetensors format for efficiency
                self.calibrated_student.save_pretrained(model_save_path, safe_serialization=True)
                self.student_tokenizer.save_pretrained(model_save_path)
            
            # Print final summary
            logger.info("\n" + "="*50)
            logger.info("EXPERIMENT RESULTS SUMMARY")
            logger.info("="*50)
            logger.info(f"Baseline accuracy: {summary['baseline_accuracy']:.4f}")
            logger.info(f"After neuron transfer (before calibration): {summary['pre_calibration_accuracy']:.4f}")
            logger.info(f"After calibration: {summary['post_calibration_accuracy']:.4f}")
            logger.info(f"Absolute improvement: {summary['absolute_improvement']:.4f}")
            logger.info(f"Relative improvement: {summary['relative_improvement']*100:.2f}%")
            logger.info(f"Total runtime: {summary['runtime_seconds']/60:.2f} minutes")
            logger.info("="*50)
            
            return {
                "config": config,
                "teacher_info": teacher_info,
                "student_info": student_info,
                "baseline_results": baseline_results,
                "pre_calibration_results": pre_calibration_results,
                "post_calibration_results": post_calibration_results,
                "critical_neurons": critical_neurons_data,
                "transferred_neurons": transferred_neurons_data,
                "summary": summary,
                "output_dir": self.output_dir
            }
        
        except Exception as e:
            logger.error(f"Experiment failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Save error information
            with open(os.path.join(self.output_dir, "experiment_error.txt"), "w") as f:
                f.write(f"Error: {str(e)}\n\n")
                f.write(traceback.format_exc())
            
            raise e


###############################################################################
# 8. Main Script
###############################################################################

def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Neuron Attribution for Arithmetic Knowledge Transfer")
    
    # Model arguments
    parser.add_argument("--teacher", type=str, default="meta-llama/Llama-13b-hf",
                       help="Teacher model name or path")
    parser.add_argument("--student", type=str, default="meta-llama/Llama-7b-hf",
                       help="Student model name or path")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory to cache models")
    
    # Experiment arguments
    parser.add_argument("--operations", type=str, default="+,-",
                       help="Comma-separated list of operations to test")
    parser.add_argument("--min-digits", type=int, default=1,
                       help="Minimum digit complexity")
    parser.add_argument("--max-digits", type=int, default=2,
                       help="Maximum digit complexity")
    parser.add_argument("--samples", type=int, default=50,
                       help="Samples per operation")
    parser.add_argument("--neurons", type=int, default=100,
                       help="Number of neurons to transfer")
    
    # Calibration arguments
    parser.add_argument("--calibration-lr", type=float, default=1e-5,
                       help="Learning rate for calibration")
    parser.add_argument("--calibration-epochs", type=int, default=1,
                       help="Number of epochs for calibration")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Parse operations
    operations = args.operations.split(",")
    
    # Create and run experiment
    experiment = ArithmeticTransferExperiment(
        teacher_model_name=args.teacher,
        student_model_name=args.student,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    results = experiment.run(
        operations=operations,
        digits_range=(args.min_digits, args.max_digits),
        samples_per_op=args.samples,
        neuron_count=args.neurons,
        calibration_lr=args.calibration_lr,
        calibration_epochs=args.calibration_epochs
    )
    
    logger.info(f"Experiment completed successfully!")
    logger.info(f"Results saved to: {results['output_dir']}")
    return results


if __name__ == "__main__":
    main()
