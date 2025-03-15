#!/usr/bin/env python
# coding: utf-8

import os
import json
import math
import random
import copy
import time
from tqdm.auto import tqdm
import numpy as np
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
    BitsAndBytesConfig
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Primary device: {device}")

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Memory cleaned. Current CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

class ArithmeticDataset(Dataset):
    def __init__(self, size=1000, digits_range=(1, 2), operations=('+',), seed=42):
        super().__init__()
        self.size = size
        self.min_digits, self.max_digits = digits_range
        self.operations = operations
        self.seed = seed
        
        random.seed(self.seed)
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        operation_names = {
            '+': "plus",
            '-': "minus",
            '*': "times",
            '/': "divided by"
        }
        
        for _ in range(self.size):
            num_digits = random.randint(self.min_digits, self.max_digits)
            
            operation = random.choice(self.operations)
            
            min_val = 10 ** (num_digits - 1)
            max_val = 10 ** num_digits - 1
            
            a = random.randint(min_val, max_val)
            
            if operation == '/':
                b = random.randint(1, min(20, a))
                a = a - (a % b)
            else:
                b = random.randint(min_val, max_val)
            
            if operation == '+':
                result = a + b
            elif operation == '-':
                if a < b:
                    a, b = b, a
                result = a - b
            elif operation == '*':
                result = a * b
            elif operation == '/':
                result = a // b
            
            formats = [
                f"Calculate {a} {operation} {b} = ",
                f"What is {a} {operation} {b}? Answer: ",
                f"{a} {operation_names[operation]} {b} equals ",
                f"Compute: {a} {operation} {b} = ",
                f"Solve: {a} {operation} {b} = "
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


def create_mistral_dataloader(dataset, tokenizer, batch_size=4, max_length=64):
    def _collate_fn(batch):
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]
        
        input_texts = []
        label_texts = []
        
        for prompt, answer in zip(prompts, answers):
            input_texts.append(prompt)
            label_texts.append(prompt + answer)
        
        input_encodings = tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        label_encodings = tokenizer(
            label_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        labels = label_encodings.input_ids.clone()
        
        for i in range(len(input_texts)):
            input_len = len(tokenizer.encode(input_texts[i])) - 2
            labels[i, :input_len] = -100  
        
        return {
            "input_ids": input_encodings.input_ids,
            "attention_mask": input_encodings.attention_mask,
            "labels": labels
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn
    )

class MistralModelManager:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            logger.warning("HUGGINGFACE_TOKEN environment variable not set. "
                          "You may encounter authentication issues when accessing gated models.")
    
    def load_model(self, model_name, use_4bit=False, device="cuda"):
        logger.info(f"Loading model: {model_name} on {device}")
        
        common_kwargs = {
            "cache_dir": self.cache_dir,
        }
        
        if self.hf_token:
            common_kwargs["token"] = self.hf_token
            logger.info("Using HuggingFace token from environment variable")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True,
            **common_kwargs
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            load_kwargs = {
                **common_kwargs,
                "quantization_config": quantization_config,
                "device_map": "auto",
            }
            logger.info("Using 4-bit quantization for memory efficiency")
        else:
            load_kwargs = {
                **common_kwargs,
                "torch_dtype": torch.float16,
                "device_map": device,
            }
        
        if torch.cuda.is_available():
            logger.info(f"Memory before loading: {torch.cuda.memory_allocated(device if isinstance(device, int) else 0) / 1e9:.2f} GB")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        if torch.cuda.is_available():
            logger.info(f"Memory after loading: {torch.cuda.memory_allocated(device if isinstance(device, int) else 0) / 1e9:.2f} GB")
            
        return model, tokenizer
    
    def get_model_size_info(self, model, tokenizer):
        config = model.config
        
        is_mixtral = hasattr(config, "num_local_experts") and config.num_local_experts > 1
        
        hidden_size = config.hidden_size
        
        if is_mixtral:
            num_experts = config.num_local_experts
            num_experts_per_token = config.num_experts_per_tok
            
            try:
                first_layer = model.model.layers[0]
                if hasattr(first_layer.mlp, "experts"):
                    first_expert = first_layer.mlp.experts.experts[0]
                    intermediate_size = first_expert.w1.weight.shape[0]
                elif hasattr(first_layer.mlp, "w1"):
                    intermediate_size = first_layer.mlp.w1.weight.shape[0]
                else:
                    intermediate_size = hidden_size * 4
            except (AttributeError, IndexError):
                intermediate_size = hidden_size * 4
                
            experts_info = {
                "num_experts": num_experts,
                "experts_per_token": num_experts_per_token
            }
        else:
            try:
                first_layer = model.model.layers[0]
                if hasattr(first_layer.mlp, "gate_proj"):
                    intermediate_size = first_layer.mlp.gate_proj.weight.shape[0]
                elif hasattr(first_layer.mlp, "up_proj"):
                    intermediate_size = first_layer.mlp.up_proj.weight.shape[0]
                else:
                    intermediate_size = hidden_size * 4
            except (AttributeError, IndexError):
                intermediate_size = hidden_size * 4
                
            experts_info = None
        
        info = {
            "model_name": model.config._name_or_path,
            "parameter_count": sum(p.numel() for p in model.parameters()) / 1_000_000,
            "layer_count": config.num_hidden_layers,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "head_dim": hidden_size // config.num_attention_heads,
            "vocab_size": len(tokenizer),
            "is_mixtral": is_mixtral,
            "experts_info": experts_info
        }
        
        return info

class MistralNeuronAnalyzer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        
        self.is_mixtral = hasattr(self.config, "num_local_experts") and self.config.num_local_experts > 1
        
        if self.is_mixtral:
            self.num_experts = self.config.num_local_experts
            self.experts_per_token = self.config.num_experts_per_tok
            logger.info(f"Detected Mixtral MoE model with {self.num_experts} experts, using {self.experts_per_token} per token")
        
        try:
            if len(model.model.layers) > 0:
                first_layer = model.model.layers[0]
                
                if self.is_mixtral:
                    if hasattr(first_layer.mlp, "experts"):
                        first_expert = first_layer.mlp.experts.experts[0]
                        if hasattr(first_expert, "w1"):
                            self.intermediate_size = first_expert.w1.weight.shape[0]
                        else:
                            self.intermediate_size = first_expert.gate_proj.weight.shape[0]
                    else:
                        self.intermediate_size = self.hidden_size * 4
                else:
                    if hasattr(first_layer.mlp, "gate_proj"):
                        self.intermediate_size = first_layer.mlp.gate_proj.weight.shape[0]
                    elif hasattr(first_layer.mlp, "up_proj"):
                        self.intermediate_size = first_layer.mlp.up_proj.weight.shape[0]
                    else:
                        self.intermediate_size = self.hidden_size * 4
            else:
                self.intermediate_size = self.hidden_size * 4
                
            logger.info(f"Detected intermediate size: {self.intermediate_size}")
        except Exception as e:
            logger.warning(f"Error detecting intermediate size: {e}")
            self.intermediate_size = self.hidden_size * 4
            logger.info(f"Using fallback intermediate size: {self.intermediate_size}")
    
    def compute_output_probability(self, text, target_token=None, intervened_neurons=None):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        if target_token is None:
            target_id = input_ids[0, -1].item()
        elif isinstance(target_token, str):
            target_encoding = self.tokenizer.encode(target_token, add_special_tokens=False)
            target_id = target_encoding[0]
        else:
            target_id = target_token
        
        if intervened_neurons and len(intervened_neurons) > 0:
            original_params = {}
            
            for layer_idx, neuron_idx, expert_idx in intervened_neurons:
                if layer_idx >= len(self.model.model.layers):
                    logger.warning(f"Layer {layer_idx} out of bounds (max {len(self.model.model.layers)-1})")
                    continue
                
                layer = self.model.model.layers[layer_idx]
                
                if self.is_mixtral:
                    if expert_idx is None or expert_idx >= self.num_experts:
                        logger.warning(f"Expert {expert_idx} out of bounds (max {self.num_experts-1})")
                        continue
                    
                    if hasattr(layer.mlp, "experts"):
                        expert = layer.mlp.experts.experts[expert_idx]
                    else:
                        logger.warning(f"Cannot access experts in layer {layer_idx}")
                        continue
                    
                    if hasattr(expert, "w1"):
                        if neuron_idx >= expert.w1.weight.shape[0]:
                            logger.warning(f"Neuron {neuron_idx} out of bounds for expert {expert_idx} (max {expert.w1.weight.shape[0]-1})")
                            continue
                        
                        original_params[(layer_idx, neuron_idx, expert_idx, 'w1')] = {
                            'weight': expert.w1.weight[neuron_idx, :].clone(),
                        }
                        
                        if hasattr(expert, "w3"):
                            original_params[(layer_idx, neuron_idx, expert_idx, 'w3')] = {
                                'weight': expert.w3.weight[neuron_idx, :].clone(),
                            }
                        
                        col_tensor = expert.w2.weight[:, neuron_idx].clone()
                        original_params[(layer_idx, neuron_idx, expert_idx, 'w2')] = {
                            'weight': col_tensor,
                        }
                    else:
                        logger.warning(f"Unexpected expert structure in layer {layer_idx}")
                        continue
                else:
                    expert_idx = None
                    
                    if hasattr(layer.mlp, "gate_proj"):
                        if neuron_idx >= layer.mlp.gate_proj.weight.shape[0]:
                            logger.warning(f"Neuron {neuron_idx} out of bounds (max {layer.mlp.gate_proj.weight.shape[0]-1})")
                            continue
                        
                        original_params[(layer_idx, neuron_idx, None, 'gate')] = {
                            'weight': layer.mlp.gate_proj.weight[neuron_idx, :].clone(),
                        }
                        
                        original_params[(layer_idx, neuron_idx, None, 'up')] = {
                            'weight': layer.mlp.up_proj.weight[neuron_idx, :].clone(),
                        }
                        
                        col_tensor = layer.mlp.down_proj.weight[:, neuron_idx].clone()
                        original_params[(layer_idx, neuron_idx, None, 'down')] = {
                            'weight': col_tensor,
                        }
                    else:
                        logger.warning(f"Unexpected MLP structure in layer {layer_idx}")
                        continue
        
        if intervened_neurons and len(intervened_neurons) > 0:
            valid_interventions = []
            
            for layer_idx, neuron_idx, expert_idx in intervened_neurons:
                if layer_idx >= len(self.model.model.layers):
                    continue
                
                try:
                    self._zero_out_neuron(layer_idx, neuron_idx, expert_idx)
                    valid_interventions.append((layer_idx, neuron_idx, expert_idx))
                except Exception as e:
                    logger.warning(f"Error zeroing out neuron ({layer_idx}, {neuron_idx}, {expert_idx}): {e}")
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            target_prob = probs[target_id].item()
        
        if intervened_neurons and len(intervened_neurons) > 0:
            with torch.no_grad():
                for (layer_idx, neuron_idx, expert_idx, proj_type), params in original_params.items():
                    layer = self.model.model.layers[layer_idx]
                    
                    if self.is_mixtral:
                        if hasattr(layer.mlp, "experts"):
                            expert = layer.mlp.experts.experts[expert_idx]
                            
                            if proj_type == 'w1':
                                weight = expert.w1.weight.clone()
                                weight[neuron_idx, :] = params['weight']
                                expert.w1.weight.copy_(weight)
                            elif proj_type == 'w3' and hasattr(expert, "w3"):
                                weight = expert.w3.weight.clone()
                                weight[neuron_idx, :] = params['weight']
                                expert.w3.weight.copy_(weight)
                            elif proj_type == 'w2':
                                weight = expert.w2.weight.clone()
                                weight[:, neuron_idx] = params['weight']
                                expert.w2.weight.copy_(weight)
                    else:
                        if proj_type == 'gate':
                            weight = layer.mlp.gate_proj.weight.clone()
                            weight[neuron_idx, :] = params['weight']
                            layer.mlp.gate_proj.weight.copy_(weight)
                        elif proj_type == 'up':
                            weight = layer.mlp.up_proj.weight.clone()
                            weight[neuron_idx, :] = params['weight']
                            layer.mlp.up_proj.weight.copy_(weight)
                        elif proj_type == 'down':
                            weight = layer.mlp.down_proj.weight.clone()
                            weight[:, neuron_idx] = params['weight']
                            layer.mlp.down_proj.weight.copy_(weight)
        
        return target_prob
    
    def _zero_out_neuron(self, layer_idx, neuron_idx, expert_idx=None):
        with torch.no_grad():
            layer = self.model.model.layers[layer_idx]
            
            if self.is_mixtral:
                if expert_idx is None:
                    raise ValueError("Expert index must be provided for Mixtral models")
                
                if hasattr(layer.mlp, "experts"):
                    expert = layer.mlp.experts.experts[expert_idx]
                    
                    if hasattr(expert, "w1"):
                        weight = expert.w1.weight.clone()
                        weight[neuron_idx, :] = 0
                        expert.w1.weight.copy_(weight)
                        
                        if hasattr(expert, "w3"):
                            weight = expert.w3.weight.clone()
                            weight[neuron_idx, :] = 0
                            expert.w3.weight.copy_(weight)
                        
                        weight = expert.w2.weight.clone()
                        weight[:, neuron_idx] = 0
                        expert.w2.weight.copy_(weight)
                    else:
                        raise ValueError(f"Unexpected expert structure in layer {layer_idx}")
                else:
                    raise ValueError(f"Cannot access experts in layer {layer_idx}")
            else:
                if hasattr(layer.mlp, "gate_proj"):
                    weight = layer.mlp.gate_proj.weight.clone()
                    weight[neuron_idx, :] = 0
                    layer.mlp.gate_proj.weight.copy_(weight)
                    
                    weight = layer.mlp.up_proj.weight.clone()
                    weight[neuron_idx, :] = 0
                    layer.mlp.up_proj.weight.copy_(weight)
                    
                    weight = layer.mlp.down_proj.weight.clone()
                    weight[:, neuron_idx] = 0
                    layer.mlp.down_proj.weight.copy_(weight)
                else:
                    raise ValueError(f"Unexpected MLP structure in layer {layer_idx}")
    
    def compute_neuron_importance(self, text, target_token=None, layer_range=None, top_k=100, batch_size=50):
        if layer_range is None:
            layer_range = (0, min(8, self.num_layers))
        
        clean_memory()
        
        baseline_prob = self.compute_output_probability(text, target_token)
        logger.info(f"Baseline probability: {baseline_prob:.6f}")
        
        if baseline_prob < 1e-5:
            logger.warning(f"Baseline probability too low ({baseline_prob:.6f}) for prompt: '{text}'")
            alternative_formats = [
                f"Calculate: {text}",
                f"Answer: {text}",
                f"Compute: {text}",
                f"Solve: {text}"
            ]
            
            for alt_format in alternative_formats:
                alt_baseline = self.compute_output_probability(alt_format, target_token)
                logger.info(f"Alternative format '{alt_format}' probability: {alt_baseline:.6f}")
                
                if alt_baseline > baseline_prob:
                    logger.info(f"Using alternative format: '{alt_format}'")
                    text = alt_format
                    baseline_prob = alt_baseline
                    break
        
        if baseline_prob < 1e-5:
            logger.warning("All baseline probabilities too low. Using artificial threshold for comparison.")
            baseline_prob = 0.01
        
        neuron_scores = []
        
        for layer_idx in tqdm(range(layer_range[0], min(layer_range[1], self.num_layers)), desc="Analyzing layers"):
            clean_memory()
            
            if layer_idx >= len(self.model.model.layers):
                logger.warning(f"Layer {layer_idx} out of bounds (max {len(self.model.model.layers)-1})")
                continue
            
            layer = self.model.model.layers[layer_idx]
            
            if self.is_mixtral:
                for expert_idx in range(self.num_experts):
                    if not hasattr(layer.mlp, "experts"):
                        logger.warning(f"Cannot access experts in layer {layer_idx}")
                        continue
                    
                    expert = layer.mlp.experts.experts[expert_idx]
                    
                    if hasattr(expert, "w1"):
                        max_neurons = expert.w1.weight.shape[0]
                    else:
                        logger.warning(f"Unexpected expert structure in layer {layer_idx}")
                        continue
                    
                    batch_indices = list(range(0, max_neurons, batch_size))
                    for start_idx in tqdm(batch_indices, desc=f"Screening layer {layer_idx}, expert {expert_idx}", leave=False):
                        end_idx = min(start_idx + batch_size, max_neurons)
                        
                        batch_neurons = [(layer_idx, i, expert_idx) for i in range(start_idx, end_idx)]
                        
                        batch_prob = self.compute_output_probability(text, target_token, batch_neurons)
                        
                        batch_importance = baseline_prob - batch_prob
                        
                        if batch_importance > 0.001:
                            for neuron_idx in range(start_idx, end_idx):
                                neuron_prob = self.compute_output_probability(
                                    text, target_token, [(layer_idx, neuron_idx, expert_idx)]
                                )
                                
                                importance = baseline_prob - neuron_prob
                                
                                if importance > 0:
                                    neuron_scores.append((layer_idx, neuron_idx, expert_idx, importance))
            else:
                if hasattr(layer.mlp, "gate_proj"):
                    max_neurons = layer.mlp.gate_proj.weight.shape[0]
                else:
                    logger.warning(f"Unexpected MLP structure in layer {layer_idx}")
                    continue
                
                batch_indices = list(range(0, max_neurons, batch_size))
                for start_idx in tqdm(batch_indices, desc=f"Screening layer {layer_idx}", leave=False):
                    end_idx = min(start_idx + batch_size, max_neurons)
                    
                    batch_neurons = [(layer_idx, i, None) for i in range(start_idx, end_idx)]
                    
                    batch_prob = self.compute_output_probability(text, target_token, batch_neurons)
                    
                    batch_importance = baseline_prob - batch_prob
                    
                    if batch_importance > 0.001:
                        for neuron_idx in range(start_idx, end_idx):
                            neuron_prob = self.compute_output_probability(
                                text, target_token, [(layer_idx, neuron_idx, None)]
                            )
                            
                            importance = baseline_prob - neuron_prob
                            
                            if importance > 0:
                                neuron_scores.append((layer_idx, neuron_idx, None, importance))
            
            clean_memory()
        
        neuron_scores.sort(key=lambda x: x[3], reverse=True)
        
        if len(neuron_scores) < min(20, top_k):
            logger.warning(f"Only found {len(neuron_scores)} critical neurons, which is less than desired {min(20, top_k)}")
            logger.info("Adding fallback neurons from early layers...")
            
            existing_neurons = set((l, n, e) for l, n, e, _ in neuron_scores)
            
            for layer_idx in range(min(3, self.num_layers)):
                if layer_idx >= len(self.model.model.layers):
                    continue
                
                layer = self.model.model.layers[layer_idx]
                
                if self.is_mixtral:
                    for expert_idx in range(self.num_experts):
                        if not hasattr(layer.mlp, "experts"):
                            continue
                        
                        expert = layer.mlp.experts.experts[expert_idx]
                        
                        if hasattr(expert, "w1"):
                            max_neurons = expert.w1.weight.shape[0]
                            
                            for neuron_idx in range(0, max_neurons, 10):
                                if (layer_idx, neuron_idx, expert_idx) not in existing_neurons:
                                    neuron_scores.append((layer_idx, neuron_idx, expert_idx, 0.0005))
                                    existing_neurons.add((layer_idx, neuron_idx, expert_idx))
                                    
                                    if len(neuron_scores) >= top_k:
                                        break
                        
                        if len(neuron_scores) >= top_k:
                            break
                else:
                    if hasattr(layer.mlp, "gate_proj"):
                        max_neurons = layer.mlp.gate_proj.weight.shape[0]
                        
                        for neuron_idx in range(0, max_neurons, 10):
                            if (layer_idx, neuron_idx, None) not in existing_neurons:
                                neuron_scores.append((layer_idx, neuron_idx, None, 0.0005))
                                existing_neurons.add((layer_idx, neuron_idx, None))
                                
                                if len(neuron_scores) >= top_k:
                                    break
                
                if len(neuron_scores) >= top_k:
                    break
            
            neuron_scores.sort(key=lambda x: x[3], reverse=True)
        
        result = []
        for layer_idx, neuron_idx, expert_idx, importance in neuron_scores[:top_k]:
            if self.is_mixtral:
                result.append((layer_idx, neuron_idx, expert_idx, importance))
            else:
                result.append((layer_idx, neuron_idx, importance))
        
        clean_memory()
        return result[:top_k]
    
    def analyze_arithmetic_neurons(self, operations=('+',), digits_range=(1, 1), 
                                  samples_per_op=3, top_k=100, layer_range=None):
        clean_memory()
        
        dataset = ArithmeticDataset(
            size=samples_per_op * len(operations), 
            digits_range=digits_range,
            operations=operations
        )
        
        all_critical_neurons = []
        
        for item in tqdm(dataset.samples, desc="Analyzing arithmetic problems"):
            prompt = item["prompt"]
            answer = item["answer"]
            
            logger.info(f"\nAnalyzing: {prompt}{answer} (Operation: {item['operation']})")
            
            clean_memory()
            
            critical_neurons = self.compute_neuron_importance(
                prompt, answer[0], layer_range=layer_range, 
                top_k=min(top_k // len(dataset.samples), 20)
            )
            
            all_critical_neurons.extend(critical_neurons)
            
            logger.info(f"Top 5 neurons for this example:")
            for i, neuron_info in enumerate(critical_neurons[:5]):
                if self.is_mixtral:
                    layer, neuron, expert, importance = neuron_info
                    logger.info(f"  {i+1}. Layer {layer}, Neuron {neuron}, Expert {expert}: {importance:.6f}")
                else:
                    layer, neuron, importance = neuron_info
                    logger.info(f"  {i+1}. Layer {layer}, Neuron {neuron}: {importance:.6f}")
        
        neuron_dict = {}
        for neuron_info in all_critical_neurons:
            if self.is_mixtral:
                layer, neuron, expert, importance = neuron_info
                key = (layer, neuron, expert)
            else:
                layer, neuron, importance = neuron_info
                key = (layer, neuron, None)
                
            neuron_dict[key] = neuron_dict.get(key, 0) + importance
        
        critical_neurons = []
        for (layer, neuron, expert), importance in neuron_dict.items():
            if self.is_mixtral:
                critical_neurons.append((layer, neuron, expert, importance))
            else:
                critical_neurons.append((layer, neuron, importance))
                
        critical_neurons.sort(key=lambda x: x[-1], reverse=True)
        
        clean_memory()
        return critical_neurons[:top_k]

    def evaluate_arithmetic_accuracy(self, dataset, max_batch_size=4):
        self.model.eval()
        
        clean_memory()
        
        correct = 0
        total = 0
        
        operation_correct = {}
        operation_total = {}
        
        digits_correct = {}
        digits_total = {}
        
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
                
                if operation not in operation_total:
                    operation_total[operation] = 0
                    operation_correct[operation] = 0
                operation_total[operation] += 1
                
                if digits not in digits_total:
                    digits_total[digits] = 0
                    digits_correct[digits] = 0
                digits_total[digits] += 1
                
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=inputs.input_ids,
                            max_new_tokens=len(expected) + 2,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    generated = outputs[0, inputs.input_ids.shape[1]:]
                    prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                    
                    is_correct = expected in prediction
                    
                    if is_correct:
                        correct += 1
                        operation_correct[operation] += 1
                        digits_correct[digits] += 1
                except Exception as e:
                    logger.warning(f"Generation failed for prompt '{prompt}': {str(e)}")
                    pass
                
                total += 1
                
            if total % 20 == 0:
                clean_memory()
        
        accuracy = correct / total if total > 0 else 0
        
        operation_accuracy = {
            op: operation_correct.get(op, 0) / cnt 
            for op, cnt in operation_total.items()
        }
        
        digits_accuracy = {
            d: digits_correct.get(d, 0) / cnt 
            for d, cnt in digits_total.items()
        }
        
        clean_memory()
        return {
            "overall_accuracy": accuracy,
            "operation_accuracy": operation_accuracy,
            "digits_accuracy": digits_accuracy,
            "total_examples": total,
            "total_correct": correct
        }

class MistralNeuronTransfer:
    def __init__(self, teacher_model, student_model, teacher_device="cuda:0", student_device="cuda:1"):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_device = teacher_device
        self.student_device = student_device
        
        self.teacher_config = teacher_model.config
        self.student_config = student_model.config
        
        self.teacher_layers = self.teacher_config.num_hidden_layers
        self.student_layers = self.student_config.num_hidden_layers
        
        self.teacher_is_mixtral = hasattr(self.teacher_config, "num_local_experts") and self.teacher_config.num_local_experts > 1
        if self.teacher_is_mixtral:
            self.teacher_num_experts = self.teacher_config.num_local_experts
            logger.info(f"Teacher is Mixtral with {self.teacher_num_experts} experts per layer")
        
        self.student_is_mixtral = hasattr(self.student_config, "num_local_experts") and self.student_config.num_local_experts > 1
        if self.student_is_mixtral:
            self.student_num_experts = self.student_config.num_local_experts
            logger.info(f"Student is Mixtral with {self.student_num_experts} experts per layer")
        
        self.teacher_hidden_size = self.teacher_config.hidden_size
        self.student_hidden_size = self.student_config.hidden_size
        
        try:
            if len(teacher_model.model.layers) > 0:
                first_layer = teacher_model.model.layers[0]
                
                if self.teacher_is_mixtral:
                    if hasattr(first_layer.mlp, "experts"):
                        first_expert = first_layer.mlp.experts.experts[0]
                        if hasattr(first_expert, "w1"):
                            self.teacher_intermediate_size = first_expert.w1.weight.shape[0]
                        else:
                            self.teacher_intermediate_size = first_expert.gate_proj.weight.shape[0]
                    else:
                        self.teacher_intermediate_size = self.teacher_hidden_size * 4
                else:
                    if hasattr(first_layer.mlp, "gate_proj"):
                        self.teacher_intermediate_size = first_layer.mlp.gate_proj.weight.shape[0]
                    elif hasattr(first_layer.mlp, "up_proj"):
                        self.teacher_intermediate_size = first_layer.mlp.up_proj.weight.shape[0]
                    else:
                        self.teacher_intermediate_size = self.teacher_hidden_size * 4
            else:
                self.teacher_intermediate_size = self.teacher_hidden_size * 4
                
            if len(student_model.model.layers) > 0:
                first_layer = student_model.model.layers[0]
                
                if self.student_is_mixtral:
                    if hasattr(first_layer.mlp, "experts"):
                        first_expert = first_layer.mlp.experts.experts[0]
                        if hasattr(first_expert, "w1"):
                            self.student_intermediate_size = first_expert.w1.weight.shape[0]
                        else:
                            self.student_intermediate_size = first_expert.gate_proj.weight.shape[0]
                    else:
                        self.student_intermediate_size = self.student_hidden_size * 4
                else:
                    if hasattr(first_layer.mlp, "gate_proj"):
                        self.student_intermediate_size = first_layer.mlp.gate_proj.weight.shape[0]
                    elif hasattr(first_layer.mlp, "up_proj"):
                        self.student_intermediate_size = first_layer.mlp.up_proj.weight.shape[0]
                    else:
                        self.student_intermediate_size = self.student_hidden_size * 4
            else:
                self.student_intermediate_size = self.student_hidden_size * 4
        except Exception as e:
            logger.warning(f"Error detecting intermediate sizes: {e}")
            self.teacher_intermediate_size = self.teacher_hidden_size * 4
            self.student_intermediate_size = self.student_hidden_size * 4
        
        self.layer_mapping = self._create_layer_mapping()
        
        logger.info(f"Teacher model: {self.teacher_layers} layers, {self.teacher_intermediate_size} neurons per layer, hidden size {self.teacher_hidden_size}")
        logger.info(f"Student model: {self.student_layers} layers, {self.student_intermediate_size} neurons per layer, hidden size {self.student_hidden_size}")
    
    def _create_layer_mapping(self):
        mapping = {}
        for teacher_layer in range(self.teacher_layers):
            student_layer = min(int(teacher_layer * self.student_layers / self.teacher_layers), 
                               self.student_layers - 1)
            mapping[teacher_layer] = student_layer
        return mapping
    
    def transfer_neurons(self, critical_neurons):
        clean_memory()
        
        modified_student = copy.deepcopy(self.student_model)
        
        transferred_neurons = []
        
        if not critical_neurons:
            logger.error("No critical neurons provided for transfer!")
            logger.info("Using fallback neurons from early layers...")
            critical_neurons = []
            
            for layer_idx in range(min(4, self.teacher_layers)):
                if layer_idx >= len(self.teacher_model.model.layers):
                    continue
                
                for neuron_idx in range(0, min(500, self.teacher_intermediate_size), 5):
                    if self.teacher_is_mixtral:
                        for expert_idx in range(min(3, self.teacher_num_experts)):
                            critical_neurons.append((layer_idx, neuron_idx, expert_idx, 0.001))
                    else:
                        critical_neurons.append((layer_idx, neuron_idx, 0.001))
                        
                    if len(critical_neurons) >= 100:
                        break
                
                if len(critical_neurons) >= 100:
                    break
        
        logger.info(f"Transferring {len(critical_neurons)} neurons from teacher to student...")
        
        for i, neuron_info in enumerate(tqdm(critical_neurons, desc="Transferring neurons")):
            if self.teacher_is_mixtral:
                layer_idx, neuron_idx, expert_idx, importance = neuron_info
            else:
                layer_idx, neuron_idx, importance = neuron_info
                expert_idx = None
            
            student_layer_idx = self.layer_mapping.get(layer_idx)
            
            if student_layer_idx is None:
                logger.warning(f"No mapping found for teacher layer {layer_idx}. Skipping...")
                continue
            
            if layer_idx >= len(self.teacher_model.model.layers):
                logger.warning(f"Teacher layer {layer_idx} out of bounds. Skipping...")
                continue
                
            if student_layer_idx >= len(modified_student.model.layers):
                logger.warning(f"Student layer {student_layer_idx} out of bounds. Skipping...")
                continue
            
            teacher_layer = self.teacher_model.model.layers[layer_idx]
            student_layer = modified_student.model.layers[student_layer_idx]
            
            if self.teacher_is_mixtral:
                if not hasattr(teacher_layer.mlp, "experts"):
                    logger.warning(f"Teacher layer {layer_idx} has no experts attribute. Skipping...")
                    continue
                
                if expert_idx >= len(teacher_layer.mlp.experts.experts):
                    logger.warning(f"Teacher expert {expert_idx} out of bounds. Skipping...")
                    continue
                
                teacher_expert = teacher_layer.mlp.experts.experts[expert_idx]
                
                if not hasattr(teacher_expert, "w1") or not hasattr(teacher_expert, "w2"):
                    logger.warning(f"Teacher expert has unexpected structure. Skipping...")
                    continue
                
                if neuron_idx >= teacher_expert.w1.weight.shape[0]:
                    logger.warning(f"Neuron index {neuron_idx} out of bounds for teacher. Skipping...")
                    continue
            else:
                if not hasattr(teacher_layer.mlp, "gate_proj"):
                    logger.warning(f"Teacher layer {layer_idx} has unexpected MLP structure. Skipping...")
                    continue
                
                if neuron_idx >= teacher_layer.mlp.gate_proj.weight.shape[0]:
                    logger.warning(f"Neuron index {neuron_idx} out of bounds for teacher. Skipping...")
                    continue
            
            student_expert_idx = 0
            
            if self.student_is_mixtral:
                if not hasattr(student_layer.mlp, "experts"):
                    logger.warning(f"Student layer {student_layer_idx} has no experts attribute. Skipping...")
                    continue
                
                if self.teacher_is_mixtral and expert_idx < len(student_layer.mlp.experts.experts):
                    student_expert_idx = expert_idx
                
                if student_expert_idx >= len(student_layer.mlp.experts.experts):
                    logger.warning(f"Student expert {student_expert_idx} out of bounds. Skipping...")
                    continue
                
                student_expert = student_layer.mlp.experts.experts[student_expert_idx]
                
                if not hasattr(student_expert, "w1") or not hasattr(student_expert, "w2"):
                    logger.warning(f"Student expert has unexpected structure. Skipping...")
                    continue
                
                if neuron_idx >= student_expert.w1.weight.shape[0]:
                    logger.warning(f"Neuron index {neuron_idx} out of bounds for student. Skipping...")
                    continue
            else:
                if not hasattr(student_layer.mlp, "gate_proj"):
                    logger.warning(f"Student layer {student_layer_idx} has unexpected MLP structure. Skipping...")
                    continue
                
                if neuron_idx >= student_layer.mlp.gate_proj.weight.shape[0]:
                    logger.warning(f"Neuron index {neuron_idx} out of bounds for student. Skipping...")
                    continue
            
            try:
                with torch.no_grad():
                    if self.teacher_is_mixtral and self.student_is_mixtral:
                        self._transfer_moe_to_moe(
                            teacher_layer, student_layer, 
                            neuron_idx, 
                            teacher_expert_idx=expert_idx, 
                            student_expert_idx=student_expert_idx
                        )
                    elif self.teacher_is_mixtral and not self.student_is_mixtral:
                        self._transfer_moe_to_dense(
                            teacher_layer, student_layer,
                            neuron_idx,
                            teacher_expert_idx=expert_idx
                        )
                    elif not self.teacher_is_mixtral and self.student_is_mixtral:
                        self._transfer_dense_to_moe(
                            teacher_layer, student_layer,
                            neuron_idx,
                            student_expert_idx=student_expert_idx
                        )
                    else:
                        self._transfer_dense_to_dense(
                            teacher_layer, student_layer,
                            neuron_idx
                        )
                
                if self.student_is_mixtral:
                    transferred_neurons.append((student_layer_idx, neuron_idx, student_expert_idx, importance))
                else:
                    transferred_neurons.append((student_layer_idx, neuron_idx, importance))
            except Exception as e:
                logger.error(f"Error transferring neuron: {e}")
                continue
            
            if (i+1) % 10 == 0:
                clean_memory()
        
        logger.info(f"Successfully transferred {len(transferred_neurons)} neurons")
        
        if len(transferred_neurons) == 0:
            logger.error("No neurons were successfully transferred!")
            logger.warning("Using unmodified student model")
            return self.student_model, []
        
        clean_memory()
        return modified_student, transferred_neurons
    
    def _transfer_moe_to_moe(self, teacher_layer, student_layer, neuron_idx, 
                            teacher_expert_idx, student_expert_idx):
        teacher_expert = teacher_layer.mlp.experts.experts[teacher_expert_idx]
        student_expert = student_layer.mlp.experts.experts[student_expert_idx]
        
        teacher_weight = teacher_expert.w1.weight[neuron_idx, :].to(self.student_device)
        student_weight = student_expert.w1.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[1]:
            if teacher_weight.shape[0] > student_weight.shape[1]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[1]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[1]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[1], device=self.student_device)
                start_idx = (student_weight.shape[1] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[neuron_idx, :] = adapted_weight
        student_expert.w1.weight.copy_(student_weight)
        
        if hasattr(teacher_expert, "w3") and hasattr(student_expert, "w3"):
            teacher_weight = teacher_expert.w3.weight[neuron_idx, :].to(self.student_device)
            student_weight = student_expert.w3.weight.clone()
            
            if teacher_weight.shape[0] != student_weight.shape[1]:
                if teacher_weight.shape[0] > student_weight.shape[1]:
                    start_idx = (teacher_weight.shape[0] - student_weight.shape[1]) // 2
                    adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[1]]
                else:
                    adapted_weight = torch.zeros(student_weight.shape[1], device=self.student_device)
                    start_idx = (student_weight.shape[1] - teacher_weight.shape[0]) // 2
                    adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
            else:
                adapted_weight = teacher_weight
            
            student_weight[neuron_idx, :] = adapted_weight
            student_expert.w3.weight.copy_(student_weight)
        
        teacher_weight = teacher_expert.w2.weight[:, neuron_idx].to(self.student_device)
        student_weight = student_expert.w2.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[0]:
            if teacher_weight.shape[0] > student_weight.shape[0]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[0]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[0]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[0], device=self.student_device)
                start_idx = (student_weight.shape[0] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[:, neuron_idx] = adapted_weight
        student_expert.w2.weight.copy_(student_weight)
    
    def _transfer_moe_to_dense(self, teacher_layer, student_layer, neuron_idx, teacher_expert_idx):
        teacher_expert = teacher_layer.mlp.experts.experts[teacher_expert_idx]
        
        teacher_weight = teacher_expert.w1.weight[neuron_idx, :].to(self.student_device)
        student_weight = student_layer.mlp.gate_proj.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[1]:
            if teacher_weight.shape[0] > student_weight.shape[1]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[1]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[1]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[1], device=self.student_device)
                start_idx = (student_weight.shape[1] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[neuron_idx, :] = adapted_weight
        student_layer.mlp.gate_proj.weight.copy_(student_weight)
        
        if hasattr(teacher_expert, "w3"):
            teacher_weight = teacher_expert.w3.weight[neuron_idx, :].to(self.student_device)
        else:
            teacher_weight = teacher_expert.w1.weight[neuron_idx, :].to(self.student_device)
            
        student_weight = student_layer.mlp.up_proj.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[1]:
            if teacher_weight.shape[0] > student_weight.shape[1]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[1]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[1]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[1], device=self.student_device)
                start_idx = (student_weight.shape[1] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[neuron_idx, :] = adapted_weight
        student_layer.mlp.up_proj.weight.copy_(student_weight)
        
        teacher_weight = teacher_expert.w2.weight[:, neuron_idx].to(self.student_device)
        student_weight = student_layer.mlp.down_proj.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[0]:
            if teacher_weight.shape[0] > student_weight.shape[0]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[0]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[0]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[0], device=self.student_device)
                start_idx = (student_weight.shape[0] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[:, neuron_idx] = adapted_weight
        student_layer.mlp.down_proj.weight.copy_(student_weight)
    
    def _transfer_dense_to_moe(self, teacher_layer, student_layer, neuron_idx, student_expert_idx):
        student_expert = student_layer.mlp.experts.experts[student_expert_idx]
        
        teacher_weight = teacher_layer.mlp.gate_proj.weight[neuron_idx, :].to(self.student_device)
        student_weight = student_expert.w1.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[1]:
            if teacher_weight.shape[0] > student_weight.shape[1]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[1]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[1]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[1], device=self.student_device)
                start_idx = (student_weight.shape[1] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[neuron_idx, :] = adapted_weight
        student_expert.w1.weight.copy_(student_weight)
        
        if hasattr(student_expert, "w3"):
            teacher_weight = teacher_layer.mlp.up_proj.weight[neuron_idx, :].to(self.student_device)
            student_weight = student_expert.w3.weight.clone()
            
            if teacher_weight.shape[0] != student_weight.shape[1]:
                if teacher_weight.shape[0] > student_weight.shape[1]:
                    start_idx = (teacher_weight.shape[0] - student_weight.shape[1]) // 2
                    adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[1]]
                else:
                    adapted_weight = torch.zeros(student_weight.shape[1], device=self.student_device)
                    start_idx = (student_weight.shape[1] - teacher_weight.shape[0]) // 2
                    adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
            else:
                adapted_weight = teacher_weight
            
            student_weight[neuron_idx, :] = adapted_weight
            student_expert.w3.weight.copy_(student_weight)
        
        teacher_weight = teacher_layer.mlp.down_proj.weight[:, neuron_idx].to(self.student_device)
        student_weight = student_expert.w2.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[0]:
            if teacher_weight.shape[0] > student_weight.shape[0]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[0]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[0]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[0], device=self.student_device)
                start_idx = (student_weight.shape[0] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[:, neuron_idx] = adapted_weight
        student_expert.w2.weight.copy_(student_weight)
    
    def _transfer_dense_to_dense(self, teacher_layer, student_layer, neuron_idx):
        teacher_weight = teacher_layer.mlp.gate_proj.weight[neuron_idx, :].to(self.student_device)
        student_weight = student_layer.mlp.gate_proj.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[1]:
            if teacher_weight.shape[0] > student_weight.shape[1]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[1]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[1]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[1], device=self.student_device)
                start_idx = (student_weight.shape[1] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[neuron_idx, :] = adapted_weight
        student_layer.mlp.gate_proj.weight.copy_(student_weight)
        
        teacher_weight = teacher_layer.mlp.up_proj.weight[neuron_idx, :].to(self.student_device)
        student_weight = student_layer.mlp.up_proj.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[1]:
            if teacher_weight.shape[0] > student_weight.shape[1]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[1]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[1]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[1], device=self.student_device)
                start_idx = (student_weight.shape[1] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[neuron_idx, :] = adapted_weight
        student_layer.mlp.up_proj.weight.copy_(student_weight)
        
        teacher_weight = teacher_layer.mlp.down_proj.weight[:, neuron_idx].to(self.student_device)
        student_weight = student_layer.mlp.down_proj.weight.clone()
        
        if teacher_weight.shape[0] != student_weight.shape[0]:
            if teacher_weight.shape[0] > student_weight.shape[0]:
                start_idx = (teacher_weight.shape[0] - student_weight.shape[0]) // 2
                adapted_weight = teacher_weight[start_idx:start_idx+student_weight.shape[0]]
            else:
                adapted_weight = torch.zeros(student_weight.shape[0], device=self.student_device)
                start_idx = (student_weight.shape[0] - teacher_weight.shape[0]) // 2
                adapted_weight[start_idx:start_idx+teacher_weight.shape[0]] = teacher_weight
        else:
            adapted_weight = teacher_weight
        
        student_weight[:, neuron_idx] = adapted_weight
        student_layer.mlp.down_proj.weight.copy_(student_weight)

class MistralCalibration:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def calibrate(self, dataset, learning_rate=5e-7, num_epochs=1, batch_size=1):
        logger.info(f"Starting memory-efficient calibration with lr={learning_rate}")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        
        clean_memory()
        
        loader = create_mistral_dataloader(dataset, self.tokenizer, batch_size=batch_size, max_length=32)
        
        initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for param in self.model.parameters():
            param.requires_grad = True
        
        try:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=0.5, 
            total_iters=len(loader) * num_epochs
        )
        
        accumulation_steps = 4
        
        self.model.train()
        total_loss = 0
        total_batches = 0
        nan_count = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            valid_batches = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Calibration epoch {epoch+1}/{num_epochs}")):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = self.model(**batch)
                        loss = outputs.loss / accumulation_steps
                    
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        logger.warning(f"NaN/Inf loss encountered in batch {batch_idx}")
                        nan_count += 1
                        if nan_count >= 3:
                            logger.error("Too many NaN losses, restoring to initial state")
                            self.model.load_state_dict(initial_model_state)
                            self.model.eval()
                            for param in self.model.parameters():
                                param.requires_grad = False
                            clean_memory()
                            return self.model
                        continue
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(loader) - 1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * accumulation_steps
                    valid_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"OOM in batch {batch_idx}: {str(e)}")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        
                        if batch_idx < 5:
                            logger.warning("OOM occurred early. Using CPU offloading instead.")
                            break
                    else:
                        logger.error(f"Error during calibration: {str(e)}")
                        if valid_batches < 3:
                            logger.error("Too many errors. Stopping calibration.")
                            self.model.load_state_dict(initial_model_state)
                            break
                
                if batch_idx % 10 == 0:
                    clean_memory()
            
            if valid_batches > 0:
                avg_epoch_loss = epoch_loss / valid_batches
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Average loss: {avg_epoch_loss:.4f}")
                total_loss += epoch_loss
                total_batches += valid_batches
            else:
                logger.warning(f"Epoch {epoch+1}/{num_epochs} had no valid batches")
                self.model.load_state_dict(initial_model_state)
                break
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        try:
            logger.info("Validating calibrated model...")
            test_input = self.tokenizer("2 + 2 = ", return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=test_input.input_ids,
                    max_new_tokens=5,
                    do_sample=False
                )
                
            prediction = self.tokenizer.decode(outputs[0, test_input.input_ids.shape[1]:], skip_special_tokens=True)
            logger.info(f"Validation test: '2 + 2 = ' -> '{prediction}'")
            
            if not any(str(digit) in prediction for digit in range(10)):
                logger.warning("Model validation failed: no digits in output. Restoring original model.")
                self.model.load_state_dict(initial_model_state)
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}. Restoring original model.")
            self.model.load_state_dict(initial_model_state)
        
        clean_memory()
        return self.model

class ArithmeticEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.analyzer = MistralNeuronAnalyzer(model, tokenizer, device)
    
    def evaluate_comprehensive(self, operations=('+', '-', '*', '/'), digits_range=(1, 2), samples_per_op=50):
        logger.info(f"Running comprehensive evaluation...")
        
        clean_memory()
        
        dataset = ArithmeticDataset(
            size=samples_per_op * len(operations),
            digits_range=digits_range,
            operations=operations
        )
        
        self.model.eval()
        
        try:
            logger.info("Validating model before evaluation...")
            test_input = self.tokenizer("2 + 2 = ", return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=test_input.input_ids,
                    max_new_tokens=5,
                    do_sample=False
                )
                
            prediction = self.tokenizer.decode(outputs[0, test_input.input_ids.shape[1]:], skip_special_tokens=True)
            logger.info(f"Basic test: '2 + 2 = ' -> '{prediction}'")
            
            if not any(str(digit) in prediction for digit in range(10)):
                logger.warning("Model may have issues with arithmetic: no digits in test output")
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            logger.error("Cannot proceed with evaluation!")
            return {
                "overall_accuracy": 0.0,
                "operation_accuracy": {op: 0.0 for op in operations},
                "digits_accuracy": {d: 0.0 for d in range(digits_range[0], digits_range[1] + 1)},
                "total_examples": 0,
                "total_correct": 0,
                "error": str(e)
            }
        
        try:
            results = self.analyzer.evaluate_arithmetic_accuracy(dataset)
            
            logger.info(f"Overall accuracy: {results['overall_accuracy']:.4f}")
            logger.info("Accuracy by operation:")
            for op, acc in results['operation_accuracy'].items():
                logger.info(f"  {op}: {acc:.4f}")
            logger.info("Accuracy by digits:")
            for digits, acc in results['digits_accuracy'].items():
                logger.info(f"  {digits}-digit: {acc:.4f}")
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            results = {
                "overall_accuracy": 0.0,
                "operation_accuracy": {op: 0.0 for op in operations},
                "digits_accuracy": {d: 0.0 for d in range(digits_range[0], digits_range[1] + 1)},
                "total_examples": 0,
                "total_correct": 0,
                "error": str(e)
            }
        
        clean_memory()
        return results

class ArithmeticTransferExperiment:
    def __init__(self, 
                teacher_model_name="mistralai/Mixtral-8x7B-v0.1", 
                student_model_name="mistralai/Mistral-7B-v0.1",
                output_dir=None, 
                cache_dir=None,
                teacher_device="cuda:0",
                student_device="cuda:1",
                hf_token=None):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.teacher_device = teacher_device
        self.student_device = student_device
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"mistral_arithmetic_transfer_experiment_{timestamp}"
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "experiment.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        
        if hf_token:
            os.environ["HUGGINGFACE_TOKEN"] = hf_token
        
        self.model_manager = MistralModelManager(cache_dir=cache_dir)
        
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
           calibration_lr=5e-7,
           calibration_epochs=1):
        start_time = time.time()
        
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
            "teacher_device": self.teacher_device,
            "student_device": self.student_device,
            "huggingface_token_provided": "HUGGINGFACE_TOKEN" in os.environ
        }
        
        with open(os.path.join(self.output_dir, "experiment_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Starting experiment with config: {config}")
        
        try:
            clean_memory()
            
            logger.info(f"Loading teacher model: {self.teacher_model_name} on {self.teacher_device}")
            use_4bit = True if "mixtral" in self.teacher_model_name.lower() else False
            self.teacher_model, self.teacher_tokenizer = self.model_manager.load_model(
                self.teacher_model_name, use_4bit=use_4bit, device=self.teacher_device
            )
            
            clean_memory()
            
            logger.info(f"Loading student model: {self.student_model_name} on {self.student_device}")
            self.student_model, self.student_tokenizer = self.model_manager.load_model(
                self.student_model_name, use_4bit=False, device=self.student_device
            )
            
            clean_memory()
            
            teacher_info = self.model_manager.get_model_size_info(self.teacher_model, self.teacher_tokenizer)
            student_info = self.model_manager.get_model_size_info(self.student_model, self.student_tokenizer)
            
            logger.info(f"Teacher model info: {teacher_info}")
            logger.info(f"Student model info: {student_info}")
            
            with open(os.path.join(self.output_dir, "model_info.json"), "w") as f:
                json.dump({"teacher": teacher_info, "student": student_info}, f, indent=2)
            
            logger.info("Evaluating baseline student performance...")
            evaluator = ArithmeticEvaluator(self.student_model, self.student_tokenizer, self.student_device)
            baseline_results = evaluator.evaluate_comprehensive(
                operations=operations,
                digits_range=digits_range,
                samples_per_op=samples_per_op
            )
            
            with open(os.path.join(self.output_dir, "baseline_results.json"), "w") as f:
                json.dump(baseline_results, f, indent=2)
            
            logger.info("Identifying critical neurons in teacher model...")
            analyzer = MistralNeuronAnalyzer(self.teacher_model, self.teacher_tokenizer, self.teacher_device)
            
            clean_memory()
            
            critical_neurons = analyzer.analyze_arithmetic_neurons(
                operations=operations,
                digits_range=digits_range,
                samples_per_op=min(3, samples_per_op),
                top_k=neuron_count,
                layer_range=(0, 8)
            )
            
            if len(critical_neurons) < min(20, neuron_count):
                logger.warning(f"Only found {len(critical_neurons)} critical neurons, less than desired {min(20, neuron_count)}")
                logger.info("Adding fallback neurons from early layers...")
                
                is_mixtral = teacher_info.get("is_mixtral", False)
                
                existing_neurons = set()
                if is_mixtral:
                    existing_neurons = set((l, n, e) for l, n, e, _ in critical_neurons)
                else:
                    existing_neurons = set((l, n) for l, n, _ in critical_neurons)
                
                for layer_idx in range(min(3, teacher_info["layer_count"])):
                    max_neurons = min(500, teacher_info["intermediate_size"])
                    for neuron_idx in range(0, max_neurons, 10):
                        if is_mixtral:
                            num_experts = teacher_info["experts_info"]["num_experts"]
                            for expert_idx in range(min(2, num_experts)):
                                if (layer_idx, neuron_idx, expert_idx) not in existing_neurons:
                                    critical_neurons.append((layer_idx, neuron_idx, expert_idx, 0.0005))
                                    if len(critical_neurons) >= neuron_count:
                                        break
                        else:
                            if (layer_idx, neuron_idx) not in existing_neurons:
                                critical_neurons.append((layer_idx, neuron_idx, 0.0005))
                                if len(critical_neurons) >= neuron_count:
                                    break
                        
                        if len(critical_neurons) >= neuron_count:
                            break
                    if len(critical_neurons) >= neuron_count:
                        break
            
            critical_neurons_data = []
            for neuron_info in critical_neurons:
                if len(neuron_info) == 4:
                    layer, neuron, expert, importance = neuron_info
                    critical_neurons_data.append({
                        "layer": int(layer),
                        "neuron": int(neuron),
                        "expert": int(expert),
                        "importance": float(importance)
                    })
                else:
                    layer, neuron, importance = neuron_info
                    critical_neurons_data.append({
                        "layer": int(layer),
                        "neuron": int(neuron),
                        "expert": None,
                        "importance": float(importance)
                    })
            
            with open(os.path.join(self.output_dir, "critical_neurons.json"), "w") as f:
                json.dump(critical_neurons_data, f, indent=2)
            
            logger.info("Transferring neurons from teacher to student...")
            transfer_manager = MistralNeuronTransfer(
                self.teacher_model, 
                self.student_model, 
                teacher_device=self.teacher_device, 
                student_device=self.student_device
            )
            
            clean_memory()
            
            try:
                self.modified_student, transferred_neurons = transfer_manager.transfer_neurons(critical_neurons)
            except Exception as e:
                logger.error(f"Neuron transfer failed: {str(e)}")
                logger.info("Using unmodified student model instead")
                self.modified_student = self.student_model
                transferred_neurons = []
            
            transferred_neurons_data = []
            for neuron_info in transferred_neurons:
                if len(neuron_info) == 4:
                    layer, neuron, expert, importance = neuron_info
                    transferred_neurons_data.append({
                        "layer": int(layer),
                        "neuron": int(neuron),
                        "expert": int(expert),
                        "importance": float(importance)
                    })
                else:
                    layer, neuron, importance = neuron_info
                    transferred_neurons_data.append({
                        "layer": int(layer),
                        "neuron": int(neuron),
                        "expert": None,
                        "importance": float(importance)
                    })
            
            with open(os.path.join(self.output_dir, "transferred_neurons.json"), "w") as f:
                json.dump(transferred_neurons_data, f, indent=2)
            
            self.teacher_model = None
            clean_memory()
            
            logger.info("Evaluating modified student (before calibration)...")
            evaluator = ArithmeticEvaluator(self.modified_student, self.student_tokenizer, self.student_device)
            pre_calibration_results = evaluator.evaluate_comprehensive(
                operations=operations,
                digits_range=digits_range,
                samples_per_op=samples_per_op
            )
            
            with open(os.path.join(self.output_dir, "pre_calibration_results.json"), "w") as f:
                json.dump(pre_calibration_results, f, indent=2)
            
            logger.info("Creating calibration dataset...")
            calibration_dataset = ArithmeticDataset(
                size=min(50, samples_per_op) * len(operations),
                digits_range=digits_range,
                operations=operations,
                seed=42
            )
            
            logger.info("Calibrating modified student model...")
            calibrator = MistralCalibration(self.modified_student, self.student_tokenizer, self.student_device)
            
            clean_memory()
            
            modified_student_backup = copy.deepcopy(self.modified_student)
            
            try:
                self.calibrated_student = calibrator.calibrate(
                    calibration_dataset,
                    learning_rate=calibration_lr,
                    num_epochs=calibration_epochs,
                    batch_size=1
                )
            except Exception as e:
                logger.error(f"Calibration failed: {str(e)}")
                logger.info("Using pre-calibration model for evaluation")
                self.calibrated_student = modified_student_backup
            
            logger.info("Validating calibrated model...")
            try:
                test_input = self.student_tokenizer("2 + 2 = ", return_tensors="pt").to(self.student_device)
                with torch.no_grad():
                    outputs = self.calibrated_student.generate(
                        input_ids=test_input.input_ids,
                        max_new_tokens=5,
                        do_sample=False
                    )
                prediction = self.student_tokenizer.decode(outputs[0, test_input.input_ids.shape[1]:], skip_special_tokens=True)
                logger.info(f"Validation test: '2 + 2 = ' -> '{prediction}'")
            except Exception as e:
                logger.error(f"Calibrated model validation failed: {str(e)}")
                logger.info("Using pre-calibration model")
                self.calibrated_student = modified_student_backup
            
            logger.info("Evaluating calibrated student...")
            evaluator = ArithmeticEvaluator(self.calibrated_student, self.student_tokenizer, self.student_device)
            
            clean_memory()
            
            try:
                post_calibration_results = evaluator.evaluate_comprehensive(
                    operations=operations,
                    digits_range=digits_range,
                    samples_per_op=samples_per_op
                )
            except Exception as e:
                logger.error(f"Post-calibration evaluation failed: {str(e)}")
                logger.info("Using previous results for post-calibration")
                post_calibration_results = pre_calibration_results.copy() if pre_calibration_results else {
                    "overall_accuracy": baseline_results.get("overall_accuracy", 0.0),
                    "operation_accuracy": baseline_results.get("operation_accuracy", {op: 0.0 for op in operations}),
                    "digits_accuracy": baseline_results.get("digits_accuracy", {d: 0.0 for d in range(digits_range[0], digits_range[1] + 1)}),
                    "total_examples": baseline_results.get("total_examples", 0),
                    "total_correct": baseline_results.get("total_correct", 0),
                    "error": str(e)
                }
            
            with open(os.path.join(self.output_dir, "post_calibration_results.json"), "w") as f:
                json.dump(post_calibration_results, f, indent=2)
            
            baseline_acc = baseline_results.get("overall_accuracy", 0.0)
            pre_cal_acc = pre_calibration_results.get("overall_accuracy", 0.0)
            post_cal_acc = post_calibration_results.get("overall_accuracy", 0.0)
            
            summary = {
                "baseline_accuracy": baseline_acc,
                "pre_calibration_accuracy": pre_cal_acc,
                "post_calibration_accuracy": post_cal_acc,
                "neurons_analyzed": len(critical_neurons),
                "neurons_transferred": len(transferred_neurons),
                "absolute_improvement": post_cal_acc - baseline_acc,
                "relative_improvement": (post_cal_acc / max(0.001, baseline_acc)) - 1,
                "runtime_seconds": time.time() - start_time
            }
            
            with open(os.path.join(self.output_dir, "experiment_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            
            if summary["absolute_improvement"] > 0:
                model_save_path = os.path.join(self.output_dir, "calibrated_model")
                logger.info(f"Saving calibrated model to {model_save_path}")
                
                clean_memory()
                
                try:
                    self.calibrated_student.save_pretrained(model_save_path, safe_serialization=True)
                    self.student_tokenizer.save_pretrained(model_save_path)
                except Exception as e:
                    logger.error(f"Failed to save calibrated model: {str(e)}")
            
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
            
            with open(os.path.join(self.output_dir, "experiment_error.txt"), "w") as f:
                f.write(f"Error: {str(e)}\n\n")
                f.write(traceback.format_exc())
            
            raise e

def main():
    parser = argparse.ArgumentParser(description="Neuron Attribution for Arithmetic Knowledge Transfer")
    
    parser.add_argument("--teacher", type=str, default="mistralai/Mixtral-8x7B-v0.1",
                       help="Teacher model name or path")
    parser.add_argument("--student", type=str, default="mistralai/Mistral-7B-v0.1",
                       help="Student model name or path")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory to cache models")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace token for accessing gated models")
    
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
    
    parser.add_argument("--calibration-lr", type=float, default=5e-7,
                       help="Learning rate for calibration")
    parser.add_argument("--calibration-epochs", type=int, default=1,
                       help="Number of epochs for calibration")
    
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    operations = args.operations.split(",")
    
    if args.token:
        os.environ["HUGGINGFACE_TOKEN"] = args.token
        logger.info("Using HuggingFace token from command line argument")
    elif "HUGGINGFACE_TOKEN" not in os.environ:
        logger.warning(
            "HUGGINGFACE_TOKEN not provided. You'll need this to access Mistral models.\n"
            "Set it using: export HUGGINGFACE_TOKEN=your_token_here\n"
            "Or pass it with --token your_token_here"
        )
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            logger.info(f"GPU {i} free memory: {free_memory / 1e9:.2f} GB")
    
    if torch.cuda.device_count() < 2:
        logger.warning(f"Only {torch.cuda.device_count()} GPU(s) available. Model parallelism requires at least 2 GPUs.")
        teacher_device = "cuda:0"
        student_device = "cuda:0"
    else:
        logger.info(f"Found {torch.cuda.device_count()} GPUs. Using model parallelism.")
        teacher_device = "cuda:0"
        student_device = "cuda:1"
    
    clean_memory()
    
    experiment = ArithmeticTransferExperiment(
        teacher_model_name=args.teacher,
        student_model_name=args.student,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        teacher_device=teacher_device,
        student_device=student_device
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
    
    clean_memory()
    
    return results

if __name__ == "__main__":
    main()
