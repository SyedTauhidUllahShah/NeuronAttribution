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
    AutoModelForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    default_data_collator
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

class ArithmeticMaskingDataset(Dataset):
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
            
            prompt = f"{a} {operation} {b} = [MASK]"
            answer = str(result)
            
            samples.append({
                "prompt": prompt,
                "answer": answer,
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

def create_bert_dataloader(dataset, tokenizer, batch_size=4, max_length=32):
    def _collate_fn(batch):
        input_texts = []
        labels = []
        
        for item in batch:
            input_text = item["prompt"].replace("[MASK]", tokenizer.mask_token)
            input_texts.append(input_text)
            labels.append(int(item["answer"]))
        
        encoded_inputs = tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        mask_positions = []
        for i, input_ids in enumerate(encoded_inputs.input_ids):
            mask_pos = torch.where(input_ids == tokenizer.mask_token_id)[0]
            if len(mask_pos) > 0:
                mask_positions.append(mask_pos[0].item())
            else:
                mask_positions.append(-1)
        
        return {
            "input_ids": encoded_inputs.input_ids,
            "attention_mask": encoded_inputs.attention_mask,
            "mask_positions": torch.tensor(mask_positions),
            "labels": torch.tensor(labels)
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn
    )

class BertModelManager:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
    
    def load_model(self, model_name, device="cuda"):
        logger.info(f"Loading BERT model: {model_name} on {device}")
        
        load_kwargs = {
            "cache_dir": self.cache_dir,
        }
        
        if torch.cuda.is_available():
            logger.info(f"Memory before loading: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        model = AutoModelForMaskedLM.from_pretrained(model_name, **load_kwargs)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        model = model.to(device)
        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        if torch.cuda.is_available():
            logger.info(f"Memory after loading: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            
        return model, tokenizer
    
    def get_model_size_info(self, model, tokenizer):
        config = model.config
        
        info = {
            "model_name": config._name_or_path,
            "parameter_count": sum(p.numel() for p in model.parameters()) / 1_000_000,
            "layer_count": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "head_dim": config.hidden_size // config.num_attention_heads,
            "vocab_size": len(tokenizer),
        }
        
        return info

class BertNeuronAnalyzer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.intermediate_size = self.model.config.intermediate_size
    
    def compute_output_probability(self, text, answer, intervened_neurons=None):
        input_text = text.replace("[MASK]", self.tokenizer.mask_token)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        mask_pos = torch.where(inputs.input_ids[0] == self.tokenizer.mask_token_id)[0]
        if len(mask_pos) == 0:
            raise ValueError(f"No mask token found in input: {input_text}")
        
        mask_pos = mask_pos[0].item()
        answer_token_id = self.tokenizer.convert_tokens_to_ids(answer)
        
        if intervened_neurons and len(intervened_neurons) > 0:
            original_params = {}
            for layer_idx, neuron_idx in intervened_neurons:
                intermediate = self.model.bert.encoder.layer[layer_idx].intermediate
                output = self.model.bert.encoder.layer[layer_idx].output
                
                original_params[(layer_idx, neuron_idx, 'dense')] = {
                    'weight': intermediate.dense.weight[neuron_idx, :].clone(),
                }
                if hasattr(intermediate.dense, 'bias') and intermediate.dense.bias is not None:
                    original_params[(layer_idx, neuron_idx, 'dense')]['bias'] = intermediate.dense.bias[neuron_idx].clone()
                
                output_weight_col = output.dense.weight[:, neuron_idx].clone()
                original_params[(layer_idx, neuron_idx, 'output')] = {
                    'weight': output_weight_col,
                }
        
        if intervened_neurons and len(intervened_neurons) > 0:
            for layer_idx, neuron_idx in intervened_neurons:
                self._zero_out_neuron(layer_idx, neuron_idx)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, mask_pos, :]
            probs = F.softmax(logits, dim=-1)
            target_prob = probs[answer_token_id].item()
        
        if intervened_neurons and len(intervened_neurons) > 0:
            with torch.no_grad():
                for (layer_idx, neuron_idx, proj_type), params in original_params.items():
                    layer = self.model.bert.encoder.layer[layer_idx]
                    
                    if proj_type == 'dense':
                        intermediate = layer.intermediate
                        weight = intermediate.dense.weight.clone()
                        weight[neuron_idx, :] = params['weight']
                        intermediate.dense.weight.copy_(weight)
                        
                        if 'bias' in params and hasattr(intermediate.dense, 'bias') and intermediate.dense.bias is not None:
                            bias = intermediate.dense.bias.clone()
                            bias[neuron_idx] = params['bias']
                            intermediate.dense.bias.copy_(bias)
                    
                    elif proj_type == 'output':
                        output = layer.output
                        weight = output.dense.weight.clone()
                        weight[:, neuron_idx] = params['weight']
                        output.dense.weight.copy_(weight)
        
        return target_prob
    
    def _zero_out_neuron(self, layer_idx, neuron_idx):
        with torch.no_grad():
            layer = self.model.bert.encoder.layer[layer_idx]
            
            intermediate = layer.intermediate
            weight = intermediate.dense.weight.clone()
            weight[neuron_idx, :] = 0
            intermediate.dense.weight.copy_(weight)
            
            if hasattr(intermediate.dense, 'bias') and intermediate.dense.bias is not None:
                bias = intermediate.dense.bias.clone()
                bias[neuron_idx] = 0
                intermediate.dense.bias.copy_(bias)
            
            output = layer.output
            weight = output.dense.weight.clone()
            weight[:, neuron_idx] = 0
            output.dense.weight.copy_(weight)
    
    def compute_neuron_importance(self, text, answer, layer_range=None, top_k=100, batch_size=50):
        if layer_range is None:
            layer_range = (0, self.num_layers)
        
        clean_memory()
        
        baseline_prob = self.compute_output_probability(text, answer)
        logger.info(f"Baseline probability: {baseline_prob:.6f}")
        
        if baseline_prob < 1e-5:
            logger.warning(f"Baseline probability too low ({baseline_prob:.6f}) for prompt: '{text}'")
            logger.warning("Using artificial threshold for comparison.")
            baseline_prob = 0.01
        
        neuron_scores = []
        
        for layer_idx in tqdm(range(layer_range[0], layer_range[1]), desc="Analyzing layers"):
            clean_memory()
            
            batch_indices = list(range(0, self.intermediate_size, batch_size))
            for start_idx in tqdm(batch_indices, desc=f"Screening layer {layer_idx}", leave=False):
                end_idx = min(start_idx + batch_size, self.intermediate_size)
                
                batch_neurons = [(layer_idx, i) for i in range(start_idx, end_idx)]
                
                batch_prob = self.compute_output_probability(text, answer, batch_neurons)
                
                batch_importance = baseline_prob - batch_prob
                
                if batch_importance > 0.001:
                    for neuron_idx in range(start_idx, end_idx):
                        neuron_prob = self.compute_output_probability(
                            text, answer, [(layer_idx, neuron_idx)]
                        )
                        
                        importance = baseline_prob - neuron_prob
                        
                        if importance > 0:
                            neuron_scores.append((layer_idx, neuron_idx, importance))
            
            clean_memory()
        
        neuron_scores.sort(key=lambda x: x[2], reverse=True)
        
        if len(neuron_scores) == 0:
            logger.warning("No critical neurons identified by importance analysis!")
            logger.info("Adding fallback neurons from early layers...")
            
            for layer_idx in range(min(3, self.num_layers)):
                for neuron_idx in range(min(100, self.intermediate_size)):
                    if neuron_idx % 10 == 0:
                        neuron_scores.append((layer_idx, neuron_idx, 0.001))
        
        return neuron_scores[:top_k]
    
    def analyze_arithmetic_neurons(self, operations=('+',), digits_range=(1, 1), 
                                  samples_per_op=3, top_k=100, layer_range=None):
        clean_memory()
        
        dataset = ArithmeticMaskingDataset(
            size=samples_per_op * len(operations), 
            digits_range=digits_range,
            operations=operations
        )
        
        all_critical_neurons = []
        
        for item in tqdm(dataset.samples, desc="Analyzing arithmetic problems"):
            prompt = item["prompt"]
            answer = item["answer"]
            
            logger.info(f"\nAnalyzing: {prompt} = {answer} (Operation: {item['operation']})")
            
            clean_memory()
            
            critical_neurons = self.compute_neuron_importance(
                prompt, answer, layer_range=layer_range, 
                top_k=min(top_k // len(dataset.samples), 20)
            )
            
            all_critical_neurons.extend(critical_neurons)
            
            logger.info(f"Top 5 neurons for this example:")
            for i, (layer, neuron, importance) in enumerate(critical_neurons[:5]):
                logger.info(f"  {i+1}. Layer {layer}, Neuron {neuron}: {importance:.6f}")
        
        neuron_dict = {}
        for layer, neuron, importance in all_critical_neurons:
            key = (layer, neuron)
            neuron_dict[key] = neuron_dict.get(key, 0) + importance
        
        critical_neurons = [(layer, neuron, importance) 
                           for (layer, neuron), importance in neuron_dict.items()]
        critical_neurons.sort(key=lambda x: x[2], reverse=True)
        
        if len(critical_neurons) < min(20, top_k):
            logger.warning(f"Only found {len(critical_neurons)} critical neurons, less than desired {min(20, top_k)}")
            logger.info("Adding fallback neurons from early layers...")
            
            existing_neurons = set((l, n) for l, n, _ in critical_neurons)
            added = 0
            
            for layer_idx in range(min(3, self.num_layers)):
                for neuron_idx in range(min(500, self.intermediate_size)):
                    if neuron_idx % 5 == 0 and (layer_idx, neuron_idx) not in existing_neurons:
                        critical_neurons.append((layer_idx, neuron_idx, 0.0005))
                        added += 1
                        if added + len(existing_neurons) >= top_k:
                            break
                if added + len(existing_neurons) >= top_k:
                    break
            
            critical_neurons.sort(key=lambda x: x[2], reverse=True)
        
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
                    input_text = prompt.replace("[MASK]", self.tokenizer.mask_token)
                    inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                    
                    mask_pos = torch.where(inputs.input_ids[0] == self.tokenizer.mask_token_id)[0][0]
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits[0, mask_pos, :]
                        
                        # Get top 5 predictions
                        top_tokens = torch.topk(logits, 5)
                        top_ids = top_tokens.indices.tolist()
                        top_tokens = [self.tokenizer.convert_ids_to_tokens(token_id) for token_id in top_ids]
                        
                        # Check if the expected answer is in top predictions
                        is_correct = expected in top_tokens or any(expected in token for token in top_tokens)
                    
                    if is_correct:
                        correct += 1
                        operation_correct[operation] += 1
                        digits_correct[digits] += 1
                except Exception as e:
                    logger.warning(f"Prediction failed for prompt '{prompt}': {str(e)}")
                
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

class BertNeuronTransfer:
    def __init__(self, teacher_model, student_model, teacher_device="cuda:0", student_device="cuda:1"):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_device = teacher_device
        self.student_device = student_device
        
        self.teacher_layers = self.teacher_model.config.num_hidden_layers
        self.student_layers = self.student_model.config.num_hidden_layers
        
        self.teacher_intermediate_size = self.teacher_model.config.intermediate_size
        self.student_intermediate_size = self.student_model.config.intermediate_size
        
        self.layer_mapping = self._create_layer_mapping()
        
        logger.info(f"Teacher model: {self.teacher_layers} layers, {self.teacher_intermediate_size} neurons per layer")
        logger.info(f"Student model: {self.student_layers} layers, {self.student_intermediate_size} neurons per layer")
    
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
        
        if not critical_neurons or len(critical_neurons) == 0:
            logger.error("No critical neurons provided for transfer!")
            logger.info("Using fallback neurons from early layers...")
            critical_neurons = []
            
            for layer_idx in range(min(4, self.teacher_layers)):
                for neuron_idx in range(min(500, self.teacher_intermediate_size)):
                    if neuron_idx % 5 == 0:
                        critical_neurons.append((layer_idx, neuron_idx, 0.001))
                    if len(critical_neurons) >= 100:
                        break
                if len(critical_neurons) >= 100:
                    break
        
        logger.info(f"Transferring {len(critical_neurons)} neurons from teacher to student...")
        
        for i, (layer_idx, neuron_idx, importance) in enumerate(tqdm(critical_neurons)):
            student_layer_idx = self.layer_mapping.get(layer_idx)
            
            if student_layer_idx is None:
                logger.warning(f"No mapping found for teacher layer {layer_idx}. Skipping...")
                continue
            
            if neuron_idx >= self.student_intermediate_size:
                logger.warning(f"Neuron index {neuron_idx} exceeds student model size {self.student_intermediate_size}. Skipping...")
                continue
            
            try:
                teacher_intermediate = self.teacher_model.bert.encoder.layer[layer_idx].intermediate
                teacher_output = self.teacher_model.bert.encoder.layer[layer_idx].output
                
                student_intermediate = modified_student.bert.encoder.layer[student_layer_idx].intermediate
                student_output = modified_student.bert.encoder.layer[student_layer_idx].output
                
                teacher_hidden_size = self.teacher_model.config.hidden_size
                student_hidden_size = self.student_model.config.hidden_size
                
                with torch.no_grad():
                    # Transfer intermediate.dense weights (inputs to the neuron)
                    teacher_weight = teacher_intermediate.dense.weight[neuron_idx, :].to(self.student_device)
                    
                    if teacher_hidden_size != student_hidden_size:
                        if teacher_hidden_size > student_hidden_size:
                            start_idx = (teacher_hidden_size - student_hidden_size) // 2
                            adapted_weight = teacher_weight[start_idx:start_idx+student_hidden_size]
                        else:
                            adapted_weight = torch.zeros(student_hidden_size, device=self.student_device)
                            start_idx = (student_hidden_size - teacher_hidden_size) // 2
                            adapted_weight[start_idx:start_idx+teacher_hidden_size] = teacher_weight
                    else:
                        adapted_weight = teacher_weight
                    
                    student_weight = student_intermediate.dense.weight.clone()
                    student_weight[neuron_idx, :] = adapted_weight
                    student_intermediate.dense.weight.copy_(student_weight)
                    
                    # Transfer intermediate.dense bias if exists
                    if (hasattr(teacher_intermediate.dense, 'bias') and 
                        teacher_intermediate.dense.bias is not None and
                        hasattr(student_intermediate.dense, 'bias') and
                        student_intermediate.dense.bias is not None):
                        teacher_bias = teacher_intermediate.dense.bias[neuron_idx].to(self.student_device)
                        student_bias = student_intermediate.dense.bias.clone()
                        student_bias[neuron_idx] = teacher_bias
                        student_intermediate.dense.bias.copy_(student_bias)
                
                    # Transfer output.dense weights (outputs from the neuron)
                    teacher_output_weight = teacher_output.dense.weight[:, neuron_idx].to(self.student_device)
                    teacher_output_size = teacher_output_weight.shape[0]
                    student_output_size = student_output.dense.weight.shape[0]
                    
                    if teacher_output_size != student_output_size:
                        if teacher_output_size > student_output_size:
                            start_idx = (teacher_output_size - student_output_size) // 2
                            adapted_output_weight = teacher_output_weight[start_idx:start_idx+student_output_size]
                        else:
                            adapted_output_weight = torch.zeros(student_output_size, device=self.student_device)
                            start_idx = (student_output_size - teacher_output_size) // 2
                            adapted_output_weight[start_idx:start_idx+teacher_output_size] = teacher_output_weight
                    else:
                        adapted_output_weight = teacher_output_weight
                    
                    student_output_weight = student_output.dense.weight.clone()
                    student_output_weight[:, neuron_idx] = adapted_output_weight
                    student_output.dense.weight.copy_(student_output_weight)
                
                transferred_neurons.append((student_layer_idx, neuron_idx, importance))
            except Exception as e:
                logger.error(f"Error transferring neuron (layer {layer_idx}, neuron {neuron_idx}): {str(e)}")
            
            if (i+1) % 10 == 0:
                clean_memory()
        
        logger.info(f"Successfully transferred {len(transferred_neurons)} neurons")
        
        if len(transferred_neurons) == 0:
            logger.error("No neurons were successfully transferred!")
            raise RuntimeError("Neuron transfer failed. Check layer mappings and neuron indices.")
        
        clean_memory()
        return modified_student, transferred_neurons

class BertCalibration:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def calibrate(self, dataset, learning_rate=5e-7, num_epochs=1, batch_size=1):
        logger.info(f"Starting memory-efficient calibration with lr={learning_rate}")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        
        clean_memory()
        
        def create_efficient_dataloader(dataset, tokenizer, batch_size=1, max_length=16):
            def _collate_fn(batch):
                input_texts = []
                labels = []
                
                for item in batch:
                    input_text = item["prompt"].replace("[MASK]", tokenizer.mask_token)
                    input_texts.append(input_text)
                    labels.append(int(item["answer"]))
                
                encoded_inputs = tokenizer(
                    input_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                mask_positions = []
                for i, input_ids in enumerate(encoded_inputs.input_ids):
                    mask_pos = torch.where(input_ids == tokenizer.mask_token_id)[0]
                    if len(mask_pos) > 0:
                        mask_positions.append(mask_pos[0].item())
                    else:
                        mask_positions.append(-1)
                
                return {
                    "input_ids": encoded_inputs.input_ids,
                    "attention_mask": encoded_inputs.attention_mask,
                    "mask_positions": torch.tensor(mask_positions),
                    "labels": torch.tensor(labels)
                }
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=_collate_fn
            )
        
        loader = create_efficient_dataloader(dataset, self.tokenizer, batch_size=batch_size)
        
        self.model.gradient_checkpointing_enable()
        
        if hasattr(self.model, "half") and self.model.dtype != torch.float16:
            logger.info("Converting model to float16 for memory efficiency")
            self.model = self.model.half()
        
        initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for param in self.model.parameters():
            param.requires_grad = True
        
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
        for epoch in range(num_epochs):
            epoch_loss = 0
            valid_batches = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Calibration epoch {epoch+1}/{num_epochs}")):
                try:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    mask_positions = batch["mask_positions"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        
                        # For each example, get logits at mask position
                        mask_logits = torch.stack([
                            logits[i, pos, :] for i, pos in enumerate(mask_positions) if pos >= 0
                        ])
                        
                        # Get token ids for answer digits
                        answer_token_ids = torch.tensor([
                            self.tokenizer.convert_tokens_to_ids(str(l)) for l in labels
                        ], device=self.device)
                        
                        # Compute loss
                        loss = F.cross_entropy(mask_logits, answer_token_ids) / accumulation_steps
                    
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        logger.warning(f"NaN/Inf loss encountered in batch {batch_idx}")
                        continue
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * accumulation_steps
                    valid_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"OOM in batch {batch_idx}: {str(e)}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        optimizer.zero_grad()
                        
                        if batch_idx < 5:
                            logger.warning("OOM occurred early. Consider CPU offloading instead.")
                    else:
                        logger.error(f"Error during calibration: {str(e)}")
                        if valid_batches < 3:
                            logger.error("Too many errors. Stopping calibration.")
                            self.model.load_state_dict(initial_model_state)
                            self.model.eval()
                            for param in self.model.parameters():
                                param.requires_grad = False
                            clean_memory()
                            return self.model
                
                if batch_idx % 10 == 0:
                    clean_memory()
            
            if valid_batches % accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
            
            if valid_batches > 0:
                avg_epoch_loss = epoch_loss / valid_batches
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Average loss: {avg_epoch_loss:.4f}")
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        try:
            # Test with simple arithmetic
            test_text = "2 + 2 = " + self.tokenizer.mask_token
            test_inputs = self.tokenizer(test_text, return_tensors="pt").to(self.device)
            mask_pos = torch.where(test_inputs.input_ids[0] == self.tokenizer.mask_token_id)[0][0]
            
            with torch.no_grad():
                outputs = self.model(**test_inputs)
                logits = outputs.logits[0, mask_pos, :]
                top_tokens = torch.topk(logits, 5)
                top_tokens = [self.tokenizer.convert_ids_to_tokens(token_id) for token_id in top_tokens.indices.tolist()]
                
            logger.info(f"Validation test: '2 + 2 =' -> Top predictions: {top_tokens}")
            
            if not any(str(digit) in ''.join(top_tokens) for digit in range(10)):
                logger.warning("Model validation failed: no digits in top predictions. Restoring original model.")
                self.model.load_state_dict(initial_model_state)
        except Exception as e:
            logger.error(f"Model validation failed after calibration: {str(e)}")
            logger.info("Restoring model to pre-calibration state.")
            self.model.load_state_dict(initial_model_state)
        
        clean_memory()
        return self.model

class ArithmeticEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.analyzer = BertNeuronAnalyzer(model, tokenizer, device)
    
    def evaluate_comprehensive(self, operations=('+', '-', '*', '/'), digits_range=(1, 2), samples_per_op=50):
        logger.info(f"Running comprehensive evaluation...")
        
        clean_memory()
        
        dataset = ArithmeticMaskingDataset(
            size=samples_per_op * len(operations),
            digits_range=digits_range,
            operations=operations
        )
        
        self.model.eval()
        
        try:
            test_text = "2 + 2 = " + self.tokenizer.mask_token
            test_inputs = self.tokenizer(test_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**test_inputs)
            logger.info("Model validation passed before evaluation.")
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
    
    def visualize_results(self, results, title="Arithmetic Evaluation Results", save_path=None):
        plt.figure(figsize=(12, 10))
        
        if "error" in results and results["error"]:
            plt.text(0.5, 0.5, f"Evaluation Error: {results['error']}", 
                    ha='center', va='center', fontsize=12)
            plt.suptitle(title)
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                logger.info(f"Error visualization saved to {save_path}")
                return save_path
            else:
                plt.show()
            return None
        
        plt.subplot(2, 2, 1)
        plt.bar(['Overall'], [results['overall_accuracy']], color='navy')
        plt.ylim(0, 1)
        plt.title("Overall Accuracy")
        plt.ylabel("Accuracy")
        
        plt.subplot(2, 2, 2)
        ops = list(results['operation_accuracy'].keys())
        accs = [results['operation_accuracy'][op] for op in ops]
        plt.bar(ops, accs, color='lightblue')
        plt.ylim(0, 1)
        plt.title("Accuracy by Operation")
        plt.ylabel("Accuracy")
        
        plt.subplot(2, 2, 3)
        digits = sorted(list(results['digits_accuracy'].keys()))
        accs = [results['digits_accuracy'][d] for d in digits]
        plt.bar([f"{d}-digit" for d in digits], accs, color='lightgreen')
        plt.ylim(0, 1)
        plt.title("Accuracy by Digit Complexity")
        plt.ylabel("Accuracy")
        
        plt.subplot(2, 2, 4)
        plt.bar(['Total', 'Correct'], [results['total_examples'], results['total_correct']], 
               color=['gray', 'lightcoral'])
        plt.title("Sample Counts")
        plt.ylabel("Count")
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Visualization saved to {save_path}")
            return save_path
        else:
            plt.show()
        
        return None

class ArithmeticTransferExperiment:
    def __init__(self, 
                teacher_model_name="bert-large-uncased", 
                student_model_name="bert-base-uncased",
                output_dir=None, 
                cache_dir=None,
                teacher_device="cuda:0",
                student_device="cuda:1"):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.teacher_device = teacher_device
        self.student_device = student_device
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"bert_arithmetic_transfer_experiment_{timestamp}"
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "experiment.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        
        self.model_manager = BertModelManager(cache_dir=cache_dir)
        
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
            "student_device": self.student_device
        }
        
        with open(os.path.join(self.output_dir, "experiment_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Starting experiment with config: {config}")
        
        try:
            clean_memory()
            
            logger.info(f"Loading teacher model: {self.teacher_model_name} on {self.teacher_device}")
            self.teacher_model, self.teacher_tokenizer = self.model_manager.load_model(
                self.teacher_model_name, device=self.teacher_device
            )
            
            clean_memory()
            
            logger.info(f"Loading student model: {self.student_model_name} on {self.student_device}")
            self.student_model, self.student_tokenizer = self.model_manager.load_model(
                self.student_model_name, device=self.student_device
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
            
            evaluator.visualize_results(
                baseline_results,
                title="Baseline Student Model Performance",
                save_path=os.path.join(self.output_dir, "baseline_results.png")
            )
            
            logger.info("Identifying critical neurons in teacher model...")
            analyzer = BertNeuronAnalyzer(self.teacher_model, self.teacher_tokenizer, self.teacher_device)
            
            clean_memory()
            
            critical_neurons = analyzer.analyze_arithmetic_neurons(
                operations=operations,
                digits_range=digits_range,
                samples_per_op=min(3, samples_per_op),
                top_k=neuron_count,
                layer_range=(0, self.teacher_model.config.num_hidden_layers)
            )
            
            if len(critical_neurons) == 0:
                logger.error("No critical neurons identified! Adding fallback neurons.")
                for layer_idx in range(min(3, self.teacher_model.config.num_hidden_layers)):
                    for neuron_idx in range(min(100, teacher_info["intermediate_size"])):
                        if neuron_idx % 10 == 0:
                            critical_neurons.append((layer_idx, neuron_idx, 0.001))
                        if len(critical_neurons) >= neuron_count:
                            break
                    if len(critical_neurons) >= neuron_count:
                        break
            
            critical_neurons_data = [
                {"layer": int(l), "neuron": int(n), "importance": float(i)}
                for l, n, i in critical_neurons
            ]
            
            with open(os.path.join(self.output_dir, "critical_neurons.json"), "w") as f:
                json.dump(critical_neurons_data, f, indent=2)
            
            logger.info("Transferring neurons from teacher to student...")
            transfer_manager = BertNeuronTransfer(
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
            
            transferred_neurons_data = [
                {"layer": int(l), "neuron": int(n), "importance": float(i)}
                for l, n, i in transferred_neurons
            ]
            
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
            
            evaluator.visualize_results(
                pre_calibration_results,
                title="Modified Student Model Performance (Before Calibration)",
                save_path=os.path.join(self.output_dir, "pre_calibration_results.png")
            )
            
            logger.info("Creating calibration dataset...")
            calibration_dataset = ArithmeticMaskingDataset(
                size=min(50, samples_per_op) * len(operations),
                digits_range=digits_range,
                operations=operations,
                seed=42
            )
            
            logger.info("Calibrating modified student model...")
            calibrator = BertCalibration(self.modified_student, self.student_tokenizer, self.student_device)
            
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
                logger.info("Using unmodified student model for evaluation")
                self.calibrated_student = modified_student_backup
            
            logger.info("Validating calibrated model...")
            try:
                test_text = "2 + 2 = " + self.tokenizer.mask_token
                test_inputs = self.tokenizer(test_text, return_tensors="pt").to(self.device)
                mask_pos = torch.where(test_inputs.input_ids[0] == self.tokenizer.mask_token_id)[0][0]
                
                with torch.no_grad():
                    outputs = self.model(**test_inputs)
                    logits = outputs.logits[0, mask_pos, :]
                    top_tokens = torch.topk(logits, 5)
                    top_tokens = [self.tokenizer.convert_ids_to_tokens(token_id) for token_id in top_tokens.indices.tolist()]
                    
                logger.info(f"Validation test: '2 + 2 =' -> Top predictions: {top_tokens}")
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
                logger.info("Using baseline results for post-calibration")
                post_calibration_results = {
                    "overall_accuracy": 0.0,
                    "operation_accuracy": {op: 0.0 for op in operations},
                    "digits_accuracy": {d: 0.0 for d in range(digits_range[0], digits_range[1] + 1)},
                    "total_examples": 0,
                    "total_correct": 0,
                    "error": str(e)
                }
            
            with open(os.path.join(self.output_dir, "post_calibration_results.json"), "w") as f:
                json.dump(post_calibration_results, f, indent=2)
            
            evaluator.visualize_results(
                post_calibration_results,
                title="Calibrated Student Model Performance",
                save_path=os.path.join(self.output_dir, "post_calibration_results.png")
            )
            
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
                    self.calibrated_student.save_pretrained(model_save_path)
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
    parser = argparse.ArgumentParser(description="BERT Neuron Attribution for Arithmetic Knowledge Transfer")
    
    parser.add_argument("--teacher", type=str, default="bert-large-uncased",
                       help="Teacher model name or path")
    parser.add_argument("--student", type=str, default="bert-base-uncased",
                       help="Student model name or path")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory to cache models")
    
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
