# run.py

import torch
import gc
import sys
from human_eval.data import read_problems, write_jsonl
from sequence_generator import SequenceGenerator
from graph_extender import GraphExtender
from evaluate_samples import evaluate_samples  # We'll create this module as well

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def run(model_name, task_ids, device_id, output_file, iterations=150):
    # Set the CUDA device
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Load the problems
    problems = read_problems()
    if not problems:
        print("Dataset loading failed. Exiting.")
        return

    # Filter the problems to only include the specified task_ids
    problems = {task_id: problems[task_id] for task_id in task_ids}

    # Initialize the sequence generator with the specified device
    sequence_generator = SequenceGenerator(model_name=model_name, rank=device_id)

    # Parameters for GraphExtender
    bottleneck_size = 2
    batch_size = 16
    max_leaf_nodes = 3000  # Adjust as needed

    samples = []

    for idx, task_id in enumerate(task_ids):
        print(f"Device {device_id} - Processing Task {idx + 1}/{len(task_ids)}")
        task = problems[task_id]
        prompt = task['prompt']
        print(f"Device {device_id} - Task ID: {task_id}")

        graph_extender = GraphExtender(
            sequence_generator=sequence_generator,
            bottleneck_size=bottleneck_size,
            batch_size=batch_size,
            max_leaf_nodes=max_leaf_nodes
        )

        generated_code = graph_extender.main(prompt, iterations)

        if generated_code is None:
            print(f"Device {device_id} - No code generated for this task.")
            continue

        # Remove the prompt from the generated code to get the completion only
        completion = generated_code[len(prompt):]

        samples.append(dict(task_id=task_id, completion=completion))
        print(f"Device {device_id} - Generated sample for {task_id}")

        # Clear memory to prevent GPU OOM
        clear_memory()

    # Write samples to the output file
    # Use a lock or append mode to prevent write conflicts if necessary
    write_jsonl(output_file, samples, append=True)
    print(f"Device {device_id} - Saved {len(samples)} samples to {output_file}")

    # Optionally, evaluate the samples (if you want each process to evaluate its own samples)
    # evaluate_samples(output_file, problem_file)
