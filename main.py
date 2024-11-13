# main.py

import os
import sys
import multiprocessing
from human_eval.data import read_problems
from run import run  # Import the run function

def main():
    model_name = "microsoft/Phi-3.5-mini-instruct"  # Updated model name
    num_tasks_to_evaluate = 64  # Adjust as needed
    iterations = 200  # Adjust as needed
    samples_file = "samples.jsonl"
    num_gpus = 8  # Number of GPUs to use

    # Load the problems
    problems = read_problems()
    if not problems:
        print("Dataset loading failed. Exiting.")
        return

    # Get the task IDs
    task_ids = list(problems.keys())[:num_tasks_to_evaluate]

    # Split the tasks among the GPUs
    task_splits = [task_ids[i::num_gpus] for i in range(num_gpus)]

    # Create a list to hold the processes
    processes = []

    # Output file for samples (ensure it's empty before starting)
    if os.path.exists(samples_file):
        os.remove(samples_file)

    # Start a process for each GPU
    for device_id in range(num_gpus):
        tasks_for_device = task_splits[device_id]
        if not tasks_for_device:
            continue  # Skip if there are no tasks assigned to this device

        # Create a process
        p = multiprocessing.Process(
            target=run,
            args=(model_name, tasks_for_device, device_id, samples_file, iterations)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All processes completed.")

    # Optionally, evaluate the samples after all processes have finished
    from evaluate_samples import evaluate_samples
    problem_file = "subset_problems.jsonl"
    save_subset_problems(problems, task_ids, problem_file)
    evaluate_samples(samples_file, problem_file)

def save_subset_problems(problems, task_ids, output_file):
    """Save a subset of problems to a JSONL file."""
    import json
    with open(output_file, 'w') as f:
        for task_id in task_ids:
            problem = problems[task_id]
            json.dump(problem, f)
            f.write('\n')

if __name__ == "__main__":
    main()
