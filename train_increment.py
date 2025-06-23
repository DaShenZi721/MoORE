import json
import os
import subprocess

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name = 'moore'
model_start_idx = 0

# Task names list
tasks = ["hellaswag", "arc-e", "piqa", "obqa", "arc-c", "csqa", "winogrande", "siqa", "boolq"]

# Base model
base_model = "meta-llama/Llama-3.1-8B-Instruct"
wandb_project_name = "Sparse-MoE-ization-increment"

# Load initial JSON data template
with open(f'moe_peft_{model_name}.json', 'r') as f:
    json_template = json.load(f)

# Loop through tasks
for i in range(len(tasks)):
    # Update the json_template
    json_template["lora"][0]["name"] = f"{model_name}_{model_start_idx + i}"
    json_template["lora"][0]["task_name"] = tasks[0] if i == 0 else ";".join(tasks[:i+1])
    
    # Save the updated template to a new JSON file
    new_json_filename = f"moe_peft_{model_name}_task_{i}.json"
    with open(new_json_filename, 'w') as f:
        json.dump(json_template, f, indent=4)
    
    # Construct the shell command
    command = f"python ./launch.py run --base_model {base_model} --config {new_json_filename} --log_wandb --wandb_project_name {wandb_project_name}"

    # Print the command for debugging purposes
    print(command)
    
    # Execute the shell command and wait for it to complete
    subprocess.run(command, shell=True, check=True)