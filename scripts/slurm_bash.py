"""
This script generates a SLURM bash script based on user-provided parameters.
Usage:

python slurm_bash.py --name my_job --cpus 4 --mem 8192 --time 02:00:00 --ntasks 1 --gpu True --gpu_type rtx_4090 --dir_name my_experiment --script_path /path/to/script.py --output_path slurm_script.sh
"""

from pathlib import Path
root_dir = Path(__file__).parent.parent
slurm_dir = root_dir / 'notes'

import sys
import argparse

def add_preamble(name, cpus, mem, time, ntasks=1, gpu=True, gpu_type=None):
    preamble = f"""
#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={name}%A_%a.out
#SBATCH --error={name}%A_%a.err
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem}
#SBATCH --ntasks={ntasks}
#SBATCH --time={time}
"""
    if gpu:
        if gpu_type:
            preamble += f"#SBATCH --gpus={gpu_type}:1"
        else:
            preamble += f"#SBATCH --gpus=rtx_4090:1"

    preamble += """\n\n
set -e  # fail fast if anything breaks

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

module load eth_proxy
"""

    return preamble

def set_environment(dir_name=None):
    env = f"""\n\n
# ---- Environment & execution ----
cd /cluster/scratch/mgadhia/{dir_name if dir_name else ''}
source env/bin/activate

echo "Activated venv ..."
"""
    return env

def specify_script(script_path):
    return f"\n\npython {script_path}\n"

def generate_slurm_script(args):
    preamble = add_preamble(args.name, args.cpus, args.mem, args.time, args.ntasks, args.gpu, args.gpu_type)
    env_setup = set_environment(args.dir_name)
    script_execution = specify_script(args.script_path)

    full_script = preamble + env_setup + script_execution
    output_path = slurm_dir / args.output_path
    print(f"Saving SLURM script to: {output_path}")
    with open(output_path, "w") as f:
        f.write(full_script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a SLURM bash script.")
    parser.add_argument("--name", type=str, required=True, help="Job name")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per task")
    parser.add_argument("--mem", type=str, default="4096", help="Memory per CPU")
    parser.add_argument("--time", type=str, default="01:00:00", help="Time limit (HH:MM:SS)")
    parser.add_argument("--ntasks", type=int, default=1, help="Number of tasks")
    parser.add_argument("--gpu", type=bool, default=True, help="Whether to use GPU")
    parser.add_argument("--gpu_type", type=str, default=None, help="Type of GPU to use (e.g., rtx_4090)")
    parser.add_argument("--dir_name", type=str, default=None, help="Directory name for execution")
    parser.add_argument("--script_path", type=str, required=True, help="Path to the Python script to execute")
    parser.add_argument("--output_path", type=str, default="slurm_script.sh", help="Path to save the generated SLURM script")

    args = parser.parse_args()

    generate_slurm_script(args)
