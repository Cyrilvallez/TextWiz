import subprocess
import time
import shlex
import sys
import argparse
import os
from datetime import date

import torch
import transformers
import flash_attn

import textwiz


def dispatch_jobs_srun(gpu_footprints: list[int], num_gpus: int, commands: list[str], cpus_per_task: int | list[int] = 2,
                       memory: float | list[float] = 35):
    """Dispatch and run all `commands` using `srun` (https://slurm.schedmd.com/srun.html), using the number of
    gpus contained in `gpu_footprints`. The dispatch of models to gpus is very naive: as soon as enough gpus
    are available to run the job that requires the less gpu, we launch it. Thus the gpu efficiency may not be the
    best possible. However, this would be extremely hard to improve on this simple strategy, especially since we do
    not know the runtime of each job.

    Parameters
    ----------
    gpu_footprints : list[int]
        List containing the number of gpus necessary for each `commands`.
    num_gpus : int
        The total number of gpus we have at our disposal.
    commands : list[str]
        The executables to run on the slrum cluster using `srun`.
    cpus_per_task : int | list[int], optional
        An int describing the number of cpus to use for all task, or a list of ints describing the number of cpus
        to use for each `commands`, by default 2.
    memory : float | list[float], optional
        A float describing the amount of RAM (GB) to use for all task, or a list of floats describing the the
        amount of RAM (GB) to use for each `commands`, by default 35.
    """

    if any([x > num_gpus for x in gpu_footprints]):
        raise ValueError('One of the function calls needs more gpus than the total number available `num_gpus`.')
    
    if len(gpu_footprints) != len(commands):
        raise ValueError('You need to specify the number of gpus for exactly each command to run.')
    
    args = (cpus_per_task, memory)
    N = len(gpu_footprints)
    # Each argument which does not have a len() of size N is cast as a list of repeating elements of size N
    iterable_args = []
    for arg in args:
        try:
            if len(arg) == N:
                iterable_args.append(arg)
            else:
                iterable_args.append([arg]*N)
        except TypeError:
            iterable_args.append([arg]*N)

    sorting = sorted(zip(gpu_footprints, commands, *iterable_args), key=lambda x: x[0])
    # Collect back the iterables
    gpu_footprints = [x[0] for x in sorting]
    commands = [x[1] for x in sorting]
    cpus_per_task = [x[2] for x in sorting]
    memory = [x[3] for x in sorting]

    # Initialize the lists we will maintain
    available_gpus = [i for i in range(num_gpus)]
    processes = []
    associated_gpus = []

    while True:

        no_sleep = False

        # In this case we have enough gpus available to launch the job that needs the less gpus
        if len(available_gpus) >= gpu_footprints[0]:

            no_sleep = True

            # Remove them from the list of models to process
            footprint = gpu_footprints.pop(0)
            cpus = cpus_per_task.pop(0)
            mem = memory.pop(0)
            executable = commands.pop(0)

            # Update gpu resources
            allocated_gpus = available_gpus[0:footprint]
            available_gpus = available_gpus[footprint:]

            # exclusive option is on by default for step allocations, and exact is implicitly set by --cpus-per-task,
            # but we still set them explicitly for completeness
            full_command = (f'srun --exclusive --exact --ntasks=1 --gpus-per-task={footprint} --cpus-per-task={cpus} '
                            f'--mem={mem}G {executable}')
            p = subprocess.Popen(shlex.split(full_command), stdout=sys.stdout, stderr=sys.stderr)

            # Add them to the list of running processes
            processes.append(p)
            associated_gpus.append(allocated_gpus)

        # Find the indices of the processes that are finished if any
        indices_to_remove = []
        for i, process in enumerate(processes):
            if process.poll() is not None:
                # the wait() is only used to clean the subprocess (avoid zombies), as it is already done at this point
                process.wait()
                indices_to_remove.append(i)

        if not len(indices_to_remove) == 0:
            # Update gpu resources
            released_gpus = [gpus for i, gpus in enumerate(associated_gpus) if i in indices_to_remove]
            available_gpus += [gpu for gpus in released_gpus for gpu in gpus]
            # Remove processes which are done
            processes = [process for i, process in enumerate(processes) if i not in indices_to_remove]
            associated_gpus = [gpus for i, gpus in enumerate(associated_gpus) if i not in indices_to_remove]

        # If we scheduled all jobs, break from the infinite loop
        if len(gpu_footprints) == 0:
            break

        # Sleep for 3 seconds before restarting the loop and check if we have enough resources to launch
        # a new job
        if not no_sleep:
            time.sleep(3)

    # Sleep until all processes are finished (they have all been scheduled at this point)
    for process in processes:
        process.wait()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Memory estimator')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    parser.add_argument('--N', type=int, default=5,
                        help='The number of time to repeat each computation for accurate estimation. By default 5.')
    
    args = parser.parse_args()
    int8 = args.int8
    int4 = args.int4
    N = args.N

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')

    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")
    
    num_gpus = torch.cuda.device_count()

    # Select models
    models = textwiz.loader.ALLOWED_MODELS

    print(f'Launching computations with {num_gpus} gpus available.')

    # Create the commands to run
    gpu_footprints = []
    commands = []
    for model in models:
        command = f'python3 -u memory_estimator.py {model} --N {N}'
        footprint = textwiz.estimate_number_of_gpus(model, int8, int4)[0]

        if model == 'command-r-plus':
            footprint = textwiz.estimate_number_of_gpus(model, int8, int4, max_fraction_gpu_0=0.9, max_fraction_gpus=0.9)[0]
            command += ' --max_gpu_0 0.9 --max_gpus 0.9'

        if model == 'bloom-176B':
            footprint = textwiz.estimate_number_of_gpus(model, int8, int4, max_fraction_gpu_0=0.95, max_fraction_gpus=0.95)[0]
            command += ' --max_gpu_0 0.95 --max_gpus 0.95'
            if not (int8 or int4):
                command += ' --int8'

        commands.append(command)
        gpu_footprints.append(footprint)

    if int8:
        commands = [c + ' --int8' for c in commands]
    if int4:
        commands = [c + ' --int4' for c in commands]

    # Save infos about the benchmark
    benchmark_info_filename = os.path.join(textwiz.helpers.utils.DATA_FOLDER, 'memory_estimator', 'infos.json')
    infos = {
        'date': str(date.today()),
        'GPU_type': 'A100 40GB',
        'transformers_version': transformers.__version__,
        'textwiz_version': textwiz.__version__,
        'torch_version': torch.__version__,
        'flash_attn_version': flash_attn.__version__,
    }
    textwiz.helpers.utils.save_json(infos, benchmark_info_filename)
    
    t0 = time.time()

    dispatch_jobs_srun(gpu_footprints, num_gpus, commands)

    dt = time.time() - t0
    print(f'Overall it took {dt/3600:.2f}h !')
