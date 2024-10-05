# Exercise: Device Query

In this exercise, we will test a simple CUDA program that queries the attached CUDA devices and gathers information about them using the CUDA Runtime API. No CUDA programming is involved; rather, the goals of this exercise are simply to demonstrate how to prepare and submit a GPU job, and to see how the Runtime API can be used to discover hardware properties.

Here is how to compile the source code, prepare the batch file, and submit the job. The instructions are specific to Frontera, but they can easily be modified for other systems.

1\. Copy and paste (or [download](exer1.txt)) the following code into a new file named _devicequery.cu_.

```bash
#include <stdio.h>

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;
}

int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
    return 0;
}
```

2\. Load the CUDA software using the module utility.

```bash
$ module load cuda
```

3\. Compile the code using the nvcc compiler, adding flags to ensure that the device has a compute capability that is known to be acceptable for running the code.

```bash
$ nvcc -arch=compute_60 -code=sm_60 devicequery.cu -o devicequery
```

4\. Prepare (or [download](exer1_batch.txt)) the batch file and save it as _batch.sh_ (or you can pick any filename). Remember to specify one of the GPU queues, such as Frontera's `rtx-dev` queue.

```bash
#!/bin/bash
#SBATCH -J gpu_query        # Job name
#SBATCH -o gpu_query.o%j    # Output and error file name
#SBATCH -N 1                # Total number of GPU nodes requested
#SBATCH -n 1                # Total cores needed for the job
#SBATCH -p rtx-dev          # Queue name
#SBATCH -t 00:01:00         # Run time (hh:mm:ss)
##SBATCH -A [account]       # Project number (uncomment to specify which one)

./devicequery
```

5\. Submit your job using the _sbatch_ command.

```bash
$ sbatch batch.sh
```

6\. Retrieve the results. If your job ran successfully, your results should be stored in the file _gpu\_query.o\[job ID\]_. Assuming you specified Frontera's `rtx-dev` queue, your output should look like the following:

```bash
CUDA Device Query...
There are 4 CUDA devices.

CUDA Device #0
Major revision number:         7
Minor revision number:         5
Name:                          Quadro RTX 5000
Total global memory:           16908615680
Total shared memory per block: 49152
Total registers per block:     65536
Warp size:                     32
Maximum memory pitch:          2147483647
Maximum threads per block:     1024
Maximum dimension 0 of block:  1024
Maximum dimension 1 of block:  1024
Maximum dimension 2 of block:  64
Maximum dimension 0 of grid:   2147483647
Maximum dimension 1 of grid:   65535
Maximum dimension 2 of grid:   65535
Clock rate:                    1815000
Total constant memory:         65536
Texture alignment:             512
Concurrent copy and execution: Yes
Number of multiprocessors:     48
Kernel execution timeout:      No

[...]
```

As you see, the program acquires device information via the CUDA Runtime API and outputs everything to STDOUT. The various device properties reported by the program have been discussed elsewhere in the Understanding GPU Architecture roadmap (except for the dimensions of blocks and grids).

Many more functions are provided in the CUDA Runtime API. Detailed [documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html#group__CUDART__DEVICE) is available from NVIDIA.