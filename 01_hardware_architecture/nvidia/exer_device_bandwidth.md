# Exercise: Device Bandwidth

In this exercise, we will build and run one of the sample programs that NVIDIA provides with CUDA, in order to test the bandwidth between the host and an attached CUDA device. In the bonus exercise, if more than one device is present, we will test the bandwidth between a pair of the devices using a second sample code.

Again, no CUDA programming is required for these exercises. The goals are just to gain experience in submitting GPU jobs, and to learn the rates at which data are transferred to and from (even within and among!) GPUs. While the exercises are again geared for Frontera, with suitable modifications they will work on a wide range of systems that include NVIDIA GPUs.

After locating the source and header files and the Makefile, we will build the code, prepare the batch file, and submit the job.

1\. Load the CUDA software using the module utility (as necessary).

```bash
module load cuda
```

2\. Set `CUDA_PATH` to the location where CUDA is installed. This can be deduced from the path to the `nvcc` compiler. The `CUDA_PATH` environment variable will be needed by the Makefile, later.

```bash
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
echo $CUDA_PATH  # should be the same as $TACC_CUDA_DIR on Frontera
```

3\. Copy the desired CUDA sample files to a directory in `$HOME`.

```bash
mkdir test
cd test
cp $CUDA_PATH/samples/1_Utilities/bandwidthTest/bandwidthTest.cu .
cp $CUDA_PATH/samples/1_Utilities/bandwidthTest/Makefile .
```

4\. Set `INCLUDES` as an environment variable for the Makefile you just copied, then build the executable with `make -e`.

```bash
export INCLUDES=-I${CUDA_PATH}/samples/common/inc
make -e
```

At the end of the make process, you are likely to see a "cannot create directory" error, which can be safely ignored.

5\. Prepare (or [download](exer2_batch.txt)) the batch file and save it as _batch\_test.sh_ (or pick any filename). Remember to submit the job to one of the GPU queues, such as Frontera's `rtx-dev` queue.

```bash
#!/bin/bash
#SBATCH -J gpu_test         # Job name
#SBATCH -o gpu_test.o%j     # Output and error file name
#SBATCH -N 1                # Total number of GPU nodes requested
#SBATCH -n 1                # Total cores needed for the job
#SBATCH -p rtx-dev          # Queue name
#SBATCH -t 00:05:00         # Run time (hh:mm:ss)
##SBATCH -A [account]       # Project number (uncomment to specify which one)

./bandwidthTest
```

6\. Submit your job using the _sbatch_ command.

```bash
sbatch batch_test.sh
```

7\. Retrieve the results. If your job ran successfully, your results should be stored in the file _gpu\_test.o\[job ID\]_. Assuming you submitted your job to Frontera's `rtx-dev` queue, your output should look like the following:

```bash
Device 0: Quadro RTX 5000
Quick Mode

Host to Device Bandwidth, 1 Device(s)
PINNED Memory Transfers
  Transfer Size (Bytes)        Bandwidth(GB/s)
  32000000          11.7

Device to Host Bandwidth, 1 Device(s)
PINNED Memory Transfers
  Transfer Size (Bytes)        Bandwidth(GB/s)
  32000000                     13.2

Device to Device Bandwidth, 1 Device(s)
PINNED Memory Transfers
  Transfer Size (Bytes)        Bandwidth(GB/s)
  32000000                     364.3
```

The final bandwidth test above has a rather deceptive name. It is _not_ a measurement of the bandwidth of a memory copy between two devices, but rather the bandwidth of a memory copy _from the device to itself_. In other words, it is a measurement of the memory bandwidth. The result appears to be a reasonably high fraction of the nominal peak rate of 448 GB/s.

## Bonus exercise

As a follow-on to the above tests, you can try building and running a different sample code that is supplied with CUDA, in order to measure the speed of transfers that go directly from each of the attached GPU devices to each of the other attached devices. NVIDIA's term for this is peer-to-peer transfers. If no direct path is present, the transfer is routed through the host.

In practice, for the types of multi-GPU platforms described in this Virtual Workshop topic, the direct transfers should take place at least as fast as the host-to-device or device-to-host transfers. However, as mentioned in the [discussion of interconnects for the RTX 5000](rtx_5000.md), it turns out that direct routing may not be faster in practice.

To complete this bonus exercise, repeat the above steps, but replace step 3 with the following:

3\. Copy the desired CUDA sample files to a second directory in `$HOME`.

```bash
mkdir test2
cd test2
cp $CUDA_PATH/samples/1_Utilities/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu .
cp $CUDA_PATH/samples/1_Utilities/p2pBandwidthLatencyTest/Makefile .
```

The only other change to the instructions is that batch script should run the executable `p2pBandwidthLatencyTest`.

In the output, you should first see a matrix displaying exactly how the attached GPUs are coupled to one another, followed by a succession of matrices displaying the bandwidths and latencies between the devices. The latter are shown with P2P (peer-to-peer) disabled and enabled. For the Frontera nodes with 4 Quadro RTX 5000 GPUs, the peer-to-peer latency improvements will likely be more encouraging than the ones for bandwidth.
