# nvidia-smi 快速入门

`nvidia-smi` 是 NVIDIA 驱动提供的命令行工具，能够帮助用户监控和管理 GPU 的状态与行为。本文整理了一些常用的 `nvidia-smi` 命令，帮助大家快速上手和高效使用。

## 1. 显示 GPU 基本信息

使用最常见的 `nvidia-smi` 命令可以展示所有 GPU 的概览信息，包括每块 GPU 的利用率、显存使用情况、风扇速度和温度等。

```bash
nvidia-smi
```

输出内容示例如下：

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:81:00.0 Off |                    0 |
| N/A   40C    P8              10W /  70W |      2MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## 2. 显示 GPU 型号信息

如果我们要获取显卡型号信息，可以使用 `-L` 参数：

```bash
nvidia-smi -L
GPU 0: Tesla T4 (UUID: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxx)
```

## 3. 显示 GPU 详细信息

如果我们需要获取更详细的信息，可以使用 `-q` 参数。这会展示每个 GPU 的所有硬件细节，包括温度、电源使用情况、性能状态等。

```bash
nvidia-smi -q
```

部分输出内容示例：

```text
==============NVSMI LOG==============

Timestamp                                 : Thu Sep 26 19:11:30 2024
Driver Version                            : 535.183.01
CUDA Version                              : 12.2

Attached GPUs                             : 1
GPU 00000000:81:00.0
    Product Name                          : Tesla T4
    Product Brand                         : NVIDIA
    Product Architecture                  : Turing
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Enabled
    ...
```

## 4. 查看特定 GPU

当系统中有多块 GPU 时，我们可以使用 `-i` 参数指定查看某块 GPU 的信息。例如，查看 ID 为 0 的 GPU：

```bash
nvidia-smi -i 0
```

输出仅限于指定的 GPU 设备。

## 5. 监控整体 GPU 使用情况

我们可以通过如下命令查看当前 GPU 资源整体使用情况。

```bash
nvidia-smi dmon -i 0 -d 2
#-i 0: 指定 GPU 的索引号，这里是索引为 0 的 GPU。
#-d 2: 设置采样间隔（采样频率），单位为秒，这里是每 2 秒采样一次。
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk
# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz
    0     15     41      -      0      0      0      0      0      0    405    300
    0     15     41      -      0      0      0      0      0      0    405    300
    0     15     40      -      0      0      0      0      0      0    405    300
    ...

#GPU 的索引号是 0。
#功耗是 15 瓦。
#GPU 温度是 41 摄氏度。
#显存温度没有显示（用 - 表示）。
#Streaming Multiprocessor 的利用率是 0%。
#显存利用率是 0%。
#视频编码引擎利用率是 0%。
#视频解码引擎利用率是 0%。
#JPEG 引擎利用率是 0%。
#Optical Flow Accelerator 的利用率是 0%。
#显存时钟频率是 405 MHz。
#GPU 核心时钟频率是 300 MHz。
```

## 6. 查看与 GPU 交互的进程

可以通过以下命令查看当前使用 GPU 资源的进程。默认情况下，它只会显示与 GPU 计算资源交互的进程。

```bash
nvidia-smi pmon -i 0
```

输出示例：

```text
# gpu        pid  type    sm   mem   enc   dec   command
# Idx         #   C/G     %     %     %     %   name
    0      2635     C     0     0     0     0   python
    ...

#gpu (Idx): GPU 的索引号。
#pid (#): 进程的进程 ID（PID）。
#type (C/G): 进程类型，C 代表计算进程，G 代表图形进程。
#sm (%): Streaming Multiprocessor 的利用率百分比。
#mem (%): 显存带宽利用率百分比。
#enc (%): 视频编码引擎利用率百分比。
#dec (%): 视频解码引擎利用率百分比。
#command (name): 进程的命令名称。
```

## 7. 实时刷新 GPU 信息

我们可以使用 `-l` 参数让 `nvidia-smi` 命令每隔固定时间刷新一次 GPU 状态。如下命令每 5 秒刷新一次：

```bash
nvidia-smi -l 5
```

这是一个很好的工具，用于实时监控 GPU 资源的使用。

## 8. 限制 GPU 的功耗

为了在使用 GPU 时更好地控制功耗，可以设置一个功耗限制。如下命令将 GPU 的最大功耗限制在 70 瓦以内：

```bash
nvidia-smi -i 0 -pl 70
```

请注意，功耗限制只能在指定的范围内设置，不同的 GPU 设备支持不同的功耗范围。

## 9. 清除已发生的错误记录

如果 GPU 上曾发生过错误（如 XID 错误），可以使用以下命令清除这些错误记录：

```bash
nvidia-smi --clear-gpu-errors
```

这是在调试和监控过程中保持 GPU 状态清晰的一个好方法。

## 10. 查看驱动版本

可以通过查询功能，查看每块 GPU 的驱动版本。命令输出的格式为 CSV 格式，便于后续处理：

```bash
nvidia-smi --query-gpu=driver_version --format=csv
```

输出内容示例：

```text
driver_version
460.39
```

## 11. 查看显存使用情况

此命令可以显示每块 GPU 的显存总量、已用显存和可用显存：

```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

输出示例：

```text
memory.total [MiB], memory.used [MiB], memory.free [MiB]
15109 MiB, 0 MiB, 15109 MiB
```

## 12. 设置 GPU 计算模式

可以为 GPU 设置不同的计算模式，以控制进程对 GPU 资源的访问权限。常用的计算模式包括：

- `0`：默认模式，所有进程均可使用 GPU；
- `1`：仅计算进程模式，只有计算任务可以使用 GPU；
- `2`：禁止模式，新的进程无法访问 GPU。

例如，将 GPU 0 设置为仅计算进程模式：

```bash
nvidia-smi -i 0 -c 1
```

## 参考

```bash
nvidia-smi -h

NVIDIA System Management Interface -- v535.104.05

NVSMI provides monitoring information for Tesla and select Quadro devices.
The data is presented in either a plain text or an XML format, via stdout or a file.
NVSMI also provides several management operations for changing the device state.

Note that the functionality of NVSMI is exposed through the NVML C-based
library. See the NVIDIA developer website for more information about NVML.
Python wrappers to NVML are also available.  The output of NVSMI is
not guaranteed to be backwards compatible; NVML and the bindings are backwards
compatible.

http://developer.nvidia.com/nvidia-management-library-nvml/
http://pypi.python.org/pypi/nvidia-ml-py/
Supported products:
- Full Support
    - All Tesla products, starting with the Kepler architecture
    - All Quadro products, starting with the Kepler architecture
    - All GRID products, starting with the Kepler architecture
    - GeForce Titan products, starting with the Kepler architecture
- Limited Support
    - All Geforce products, starting with the Kepler architecture
nvidia-smi [OPTION1 [ARG1]] [OPTION2 [ARG2]] ...

    -h,   --help                Print usage information and exit.

  LIST OPTIONS:

    -L,   --list-gpus           Display a list of GPUs connected to the system.

    -B,   --list-excluded-gpus  Display a list of excluded GPUs in the system.

  SUMMARY OPTIONS:

    <no arguments>              Show a summary of GPUs connected to the system.

    [plus any of]

    -i,   --id=                 Target a specific GPU.
    -f,   --filename=           Log to a specified file, rather than to stdout.
    -l,   --loop=               Probe until Ctrl+C at specified second interval.

  QUERY OPTIONS:

    -q,   --query               Display GPU or Unit info.

    [plus any of]

    -u,   --unit                Show unit, rather than GPU, attributes.
    -i,   --id=                 Target a specific GPU or Unit.
    -f,   --filename=           Log to a specified file, rather than to stdout.
    -x,   --xml-format          Produce XML output.
          --dtd                 When showing xml output, embed DTD.
    -d,   --display=            Display only selected information: MEMORY,
                                    UTILIZATION, ECC, TEMPERATURE, POWER, CLOCK,
                                    COMPUTE, PIDS, PERFORMANCE, SUPPORTED_CLOCKS,
                                    PAGE_RETIREMENT, ACCOUNTING, ENCODER_STATS,
                                    SUPPORTED_GPU_TARGET_TEMP, VOLTAGE
                                    FBC_STATS, ROW_REMAPPER, RESET_STATUS
                                Flags can be combined with comma e.g. ECC,POWER.
                                Sampling data with max/min/avg is also returned
                                for POWER, UTILIZATION and CLOCK display types.
                                Doesn't work with -u or -x flags.
    -l,   --loop=               Probe until Ctrl+C at specified second interval.

    -lms, --loop-ms=            Probe until Ctrl+C at specified millisecond interval.

  SELECTIVE QUERY OPTIONS:

    Allows the caller to pass an explicit list of properties to query.

    [one of]

    --query-gpu                 Information about GPU.
                                Call --help-query-gpu for more info.
    --query-supported-clocks    List of supported clocks.
                                Call --help-query-supported-clocks for more info.
    --query-compute-apps        List of currently active compute processes.
                                Call --help-query-compute-apps for more info.
    --query-accounted-apps      List of accounted compute processes.
                                Call --help-query-accounted-apps for more info.
                                This query is not supported on vGPU host.
    --query-retired-pages       List of device memory pages that have been retired.
                                Call --help-query-retired-pages for more info.
    --query-remapped-rows       Information about remapped rows.
                                Call --help-query-remapped-rows for more info.

    [mandatory]

    --format=                   Comma separated list of format options:
                                  csv - comma separated values (MANDATORY)
                                  noheader - skip the first line with column headers
                                  nounits - don't print units for numerical
                                             values

    [plus any of]

    -i,   --id=                 Target a specific GPU or Unit.
    -f,   --filename=           Log to a specified file, rather than to stdout.
    -l,   --loop=               Probe until Ctrl+C at specified second interval.
    -lms, --loop-ms=            Probe until Ctrl+C at specified millisecond interval.

  DEVICE MODIFICATION OPTIONS:

    [any one of]

    -pm,  --persistence-mode=   Set persistence mode: 0/DISABLED, 1/ENABLED
    -e,   --ecc-config=         Toggle ECC support: 0/DISABLED, 1/ENABLED
    -p,   --reset-ecc-errors=   Reset ECC error counts: 0/VOLATILE, 1/AGGREGATE
    -c,   --compute-mode=       Set MODE for compute applications:
                                0/DEFAULT, 1/EXCLUSIVE_THREAD (DEPRECATED),
                                2/PROHIBITED, 3/EXCLUSIVE_PROCESS
          --gom=                Set GPU Operation Mode:
                                    0/ALL_ON, 1/COMPUTE, 2/LOW_DP
    -r    --gpu-reset           Trigger reset of the GPU.
                                Can be used to reset the GPU HW state in situations
                                that would otherwise require a machine reboot.
                                Typically useful if a double bit ECC error has
                                occurred.
                                Reset operations are not guarenteed to work in
                                all cases and should be used with caution.
    -vm   --virt-mode=          Switch GPU Virtualization Mode:
                                Sets GPU virtualization mode to 3/VGPU or 4/VSGA
                                Virtualization mode of a GPU can only be set when
                                it is running on a hypervisor.
    -lgc  --lock-gpu-clocks=    Specifies <minGpuClock,maxGpuClock> clocks as a
                                    pair (e.g. 1500,1500) that defines the range
                                    of desired locked GPU clock speed in MHz.
                                    Setting this will supercede application clocks
                                    and take effect regardless if an app is running.
                                    Input can also be a singular desired clock value
                                    (e.g. <GpuClockValue>). Optionally, --mode can be
                                    specified to indicate a special mode.
    -m    --mode=               Specifies the mode for --locked-gpu-clocks.
                                    Valid modes: 0, 1
    -rgc  --reset-gpu-clocks
                                Resets the Gpu clocks to the default values.
    -lmc  --lock-memory-clocks=  Specifies <minMemClock,maxMemClock> clocks as a
                                    pair (e.g. 5100,5100) that defines the range
                                    of desired locked Memory clock speed in MHz.
                                    Input can also be a singular desired clock value
                                    (e.g. <MemClockValue>).
    -rmc  --reset-memory-clocks
                                Resets the Memory clocks to the default values.
    -lmcd --lock-memory-clocks-deferred=
                                    Specifies memClock clock to lock. This limit is
                                    applied the next time GPU is initialized.
                                    This is guaranteed by unloading and reloading the kernel module.
                                    Requires root.
    -rmcd --reset-memory-clocks-deferred
                                Resets the deferred Memory clocks applied.
    -ac   --applications-clocks= Specifies <memory,graphics> clocks as a
                                    pair (e.g. 2000,800) that defines GPU's
                                    speed in MHz while running applications on a GPU.
    -rac  --reset-applications-clocks
                                Resets the applications clocks to the default values.
    -pl   --power-limit=        Specifies maximum power management limit in watts.
                                Takes an optional argument --scope.
    -sc   --scope=              Specifies the device type for --scope: 0/GPU, 1/TOTAL_MODULE (Grace Hopper Only)
    -cc   --cuda-clocks=        Overrides or restores default CUDA clocks.
                                In override mode, GPU clocks higher frequencies when running CUDA applications.
                                Only on supported devices starting from the Volta series.
                                Requires administrator privileges.
                                0/RESTORE_DEFAULT, 1/OVERRIDE
    -am   --accounting-mode=    Enable or disable Accounting Mode: 0/DISABLED, 1/ENABLED
    -caa  --clear-accounted-apps
                                Clears all the accounted PIDs in the buffer.
          --auto-boost-default= Set the default auto boost policy to 0/DISABLED
                                or 1/ENABLED, enforcing the change only after the
                                last boost client has exited.
          --auto-boost-permission=
                                Allow non-admin/root control over auto boost mode:
                                0/UNRESTRICTED, 1/RESTRICTED
    -mig  --multi-instance-gpu= Enable or disable Multi Instance GPU: 0/DISABLED, 1/ENABLED
                                Requires root.
    -gtt  --gpu-target-temp=    Set GPU Target Temperature for a GPU in degree celsius.
                                Requires administrator privileges

   [plus optional]

    -i,   --id=                 Target a specific GPU.
    -eow, --error-on-warning    Return a non-zero error for warnings.

  UNIT MODIFICATION OPTIONS:

    -t,   --toggle-led=         Set Unit LED state: 0/GREEN, 1/AMBER

   [plus optional]

    -i,   --id=                 Target a specific Unit.

  SHOW DTD OPTIONS:

          --dtd                 Print device DTD and exit.

     [plus optional]

    -f,   --filename=           Log to a specified file, rather than to stdout.
    -u,   --unit                Show unit, rather than device, DTD.

    --debug=                    Log encrypted debug information to a specified file.

 Device Monitoring:
    dmon                        Displays device stats in scrolling format.
                                "nvidia-smi dmon -h" for more information.

    daemon                      Runs in background and monitor devices as a daemon process.
                                This is an experimental feature. Not supported on Windows baremetal
                                "nvidia-smi daemon -h" for more information.

    replay                      Used to replay/extract the persistent stats generated by daemon.
                                This is an experimental feature.
                                "nvidia-smi replay -h" for more information.

 Process Monitoring:
    pmon                        Displays process stats in scrolling format.
                                "nvidia-smi pmon -h" for more information.

 TOPOLOGY:
    topo                        Displays device/system topology. "nvidia-smi topo -h" for more information.

 DRAIN STATES:
    drain                       Displays/modifies GPU drain states for power idling. "nvidia-smi drain -h" for more information.

 NVLINK:
    nvlink                      Displays device nvlink information. "nvidia-smi nvlink -h" for more information.

 C2C:
    c2c                         Displays device C2C information. "nvidia-smi c2c -h" for more information.

 CLOCKS:
    clocks                      Control and query clock information. "nvidia-smi clocks -h" for more information.

 ENCODER SESSIONS:
    encodersessions             Displays device encoder sessions information. "nvidia-smi encodersessions -h" for more information.

 FBC SESSIONS:
    fbcsessions                 Displays device FBC sessions information. "nvidia-smi fbcsessions -h" for more information.

 GRID vGPU:
    vgpu                        Displays vGPU information. "nvidia-smi vgpu -h" for more information.

 MIG:
    mig                         Provides controls for MIG management. "nvidia-smi mig -h" for more information.

 COMPUTE POLICY:
    compute-policy              Control and query compute policies. "nvidia-smi compute-policy -h" for more information.

 BOOST SLIDER:
    boost-slider                Control and query boost sliders. "nvidia-smi boost-slider -h" for more information.

 POWER HINT:
    power-hint                  Estimates GPU power usage. "nvidia-smi power-hint -h" for more information.

 BASE CLOCKS:
    base-clocks                 Query GPU base clocks. "nvidia-smi base-clocks -h" for more information.

 CONFIDENTIAL COMPUTE:
    conf-compute                Control and query confidential compute. "nvidia-smi conf-compute -h" for more information.

 GPU PERFORMANCE MONITORING:
    gpm                         Control and query GPU performance monitoring unit. "nvidia-smi gpm -h" for more information.

Please see the nvidia-smi(1) manual page for more detailed information.
```
