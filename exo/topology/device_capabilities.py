from exo import DEBUG
from dataclasses import dataclass, asdict
import psutil
import subprocess
import json

TFLOPS = 1.00

@dataclass
class DeviceFlops:
    # units of TFLOPS
    fp32: float
    fp16: float
    int8: float

    def __str__(self):
        return f"fp32: {self.fp32 / TFLOPS:.2f} TFLOPS, fp16: {self.fp16 / TFLOPS:.2f} TFLOPS, int8: {self.int8 / TFLOPS:.2f} TFLOPS"

    def to_dict(self):
        return asdict(self)


@dataclass
class DeviceCapabilities:
    model: str
    chip: str
    memory: int
    flops: DeviceFlops

    def __str__(self):
        return f"Model: {self.model}. Chip: {self.chip}. Memory: {self.memory}MB. Flops: {self.flops}"

    def __post_init__(self):
        if isinstance(self.flops, dict):
            self.flops = DeviceFlops(**self.flops)

    def to_dict(self):
        return {
            'model': self.model,
            'chip': self.chip,
            'memory': self.memory,
            'flops': self.flops.to_dict()
        }

UNKNOWN_DEVICE_CAPABILITIES = DeviceCapabilities(model="Unknown Model", chip="Unknown Chip", memory=0, flops=DeviceFlops(fp32=0, fp16=0, int8=0))

CHIP_FLOPS = {
    # Source: https://www.cpu-monkey.com
    # Note: currently no distinction between variants of M3 Max and M3 Pro, we pick the lower one to be conservative
    ### M chips
    "Apple M1": DeviceFlops(fp32=2.29*TFLOPS, fp16=4.58*TFLOPS, int8=9.16*TFLOPS),
    "Apple M1 Pro": DeviceFlops(fp32=4.5 * TFLOPS, fp16=9.0 * TFLOPS, int8=18.0 * TFLOPS),
    "Apple M1 Max": DeviceFlops(fp32=10.4 * TFLOPS, fp16=20.8 * TFLOPS, int8=41.6 * TFLOPS),
    "Apple M1 Ultra": DeviceFlops(fp32=21.20*TFLOPS, fp16=42.40*TFLOPS, int8=84.80*TFLOPS),
    "Apple M2": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
    "Apple M2 Pro": DeviceFlops(fp32=5.68*TFLOPS, fp16=11.36*TFLOPS, int8=22.72*TFLOPS),
    "Apple M2 Max": DeviceFlops(fp32=13.49*TFLOPS, fp16=26.98*TFLOPS, int8=53.96*TFLOPS),
    "Apple M2 Ultra": DeviceFlops(fp32=26.98*TFLOPS, fp16=53.96*TFLOPS, int8=107.92*TFLOPS),
    "Apple M3": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
    "Apple M3 Pro": DeviceFlops(fp32=4.97*TFLOPS, fp16=9.94*TFLOPS, int8=19.88*TFLOPS),
    "Apple M3 Max": DeviceFlops(fp32=14.20*TFLOPS, fp16=28.40*TFLOPS, int8=56.80*TFLOPS),

    ### A chips
    "Apple A13 Bionic": DeviceFlops(fp32=0.69*TFLOPS, fp16=1.38*TFLOPS, int8=2.76*TFLOPS),
    "Apple A14 Bionic": DeviceFlops(fp32=0.75*TFLOPS, fp16=1.50*TFLOPS, int8=3.00*TFLOPS),
    "Apple A15 Bionic": DeviceFlops(fp32=1.37*TFLOPS, fp16=2.74*TFLOPS, int8=5.48*TFLOPS),
    "Apple A16 Bionic": DeviceFlops(fp32=1.79*TFLOPS, fp16=3.58*TFLOPS, int8=7.16*TFLOPS),
    "Apple A17 Pro": DeviceFlops(fp32=2.15*TFLOPS, fp16=4.30*TFLOPS, int8=8.60*TFLOPS),
    ### NVIDIA GPUs
    #RTX 40 series
    "Nvidia GeForce RTX 4090": DeviceFlops(fp32=82.58*TFLOPS, fp16=165.16*TFLOPS, int8=330.32*TFLOPS),
    "Nvidia GeForce RTX 4080": DeviceFlops(fp32=48.74*TFLOPS, fp16=97.48*TFLOPS, int8=194.96*TFLOPS),
    "Nvidia GeForce RTX 4080 Super": DeviceFlops(fp32=52.0*TFLOPS, fp16=104.0*TFLOPS, int8=208.0*TFLOPS),
    "Nvidia GeForce RTX 4070 Ti Super": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
    "Nvidia GeForce RTX 4070 Ti": DeviceFlops(fp32=39.43*TFLOPS, fp16=78.86*TFLOPS, int8=157.72*TFLOPS),
    "Nvidia GeForce RTX 4070 Super": DeviceFlops(fp32=30.0*TFLOPS, fp16=60.0*TFLOPS, int8=120.0*TFLOPS),
    "Nvidia GeForce RTX 4070": DeviceFlops(fp32=29.0*TFLOPS, fp16=58.0*TFLOPS, int8=116.0*TFLOPS),
    "Nvidia GeForce RTX 4060 Ti 16GB": DeviceFlops(fp32=22.0*TFLOPS, fp16=44.0*TFLOPS, int8=88.0*TFLOPS),
    #RTX 30 series
    "Nvidia GeForce RTX 3050": DeviceFlops(fp32=9.11*TFLOPS, fp16=18.22*TFLOPS, int8=36.44*TFLOPS),
    "Nvidia GeForce RTX 3060": DeviceFlops(fp32=13.0*TFLOPS, fp16=26.0*TFLOPS, int8=52.0*TFLOPS),
    "Nvidia GeForce RTX 3060 Ti": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
    "Nvidia GeForce RTX 3070": DeviceFlops(fp32=20.3*TFLOPS, fp16=40.6*TFLOPS, int8=81.2*TFLOPS),
    "Nvidia GeForce RTX 3070 Ti": DeviceFlops(fp32=21.8*TFLOPS, fp16=43.6*TFLOPS, int8=87.2*TFLOPS),
    "Nvidia GeForce RTX 3080 (10 GB)": DeviceFlops(fp32=29.8*TFLOPS, fp16=59.6*TFLOPS, int8=119.2*TFLOPS),
    "Nvidia GeForce RTX 3080 (12 GB)": DeviceFlops(fp32=30.6*TFLOPS, fp16=61.2*TFLOPS, int8=122.4*TFLOPS),
    "Nvidia GeForce RTX 3080 Ti": DeviceFlops(fp32=34.1*TFLOPS, fp16=68.2*TFLOPS, int8=136.4*TFLOPS),
    "Nvidia GeForce RTX 3090": DeviceFlops(fp32=35.6*TFLOPS, fp16=71.2*TFLOPS, int8=142.4*TFLOPS),
    "Nvidia GeForce RTX 3090 Ti": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
        # ... add more devices if needed ...
    ### AMD GPUs
    # RX 6000 series
    "AMD Radeon RX 6900 XT": DeviceFlops(fp32=23.04*TFLOPS, fp16=46.08*TFLOPS, int8=92.16*TFLOPS),
    "AMD Radeon RX 6800 XT": DeviceFlops(fp32=20.74*TFLOPS, fp16=41.48*TFLOPS, int8=82.96*TFLOPS),
    "AMD Radeon RX 6800": DeviceFlops(fp32=16.17*TFLOPS, fp16=32.34*TFLOPS, int8=64.68*TFLOPS),
    "AMD Radeon RX 6700 XT": DeviceFlops(fp32=13.21*TFLOPS, fp16=26.42*TFLOPS, int8=52.84*TFLOPS),
    "AMD Radeon RX 6700": DeviceFlops(fp32=11.4*TFLOPS, fp16=22.8*TFLOPS, int8=45.6*TFLOPS),
    "AMD Radeon RX 6600 XT": DeviceFlops(fp32=10.6*TFLOPS, fp16=21.2*TFLOPS, int8=42.4*TFLOPS),
    "AMD Radeon RX 6600": DeviceFlops(fp32=8.93*TFLOPS, fp16=17.86*TFLOPS, int8=35.72*TFLOPS),
    "AMD Radeon RX 6500 XT": DeviceFlops(fp32=5.77*TFLOPS, fp16=11.54*TFLOPS, int8=23.08*TFLOPS),
    "AMD Radeon RX 6400": DeviceFlops(fp32=3.57*TFLOPS, fp16=7.14*TFLOPS, int8=14.28*TFLOPS),
    # RX 7000 series
    "AMD Radeon RX 7900 XTX": DeviceFlops(fp32=61.4*TFLOPS, fp16=122.8*TFLOPS, int8=245.6*TFLOPS),
    "AMD Radeon RX 7900 XT": DeviceFlops(fp32=53.4*TFLOPS, fp16=106.8*TFLOPS, int8=213.6*TFLOPS),
    "AMD Radeon RX 7800 XT": DeviceFlops(fp32=42.6*TFLOPS, fp16=85.2*TFLOPS, int8=170.4*TFLOPS),
    "AMD Radeon RX 7700 XT": DeviceFlops(fp32=34.2*TFLOPS, fp16=68.4*TFLOPS, int8=136.8*TFLOPS),
    "AMD Radeon RX 7600": DeviceFlops(fp32=21.5*TFLOPS, fp16=43.0*TFLOPS, int8=86.0*TFLOPS),
    "AMD Radeon RX 7500": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
    # ... add more devices if needed ...
    ### Qualcomm embedded chips: TODO
}


def device_capabilities() -> DeviceCapabilities:
    if psutil.MACOS:
        return mac_device_capabilities()
    elif psutil.LINUX:
        return linux_device_capabilities()
    else:
        return DeviceCapabilities(model=f"Unknown Device", chip=f"Unknown Chip", memory=psutil.virtual_memory().total // 2**20, flops=DeviceFlops(fp32=0, fp16=0, int8=0))


def mac_device_capabilities() -> DeviceCapabilities:
    hw_info_json = subprocess.check_output(['system_profiler', 'SPHardwareDataType', '-json']).decode('utf-8')
    hw_info = json.loads(hw_info_json)

    # Extract relevant information from the JSON
    hardware_data = hw_info['SPHardwareDataType'][0]
    model_id = hardware_data.get('machine_model', 'Unknown Model')
    chip_id = hardware_data.get('chip_type', 'Unknown Chip')
    cores = hardware_data.get('number_processors', 0)
    memory_str = hardware_data.get('physical_memory', '0 GB')
    memory = int(memory_str.split()[0]) * 1024  # Convert GB to MB

    # If chip_type is not available, try to infer it
    if chip_id == 'Unknown':
        sysctl_output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
        if "Apple M" in sysctl_output:
            chip_id = sysctl_output.split("Apple M")[1].split()[0]
            chip_id = f"Apple M{chip_id}"
            if "Max" in sysctl_output:
                chip_id += " Max"
            elif "Pro" in sysctl_output:
                chip_id += " Pro"
            elif "Ultra" in sysctl_output:
                chip_id += " Ultra"

    flops = CHIP_FLOPS.get(chip_id, DeviceFlops(fp32=0, fp16=0, int8=0))

    if DEBUG >= 1:
        print(f"\nDetailed Mac Device Capabilities:")
        print(f"Model: {model_id}")
        print(f"Chip: {chip_id}")
        print(f"Total Cores: {cores}")
        print(f"Memory: {memory} MB")
        print(f"TFLOPS Calculations:")
        print(f"  FP32: {flops.fp32 / TFLOPS:.2f} TFLOPS")
        print(f"  FP16: {flops.fp16 / TFLOPS:.2f} TFLOPS")
        print(f"  INT8: {flops.int8 / TFLOPS:.2f} TFLOPS")

    return DeviceCapabilities(model=model_id, chip=chip_id, memory=memory, flops=flops)


def linux_device_capabilities() -> DeviceCapabilities:
    import psutil
    from tinygrad import Device

    if DEBUG >= 2: print(f"tinygrad {Device.DEFAULT=}")
    if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT=="GPU":
        import pynvml, pynvml_utils
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        if DEBUG >= 2: print(f"NVIDIA device {gpu_name=} {gpu_memory_info=}")

        return DeviceCapabilities(model=f"Linux Box ({gpu_name})", chip=gpu_name, memory=gpu_memory_info.total // 2**20, flops=CHIP_FLOPS.get(gpu_name, DeviceFlops(fp32=0, fp16=0, int8=0)))
    elif Device.DEFAULT == "AMD":
        # TODO AMD support
        return DeviceCapabilities(model="Linux Box (AMD)", chip="Unknown AMD", memory=psutil.virtual_memory().total // 2**20, flops=DeviceFlops(fp32=0, fp16=0, int8=0))
    else:
        return DeviceCapabilities(model=f"Linux Box (Device: {Device.DEFAULT})", chip=f"Unknown Chip (Device: {Device.DEFAULT})", memory=psutil.virtual_memory().total // 2**20, flops=DeviceFlops(fp32=0, fp16=0, int8=0))
