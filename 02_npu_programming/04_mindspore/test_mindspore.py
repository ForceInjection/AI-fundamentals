import mindspore as ms

print(f"MindSpore version: {ms.__version__}")
ascend_ok = ms.hal.is_available("Ascend")
print(f"Ascend available: {ascend_ok}")
print(f"CPU available: {ms.hal.is_available('CPU')}")

if ascend_ok:
    ms.set_context(device_target="Ascend", device_id=0)
    print(f"Ascend device count: {ms.hal.device_count('Ascend')}")

    # Create a simple tensor on Ascend
    x = ms.Tensor([1.0, 2.0, 3.0], ms.float32)
    print(f"Tensor: {x}")
    print("MindSpore on Ascend OK!")
else:
    print("Ascend not available, running on CPU")
    ms.set_context(device_target="CPU")
