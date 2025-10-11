#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 简单验证pynvml - 打印显卡数量
import pynvml
pynvml.nvmlInit()
print("GPU Count: ", pynvml.nvmlDeviceGetCount())
pynvml.nvmlShutdown()

# ----------------------------【分割线】----------------------------

# 简单使用 - 获取显卡的各项信息
import pynvml
pynvml.nvmlInit()
print("GPU Driver Version: ", pynvml.nvmlSystemGetDriverVersion())  # 显示驱动信息 Driver:  572.83

# 查看显卡设备列表
deviceCount = pynvml.nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    print("\n", f"GPU {i}:", pynvml.nvmlDeviceGetName(handle)) # GPU 0 : NVIDIA GeForce RTX 2080 Ti

    # 查看显存、温度、风扇、电源
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("\t", "Memory Total: ", info.total, "|", f"{float(info.total)/1024/1024/1024:.2f}", "GB")
    print("\t", "Memory Free: ", info.free, "|", f"{float(info.free)/1024/1024/1024:.2f}", "GB")
    print("\t", "Memory Used: ", info.used, "|", f"{float(info.used)/1024/1024/1024:.2f}", "GB")
    print("\t", "Temperature: %d℃" % pynvml.nvmlDeviceGetTemperature(handle, 0))
    print("\t", "Fan Speed: ", pynvml.nvmlDeviceGetFanSpeed(handle))
    print("\t", "Power State: ", pynvml.nvmlDeviceGetPowerState(handle))

# 最后要关闭管理工具
pynvml.nvmlShutdown()

# nvmlDeviceXXX有一系列函数可以调用，包括了NVML的大多数函数。
# 具体可以参考：https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries

"""
输出示例：
GPU Count:  2
GPU Driver Version:  572.83

 GPU 0: NVIDIA GeForce RTX 2080 Ti
         Memory Total:  23622320128 | 22.00 GB
         Memory Free:  8432332800 | 7.85 GB
         Memory Used:  15189987328 | 14.15 GB
         Temperature: 37℃
         Fan Speed:  27
         Power State:  8

 GPU 1: NVIDIA GeForce RTX 2080 Ti
         Memory Total:  23622320128 | 22.00 GB
         Memory Free:  8433250304 | 7.85 GB
         Memory Used:  15189069824 | 14.15 GB
         Temperature: 37℃
         Fan Speed:  27
         Power State:  8
"""