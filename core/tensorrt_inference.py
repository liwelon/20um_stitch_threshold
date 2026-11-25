from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

import numpy as np


class BlindCircuitTensorRt:
    def __init__(self, model_path: Path) -> None:
        if sys.platform != "win32":
            raise RuntimeError("盲元电路区 TensorRT 推理仅支持 Windows。")

        bundle_dir = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else None
        trt_dir = Path(os.environ.get(
            "TENSORRT_LIB_DIR",
            r"D:\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8\TensorRT-8.6.1.6\lib",
        ))
        cuda_dir = Path(os.environ.get(
            "CUDA_BIN_DIR",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        ))
        library_dirs = [
            directory for directory in (bundle_dir, trt_dir, cuda_dir)
            if directory is not None and directory.is_dir()
        ]
        self._dll_handles = [
            os.add_dll_directory(str(directory))
            for directory in library_dirs
        ]
        priority_dirs = list(dict.fromkeys(str(directory) for directory in library_dirs))
        priority_names = {directory.casefold() for directory in priority_dirs}
        existing_dirs = [
            directory for directory in os.environ.get("PATH", "").split(os.pathsep)
            if directory and directory.casefold() not in priority_names
        ]
        os.environ["PATH"] = os.pathsep.join([*priority_dirs, *existing_dirs])
        try:
            import tensorrt as trt
        except Exception as exc:
            for handle in self._dll_handles:
                handle.close()
            raise RuntimeError("当前 Python 环境无法加载 TensorRT 8.6。") from exc

        cuda_binary = next(
            (
                directory / "cudart64_110.dll"
                for directory in (bundle_dir, cuda_dir)
                if directory is not None and (directory / "cudart64_110.dll").is_file()
            ),
            Path("cudart64_110.dll"),
        )
        self._cuda = ctypes.WinDLL(str(cuda_binary))
        self._cuda.cudaGetErrorString.argtypes = [ctypes.c_int]
        self._cuda.cudaGetErrorString.restype = ctypes.c_char_p
        self._cuda.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._cuda.cudaStreamCreate.restype = ctypes.c_int
        self._cuda.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._cuda.cudaStreamDestroy.restype = ctypes.c_int
        self._cuda.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._cuda.cudaStreamSynchronize.restype = ctypes.c_int
        self._cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._cuda.cudaMalloc.restype = ctypes.c_int
        self._cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self._cuda.cudaFree.restype = ctypes.c_int
        self._cuda.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,
        ]
        self._cuda.cudaMemcpyAsync.restype = ctypes.c_int

        self._runtime = None
        self._engine = None
        self._context = None
        self._stream = ctypes.c_void_p()
        self._device_input = ctypes.c_void_p()
        self._device_output = ctypes.c_void_p()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(logger)
            self._engine = self._runtime.deserialize_cuda_engine(model_path.read_bytes())
            if self._engine is None:
                raise RuntimeError(f"无法加载盲元电路区 engine：{model_path}")
            if (
                self._engine.num_io_tensors != 2
                or self._engine.get_tensor_mode("image") != trt.TensorIOMode.INPUT
                or self._engine.get_tensor_mode("probability") != trt.TensorIOMode.OUTPUT
                or self._engine.get_tensor_dtype("image") != trt.float32
                or self._engine.get_tensor_dtype("probability") != trt.float32
            ):
                raise RuntimeError("盲元电路区 engine 输入输出不匹配。")
            self._context = self._engine.create_execution_context()
            if self._context is None or not self._context.set_input_shape(
                "image", (1, 3, 512, 512)
            ):
                raise RuntimeError("无法设置盲元电路区 engine 输入尺寸。")
            if tuple(self._context.get_tensor_shape("probability")) != (1, 1, 512, 512):
                raise RuntimeError("盲元电路区 engine 输出尺寸不匹配。")

            self._check(self._cuda.cudaStreamCreate(ctypes.byref(self._stream)))
            self._check(self._cuda.cudaMalloc(
                ctypes.byref(self._device_input), 3 * 512 * 512 * 4,
            ))
            self._check(self._cuda.cudaMalloc(
                ctypes.byref(self._device_output), 512 * 512 * 4,
            ))
            if (
                not self._context.set_tensor_address("image", self._device_input.value)
                or not self._context.set_tensor_address(
                    "probability", self._device_output.value
                )
            ):
                raise RuntimeError("无法绑定盲元电路区 engine 输入输出。")
        except Exception:
            self.close()
            raise

    def _check(self, result: int) -> None:
        if result != 0:
            message = self._cuda.cudaGetErrorString(result)
            detail = message.decode("utf-8", errors="replace") if message else str(result)
            raise RuntimeError(f"TensorRT CUDA 错误：{detail}")

    def infer(self, blob: np.ndarray) -> np.ndarray:
        if blob.shape != (1, 3, 512, 512) or blob.dtype != np.float32:
            raise ValueError("盲元电路区 engine 输入格式不匹配。")
        source = np.ascontiguousarray(blob)
        output = np.empty((1, 1, 512, 512), dtype=np.float32)
        self._check(self._cuda.cudaMemcpyAsync(
            self._device_input, ctypes.c_void_p(source.ctypes.data),
            source.nbytes, 1, self._stream,
        ))
        if not self._context.execute_async_v3(self._stream.value):
            raise RuntimeError("盲元电路区 engine 推理失败。")
        self._check(self._cuda.cudaMemcpyAsync(
            ctypes.c_void_p(output.ctypes.data), self._device_output,
            output.nbytes, 2, self._stream,
        ))
        self._check(self._cuda.cudaStreamSynchronize(self._stream))
        return output

    def close(self) -> None:
        if self._device_output.value:
            self._cuda.cudaFree(self._device_output)
            self._device_output = ctypes.c_void_p()
        if self._device_input.value:
            self._cuda.cudaFree(self._device_input)
            self._device_input = ctypes.c_void_p()
        if self._stream.value:
            self._cuda.cudaStreamDestroy(self._stream)
            self._stream = ctypes.c_void_p()
        self._context = None
        self._engine = None
        self._runtime = None
        for handle in self._dll_handles:
            handle.close()
        self._dll_handles.clear()

    def __enter__(self) -> BlindCircuitTensorRt:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
