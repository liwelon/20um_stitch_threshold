# -*- mode: python ; coding: utf-8 -*-
import os
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

if sys.version_info[:2] != (3, 11):
    raise RuntimeError('打包必须使用 Python 3.11，TensorRT 8.6 不支持当前 Python 版本。')
if find_spec('tensorrt') is None:
    raise RuntimeError('当前 Python 3.11 环境没有安装 TensorRT 8.6。')

project_root = Path.cwd()
trt_lib_dir = Path(os.environ.get(
    'TENSORRT_LIB_DIR',
    r'D:\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8\TensorRT-8.6.1.6\lib',
))
cuda_bin_dir = Path(os.environ.get(
    'CUDA_BIN_DIR',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
))
cudnn_bin_dir = Path(os.environ.get(
    'CUDNN_BIN_DIR',
    r'D:\cudnn-windows-x86_64-8.9.6.50_cuda11-archive\cudnn-windows-x86_64-8.9.6.50_cuda11-archive\bin',
))
trt_binaries = [
    (str(directory / name), '.')
    for directory, names in (
        (trt_lib_dir, (
            'nvinfer.dll', 'nvinfer_plugin.dll',
            'nvonnxparser.dll', 'nvparsers.dll',
        )),
        (cuda_bin_dir, (
            'cudart64_110.dll', 'cublas64_11.dll', 'cublasLt64_11.dll',
        )),
        (cudnn_bin_dir, ('cudnn64_8.dll',)),
    )
    for name in names
]
for source_path, _destination in trt_binaries:
    if not Path(source_path).is_file():
        raise FileNotFoundError(f'打包依赖不存在：{source_path}')
hook_dir = project_root / 'build' / 'packaging_hooks'
hook_dir.mkdir(parents=True, exist_ok=True)
(hook_dir / 'hook-torch.py').write_text(
    "from PyInstaller.utils.hooks import collect_dynamic_libs\n"
    "\n"
    "binaries = collect_dynamic_libs('torch')\n"
    "module_collection_mode = 'pyz+py'\n",
    encoding='utf-8',
)
runtime_hook = hook_dir / 'runtime_torch.py'
runtime_hook.write_text(
    "import os\n"
    "\n"
    "os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'\n",
    encoding='utf-8',
)
dist_app = project_root / 'dist' / 'ImageBinarizationTool'
config_dir = dist_app / 'config'
# 放在 packaging 下，避免被 pyinstaller --clean 清掉 build/
config_backup = project_root / 'packaging' / '_config_backup'

if config_dir.is_dir():
    if config_backup.exists():
        shutil.rmtree(config_backup)
    shutil.copytree(config_dir, config_backup)

a = Analysis(
    [str(project_root / 'main.py')],
    pathex=[str(project_root)],
    binaries=trt_binaries,
    datas=[
        (str(path), 'resources')
        for path in (project_root / 'resources').glob('params_*um.json')
    ],
    hiddenimports=['tensorrt'],
    hookspath=[str(hook_dir)],
    hooksconfig={},
    runtime_hooks=[str(runtime_hook)],
    excludes=[
        'PyQt5',
        'IPython',
        'astropy',
        'bokeh',
        'dask',
        'distributed',
        'docutils',
        'fsspec',
        'h5py',
        'intake',
        'ipykernel',
        'jsonschema',
        'jupyterlab',
        'lxml',
        'matplotlib',
        'narwhals',
        'nbconvert',
        'nbformat',
        'notebook',
        'numba',
        'llvmlite',
        'pandas',
        'panel',
        'plotly',
        'pyarrow',
        'pydantic',
        'pyviz_comms',
        'pytest',
        'scipy',
        'skimage',
        'sklearn',
        'sphinx',
        'shapely',
        'sqlalchemy',
        'statsmodels',
        'torch.utils.tensorboard',
        'torchvision',
        'xarray',
        'xyzservices',
    ],
    noarchive=False,
    optimize=0,
)
a.binaries = [
    entry for entry in a.binaries
    if Path(entry[0]).name.lower() not in {'icuuc.dll', 'icudt78.dll'}
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ImageBinarizationTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ImageBinarizationTool',
)

# COLLECT 会重建 dist 目录；把打包前的 config 还原回去
if config_backup.is_dir():
    restored = dist_app / 'config'
    if restored.exists():
        shutil.rmtree(restored)
    shutil.copytree(config_backup, restored)
    shutil.rmtree(config_backup)
