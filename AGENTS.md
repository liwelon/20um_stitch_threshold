# 项目协作说明

## 项目概览

这是一个基于 PySide6、OpenCV 和 NumPy 的 Windows 桌面工具，用于显微小图拼接、单图加载、旋转、二值化预览和 ROI 裁剪。

入口是根目录的 `main.py`。主窗口采用 Qt Widgets 构建，图像处理和拼接算法与界面分离。

## 代码导航

### 启动与主界面

- `main.py`
  - `main()`：创建 `QApplication`、显示 `MainWindow`。
  - `place_window()`：设置窗口初始位置。
- `ui/main_window.py`
  - `MainWindow._build_ui()`：右侧控制面板和左侧图像视图的全部布局。
  - `MainWindow.start_stitch()`：启动后台拼接线程。
  - `_on_stitch_batch_finished()`、`_on_stitch_finished()`、`_on_stitch_failed()`：处理拼接线程结果。
  - `load_image()`、`set_working_image()`：加载和切换当前工作图像。
  - `update_binary_image()`：按阈值更新二值图。
  - `set_display_mode()`：切换原图、二值图和并排显示。
  - `crop_and_save()`、`_on_roi_changed()`：ROI 裁剪及偏移显示。

### 图像基础工具

- `core/image_utils.py`
  - `imread_unicode()`、`imwrite_unicode()`：支持 Unicode 路径的图片读写。
  - `natural_sort_key()`：小图文件名自然排序。
  - `numpy_to_qimage()`：NumPy 图像转换为 Qt 图像。
  - `rotate_image()`：图像旋转。

### 拼接算法与参数

- `core/stitching/image_jointer.py`
  - `ImageJointer.horizontal_joint()`、`vertical_joint()`：横向和纵向拼接。
  - `joint_up_down()`：上下层拼接。
  - `joint_single_layer()`：单层拼接。
- `core/stitching/params.py`
  - `JointParams`：拼接参数数据结构。
  - `params_json_path()`：定位当前分辨率参数文件。
  - `get_joint_params()`、`get_machine_presets()`：读取机台配方。
  - `save_machine_preset()`、`delete_machine_preset()`：保存和删除配方。
- `config/params_5um.json`、`resources/params_*um.json`：机台配方数据。修改参数前先确认分辨率和机台 ID。

### 后台任务

- `workers/stitch_worker.py`
  - `StitchWorker.run()`：读取图片、调用 `ImageJointer`，通过 Qt 信号回传批次结果。

### 自定义控件与对话框

- `widgets/image_view.py`：缩放、平移、双视图同步和 ROI 交互。
- `dialogs/recipe_manager.py`：机台配方增删改和未保存变更确认。
- `dialogs/stitch_calibration.py`：拼接标定画布和参数写入。

### 打包

- `packaging/ImageBinarizationTool.spec`：PyInstaller 配置。
- 打包命令：`pyinstaller --noconfirm --clean packaging\\packaging.spec`
- 输出：`dist/ImageBinarizationTool/`

## 按需求定位文件

| 需求 | 首先读取 |
| --- | --- |
| 主界面布局、控件分组、样式 | `ui/main_window.py` 的 `_build_ui()` |
| 拼接按钮、进度、结果切换 | `ui/main_window.py` 的拼接相关方法；必要时再读 `workers/stitch_worker.py` |
| 单张图、阈值、显示模式、ROI | `ui/main_window.py` 的 `load_image()`、`update_binary_image()`、`set_display_mode()`、`crop_and_save()` |
| 拼接位置、尺寸、重叠参数 | `core/stitching/image_jointer.py` 和 `core/stitching/params.py` |
| 配方管理或标定对话框 | 只读对应的 `dialogs/*.py` |
| 图片路径、编码或旋转 | `core/image_utils.py` |
| 打包失败 | `packaging/ImageBinarizationTool.spec`，然后查看 `build/ImageBinarizationTool/warn-ImageBinarizationTool.txt` |

不要为了修改主界面而读取 `cpp/`、全部对话框实现或完整算法文件；只有需求涉及对应功能时再展开读取。

## 修改约定

- 修改生产代码前，先列出具体改动点并等待确认。
- 不修改算法常量，除非需求明确要求；代码语言转换时保持常量原值。
- 遵循现有 PySide6、类型标注和命名风格，不新增无必要的抽象层。
- 禁止 emoji、TODO、占位实现和无关重构。
- UI 修改优先保持现有控件对象名和信号连接，避免破坏业务逻辑。
- 修改后至少运行 `python -m compileall -q <changed_file>`；界面改动可使用 `QT_QPA_PLATFORM=offscreen` 做初始化检查。
- 打包改动后执行 spec 打包命令，并记录输出目录和关键警告。

## 运行环境

- Windows，时区 `Asia/Shanghai`，编码 UTF-8。
- Python 依赖见 `requirements.txt`。
- 中国网络环境安装依赖时使用清华 pip 镜像：`https://pypi.tuna.tsinghua.edu.cn/simple`。
