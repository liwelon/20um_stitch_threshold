#pragma once
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;


class CommonParmas
{
public:
	CommonParmas() = default;
	~CommonParmas() = default;

	// 固定参数
	// 镜头参数
	const float fovW = 11.28F;				// 5um视场: 6.452f * 5.857f, 单位：mm
	const float fovH = 10.24F;				// 镜头视野宽高，单位：mm

	// 扫描参数
	const float scanOverlap = 1.0F;			// 扫描重叠大小，单位：mm

	const int singleW = 4512;
	const int singleH = 4096;
	const int channel = 3;					// 单张图片的高宽和通道数（根据相机而来，固定数据）

	// 使用参数
	// 大载盘参数，单位：mm
	float largeW = 320.0F;
	float largeH = 350.0F;				// 宽高
	float largeLeftMargin = 5.5F;
	float largeRightMargin = 5.5F;
	float largeTopMargin = 15.0F;
	float largeBottomMargin = 15.0F;	// 左右上下面边距
	float largeStepX = 80.0F;			// 79.0 -> 80.0
	float largeStepY = 110.0F;			// X轴和Y轴步长 110.0 -> 109.8
	int largeRows = 3;
	int largeCols = 4;					// 行数和列数
	//float largeCellW = 70.4f;
	//float largeCellH = 98.4f;			// 大载盘中每个孔的宽高, 后面更新的时候要加上

	// 小载盘参数，单位：mm
	float smallW = 70.0f;
	float smallH = 98.0f;				// 宽高
	float smallLeftMargin = 3.0f;
	float smallRightMargin = 3.0f;
	float smallTopMargin = 4.0f;
	float smallBottomMargin = 4.0f;		// 左右上下面边距
	float smallStepX = 12.85f;
	float smallStepY = 13.22f;			// X轴和Y轴步长R
	int smallRows = 7;
	int smallCols = 5;					// 行数和列数
	//float smallCellW = 11.6f;
	//float smallCellH = 10.9f;			// 小载盘中每个孔的宽高

	// 芯片参数，单位：mm
	float chipW = 11.6f;
	float chipH = 10.9f;
	float chipDZ = 1.5f;				// 对焦时的Z轴高度，单位：mm
	double capZeroPos = 0.0;			// cap高度距离z轴零点的距离，单位：mm

	// 图像拼接位移，由芯片配方传入，单位：pixel
	int hDx = 0;
	int hDy = 0;
	int vDy = 0;
	int vDx = 0;

	// 图片拼接参数（计算而来的参数）
	int jointRows = 0;
	int jointCols = 0;							// 水平和垂直拼接的行数和列数，根据载盘参数和芯片参数计算得到

	// 各种文件路径
	std::string tempPath = "datas/temp.jpg";	// 模板图片路径（芯片目标）
	std::string routingPath = "";				// 运动路径保存路径
	std::string chipRoutingPath = "";			// 芯片内运动路径保存路径（一块芯片的运动）

	// 根据模板派生的参数
	int tempW0 = 0;
	int tempH0 = 0;								// 也可以直接从spliter中获取，这里直接读取，保持模块独立性

	/* ########################## 以上为芯片配方中保存的数据 ########################## */
	/* ############################ 以下为缺陷检测参数数据 ########################## */

	// 图片拼接参数(用户输入参数）
	cv::Size stackKSize = cv::Size(21, 21);
	int stackSigma = 0;					// 对焦融合的高斯模糊核大小和sigma（根据经验而来，固定数据）

	// 目标裁剪参数
	double ratio = 0.1;					// 数据处理时，图像的压缩倍数，加速计算
	cv::Size ksize = cv::Size(5, 5);
	double sigmax = 0;					// 高斯模糊核大小和sigma（根据经验而来，固定数据）

	double matchedThresh = 0.65;				// 模板匹配的阈值，低于该阈值认为匹配结果置信度低

	int correctOffset = 30;						// 匹配到的cmos初始点 x - correctOffset
	int correctAngleSearchRange = 150;			// 查找pad的宽度以及上下偏移的范围，单位：pixel（只检查一边）
	int correctGrayThresh = 220;				// pad寻找时，灰度阈值
	double padMinArea = 700;					// pad寻找时，最小面积，单位：pixel
	int padMaxDist = 20;						// pad寻找时，两个pad之间的最大距离，单位：pixel
	int padMinNum = 5;							// pad寻找时，最少pad数量

	//double capGrayThreshRatio = 0.9;			// cap灰度均值的 90% 作为阈值
	double capSplitGrayThresh = 100;

	cv::Scalar capLowerHSV;
	cv::Scalar capUpperHSV;

	double capMinAreaRatio = 0.75;
	double capMaxAreaRatio = 0.9;				// temp面积的 75% ~ 90% 作为cap面积范围

	int capOpenSize = 11;						// cap开运算的核大小，单位：pixel
	int capOpenIteration = 5;					// cap开运算的迭代次数

	// 缺陷检测参数
	std::string onnxPath = "";					// onnx模型路径
	double confThreshold = 0.3;					// 缺陷检测时，置信度阈值，低于该阈值认为是伪缺陷
	double nmsThreshold = 0.5;					// 缺陷检测时，nms的阈值，iou大于该值认为是重复框
	double baseFactor = 1.0;

	bool filter = false;                        // true:不显示成像区缺陷 false:显示全部缺陷

	double umPerPix = 2.5;						// 固定参数，根据相机和镜头计算而来，单位：um/pix
	float imagingLeft = 1.1;
	float imagingTop = 0.9;
	float imagingRight = 1.1;
	float imagingBottom = 2.2;					// 成像区边距，单位：mm，转换为像素时除以umPerPix

	cv::Rect cmosRect;							// 人工画出CMOS区域（CMOS缺陷检测用）
	cv::Rect cornerRect;						// corner mask匹配区域（独立于cmosRect）, 空=用整张模板

	double warpSearchRatio = 0.3;				// 翘起检测时，搜索区域占整个图像的比例（两侧的比例）
	int warpGrayThresh = 70;					// 翘起检测时，灰度阈值

	int padOffsetPix = 50;						// pad的内边界距离cmos的距离，单位：pixel
	int offsetThreshUm = 260;					// offset检测时，允许的最大偏移，单位：um

	int blurErode = 20;							// cap时，消除胶影响，内缩范围，单位：pixel
	int capEdgeEps = 30;						// cap时，小于该值的点认为是边缘点，已加上 blurErode
	float capChipLength = 1.0f;					// cap崩边的最小长度，单位：mm，转换为像素时除以umPerPix
	float capChipWidth = 0.3f;					// cap崩边的最小宽度，单位：mm，转换为像素时除以umPerPix

	int capGrayThresh = 100;					// 传统算法检测时，二值化的灰度阈值
	int capBinaryThresh = 96;                   // 模型检测时，二值化阈值

	double capResizeR = 0.1;					// cap缺陷检测时，图像缩放倍数，加速计算，同时，忽略小缺陷并且合并离散缺陷
	int capDefectMinArea = 200;					// cap缺陷检测时，最小缺陷面积，单位：pixel
	double capPointMinArea = 100;
	double capDirtyMinArea = 100;
	//double capScratchMinArea = 100;

	// 划伤最小粗细阈值，单位：mm，小于该值过滤
	double capScratchMinWidth = 0.02;
	//double capBinaryWhiteRatio = 0.30;          // 二值化后缺陷面积1占比
        double capScratchMinLength = 1; // 划伤长度过滤
        int capScratchCount = 3;// 划伤数量

    // 缺陷颜色深浅过滤阈值（0~1）
    // 值越大表示缺陷越深
	double capPointGrayThresh = 0.4;
	double capDirtyGrayThresh = 0.4;
	double capScratchGrayThresh = 0.4;
	int blurExpand = 100;						// cmos缺陷检测时，扩展缺陷边界的范围，单位：pixel
	int cmosGrayThresh = 20;					// cmos缺陷检测时，二值化的灰度阈值
	int cmosEdgeX0 = 10;						// 小于该值认为是左边缘，右边缘阈值，根据左边缘计算
	int cmosEdgeY0 = 10;
	int cmosChipMinWidth = 40;					// cmos崩边的最小宽度，单位：pixel（超过安全线为崩边，这里直接转换为具体值）
	int cmosDirtyMinArea = 100;					// cmos脏污的最小面积，单位：pixel

	int cornerW = 0;                     // 四角mask矩形宽(原图px), 0=禁用corner matching
	int cornerH = 0;                     // 四角mask矩形高(原图px)

	double capPointMinArea2 = 0.02;
    double capDirtyMinArea2 = 0.05;
	double capPointGrayThresh2 = 0.4;
	double capDirtyGrayThresh2 = 0.4;

	int offsetY0 = 0; // 窗片偏移

	// 参数方法
	void readAndUpdateParams(const json& config);
	void readAndUpdateParams(const std::string configPath);	// 读取配置文件并更新参数
};




