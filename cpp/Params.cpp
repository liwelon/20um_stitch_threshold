#include "Params.h"
#include <algorithm>

//CommonParmas::CommonParmas(const std::string jsonPath)
//{
//	if (jsonPath.empty()) {
//		std::cerr << "json file is emptyp." << std::endl;
//		return;
//	}
//
//	// 读取配置文件
//	std::ifstream configFile(jsonPath);
//	if (!configFile.is_open()) {
//		std::cerr << "Failed to open config file." << std::endl;
//		return;
//	}
//	json config;
//	configFile >> config;
//	configFile.close();
//
//	// 更新配置到成员变量
//	// 配方参数
//	largeW = config["largeW"].get<float>();
//	largeH = config["largeH"].get<float>();
//	largeLeftMargin = config["largeLeftMargin"].get<float>();
//	largeRightMargin = config["largeRightMargin"].get<float>();
//	largeTopMargin = config["largeTopMargin"].get<float>();
//	largeBottomMargin = config["largeBottomMargin"].get<float>();
//	largeStepX = config["largeStepX"].get<float>();
//	largeStepY = config["largeStepY"].get<float>();
//	largeRows = config["largeRows"].get<int>();
//	largeCols = config["largeCols"].get<int>();
//
//	smallW = config["smallW"].get<float>();
//	smallH = config["smallH"].get<float>();
//	smallLeftMargin = config["smallLeftMargin"].get<float>();
//	smallRightMargin = config["smallRightMargin"].get<float>();
//	smallTopMargin = config["smallTopMargin"].get<float>();
//	smallBottomMargin = config["smallBottomMargin"].get<float>();
//	smallStepX = config["smallStepX"].get<float>();
//	smallStepY = config["smallStepY"].get<float>();
//	smallRows = config["smallRows"].get<int>();
//	smallCols = config["smallCols"].get<int>();
//
//	chipW = config["chipW"].get<float>();
//	chipH = config["chipH"].get<float>();
//
//	// 图片拼接参数
//	hDx = config["hDx"].get<int>();
//	hDy = config["hDy"].get<int>();
//	vDy = config["vDy"].get<int>();
//	vDx = config["vDx"].get<int>();
//
//	singleH = config["singleH"].get<int>();
//	singleW = config["singleW"].get<int>();
//	channel = config["channel"].get<int>();
//
//	stackKSize = cv::Size(config["statckSize"].get<int>(), config["statckSize"].get<int>());
//	stackSigma = config["stackSigma"].get<int>();
//
//	// 目标裁剪参数
//	ratio = config["ratio"].get<double>();
//	ksize = cv::Size(config["ksize"].get<int>(), config["ksize"].get<int>());
//	sigmax = config["sigmax"].get<double>();
//
//	tempPath = config["tempPath"].get<std::string>();
//	matchedThresh = config["matchedThresh"].get<double>();
//
//	correctOffset = config["correctOffset"].get<int>();
//	correctAngleSearchRange = config["correctAngleSearchRange"].get<int>();
//	correctGrayThresh = config["correctGrayThresh"].get<int>();
//	padMinArea = config["padMinArea"].get<double>();
//	padMaxDist = config["padMaxDist"].get<int>();
//	padMinNum = config["padMinNum"].get<int>();
//
//	capGrayThreshRatio = config["capGrayThreshRatio"].get<double>();
//	capMinAreaRatio = config["capMinAreaRatio"].get<double>();
//	capMaxAreaRatio = config["capMaxAreaRatio"].get<double>();
//
//	// 缺陷检测参数
//	umPerPix = config["umPerPix"].get<double>();
//	imagingLeft = config["imagingLeft"].get<float>();
//	imagingTop = config["imagingTop"].get<float>();
//	imagingRight = config["imagingRight"].get<float>();
//	imagingBottom = config["imagingBottom"].get<float>();
//
//	warpSearchRatio = config["warpSearchRatio"].get<double>();
//	warpGrayThresh = config["warpGrayThresh"].get<int>();
//
//	padOffsetPix = config["padOffsetPix"].get<int>();
//	offsetThreshUm = config["offsetThreshUm"].get<double>();
//
//	blurErode = config["blurErode"].get<int>();
//	capEdgeEps = config["capEdgeEps"].get<int>();
//	capChipLength = config["capChipLength"].get<int>();
//	capChipWidth = config["capChipWidth"].get<int>();
//
//	capGrayThresh = config["capGrayThresh"].get<int>();
//	capResizeR = config["capResizeR"].get<double>();
//	capDefectMinArea = config["capDefectMinArea"].get<int>();
//
//	blurExpand = config["blurExpand"].get<int>();
//	cmosGrayThresh = config["cmosGrayThresh"].get<int>();
//	cmosEdgeX0 = config["cmosEdgeX0"].get<int>();
//	cmosChipMinWidth = config["cmosChipMinWidth"].get<int>();
//	cmosDirtyMinArea = config["cmosDirtyMinArea"].get<int>();
//
//	// 根据配置参数，计算并更新相关的派生参数
//}

//void CommonParmas::readReceipe(const std::string receipePath)
//{
//	// 读取配方文件
//	std::ifstream configFile(receipePath);
//	if (!configFile.is_open()) {
//		std::cerr << "Failed to open config file." << std::endl;
//		return;
//	}
//
//	json receipe;
//	configFile >> receipe;
//	configFile.close();
//
//	// 配方参数
//	largeW = receipe["largeW"].get<float>();
//	largeH = receipe["largeH"].get<float>();
//	largeLeftMargin = receipe["largeLeftMargin"].get<float>();
//	largeRightMargin = receipe["largeRightMargin"].get<float>();
//	largeTopMargin = receipe["largeTopMargin"].get<float>();
//	largeBottomMargin = receipe["largeBottomMargin"].get<float>();
//	largeStepX = receipe["largeStepX"].get<float>();
//	largeStepY = receipe["largeStepY"].get<float>();
//	largeRows = receipe["largeRows"].get<int>();
//	largeCols = receipe["largeCols"].get<int>();
//
//	smallW = receipe["smallW"].get<float>();
//	smallH = receipe["smallH"].get<float>();
//	smallLeftMargin = receipe["smallLeftMargin"].get<float>();
//	smallRightMargin = receipe["smallRightMargin"].get<float>();
//	smallTopMargin = receipe["smallTopMargin"].get<float>();
//	smallBottomMargin = receipe["smallBottomMargin"].get<float>();
//	smallStepX = receipe["smallStepX"].get<float>();
//	smallStepY = receipe["smallStepY"].get<float>();
//	smallRows = receipe["smallRows"].get<int>();
//	smallCols = receipe["smallCols"].get<int>();
//
//	chipW = receipe["chipW"].get<float>();
//	chipH = receipe["chipH"].get<float>();
//}
//
//void CommonParmas::buildNewConfig(const std::string receipePath, const std::string routingDir, const std::string configPath)
//{
//	this->readReceipe(receipePath);
//	this->movePath.updateParams(largeW, largeH, largeLeftMargin, largeRightMargin, largeTopMargin, largeBottomMargin, largeStepX, largeStepY, largeRows, largeCols,
//		smallW, smallH, smallLeftMargin, smallRightMargin, smallTopMargin, smallBottomMargin, smallStepX, smallStepY, smallRows, smallCols,
//		chipW, chipH,
//		fovW, fovH,
//		scanOverlap);
//	this->movePath.calcMoveRouting(routingDir);
//
//	// 更新其他参数
//	this->movePath.getMoveInChipInfo(jointRows, jointCols);		// 关键参数，后面很多参数都需要用到
//
//}

void CommonParmas::readAndUpdateParams(const std::string configPath)
{
	// 读取配置文件
	std::ifstream configFile(configPath);
	if (!configFile.is_open()) {
		std::cerr << "Failed to open config file." << std::endl;
		return;
	}
	json config;
	configFile >> config;
	configFile.close();
	readAndUpdateParams(config);
}

void CommonParmas::readAndUpdateParams(const json& config)
{

	//std::cout << config.dump() << "\n";

	// 读取所有参数（包括要计算的参数）
	// 配方参数
	largeW = config["largeW"].get<float>();
	largeH = config["largeH"].get<float>();
	largeLeftMargin = config["largeLeftMargin"].get<float>();
	largeRightMargin = config["largeRightMargin"].get<float>();
	largeTopMargin = config["largeTopMargin"].get<float>();
	largeBottomMargin = config["largeBottomMargin"].get<float>();
	largeStepX = config["largeStepX"].get<float>();
	largeStepY = config["largeStepY"].get<float>();
	largeRows = config["largeRows"].get<int>();
	largeCols = config["largeCols"].get<int>();

	smallW = config["smallW"].get<float>();
	smallH = config["smallH"].get<float>();
	smallLeftMargin = config["smallLeftMargin"].get<float>();
	smallRightMargin = config["smallRightMargin"].get<float>();
	smallTopMargin = config["smallTopMargin"].get<float>();
	smallBottomMargin = config["smallBottomMargin"].get<float>();
	smallStepX = config["smallStepX"].get<float>();
	smallStepY = config["smallStepY"].get<float>();
	smallRows = config["smallRows"].get<int>();
	smallCols = config["smallCols"].get<int>();

	chipW = config["chipW"].get<float>();
	chipH = config["chipH"].get<float>();
	chipDZ = config["chipDZ"].get<float>();
	capZeroPos = config["capZeroPos"].get<double>();
	hDx = config["hDx"].get<int>();
	hDy = config["hDy"].get<int>();
	vDx = config["vDx"].get<int>();
	vDy = config["vDy"].get<int>();

	jointRows = config["jointRows"].get<int>();
	jointCols = config["jointCols"].get<int>();

	tempPath = config["tempPath"].get<std::string>();
	routingPath = config["routingPath"].get<std::string>();
	chipRoutingPath = config["chipRoutingPath"].get<std::string>();

	// 目标裁剪参数
	ratio = config["ratio"].get<double>();
	ksize = cv::Size(config["ksize"].get<int>(), config["ksize"].get<int>());
	sigmax = config["sigmax"].get<double>();

	matchedThresh = config["matchedThresh"].get<double>();

	correctOffset = config["correctOffset"].get<int>();
	correctAngleSearchRange = config["correctAngleSearchRange"].get<int>();
	correctGrayThresh = config["correctGrayThresh"].get<int>();
	padMinArea = config["padMinArea"].get<double>();
	padMaxDist = config["padMaxDist"].get<int>();
	padMinNum = config["padMinNum"].get<int>();

	//capGrayThreshRatio = config["capGrayThreshRatio"].get<double>();
	//capSplitGrayThresh = config["capSplitGrayThresh"].get<double>();		// 灰度阈值不稳定（溢胶不稳定）
	int capSplitH0 = config.value("capSplitH0", 75);
	int capSplitS0 = config.value("capSplitS0", 100);
	int capSplitV0 = config.value("capSplitV0", 110);
	int capSplitH1 = config.value("capSplitH1", 85);
	int capSplitS1 = config.value("capSplitS1", 130);
	int capSplitV1 = config.value("capSplitV1", 150);

	capLowerHSV = cv::Scalar(capSplitH0, capSplitS0, capSplitV0);
	capUpperHSV = cv::Scalar(capSplitH1, capSplitS1, capSplitV1);

	capMinAreaRatio = config.value("capMinAreaRatio", 0.75);
	capMaxAreaRatio = config.value("capMaxAreaRatio", 0.9);

	capOpenSize = config.value("capOpenSize", 11);
	capOpenIteration = config.value("capOpenIteration", 5);

	// 缺陷检测参数
	onnxPath = config["onnxPath"].get<std::string>();
	confThreshold = config["confThreshold"].get<double>();
	nmsThreshold = config["nmsThreshold"].get<double>();
	baseFactor = config["baseFactor"].get<double>();
	filter = config.value("filter", false);

	umPerPix = config["umPerPix"].get<double>();
	imagingLeft = config["imagingLeft"].get<float>();
	imagingTop = config["imagingTop"].get<float>();
	imagingRight = config["imagingRight"].get<float>();
	imagingBottom = config["imagingBottom"].get<float>();

	int cmosLeftPix = config["cmosLeftPix"].get<int>();
	int cmosTopPix = config["cmosTopPix"].get<int>();
	int cmosRightPix = config["cmosRightPix"].get<int>();
	int cmosBottomPix = config["cmosBottomPix"].get<int>();
	cmosRect = cv::Rect(cmosLeftPix, cmosTopPix, cmosRightPix - cmosLeftPix, cmosBottomPix - cmosTopPix);

	int cornerLeftPix = config.value("cornerLeftPix", 0);
	int cornerTopPix = config.value("cornerTopPix", 0);
	int cornerRightPix = config.value("cornerRightPix", 0);
	int cornerBottomPix = config.value("cornerBottomPix", 0);
	cornerRect = cv::Rect(cornerLeftPix, cornerTopPix,
	                      cornerRightPix - cornerLeftPix,
	                      cornerBottomPix - cornerTopPix);

	warpSearchRatio = config["warpSearchRatio"].get<double>();
	warpGrayThresh = config["warpGrayThresh"].get<int>();

	padOffsetPix = config["padOffsetPix"].get<int>();
	offsetThreshUm = config["offsetThreshUm"].get<double>();
	offsetY0 = config.value("offsetY0", 0);

	blurErode = config["blurErode"].get<int>();
	capEdgeEps = config["capEdgeEps"].get<int>();
	capChipLength = config["capChipLength"].get<float>();
	capChipWidth = config["capChipWidth"].get<float>();

	capGrayThresh = config["capGrayThresh"].get<int>();
	capBinaryThresh = config.value("capBinaryThresh", 96);
	capResizeR = config["capResizeR"].get<double>();
	capDefectMinArea = config["capDefectMinArea"].get<int>();
	capPointMinArea = config.value("capPointMinArea", 0.02);
	capDirtyMinArea = config.value("capDirtyMinArea", 0.05);
	capPointMinArea2 = config.value("capPointMinArea2", capPointMinArea);
	capDirtyMinArea2 = config.value("capDirtyMinArea2", capDirtyMinArea);
	//capScratchMinArea = config["capScratchMinArea"].get<double>();

	// 划伤最小粗细阈值，单位：mm
	capScratchMinWidth = config.value("capScratchMinWidth", 0.02);
	//capBinaryWhiteRatio = config.value("capBinaryWhiteRatio", 0.30);
        capScratchMinLength = config.value("capScratchMinLength", 1);
        capScratchCount = config.value("capScratchCount", 3);

	capPointGrayThresh = config.value("capPointGrayThresh", 0.4);
	capDirtyGrayThresh = config.value("capDirtyGrayThresh", 0.4);
	capPointGrayThresh2 = config.value("capPointGrayThresh2", capPointGrayThresh);
	capDirtyGrayThresh2 = config.value("capDirtyGrayThresh2", capDirtyGrayThresh);
	capScratchGrayThresh = config.value("capScratchGrayThresh", 0.4);


	blurExpand = config["blurExpand"].get<int>();
	cmosGrayThresh = config["cmosGrayThresh"].get<int>();
	cmosEdgeX0 = config["cmosEdgeX0"].get<int>();
	cmosEdgeY0 = config["cmosEdgeY0"].get<int>();
	cmosChipMinWidth = config["cmosChipMinWidth"].get<int>();
	cmosDirtyMinArea = config["cmosDirtyMinArea"].get<int>();

	cornerW = config.value("cornerW", 0);
		cornerH = config.value("cornerH", 0);

	// 派生参数
	cv::Mat temp = cv::imread(tempPath, cv::IMREAD_GRAYSCALE);
	if (temp.empty()) {
		std::cerr << "[CommonParmas] Failed to read temp image: " << tempPath << std::endl;
		return;
	}
	tempW0 = temp.cols;
	tempH0 = temp.rows;
}
