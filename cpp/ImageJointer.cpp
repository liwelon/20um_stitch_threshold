#include "ImageJointer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

ImageJointer::ImageJointer(CommonParmas& params)
{
    // 更新参数
	hDx = params.meachineParams.hDx;
    hDy = params.meachineParams.hDy;
    vDy = params.meachineParams.vDy;
    vDx = params.meachineParams.vDx;

    rows = params.jointParams.jointRows;
    cols = params.jointParams.jointCols;
    layers = params.jointParams.layers;

    singleW = params.meachineParams.singleW;
    singleH = params.meachineParams.singleH;
    channel = params.meachineParams.channel;

	// 根据参数计算相关的派生参数
	jointCount = rows * cols * layers;       // 上下两层，每层rows*cols张图
    calcIdArray();
    calcJointedSize();
}

bool ImageJointer::genHorizontalMask(int width, int height, cv::Mat& mask1, cv::Mat& mask2)
{
    if (width <= 0 || height <= 0) {
        return false;
    }

    mask1 = cv::Mat(height, width, CV_64FC3);
    mask2 = cv::Mat(height, width, CV_64FC3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double v1 = (width == 1) ? 1.0 : 1.0 - static_cast<double>(x) / static_cast<double>(width - 1);
            double v2 = (width == 1) ? 0.0 : static_cast<double>(x) / static_cast<double>(width - 1);
            mask1.at<cv::Vec3d>(y, x) = cv::Vec3d(v1, v1, v1);
            mask2.at<cv::Vec3d>(y, x) = cv::Vec3d(v2, v2, v2);
        }
    }
    return true;
}

bool ImageJointer::genVerticalMask(int width, int height, cv::Mat& mask1, cv::Mat& mask2)
{
    if (width <= 0 || height <= 0) {
        return false;
    }

    mask1 = cv::Mat(height, width, CV_64FC3);
    mask2 = cv::Mat(height, width, CV_64FC3);

    for (int y = 0; y < height; ++y) {
        double v1 = (height == 1) ? 1.0 : 1.0 - static_cast<double>(y) / static_cast<double>(height - 1);
        double v2 = (height == 1) ? 0.0 : static_cast<double>(y) / static_cast<double>(height - 1);
        for (int x = 0; x < width; ++x) {
            mask1.at<cv::Vec3d>(y, x) = cv::Vec3d(v1, v1, v1);
            mask2.at<cv::Vec3d>(y, x) = cv::Vec3d(v2, v2, v2);
        }
    }
    return true;
}

bool ImageJointer::calcJointedSize()
{
    hOverlap = singleW - std::abs(hDx);
    hJointedW = singleW * cols - (cols - 1) * hOverlap;
    hJointedH = singleH - (cols - 1) * std::abs(hDy);
    if (!genHorizontalMask(hOverlap, hJointedH, hOverlapMask1, hOverlapMask2)) {
        return false;
    }

    vOverlap = singleH - vDy - (cols - 1) * std::abs(hDy);
    vJointedH = hJointedH * rows - (rows - 1) * vOverlap;
    vJointedW = hJointedW - (rows - 1) * std::abs(vDx);
    if (!genVerticalMask(vJointedW, vOverlap, vOverlapMask1, vOverlapMask2)) {
        return false;
    }

    return true;
}

void ImageJointer::calcIdArray()
{
    // 固定两层，按照先上的顺序开始，上下下上上下...，获得上下两层的id
    std::vector<int> upIds;
    std::vector<int> downIds;
    int id = 0;
    for (int i = 0; i < rows * cols; ++i) {
        if (i % 2 == 0) {
            upIds.push_back(id++);
            downIds.push_back(id++);
        }
        else {
            downIds.push_back(id++);
            upIds.push_back(id++);
        }
    }

    idArray.clear();
    idArray.resize(2, std::vector<std::vector<int>>(rows, std::vector<int>(cols, 0)));

    // 单看一层，拼图顺序应该是偶数行从左到右，奇数行从右到左（索引）
    for (int i = 0; i < rows; i++) {
        if (i % 2 == 0)				// 偶数行
        {
            int startJ = i * cols;
            for (int j = 0; j < cols; j++) {
                idArray[0][i][j] = upIds[startJ];
                idArray[1][i][j] = downIds[startJ];
                ++startJ;
            }
        }
        else						// 奇数行	
        {
            int startJ = (i + 1) * cols - 1;
            for (int j = 0; j < cols; j++) {
                idArray[0][i][j] = upIds[startJ];
                idArray[1][i][j] = downIds[startJ];
                --startJ;
            }
        }
    }
}

bool ImageJointer::horizontalJoint(const std::vector<cv::Mat>& imgs, cv::Mat& jointed) const
{
    if (static_cast<int>(imgs.size()) != cols) {
        return false;
    }

    std::vector<cv::Mat> clippedImgs;
    int endId = cols - 1;

    if (hDy > 0) {
        for (int i = 0; i < cols; ++i) {
            int y0 = (endId - i) * hDy;
            int y1 = singleH - i * hDy;
            clippedImgs.push_back(imgs[i](cv::Range(y0, y1), cv::Range::all()).clone());
        }
    }
    else if (hDy < 0) {
        for (int i = 0; i < cols; ++i) {
            int y0 = -i * hDy;
            int y1 = singleH + (endId - i) * hDy;
            clippedImgs.push_back(imgs[i](cv::Range(y0, y1), cv::Range::all()).clone());
        }
    }
    else {
        clippedImgs = imgs;
    }

    jointed = cv::Mat::zeros(hJointedH, hJointedW, CV_8UC3);

    for (int i = 0; i < cols - 1; ++i) {
        const cv::Mat& img1 = clippedImgs[i];
        const cv::Mat& img2 = clippedImgs[i + 1];

        cv::Mat overlap1 = img1(cv::Range::all(), cv::Range(img1.cols - hOverlap, img1.cols));
        cv::Mat overlap2 = img2(cv::Range::all(), cv::Range(0, hOverlap));

        cv::Mat overlap1F;
        cv::Mat overlap2F;
        overlap1.convertTo(overlap1F, CV_64FC3);
        overlap2.convertTo(overlap2F, CV_64FC3);

        cv::Mat overlapF = hOverlapMask1.mul(overlap1F) + hOverlapMask2.mul(overlap2F);
        cv::Mat overlap;
        overlapF.convertTo(overlap, CV_8UC3);

        if (i == 0) {
            img1(cv::Range::all(), cv::Range(0, hDx)).copyTo(jointed(cv::Range::all(), cv::Range(0, hDx)));
            overlap.copyTo(jointed(cv::Range::all(), cv::Range(hDx, singleW)));
        }
        else {
            int x0 = singleW + (i - 1) * hDx;
            int x1 = x0 + hDx - hOverlap;
            int x2 = x0 + hDx;
            img1(cv::Range::all(), cv::Range(hOverlap, img1.cols - hOverlap))
                .copyTo(jointed(cv::Range::all(), cv::Range(x0, x1)));
            overlap.copyTo(jointed(cv::Range::all(), cv::Range(x1, x2)));
        }

        if (i == cols - 2) {
            int x0 = (i + 1) * hDx + hOverlap;
            int x1 = hJointedW;
            img2(cv::Range::all(), cv::Range(hOverlap, img2.cols))
                .copyTo(jointed(cv::Range::all(), cv::Range(x0, x1)));
        }
    }

    return true;
}

bool ImageJointer::verticalJoint(const std::vector<cv::Mat>& imgs, cv::Mat& jointed) const
{
    if (static_cast<int>(imgs.size()) != rows) {
        return false;
    }

    std::vector<cv::Mat> clippedImgs;
    int endId = rows - 1;

    if (vDx > 0) {
        for (int i = 0; i < rows; ++i) {
            int x0 = (endId - i) * vDx;
            int x1 = hJointedW - i * vDx;
            clippedImgs.push_back(imgs[i](cv::Range::all(), cv::Range(x0, x1)).clone());
        }
    }
    else if (vDx < 0) {
        for (int i = 0; i < rows; ++i) {
            int x0 = -i * vDx;
            int x1 = hJointedW + (endId - i) * vDx;
            clippedImgs.push_back(imgs[i](cv::Range::all(), cv::Range(x0, x1)).clone());
        }
    }
    else {
        clippedImgs = imgs;
    }

    jointed = cv::Mat::zeros(vJointedH, vJointedW, CV_8UC3);

    for (int i = 0; i < rows - 1; ++i) {
        const cv::Mat& img1 = clippedImgs[i];
        const cv::Mat& img2 = clippedImgs[i + 1];

        cv::Mat overlap1 = img1(cv::Range(img1.rows - vOverlap, img1.rows), cv::Range::all());
        cv::Mat overlap2 = img2(cv::Range(0, vOverlap), cv::Range::all());

        cv::Mat overlap1F;
        cv::Mat overlap2F;
        overlap1.convertTo(overlap1F, CV_64FC3);
        overlap2.convertTo(overlap2F, CV_64FC3);

        cv::Mat overlapF = vOverlapMask1.mul(overlap1F) + vOverlapMask2.mul(overlap2F);
        cv::Mat overlap;
        overlapF.convertTo(overlap, CV_8UC3);

        if (i == 0) {
            img1(cv::Range(0, vDy), cv::Range::all()).copyTo(jointed(cv::Range(0, vDy), cv::Range::all()));
            overlap.copyTo(jointed(cv::Range(vDy, hJointedH), cv::Range::all()));
        }
        else {
            int y0 = hJointedH + (i - 1) * vDy;
            int y1 = y0 + vDy - vOverlap;
            int y2 = y0 + vDy;
            img1(cv::Range(vOverlap, img1.rows - vOverlap), cv::Range::all())
                .copyTo(jointed(cv::Range(y0, y1), cv::Range::all()));
            overlap.copyTo(jointed(cv::Range(y1, y2), cv::Range::all()));
        }

        if (i == rows - 2) {
            int y0 = (i + 1) * vDy + vOverlap;
            int y1 = vJointedH;
            img2(cv::Range(vOverlap, img2.rows), cv::Range::all())
                .copyTo(jointed(cv::Range(y0, y1), cv::Range::all()));
        }
    }

    return true;
}

bool ImageJointer::jointUpDown(const std::vector<cv::Mat>& imgs, cv::Mat& upImg, cv::Mat& downImg) const
{
    if (imgs.size() != jointCount) {
        return false;
    }

    std::vector<cv::Mat> lineImgs;          // 一行的数据
	std::vector<cv::Mat> lineJointedImgs;	// 一行拼接后的数据

    // 上层图片
    for (int i = 0; i < rows; ++i) {
		// 取一行的图片
        lineImgs.clear();
        for (int j = 0; j < cols; ++j) {
			lineImgs.push_back(imgs[idArray[0][i][j]]);
        }
		// 水平拼接
		cv::Mat lineJointed;
        if (!horizontalJoint(lineImgs, lineJointed)) {
            return false;
		}
        // 所有行拼接的图
		lineJointedImgs.push_back(lineJointed);
	}
	// 垂直拼接图片
    if (!verticalJoint(lineJointedImgs, upImg)) {
        return false;
    }

	// 下层图片
    lineJointedImgs.clear();
    for (int i = 0; i < rows; ++i) {
        lineImgs.clear();
        for (int j = 0; j < cols; ++j) {
            lineImgs.push_back(imgs[idArray[1][i][j]]);
        }
        cv::Mat lineJointed;
        if (!horizontalJoint(lineImgs, lineJointed)) {
            return false;
        }
        lineJointedImgs.push_back(lineJointed);
    }
    if (!verticalJoint(lineJointedImgs, downImg)) {
        return false;
    }
    return true;
}

//bool ImageJointer::focusStack(const cv::Mat& imgHigh, const cv::Mat& imgLow, cv::Mat& result) const
//{
//    if (imgHigh.empty() || imgLow.empty()) {
//        return false;
//    }
//
//    cv::Mat grayHigh;
//    cv::Mat grayLow;
//    cv::cvtColor(imgHigh, grayHigh, cv::COLOR_BGR2GRAY);
//    cv::cvtColor(imgLow, grayLow, cv::COLOR_BGR2GRAY);
//
//    cv::Mat laplacianHigh;
//    cv::Mat laplacianLow;
//    cv::Laplacian(grayHigh, laplacianHigh, CV_64F);
//    cv::Laplacian(grayLow, laplacianLow, CV_64F);
//
//    cv::Mat clarityHigh = cv::abs(laplacianHigh);
//    cv::Mat clarityLow = cv::abs(laplacianLow);
//
//    cv::GaussianBlur(clarityHigh, clarityHigh, stackKSize, stackSigma);
//    cv::GaussianBlur(clarityLow, clarityLow, stackKSize, stackSigma);
//
//    cv::Mat maskU8 = (clarityHigh > clarityLow);
//    cv::Mat mask;
//    maskU8.convertTo(mask, CV_32F, 1.0 / 255.0);
//
//    cv::Mat mask3Ch;
//    std::vector<cv::Mat> channels = { mask, mask, mask };
//    cv::merge(channels, mask3Ch);
//
//    cv::Mat imgHighF;
//    cv::Mat imgLowF;
//    imgHigh.convertTo(imgHighF, CV_32FC3);
//    imgLow.convertTo(imgLowF, CV_32FC3);
//
//    cv::Mat imgResult = imgHighF.mul(mask3Ch) + imgLowF.mul(cv::Scalar(1.0, 1.0, 1.0) - mask3Ch);
//    imgResult.convertTo(result, CV_8UC3);
//
//    return true;
//}
//
//bool ImageJointer::jointToSingle(const std::vector<cv::Mat>& imgs, cv::Mat& result) const
//{
//    if (imgs.size() != 8) {
//        return false;
//    }
//
//    cv::Mat upImg;
//    cv::Mat downImg;
//    if (!jointUpDown(imgs, upImg, downImg)) {
//        return false;
//    }
//
//    return focusStack(upImg, downImg, result);
//}