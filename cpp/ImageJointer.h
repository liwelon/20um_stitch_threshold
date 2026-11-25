#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "Params.h"

class ImageJointer
{
public:
    ImageJointer(CommonParmas& params);
    bool jointUpDown(const std::vector<cv::Mat>& imgs, cv::Mat& upImg, cv::Mat& downImg) const;  // 拼接图像，获得上下两层图片

    int getJointCount() const { return jointCount; }  // 获取拼接图片的总数

private:
    static bool genHorizontalMask(int width, int height, cv::Mat& mask1, cv::Mat& mask2);
    static bool genVerticalMask(int width, int height, cv::Mat& mask1, cv::Mat& mask2);
    bool calcJointedSize();
	void calcIdArray();
    bool horizontalJoint(const std::vector<cv::Mat>& imgs, cv::Mat& jointed) const;
    bool verticalJoint(const std::vector<cv::Mat>& imgs, cv::Mat& jointed) const;

private:
    int hDx;
    int hDy;
    int vDy;
    int vDx;

    int rows;
    int cols;
    int layers;

    int singleH;
    int singleW;
    int channel;

    int jointCount;
    std::vector<std::vector<std::vector<int>>> idArray;

    int hJointedW;
    int hJointedH;
    int vJointedH;
    int vJointedW;

    int hOverlap;
    int vOverlap;
    cv::Mat hOverlapMask1;
    cv::Mat hOverlapMask2;
    cv::Mat vOverlapMask1;
    cv::Mat vOverlapMask2;
};