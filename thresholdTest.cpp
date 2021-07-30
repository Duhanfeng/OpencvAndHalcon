
#define NOMINMAX

#include "thresholdTest.h"
#include <opencv2/opencv.hpp>
#include <HalconCpp.h>
#include "timer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>

using namespace std;
using namespace cv;
using namespace HalconCpp;

void _threshold(unsigned char* src, unsigned char* dst, const int32_t start_row, const int32_t thread_stride, const int32_t stride, const int32_t width,
    const uint8_t min, const uint8_t max)
{
    for (size_t r = start_row; r < start_row + thread_stride; r++)
    {
        uchar* srcData = src + r * stride;
        uchar* dstData = dst + r * stride;

        for (int c = 0; c < width; c++)
        {
            if ((*srcData >= min) && (*srcData <= max))
            {
                *dstData = 255;
            }

            srcData++;
            dstData++;
        }
    }
}

/// <summary>
/// 二值化
/// </summary>
/// <param name="src">输入图像,要求单通道图像</param>
/// <param name="dst">输出图像</param>
/// <param name="threshold1">低阈值</param>
/// <param name="threshold2">高阈值</param>
void threshold(cv::InputArray src, cv::OutputArray dst, double threshold1, double threshold2)
{
    if (src.channels() != 1)
    {
        throw invalid_argument("输入图像必须为单通道");
    }

    uchar minT = (uchar)std::min(threshold1, threshold2);
    uchar maxT = (uchar)std::max(threshold1, threshold2);

    cv::Mat srcMat = src.getMat();
    int width = srcMat.cols;
    int height = srcMat.rows;
    int stride = srcMat.step;
    cv::Mat dstMat = cv::Mat::zeros(height, width, CV_8U);

    const int32_t hw_concur = (std::min)(height >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));
    std::vector<std::future<void>> fut(hw_concur);
    const int thread_stride = (height - 1) / hw_concur + 1;

    //创建异步线程租
    int i = 0, start = 0;
    for (; i < (std::min)(height, hw_concur); i++, start += thread_stride)
    {
        fut[i] = std::async(std::launch::async, _threshold, srcMat.data, dstMat.data, start, thread_stride, stride, width, minT, maxT);
    }
    //等待所有的数据执行完成
    for (int j = 0; j < i; ++j)
    {
        fut[j].wait();
    }

    if (dst.kind() == cv::_InputArray::MAT)
    {
        dst.assign(dstMat);
    }
}

/// <summary>
/// 二值化
/// </summary>
/// <param name="src">输入图像,要求单通道图像</param>
/// <param name="dst">输出图像</param>
/// <param name="threshold1">低阈值</param>
/// <param name="threshold2">高阈值</param>
void threshold2(cv::InputArray src, cv::OutputArray dst, double threshold1, double threshold2)
{
    if (src.channels() != 1)
    {
        throw invalid_argument("输入图像必须为单通道");
    }

    double minThreshold = std::min(threshold1, threshold2);
    double maxThreshold = std::max(threshold1, threshold2);

    cv::Mat srcMat = src.getMat();
    cv::Mat dstMat = cv::Mat::zeros(srcMat.rows, srcMat.cols, CV_8U);
    int rows = srcMat.rows;
    int cols = srcMat.cols;

    uchar minT = (uchar)minThreshold;
    uchar maxT = (uchar)maxThreshold;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
        {
            for (int index = range.start; index < range.end; index++) //这是需要并行计算的for循环
            {
                uchar* srcData = srcMat.ptr<uchar>(index);
                uchar* dstData = dstMat.ptr<uchar>(index);
                for (int c = 0; c < cols; c++)
                {
                    if ((*srcData >= minT) && (*srcData <= maxT))
                    {
                        *dstData = 255;
                    }

                    srcData++;
                    dstData++;
                }
            }
        });

    if (dst.kind() == cv::_InputArray::MAT)
    {
        dst.assign(dstMat);
    }
}

void thresholdTest()
{
    std::string file = "E:/临时文件/匹配/Pic_2021_07_27_103526_1.bmp";

    //读取opencv图像
    cv::Mat cImage = imread(file, IMREAD_GRAYSCALE);
    int step = cImage.step;
    int width = cImage.cols;
    int height = cImage.rows;
    cout << step << "," << width << endl;

    //读取halcon图像
    HTuple hfile = HTuple(file.c_str());
    HObject hImage;
    HalconCpp::ReadImage(&hImage, hfile);
    HalconCpp::Rgb1ToGray(hImage, &hImage);

    //延时让系统稳定(刚进入debug模式下,VS需要加载各种窗口及资源,此时无法跑满系统性能)
    _sleep(1000);

    int circleTime = 500;

    Timer timer1;
    for (size_t i = 0; i < circleTime; i++)
    {
        HObject region;
        HalconCpp::Threshold(hImage, &region, 0, 100);
        HObject thrImage;
        RegionToBin(region, &thrImage, 255, 0, width, height);
    }
    timer1.out("halcon");

    Timer timer2;
    for (size_t i = 0; i < circleTime; i++)
    {
        cv::Mat thrImage;
        threshold(cImage, thrImage, 0, 100);
    }
    timer2.out("opencv-custom");

    Timer timer4;
    for (size_t i = 0; i < circleTime; i++)
    {
        cv::Mat thrImage;
        threshold2(cImage, thrImage, 0, 100);
    }
    timer4.out("opencv-custom2");

    Timer timer3;
    for (size_t i = 0; i < circleTime; i++)
    {
        cv::Mat thrImage;
        cv::Mat thrImage2;
        cv::threshold(cImage, thrImage, 100, 255, THRESH_BINARY_INV);
        cv::threshold(cImage, thrImage2, 0, 255, THRESH_BINARY);
        thrImage = thrImage & thrImage2;
    }
    timer3.out("opencv");

}
