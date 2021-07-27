// OpencvAndHalcon.cpp: 定义应用程序的入口点。
//

#include "OpencvAndHalcon.h"

#define NDEBUG
#include <opencv2/opencv.hpp>
#include <HalconCpp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include "timer.h"

#include <opencv2/core/hal/intrin.hpp>

using namespace std;
using namespace cv;
using namespace HalconCpp;


void _threshold(unsigned char* src, unsigned char* dst, const int32_t start_row, const int32_t thread_stride, const int32_t stride, const int32_t width,
    const uint8_t min, const uint8_t max)
{
    v_uint8 maxval16 = vx_setall_u8(255);
    v_uint8 thresh_min = vx_setall_u8(min);

    for (size_t r = start_row; r < start_row + thread_stride; r++)
    {
        uchar* srcData = src + r * stride;
        uchar* dstData = dst + r * stride;

        //for (int j = 0; j <= width - v_uint8::nlanes; j += v_uint8::nlanes)
        //{
        //    v_uint8 v0;
        //    v0 = vx_load(srcData + j);
        //    v0 = thresh_min < v0;
        //    v0 = v0 & maxval16;
        //    v_store(dstData + j, v0);
        //}

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

    uchar minT = (uchar)(std::min)(threshold1, threshold2);
    uchar maxT = (uchar)(std::max)(threshold1, threshold2);

    cv::Mat srcMat = src.getMat();
    int width = srcMat.cols;
    int height = srcMat.rows;
    int stride = srcMat.step;
    cv::Mat dstMat = cv::Mat::zeros(height, width, CV_8U);

    const int32_t hw_concur = (std::min)(height >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));
    std::vector<std::future<void>> fut(hw_concur);
    const int thread_stride = (height - 1) / hw_concur + 1;

    int i = 0, start = 0;
    for (; i < (std::min)(height, hw_concur); i++, start += thread_stride)
    {
        fut[i] = std::async(std::launch::async, _threshold, srcMat.data, dstMat.data, start, thread_stride, stride, width, minT, maxT);
    }
    for (int j = 0; j < i; ++j)
    {
        fut[j].wait();
    }

    if (dst.kind() == cv::_InputArray::MAT)
    {
        dst.assign(dstMat);
    }
}

int main()
{
    std::string file = "G:/临时文件/匹配/1.1.bmp";

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

    //延时让系统稳定
    _sleep(1000);

    Timer timer1;
    for (size_t i = 0; i < 1000; i++)
    {
        HObject region;
        HalconCpp::Threshold(hImage, &region, 0, 100);
        HObject thrImage;
        RegionToBin(region, &thrImage, 255, 0, width, height);
    }
    timer1.out("halcon");

    Timer timer2;
    for (size_t i = 0; i < 1000; i++)
    {
        cv::Mat thrImage;
        threshold(cImage, thrImage, 0, 100);

        cv::Mat resizeImage;
        resize(thrImage, resizeImage, {}, 0.2, 0.2);
    }
    timer2.out("opencv");

	return 0;
}
