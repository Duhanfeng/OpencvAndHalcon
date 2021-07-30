
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
#include "halconConverter.h"

using namespace std;
using namespace cv;
using namespace HalconCpp;

/// <summary>
/// 弧度转角度
/// </summary>
/// <param name="rad">弧度</param>
/// <returns>角度</returns>
inline double rad2deg(double rad)
{
    return rad * 180.0 / CV_PI;
}

/// <summary>
/// 角度转弧度
/// </summary>
/// <param name="deg">角度</param>
/// <returns>弧度</returns>
inline double deg2rad(double deg)
{
    return deg * CV_PI / 180.0;
}

/// <summary>
/// 向量转角度
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns></returns>
inline double vector2angle(double x, double y)
{

    return std::atan2(y, x);
}

/// <summary>
/// BGR颜色图像转HSV图像
/// </summary>
/// <param name="bgr">BGR图像</param>
/// <param name="hsv">HSV图像</param>
void bgr2hsv(cv::InputArray bgr, cv::OutputArray hsv)
{
    cv::Mat bgrMat = bgr.getMat();

    if (bgrMat.empty())
    {
        throw std::invalid_argument("输入图像无效");
    }

    if (bgrMat.channels() == 1)
    {
        cv::cvtColor(bgrMat, bgrMat, cv::ColorConversionCodes::COLOR_GRAY2BGR);
    }

    //分割BGR图像
    std::vector<cv::Mat> bgrChannels;
    cv::split(bgrMat, bgrChannels);
    cv::Mat b = bgrChannels[0];
    cv::Mat g = bgrChannels[1];
    cv::Mat r = bgrChannels[2];

    //创建浮点类型的HSV图像
    cv::Mat fh = cv::Mat::zeros(b.size(), CV_32F);
    cv::Mat fs = cv::Mat::zeros(b.size(), CV_32F);
    cv::Mat fv = cv::Mat::zeros(b.size(), CV_32F);

    //进行类型转换
    int rows = bgrMat.rows;
    int cols = bgrMat.cols;
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
        {
            for (int row = range.start; row < range.end; row++) //这是需要并行计算的for循环
            {
                uchar* bc = b.ptr<uchar>(row);
                uchar* gc = g.ptr<uchar>(row);
                uchar* rc = r.ptr<uchar>(row);

                float* hc = fh.ptr<float>(row);
                float* sc = fs.ptr<float>(row);
                float* vc = fv.ptr<float>(row);

                for (int col = 0; col < cols; col++)
                {
                    uchar minValue = std::min(std::min(*bc, *gc), *rc);
                    uchar maxValue = std::max(std::max(*bc, *gc), *rc);
                    *vc = maxValue;

                    if (minValue == maxValue)
                    {
                        *hc = 0;
                        *sc = 0;
                    }
                    else
                    {
                        *sc = (float)(maxValue - minValue) / maxValue;

                        if (*rc == maxValue)
                        {
                            *hc = ((float)(*gc - *bc) / (maxValue - minValue)) * deg2rad(60);
                        }
                        else if (*gc == maxValue)
                        {
                            *hc = (2.0f + (float)(*bc - *rc) / (maxValue - minValue)) * deg2rad(60);
                        }
                        else
                        {
                            *hc = (4.0f + (float)(*rc - *gc) / (maxValue - minValue)) * deg2rad(60);
                        }

                        if (*hc < 0)
                        {
                            *hc += 2 * CV_PI;
                        }
                    }

                    bc++;
                    gc++;
                    rc++;
                    hc++;
                    sc++;
                    vc++;
                }
            }

        });

    //H的范围为[0,2PI],S的范围为[0,1],V的范围为[0,255],将其统一归一化到[0,255]
    fh = fh * 255.0 / (2 * CV_PI);
    fs = fs * 255.0;
    cv::Mat h, s, v;
    fh.convertTo(h, CV_8UC1);
    fs.convertTo(s, CV_8UC1);
    fv.convertTo(v, CV_8UC1);

    //合并图像
    std::vector<cv::Mat> hsvChannels;
    hsvChannels.push_back(h);
    hsvChannels.push_back(s);
    hsvChannels.push_back(v);
    cv::merge(hsvChannels, hsv);

}

/// <summary>
/// BGR颜色图像转HSV图像
/// </summary>
/// <param name="bgr">BGR图像</param>
/// <param name="hsv">HSV图像</param>
void bgr2hsv2(cv::InputArray bgr, cv::OutputArray hsv)
{
    cv::Mat bgrMat = bgr.getMat();

    if (bgrMat.empty())
    {
        throw std::invalid_argument("输入图像无效");
    }

    if (bgrMat.channels() == 1)
    {
        cv::cvtColor(bgrMat, bgrMat, cv::ColorConversionCodes::COLOR_GRAY2BGR);
    }

    //进行类型转换
    int rows = bgrMat.rows;
    int cols = bgrMat.cols;

    cv::Mat hsvMat = cv::Mat::zeros(bgrMat.size(), CV_32FC3);

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
        {
            for (int row = range.start; row < range.end; row++) //这是需要并行计算的for循环
            {
                uchar* bgr = bgrMat.ptr<uchar>(row);
                uchar* hsv = hsvMat.ptr<uchar>(row);

                for (int col = 0; col < cols; col++)
                {
                    uchar minValue = std::min(std::min(bgr[0], bgr[1]), bgr[2]);
                    uchar maxValue = std::max(std::max(bgr[0], bgr[1]), bgr[2]);
                    hsv[2] = maxValue;

                    uchar valueDiff = maxValue - minValue;

                    if (minValue == maxValue)
                    {
                        hsv[0] = 0;
                        hsv[1] = 0;
                    }
                    else
                    {
                        hsv[1] = 255 * valueDiff / maxValue;

                        short h = 0;
                        if (bgr[2] == maxValue)
                        {
                            h = 30 * (bgr[1] - bgr[0]) / valueDiff;
                        }
                        else if (bgr[1] == maxValue)
                        {
                            h = 60 + 30 * (bgr[1] - bgr[0]) / valueDiff;
                        }
                        else
                        {
                            h = 120 + 30 * (bgr[1] - bgr[0]) / valueDiff;
                        }
                        hsv[0] = (h < 0) ? h + 180 : h;
                    }

                    bgr += 3;
                    hsv += 3;
                }
            }

        });

    if (hsv.kind() == _InputArray::KindFlag::MAT)
    {
        hsv.assign(hsvMat);
    }
}

void bgr2hsvTest()
{
    std::string file = "E:/临时文件/匹配/Pic_2021_07_27_103526_1.bmp";

    //读取opencv图像
    cv::Mat cImage = imread(file, IMREAD_COLOR);
    int step = cImage.step;
    int width = cImage.cols;
    int height = cImage.rows;
    cout << step << "," << width << endl;

    //读取halcon图像
    HTuple hfile = HTuple(file.c_str());
    HObject hImage;
    HalconCpp::ReadImage(&hImage, hfile);

    //延时让系统稳定(刚进入debug模式下,VS需要加载各种窗口及资源,此时无法跑满系统性能)
    _sleep(1000);

    int circleTime = 10;

    Timer timer1;
    for (size_t i = 0; i < circleTime; i++)
    {
        HObject red;
        HObject green;
        HObject blue;
        HalconCpp::Decompose3(hImage, &red, &green, &blue);
        HObject hue;
        HObject saturation;
        HObject value;
        HalconCpp::TransFromRgb(red, green, blue, &hue, &saturation, &value, "hsv");
    }
    timer1.out("halcon");

    Timer timer2;
    for (size_t i = 0; i < circleTime; i++)
    {
        cv::Mat hsv;
        bgr2hsv(cImage, hsv);
    }
    timer2.out("opencv-custom");

    Timer timer3;
    for (size_t i = 0; i < circleTime; i++)
    {
        cv::Mat hsv;
        cv::cvtColor(cImage, hsv, ColorConversionCodes::COLOR_BGR2HSV);
    }
    timer3.out("opencv");

    Timer timer4;
    for (size_t i = 0; i < circleTime; i++)
    {
        cv::Mat hsv;
        bgr2hsv2(cImage, hsv);
    }
    timer4.out("opencv-custom2");

    //数据比较
    cv::Mat hsv1;
    bgr2hsv(cImage, hsv1);

    cv::Mat hsv2;
    cv::cvtColor(cImage, hsv2, ColorConversionCodes::COLOR_BGR2HSV);
    
    cv::Mat hsv3;
    bgr2hsv2(cImage, hsv3);

    HObject red;
    HObject green;
    HObject blue;
    HalconCpp::Decompose3(hImage, &red, &green, &blue);
    HObject hue;
    HObject saturation;
    HObject value;
    HalconCpp::TransFromRgb(red, green, blue, &hue, &saturation, &value, "hsv");

    std::vector<cv::Mat> channel1;
    cv::split(hsv1, channel1);
    cv::Mat h1 = channel1[0];
    cv::Mat s1 = channel1[1];
    cv::Mat v1 = channel1[2];

    std::vector<cv::Mat> channel2;
    cv::split(hsv2, channel2);
    cv::Mat h2 = channel2[0] / 180.0 * 255;
    cv::Mat s2 = channel2[1];
    cv::Mat v2 = channel2[2];

    std::vector<cv::Mat> channel3;
    cv::split(hsv3, channel3);
    cv::Mat h3 = channel3[0];
    cv::Mat s3 = channel3[1];
    cv::Mat v3 = channel3[2];

    cv::Mat h4 = hobject2Mat(hue);
    cv::Mat s4 = hobject2Mat(saturation);
    cv::Mat v4 = hobject2Mat(value);

    //计算偏差
    cv::Mat h24 = h2 - h4;
    cv::Mat s24 = s2 - s4;
    cv::Mat v24 = v2 - v4;

    cv::Mat h42 = h4 - h2;
    cv::Mat s42 = s4 - s2;
    cv::Mat v42 = v4 - v2;

    waitKey();

}
