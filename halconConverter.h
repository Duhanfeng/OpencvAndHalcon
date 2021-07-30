#pragma once

#include <HalconCpp.h>
#include <opencv2/opencv.hpp>

//Mat与halcon类型的转换
inline cv::Mat hobject2Mat(HalconCpp::HObject Hobj)
{
	HalconCpp::HTuple htCh;
	HalconCpp::HString cType;
	cv::Mat Image;
	HalconCpp::ConvertImageType(Hobj, &Hobj, "byte");
	HalconCpp::CountChannels(Hobj, &htCh);
	Hlong wid = 0;
	Hlong hgt = 0;
	if (htCh[0].I() == 1)
	{
		HalconCpp::HImage hImg(Hobj);
		void* ptr = hImg.GetImagePointer1(&cType, &wid, &hgt);//GetImagePointer1(Hobj, &ptr, &cType, &wid, &hgt);
		int W = wid;
		int H = hgt;
		Image.create(H, W, CV_8UC1);
		unsigned char* pdata = static_cast<unsigned char*>(ptr);
		memcpy(Image.data, pdata, W * H);
	}
	else if (htCh[0].I() == 3)
	{
		void* Rptr;
		void* Gptr;
		void* Bptr;
		HalconCpp::HImage hImg(Hobj);
		hImg.GetImagePointer3(&Rptr, &Gptr, &Bptr, &cType, &wid, &hgt);
		int W = wid;
		int H = hgt;
		Image.create(H, W, CV_8UC3);
		std::vector<cv::Mat> VecM(3);
		VecM[0].create(H, W, CV_8UC1);
		VecM[1].create(H, W, CV_8UC1);
		VecM[2].create(H, W, CV_8UC1);
		unsigned char* R = (unsigned char*)Rptr;
		unsigned char* G = (unsigned char*)Gptr;
		unsigned char* B = (unsigned char*)Bptr;
		memcpy(VecM[2].data, R, W * H);
		memcpy(VecM[1].data, G, W * H);
		memcpy(VecM[0].data, B, W * H);
		cv::merge(VecM, Image);
	}
	return Image;
}

inline HalconCpp::HObject mat2HObject(const cv::Mat& image)
{
	HalconCpp::HObject Hobj = HalconCpp::HObject();

	if (image.type() == CV_8UC1)
	{
		HalconCpp::GenImage1(&Hobj, "byte", image.cols, image.rows, (Hlong)image.data);
	}
	else if (image.type() == CV_8UC3)
	{
		std::vector<cv::Mat> imgchannel;
		split(image, imgchannel);
		HalconCpp::GenImage3(&Hobj, "byte", image.cols, image.rows, (Hlong)imgchannel[2].ptr(), (Hlong)imgchannel[1].ptr(), (Hlong)imgchannel[0].ptr());
	}

	//int hgt = image.rows;
	//int wid = image.cols;
	//int i;
	//// CV_8UC3
	//if (image.type() == CV_8UC3)
	//{
	//	std::vector<cv::Mat> imgchannel;
	//	split(image, imgchannel);
	//	cv::Mat imgB = imgchannel[0];
	//	cv::Mat imgG = imgchannel[1];
	//	cv::Mat imgR = imgchannel[2];
	//	uchar* dataR = new uchar[hgt * wid];
	//	uchar* dataG = new uchar[hgt * wid];
	//	uchar* dataB = new uchar[hgt * wid];
	//	for (i = 0; i < hgt; i++)
	//	{
	//		memcpy(dataR + wid * i, imgR.data + imgR.step * i, wid);
	//		memcpy(dataG + wid * i, imgG.data + imgG.step * i, wid);
	//		memcpy(dataB + wid * i, imgB.data + imgB.step * i, wid);
	//	}
	//	HalconCpp::GenImage3(&Hobj, "byte", wid, hgt, (Hlong)dataR, (Hlong)dataG, (Hlong)dataB);
	//	delete[]dataR;
	//	delete[]dataG;
	//	delete[]dataB;
	//	dataR = nullptr;
	//	dataG = nullptr;
	//	dataB = nullptr;
	//}
	//// CV_8UCU1
	//else if (image.type() == CV_8UC1)
	//{
	//	uchar* data = new uchar[hgt * wid];
	//	for (i = 0; i < hgt; i++)
	//		memcpy(data + wid * i, image.data + image.step * i, wid);
	//	HalconCpp::GenImage1(&Hobj, "byte", wid, hgt, (Hlong)data);
	//	delete[] data;
	//	data = nullptr;
	//}
	return Hobj;
}