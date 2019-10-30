#include  <opencv2/opencv.hpp>  
#include "vibe.h"    
#include "EulerianMotionMag.h"
#include <fstream>
#include <iostream>    
#include <cstdio>    
#include <stdlib.h>  
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	string input_filename = argv[1];
	string output_filename = "./baby_out.avi";
	int input_width = 0;
	int input_height = 0;
	int output_width = 0;
       	int output_height = 0;
	double alpha = 20;
	double lambda_c = 16;
	double cutoff_freq_low = 0.05;
	double cutoff_freq_high = 0.4;
	double chrom_attenuation = 0.1;
	double exaggeration_factor = 2.0;
	double delta = 0;
	double lambda = 0;
	int levels = 5;

	if (argc <= 1)
	{
        	cerr << "Error: Number of arguments is not enough!" << endl;
		return -1;
	}

	// EulerianMotionMag
	EulerianMotionMag* motion_mag = new EulerianMotionMag();

	// Set params
	motion_mag->setInputFileName(input_filename);
	motion_mag->setOutputFileName(output_filename);
	motion_mag->setInputImgWidth(input_width);
	motion_mag->setInputImgHeight(input_height);
	motion_mag->setOutputImgWidth(output_width);
	motion_mag->setOutputImgHeight(output_height);
	motion_mag->setAlpha(alpha);
	motion_mag->setLambdaC(lambda_c);
	motion_mag->setCutoffFreqLow(cutoff_freq_low);
	motion_mag->setCutoffFreqHigh(cutoff_freq_high);
	motion_mag->setChromAttenuation(chrom_attenuation);
	motion_mag->setExaggerationFactor(exaggeration_factor);
	motion_mag->setDelta(delta);
	motion_mag->setLambda(lambda);
	motion_mag->setLapPyramidLevels(levels);

	// Init Motion Magnification object
	bool init_status = motion_mag->init();
	if (!init_status)
		return -1;

	//Run Motion Magnification
	motion_mag->run();

	delete motion_mag;

	Mat frame, gray, mask;
	VideoCapture capture;
	capture.open("./baby_out.avi");    //输入视频名称

	if (!capture.isOpened())
	{
	    cout << "No camera or video input!\n" << endl;
	    return -1;
	}

	ViBe_BGS Vibe_Bgs; //定义一个背景差分对象  
	int count = 0; //帧计数器，统计为第几帧   
	
	while (1)
	{
	    count++;
	    capture >> frame;
	    if (frame.empty())
	        break;
	    cvtColor(frame, gray, CV_RGB2GRAY); //转化为灰度图像   
	
	    if (count == 1)  //若为第一帧  
	    {
	        Vibe_Bgs.init(gray);
	        Vibe_Bgs.processFirstFrame(gray); //背景模型初始化   
	       // cout << " Training GMM complete!" << endl;
	    }
	    else
	    {
	        double t = getTickCount();
	        Vibe_Bgs.testAndUpdate(gray);
	        mask = Vibe_Bgs.getMask();    //计算前景
	        morphologyEx(mask, mask, MORPH_OPEN, Mat());   //形态学处理消除前景图像中的小噪声，这里用的开运算 
	        imshow("mask", mask);
	        t = (getTickCount() - t) / getTickFrequency();
	        cout << "imgpro time: " << t << endl;
	    }
	
	    imshow("input", frame);
	
	    if (cvWaitKey(10) == 'q')    //键盘键入q,则停止运行，退出程序
	        break;
	}
	system("pause");
	return 0;
}
