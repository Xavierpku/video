#include  <opencv2/opencv.hpp>  
#include "vibe.h"    
#include "EulerianMotionMag.h"
#include <cstdio>    
#include <stdlib.h>  
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	//Set default param
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
        	printf("Error: Number of arguments is not enough!\n");
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

	//Run Motion Magnification and vibe, note that every operation are all in the motion_mag module
	motion_mag->run();

	delete motion_mag;
	
	return 0;
}
