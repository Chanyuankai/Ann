#include <windows.h>
#include <io.h>
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\ml.hpp>

using namespace cv;
using namespace std;
using namespace ml;

//训练样本图片
void ann10(void)
{
	const string fileform = "*.bmp";
	const string perfileReadPath = "D:\\resource\\训练样本图片\\";

	const int sample_mun_perclass = 50;//训练字符每类数量 
	const int class_mun = 14;//训练字符类数 数字0-9 字母C和L 冒号 空格 

	const int image_cols = 14;
	const int image_rows = 28;
	string  fileReadName, fileReadPath;
	char temp[256];

	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } };//每一行一个训练样本  
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };//训练样本标签  

	for (int i = 0; i <= class_mun-1; i++)//不同类  
	{
		//读取每个类文件夹下所有图像  
		int j = 0;//每一类读取图像个数计数  
		if (i <= 9)//0-9  
		{
			sprintf_s(temp, "%d",i);
			//printf("%d\n", i);  
		}
		else//C、L  
		{
			if(i==10) sprintf_s(temp, "%c", 67);           //C
			else if (i == 11) sprintf_s(temp, "%c", 76);        //L
			else if (i == 12) sprintf_s(temp, "%c", 77);        //M :
			else if (i == 13) sprintf_s(temp, "%c", 75);        //K  
			//printf("%c\n", i+55);  
		}
		fileReadPath = perfileReadPath + temp + "\\" + to_string(j) +  ".bmp";
		cout << "文件夹" << fileReadPath << endl;
		//do-while循环读取  
		do
		{

			j++;//读取一张图  
			Mat srcImage = imread(perfileReadPath +  temp + "\\" + to_string(j)  + ".bmp",0);
			//imshow("11", srcImage);
			cout << "文件夹1" << perfileReadPath + temp + "\\" + to_string(j) + ".bmp" << endl;
			//Mat trainImage;
			Mat result;

			for (int k = 0; k < image_rows*image_cols; k++)
			{
				trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)srcImage.data[k];
			}
		} while (j < sample_mun_perclass);//如果设置读入的图片数量，则以设置的为准 
	}
	// Set up training data Mat  
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout << "trainingDataMat——OK！" << endl;
	for (int i = 0; i <= class_mun - 1; ++i)
	{
		for (int j = 0; j <= sample_mun_perclass - 1; ++j)
		{
			for (int k = 0; k < class_mun; ++k)
			{
				if (k == i)
					labels[i*sample_mun_perclass + j][k] = 1;

				else
					labels[i*sample_mun_perclass + j][k] = 0;
			}
		}
	}
	Mat labelsMat(class_mun*sample_mun_perclass, class_mun, CV_32FC1, labels);
	cout << "labelsMat:" << endl;
	ofstream outfile("out.txt"); 
	outfile << labelsMat;
	//cout<<labelsMat<<endl;  
	cout << "labelsMat——OK！" << endl;

	//训练代码  
	cout << "training start...." << endl;
	Ptr<ANN_MLP> bp = ANN_MLP::create();
	//4层，输入层神经元个数5，第一个隐层为6,第二个隐层为5,第三个隐层为4,

	//设置层数
	Mat layerSizes = (Mat_<int>(1, 5) << image_rows * image_cols, 128, 128, 128, class_mun);
	bp->setLayerSizes(layerSizes);
	//设置各种参数
	bp->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001,0.1);
	bp->setActivationFunction(ANN_MLP::SIGMOID_SYM,1.0,1.0);
	//终止训练的条件
	bp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001));
	bool trained = bp->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat);
	bp->save("E:\\数据挖掘考试资料\\字符识别\\训练程序\\temp\\temp\\MLPModel.xml");
	cout << "training finish...bpModel.xml saved " << endl;



	////测试
	Mat test, dst;
	test = imread("D:\\resource\\训练样本图片\\\\0\\4.bmp",0);
	if (test.empty())
	{
		std::cout << "can not load image \n" << std::endl;
		/*return -1;*/
	}
	//将测试图像转化为1*128的向量
	Mat_<float> testMat(1, image_rows * image_cols);
	for (int i = 0; i < image_rows * image_cols; i++)
	{
		testMat.at<float>(0, i) = (float)test.at<uchar>(i / 14, i % 14);
	}
	//使用训练好的MLP model预测测试图像
	bp->predict(testMat, dst);
	std::cout << "testMat: \n" << testMat << "\n" << std::endl;
	std::cout << "dst: \n" << dst << "\n" << std::endl;
	double maxVal = 0;
	Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	std::cout << "测试结果：" << maxLoc.x << "置信度:" << maxVal * 100 << "%" << std::endl;
	imshow("test", test);	
}


int main()
{
	ann10();
	
	waitKey(0);

	return 0;
}


