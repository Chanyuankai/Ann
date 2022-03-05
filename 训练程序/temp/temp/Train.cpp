#include <windows.h>
#include <io.h>
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\ml.hpp>

using namespace cv;
using namespace std;
using namespace ml;

//ѵ������ͼƬ
void ann10(void)
{
	const string fileform = "*.bmp";
	const string perfileReadPath = "D:\\resource\\ѵ������ͼƬ\\";

	const int sample_mun_perclass = 50;//ѵ���ַ�ÿ������ 
	const int class_mun = 14;//ѵ���ַ����� ����0-9 ��ĸC��L ð�� �ո� 

	const int image_cols = 14;
	const int image_rows = 28;
	string  fileReadName, fileReadPath;
	char temp[256];

	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } };//ÿһ��һ��ѵ������  
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };//ѵ��������ǩ  

	for (int i = 0; i <= class_mun-1; i++)//��ͬ��  
	{
		//��ȡÿ�����ļ���������ͼ��  
		int j = 0;//ÿһ���ȡͼ���������  
		if (i <= 9)//0-9  
		{
			sprintf_s(temp, "%d",i);
			//printf("%d\n", i);  
		}
		else//C��L  
		{
			if(i==10) sprintf_s(temp, "%c", 67);           //C
			else if (i == 11) sprintf_s(temp, "%c", 76);        //L
			else if (i == 12) sprintf_s(temp, "%c", 77);        //M :
			else if (i == 13) sprintf_s(temp, "%c", 75);        //K  
			//printf("%c\n", i+55);  
		}
		fileReadPath = perfileReadPath + temp + "\\" + to_string(j) +  ".bmp";
		cout << "�ļ���" << fileReadPath << endl;
		//do-whileѭ����ȡ  
		do
		{

			j++;//��ȡһ��ͼ  
			Mat srcImage = imread(perfileReadPath +  temp + "\\" + to_string(j)  + ".bmp",0);
			//imshow("11", srcImage);
			cout << "�ļ���1" << perfileReadPath + temp + "\\" + to_string(j) + ".bmp" << endl;
			//Mat trainImage;
			Mat result;

			for (int k = 0; k < image_rows*image_cols; k++)
			{
				trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)srcImage.data[k];
			}
		} while (j < sample_mun_perclass);//������ö����ͼƬ�������������õ�Ϊ׼ 
	}
	// Set up training data Mat  
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout << "trainingDataMat����OK��" << endl;
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
	cout << "labelsMat����OK��" << endl;

	//ѵ������  
	cout << "training start...." << endl;
	Ptr<ANN_MLP> bp = ANN_MLP::create();
	//4�㣬�������Ԫ����5����һ������Ϊ6,�ڶ�������Ϊ5,����������Ϊ4,

	//���ò���
	Mat layerSizes = (Mat_<int>(1, 5) << image_rows * image_cols, 128, 128, 128, class_mun);
	bp->setLayerSizes(layerSizes);
	//���ø��ֲ���
	bp->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001,0.1);
	bp->setActivationFunction(ANN_MLP::SIGMOID_SYM,1.0,1.0);
	//��ֹѵ��������
	bp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001));
	bool trained = bp->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat);
	bp->save("E:\\�����ھ�������\\�ַ�ʶ��\\ѵ������\\temp\\temp\\MLPModel.xml");
	cout << "training finish...bpModel.xml saved " << endl;



	////����
	Mat test, dst;
	test = imread("D:\\resource\\ѵ������ͼƬ\\\\0\\4.bmp",0);
	if (test.empty())
	{
		std::cout << "can not load image \n" << std::endl;
		/*return -1;*/
	}
	//������ͼ��ת��Ϊ1*128������
	Mat_<float> testMat(1, image_rows * image_cols);
	for (int i = 0; i < image_rows * image_cols; i++)
	{
		testMat.at<float>(0, i) = (float)test.at<uchar>(i / 14, i % 14);
	}
	//ʹ��ѵ���õ�MLP modelԤ�����ͼ��
	bp->predict(testMat, dst);
	std::cout << "testMat: \n" << testMat << "\n" << std::endl;
	std::cout << "dst: \n" << dst << "\n" << std::endl;
	double maxVal = 0;
	Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	std::cout << "���Խ����" << maxLoc.x << "���Ŷ�:" << maxVal * 100 << "%" << std::endl;
	imshow("test", test);	
}


int main()
{
	ann10();
	
	waitKey(0);

	return 0;
}


