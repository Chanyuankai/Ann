

#include "pch.h"
#include <windows.h>
#include <io.h>
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\ml.hpp>

using namespace cv;
using namespace std;

//定义全局变量
Mat ROI[40];//存储字符图像的矩阵变量
int CharactersNumber1 = 0, CharactersNumber2 = 0; //第一第二行字符个数，即画矩形的个数
string FilePath = "E:\\数据挖掘考试资料\\字符识别\\处理后的图片\\";
string ResizeFilePath = "E:\\数据挖掘考试资料\\字符识别\\处理后的图片\\resize\\";
int zhongzhi;
void fenge(Mat out1, uint invalidBlank, string SavePath, string Saveformat, uint ChartWidth); //out：输入图像  invalidBlank：无效空格的列数，用于区分有用空格和无效间隔 z:分割后的字符图像保存位置及格式
int* xifen(Mat img, int x1, int x2, int hang); //二值化图像 起始列 终止列 第几行
void predictann1(Mat testroi);
int blankimage(Mat inputimage, int point1, int point2, int point3, int point4);

int main()
{
	Mat pSrcImg = imread("E:\\数据挖掘考试资料\\字符识别\\识别图片\\source11.bmp", 0);//从本地读取灰度图
	imshow("原图", pSrcImg);

	//均值滤波
	Mat dst;
	blur(pSrcImg,dst,Size(7,7));
	imwrite(FilePath+"remove1.bmp", dst);
	imshow("remove1", dst);

	//二值化部分 数字黑色0 背景白色255
	Mat pDecImg;
	pDecImg.create(pSrcImg.size(), pSrcImg.type()); //1通道
	pDecImg = pSrcImg.clone();
	imshow("二值化", pDecImg);
	threshold(pSrcImg, pDecImg,0,255,cv::THRESH_OTSU);
	imshow("binary1", pDecImg);
	imwrite(FilePath+"binary1.bmp", pDecImg);

	//腐蚀去噪
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat out1;
	dilate(pDecImg, out1, element);
	imshow("腐蚀", out1);

	//图像分割，分割后的图片在数组Mat ROI[]中
	fenge(out1, 60, FilePath, ".bmp", 110);
	
	//ANN识别
	Mat Fengeimg[40], Resizeimage[40];
	string Pathfile1,PathfileResize;
	cout<<"识别结果为："<<endl;
	for(int i = 0; i< CharactersNumber1+ CharactersNumber2; i++)
	{
		Pathfile1 = FilePath + to_string(i+1) + ".bmp";
		PathfileResize = FilePath +"resize\\" + to_string(i + 1) + ".bmp";
		Fengeimg[i] = imread(Pathfile1, 0);
		resize(Fengeimg[i], Resizeimage[i], Size(14, 28));

		imshow("字符"+to_string(i+1), Resizeimage[i]);
		imwrite(PathfileResize, Resizeimage[i]);
		//
		predictann1(Resizeimage[i]);
		//cout << Resizeimage[i] << endl;
		if (i == CharactersNumber1-1) cout<<endl;
	}

	waitKey(0);
	return 0;
}


void predictann1(Mat testroi)
{
	Ptr<ml::ANN_MLP> bp = ml::ANN_MLP::load("E:/数据挖掘考试资料/字符识别/训练程序/temp/temp/MLPModel.xml");
	const int image_cols = 14;
	const int image_rows = 28;
	Mat  dst;
	//将测试图像转化为1*14*28的向量
	Mat_<float> testMat(1, image_rows * image_cols);
	for (int i = 0; i < image_rows * image_cols; i++)
	{
		testMat.at<float>(0, i) = (float)testroi.at<uchar>(i / 14, i % 14);
	}
	//使用训练好的MLP model预测测试图像
	bp->predict(testMat, dst);
	double maxVal = 0;
	Point maxLoc,minLoc;
	minMaxLoc(dst, NULL, &maxVal, &minLoc, &maxLoc);
	char temp[256];
	if (maxLoc.x <= 9)//0-9  
	{
		sprintf_s(temp, "%d", maxLoc.x);
	}
	else//C、L  
	{
		if (maxLoc.x == 10) sprintf_s(temp, "%c", 67);           //C
		else if (maxLoc.x == 11) sprintf_s(temp, "%c", 76);        //L
		else if (maxLoc.x == 12) sprintf_s(temp, "%c", 58);        //M :
		else if (maxLoc.x == 13) sprintf_s(temp, "%c", 32);        //K  
	}
	//std::cout << "测试结果：" << temp << "置信度:" << maxVal * 100 << "%" << std::endl;
	cout<<temp;
}

void fenge(Mat out1, uint invalidBlank, string SavePath, string Saveformat, uint ChartWidth) //out：输入图像  invalidBlank：无效空格的列数，用于区分有用空格和无效间隔
{
	//第一步：分行 找出两行字符间的空白行，以中值作为分行处
	int p1, flag1,hang[500],t=0;   //zhongzhi=两行间隔的中间值  flag1：空白行标志位 hang[500]：空白行数组 t=0：计数空白行行数
	int step = out1.step;//每行所占字节数  
	uchar* out1_data = (uchar*)out1.data;
	for (int i = 10; i < out1.rows-10; i++)
	{
		flag1 = 0;
		for (int j = 0; j < out1.cols; j++)
		{
			p1 = out1_data[i*step + j];  //像数值
			if (p1 == 0) flag1++;
		}
		if (flag1 ==0) hang[t]=i,t++;
		zhongzhi = hang[t/2];
	}
	/*cout << "zhongzhi=" << zhongzhi << endl;*/

	//第二步：判断第一行空白列，并将对应的列数记录
	uint p, flag, blanklie1 = 0, lie1[500], blanklie2 = 0, lie2[500];  //p：像数值 flag:空白列标志位 blanklie：空白列数 lie[400]：存放空白列
	for (int j = 0; j < out1.cols; j++)
	{
		flag = 0;
		for (int i = 0; i < zhongzhi; i++)
		{
			p = out1_data[i*step + j];
			if (p == 0) flag++;
		}
		if (flag == 0) blanklie1++, lie1[blanklie1] = j;
	}
	//判断第二行空白列
	for (int j = 0; j < out1.cols; j++)
	{
		flag = 0;
		for (int i = out1.rows - zhongzhi; i < out1.rows; i++)
		{
			p = out1_data[i*step + j];
			if (p == 0) flag++;
		}
		if (flag == 0) blanklie2++, lie2[blanklie2] = j;
	}

	//第三步 把第一行零界点存入数组pointZero1
	uint a1 = 0, a2 = 0, pointZero1[400], pointZero2[400]; //a1,a2：第一第二行零界点个数  pointZero1[400], pointZero2[400]：存放零界点
	for (int m = 1; m <= blanklie1; m++)
	{
		if (lie1[m + 1] != lie1[m] + 1 || lie1[m] != lie1[m - 1] + 1)
		{
			pointZero1[a1] = lie1[m];
			a1++;
		}
	}
	//第二行把零界点存入数组pointZero2
	for (int m = 1; m <= blanklie2; m++)
	{
		if (lie2[m + 1] != lie2[m] + 1 || lie2[m] != lie2[m - 1] + 1)
		{
			pointZero2[a2] = lie2[m];
			a2++;
		}
	}
	//第三步 把零界点转化成相应的坐标点，画矩形。去掉数字间的间隙，保留空格，并提取矩形区域
	//int CharactersNumber1 = 0, CharactersNumber2 = 0; //第一第二行字符个数，即画矩形的个数
	Point pointStart1[20], pointStart2[20];//第一第二行矩形起点第x列
	Point pointEnd1[20], pointEnd2[20];//第一第二行矩形终点第y列
	//Mat ROI[40];//存储字符图像的矩阵变量
	int *fg1,*fg2;
	for (int j = 0; j < a1 - 2; j++)
	{
		int xx = blankimage(out1, 0, zhongzhi, pointZero1[j], pointZero1[j + 1]);
		if (((pointZero1[j + 1] - pointZero1[j]) <= invalidBlank && xx==1) || j == 0) j++;
		pointStart1[CharactersNumber1] = Point(pointZero1[j], 0);
		pointEnd1[CharactersNumber1] = Point(pointZero1[j + 1], zhongzhi);
		
		ROI[CharactersNumber1] = out1(Range(0, zhongzhi), Range(pointZero1[j], pointZero1[j + 1]));//将其作为感兴趣区提取出来

		//判断一副图像字符有几个
		if(pointZero1[j + 1] - pointZero1[j] > ChartWidth&&(blankimage(out1, 0, zhongzhi, pointZero1[j], pointZero1[j + 1])==0))
		{
			int breakpoint;
			fg1 = xifen(out1, pointZero1[j], pointZero1[j + 1], 1);
			/*cout << "分隔列 shuzhi=" << fg1[0]<< fg1[1] << endl;*/
			breakpoint = fg1[0] - fg1[1] / 2;
			ROI[CharactersNumber1] = out1(Range(0, zhongzhi), Range(pointZero1[j], breakpoint));//将其作为感兴趣区提取出来
			//rectangle(out1, Point(pointZero1[j], 0), Point(breakpoint, zhongzhi), Scalar(0, 0, 255), 1, 4);//用矩形标记字符
			CharactersNumber1++;
			ROI[CharactersNumber1] = out1(Range(0, zhongzhi), Range(breakpoint, pointZero1[j + 1]));//将其作为感兴趣区提取出来
			//rectangle(out1, Point(breakpoint, 0), Point(pointZero1[j + 1], zhongzhi), Scalar(0, 0, 255), 1, 4);//用矩形标记字符
		}
		//else rectangle(out1, pointStart1[CharactersNumber1], pointEnd1[CharactersNumber1], Scalar(0, 0, 255), 1, 4);//用矩形标记字符
		CharactersNumber1++;
	}
	//处理第二行
	for (int i = 0; i < a2 - 2; i++)
	{
		int xx = blankimage(out1, zhongzhi + 3, out1.rows, pointZero2[i], pointZero2[i + 1]);
		if ((pointZero2[i + 1] - pointZero2[i]) <= invalidBlank && xx == 1 || i == 0) i++;
		pointStart2[CharactersNumber2] = Point(pointZero2[i], zhongzhi+3);
		pointEnd2[CharactersNumber2] = Point(pointZero2[i + 1], out1.rows);
		ROI[CharactersNumber1 + CharactersNumber2] = out1(Range(zhongzhi + 3, out1.rows), Range(pointZero2[i], pointZero2[i + 1]));
		
		//判断一副字符非空格图像字符有几个
		if (pointZero2[i + 1] - pointZero2[i] > ChartWidth&&(blankimage(out1, zhongzhi + 3, out1.rows, pointZero2[i], pointZero2[i + 1])==0))
		{
			int breakpoint2;
			fg2 = xifen(out1, pointZero2[i], pointZero2[i + 1], 2);
			/*cout << "分隔列 shuzhi=" << fg2[0] << fg2[1] << endl;*/
			breakpoint2 = fg2[0] - fg2[1] / 2;
			ROI[CharactersNumber1 + CharactersNumber2] = out1(Range(zhongzhi + 3, out1.rows), Range(pointZero2[i], breakpoint2));
			//rectangle(out1, Point(pointZero2[i], zhongzhi+3), Point(breakpoint2, out1.rows), Scalar(0, 0, 255), 1, 4);
			CharactersNumber2++;
			ROI[CharactersNumber1 + CharactersNumber2] = out1(Range(zhongzhi + 3, out1.rows), Range(breakpoint2, pointZero2[i + 1]));
			//rectangle(out1, Point(breakpoint2, zhongzhi+3), Point(pointZero2[i + 1], out1.rows), Scalar(0, 0, 255), 1, 4);
		}

		//else rectangle(out1, pointStart2[CharactersNumber2], pointEnd2[CharactersNumber2], Scalar(0, 0, 255), 1, 4);
		CharactersNumber2++;
	}
	//显示图像
	//imshow("矩形", out1);
	//第一第二行字符图片保存
	uint x = 1;//图像起始编号
	string path;
	for (int number1 = 0; number1 < CharactersNumber1+ CharactersNumber2; number1++)//总共有CharactersNumber1+ CharactersNumber2个字符图片
	{
		//imshow("字符", ROI[number]);
		path = SavePath + to_string(x) + Saveformat;
		imwrite(path, ROI[number1]);
		x++;
	}
	
}

//对于第一次分割后还有两个字符的图像进行细分
int* xifen(Mat img, int x1, int x2, int hang) //二值化图像 起始列 终止列 第几行
{
	uint dd=0, mubiaohang[100],DisHang[300], DisHangNum[300], DisHangNum1[300], DisNum = 0, q1 = 0, yuzhi;
	int qi, zhi, step = img.step;//每行所占字节数  
	uchar* img_data = (uchar*)img.data;
	if (hang == 1) qi = 0, zhi = zhongzhi;
	else qi = zhongzhi, zhi = img.rows;
	if (x2 - x1 > 110)
	{
		for (int ii = x1 + 15; ii <= x2 - 15; ii++)
		{
			DisNum = 0;
			for (int kk = qi; kk < zhi; kk++)
			{
				if (img_data[kk*step + ii] == 0)
				{
					DisNum++;
				}
			}
			DisHangNum[q1] = DisNum;
			DisHangNum1[q1] = DisNum;
			DisHang[q1] = ii;
			q1++;
			//cout << ii << "DisNum = " << DisNum << endl;
		}
		std::sort(DisHangNum1, DisHangNum1 + q1);
		yuzhi = DisHangNum1[10];
		//cout<<"yuzhi="<<yuzhi<<endl;
		for (int w = 0; w < q1; w++)
		{
			if (DisHangNum[w] <= yuzhi)
			{
				mubiaohang[dd] = DisHang[w];
				/*cout<< mubiaohang[dd] <<endl;*/
				dd++;
			}
		}
		int zhongdian,lianxu=0, last_lianxu = 0, dNum=0;
		for (int zz = 0; zz < dd - 1; zz++)
		{
			if (mubiaohang[zz] + 1 == mubiaohang[zz + 1])
			{
				lianxu++;
			}
			if (lianxu == dd - 1)
			{
				zhongdian = mubiaohang[zz+1];
				last_lianxu = dd - 1;
				break;
			}
			if(mubiaohang[zz] + 1 != mubiaohang[zz + 1])
			{
				zhongdian = mubiaohang[zz];
				if (lianxu > last_lianxu) last_lianxu = lianxu;
				lianxu = 0;
			}
		}
		int fengedian;
		/*cout << "zhongdian="<<zhongdian<<"     最大连续的段=" << last_lianxu << endl;*/
		fengedian = zhongdian - last_lianxu / 2;
		static int zhishu[2];
		zhishu[0] = fengedian;
		zhishu[1] = last_lianxu;
		
		return zhishu;
	}
}

//判断图像是否是空白图像的
int blankimage(Mat inputimage, int point1, int point2, int point3, int point4)
{
	Mat img = inputimage(Range(point1, point2), Range(point3, point4));
	int p1, flag=0;
	int step = img.step;//每行所占字节数  
	uchar* img_data = (uchar*)img.data;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			p1 = img_data[i*step + j];  //像数值
			if (p1 == 0) flag++;
		}
	}
	if (flag <= 10) return 1;
	else return 0;
}
