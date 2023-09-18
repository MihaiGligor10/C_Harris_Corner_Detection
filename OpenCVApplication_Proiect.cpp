// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine

		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i * width + j;

				dstDataPtrH[gi] = hsvDataPtr[hi] * 510 / 360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void addFactorAtGreyChannel()
{
	int factor;
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		printf("\n Additive factor: ");
		scanf("%d", &factor);

		imshow("image1", src);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<unsigned char>(i, j) + factor >= 255)
				{
					src.at<unsigned char>(i, j) = 255;
				}
				else
				{
					if (src.at<unsigned char>(i, j) + factor <= 0)
					{
						src.at<unsigned char>(i, j) = 0;
					}
					else
					{
						src.at<unsigned char>(i, j) = src.at<unsigned char>(i, j) + factor;
					}
				}
			}
		}

		imshow("image2", src);
	}
}

void multiplyFactorAtGreyChannel()
{
	float factor;
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		printf("\n Multiplying factor: ");
		scanf("%f", &factor);

		imshow("image1", src);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<unsigned char>(i, j) * factor >= 255)
				{
					src.at<unsigned char>(i, j) = 255;
				}
				else
				{
					if (src.at<unsigned char>(i, j) * factor <= 0)
					{
						src.at<unsigned char>(i, j) = 0;
					}
					else
					{
						src.at<unsigned char>(i, j) = src.at<unsigned char>(i, j) * factor;
					}
				}
			}
		}

		imshow("image2", src);
		imwrite("C:\\Users\\Mihai\\Desktop\\Desktop\\Faculta\\Procesarea imaginilor\\OpenCVApplication-VS2019_OCV3411_basic\\Images\\img.png", src);
	}
}

void fourCadrans()
{
	Mat img(256, 256, CV_8UC3);

	for (int i = 0; i < 128; i++)
	{
		for (int j = 0; j < 128; j++)
		{
			img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}

	for (int i = 0; i < 128; i++)
	{
		for (int j = 128; j < 256; j++)
		{
			img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
		}
	}

	for (int i = 128; i < 256; i++)
	{
		for (int j = 0; j < 128; j++)
		{
			img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
		}
	}

	for (int i = 128; i < 256; i++)
	{
		for (int j = 128; j < 256; j++)
		{
			img.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
		}
	}

	imshow("image2", img);
	waitKey(0);
}


void descompunereInTrei()
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		Mat iB(src.rows, src.cols, CV_8UC3);
		Mat iG(src.rows, src.cols, CV_8UC3);
		Mat iR(src.rows, src.cols, CV_8UC3);


		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				Vec3b var = src.at<Vec3b>(i, j);
				iB.at<Vec3b>(i, j) = Vec3b(var[0], 0, 0);
				iG.at<Vec3b>(i, j) = Vec3b(0, var[1], 0);
				iR.at<Vec3b>(i, j) = Vec3b( 0, 0,var[2]);

			}
		}

		imshow("image0", src);
		imshow("image1", iB);
		imshow("image2", iG);
		imshow("image3", iR);
		waitKey(0);

		//imwrite("C:\\Users\\Mihai\\Desktop\\Desktop\\Faculta\\Procesarea imaginilor\\OpenCVApplication-VS2019_OCV3411_basic\\Images\\img.png", src);
	}


}

void grayscale() {

	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		Mat gray(src.rows, src.cols, CV_8UC1);
		

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				Vec3b var = src.at<Vec3b>(i, j);
				gray.at<uchar>(i, j) = (var[0]+var[1]+var[2])/3;
			
			}
		}

		imshow("image0", src);
		imshow("image0", gray);
		waitKey(0);

		//imwrite("C:\\Users\\Mihai\\Desktop\\Desktop\\Faculta\\Procesarea imaginilor\\OpenCVApplication-VS2019_OCV3411_basic\\Images\\img.png", src);
	}

}

void grayToBlackWhite()
{
	Mat src;
	int th;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		Mat bw(src.rows, src.cols, CV_8UC1);

		printf("\n prag: ");
		scanf("%d", &th);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar var = src.at<uchar>(i, j);
				if ((int)var >= th)
				{
					bw.at<uchar>(i, j) =0;
				}
				else {
					bw.at<uchar>(i, j) =255;
				}

				

			}
		}

		imshow("image0", src);
		imshow("image0", bw);
		waitKey(0);

		//imwrite("C:\\Users\\Mihai\\Desktop\\Desktop\\Faculta\\Procesarea imaginilor\\OpenCVApplication-VS2019_OCV3411_basic\\Images\\img.png", src);
	}
}


void RGB_to_HSV() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		float b;
		float g;
		float r;
		float M;
		float m;
		float C;
		float V;
		float S;
		float H;
		float H_norm;
		float S_norm;
		float V_norm;

		Mat hue(src.rows, src.cols, CV_8UC1);
		Mat saturation(src.rows, src.cols, CV_8UC1);
		Mat value(src.rows, src.cols, CV_8UC1);


		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				Vec3b var = src.at<Vec3b>(i, j);
				b = (float)var[0] / 255;
				g = (float)var[1] / 255;
				r = (float)var[2] / 255;

				M = max(max(b,g),r);
				m = min(min(b,g),r);

				C = M - m;
				V = M;
				if (V != 0) {
					S = C / V;
				}
				else
				{
					S = 0;
				}

				if(C != 0) {
					if (M == r) H = 60 * (g - b) / C;
					if (M == g) H = 120 + 60 * (b - r) / C;
					if (M == b) H = 240 + 60 * (r - g) / C;
				}
				else // grayscale
					H = 0;
				if(H < 0)
					H = H + 360;

				H_norm = H * 255 / 360;
				S_norm = S * 255;
				V_norm = V * 255;

				hue.at<uchar>(i, j) = H_norm;
				saturation.at<uchar>(i, j) = S_norm;
				value.at<uchar>(i, j) = V_norm;

			}
		}

		imshow("image0", src);
		imshow("hue", hue);
		imshow("saturation", saturation);
		imshow("value", value);
		
		waitKey(0);

		//imwrite("C:\\Users\\Mihai\\Desktop\\Desktop\\Faculta\\Procesarea imaginilor\\OpenCVApplication-VS2019_OCV3411_basic\\Images\\img.png", src);
	}
}

bool isInside(Mat img, int i, int j) {
	if ((i < img.rows) && (j < img.cols) && (i >= 0) && (j >= 0))
		return true;
	else
		return false;

}


void showHistogram(const string& name, int* hist, const int hist_cols,

	const int hist_height) {

	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
	// constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];

	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
		// colored in magenta

	}
	imshow(name, imgHist);
}



void calcul_histograma(Mat src ,int* x )
{

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				x[src.at<uchar>(i, j)]++;
			}
		}
}

void calcul_histograma_acumulator(Mat src, int* x,int m)
{

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			x[src.at<uchar>(i, j)*m/256]++;
		}
	}
}

void FDP(Mat src, int* hist, float* d)
{
	 int m = src.cols * src.rows;

	for (int i = 0; i < 256; i++)
	{
		d[i] = (float)hist[i] / (float)m;
	}
	
}

void afisare_histograma() {
		char fname[MAX_PATH];

		while (openFileDlg(fname)) {
			Mat img = imread(fname, IMREAD_GRAYSCALE);
			int* hist = (int*)calloc(256, sizeof(int));
			float* x1 = (float*)calloc(256, sizeof(float));
			calcul_histograma(img, hist);
			FDP(img, hist, x1);
			showHistogram("histogram", hist, 256, 300);
			waitKey(0);

			free(hist);
			hist = NULL;
			free(x1);
			x1 = NULL;
		}
		
}

void afisare_histograma_acumulator() {
	char fname[MAX_PATH];
	
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		int* hist = (int*)calloc(128, sizeof(int));
		float* x1 = (float*)calloc(256, sizeof(float));
		calcul_histograma_acumulator(img, hist,128);
		FDP(img, hist, x1);
		showHistogram("histogram", hist, 128,256);
		waitKey(0);

		free(hist);
		hist = NULL;
		free(x1);
		x1 = NULL;
	}

}

bool maxima_locala(float* fdp, int k,int wh)
{
	bool t = true;
	for(int i = k - wh; i < k + wh; i++)
	{
		if (fdp[k] < fdp[i])
			t = false;
	}
	return t;
}


int find_closest_histogram_maximum(int oldpixel, std::vector<int>maxima, std::vector<int>thresholds)
{
	
	for (int i = 0; i < thresholds.size(); i++) {
		if (oldpixel < thresholds.at(i))
		{
			return maxima.at(i);
		}
	}
	return 255;
}

void reducere_niveluri_gri()
{
	// Compute fdp     int wh = 5;

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		Mat aux(img.rows, img.cols, CV_8UC1);
		int* hist = (int*)calloc(256, sizeof(int));
		float* x1 = (float*)calloc(256, sizeof(float));
		calcul_histograma(img, hist);
		FDP(img, hist, x1);
		//showHistogram("histogram", hist, 128, 256);
		//waitKey(0);

		int wh = 5;
		float th = 0.0003;
		std::vector<int> maxima;
		maxima.push_back(0);
		// Compute maxima vector  
		for (int k = 0 + wh; k < 255 - wh; k++)
		{
			float m=0;
			// compute average 
			for (int i = k - wh; i < k + wh; i++)
			{ 
				m = m + x1[i];
			}
			m = m / (2 * wh+1);

			       
			bool is_local_maxima=maxima_locala(x1,k,wh);
			// Check if is local maxima       
			// ... todo       

			if (x1[k] > m + th && is_local_maxima)
			{
				maxima.push_back(k);
			}
		}

		maxima.push_back(255);
		std::vector<int> thresholds;
		for (int i = 0; i < maxima.size() - 1; i++)
		{
			thresholds.push_back((maxima.at(i) + maxima.at(i + 1)) / 2);
			printf("%d \n", thresholds.at(i));
		}
		

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				aux.at<uchar>(i, j)=find_closest_histogram_maximum((int)img.at<uchar>(i,j),maxima,thresholds);
			}
		}
        free(hist);
		hist = NULL;
		free(x1);
		x1 = NULL;
		imshow("image0", img);
		imshow("image1", aux);
		waitKey(0);
	}

}

	
void reducere_niveluri_gri_Floyd_Steinberg()
{
	// Compute fdp     int wh = 5;

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		Mat aux(img.rows, img.cols, CV_8UC1);
		int* hist = (int*)calloc(256, sizeof(int));
		float* x1 = (float*)calloc(256, sizeof(float));
		calcul_histograma(img, hist);
		FDP(img, hist, x1);
		
		int wh = 5;
		float th = 0.0003;
		std::vector<int> maxima;
		maxima.push_back(0);
		// Compute maxima vector  
		for (int k = 0 + wh; k < 255 - wh; k++)
		{
			float m = 0;
			// compute average 
			for (int i = k - wh; i < k + wh; i++)
			{
				m = m + x1[i];
			}
			m = m / (2 * wh + 1);


			bool is_local_maxima = maxima_locala(x1, k, wh);
		     

			if (x1[k] > m + th && is_local_maxima)
			{
				maxima.push_back(k);
			}
		}

		free(hist);
		hist = NULL;
		free(x1);
		x1 = NULL;

		maxima.push_back(255);
		std::vector<int> thresholds;
		for (int i = 0; i < maxima.size() - 1; i++)
		{
			thresholds.push_back((maxima.at(i) + maxima.at(i + 1)) / 2);
			printf("%d \n", thresholds.at(i));
		}


		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) 
			{
				int o = img.at<uchar>(i, j);
				aux.at<uchar>(i, j) = o;
				int n = find_closest_histogram_maximum(o, maxima, thresholds);
			
				img.at<uchar>(i, j) = n;
				int eroare = o - n;
				if (isInside(img, i, j + 1) == 1)
				{
					img.at<uchar>(i, j + 1) = img.at<uchar>(i, j + 1) + 7 * eroare / 16;
				}
				if (isInside(img, i + 1, j - 1) == 1)
				{
					img.at<uchar>(i + 1, j - 1) = img.at<uchar>(i + 1, j - 1) + 3 * eroare / 16;
				}
				if (isInside(img, i + 1, j) == 1)
				{
					img.at<uchar>(i + 1, j) = img.at<uchar>(i + 1, j) + 5 * eroare / 16;
				}
				if (isInside(img, i + 1, j + 1) == 1)
				{
					img.at<uchar>(i + 1, j + 1) = img.at<uchar>(i + 1, j + 1) + eroare / 16;
				}
			}
		}
		
		imshow("image0", img);
		imshow("image1", aux);
		waitKey(0);
	}

}

void reducere_niveluri_gri_pe_canalul_hue() {

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		Mat aux(img.rows, img.cols, CV_8UC3);
		cvtColor(img, aux, COLOR_BGR2HSV);


		imshow("aux initial", aux);
		waitKey(0);
		

		Mat channels[3];
		split(aux, channels);


		int* hist = (int*)calloc(256, sizeof(int));
		float* x1 = (float*)calloc(256, sizeof(float));
		calcul_histograma(channels[0], hist);
		FDP(channels[0], hist, x1);
		//showHistogram("histogram", hist, 128, 256);
		//waitKey(0);

		int wh = 5;
		float th = 0.0003;
		std::vector<int> maxima;
		maxima.push_back(0);
		// Compute maxima vector  
		for (int k = 0 + wh; k < 255 - wh; k++)
		{
			float m = 0;
			// compute average 
			for (int i = k - wh; i < k + wh; i++)
			{
				m = m + x1[i];
			}
			m = m / (2 * wh + 1);


			bool is_local_maxima = maxima_locala(x1, k, wh);
			// Check if is local maxima       
			// ... todo       

			if (x1[k] > m + th && is_local_maxima)
			{
				maxima.push_back(k);
			}
		}

		maxima.push_back(255);
		std::vector<int> thresholds;
		for (int i = 0; i < maxima.size() - 1; i++)
		{
			thresholds.push_back((maxima.at(i) + maxima.at(i + 1)) / 2);
			printf("%d \n", thresholds.at(i));
		}


		for (int i = 0; i < channels[0].rows; i++) {
			for (int j = 0; j < channels[0].cols; j++) {
				channels[0].at<uchar>(i, j) = find_closest_histogram_maximum((int)channels[0].at<uchar>(i, j), maxima, thresholds);
			}
		}
		free(hist);
		hist = NULL;
		free(x1);
		x1 = NULL;

		merge(channels,3,aux);

		cvtColor(aux, aux, COLOR_HSV2BGR);

		imshow("img", img);
		imshow("aux", aux);
		waitKey(0);
	}

}



void DrawCross(Mat& img, Point p, int size, Scalar color, int thickness)
{
	line(img, Point(p.x - size / 2, p.y), Point(p.x + size / 2, p.y), color, thickness, 8);
	line(img, Point(p.x, p.y - size / 2), Point(p.x, p.y + size / 2), color, thickness, 8);
}


void lab4(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	Mat copy = Mat::zeros(src->rows, src->cols, CV_8UC3);
	Vec3b pixel;
	int r = 0;
	int c = 0;
	int r_ = 0;
	int c_ = 0;
	int perimetru=0;
	float T = 0;
	float elongatia=0;
	int cmin = 999999;
	int cmax = 0;
	int rmin = 999999;
	int rmax = 0;
	int aux = 0;
	int aux2 = 0;
	float phi = 0;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		int aria=0;
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
		pixel = Vec3b((*src).at<Vec3b>(y, x)[0], (*src).at<Vec3b>(y, x)[1], (*src).at<Vec3b>(y, x)[2]);


		if (pixel[0] ==255 && pixel[1] ==255 && pixel[2] == 255)
		{
			printf("Background\n");
		}
		else
		{
			
			for (int i = 0; i < (*src).rows; i++) {
				for (int j = 0; j < (*src).cols; j++) {
					if (pixel == (*src).at<Vec3b>(i, j))
					{
						aria++;
						r = r + i;
						c = c + j;


						if (((*src).at<Vec3b>(i + 1, j) != pixel) || ((*src).at<Vec3b>(i, j + 1) != pixel) ||
							((*src).at<Vec3b>(i - 1, j) != pixel) || ((*src).at<Vec3b>(i, j - 1) != pixel) ||
							((*src).at<Vec3b>(i - 1, j - 1) != pixel) || ((*src).at<Vec3b>(i + 1, j + 1) != pixel) ||
							((*src).at<Vec3b>(i - 1, j + 1) != pixel) || ((*src).at<Vec3b>(i + 1, j - 1) != pixel))
						{
							perimetru++;
							copy.at<Vec3b>(i, j) = Vec3b(255, 255, 255);

						}

						if (i < rmin)
							rmin = i;

						if (i > rmax)
							rmax = i;

						if (i < cmin)
							cmin = j;

						if (i > cmax)
							cmax = j;

						
					}
						
				}
				
			}
			if (aria != 0)
			{
				r_ = r / aria;
				c_ = c / aria;
			}

			
			copy.at<Vec3b>(r_, c_) = Vec3b(255, 255, 255);
			copy.at<Vec3b>(r_-1, c_) = Vec3b(255, 255, 255);
			copy.at<Vec3b>(r_+1, c_) = Vec3b(255, 255, 255);
			copy.at<Vec3b>(r_-2, c_) = Vec3b(255, 255, 255);
			copy.at<Vec3b>(r_+2, c_) = Vec3b(255, 255, 255);
			copy.at<Vec3b>(r_, c_+1) = Vec3b(255, 255, 255);
			copy.at<Vec3b>(r_, c_-1) = Vec3b(255, 255, 255);
			copy.at<Vec3b>(r_, c_+2) = Vec3b(255, 255, 255);
			copy.at<Vec3b>(r_, c_-2) = Vec3b(255, 255, 255);


			T = 4 * CV_PI * ((float)aria / (perimetru * perimetru));
			elongatia = (float)(cmax - cmin + 1) / (rmax - rmin + 1);
			for (int i = 0; i < (*src).rows; i++) {
				for (int j = 0; j < (*src).cols; j++) {
					if (pixel == (*src).at<Vec3b>(i, j))
					{
						aux += (i - r_) * (j - c_);
						aux2 += pow((j - c_), 2) - pow((i - r_), 2);
					}
				}
			}
			aux = aux * 2;
			float tangenta = atan2(aux, aux2);
			phi = tangenta * (180 / CV_PI);
			phi = phi / 2;
	
		}
		printf("aria este = %d \n",aria);		
		printf("centrele de masa sunt: \n rand: %d\n coloana: %d\n", r_, c_);
		printf("perimetrul este: %d \n", perimetru);
		printf("factorul de subtiere este: %f \n", T);
		printf("elongatia este: %f \n", elongatia);
		printf("theta este: %f \n", phi);

		imshow("copy", copy);
		waitKey(0);
		//printf("cmax este: %d \n", cmax);
		//printf("elongatia este: %f \n", elongatia);

	}

}

void MouseClick()
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		namedWindow("My Window", 1);
		setMouseCallback("My Window", lab4, &src);
		imshow("My Window", src);
		waitKey(0);
	}

}


void etichetare()
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname,IMREAD_GRAYSCALE);
		int label = 0;
		Mat labels = Mat::zeros(src.rows, src.cols, CV_16UC1);
		Mat c(src.rows, src.cols, CV_8UC3); 
		
		
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j)== 0 && labels.at<ushort>(i, j) == 0)
				{
					label++;
					std::queue<Point> Q;
					labels.at<ushort>(i, j) = label;
					Q.push({i,j});
					while(!Q.empty()){
						Point p = Q.front(); 
						Q.pop();
						int di[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
						int dj[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
						for (int k = 0; k < 8; k++)
						{
							if ((src.at<uchar>(p.x + di[k], p.y + dj[k]) == 0) && (labels.at<ushort>(p.x + di[k], p.y + dj[k]) == 0)) {
								Point neighbor = Point(p.x + di[k], p.y + dj[k]);
								labels.at<ushort>(p.x + di[k], p.y + dj[k]) = label;
								Q.push( neighbor );
							}
							
						}
							
					}

				}

			}
		}


		std::vector<Vec3b> culori;
		culori.push_back(Vec3b(255, 255, 255));

		for (int i = 0; i < label; i++)
		{
			culori.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
		}

		for (int i = 0; i < labels.rows; i++) {
			for (int j = 0; j < labels.cols; j++)
			{
				c.at<Vec3b>(i, j)=culori.at(labels.at<ushort>(i,j));
			}
		}
		imshow("My Window", src);
		imshow("copia", c);
		waitKey(0);

	}
}

void contur()
{
	Mat src;
	
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		Mat	aux(src.rows, src.cols, CV_8UC1, Scalar(255));


		int di[8] = { 0,-1,-1,-1, 0, 1, 1, 1 };
		int dj[8] = { 1, 1, 0,-1,-1,-1, 0, 1 };
		
		int dir = 7;
		//int derivativeDirValue;

		Point p1,p2,p3,p4;
		bool obiect = false;

		std::vector<Point> contur;
		std::vector<int> dirVec;

		for (int i = 1; i < src.rows; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				if (src.at<uchar>(i, j) !=255) {
					p1.x = i;
					p1.y = j;
					obiect = true;
					break;
				}
			}
			if (obiect == true)
				break;
		}

		

		for (int i = 0; i < 8; i++) {
			int x = p1.x + di[(dir+6) % 8];
			int y = p1.y + dj[(dir + 6) % 8];
			if (src.at<uchar>(x, y) == 0)
			{
				p2.x = x;
				p2.y = y;
				break;
			}
			dir++;
		}
		
		contur.push_back(p1);
		contur.push_back(p2);

		dirVec.push_back(dir);

		p4 = p2;

		while (!((p3.x == p1.x) && (p3.y == p1.y) && (p4.x == p2.x) && (p4.y == p2.y))) {
			p3 = p4;
			if (dir % 2 != 1) {
				for (int k = 0; k < 8; k++) {
					int x = p3.x + di[(dir + 7) % 8];
					int y = p3.y + dj[(dir + 7) % 8];
					if (src.at<uchar>(x, y) == 0) {
						p4.x = x;
						p4.y = y;
						break;
					}
					dir++;
				}
				dir = (dir + 7) % 8;
			}
			else {
				for (int k = 0; k < 8; k++) {
					int x = p3.x + di[(dir + 6) % 8];
					int y = p3.y + dj[(dir + 6) % 8];
					if (src.at<uchar>(x, y) == 0) {
						p4.x = x;
						p4.y = y;
						break;
					}
					dir++;
				}
				dir = (dir + 6) % 8;
			}
			contur.push_back(p4);
			dirVec.push_back(dir);
		}

		for (int i = 0; i < contur.size(); i++)
		{
			aux.at<uchar>(contur[i].x, contur[i].y) = 0;
		}

		
		printf("Dir vector = ");
		for (int i = 0; i < dirVec.size()-2; i++)
		{
			printf("%d ", dirVec[i]);
		}


		printf("\nDerivata dir vector = ");
		for (int i = 1; i < dirVec.size()-2; i++)
		{
			printf("%d ", (8 + dirVec[i] - dirVec[i - 1]) % 8);
		}

		imshow("My Window", src);
		imshow("aux", aux);
		waitKey(0);
	}	
}

Mat dilatareCalcul(Mat src)
{
	Mat dst(src.rows, src.cols, CV_8UC1);
	uchar val;
	int di[8] = { -1,-1,-1, 0, 0, 1,1,1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			dst.at<unsigned char>(i, j) = 255;
		}
	}

	
	for (int i = 1; i < dst.rows-1; i++)
	{
		for (int j = 1; j < dst.cols-1; j++)
		{
			val = src.at<uchar>(i, j);
			if (val == 0) {
				for (int k = 0; k < 8; k++) {
					dst.at<uchar>(i + di[k], j + dj[k]) = 0;
				}
			}
		}
	}
	
	return dst;
}

void dilatare()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("img", dilatareCalcul(src));
		imshow("imgOriginala", src);
		waitKey(0);
	}
}


Mat eroziuneCalcul(Mat src)
{
	Mat dst(src.rows, src.cols, CV_8UC1);
	uchar val;
	int di[8] = { -1,-1,-1, 0, 0, 1,1,1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	bool test= false;
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			dst.at<unsigned char>(i, j) = 255;
		}
	}


	for (int i = 1; i < dst.rows - 1; i++)
	{
		for (int j = 1; j < dst.cols - 1; j++)
		{
			bool test = false;
			if (src.at<uchar>(i, j) == 0)
			{
					for (int k = 0; k < 8; k++) {
						if (src.at<uchar>(i + di[k], j + dj[k]) == 255)
						{
							test = true;
							break;
						}

					}
					
					if (test == true)
					{
						dst.at<uchar>(i, j) = 255;
					}
					else
					{
						dst.at<uchar>(i, j) = 0;
					}
					}
			
		}
	}

	return dst;
}

void eroziune()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("img", eroziuneCalcul(src));
		imshow("imgOriginala", src);
		waitKey(0);
	}
}
Mat inchidereCalcul(Mat src) {
	return eroziuneCalcul(dilatareCalcul(src));
}
void inchidere() {
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("img", eroziuneCalcul(dilatareCalcul(src)));
		imshow("imgOriginala", src);
		waitKey(0);
	}
}


Mat deschidereCalcul(Mat src) {
	return dilatareCalcul(eroziuneCalcul(src));
}
void deschidere() {
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("img", deschidereCalcul(src));
		imshow("imgOriginala", src);
		waitKey(0);
	}
}



void dilatare_eroziune_n_ori() {
	char fname[MAX_PATH];
	Mat src;
	int a,b;
	printf("introduceti 1 pentru dilatare\nintroduceti 2 pentru eroziune\nintroduceti 3 pentru deschidere\nintroduceti 4 pentru inchidere\ndaca introduceti orice alt numar operatia default este dilatarea\n > ");
	scanf("%d",&a);
	printf("%d", a);
	printf("\nintroduceti de cate ori vreti sa aplicati operatia :\n > \n");
	scanf("%d", &b);

	if (a == 2)
	{
		while (openFileDlg(fname))
		{

			src = imread(fname, IMREAD_GRAYSCALE);
			for (int i = 0; i < b; i++)
			{
				 src=eroziuneCalcul(src);
			}
			imshow("imagine erodata", src);
			waitKey(0);
		}
	}
	else if(a==3)
	{
		while (openFileDlg(fname))
		{

			src = imread(fname, IMREAD_GRAYSCALE);
			for (int i = 0; i < b; i++)
			{
				src=deschidereCalcul(src);
			}
			imshow("imagine open", src);
			waitKey(0);
		}
	}
	else if (a == 4)
	{
		while (openFileDlg(fname))
		{

			src = imread(fname, IMREAD_GRAYSCALE);
			for (int i = 0; i < b; i++)
			{
				src = inchidereCalcul(src);
			}
			imshow("imagine close", src);
			waitKey(0);
		}
	}
	else
	{
		while (openFileDlg(fname))
		{

			src = imread(fname, IMREAD_GRAYSCALE);
			for (int i = 0; i < b; i++)
			{
				src = dilatareCalcul(src);
			}
			imshow("imagine dilatata", src);
			waitKey(0);
		}
	}

	
}

void extragerea_conturului() {
	char fname[MAX_PATH];
	Mat src,dst;
	while (openFileDlg(fname))
	{

		src = imread(fname, IMREAD_GRAYSCALE);
		Mat contur(src.rows, src.cols, CV_8UC1);
		dst = eroziuneCalcul(src);

		for (int i = 0; i < dst.rows; i++)
		{
			for (int j = 0; j < dst.cols; j++)
			{
				contur.at<uchar>(i, j) = 255;
			}
		}
		
		for (int i = 0; i < dst.rows ; i++)
		{
			for (int j = 0; j < dst.cols; j++)
			{
				if (src.at<uchar>(i,j)==0) {
					if (dst.at<uchar>(i, j) == 255)
					{
						contur.at<uchar>(i, j) = 0;
					}
				}
			}
		}

		imshow("imagine dst", dst);
		imshow("imagine contur", contur);
		imshow("imagine", src);


		waitKey(0);
	}
}


void media_dev_standard_histograma() {
	char fname[MAX_PATH];
	Mat src;
	
	while (openFileDlg(fname))
	{
		float media=0, aux=0, nrPixeli=0,deviatia=0;
		int* vec = (int*)calloc(256, sizeof(int));
		int* cumulativa = (int*)calloc(256, sizeof(int));
		
		src = imread(fname, IMREAD_GRAYSCALE);
	
		nrPixeli = src.rows * src.cols;
		calcul_histograma(src, vec);

		for (int i = 0; i < 255; i++)
		{
			aux = aux + vec[i] * i;
		}
		
		media = aux / nrPixeli;
		printf("media = %f\n", media);


		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				deviatia = deviatia + (src.at<uchar>(i, j) - media)*(src.at<uchar>(i, j) - media);
			}
		}
		deviatia = deviatia / nrPixeli;
		deviatia=sqrt(deviatia);
		printf("deviatia = %f\n", deviatia);

		

		for (int i = 0; i < 255; i++)
		{
			for (int j = 0; j < i; j++)
			{
				cumulativa[i]+= vec[j] ;
			}
		}

		showHistogram("histogramCumulativ", cumulativa, 256, 300);
		showHistogram("histogram", vec, 256, 300);
		waitKey(0);
		free(vec);
		vec = NULL;

		free(cumulativa);
		cumulativa = NULL;
	}
}


void determinare_prag_binarizare() {
	char fname[MAX_PATH];
	Mat src,dst;

	while (openFileDlg(fname))
	{
		int* vec = (int*)calloc(256, sizeof(int));
		float T1 = 0, T2 = 0, uG1 = 0, uG2=0,imax = 0, imin = 0;
		src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst(src.rows, src.cols, CV_8UC1);


		calcul_histograma(src, vec);

		for (int i = 0; i < 255; i++)
		{
			if (vec[i] != 0)
			{
				imin = i;
				break;
			}
			
		}

		for (int i = 255; i > 0; i--)
		{
			if (vec[i] != 0)
			{
				imax = i;
				break;
			}

		}
		T1 = (imin + imax) / 2;

		float m1 = 0, m2 = 0,tt=0;


		do {

			T2=T1;
			for (int j = imin; j <= T1; j++){
				m1 += vec[j] ;
				uG1 += vec[j] * j;
			}
	
			uG1 = uG1 / m1;


			for (int i = T1+1; i < imax; i++){
				m2 += vec[i];
				uG2 += vec[i] * i;
			}
	
			uG2 = uG2 / m2;

			T1= (uG1 + uG2) / 2;
			tt = (T1 - T2);
		} while (tt > 0.1);


		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) < T1)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else
				{
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
		printf("%f",T1);
		
		imshow("original", src);
		imshow("binarizata", dst);
		waitKey(0);
		free(vec);
		vec = NULL;	
	}
}


void functii_transformare_histograma()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		float gamma;
		int gmin = 0, gmax = 0, imin=0, imax=0, offset;
		int* vec = (int*)calloc(256, sizeof(int));
		int* contrast = (int*)calloc(256, sizeof(int));
		int* g = (int*)calloc(256, sizeof(int));
		int* intensitate = (int*)calloc(256, sizeof(int));
		int* negativ = (int*)calloc(256, sizeof(int));
		src = imread(fname, IMREAD_GRAYSCALE);

		
		
		calcul_histograma(src,vec);
;
		int M = src.rows * src.cols;

		for (int i = 0; i < 256; i++) {
			negativ[i] = vec[255 - i];
		}


		printf("insert new contrast interval: ");
		scanf("%d %d", &gmin, &gmax);

		for (int i = 0; i < 255; i++)
		{
			if (vec[i] != 0)
			{
				imin = i;
				break;
			}

		}

		for (int i = 255; i > 0; i--)
		{
			if (vec[i] != 0)
			{
				imax = i;
				break;
			}

		}

		for (int i = 0; i < 256; i++) {
			int index;
			index = gmin + (i - imin) * ((gmax - gmin) / (float)(imax - imin));
			if ((index >= 0) && (index < 256))
				contrast[index] = vec[i];
		}


		printf("\ninsert offset: ");
		scanf("%d", &offset);

		for (int i = 0; i < 256; i++) {
			if ((i + offset > 255) || (i + offset < 0))
				intensitate[i] = 0;
			else
				intensitate[i] = vec[i + offset];
		}


		printf("insert gamma correction: ");
		scanf("%f", &gamma);


		for (int i = 0; i < 256; i++) {
			int index;
			index = 255 * pow((i / 255.0), gamma);

			if ((index >= 0) && (index < 256))
				g[index] = vec[i];
		}


	
		
		showHistogram("originala_hist", vec, 256, 300);
		showHistogram("negativ_hist", negativ, 256, 300);
		showHistogram("contrast_hist", contrast, 256, 300);
		showHistogram("gamma_hist", g, 256, 300);
		showHistogram("luminozitate_hist", intensitate, 256, 300);
		//imshow("negativ_img", n_img);
		waitKey(0);

	}
}


int H[7][7] = { { 1, 2, 1, 1, 1, 1, 1 },
				{ 2, 4, 2, 1, 1, 1, 1 },
				{ 1, 2, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 1, 1 } };

int H1[7][7] = { { 0, -1, 0, 1, 1, 1, 1 },
				{ -1, 4, -1, 1, 1, 1, 1 },
				{ 0, -1, 0, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 1, 1 } };


void low_pass_filter(int H[7][7])
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst(src.rows, src.cols, CV_8UC1);
		
		int w = 3, c = 0;
		int k = w / 2;

		for (int i = 0; i < w; i++)
		{
			for (int j = 0; j < w; j++)
			{
				c=c + H[i][j];
			}
		}

		for (int i = k; i < src.rows-k; i++)
		{
			for (int j = k; j < src.cols-k; j++)
			{
				float suma = 0;
				for (int u = 0; u < w; u++)
				{
					for (int v = 0; v < w; v++)
					{
						suma = suma + H[u][v] * src.at<uchar>(i + u - k, j + v - k);
					}
				}
				suma = suma / c;
				dst.at<uchar>(i, j) = (uchar)suma;
			}
		}

		imshow("imagine src",src);
		imshow("imagine dst", dst);
		waitKey(0);
	}

}



void high_pass_filter(int H1[7][7])
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst(src.rows, src.cols, CV_8UC1);

		int w = 3, sum_positive = 0,sum_negative=0;
		float S = 0.0f;
		int k = w / 2;



		for (int i = 0; i < w; i++)
		{
			for (int j = 0; j < w; j++)
			{
				(H1[i][j] > 0) ? (sum_positive += H1[i][j]) : (sum_negative -= H1[i][j]);
			}
		}
		S = 1 / (float)(2 * max(sum_positive, sum_negative));

		for (int i = k; i < src.rows - k; i++)
		{
			for (int j = k; j < src.cols - k; j++)
			{
				float suma = 0;
				for (int u = 0; u < w; u++)
				{
					for (int v = 0; v < w; v++)
					{
						suma = suma + H1[u][v] * src.at<uchar>(i + u - k, j + v - k);
					}
				}
				suma = S* suma +128;
				dst.at<uchar>(i, j) = (uchar)suma;
			}
		}

		imshow("imagine src", src);
		imshow("imagine dst", dst);
		waitKey(0);
	}

}

void filtru_median()
{
	Mat src;
	Mat finalImage;

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		src = imread(fname, 0);
		int w;
		finalImage = src.clone();


		while (true)
		{
			printf("\nalege dimensiunea filtrului (3 , 5 sau 7): ");
			scanf("%d", &w);
			
			if (w == 3 || w == 5 || w == 7)
			{
				break;
			}
		}

		double t = (double)getTickCount();

		for (int i = 0; i < src.rows - w; i++) {
			for (int j = 0; j < src.cols - w; j++) {
				std::vector<int> valori;
				for (int x = 0; x < w; x++) {
					for (int y = 0; y < w; y++) {
						valori.push_back(src.at<uchar>(i + x, j + y));
					}
				}
				sort(valori.begin(), valori.end());
				finalImage.at<uchar>(i + w / 2, j + w / 2) = valori.at(pow(w, 2) / 2);
			}
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.2f [ms] \n", t * 1000);
		imshow("finala", finalImage);
		imshow("originala", src);
		waitKey(0);
	}
}

void filtru_gaussian()
{
	Mat src;
	Mat finalImage;

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		src = imread(fname, 0);
		

		finalImage = src.clone();

		int w = 0;
		while (true)
		{
			printf("\nalege dimensiunea filtrului (3 , 5 sau 7): ");
			scanf("%d", &w);

			if (w == 3 || w == 5 || w == 7)
			{
				break;
			}
		}
		double t = (double)getTickCount();

		float nucleu[300][300];
		float row = (float)w / 6;
		int mid = w / 2;

		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {

				nucleu[i][j] = (1.0 / (2 * 3.14 * pow(row, 2))) * exp(-(float)(pow((i - mid), 2) + pow((j - mid), 2)) / (2.0 * pow(row, 2)));

			}
		}

	

		for (int i = w / 2; i < src.rows - w / 2; i++) {
			for (int j = w / 2; j < src.cols - w / 2; j++) {

				float aux = 0;

				for (int u = 0; u < w; u++) {
					for (int v = 0; v < w; v++) {
						aux = aux + nucleu[u][v] * src.at<uchar>(i + u - w / 2, j + v - w / 2);
					}
				}
				finalImage.at<uchar>(i, j) = (char)(aux);

			}
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.2f [ms] \n", t * 1000);
		imshow("finala", finalImage);
		imshow("originala", src);
		waitKey(0);

	}
}

void metoda_Canny_de_detectie_a_muchiilor()
{
	Mat src;
	Mat blur,mod,mf,directie,aux;
	float x, y;
	int d;

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		src = imread(fname, 0);


		blur = src.clone();
		GaussianBlur(src, blur, Size(3, 3),0,0);
		/*int w = 0;
		while (true)
		{
			printf("\nalege dimensiunea filtrului (3 , 5 sau 7): ");
			scanf("%d", &w);

			if (w == 3 || w == 5 || w == 7)
			{
				break;
			}
		}
		double t = (double)getTickCount();

		float nucleu[300][300];
		float row = (float)w / 6;
		int mid = w / 2;

		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {

				nucleu[i][j] = (1.0 / (2 * 3.14 * pow(row, 2))) * exp(-(float)(pow((i - mid), 2) + pow((j - mid), 2)) / (2.0 * pow(row, 2)));

			}
		}



		for (int i = w / 2; i < src.rows - w / 2; i++) {
			for (int j = w / 2; j < src.cols - w / 2; j++) {

				float aux = 0;

				for (int u = 0; u < w; u++) {
					for (int v = 0; v < w; v++) {
						aux = aux + nucleu[u][v] * src.at<uchar>(i + u - w / 2, j + v - w / 2);
					}
				}
				blur.at<uchar>(i, j) = (char)(aux);

			}
		}
		*/
	
		mod = src.clone();
		directie = Mat::zeros(src.size(), CV_8UC1);
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {

				x = -blur.at<uchar>(i - 1, j - 1) - 2 * blur.at<uchar>(i , j-1) - blur.at<uchar>(i + 1, j - 1) + blur.at<uchar>(i - 1, j + 1) + 2 * blur.at<uchar>(i , j+1) + blur.at<uchar>(i + 1, j + 1);

				y = blur.at<uchar>(i - 1, j - 1) + 2 * blur.at<uchar>(i - 1, j) + blur.at<uchar>(i - 1, j + 1) - blur.at<uchar>(i + 1, j - 1) - 2 * blur.at<uchar>(i + 1, j) - blur.at<uchar>(i + 1, j + 1);

				mod.at<uchar>(i, j) = sqrt(x * x + y * y) / 5.65;

				float teta = atan2((float)y, (float)x);
				if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) d = 0;
				if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) d = 1;
				if ((teta > -PI / 8 && teta < PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) d = 2;
				if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8)) d = 3;
				directie.at<uchar>(i, j) = d;



			}
		}
		
		mf = mod.clone();
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				if (directie.at<uchar>(i, j) == 0) {
					if (mf.at<uchar>(i, j) < mf.at<uchar>(i + 1, j) || mf.at<uchar>(i, j) < mf.at<uchar>(i - 1, j)) {
						mf.at<uchar>(i, j) = 0;
					}
				}
				if (directie.at<uchar>(i, j) == 3) {
					if (mf.at<uchar>(i, j) < mf.at<uchar>(i - 1, j - 1) || mf.at<uchar>(i, j) < mf.at<uchar>(i + 1, j + 1)) {
						mf.at<uchar>(i, j) = 0;
					}
				}
				if (directie.at<uchar>(i, j) == 2) {
					if (mf.at<uchar>(i, j) < mf.at<uchar>(i, j - 1) || mf.at<uchar>(i, j) < mf.at<uchar>(i, j + 1)) {
						mf.at<uchar>(i, j) = 0;
					}
				}
				if (directie.at<uchar>(i, j) == 1) {
					if (mf.at<uchar>(i, j) < mf.at<uchar>(i - 1, j + 1) || mf.at<uchar>(i, j) < mf.at<uchar>(i + 1, j - 1)) {
						mf.at<uchar>(i, j) = 0;
					}
				}

			}
		}


		




		aux = mf.clone();
		//t = ((double)getTickCount() - t) / getTickFrequency();
		//printf("Time = %.2f [ms] \n", t * 1000);
		
		
		float p = 0.1;

		int hist[256] = {};

		for (int i = 1; i < mod.rows - 1; i++) {
			for (int j = 1; j < mod.cols - 1; j++) {
				hist[mf.at<uchar>(i, j)]++;
			}
		}


		int nrNonMuchie = (1 - p) * ((mod.rows - 2) * (mod.cols - 2) - hist[0]);


		int s = 0;
		int index;
		for (index = 1; index < 256; index++) {

			s += hist[index];
			if (s > nrNonMuchie)
				break;
		}
		//int pH = index;
		//int pL = 0.4 * pH;

		int pH = 28;
		int pL = 11;


		printf("high = %d  low = %d \n", pH,pL);


		for (int i = 0; i < mod.rows; i++) {
			for (int j = 0; j < mod.cols; j++) {
				int value = mf.at<uchar>(i, j);

				if (value < pL)
					mf.at<uchar>(i, j) = 0;
				else if (value > pH)
					mf.at<uchar>(i, j) = 255;
				else
					mf.at<uchar>(i, j) = 128;
			}
		}



		Mat	labels(src.rows, src.cols, CV_8UC1);
		labels = Mat::zeros(src.rows, src.cols, CV_8UC1);

		int di[8] = { -1,0,1,0,-1,1,-1,1 };
		int dj[8] = { 0,-1,0,1,-1,1,1,-1 };

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if ((mf.at<uchar>(i, j) == 255) && (labels.at<uchar>(i, j) == 0)) {
					std::queue<Point> Q;
					labels.at<uchar>(i, j) = 1;
					Q.push({ i,j });
					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();

						for (int k = 0; k < 8; k++)
							if ((mf.at<uchar>(q.x + di[k], q.y + dj[k]) == 128)
								&& (labels.at<uchar>(q.x + di[k], q.y + dj[k]) == 0)) {
								mf.at<uchar>(q.x + di[k], q.y + dj[k]) = 255;
								labels.at<uchar>(q.x + di[k], q.y + dj[k]) = 1;
								Q.push({ q.x + di[k], q.y + dj[k] });
							}
					}
				}
			}
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (mf.at<uchar>(i, j) == 128)
					mf.at<uchar>(i, j) = 0;
			}
		}

		
		imshow("Imaginea Initiala", src);
		imshow("Gaus", blur);
		imshow("Modul", mod);
		imshow("Modul Final", aux);
		imshow("edges", mf);
		waitKey(0);

		
		
	}
}




void proiect_detectare_colturi()
{
	char fname[MAX_PATH];
	Mat src, src_gray, R;
	int thresh = 128;
	const char* corners_window = "Corners detected";

	while (openFileDlg(fname)) {
		//citim imaginea
		src = imread(fname, IMREAD_COLOR);
		if (src.empty())
		{
			cout << "Could not open or find the image!\n" << endl;
			return;
		}

		//convertim imaginea in imagine grayscale
		cvtColor(src, src_gray, COLOR_BGR2GRAY);


		double k = 0.04;
		Mat deriv1_x, deriv1_y, x2y2, xy, mtrace;
		Mat deriv_pow2_x, deriv_pow2_y, xy_derivative, x2gauss_deriv, y2gauss_deriv, xy_gauss_deriv;

		//folosim functia Sobel pentru a calcula derivatele x si y ale imaginii
		//Sobel( img_sursa, img_destinatie, depthofimage -1 means same as input, xorder 1,yorder 0,kernelsize 3, BORDER_DEFAULT);
		Sobel(src_gray, deriv1_x, CV_32FC1, 1, 0, 3, BORDER_DEFAULT);
		Sobel(src_gray, deriv1_y, CV_32FC1, 0, 1, 3, BORDER_DEFAULT);

		//calculam valorile din M
		pow(deriv1_x, 2.0, deriv_pow2_x);
		pow(deriv1_y, 2.0, deriv_pow2_y);
		multiply(deriv1_x, deriv1_y, xy_derivative);

		//aplicam filtru gaussian pentru a scapa de posibilele zgomote
		GaussianBlur(deriv_pow2_x, x2gauss_deriv, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
		GaussianBlur(deriv_pow2_y, y2gauss_deriv, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
		GaussianBlur(xy_derivative, xy_gauss_deriv, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);

		

		//calculam elementele necesare pentru a afla determinantul matricii M
		multiply(x2gauss_deriv, y2gauss_deriv, x2y2);
		multiply(xy_gauss_deriv, xy_gauss_deriv, xy);

		//calculam mtrace  (lambda1 + lambda2) si ridicam la putere 
		pow((x2gauss_deriv + y2gauss_deriv), 2.0, mtrace);

		//calculam formula R
		R = (x2y2 - xy) - k * mtrace;


		Mat dst_norm;

		//normalizam matricea R
		//InputArray,	InputOutputArray , alpha = 1 ,beta = 0 , norm_type ,dtype , InputArray 	mask = noArray()
		normalize(R, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

		//parcurgem imaginea
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if ((int)dst_norm.at<float>(i, j) > thresh)
				{
					//parametrii : imaginea , pozitia , culoare , grosime , line type , shift
					circle(src, Point(j, i), 2, Scalar(200), 1, 8, 0);
				}
			}

		}
		namedWindow(corners_window);
		imshow(corners_window, src);
	}
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Lab1 GrayScale Additive\n");
		printf(" 11 - Lab1 GrayScale Multiply\n");
		printf(" 12 - Lab1 4 Cadrans\n");
		

		printf(" 13 - Lab2 Descompunere pe culori\n");
		printf(" 14 - Lab2 Grayscale\n");
		printf(" 15 - Lab2 Grayscale to Black and White\n");
		printf(" 16 - Lab2 Conversie RGB to HSV\n");
		printf(" 17 - Lab2 Verificare daca punctul e in interiorul immaginii\n");

		printf(" 18 - Lab3 afisare histograma \n");
		printf(" 19 - Lab3 afisare histograma acumulatori \n");
		printf(" 20 - Lab3 algoritm reducere niveluri de gri \n");
		printf(" 21 - Lab3 algoritm reducere niveluri de gri Floyd Steinberg\n");
		printf(" 22 - Lab3 algoritm reducere niveluri de gri hue\n");

		printf(" 23 - Lab4 test mouse click\n");

		printf(" 24 - Lab5 etichetare\n");

		printf(" 25 - Lab6 contur\n");

		printf(" 26 - Lab7 dilatare\n");
		printf(" 27 - Lab7 eroziune\n");
		printf(" 28 - Lab7 deschidere\n");
		printf(" 29 - Lab7 inchidere\n");
		printf(" 30 - Lab7 dilatare , eroziune , deschidere sau inchidere de n ori\n");
		printf(" 31 - Lab7 Extragerea conturului\n");

		printf(" 32 - Lab8 Media, deviatia standard, histograma si histograma cumulativa a nivelurilor de intensitate.\n");
		printf(" 33 - Lab8 Determinare automata a pragului de binarizare\n");
		printf(" 34 - Lab8 Functiile de transformare a histogramei\n");

		printf(" 35 - Lab9 Filtru low pass\n");
		printf(" 36 - Lab9 Filtru high pass\n");

		printf(" 37 - Lab10 filtru median \n");
		printf(" 38 - Lab10 filtru gaussian \n");

		printf(" 39 - Lab11 metoda Canny de detectie a muchiior \n");
		printf(" 40 - Proiect : detectarea colturilor\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle();
			break;
		case 4:
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			addFactorAtGreyChannel();
			break;
		case 11:
			multiplyFactorAtGreyChannel();
			break;
		case 12:
			fourCadrans();
			break;
		case 13:
			descompunereInTrei();
			break;
		case 14:
			grayscale();
			break;
		case 15:
			grayToBlackWhite();
			break;
		case 16:
			RGB_to_HSV();
			break;
		case 17:
			printf("%d \n", isInside(Mat(256, 256, CV_8UC1), 256, 100));
			Sleep(2000);
			break;
		case 18:
			afisare_histograma();
			break;
		case 19:
			afisare_histograma_acumulator();
			break;
		case 20:
			reducere_niveluri_gri();
			break;
		case 21:
			reducere_niveluri_gri_Floyd_Steinberg();
			break;
		case 22:
			reducere_niveluri_gri_pe_canalul_hue();
			break;
		case 23:
			MouseClick();
			break;
		case 24:
			etichetare();
			break;
		case 25:
			contur();
			break;
		case 26:
			dilatare();
			break;
		case 27:
			eroziune();
			break;
		case 28:
			deschidere();
			break;
		case 29:
			inchidere();
			break;
		case 30:
			dilatare_eroziune_n_ori();
			break;
		case 31:
			extragerea_conturului();
			break;
		case 32:
			media_dev_standard_histograma();
			break;
		case 33:
			determinare_prag_binarizare();
			break;
		case 34:
			functii_transformare_histograma();
			break;
		case 35:
			low_pass_filter(H);
			break;
		case 36:
			high_pass_filter(H1);
			break;
		case 37:
			filtru_median();
			break;
		case 38:
			filtru_gaussian();
			break;
		case 39:
			metoda_Canny_de_detectie_a_muchiilor();
			break;
		case 40:
			proiect_detectare_colturi();
			break;
		}

	} while (op != 0);
	return 0;
}