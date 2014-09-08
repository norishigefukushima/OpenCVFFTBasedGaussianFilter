#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
using namespace cv;
using namespace std; 

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_viz"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_videostab"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_video"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ts"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_superres"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_stitching"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ocl"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_objdetect"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_nonfree"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ml"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_legacy"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
#pragma comment(lib, "opencv_gpu"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_flann"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_features2d"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_calib3d"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_viz"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_videostab"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_video"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ts"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_superres"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_stitching"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ocl"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_objdetect"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_nonfree"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ml"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_legacy"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
#pragma comment(lib, "opencv_gpu"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_flann"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_features2d"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_calib3d"CV_VERSION_NUMBER".lib")
#endif

void fftShift(Mat magI)
{

    // crop if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void imshowFFTSpectrum(string wname, const Mat& complex )
{
	Mat magI;
	Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
	split(complex, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

    magnitude(planes[0], planes[1], magI);    // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)

	// switch to logarithmic scale: log(1 + magnitude)
	magI += Scalar::all(1.0);
    log(magI, magI);

	fftShift(magI);
    normalize(magI, magI, 1, 0, NORM_INF); // Transform the matrix with float values into a
                                              // viewable image form (float between values 0 and 1).
    imshow(wname, magI);
}

void computeIDFT(Mat& complex, Mat& dest)
{
	idft(complex, dest,DFT_REAL_OUTPUT+DFT_SCALE);

	//dft(complex, work, DFT_INVERSE + DFT_SCALE);
	//Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
	//split(work, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
	//magnitude(planes[0], planes[1], work);	  // === sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
}

void computeDFT(Mat& image, Mat& dest)
{
	Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols ); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    merge(planes, 2, dest);         // Add to the expanded another plane with zeros

	dft(dest, dest, DFT_COMPLEX_OUTPUT);  // furier transform
}

Mat createGaussFilterMask(Size imsize, int radius, bool normalization, bool invert)
{
	// call openCV gaussian kernel generator
	//double sigma = -1;
	double sigma = radius/6.0;
	Mat kernelX = getGaussianKernel(2*radius+1, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(2*radius+1, sigma, CV_32F);
	// create 2d gaus
	Mat kernel = kernelX * kernelY.t();
	//kernel*=255;

	int w = imsize.width-kernel.cols;
	int h = imsize.height-kernel.rows;

	int r = w/2;
	int l = imsize.width-kernel.cols -r;

	int b = h/2;
	int t = imsize.height-kernel.rows -b;

	Mat ret;
	copyMakeBorder(kernel,ret,t,b,l,r,BORDER_CONSTANT,Scalar::all(0));

	// transform mask to range 0..1
	if(normalization) {
		normalize(ret,ret, 0, 1, NORM_MINMAX);
	}

	// invert mask
	if(invert) {
		ret = Mat::ones(ret.size(), CV_32F) - ret;
	}

	return ret;
}


void deconvolute(Mat& img, Mat& kernel)
{
	int width = img.cols;
	int height=img.rows;

	Mat_<Vec2f> src = kernel;
	Mat_<Vec2f> dst = img;

	float eps =  + 0.0001;
	float power, factor, tmp;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			power = src(y,x)[0] * src(y,x)[0] + src(y,x)[1] * src(y,x)[1]+eps;
			factor = 1.f / power;

			tmp = dst(y,x)[0];
			dst(y,x)[0] = (src(y,x)[0] * tmp + src(y,x)[1] * dst(y,x)[1]) * factor;
			dst(y,x)[1] = (src(y,x)[0] * dst(y,x)[1] - src(y,x)[1] * tmp) * factor;	
		}
	}
}

void deconvoluteWiener(Mat& img, Mat& kernel,float snr)
{
	int width = img.cols;
	int height=img.rows;

	Mat_<Vec2f> src = kernel;
	Mat_<Vec2f> dst = img;

	float eps =  + 0.0001f;
	float power, factor, tmp;
	float inv_snr = 1.f / (snr + 0.00001f);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			power = src(y,x)[0] * src(y,x)[0] + src(y,x)[1] * src(y,x)[1]+eps;
			factor = (1.f / power)*(1.f-inv_snr/(power*power + inv_snr));

			tmp = dst(y,x)[0];
			dst(y,x)[0] = (src(y,x)[0] * tmp + src(y,x)[1] * dst(y,x)[1]) * factor;
			dst(y,x)[1] = (src(y,x)[0] * dst(y,x)[1] - src(y,x)[1] * tmp) * factor;	
		}
	}
}

void fftTest(Mat& image)
{
	Mat imgf;image.convertTo(imgf,CV_32F);

	string wname = "fft test";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int r = 20; createTrackbar("r",wname,&r,500);

	int key = 0;
	Mat show;
	Mat dftmat;
	Mat destf;
	while(key!='q')
	{
		GaussianBlur(imgf,destf,Size(2*r+1,2*r+1),r/6.0);

		computeDFT(image,dftmat);
		
		Mat mask;
		/*
		Mat mk = Mat::zeros(dftmat.size(),CV_8U);
		circle(mk, Point(mk.cols/2,mk.rows/2), r,Scalar::all(255),CV_FILLED);
		mk.convertTo(mask,CV_32F,1.0/255.0);
		*/
		mask = createGaussFilterMask(dftmat.size(),r, false, false);
		Mat gmask;
		fftShift(mask);
		computeDFT(mask,gmask);
				
		mulSpectrums(dftmat, gmask, dftmat, DFT_ROWS); // only DFT_ROWS accepted
		//fftShift(dftmat);
		imshowFFTSpectrum("spectrum filtered",dftmat);// show spectrum
		
		computeIDFT(dftmat,show);		// do inverse transform
		
		Mat dest;destf.convertTo(dest,CV_8U);

		Mat showu;
		show.convertTo(showu,CV_8U);
		addWeighted(dest,a/100.0, showu, 1.0-a/100.0,0.0,dest);//0 fft, 1, FIR
		
		imshow(wname,dest);
		key = waitKey(1);
	}
}


void deconvTest(Mat& image)
{
	Mat imgf;image.convertTo(imgf,CV_32F);

	string wname = "fft test";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int r = 20; createTrackbar("r",wname,&r,500);

	int key = 0;
	Mat show;
	Mat dftmat;
	Mat destf;
	while(key!='q')
	{
		GaussianBlur(imgf,destf,Size(2*r+1,2*r+1),r/6.0);

		computeDFT(destf,dftmat);
		
		Mat mask;
		mask = createGaussFilterMask(dftmat.size(),r, false, false);
		Mat gmask;
		fftShift(mask);
		computeDFT(mask,gmask);
				

		deconvolute(dftmat,gmask);
		imshowFFTSpectrum("inv spec",dftmat);

		computeIDFT(dftmat,show);		// do inverse transform
		Mat showu;
		show.convertTo(showu,CV_8U);
		imshow("inv",showu);

		computeIDFT(dftmat,show);		// do inverse transform
		
		Mat dest;destf.convertTo(dest,CV_8U);

		addWeighted(dest,a/100.0, showu, 1.0-a/100.0,0.0,dest);//0 fft, 1, FIR		
		imshow(wname,dest);
		key = waitKey(1);
	}
}
int main()
{
	Mat src = imread("lenna.png",0);
	fftTest(src);
	deconvTest(src);
	return 0;
}