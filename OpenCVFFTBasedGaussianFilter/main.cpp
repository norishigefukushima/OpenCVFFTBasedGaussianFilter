#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std; 

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#endif

#define SIGMA_CLIP 6.0f
inline int sigma2radius(float sigma)
{
	return (int)(SIGMA_CLIP*sigma+0.5f);
}

inline float radius2sigma(int r)
{
	return (r/SIGMA_CLIP+0.5f);
}

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

void magnitudeFFT(const Mat& complex, Mat& dest)
{
	vector<Mat> planes;
	split(complex, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], dest);    // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
}

void imshowFFTSpectrum(string wname, const Mat& complex )
{
	Mat magI;
	magnitudeFFT(complex, magI);// sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
	
	// switch to logarithmic scale: log(1 + magnitude)
	log(magI+1.0, magI);

	fftShift(magI);
	normalize(magI, magI, 1, 0, NORM_INF); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).
	imshow(wname, magI);
}

void computeIDFT(Mat& complex, Mat& dest)
{
	idft(complex, dest,DFT_REAL_OUTPUT+DFT_SCALE);

	//following is same process of idft:DFT_REAL_OUTPUT+DFT_SCALE
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

	copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_REPLICATE);

	Mat imgf;
	padded.convertTo(imgf,CV_32F);	
	dft(imgf, dest, DFT_COMPLEX_OUTPUT);  // furier transform

	//other implimentation
	//Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	//merge(planes, 2, dest);         // Add to the expanded another plane with zeros
	//dft(dest, dest, DFT_COMPLEX_OUTPUT);  // furier transform
}

Mat createCircleMask(Size imsize, int radius)
{
	Mat ret;
	Mat mk = Mat::zeros(imsize,CV_8U);
	circle(mk, Point(mk.cols/2,mk.rows/2), radius,Scalar::all(1.f),CV_FILLED);
	mk.convertTo(ret,CV_32F,1.0);

	return ret;
}

Mat createGaussFilterMask(Size imsize, int radius)
{
	// call openCV gaussian kernel generator
	double sigma = radius2sigma(radius);
	Mat kernelX = getGaussianKernel(2*radius+1, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(2*radius+1, sigma, CV_32F);
	// create 2d gaus
	Mat kernel = kernelX * kernelY.t();

	int w = imsize.width-kernel.cols;
	int h = imsize.height-kernel.rows;

	int r = w/2;
	int l = imsize.width-kernel.cols -r;

	int b = h/2;
	int t = imsize.height-kernel.rows -b;

	Mat ret;
	copyMakeBorder(kernel,ret,t,b,l,r,BORDER_CONSTANT,Scalar::all(0));

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

void fftConvolutionTest(Mat& image)
{
	Mat imgf;image.convertTo(imgf,CV_32F);

	string wname = "fft test";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int r = 20; createTrackbar("r",wname,&r,500);
	int sw = 0; createTrackbar("sw",wname,&sw,2);

	Mat dftmat;
	Mat filterdFIR;
	Mat filteredFFT;

	Mat filterdFIR_8u;
	Mat filteredFFT_8u;

	int key = 0;
	while(key!='q')
	{
		//reference Gaussian filter of FIR
		GaussianBlur(imgf,filterdFIR,Size(2*r+1,2*r+1),radius2sigma(r));

		computeDFT(image,dftmat);

		Mat mask;
		Mat kernel;

		if(sw == 0)
		{
			//Gaussian kernel
			mask = createGaussFilterMask(dftmat.size(),r);
			fftShift(mask);
			computeDFT(mask,kernel);
		}
		else if(sw == 1)
		{
			//circle kernel
			mask = createCircleMask(dftmat.size(),r);
			fftShift(mask);
			vector<Mat> v;
			v.push_back(mask);
			v.push_back(Mat::zeros(dftmat.size(),CV_32FC1));

			merge(v,kernel);
		}
		else
		{
			//no kernel
			kernel = Mat::ones(dftmat.size(),CV_32FC2);
		}	

		mulSpectrums(dftmat, kernel, dftmat, DFT_ROWS); // only DFT_ROWS accepted
		imshowFFTSpectrum("spectrum filtered",dftmat);// show spectrum

		computeIDFT(dftmat,filteredFFT);		// do inverse transform

		//for visualization
		filterdFIR.convertTo(filterdFIR_8u,CV_8U);
		filteredFFT.convertTo(filteredFFT_8u,CV_8U);

		addWeighted(filterdFIR_8u,a/100.0, filteredFFT_8u, 1.0-a/100.0,0.0,filteredFFT_8u);//0 fft, 1, FIR		

		imshow(wname,filteredFFT_8u);
		key = waitKey(1);
	}
	destroyAllWindows();
}

void fftDeconvolutionTest(Mat& image)
{
	Mat imgf;image.convertTo(imgf,CV_32F);

	string wname = "fft test";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int r = 20; createTrackbar("r",wname,&r,500);
	int noise_sigma = 0; createTrackbar("noise sigma",wname,&noise_sigma,1000);
	int sw = 0; createTrackbar("switch",wname, &sw,1);//switch deconv method
	int snr = 0; createTrackbar("snr:Weiner",wname, &snr,10000);//parameter for Weiner filter

	Mat dftmat;
	Mat filterdFIR;
	Mat filteredFFT;

	Mat filterdFIR_8u;
	Mat filteredFFT_8u;

	int key = 0;

	Mat GaussNoise = Mat::zeros(imgf.size(),CV_32F);
	while(key!='q')
	{
		//reference Gaussian filter of FIR
		GaussianBlur(imgf,filterdFIR,Size(2*r+1,2*r+1),radius2sigma(r));

		//add Gaussian noise
		randn(GaussNoise,0,noise_sigma/10.0);
		filterdFIR += GaussNoise;
		
		//FFT input signal
		computeDFT(filterdFIR, dftmat);

		//generating kernel
		Mat mask = createGaussFilterMask(dftmat.size(),r);
		fftShift(mask);

		Mat kernel;
		computeDFT(mask,kernel);//generate Gaussian Kernel


		//deconvolution
		if(sw==0)
		{
			deconvolute(dftmat,kernel);//normal deconvolution
		}
		else
		{
			deconvoluteWiener(dftmat,kernel,snr*snr/100.0);//for Wiener filter
		}

		imshowFFTSpectrum("spectrum filtered",dftmat);// show spectrum

		computeIDFT(dftmat,filteredFFT);		// do inverse transform

		//for visualization
		filterdFIR.convertTo(filterdFIR_8u,CV_8U);
		filteredFFT.convertTo(filteredFFT_8u,CV_8U);

		addWeighted(filterdFIR_8u,a/100.0, filteredFFT_8u, 1.0-a/100.0,0.0,filteredFFT_8u);//0 fft, 1, FIR		

		imshow(wname,filteredFFT_8u);
		key = waitKey(1);
	}
	destroyAllWindows();
}

int main()
{
	Mat src = imread("lenna.png",0);

	fftConvolutionTest(src);//convolution test

	fftDeconvolutionTest(src);//deconvolution test

	return 0;
}