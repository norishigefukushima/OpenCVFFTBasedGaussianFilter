OpenCVFFTBasedGaussianFilter
============================
The code is a OpenCV's sample of filtering with DFT by using cv::dct function.   


Note
====
The code contains two demos; one is FFT based convolution (fftConvolutionTest) and anothor is FFT based deconvolution (fftDeconvolutionTest).  

The first demo can convolute an image by the Gaussian kernel and the circle kernel.  
The seconde demo can deconvolute (i.e. deblur) Gaussian blur with normal or Weiner filter.  

The code is tested on OpenCV2.4.9.  

Relust
======
![input image](result/conv_input.png "Input image")  
###Input image  
![input spec](result/conv_input_spec.png "Input image Spectrum")  
###Input image spectrum

![Gaussian convolution image](result/conv_Gaussian.png "Gaussian convoluted image")  
###Gaussian convoluted image  
![Gaussian convolution spec](result/conv_Gaussian_spec.png "Gaussian convolution spectrum")  
###Gaussian convoluted spectrum

![Deconvolution image](result/deconv_Gaussian.png "Gaussian deconvoluted image")  
###Gaussian deconvoluted image  

![N-Deconvolution image](result/deconv_noisy.png "Noisy Gaussian deconvoluted image")  
###Noisy Gaussian deconvoluted image  (Gaussian noise sigma = 5)
![N-Deconvolution spec](result/deconv_noisy_spec.png "Noisy Gaussian deconvoluted spectrum")  
###Noisy Gaussian deconvoluted Spectrum  (Gaussian noise sigma = 5)

![N-Deconvolution image](result/deconv_noisy_Weiner.png "Noisy Weiner deconvoluted image")  
###Noisy Gaussian deconvoluted image  (Gaussian noise sigma = 5)
![N-Deconvolution spec](result/deconv_noisy_Weiner_spec.png "Noisy Weiner deconvoluted spectrum")  
###Noisy Gaussian deconvoluted spectrum  (Gaussian noise sigma = 5)

Todo
====

+ Add more better baundary treatment for deconvolution. Now, We cannot ignore ringing.  
+ Modify generating funcation of Gaussian kernel by directly generation without FFT.

