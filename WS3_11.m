
% Question 11 -------------------------------------------------------------

% Load the image
im = imread('Cards.png');

% Convert the image to grayscale
gray = rgb2gray(im);

% Apply Otsu's thresholding
bw = imbinarize(gray);

% Compute the Euclidean Distance Transform
D = bwdist(~bw);

% Blur the distance transform to reduce noise
sigma = 10;
kernel = fspecial('gaussian', 4*sigma+1, sigma);
D_blurred = imfilter(D, kernel, 'symmetric');

% Apply the watershed algorithm
L = watershed(max(D_blurred(:)) - D_blurred);

% Display the segmented image
imshow(labeloverlay(im, L));
title('Segmented Cards');
