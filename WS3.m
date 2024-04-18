% Question 1 --------------------------------------------------------------

img = imread('x31_f18.tif');
SE_erode = strel('disk',8);
eroded = imerode(img,SE_erode);
imshow(eroded);

SE_dilate = strel('disk',8);
dilated = imdilate(img,SE_dilate);
imshow(dilated);

% Question 2 --------------------------------------------------------------

freemanCode  = imread('FreemanCode.png');
SE = strel('disk',2);
dilate = imdilate(freemanCode,SE);
erode = imerode(freemanCode,SE);
diff = dilate - erode;
figure;
subplot(1,2,1);
imshow(freemanCode);
title('Original Image');
subplot(1,2,2);
imshow(diff);
title('Boundary Image');


% Question 3 --------------------------------------------------------------

coins = imread('coins.png');
bin = imbinarize(coins);
fill = imfill(bin,'holes');
figure; imshow(fill);

% Question 7 (A) ----------------------------------------------------------

sigma = 2.5
Gaussian_filter = fspecial('gaussian', [5 5], sigma);
Gaussian_img = imfilter(img, Gaussian_filter, 'same');
Laplacian = [-1 -1 -1; -1 8 -1; -1 -1 -1];
Laplacian_img = conv2(Gaussian_img, Laplacian, 'same');
edges = Laplacian_img

figure;
subplot(1,2,1); imshow(img); title('Original Image');
subplot(1,2,2); imshow(Laplacian_img); title('Filtered Image');

% Question 7 (B)

edges_canny = edge(img, 'canny', 0.5, 1.2);
imshow(edges_canny);
title('Canny Filter');

% Question 8 (A) ----------------------------------------------------------

% Perform the Hough Transform
[H,theta,rho] = hough(edges_canny);

% Find peaks in the Hough Transform
P = houghpeaks(H, 50);

% Find lines corresponding to the peaks
lines = houghlines(edges_canny,theta,rho,P);

% Display the original image
figure;
imshow(img);

hold on;

% Plot the edges
plot(lines(1).point1(1), lines(1).point1(2), 'g*', 'LineWidth', 2);
plot(lines(1).point2(1), lines(1).point2(2), 'g*', 'LineWidth', 2);

% Plot the lines
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
end

hold off;

% Question 8 (B)

threshold = graythresh(img);
BW = imbinarize(img, threshold);
BW = edge(BW, 'canny', 0.5, 1.2);
[H,theta,rho] = hough(BW);
P = houghpeaks(H, 50);
lines = houghlines(BW,theta,rho,P);
figure;
imshow(img);

hold on;
% Plot the edges
plot(lines(1).point1(1), lines(1).point1(2), 'g*', 'LineWidth', 2);
plot(lines(1).point2(1), lines(1).point2(2), 'g*', 'LineWidth', 2);

% Plot the lines
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
end
hold off;


% Question 9 --------------------------------------------------------------

cameraman = imread('cameraman.tif');

% Simple Global Thresholding
level = graythresh(cameraman);
bw_simple = imbinarize(cameraman, level);

% Global Thresholding
threshold_value = 128;
bw_global = cameraman > threshold_value;

% Otsu's Method
bw_otsu = imbinarize(cameraman, level);

% Edge-guided Thresholding
edges = edge(cameraman, 'sobel');
bw_edge_guided = imbinarize(cameraman, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.5);

% Multiple Thresholding
numThresholds = 3;
thresholds = multithresh(cameraman, numThresholds);
binaryImage = imquantize(cameraman, thresholds);

% Create a figure with subplots
figure;

subplot(2, 3, 1);
imshow(cameraman);
title('Original Image');

subplot(2, 3, 2);
imshow(bw_simple);
title('Simple Global Thresholding');

subplot(2, 3, 3);
imshow(bw_global);
title('Global Thresholding');

subplot(2, 3, 4);
imshow(bw_otsu);
title('Otsu''s Method');

subplot(2, 3, 5);
imshow(binaryImage, []);
title('Multiple Thresholding');

subplot(2, 3, 6);
imshow(bw_edge_guided);
title('Edge-guided Thresholding');


% Question 10 -------------------------------------------------------------

% Read the original image
butterfly = imread('butterfly.png');

% Perform K-means clustering with k=3
[L_k3, Centers_k3] = imsegkmeans(butterfly, 3);

% Perform K-means clustering with k=6
[L_k6, Centers_k6] = imsegkmeans(butterfly, 6);

% Display the original image
subplot(1, 3, 1);
imshow(butterfly);
title('Original Image');

% Display the segmented image with k=3
subplot(1, 3, 2);
B_k3 = labeloverlay(butterfly, L_k3);
imshow(B_k3);
title('Segmented Image with k=3');

% Display the segmented image with k=6
subplot(1, 3, 3);
B_k6 = labeloverlay(butterfly, L_k6);
imshow(B_k6);
title('Segmented Image with k=6');





