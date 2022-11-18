function HeatMap = CAMheatmap_squeezenet(image,act,scores,Weights)
%#codegen
imageActivations = act;
% check the classified category from scores
if scores(1) > scores(2)
    classIndex      = 1;
else
    classIndex      = 2;
end
% get classified category's Weight data at the fully conntected layer
weightVector = Weights(classIndex,:);
%weightVector    = mynet.Layers(67).Weights(classIndex,:); 
% calculate Classification Activation Map
weightVectorSize = size(weightVector); 
weightVector = reshape(weightVector,[1 weightVectorSize]); 
dotProduct = bsxfun(@times,imageActivations,weightVector); 
classActivationMap = sum(dotProduct,3); 
originalSize = size(image);
% resize 
classActivationMap = imresize(classActivationMap,originalSize(1:2)); 
imgmap = double(classActivationMap);
range = [min(imgmap(:)) max(imgmap(:))];
heatmap_gray = imgmap - range(1);
heatmap_gray = heatmap_gray/(range(2)-range(1));
heatmap_x = round(heatmap_gray * 255);
%persistent tbl;
%if isempty(tbl)
    tbl = coder.const(jet(256));
%end
% Make sure A is in the range from 1 to size(cm,1)
a = max(1,min(heatmap_x,size(tbl,1)));
% Extract r,g,b components
r = tbl(a,1);
r = reshape(r,[224 224]);
g = tbl(a,2);
g = reshape(g,[224 224]);
b = tbl(a,3);
b = reshape(b,[224 224]);
tmp = im2double(image)*0.3 + (cat(3, r, g, b) * 0.5);
HeatMap = uint8(tmp*255);
end
%% 
%%
#Prediction
function [out, act] = dcnn_predict(in) %#codegen
%
    
%coder.gpu.kernelfun
persistent mynet1;
if isempty(mynet1)
    mynet1 = coder.loadDeepLearningNetwork('NDNet.mat','convnet');
end
out = mynet1.predict(in);
act = mynet1.activations(in, 'relu_conv5','OutputAs','channels');
end
function remoteExePath = getTargetDir()
    targetWorkspaceDir = codertarget.camera.getRemoteBuildDir;
    remoteExePath = codertarget.camera.internal.fullLnxFile(targetWorkspaceDir, pwd);
    remoteExePath = codertarget.camera.internal.w2l(remoteExePath);
end
function out = mat2ocv(img)
    % Converting image data to b2gray
    % 
    sz = size(img);
    Iocv = zeros([prod(sz), 1], 'uint8');
    
    % Reshape input data
    Ir = permute(img, [2 1 3]);
    
    % RRR...GGG... format to BGRBGR...(openCV) format
    Iocv(1:3:end) = Ir(:,:,3);
    Iocv(2:3:end) = Ir(:,:,2);
    Iocv(3:3:end) = Ir(:,:,1);
    out = reshape(Iocv, sz);
end
function [imgPacked] = myNDNet_Postprocess(Iori, num, bbox, scores, wi, he, ch, HeatMap)
% Draw the bounding box around the detected solar cell panels.
% 
% 
%#c
%% Specify constant values
labeltbl = {'Defective';'Good'};
colortbl = [136 8 8; 51 29 248];
%% Draw the bounding box around the detected nuts
img2 = Iori;
for i = 1:num
    idx = (scores(1, i) < scores(2, i)) + 1;
    bbox = round(bbox);
    img2 = insertObjectAnnotation(img2, 'rectangle', bbox(i,:),...
        labeltbl{idx}, 'FontSize', 18, 'Color', colortbl(idx,:));
    tmp = imresize(HeatMap(:,:,:,i), [bbox(i,4)+1 bbox(i,3)+1]);
    img2(bbox(i,2):bbox(i,2)+bbox(i,4),bbox(i,1):bbox(i,1)+bbox(i,3),:) = tmp;    
end
%% Converting image data to opencv
imgPacked = coder.nullcopy(zeros([1,he*wi*ch], 'uint8'));
for i = 1:he
    for j = 1:wi
        imgPacked((i-1)*wi*ch + (j-1)*ch + 3) = img2(i,j,1);
        imgPacked((i-1)*wi*ch + (j-1)*ch + 2) = img2(i,j,2);
        imgPacked((i-1)*wi*ch + (j-1)*ch + 1) = img2(i,j,3);
    end
end
end
function [Iori, solar cell, tnum, bbox] = myNDNet_Preprocess(inImg)
% Extract solar cell busbar and finger for ROI
% processing
% 
    
%#
%% Parameters related to input image
[he, wi, ch] = size(inImg);
ratio = he*wi*ch / 921600;
Iori = coder.nullcopy(zeros([he,wi,ch], 'uint8'));
for i = 1:he
    for j = 1:wi
        Iori(i,j,3) = inImg((i-1)*wi*ch + (j-1)*ch+1);
        Iori(i,j,2) = inImg((i-1)*wi*ch + (j-1)*ch+2);
        Iori(i,j,1) = inImg((i-1)*wi*ch + (j-1)*ch+3);
    end
end
%% Extract solar cell busbar and finger for ROI
% Convert RGB image to grayscale
gray = rgb2gray(Iori);
gray = imadjust(gray);
% Binarize image by using saturation value 
hsv = rgb2hsv(Iori);
%th = otsuthresh(imhist(hsv(:,:,2)));
%bw = hsv(:,:,2) > th;
bw = hsv(:,:,1) > 0.4 & hsv(:,:,1) < 0.75;
% Remove small objects
bw = bwareaopen(bw, 18000*ratio);
[bwl, num]= bwlabel(bw);
% Measure propaties of ROI
stats = regionprops(bwl, 'Area', 'BoundingBox');
bbox = zeros(4);
nuts = coder.nullcopy(zeros([224,224,4], 'uint8'));
tnum = 0;
for i = 1:num
    area = stats(i).Area;
    % Crop the image if measured area is in a specified range 
    if area < 30000*ratio
        if tnum < 4
            bbox(i,:) = stats(i).BoundingBox;
            tmp = imcrop(gray, bbox(i,:));
            nuts(:,:,i) = imresize(tmp, [224 224]);
            tnum = tnum + 1;
        end
    end
end
end
function out = ocv2mat(img, sz)
    % OCR
    sec = sz(1)*sz(2);
    Im = zeros([sz(2), sz(1), sz(3)], 'uint8');
    
    % BGRBGR... format to RRR...GGG... format  
    Im(1:sec) = img(3:3:end);
    Im(sec+1:sec*2) = img(2:3:end);
    Im(sec*2+1:sec*3) = img(1:3:end);
    
    % Reshape matrix
    out = permute(Im, [2 1 3]);
end

function out = targetFunction(img, Weights, arm) %#codegen
if arm
    % Update buildinfo to link with OpenCV library available on target.    
    opencv_linkflags = '`pkg-config --cflags --libs opencv`';
    coder.updateBuildInfo('addLinkFlags',opencv_linkflags);
end
%coder.inline('never');
%coder.gpu.kernelfun
wi = 320;
he = 240;
ch = 3;
%extract ROI as an pre-prosessing
[Iori, solar cell, num, bbox] = myNDNet_Preprocess(img);
%classify detected solar cell busbar and finger
scores = coder.nullcopy(zeros(2,4));
HeatMap = coder.nullcopy(zeros(224,224,4));
assert (num < 4);
for i = 1:num
    indata = repmat(solarcell(:,:,i), [1 1 3]);
    [scores(:,i), act] = dcnn_predict(indata);
    HeatMap(:,:,:,i) = CAMheatmap_squeezenet(indata,act,scores(:,i),Weights);
end
    
%insert annotation as an post-processing
out = myNDNet_Postprocess(Iori, solar cell, bbox, scores, wi, he, ch, HeatMap);
end
