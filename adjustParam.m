function [kernelC1Adjust,kernelC2Adjust,biasC1Adjust,biasC2Adjust,weight1Adjust,weight2Adjust] = adjustParam(yita,Error,label,...
                                                                            img,trainMatC1,trainMatS1,...
                                                                            trainMatB2,trainMatOutPut,output,...
                                                                            kernelC1,kernelC2,biasC1,biasC2,weight1,weight2)
%结点数目
layer_C2 = size(trainMatOutPut,2);
layer_Out = size(output,2);

[c1_row,c1_col,layer_C1] = size(trainMatC1);
[s1_row,s1_col,layer_S1] = size(trainMatS1);
[kernelC1_row,kernelC1_col,~] = size(kernelC1);
[kernelC2_row,kernelC2_col,~] = size(kernelC2);

%网络权值与卷积核的暂存
weight1Temp = weight1;
weight2Temp = weight2;
kernel1Temp = kernelC1;
kernel2Temp = kernelC2;

%Error计算
labelTrue = zeros(1,layer_Out);
labelTrue(1,label+1) = 1;
delta_output = output - labelTrue;

%预处理
delta_weight2_temp = zeros(layer_C2,layer_Out);%64*10
delta_biasC2 = zeros(1,layer_C2);%1*64
delta_kernelC2 = zeros(14,14,layer_C2);%14*14*64
delta_layerC2_temp = zeros(14,14,layer_C2);
delta_weight1 = zeros(layer_S1,layer_C2);%32*64
delta_layerC2 = zeros(14,14,layer_S1);
delta_layerC1 = zeros(28,28,layer_S1);%28*28*32
delta_biasC1 = zeros(1,layer_S1);%1*32
delta_kernelC1 = zeros(3,3,layer_C1);%3*3*32

%更新weight2
for n = 1:layer_Out
    delta_weight2_temp(:,n) = delta_output(1,n) * trainMatOutPut';
end
weight2Temp = weight2Temp - yita * delta_weight2_temp;

%更新偏移量biasC2与卷积核kernelC2
for n = 1:layer_C2
    count = delta_output(1,:) * weight2(n,:)';
    delta_biasC2(1,n) = count * (1 - tanh(trainMatOutPut(1,n)).^2);
    delta_kernelC2(:,:,n) = delta_biasC2(1,n) * trainMatB2(:,:,n);
end
biasC2Adjust = biasC2 - yita * delta_biasC2;
kernel2Temp = kernel2Temp - yita * delta_kernelC2;

%更新weight1
for n = 1:layer_C2
    delta_layerC2_temp(:,:,n) = delta_biasC2(1,n) * kernelC2(:,:,n);
end
for n = 1:layer_S1
    for m = 1:layer_C2
        delta_weight1(n,m) = sum(sum(delta_layerC2_temp(:,:,m) .* trainMatS1(:,:,n)));
    end
end
weight1Temp = weight1Temp - yita * delta_weight1;

%更新偏移量biasC1
for n = 1:layer_S1
    count = 0;
    for m = 1:layer_C2
        count = count + delta_layerC2_temp(:,:,m) * weight1(n,m);
    end
    delta_layerC2(:,:,n) = count;
    delta_layerC1(:,:,n) = kron(delta_layerC2(:,:,n),ones(2,2) / 4) .* (1 - tanh(trainMatC1(:,:,n)) .^ 2);
    delta_biasC1(1,n) = sum(sum(delta_layerC1(:,:,n)));
end
biasC1Adjust = biasC1 - yita * delta_biasC1;

%更新kernelC1
img2 = zeros(length(img)+2);
img2(2:end-1,2:end-1) = img;
for n = 1:layer_C1
    delta_kernelC1(:,:,n) = rot90(conv2(img2,rot90(delta_layerC1(:,:,n),2),'valid'),2);
end
kernel1Temp = kernel1Temp - yita * delta_kernelC1;

%网络权值更新
kernelC1Adjust = kernel1Temp;
kernelC2Adjust = kernel2Temp;
weight1Adjust = weight1Temp;
weight2Adjust = weight2Temp;

end

