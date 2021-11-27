function main
    clear all;
    clc;
    %读取测试集
    [testMat,testLabel]=readFileTest;
    imgNum = size(testMat,1);
    %初始化参数
    testNum = 3; %测试循环次数
    yita=0.01; %初始化步进
    %初始化卷积核
    kernelC1 = initKernel(32,3);
    kernelC2 = initKernel(64,14);
    %初始化偏移量
    biasC1 = (2*rand(1,32) - ones(1,32))/sqrt(20);
    biasC2 = (2*rand(1,64) - ones(1,64))/sqrt(20);
    %初始化权值
    weight1 = (2*rand(32,64) - ones(32,64))/sqrt(20);
    weight2 = (2*rand(64,10) - ones(64,10))/sqrt(20);
    for n = 1:testNum
        for i = 1:imgNum
            img(:,:) = testMat(i,:,:);
            label = testLabel(i,1);
            %预处理
            img = preTreatment(img);
            trainMatC1 = zeros(28,28,32);
            trainMatS1 = zeros(14,14,32);
            trainMatB2 = zeros(14,14,64);
            trainMatC2 = zeros(1,1,64);
            trainMatOutPut = zeros(1,64);
            conImg = zeros(14);
            output = zeros(1,10);
            %全连接层1
            for j = 1:32
                trainMatC1(:,:,j) = convolution(img,kernelC1(:,:,j),1); %卷积
                trainMatC1(:,:,j) = tanh(trainMatC1(:,:,j)+biasC1(1,j)); %激活
                trainMatS1(:,:,j) = maxPooling(trainMatC1(:,:,j),2); %池化
            end
            %全连接层2
            for j = 1:64
                for k = 1:32
                    conImg = conImg + trainMatS1(:,:,k) * weight1(k,j);
                end
                trainMatB2(:,:,j) = conImg;
                trainMatC2(:,:,j) = convolution(conImg,kernelC2(:,:,j),0); %卷积
                trainMatOutPut(1,j) = tanh(trainMatC2(:,:,j)+biasC2(1,j)); %激活
            end
            %进入softmax层
            for j = 1:10
                output(1,j) = exp(trainMatOutPut*weight2(:,j))/sum(exp(trainMatOutPut*weight2));
            end
            %误差计算
            Error = output(1,label + 1);
            if(Error > 0.98)
                k = k + 1;
            end
            if (mod(k,1000)==0)
                disp(k);
            end
            if(mod(i,20)==0)
               disp(Error); 
            end
            %参数调整
            [kernelC1,kernelC2,biasC1,biasC2,weight1,weight2] = adjustParam(yita,Error,label,...
                                                                            img,trainMatC1,trainMatS1,...
                                                                            trainMatB2,trainMatOutPut,output,...
                                                                            kernelC1,kernelC2,biasC1,biasC2,weight1,weight2);
        end
    end
    train = {kernelC1,kernelC2,biasC1,biasC2,weight1,weight2};
    save trainMat train
    checkNetwork
end

