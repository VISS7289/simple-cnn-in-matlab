function main
    clear all;
    clc;
    %��ȡ���Լ�
    [testMat,testLabel]=readFileTest;
    imgNum = size(testMat,1);
    %��ʼ������
    testNum = 3; %����ѭ������
    yita=0.01; %��ʼ������
    %��ʼ�������
    kernelC1 = initKernel(32,3);
    kernelC2 = initKernel(64,14);
    %��ʼ��ƫ����
    biasC1 = (2*rand(1,32) - ones(1,32))/sqrt(20);
    biasC2 = (2*rand(1,64) - ones(1,64))/sqrt(20);
    %��ʼ��Ȩֵ
    weight1 = (2*rand(32,64) - ones(32,64))/sqrt(20);
    weight2 = (2*rand(64,10) - ones(64,10))/sqrt(20);
    for n = 1:testNum
        for i = 1:imgNum
            img(:,:) = testMat(i,:,:);
            label = testLabel(i,1);
            %Ԥ����
            img = preTreatment(img);
            trainMatC1 = zeros(28,28,32);
            trainMatS1 = zeros(14,14,32);
            trainMatB2 = zeros(14,14,64);
            trainMatC2 = zeros(1,1,64);
            trainMatOutPut = zeros(1,64);
            conImg = zeros(14);
            output = zeros(1,10);
            %ȫ���Ӳ�1
            for j = 1:32
                trainMatC1(:,:,j) = convolution(img,kernelC1(:,:,j),1); %���
                trainMatC1(:,:,j) = tanh(trainMatC1(:,:,j)+biasC1(1,j)); %����
                trainMatS1(:,:,j) = maxPooling(trainMatC1(:,:,j),2); %�ػ�
            end
            %ȫ���Ӳ�2
            for j = 1:64
                for k = 1:32
                    conImg = conImg + trainMatS1(:,:,k) * weight1(k,j);
                end
                trainMatB2(:,:,j) = conImg;
                trainMatC2(:,:,j) = convolution(conImg,kernelC2(:,:,j),0); %���
                trainMatOutPut(1,j) = tanh(trainMatC2(:,:,j)+biasC2(1,j)); %����
            end
            %����softmax��
            for j = 1:10
                output(1,j) = exp(trainMatOutPut*weight2(:,j))/sum(exp(trainMatOutPut*weight2));
            end
            %������
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
            %��������
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

