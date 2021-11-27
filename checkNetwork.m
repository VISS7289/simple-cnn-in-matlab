function [accuracy] = checkNetwork
    %��ȡ����
    parameter = load('trainMat.mat');
    train = parameter.train;
    [kernelC1,kernelC2,biasC1,biasC2,weight1,weight2] = train{1,:};
    %��ȡ��⼯
    [checkMat,checkLabel]=readFileCheck;
    imgNum = size(checkMat,1);
    numTrue = 0; %��ȷ����������
    for i = 1:imgNum
        img(:,:) = checkMat(i,:,:);
        label = checkLabel(i,1);
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
        [~,maxcol] = find(output==max(output));
        predict = maxcol - 1;
        if (label == predict) 
            numTrue = numTrue + 1;
        end
         if (mod(i,100) == 0)
             disp(numTrue/i);
        end
    end
    accuracy = numTrue / imgNum;
end

