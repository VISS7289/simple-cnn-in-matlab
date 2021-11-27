%��ȡ���Լ�
function [checkMat,checkLabel]=readFileCheck
    %��ȡimg
    filename = 't10k-images.idx3-ubyte';
    fileImg = fopen(filename,'r');
    %��ȡ�ļ�ǰ16���ֽ�
    f = fread(fileImg,16,'uint8');
    %ǰ���ֽ���idx�ļ���ʽ˵��
    MagicNum = ((f(1)*256+f(2))*256+f(3))*256+f(4);
    %ͼƬ����
    ImageNum = ((f(5)*256+f(6))*256+f(7))*256+f(8);
    %���������ֽڣ���32λ���ͱ���������ֵΪ28
    rows = ((f(9)*256+f(10))*256+f(11))*256+f(12);
    %���������ֽڣ���32λ���ͱ���������ֵΪ28
    cols = ((f(13)*256+f(14))*256+f(15))*256+f(16);
    %��ȡlabel
    filename1 = 't10k-labels.idx1-ubyte';
    fileLab = fopen(filename1,'r');
    %��ȡ�ļ�ǰ8���ֽ�
    f = fread(fileLab,8,'uint8');
    %ǰ���ֽ���idx�ļ���ʽ˵��
    MagicNum1 = ((f(1)*256+f(2))*256+f(3))*256+f(4);
    %ͼƬ����
    ImageNum1 = ((f(5)*256+f(6))*256+f(7))*256+f(8);
    test = {};
    checkMat = zeros(ImageNum,28,28);
    checkLabel = zeros(ImageNum,1);
    for i=1:ImageNum
        img = fread(fileImg,rows*cols,'uint8');
        label = fread(fileLab,1,'uint8');
        imgMat = reshape(img,[rows,cols]);
        %��ȡ����ͼƬ�������෴��
        checkMat(i,:,:) = imgMat';
        checkLabel(i,1) = label;
%         if(mod(i,1000)==0)
%             disp(i);
%         end
    end
    check = {checkMat checkLabel};
    save checkMat check
end

