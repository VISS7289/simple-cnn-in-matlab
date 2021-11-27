%读取测试集
function [testMat,testLabel]=readFileTest
    %读取img
    filename = 'train-images.idx3-ubyte';
    fileImg = fopen(filename,'r');
    %读取文件前16个字节
    f = fread(fileImg,16,'uint8');
    %前四字节是idx文件格式说明
    MagicNum = ((f(1)*256+f(2))*256+f(3))*256+f(4);
    %图片数量
    ImageNum = ((f(5)*256+f(6))*256+f(7))*256+f(8);
    %行数（四字节，用32位整型表述）：此值为28
    rows = ((f(9)*256+f(10))*256+f(11))*256+f(12);
    %列数（四字节，用32位整型表述）：此值为28
    cols = ((f(13)*256+f(14))*256+f(15))*256+f(16);
    %读取label
    filename1 = 'train-labels.idx1-ubyte';
    fileLab = fopen(filename1,'r');
    %读取文件前8个字节
    f = fread(fileLab,8,'uint8');
    %前四字节是idx文件格式说明
    MagicNum1 = ((f(1)*256+f(2))*256+f(3))*256+f(4);
    %图片数量
    ImageNum1 = ((f(5)*256+f(6))*256+f(7))*256+f(8);
    test = {};
    testMat = zeros(ImageNum,28,28);
    testLabel = zeros(ImageNum,1);
    for i=1:ImageNum
        img = fread(fileImg,rows*cols,'uint8');
        label = fread(fileLab,1,'uint8');
        imgMat = reshape(img,[rows,cols]);
        %读取到的图片行列是相反的
        testMat(i,:,:) = imgMat';
        testLabel(i,1) = label;
%         if(mod(i,1000)==0)
%             disp(i);
%         end
    end
    test = {testMat testLabel};
    save testMat test
end

