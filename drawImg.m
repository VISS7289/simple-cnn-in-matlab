%�����Լ��е�����ͼ��20�ţ�
function drawImg
test = load('checkMat.mat');
testImg = test.check;
testImg = testImg{1,1};
test = 'mean';
name = 'checkImg';
format = '.bmp';
    for i = 1:20
        img(:,:) = testImg(i,:,:);
        img = img - mean(mean(img)) * ones(28);
        imwrite(img,[test,name,num2str(i),format]);
    end
end

