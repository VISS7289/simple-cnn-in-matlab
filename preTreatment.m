%��һ��+ȥ��ֵ��ʹ��ֵ����0��
function res = preTreatment(mat)
    [matRow,matCol] = size(mat);
    mat = mat/(max(max(abs(mat))));
    res = mat - mean(mean(mat))*ones(matRow,matCol);
end

