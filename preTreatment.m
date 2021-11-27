%归一化+去均值（使均值趋于0）
function res = preTreatment(mat)
    [matRow,matCol] = size(mat);
    mat = mat/(max(max(abs(mat))));
    res = mat - mean(mean(mat))*ones(matRow,matCol);
end

