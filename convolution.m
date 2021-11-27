%全连接层（实现卷积操作）
function res = convolution(data,Kernel,paddingSize)
    [dataRow,dataCol] = size(data);
    [kernelRow,kernelCol] = size(Kernel);
    res = zeros(dataRow - kernelRow + paddingSize + 1,dataCol - kernelCol + paddingSize + 1);
    padding = zeros(dataRow + 2 * paddingSize,dataCol + 2 * paddingSize);
    padding(paddingSize + 1:paddingSize + dataRow,paddingSize + 1:paddingSize + dataCol) = data;
    for i = 1:(dataRow - kernelRow + 2 * paddingSize + 1)
        for j = 1:(dataCol - kernelCol + 2 * paddingSize + 1)
            res(i,j) = sum(sum(padding(i:i + kernelRow - 1,j:j + kernelRow - 1).*Kernel));
        end
    end
end

