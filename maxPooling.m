%³Ø»¯²ã
function res = maxPooling(data,poolingSize)
    [dataRow,dataCol] = size(data);
    res = zeros(dataRow / poolingSize,dataCol / poolingSize);
    for i = 1:dataRow / poolingSize
        for j = 1:dataCol / poolingSize
            res(i,j) = max(max(data(1 + (i - 1) * poolingSize:i * poolingSize,1 + (j - 1) * poolingSize:j * poolingSize)));
        end
    end
end

