%³õÊ¼»¯¾í»ýºË
function Kernel = initKernel(layerKernel,sizeKernel)
    Kernel = zeros(sizeKernel,sizeKernel,layerKernel);
    for i = 1:layerKernel
        Kernel(:,:,i)=2*rand(sizeKernel)-ones(sizeKernel);
    end
end

