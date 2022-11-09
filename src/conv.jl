const MIOpenFloat = Union{Float16, Float32}

function nnlib_padding(dims)
    pd = NNlib.padding(dims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn """
        MIOpen does not support asymmetric padding, defaulting to symmetric choice:
        $pd -> $(pd[1:2:end]).
        """ maxlog=1
    end
    pd[1:2:end]
end

function NNlib.conv!(
    y::ROCArray{T}, x::ROCArray{T}, w::ROCArray{T}, cdims::DenseConvDims;
    alpha = 1f0, beta = 0f0,
) where T <: MIOpenFloat
    NNlib.flipkernel(cdims) || throw(ArgumentError( # TODO do something about it?
        "MIOpen supports only cross-correlation as its convolution implementation."))
    MIOpen.convolution!(
        y, x, w; padding=nnlib_padding(cdims), stride=NNlib.stride(cdims),
        dilation=NNlib.dilation(cdims), groups=NNlib.groupcount(cdims), alpha, beta)
end

function NNlib.∇conv_data!(
    dx::ROCArray{T}, dy::ROCArray{T}, w::ROCArray{T}, cdims::DenseConvDims;
    alpha = 1f0, beta = 0f0,
) where T <: MIOpenFloat
    NNlib.flipkernel(cdims) || throw(ArgumentError(
        "MIOpen supports only cross-correlation as its convolution implementation."))
    MIOpen.∇convolution_data!(
        dx, dy, w; padding=nnlib_padding(cdims), stride=NNlib.stride(cdims),
        dilation=NNlib.dilation(cdims), groups=NNlib.groupcount(cdims), alpha, beta)
end

function NNlib.∇conv_filter!(
    dw::ROCArray{T}, x::ROCArray{T}, dy::ROCArray{T}, cdims::DenseConvDims;
    alpha = 1f0, beta = 0f0,
) where T <: MIOpenFloat
    NNlib.flipkernel(cdims) || throw(ArgumentError(
        "MIOpen supports only cross-correlation as its convolution implementation."))
    MIOpen.∇convolution_weight!(
        dw, dy, x; padding=nnlib_padding(cdims), stride=NNlib.stride(cdims),
        dilation=NNlib.dilation(cdims), groups=NNlib.groupcount(cdims), alpha, beta)
end
