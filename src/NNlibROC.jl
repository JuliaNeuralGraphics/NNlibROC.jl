module NNlibROC

using Adapt
using Adapt: WrappedArray
using AMDGPU
using AMDGPU.MIOpen
using NNlib
using NNlib: BatchedAdjoint, BatchedTranspose, BatchedAdjOrTrans
using NNlib: DenseConvDims

const ROCBatchedAdjoint{T} = BatchedAdjoint{T, <: ROCArray{T}}
const ROCBatchedTranspose{T} = BatchedTranspose{T, <: ROCArray{T}}
const ROCBatchedAdjOrTrans{T} = Union{
    ROCBatchedAdjoint{T}, ROCBatchedTranspose{T}}
const WrappedROCBatchedAdjOrTrans{T, N} = WrappedArray{
    T, N, ROCBatchedAdjOrTrans{T}, ROCBatchedAdjOrTrans{T}}
const AnyROCBatchedAdjOrTrans = Union{
    ROCBatchedAdjOrTrans, WrappedROCBatchedAdjOrTrans}

function Base.convert(::Type{T}, b::AnyROCBatchedAdjOrTrans) where {T <: Array}
    Base.convert(T, adapt(Array, b))
end

function Base.Array{T, N}(b::AnyROCBatchedAdjOrTrans) where {T, N}
    Array{T, N}(adapt(Array, b))
end

Base.collect(b::AnyROCBatchedAdjOrTrans) = collect(adapt(Array, b))

# FIXME does not preserve ROCArray type
function Base.show(
    io::IO, mime::MIME{Symbol("text/plain")}, x::AnyROCBatchedAdjOrTrans,
)
    show(io, mime, adapt(Array, x))
end

Base.show(io::IO, x::AnyROCBatchedAdjOrTrans) = show(io, adapt(Array, x))

Base.display(x::AnyROCBatchedAdjOrTrans) = display(adapt(Array, x))

function NNlib._batched_gemm!(
    ::Type{<: ROCArray}, transA::Char, transB::Char, α, A, B, β, C,
)
    AMDGPU.rocBLAS.gemm_batched!(transA, transB, α, A, B, β, C)
end

include("conv.jl")

end
