using AMDGPU
using NNlib
using NNlib: batched_adjoint, batched_mul, batched_mul!, batched_transpose
using NNlib: is_strided, storage_type
using NNlibROC
using LinearAlgebra
using Test

AMDGPU.allowscalar(false)

@testset "NNlibROC" begin
    include("storage_type.jl")
    include("batched_repr.jl")
    include("batched_mul.jl")
    include("conv.jl")
end
