@testset "Convolutions" begin
    x = rand(Float32, 10, 10, 3, 1)
    w = rand(Float32, 2, 2, 3, 4)
    y = rand(Float32, 9, 9, 4, 1)
    dx, dw, dy = ROCArray.((x, w, y))
    cdims = NNlib.DenseConvDims(x, w, flipkernel=true)

    @test NNlib.conv(x, w, cdims) ≈ Array(NNlib.conv(dx, dw, cdims))
    @test NNlib.∇conv_data(y, w, cdims) ≈ Array(NNlib.∇conv_data(dy, dw, cdims))
    @test NNlib.∇conv_filter(x, y, cdims) ≈ Array(NNlib.∇conv_filter(dx, dy, cdims))
end
