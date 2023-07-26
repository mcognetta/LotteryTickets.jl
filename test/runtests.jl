using LotteryTickets, SparseArrays, Flux, Optimisers
using Test

_invertmask(mask) = mask .== false

@testset "Dense Gradient" begin

    s1 = sprand(Float32, 20, 20, 0.5)
    s2 = sprand(Float32, 5, 20, 0.5)

    mask1 = Matrix(s1 .!= 0.0)
    mask2 = Matrix(s2 .!= 0.0)

    s_d1 = Dense(s1, zeros(Float32, 20), relu)
    s_d2 = Dense(s2, zeros(Float32, 5), relu)

    d_d1 = Dense(Matrix(s1), zeros(Float32, 20), relu)
    d_d2 = Dense(Matrix(s2), zeros(Float32, 5), relu)
    d_d1.bias .= s_d1.bias
    d_d2.bias .= s_d2.bias

    p1 = PrunableDense(d_d1)
    p2 = PrunableDense(d_d2)

    p1.mask .= mask1
    p2.mask .= mask2

    s_chain = Chain(s_d1, s_d2)
    p_chain = Chain(p1, p2)

    s_chain_orig = deepcopy(s_chain)

    data = [(rand(Float32, 20), rand(Float32, 5)) for _ = 1:10]

    loss(m, x, y) = Flux.mse(m(x), y)

    Optimisers.maywrite(::SparseMatrixCSC{Float32,Int64}) = true

    opt_state_s = Optimisers.setup(Flux.Descent(), s_chain)
    opt_state_p = Optimisers.setup(Flux.Descent(), p_chain)

    Flux.train!(loss, s_chain, data, opt_state_s)
    Flux.train!(loss, p_chain, data, opt_state_p)

    @test p1.d.weight ≈ s_d1.weight
    @test p1.d.bias ≈ s_d1.bias
    @test p2.d.weight ≈ s_d2.weight
    @test p2.d.bias ≈ s_d2.bias

    @test all(iszero, p1.d.weight[findall(_invertmask(mask1))])
    @test all(iszero, p2.d.weight[findall(_invertmask(mask2))])

    @test p1.d.weight ≉ p1.orig
    @test p2.d.weight ≉ p2.orig

    @test p1.d.weight ≉ s_chain_orig[1].weight
    @test p1.d.bias ≉ s_chain_orig[1].bias
    @test p2.d.weight ≉ s_chain_orig[2].weight
    @test p2.d.bias ≉ s_chain_orig[2].bias
end

@testset "Pruner" begin
    # randomly shuffled ints from -20 to 19 (inclusive)
    weights = Float32[
        12,
        2,
        -11,
        3,
        5,
        13,
        -4,
        8,
        -20,
        11,
        15,
        -16,
        6,
        0,
        -17,
        4,
        -14,
        -13,
        7,
        -3,
        18,
        -18,
        16,
        -9,
        9,
        -15,
        -12,
        -7,
        10,
        -19,
        -10,
        -5,
        14,
        -8,
        -2,
        19,
        -1,
        17,
        1,
        -6,
    ]

    # split into two rectangular matrices
    d1 = reshape(weights[1:30], 5, 6)
    d2 = reshape(weights[31:end], 2, 5)

    D1 = PrunableDense(d1)
    D2 = PrunableDense(d2)

    @test all(D1.mask)
    @test all(D2.mask)

    @assert D1.orig == Float32[
        12.0 13.0 15.0 4.0 18.0 -15.0
        2.0 -4.0 -16.0 -14.0 -18.0 -12.0
        -11.0 8.0 6.0 -13.0 16.0 -7.0
        3.0 -20.0 0.0 7.0 -9.0 10.0
        5.0 11.0 -17.0 -3.0 9.0 -19.0
    ]
    @assert D2.orig == Float32[-10.0 14.0 -2.0 -1.0 1.0; -5.0 -8.0 19.0 17.0 -6.0]

    # prune the bottom 4 values by magnitude (0.0, 1.0, -1.0, 2.0)
    g = MagnitudePruneGroup([D1, D2], 0.1)
    prune!(g)

    @test !all(D1.mask)
    @test !all(D2.mask)

    @test all(
        x -> x[1] == x[2],
        zip(D1.d.weight[findall(D1.mask)], D1.orig[findall(D1.mask)]),
    )
    @test all(
        x -> x[1] == x[2],
        zip(D2.d.weight[findall(D2.mask)], D2.orig[findall(D2.mask)]),
    )

    # 2.0 and 0.0 are removed
    @assert D1.d.weight == Float32[
        12.0 13.0 15.0 4.0 18.0 -15.0
        0.0 -4.0 -16.0 -14.0 -18.0 -12.0
        -11.0 8.0 6.0 -13.0 16.0 -7.0
        3.0 -20.0 0.0 7.0 -9.0 10.0
        5.0 11.0 -17.0 -3.0 9.0 -19.0
    ]

    # -1.0 and 1.0 are removed
    @assert D2.d.weight == Float32[-10.0 14.0 -2.0 0.0 0.0; -5.0 -8.0 19.0 17.0 -6.0]

    # remove the next 4 (-2.0, 3.0, -3.0, -4.0)
    prune!(g)

    # 3.0, -3.0, -4.0 removed
    @assert D1.d.weight == Float32[
        12.0 13.0 15.0 4.0 18.0 -15.0
        0.0 0.0 -16.0 -14.0 -18.0 -12.0
        -11.0 8.0 6.0 -13.0 16.0 -7.0
        0.0 -20.0 0.0 7.0 -9.0 10.0
        5.0 11.0 -17.0 0.0 9.0 -19.0
    ]

    # -2.0 removed
    @assert D2.d.weight == Float32[-10.0 14.0 0.0 0.0 0.0; -5.0 -8.0 19.0 17.0 -6.0]
end

@testset "applymask and rewind" begin

    s1 = sprand(Float32, 20, 20, 0.5)
    s2 = sprand(Float32, 5, 20, 0.5)

    mask1 = Matrix(s1 .!= 0.0)
    mask2 = Matrix(s2 .!= 0.0)

    d_d1 = Dense(Matrix(s1))
    d_d2 = Dense(Matrix(s2))

    p1 = PrunableDense(d_d1)
    p2 = PrunableDense(d_d2)

    p1.mask .= mask1
    p2.mask .= mask2

    p1.d.weight .= rand(Float32, 20, 20)
    p2.d.weight .= rand(Float32, 5, 20)

    @test p1.d.weight ≉ p1.orig
    @test p2.d.weight ≉ p2.orig

    g = MagnitudePruneGroup([p1, p2], 0.1)

    rewind!(g)

    @test p1.d.weight[findall(p1.mask)] ≈ p1.orig[findall(p1.mask)]
    @test p2.d.weight[findall(p2.mask)] ≈ p2.orig[findall(p2.mask)]

    @test iszero(p1.d.weight[findall(_invertmask(p1.mask))])
    @test iszero(p2.d.weight[findall(_invertmask(p2.mask))])
end
