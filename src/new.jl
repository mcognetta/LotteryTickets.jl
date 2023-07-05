

struct FakeDense
    d::Dense
    orig
    mask
end

(f::FakeDense)(x) = (f.d.weight .*= f.mask; f.d(x))

function FakeDense(d::Dense)
    orig = deepcopy(d.weight)
    mask = similar(orig, Bool) .= true
    FakeDense(d, orig, mask)
end

Zygote.@adjoint function (f::FakeDense)(x)
    f.d.weight .*= f.mask
    # println("YO ", x)
    # println(f)
    out, pb = Zygote._pullback(f.d, x)
    # println("OUT ", out)
    # (f(x), ŷ -> (println("PB ", pb(ŷ)[1][1]); (pb(ŷ)[1][1] .* f.mask, pb(ŷ .* f.mask)[2])))
    f(x), ŷ -> (pb(ŷ)[1][1] .* f.mask, pb(ŷ .* f.mask)[2])
end

struct FakeConv2
    c
    orig
    mask
end

function FakeConv2(c)
    orig = deepcopy(c.weight)
    mask = similar(orig, Bool) .= true
    return FakeConv2(c, orig, mask)
end

(f::FakeConv2)(x) = (f.c.weight .*= f.mask; f.c(x))

Flux.@functor FakeConv2

Flux.trainable(c::FakeConv2) = Flux.trainable(c.c)


julia> Zygote.@adjoint function (f::FakeConv2)(x)
    f.c.weight .*= f.mask
    # println("YO ", x)
    # println(f)
    out, pb = Zygote._pullback(f.c, x)
    # println("OUT ", out)
    # println("f(x)", f(x))
    # (f(x), ŷ -> (println(ŷ); println(size(ŷ)); println("YHAT ", size(pb(ŷ)[1][2])); (pb(ŷ)[1][2] .* f.mask, pb(ŷ)[2])))
    f(x), ŷ -> (pb(ŷ)[1][2] .* f.mask, pb(ŷ)[2])
end

"""

julia> Zygote.@adjoint function (f::FakeConv2)(x)
    f.c.weight .*= f.mask
    println("YO ", x)
    # println(f)
    out, pb = Zygote._pullback(f.c, x)
    println("OUT ", out)
    println("f(x)", f(x))
    (f(x), ŷ -> (println(ŷ); println(size(ŷ)); println("YHAT ", size(pb(ŷ)[1][2])); (pb(ŷ)[1][2] .* f.mask, pb(ŷ)[2])))
end
"""