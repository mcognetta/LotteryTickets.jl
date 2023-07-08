

# struct FakeDense
#     d::Dense
#     orig
#     mask
# end

# # @adjoint FakeDense(d) = FakeDense(d), p -> (p.d, p.orig, p.mask)

# function (f::FakeDense)(x)
#     f.d.weight .*= f.mask
#     f.d(x)
#     # return getbase(f)(x)
# end

# # getbase(f::FakeDense) = f.d

# # Zygote.@adjoint getbase(f::FakeDense) = (println("THIS ONE"); (f.d, x ->(FakeDense(x),)))

# function FakeDense(d::Dense)
#     orig = deepcopy(d.weight)
#     mask = similar(orig, Bool) .= true
#     FakeDense(d, orig, mask)
# end

# function FakeDense(x::AbstractMatrix)
#     FakeDense(Dense(x))
# end

# function Adapt.adapt_structure(to, itp::FakeDense)
#     println("YO")
#     d = Adapt.adapt_structure(to, Adapt.adapt_structure(to, itp.d))
#     orig = Adapt.adapt_structure(to, itp.orig)
#     mask = Adapt.adapt_structure(to, itp.mask)
#     FakeDense(d, orig, mask)
# end

# prunableweights(f::FakeDense) = (f.d.weight,)

# Flux.@functor FakeDense

# # Flux.trainable(f::FakeDense) = (; d = f.d)

# Zygote.@adjoint function (f::FakeDense)(x)

#     pb = Zygote._pullback(f.d, x)[2]

#     function inner_pb(y)
#         grad, val = pb(y)
#         @show(grad)
#         display(grad.weight)
        
#         display(grad.weight .* f.mask)
#         @show(val)
#         weight = grad.weight
#         newgrad = (; d = merge(grad, (; weight = weight .* f.mask)), orig = nothing, mask = nothing)
#         return newgrad, val
#     end

#     return f(x), inner_pb
# end


# struct FakeRNNCell
#     cell
#     mask_h
#     mask_i
#     orig_h
#     orig_i
# end

# # (f::FakeRNNCell)(x) = f.cell(x)
# (f::FakeRNNCell)(x, y) = (f.cell.Wh .*= f.mask_h; f.cell.Wi .*= f.mask_i; f.cell(x, y))

# prunableweights(f::FakeDense) = (f.d.weight,)

# function FakeRNNCell(c)
#     orig_h = deepcopy(c.Wh)
#     orig_i = deepcopy(c.Wi)
#     mask_h = similar(orig_h, Bool) .= true
#     mask_i = similar(orig_i, Bool) .= true
#     FakeRNNCell(c, mask_h, mask_i, orig_h, orig_i)
# end

# Flux.@functor FakeRNNCell


# Flux.reset!(c::FakeRNNCell) = Flux.reset!(c.cell)

# Flux.reset!(r::Flux.Recur{<:FakeRNNCell}) = (r.state = r.cell.cell.state0)

# FakeRNN(r) = Flux.Recur(FakeRNNCell(r.cell), deepcopy(r.state))

# # struct FakeRNN{R<:Flux.Recur{<:Flux.RNNCell}}
# #     r::R
# #     mask_h
# #     mask_i
# #     orig_h
# #     orig_i
# # end

# # Flux.reset!(r::FakeRNN) = Flux.reset!(r.r)

# # function FakeRNN(r::R) where R<:Flux.Recur{<:Flux.RNNCell}
# #     orig_h = deepcopy(r.cell.Wh)
# #     orig_i = deepcopy(r.cell.Wi)
# #     mask_h = similar(orig_h, Bool) .= true
# #     mask_i = similar(orig_i, Bool) .= true
# #     FakeRNN(r, mask_h, mask_i, orig_h, orig_i)
# # end

# # Flux.@functor FakeRNN

# # function (f::FakeRNN)(x)
# #     println("YOYOYOYOYO")
# #     f.r.cell.Wh .*= f.mask_h
# #     f.r.cell.Wi .*= f.mask_i
# #     f.r(x)
# # end

# Zygote.@adjoint function (f::FakeRNNCell)(x)

#     pb = Zygote._pullback(f.r, x)[2]

#     function inner_pb(y)
#         grad, val = pb(y)
#         newgrad = (; d = merge(grad, (; Wi = cell.Wi .* f.mask_i, Wh = cell.Wh .* f.mask_h)), orig = nothing, mask = nothing)
#         return newgrad, val
#     end
#     return f(x), inner_pb
# end




# struct FakeConv
#     c
#     orig
#     mask
# end

# function FakeConv(c)
#     orig = deepcopy(c.weight)
#     mask = similar(orig, Bool) .= true
#     return FakeConv(c, orig, mask)
# end

# (f::FakeConv)(x) = (f.c.weight .*= f.mask; f.c(x))

# Flux.@functor FakeConv

# Flux.trainable(c::FakeConv) = Flux.trainable(c.c)


# Zygote.@adjoint function (f::FakeConv)(x)
#     f.c.weight .*= f.mask
#     # println("YO ", x)
#     # println(f)
#     out, pb = Zygote._pullback(f.c, x)
#     # println("OUT ", out)
#     # println("f(x)", f(x))
#     # (f(x), ŷ -> (println(ŷ); println(size(ŷ)); println("YHAT ", size(pb(ŷ)[1][2])); (pb(ŷ)[1][2] .* f.mask, pb(ŷ)[2])))
#     out, ŷ -> (pb(ŷ)[1][2] .* f.mask, pb(ŷ)[2])
# end