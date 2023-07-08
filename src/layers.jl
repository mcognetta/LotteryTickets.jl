# TODO:
#    - add matching Flux constructors
#    - maybe rename "PrunableDense", etc.
#    - correct interface for non-prunable layers
#    - add typing information to mask / orig weights

# maybe make these error?
sparsify(x) = x
checkpoint!(x) = x
rewind!(x) = x

abstract type AbstractPrunableLayer end

#####################################
#
# Dense Layer
#
#####################################

struct PrunableDense <: AbstractPrunableLayer
    d::Dense
    orig
    mask
end

Flux.@functor PrunableDense

PrunableDense(w::AbstractMatrix, b, σ) = PrunableDense(Dense(w, b, σ))

function PrunableDense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
    init = glorot_uniform, bias = true)
    PrunableDense(Dense(init(out, in), bias, σ))
end

function sparsify(d::PrunableDense)
    Dense(sparse(d.weight), d.bias, d.σ)
end

prunableweights(f::PrunableDense) = (f.d.weight,)
prunableweightmasks(f::PrunableDense) = (f.mask,)
prunableweightorigins(f::PrunableDense) = (f.orig,)

checkpoint!(f::PrunableDense) = (f.orig .= f.d.weight; f)
checkpoint(f::PrunableDense) = checkpoint!(PrunableDense(deepcopy(f)))
rewind!(f::PrunableDense) = (f.d.weight .= f.orig; f)

function Base.copyto!(dest::PrunableDense, orig::PrunableDense)
    # # simplify when copyto! is upstreamed to Flux
    # # copyto!(dest.d, orig.d)
    copyto!(dest.d.weight, orig.d.weight)
    copyto!(dest.d.bias, orig.d.bias)

    copyto!(dest.orig, orig.orig)
    copyto!(dest.mask, orig.mask)
    dest
end

function (f::PrunableDense)(x)
    f.d.weight .*= f.mask
    f.d(x)
end

function PrunableDense(d::Dense)
    orig = deepcopy(d.weight)
    mask = similar(orig, Bool) .= true
    PrunableDense(d, orig, mask)
end

# function PrunableDense(x::AbstractMatrix)
#     PrunableDense(Dense(x))
# end

Zygote.@adjoint function (f::PrunableDense)(x)

    pb = Zygote._pullback(f.d, x)[2]

    function inner_pb(y)
        grad, val = pb(y)
        weight = grad.weight
        newgrad = (;
            d = merge(grad, (; weight = weight .* f.mask)),
            orig = nothing,
            mask = nothing,
        )
        return newgrad, val
    end

    return f(x), inner_pb
end

#####################################
#
# Recurrent Layers
#
#####################################

abstract type AbstractPrunableRecurrentCell <: AbstractPrunableLayer end

Flux.reset!(r::Flux.Recur{<:AbstractPrunableRecurrentCell}) = (r.state = r.cell.cell.state0)

struct PrunableRNNCell <: AbstractPrunableRecurrentCell
    cell::Any
    mask_h::Any
    mask_i::Any
    orig_h::Any
    orig_i::Any
end

Flux.@functor PrunableRNNCell

prunableweights(f::PrunableRNNCell) = (f.cell.Wh, f.cell.Wi,)
prunableweightmasks(f::PrunableRNNCell) = (f.mask_h, f.mask_i,)
prunableweightorigins(f::PrunableRNNCell) = (f.orig_h, f.orig_i,)

function checkpoint!(c::PrunableRNNCell)
    c.orig_h .= c.cell.Wh
    c.orig_i .= c.cell.Wi
    c
end
checkpoint(f::PrunableRNNCell) = checkpoint!(PrunableRNNCell(deepcopy(f)))

function rewind!(c::PrunableRNNCell)
    c.cell.Wh .= c.orig_h
    c.cell.Wi .= c.orig_i
    c
end

function (f::PrunableRNNCell)(x, y) 
    f.cell.Wh .*= f.mask_h
    f.cell.Wi .*= f.mask_i
    f.cell(x, y)
end

function PrunableRNNCell(c::Flux.RNNCell)
    orig_h = deepcopy(c.Wh)
    orig_i = deepcopy(c.Wi)
    mask_h = similar(orig_h, Bool) .= true
    mask_i = similar(orig_i, Bool) .= true
    PrunableRNNCell(c, mask_h, mask_i, orig_h, orig_i)
end

FakeRNN(r) = Flux.Recur(PrunableRNNCell(r.cell), deepcopy(r.state))

Zygote.@adjoint function (f::PrunableRNNCell)(x)

    pb = Zygote._pullback(f.r, x)[2]

    function inner_pb(y)
        grad, val = pb(y)
        newgrad = (;
            d = merge(grad, (; Wi = cell.Wi .* f.mask_i, Wh = cell.Wh .* f.mask_h)),
            orig = nothing,
            mask = nothing,
        )
        return newgrad, val
    end
    return f(x), inner_pb
end

#####################################
#
# Convolutional Layers
#
#####################################

struct PrunableConv <: AbstractPrunableLayer
    c::Any
    orig::Any
    mask::Any
end

Flux.@functor PrunableConv

function PrunableConv(c)
    orig = deepcopy(c.weight)
    mask = similar(orig, Bool) .= true
    return PrunableConv(c, orig, mask)
end

prunableweights(f::PrunableConv) = (f.c.weight,)
prunableweightmasks(f::PrunableConv) = (f.mask,)
prunableweightorigins(f::PrunableConv) = (f.orig,)

checkpoint!(c::PrunableConv) = (c.orig .= c.c.weight; c)
checkpoint(c::PrunableConv) = checkpoint!(PrunableConv(deepcopy(c)))

rewind!(c::PrunableConv) = (c.c.weight .= c.orig; c)

(f::PrunableConv)(x) = (f.c.weight .*= f.mask; f.c(x))

Zygote.@adjoint function (f::PrunableConv)(x)

    pb = Zygote._pullback(f.d, x)[2]

    function inner_pb(y)
        grad, val = pb(y)
        weight = grad.weight
        newgrad = (;
            d = merge(grad, (; weight = weight .* f.mask)),
            orig = nothing,
            mask = nothing,
        )
        return newgrad, val
    end

    return f(x), inner_pb
end
