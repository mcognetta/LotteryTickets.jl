# TODO:
#    - add matching Flux constructors
#    - correct interface for non-prunable layers
#    - add typing information to mask / orig weights
#    - add disk-stored layer origins

# maybe make these error?
sparsify(x) = x
checkpoint!(x) = x
rewind!(x) = x
prunableweights(x) = nothing
prunableweightmasks(x) = nothing
prunableweightorigins(x) = nothing

abstract type AbstractPrunableLayer end

#####################################
#
# Dense Layer
#
#####################################

struct PrunableDense <: AbstractPrunableLayer
    d::Dense
    orig::Any
    mask::Any
end

Flux.@functor PrunableDense (d,)

PrunableDense(w::AbstractMatrix, b, σ) = PrunableDense(Dense(w, b, σ))

function PrunableDense(
    (in, out)::Pair{<:Integer,<:Integer},
    σ = identity;
    init = Flux.glorot_uniform,
    bias = true,
)
    PrunableDense(Dense(init(out, in), bias, σ))
end

function sparsify(d::PrunableDense)
    Dense(sparse(d.weight), d.bias, d.σ)
end

prunableweights(f::PrunableDense) = (f.d.weight,)
prunableweightmasks(f::PrunableDense) = (f.mask,)
prunableweightorigins(f::PrunableDense) = (f.orig,)

checkpoint!(f::PrunableDense) = (f.orig .= f.d.weight; f)
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

# RNN

struct PrunableRNNCell <: AbstractPrunableRecurrentCell
    cell::Any
    mask_h::Any
    mask_i::Any
    orig_h::Any
    orig_i::Any
end

PrunableRNNCell(
    (in, out)::Pair,
    σ = tanh;
    init = Flux.glorot_uniform,
    initb = zeros32,
    init_state = zeros32,
) = PrunableRNNCell(
    Flux.RNNCell(in => out, σ; init = init, initb = initb, init_state = init_state),
)

Flux.@functor PrunableRNNCell

prunableweights(f::PrunableRNNCell) = (f.cell.Wh, f.cell.Wi)
prunableweightmasks(f::PrunableRNNCell) = (f.mask_h, f.mask_i)
prunableweightorigins(f::PrunableRNNCell) = (f.orig_h, f.orig_i)

function checkpoint!(c::PrunableRNNCell)
    c.orig_h .= c.cell.Wh
    c.orig_i .= c.cell.Wi
    c
end

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

# LSTM

struct PrunableLSTMCell <: AbstractPrunableRecurrentCell
    cell::Any
    mask_h::Any
    mask_i::Any
    orig_h::Any
    orig_i::Any
end

PrunableLSTMCell(
    (in, out)::Pair;
    init = Flux.glorot_uniform,
    initb = zeros32,
    init_state = zeros32,
) = PrunableLSTMCell(
    Flux.LSTMCell(in => out; init = init, initb = initb, init_state = init_state),
)

Flux.@functor PrunableLSTMCell

prunableweights(f::PrunableLSTMCell) = (f.cell.Wh, f.cell.Wi)
prunableweightmasks(f::PrunableLSTMCell) = (f.mask_h, f.mask_i)
prunableweightorigins(f::PrunableLSTMCell) = (f.orig_h, f.orig_i)

function checkpoint!(c::PrunableLSTMCell)
    c.orig_h .= c.cell.Wh
    c.orig_i .= c.cell.Wi
    c
end

function rewind!(c::PrunableLSTMCell)
    c.cell.Wh .= c.orig_h
    c.cell.Wi .= c.orig_i
    c
end

function (f::PrunableLSTMCell)(x, y)
    f.cell.Wh .*= f.mask_h
    f.cell.Wi .*= f.mask_i
    f.cell(x, y)
end

function PrunableLSTMCell(c::Flux.LSTMCell)
    orig_h = deepcopy(c.Wh)
    orig_i = deepcopy(c.Wi)
    mask_h = similar(orig_h, Bool) .= true
    mask_i = similar(orig_i, Bool) .= true
    PrunableLSTMCell(c, mask_h, mask_i, orig_h, orig_i)
end

Zygote.@adjoint function (f::PrunableLSTMCell)(x)

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

# GRU

struct PrunableGRUCell <: AbstractPrunableRecurrentCell
    cell::Any
    mask_h::Any
    mask_i::Any
    orig_h::Any
    orig_i::Any
end

PrunableGRUCell(
    (in, out)::Pair;
    init = Flux.glorot_uniform,
    initb = zeros32,
    init_state = zeros32,
) = PrunableGRUCell(
    Flux.GRUCell(in => out);
    init = init,
    initb = initb,
    init_state = init_state,
)

Flux.@functor PrunableGRUCell

prunableweights(f::PrunableGRUCell) = (f.cell.Wh, f.cell.Wi)
prunableweightmasks(f::PrunableGRUCell) = (f.mask_h, f.mask_i)
prunableweightorigins(f::PrunableGRUCell) = (f.orig_h, f.orig_i)

function checkpoint!(c::PrunableGRUCell)
    c.orig_h .= c.cell.Wh
    c.orig_i .= c.cell.Wi
    c
end

function rewind!(c::PrunableGRUCell)
    c.cell.Wh .= c.orig_h
    c.cell.Wi .= c.orig_i
    c
end

function (f::PrunableGRUCell)(x, y)
    f.cell.Wh .*= f.mask_h
    f.cell.Wi .*= f.mask_i
    f.cell(x, y)
end

function PrunableGRUCell(c::Flux.GRUCell)
    orig_h = deepcopy(c.Wh)
    orig_i = deepcopy(c.Wi)
    mask_h = similar(orig_h, Bool) .= true
    mask_i = similar(orig_i, Bool) .= true
    PrunableGRUCell(c, mask_h, mask_i, orig_h, orig_i)
end

Zygote.@adjoint function (f::PrunableGRUCell)(x)

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

# GRUv3

struct PrunableGRUv3Cell <: AbstractPrunableRecurrentCell
    cell::Any
    mask_h::Any
    mask_i::Any
    mask_hh::Any
    orig_h::Any
    orig_i::Any
    orig_hh::Any
end

PrunableGRUv3Cell(
    (in, out)::Pair;
    init = Flux.glorot_uniform,
    initb = zeros32,
    init_state = zeros32,
) = PrunableGRUv3Cell(
    Flux.GRUv3Cell(in => out; init = init, initb = initb, init_state = init_state),
)

Flux.@functor PrunableGRUv3Cell

prunableweights(f::PrunableGRUv3Cell) = (f.cell.Wh, f.cell.Wi)
prunableweightmasks(f::PrunableGRUv3Cell) = (f.mask_h, f.mask_i)
prunableweightorigins(f::PrunableGRUv3Cell) = (f.orig_h, f.orig_i)

function checkpoint!(c::PrunableGRUv3Cell)
    c.orig_h .= c.cell.Wh
    c.orig_i .= c.cell.Wi
    c.orig_hh .= c.cell.Wh_h̃
    c
end

function rewind!(c::PrunableGRUv3Cell)
    c.cell.Wh .= c.orig_h
    c.cell.Wi .= c.orig_i
    c.cell.Wh_h̃ .= c.orig_hh
    c
end

function (f::PrunableGRUv3Cell)(x, y)
    f.cell.Wh .*= f.mask_h
    f.cell.Wi .*= f.mask_i
    f.cell.Wh_h̃ .*= f.mask_hh
    f.cell(x, y)
end

function PrunableGRUCell(c::Flux.GRUv3Cell)
    orig_h = deepcopy(c.Wh)
    orig_i = deepcopy(c.Wi)
    orig_hh = deepcopy(c.Wh_h̃)
    mask_h = similar(orig_h, Bool) .= true
    mask_i = similar(orig_i, Bool) .= true
    mask_hh = similar(orig_hh, Bool) .= true
    PrunableGRUCell(c, mask_h, mask_i, mask_hh, orig_h, orig_i, orig_hh)
end

Zygote.@adjoint function (f::PrunableGRUv3Cell)(x)

    pb = Zygote._pullback(f.r, x)[2]

    function inner_pb(y)
        grad, val = pb(y)
        newgrad = (;
            d = merge(
                grad,
                (;
                    Wi = cell.Wi .* f.mask_i,
                    Wh = cell.Wh .* f.mask_h,
                    Wh_h̃ = cell.Wh_h̃ .* f.mask_hh,
                ),
            ),
            orig = nothing,
            mask = nothing,
        )
        return newgrad, val
    end
    return f(x), inner_pb
end

# Recur wrappers

for (name, cell) in (
    (:PrunableRNN, :PrunableRNNCell),
    (:PrunableLSTM, :PrunableLSTMCell),
    (:PrunableGRU, :PrunableGRUCell),
    (:PrunableGRUv3, :PrunableGRUv3Cell),
)
    @eval $name(a...; ka...) = Flux.Recur($cell(a...; ka...))
    @eval Flux.Recur(r::$cell) = Flux.Recur(r, r.cell.state0)
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
