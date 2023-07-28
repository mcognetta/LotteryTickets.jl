# TODO:
#    - add disk-stored layer origins
#    - macro
#    - embedding(bag)?

########################################
#
# Abstract Type + Interface
#
########################################

abstract type AbstractPrunableLayer end

_prunable(l) = false
_prunable(::Union{Flux.Recur{<:AbstractPrunableLayer},AbstractPrunableLayer}) = true

sparsify(m) = Functors.fmap(_sparsify, m; exclude = _prunable)
sparsify(l::AbstractPrunableLayer) = _sparsify(l)

checkpoint!(m) = Functors.fmap(_checkpoint!, m; exclude = _prunable)
checkpoint!(l::AbstractPrunableLayer) = _checkpoint!(l)

rewind!(m) = Functors.fmap(_rewind!, m; exclude = _prunable)
rewind!(l::AbstractPrunableLayer) = _rewind!(l)

applymask!(m) = Functors.fmap(_applymask!, m; exclude = _prunable)
applymask!(l::AbstractPrunableLayer) = _applymask!(l)

function _applymask!(l::AbstractPrunableLayer)
    for (w, m) in zip(prunableweights(l), prunableweightmasks(l))
        w .*= m
    end
    l
end

function _rewind!(l::AbstractPrunableLayer)
    for (w, o) in zip(prunableweights(l), prunableweightorigins(l))
        w .= o
    end
    applymask!(l)
    l
end

function _checkpoint!(l::AbstractPrunableLayer)
    for (o, w) in zip(prunableweightorigins(l), prunableweights(l))
        o .= w
    end
    l
end

_sparsify(x) = x
_checkpoint!(x) = x
_rewind!(x) = x

# maybe these should error
prunableweights(x) = nothing
prunableweightmasks(x) = nothing
prunableweightorigins(x) = nothing


#####################################
#
# Dense Layer
#
#####################################

struct PrunableDense{D<:Flux.Dense,O,M} <: AbstractPrunableLayer
    d::D
    orig::O
    mask::M
end

Flux.@functor PrunableDense

Flux.trainable(d::PrunableDense) = (; d = d.d)

PrunableDense(w::AbstractMatrix, bias = true, σ = identity) =
    PrunableDense(Dense(w, bias, σ))

function PrunableDense(
    (in, out)::Pair{<:Integer,<:Integer},
    σ = identity;
    init = Flux.glorot_uniform,
    bias = true,
)
    PrunableDense(Dense(init(out, in), bias, σ))
end

function PrunableDense(d::Dense)
    orig = deepcopy(d.weight)
    mask = similar(orig, Bool) .= true
    PrunableDense(d, orig, mask)
end

prunableweights(f::PrunableDense) = (f.d.weight,)
prunableweightmasks(f::PrunableDense) = (f.mask,)
prunableweightorigins(f::PrunableDense) = (f.orig,)

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
    # f.d.weight .*= f.mask
    f.d(x)
end

Zygote.@adjoint function (f::PrunableDense)(x)

    pb = Zygote._pullback(f.d, x)[2]

    function inner_pb(y)
        grad, val = pb(y)
        weight = grad.weight
        newgrad = (; d = merge(grad, (; weight = weight .* f.mask)))
        return newgrad, val
    end

    return f(x), inner_pb
end

function _sparsify(d::PrunableDense)
    applymask!(d)
    Flux.Dense(sparse(d.d.weight), d.d.bias, d.d.σ)
end

#####################################
#
# Bilinear
#
#####################################

struct PrunableBilinear{B<:Flux.Bilinear,O,M} <: AbstractPrunableLayer
    b::B
    orig::O
    mask::M
end

Flux.@functor PrunableBilinear

Flux.trainable(b::PrunableBilinear) = (; b = b.b)

PrunableBilinear(w::AbstractArray, bias = true, σ = identity) =
    PrunableBilinear(Flux.Bilinear(w, bias, σ))

function PrunableBilinear(
    ((in1, in2), out)::Pair{<:Tuple,<:Integer},
    σ = identity;
    init = Flux.glorot_uniform,
    bias = true,
)
    PrunableBilinear(Flux.Bilinear(init(out, in1, in2), bias, σ))
end

function PrunableBilinear(b::Flux.Bilinear)
    orig = deepcopy(b.weight)
    mask = similar(orig, Bool) .= true
    PrunableBilinear(b, orig, mask)
end

prunableweights(p::PrunableBilinear) = (p.b.weight,)
prunableweightmasks(p::PrunableBilinear) = (p.mask,)
prunableweightorigins(p::PrunableBilinear) = (p.orig,)

function Base.copyto!(dest::PrunableBilinear, orig::PrunableBilinear)
    # # simplify when copyto! is upstreamed to Flux
    # # copyto!(dest.d, orig.d)
    copyto!(dest.b.weight, orig.b.weight)
    copyto!(dest.b.bias, orig.b.bias)

    copyto!(dest.orig, orig.orig)
    copyto!(dest.mask, orig.mask)
    dest
end

function (f::PrunableBilinear)(x)
    # f.b.weight .*= f.mask
    f.b(x)
end

function (f::PrunableBilinear)(x, y)
    # f.b.weight .*= f.mask
    f.b(x, y)
end

Zygote.@adjoint function (f::PrunableBilinear)(x)

    pb = Zygote._pullback(f.b, x)[2]

    function inner_pb(y)
        grad, val = pb(y)
        weight = grad.weight
        newgrad = (; b = merge(grad, (; weight = weight .* f.mask)))
        return newgrad, val
    end

    return f(x), inner_pb
end


Zygote.@adjoint function (f::PrunableBilinear)(x, y)

    pb = Zygote._pullback(f.b, x, y)[2]

    function inner_pb(y)
        grad, val = pb(y)
        weight = grad.weight
        newgrad = (; b = merge(grad, (; weight = weight .* f.mask)))
        return newgrad, val
    end

    return f(x, y), inner_pb
end

# Bilinear is implemented as a 3d tensor, so it cannot be sparsified
function _sparsify(b::PrunableBilinear)
    applymask!(b)
    b.b
end

#####################################
#
# Recurrent Layers
#
#####################################

abstract type AbstractPrunableRecurrentCell <: AbstractPrunableLayer end

Flux.reset!(r::Flux.Recur{<:AbstractPrunableRecurrentCell}) = (r.state = r.cell.cell.state0)

# RNN

struct PrunableRNNCell{C<:Flux.RNNCell,MH,MI,OH,OI} <: AbstractPrunableRecurrentCell
    cell::C
    mask_h::MH
    mask_i::MI
    orig_h::OH
    orig_i::OI
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

function PrunableRNNCell(c::Flux.RNNCell)
    orig_h = deepcopy(c.Wh)
    orig_i = deepcopy(c.Wi)
    mask_h = similar(orig_h, Bool) .= true
    mask_i = similar(orig_i, Bool) .= true
    PrunableRNNCell(c, mask_h, mask_i, orig_h, orig_i)
end

function Base.copyto!(dest::PrunableRNNCell, orig::PrunableRNNCell)
    # # simplify when copyto! is upstreamed to Flux
    # # copyto!(dest.d, orig.d)
    copyto!(dest.cell.Wh, orig.cell.Wh)
    copyto!(dest.cell.Wi, orig.cell.Wi)
    copyto!(dest.cell.b, orig.cell.b)
    copyto!(dest.cell.state0, orig.cell.state0)

    copyto!(dest.orig_h, orig.orig_h)
    copyto!(dest.orig_i, orig.orig_i)
    copyto!(dest.mask_h, orig.mask_h)
    copyto!(dest.mask_i, orig.mask_i)
    dest
end


Flux.@functor PrunableRNNCell
Flux.trainable(r::PrunableRNNCell) = (; cell = r.cell)

prunableweights(f::PrunableRNNCell) = (f.cell.Wh, f.cell.Wi)
prunableweightmasks(f::PrunableRNNCell) = (f.mask_h, f.mask_i)
prunableweightorigins(f::PrunableRNNCell) = (f.orig_h, f.orig_i)

function checkpoint!(c::PrunableRNNCell)
    c.orig_h .= c.cell.Wh
    c.orig_i .= c.cell.Wi
    c
end

function rewind!(c::PrunableRNNCell)
    c.cell.Wh .= c.orig_h .* c.mask_h
    c.cell.Wi .= c.orig_i .* c.mask_i
    c
end

function (f::PrunableRNNCell)(x, y)
    # f.cell.Wh .*= f.mask_h
    # f.cell.Wi .*= f.mask_i
    f.cell(x, y)
end

Zygote.@adjoint function (f::PrunableRNNCell)(h, x)
    out, pb = Zygote._pullback(f.cell, h, x)
    function inner_pb(y)
        grad, val... = pb(y)
        newgrad =
            (; cell = merge(grad, (; Wi = grad.Wi .* f.mask_i, Wh = grad.Wh .* f.mask_h)))
        return newgrad, val...
    end
    return f(h, x), inner_pb
end

function _sparsify(r::PrunableRNNCell)
    applymask!(r)
    Flux.RNNCell(r.cell.σ, sparse(r.cell.Wi), sparse(r.cell.Wh), r.cell.b, r.cell.state0)
end

# LSTM

struct PrunableLSTMCell{C<:Flux.LSTMCell,MH,MI,OH,OI} <: AbstractPrunableRecurrentCell
    cell::C
    mask_h::MH
    mask_i::MI
    orig_h::OH
    orig_i::OI
end

PrunableLSTMCell(
    (in, out)::Pair;
    init = Flux.glorot_uniform,
    initb = zeros32,
    init_state = zeros32,
) = PrunableLSTMCell(
    Flux.LSTMCell(in => out; init = init, initb = initb, init_state = init_state),
)

function PrunableLSTMCell(c::Flux.LSTMCell)
    orig_h = deepcopy(c.Wh)
    orig_i = deepcopy(c.Wi)
    mask_h = similar(orig_h, Bool) .= true
    mask_i = similar(orig_i, Bool) .= true
    PrunableLSTMCell(c, mask_h, mask_i, orig_h, orig_i)
end

function Base.copyto!(dest::PrunableLSTMCell, orig::PrunableLSTMCell)
    # # simplify when copyto! is upstreamed to Flux
    # # copyto!(dest.d, orig.d)
    copyto!(dest.cell.Wh, orig.cell.Wh)
    copyto!(dest.cell.Wi, orig.cell.Wi)
    copyto!(dest.cell.b, orig.cell.b)
    copyto!.(dest.cell.state0, orig.cell.state0)

    copyto!(dest.orig_h, orig.orig_h)
    copyto!(dest.orig_i, orig.orig_i)
    copyto!(dest.mask_h, orig.mask_h)
    copyto!(dest.mask_i, orig.mask_i)
    dest
end

Flux.@functor PrunableLSTMCell
Flux.trainable(r::PrunableLSTMCell) = (; cell = r.cell)

prunableweights(f::PrunableLSTMCell) = (f.cell.Wh, f.cell.Wi)
prunableweightmasks(f::PrunableLSTMCell) = (f.mask_h, f.mask_i)
prunableweightorigins(f::PrunableLSTMCell) = (f.orig_h, f.orig_i)

function (f::PrunableLSTMCell)(x, y)
    # f.cell.Wh .*= f.mask_h
    # f.cell.Wi .*= f.mask_i
    f.cell(x, y)
end

Zygote.@adjoint function (f::PrunableLSTMCell)(h, x)
    pb = Zygote._pullback(f.cell, h, x)[2]

    function inner_pb(y)
        grad, val... = pb(y)
        newgrad =
            (; cell = merge(grad, (; Wi = grad.Wi .* f.mask_i, Wh = grad.Wh .* f.mask_h)))
        return newgrad, val...
    end
    return f(h, x), inner_pb
end

function _sparsify(r::PrunableLSTMCell)
    applymask!(r)
    Flux.LSTMCell(sparse(r.cell.Wi), sparse(r.cell.Wh), r.cell.b, r.cell.state0)
end

# GRU

struct PrunableGRUCell{C<:Flux.GRUCell,MH,MI,OH,OI} <: AbstractPrunableRecurrentCell
    cell::C
    mask_h::MH
    mask_i::MI
    orig_h::OH
    orig_i::OI
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

function PrunableGRUCell(c::Flux.GRUCell)
    orig_h = deepcopy(c.Wh)
    orig_i = deepcopy(c.Wi)
    mask_h = similar(orig_h, Bool) .= true
    mask_i = similar(orig_i, Bool) .= true
    PrunableGRUCell(c, mask_h, mask_i, orig_h, orig_i)
end

function Base.copyto!(dest::PrunableGRUCell, orig::PrunableGRUCell)
    # # simplify when copyto! is upstreamed to Flux
    # # copyto!(dest.d, orig.d)
    copyto!(dest.cell.Wh, orig.cell.Wh)
    copyto!(dest.cell.Wi, orig.cell.Wi)
    copyto!(dest.cell.b, orig.cell.b)
    copyto!(dest.cell.state0, orig.cell.state0)

    copyto!(dest.orig_h, orig.orig_h)
    copyto!(dest.orig_i, orig.orig_i)
    copyto!(dest.mask_h, orig.mask_h)
    copyto!(dest.mask_i, orig.mask_i)
    dest
end

Flux.@functor PrunableGRUCell
Flux.trainable(c::PrunableGRUCell) = (; cell = c.cell)

prunableweights(f::PrunableGRUCell) = (f.cell.Wh, f.cell.Wi)
prunableweightmasks(f::PrunableGRUCell) = (f.mask_h, f.mask_i)
prunableweightorigins(f::PrunableGRUCell) = (f.orig_h, f.orig_i)

function checkpoint!(c::PrunableGRUCell)
    c.orig_h .= c.cell.Wh
    c.orig_i .= c.cell.Wi
    c
end

function rewind!(c::PrunableGRUCell)
    c.cell.Wh .= c.orig_h .*= c.mask_h
    c.cell.Wi .= c.orig_i .*= c.mask_i
    c
end

function (f::PrunableGRUCell)(x, y)
    # f.cell.Wh .*= f.mask_h
    # f.cell.Wi .*= f.mask_i
    f.cell(x, y)
end

Zygote.@adjoint function (f::PrunableGRUCell)(h, x)
    pb = Zygote._pullback(f.cell, h, x)[2]

    function inner_pb(y)
        grad, val... = pb(y)
        newgrad =
            (; cell = merge(grad, (; Wi = grad.Wi .* f.mask_i, Wh = grad.Wh .* f.mask_h)))
        return newgrad, val...
    end
    return f(h, x), inner_pb
end

function _sparsify(r::PrunableGRUCell)
    applymask!(r)
    Flux.GRUCell(sparse(r.cell.Wi), sparse(r.cell.Wh), r.cell.b, r.cell.state0)
end

# GRUv3

struct PrunableGRUv3Cell{C<:Flux.GRUv3Cell,MH,MI,MHH,OH,OI,OHH} <:
       AbstractPrunableRecurrentCell
    cell::C
    mask_h::MH
    mask_i::MI
    mask_hh::MHH
    orig_h::OH
    orig_i::OI
    orig_hh::OHH
end

PrunableGRUv3Cell(
    (in, out)::Pair;
    init = Flux.glorot_uniform,
    initb = zeros32,
    init_state = zeros32,
) = PrunableGRUv3Cell(
    Flux.GRUv3Cell(in => out; init = init, initb = initb, init_state = init_state),
)

function PrunableGRUv3Cell(c::Flux.GRUv3Cell)
    orig_h = deepcopy(c.Wh)
    orig_i = deepcopy(c.Wi)
    orig_hh = deepcopy(c.Wh_h̃)
    mask_h = similar(orig_h, Bool) .= true
    mask_i = similar(orig_i, Bool) .= true
    mask_hh = similar(orig_hh, Bool) .= true
    PrunableGRUv3Cell(c, mask_h, mask_i, mask_hh, orig_h, orig_i, orig_hh)
end

function Base.copyto!(dest::PrunableGRUv3Cell, orig::PrunableGRUv3Cell)
    # # simplify when copyto! is upstreamed to Flux
    # # copyto!(dest.d, orig.d)
    copyto!(dest.cell.Wh, orig.cell.Wh)
    copyto!(dest.cell.Wi, orig.cell.Wi)
    copyto!(dest.cell.Wh_h̃, orig.cell.Wh_h̃)
    copyto!(dest.cell.b, orig.cell.b)
    copyto!(dest.cell.state0, orig.cell.state0)

    copyto!(dest.orig_h, orig.orig_h)
    copyto!(dest.orig_i, orig.orig_i)
    copyto!(dest.orig_hh, orig.orig_hh)
    copyto!(dest.mask_h, orig.mask_h)
    copyto!(dest.mask_i, orig.mask_i)
    copyto!(dest.mask_hh, orig.mask_hh)
    dest
end

Flux.@functor PrunableGRUv3Cell
Flux.trainable(c::PrunableGRUv3Cell) = (; cell = c.cell)

prunableweights(f::PrunableGRUv3Cell) = (f.cell.Wh, f.cell.Wi)
prunableweightmasks(f::PrunableGRUv3Cell) = (f.mask_h, f.mask_i)
prunableweightorigins(f::PrunableGRUv3Cell) = (f.orig_h, f.orig_i)

function (f::PrunableGRUv3Cell)(x, y)
    # f.cell.Wh .*= f.mask_h
    # f.cell.Wi .*= f.mask_i
    # f.cell.Wh_h̃ .*= f.mask_hh
    f.cell(x, y)
end

Zygote.@adjoint function (f::PrunableGRUv3Cell)(h, x)
    pb = Zygote._pullback(f.cell, h, x)[2]

    function inner_pb(y)
        grad, val... = pb(y)
        newgrad = (;
            cell = merge(
                grad,
                (;
                    Wi = grad.Wi .* f.mask_i,
                    Wh = grad.Wh .* f.mask_h,
                    Wh_h̃ = grad.Wh_h̃ .* f.mask_hh,
                ),
            )
        )
        return newgrad, val...
    end
    return f(h, x), inner_pb
end

function _sparsify(r::PrunableGRUv3Cell)
    applymask!(r)
    Flux.GRUv3Cell(
        sparse(r.cell.Wi),
        sparse(r.cell.Wh),
        r.cell.b,
        sparse(r.cell.Wh_h̃)r.cell.state0,
    )
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

    @eval function Base.copyto!(dest::Flux.Recur{<:$cell}, orig::Flux.Recur{<:$cell})
        copyto!(dest.cell, orig.cell)
        if dest.state isa Tuple
            copyto!.(dest.state, orig.state)
        else
            copyto!(dest.state, orig.state)
        end
        dest
    end
end

for func in (
    :applymask!,
    :checkpoint!,
    :rewind!,
    :prunableweights,
    :prunableweightmasks,
    :prunableweightorigins,
)
    @eval $func(r::Flux.Recur{<:AbstractPrunableRecurrentCell}) = $func(r.cell)
end

_sparsify(r::Flux.Recur{<:AbstractPrunableRecurrentCell}) = Flux.Recur(sparsify(r.cell))

#####################################
#
# Convolutional Layers
#
#####################################

struct PrunableConv{C<:Flux.Conv,O,M} <: AbstractPrunableLayer
    c::C
    orig::O
    mask::M
end

Flux.@functor PrunableConv
Flux.trainable(c::PrunableConv) = (; c = c.c)

function PrunableConv(c)
    orig = deepcopy(c.weight)
    mask = similar(orig, Bool) .= true
    return PrunableConv(c, orig, mask)
end

prunableweights(f::PrunableConv) = (f.c.weight,)
prunableweightmasks(f::PrunableConv) = (f.mask,)
prunableweightorigins(f::PrunableConv) = (f.orig,)

function (f::PrunableConv)(x)
    # f.c.weight .*= f.mask
    f.c(x)
end

Zygote.@adjoint function (f::PrunableConv)(x)

    pb = Zygote._pullback(f.d, x)[2]

    function inner_pb(y)
        grad, val = pb(y)
        weight = grad.weight
        newgrad = (; d = merge(grad, (; weight = weight .* f.mask)))
        return newgrad, val
    end

    return f(x), inner_pb
end

function _sparsify(f::PrunableConv)
    applymask!(f)
    Flux.Conv(f.c.σ, f.c.weight, f.c.bias, f.c.stride, f.c.pad, f.c.dilation, f.c.groups)
end

#########################################
#
# Attention
#
#########################################

struct PrunableMultiHeadAttention{M} <: AbstractPrunableLayer
    mha::M

    function PrunableMultiHeadAttention(mha::Flux.MultiHeadAttention)

        replaced = Flux.MultiHeadAttention(
            mha.nheads,
            PrunableDense(mha.q_proj),
            PrunableDense(mha.k_proj),
            PrunableDense(mha.v_proj),
            mha.attn_drop,
            PrunableDense(mha.out_proj),
        )

        return new{typeof(replaced)}(replaced)
    end
end

PrunableMultiHeadAttention(
    dims;
    nheads::Int = 8,
    bias::Bool = false,
    init = Flux.glorot_uniform,
    dropout_prob = 0.0,
) = PrunableMultiHeadAttention(
    Flux.MultiHeadAttention(
        dims;
        nheads = nheads,
        init = init,
        dropout_prob = dropout_prob,
    ),
)

Flux.@functor PrunableMultiHeadAttention
Flux.trainable(m::PrunableMultiHeadAttention) = (; mha = m.mha)

(m::PrunableMultiHeadAttention)(q_in; kws...) = m.mha(q_in; kws...)
(m::PrunableMultiHeadAttention)(q_in, k_in; kws...) = m.mha(q_in, k_in; kws...)
(m::PrunableMultiHeadAttention)(q_in, k_in, z_in, bias = nothing; mask = nothing) =
    m.mha(x, y, z, bias = bias; mask = mask)

for func in (:prunableweights, :prunableweightorigins, :prunableweightmasks)
    @eval $func(m::PrunableMultiHeadAttention) = (
        $func(m.mha.q_proj)...,
        $func(m.mha.k_proj)...,
        $func(m.mha.v_proj)...,
        $func(m.mha.out_proj)...,
    )
end

function _sparsify(m::PrunableMultiHeadAttention)
    Flux.MultiHeadAttention(
        m.mha.nheads,
        sparsify(m.mha.q_proj),
        sparsify(m.mha.k_proj),
        sparsify(m.mha.v_proj),
        m.mha.attn_drop,
        sparsify(m.mha.out_proj),
    )
end

#########################################
#
# MISC
#
#########################################

_prunablecounterpart(l) = l
_prunablecounterpart(l::AbstractPrunableLayer) = l

_convertableorprunable(x) = _convertable(x) || _prunable(x)

_convertable(x) = false

for (flux, lotto) in (
    (Flux.Dense, PrunableDense),
    (Flux.Bilinear, PrunableBilinear),
    (Flux.RNNCell, PrunableRNNCell),
    (Flux.LSTMCell, PrunableLSTMCell),
    (Flux.GRUCell, PrunableGRUCell),
    (Flux.Conv, PrunableConv),
    (Flux.MultiHeadAttention, PrunableMultiHeadAttention),
)
    @eval _convertable(::$flux) = true
    @eval _prunablecounterpart(l::$flux) = $lotto(l)
end

_convertable(l::Flux.Recur) = true
_convertable(l::Flux.Recur{<:AbstractPrunableRecurrentCell}) = false

_prunablecounterpart(l::Flux.Recur) = Flux.Recur(_prunablecounterpart(l.cell))
_prunablecounterpart(l::Flux.Recur{<:AbstractPrunableRecurrentCell}) = l
prunablecounterpart(l::AbstractPrunableLayer) = _prunablecounterpart(l)
prunablecounterpart(l) = Functors.fmap(_prunablecounterpart, l; exclude = _convertableorprunable)

  macro prunable(m)
    :(prunablecounterpart($(esc(m))))
  end