
struct MaskedMatrix{T<:Number, N, W<:AbstractArray{T, N}, M<:AbstractArray{Bool, N}} <: AbstractArray{T, N}
    w::W
    mask::M
    orig::W
end
Flux.@functor MaskedMatrix


function MaskedMatrix(x::W) where {T, N, W <: AbstractArray{T, N}}
    mask = similar(x, Bool) .= true
    MaskedMatrix{eltype(x), N, W, typeof(mask)}(x,mask,copy(x))
end

MaskedMatrix(m::MaskedMatrix) = MaskedMatrix(m.w, m.mask, m.orig)

MaskedMatrix(x, m) = MaskedMatrix(x, m, deepcopy(x))
MaskedMatrix{T, V}(x::A) where {T, N, A<:AbstractArray{T, N}, V<:AbstractArray{T, N}} = MaskedMatrix{T, A}(x, similar(x, Bool) .= true)
MaskedMatrix{T, M}(u::UndefInitializer, dims) where {T, M<:CuArray} = MaskedMatrix(CuMatrix{T}(zeros(T, dims...)))
MaskedMatrix{T, M}(u::UndefInitializer, dims) where {T, M<:AbstractArray} = MaskedMatrix(Matrix{T}(zeros(T, dims...)))
# (::Type{M})(u::UndefInitializer, dims) where {T, M<:MaskedMatrix{T}} = (println("other one", M); MaskedMatrix(Matrix{T}(u, dims...)))

# AbstractMatrix Interface
Base.size(m::MaskedMatrix) = size(m.w)
Base.getindex(m::MaskedMatrix, i::Int, j::Int)     = getindex(m.w, i, j) * getindex(m.mask, i, j)
Base.setindex!(m::MaskedMatrix, v, i::Int, j::Int) = setindex!(m.w, v * getindex(m.mask, i, j), i, j)

Base.getindex(m::MaskedMatrix, i1::Int64, i2::Int64, I::Int64...) = getindex(m.w, i1, i2, I...) * getindex(m.mask, i1, i2, I...)

for op in (:+, :-, :*)
    @eval begin
        ($op)(m::MaskedMatrix, x::AbstractMatrix) = ($op)(m.w .* m.mask, x)
        ($op)(m::MaskedMatrix, x::AbstractVector) = ($op)(m.w .* m.mask, x)
        ($op)(x::AbstractMatrix, m::MaskedMatrix) = ($op)(x, m.w .* m.mask)
        ($op)(x::AbstractVector, m::MaskedMatrix) = ($op)(x, m.w .* m.mask)
        ($op)(m::MaskedMatrix, x::OneHotArrays.OneHotMatrix)  = ($op)(m.w .* m.mask, x)
        ($op)(m::MaskedMatrix, x::OneHotArrays.OneHotVector)  = ($op)(m.w .* m.mask, x)
    end
end

Base.:(==)(m::MaskedMatrix, x::AbstractVecOrMat) = (m.w .* m.mask) == x

for op in (:+, :-)
    @eval ($op)(m::MaskedMatrix, n::MaskedMatrix) = MaskedMatrix($op(m.w .* m.mask, n.w .* n.mask), _intersect(m.mask, n.mask))
end
Base.:*(n::Number, m::MaskedMatrix) = MaskedMatrix((m.w * n) .* m.mask, m.mask)
Base.:*(m::MaskedMatrix, n::Number) = MaskedMatrix((m.w * n) .* m.mask, m.mask)

# Base.:*(m::MaskedMatrix, x::AbstractMatrix) =(m.w .* m.mask) * x
# Base.:*(m::MaskedMatrix, x::AbstractVector) = (m.w .* m.mask) * x

# Base.:*(n::Adjoint{T, MaskedMatrix{T}}, x::AbstractVecOrMat) where T = (m = n.parent; (m.mask .* m.w)' * x)

# Base.:*(n::Adjoint{<:T, <:MaskedMatrix{T, S, W} where {S, W<:AbstractMatrix{T}}}, x::AbstractVector{S}) where {T, T, S}) = (m = n.parent; (m.mask .* m.w)' * x)
Base.:+(m::MaskedMatrix) = MaskedMatrix(+m.w, m.mask)
Base.:-(m::MaskedMatrix) = MaskedMatrix(-m.w, m.mask) # invert weights first to avoid -0.0


Base.copy(m::MaskedMatrix) = MaskedMatrix(copy(m.w), copy(m.mask), copy(m.orig))
Base.copyto!(m::MaskedMatrix, x::AbstractArray) = (m.w .= x; m)
Base.copyto!(m::MaskedMatrix, n::MaskedMatrix) = (m.w .= n.w; m.mask .= n.mask; m.orig .= n.orig; m)
# function Base.copyto!(dest::MaskedMatrix, B::Base.Broadcast.Broadcasted)
#     copyto!(dest.w, B)
#     dest
# end

# @inline function Base.copyto!(dest::MaskedMatrix, B::Base.Broadcast.Broadcasted{<:AbstractGPUArrayStyle})
#     copyto!(dest.w, B)
#     dest
# end



Base.vec(m::MaskedMatrix) = Base.vec(m.w)

Base.similar(m::MaskedMatrix) = MaskedMatrix(similar(m.w), similar(m.mask) .= true)
Base.similar(m::MaskedMatrix, ::Type{T}) where T = MaskedMatrix(similar(m.w, T), similar(m.mask) .= true)
Base.similar(::Type{M}, dims::Union{Dims{2}, Tuple{I, I}}) where {T, A<:AbstractArray{T, 2}, M<:MaskedMatrix{T, A}, I<:Integer} = MaskedMatrix(similar(A, dims))

function ArrayInterfaceCore.restructure(m::MaskedMatrix, y)
    out = similar(m, eltype(y))
    
    vec(out.w) .= vec(y)
    out
end

function Flux.update!(opt::Flux.Optimise.AbstractOptimiser, x::MaskedMatrix, x̄::MaskedMatrix)
    x̄r = copy(x̄)
    x.w .= (x.w .* x.mask)
    x.w .-= Flux.Optimise.apply!(opt, x.w, x̄r.w .* x̄r.mask)
end

Flux.Optimisers.maywrite(::MaskedMatrix) = true

# other
Base.Matrix(m::MaskedMatrix) = m.w .* m.mask
Base.convert(::Type{T}, a::AbstractMatrix) where T <: MaskedMatrix = MaskedMatrix(a)

sparsify(m::MaskedMatrix) = SparseMatrixCSC(Base.Matrix(m))

# prune the bottom k

# workaround since partialsortperm isn't yet supproted by CUDA.jl
partialsortperm_cuda(c, k) = sortperm(c)[k]


abstract type AbstractPruner end

struct PruneGroup <: AbstractPruner
    matrices::Array
end

Base.length(g::PruneGroup) = length(g.matrices)

prune!(x, p::Float64, zerooutweights::Bool) = nothing
prune_and_restore!(x, p, zerooutweights) = nothing

# TODO: gpu and cpu dispatch, so we don't use partialsortperm_cuda
function prune!(m::MaskedMatrix, p::Float64, zerooutweights::Bool=false)
    # @assert 0 <= p < 1, "p must be in [0, 1)"
    indices = findall(view(m.mask, :))
    k = round(Int, p * length(indices))
    before = sum(m.mask)
    bot_k = partialsortperm_cuda(abs.(reshape(m.w[indices], :)), 1:k)
    view(m.mask, :)[indices[bot_k]] .= zero(Bool)
    zerooutweights || (m.w .= (m.w .* m.mask))
    after = sum(m.mask)
    m
end

restore!(m::MaskedMatrix) = (m.w .= m.orig; m)

function prune_and_restore!(m::MaskedMatrix, p::Float64, zerooutweights::Bool=false)
    prune!(m, p)
    restore!(m)
    zerooutweights || (m.w .= (m.w .* m.mask))
    m
end



function _gpu_prune!(g::PruneGroup, p::Float64, zerooutweights::Bool = false)

    cpu_group = PruneGroup(cpu.(g.matrices))

    _cpu_prune!(cpu_group, p, zerooutweights)
    for (cm, gm) in zip(cpu_group.matrices, g.matrices)
        copyto!(gm, gpu(cm))
    end
    g
end

function _cpu_prune!(g::PruneGroup, p::Float64, zerooutweights::Bool=false)
    L = 0

    v = Vector{Tuple{eltype(g.matrices[1]), Int, Int}}()

    for (idx, m) in enumerate(g.matrices)
        indices = findall(view(m.mask, :))
        L += length(indices)
        append!(v, collect(zip(m[indices], Iterators.repeated(idx), indices)))
    end

    k = min(L, round(Int, p * L, RoundUp))
    
    indices = sortperm(v, by=x->x[1])[1:k]

    for (_, m_idx, idx) in v[indices]
        zerooutweights && (g.matrices[m_idx].w[idx] = 0.0)
        g.matrices[m_idx].mask[idx] = false
    end
    g
end

function prune!(g::PruneGroup, p::Float64, zerooutweights::Bool=false)
    println("CALLING PRUNE!")
    sizes = sum.(m.mask for m in g.matrices)
    if all(m.w isa CuArray for m in g.matrices)
        _gpu_prune!(g, p, zerooutweights)
    else
        _cpu_prune!(g, p, zerooutweights)
    end
    after = sum.(m.mask for m in g.matrices)
    original = length.(m.w for m in g.matrices)
    for (orig, before, after) in zip(original, sizes, after)
        println("LAYER $orig: $before -> $after")
    end
    g
end

function prune_and_restore!(g::PruneGroup, p::Float64, zerooutweights::Bool=false)
    prune!(g, p, zerooutweights)
    for m in g.matrices
        restore!(m)
    end
    g
end

adjoint(m::MaskedMatrix) = MaskedMatrix(adjoint(m.w), adjoint(m.mask))
transpose(m::MaskedMatrix) = MaskedMatrix(transpose(m.w), transpose(m.mask))

# rewind!
rewind!(x, rng = nothing) = nothing
rewind(x, rng = nothing) = nothing

glor_rand(x::AbstractMatrix) = Flux.glorot_uniform(size(x)...)
glor_rand(x::CUDA.CuArray) = cu(Flux.glorot_uniform(size(x)...))


function rewind!(m::MaskedMatrix, rng = glor_rand)
    m.w .= rng(m.w)
    m.w .= (m.w .* m.mask)
    m
end
rewind(m::MaskedMatrix, rng = glor_rand) = rewind!(copy(m), rng)

function _intersect(a::T, b::T) where T<:AbstractMatrix{Bool}
    T(a .& b)
end

function _union(a::T, b::T) where T<:AbstractMatrix{Bool}
    T(a .| b)
end

function Base.fill!(m::MaskedMatrix, v)
    fill!(m.w, v)
    m
end

Base.broadcast(f::Tf, m::MaskedMatrix, v) where Tf = broadcast(x -> f(x, T), A)
Base.BroadcastStyle(::Type{M}) where {T, A<:AbstractMatrix{T}, M<:MaskedMatrix{T, A}} = Broadcast.ArrayStyle{MaskedMatrix{T,A}}()


Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedMatrix{T, A}}}, ::Type{T}) where {T, A} = similar(MaskedMatrix{T, A}, axes(bc))

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedMatrix{T}}}, ::Type{T}, dims) where {T} = similar(MaskedMatrix{T}, dims)
# Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedMatrix}}, ::Type{T}, dims) where {T} = (println("bc3"); similar(MaskedMatrix{T}, dims))
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{M}}, ::Type{T}, dims) where {T, M<:MaskedMatrix} = similar(M, dims)
    # MaskedMatrix{T}(undef, dims)

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedMatrix{T,A}}}, ::Type{ElType}) where {T, A, ElType}
    # Scan the inputs for the ArrayAndChar:
    M = find_mm(bc)
    # Use the char field of A to create the output
    MaskedMatrix(similar(A{ElType}, axes(bc)), M.mask)
end

Base.show(io::IO, mime::MIME"text/plain", m::MaskedMatrix) = (println("MASKED MATRIX"); Base.show(io, mime, (m.w .* m.mask)))
Base.show(io::IO, m::MaskedMatrix) = Base.show(io, (m.w .* m.mask) .+ 10)

# Adapt.adapt_storage(::Type{<:MaskedMatrix}, xs) = MaskedMatrix(xs)
# Adapt.adapt_structure(::Type{<:MaskedMatrix}, xs) = MaskedMatrix(xs)
# Adapt.adapt_structure(to, x::M) where {T, M<:MaskedMatrix{T}} = MaskedMatrix{T}(adapt(to, x.w))

find_mm(bc::Base.Broadcast.Broadcasted) = find_mm(bc.args)
find_mm(args::Tuple) = find_mm(find_mm(args[1]), Base.tail(args))
find_mm(x) = x
find_mm(::Tuple{}) = nothing
find_mm(a::MaskedMatrix, rest) = a
find_mm(::Any, rest) = find_mm(rest)