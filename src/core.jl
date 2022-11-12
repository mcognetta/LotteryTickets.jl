
struct MaskedMatrix{T<:Number, W<:AbstractMatrix{T}, M<:AbstractMatrix{Bool}} <: AbstractMatrix{T}
    w::W
    mask::M
end
Flux.@functor MaskedMatrix


function MaskedMatrix(x::W) where W
    mask = similar(x, Bool) .= true
    MaskedMatrix{eltype(x), W, typeof(mask)}(x,mask)
end

# AbstractMatrix Interface
Base.size(m::MaskedMatrix) = size(m.w)
Base.getindex(m::MaskedMatrix, i::Int, j::Int) = getindex(m.w, i, j) * getindex(m.mask, i, j)
Base.setindex!(m::MaskedMatrix, v, i::Int, j::Int) = setindex!(m.w, v * getindex(m.mask, i, j), i, j)

for op in (:+, :-, :*, :(==))
    @eval begin
        ($op)(m::MaskedMatrix, x::AbstractVecOrMat) = ($op)(m.w .* m.mask, x)
        ($op)(x::AbstractVecOrMat, m::MaskedMatrix) = ($op)(x, m.w .* m.mask)
    end
end

for op in (:+, :-)
    @eval ($op)(m::MaskedMatrix, n::MaskedMatrix) = MaskedMatrix($op(m.w .* m.mask, n.w .* n.mask), _intersect(m.mask, n.mask))
end
Base.:*(n::Number, m::MaskedMatrix) = MaskedMatrix((m.w * n) .* m.mask, m.mask)
Base.:*(m::MaskedMatrix, n::Number) = MaskedMatrix((m.w * n) .* m.mask, m.mask)

Base.:*(m::MaskedMatrix, x::AbstractMatrix) = (m.w .* m.mask) * x
Base.:*(m::MaskedMatrix, x::AbstractVector) = (m.w .* m.mask) * x

# Base.:*(n::Adjoint{T, MaskedMatrix{T}}, x::AbstractVecOrMat) where T = (m = n.parent; (m.mask .* m.w)' * x)

# Base.:*(n::Adjoint{<:T, <:MaskedMatrix{T, S, W} where {S, W<:AbstractMatrix{T}}}, x::AbstractVector{S}) where {T, T, S}) = (m = n.parent; (m.mask .* m.w)' * x)
Base.:+(m::MaskedMatrix) = +(m.w .* m.mask)
Base.:-(m::MaskedMatrix) = (-m.w .* m.mask) # invert weights first to avoid -0.0

Base.copy(m::MaskedMatrix) = MaskedMatrix(copy(m.w), copy(m.mask))

Base.vec(m::MaskedMatrix) = Base.vec(m.w .* m.mask)

Base.similar(m::MaskedMatrix) = MaskedMatrix(similar(m.w), similar(m.mask) .= true)
Base.similar(m::MaskedMatrix, ::Type{T}) where T = MaskedMatrix(similar(m.w, T), similar(m.mask) .= true)

function ArrayInterfaceCore.restructure(m::MaskedMatrix, y)
    out = similar(m, eltype(y))
    vec(out.w) .= vec(y)
    out
end

# other
Base.Matrix(m::MaskedMatrix) = m.w .* m.mask
sparsify(m::MaskedMatrix) = SparseMatrixCSC(Base.Matrix(m))

# prune the bottom k

prune!(x, p::Float64, zerooutweights::Bool) = nothing
function prune!(m::MaskedMatrix, p::Float64, zerooutweights::Bool=false)
    # @assert 0 <= p < 1, "p must be in [0, 1)"
    indices = findall(view(m.mask, :))
    k = round(Int, p * length(indices))
    before = sum(m.mask)
    # println("K = $k")
    # bot_k = partialsortperm(view(m.w, :)[indices], 1:k)
    bot_k = partialsortperm(abs.(reshape(m.w[indices], :)), 1:k)
    view(m.mask, :)[indices[bot_k]] .= zero(Bool)
    zerooutweights || (m.w .= (m.w .* m.mask))
    after = sum(m.mask)
    println("K = $k, before/after $before -> $after")
    m
end

adjoint(m::MaskedMatrix) = MaskedMatrix(adjoint(m.w), adjoint(m.mask))
transpose(m::MaskedMatrix) = MaskedMatrix(transpose(m.w), transpose(m.mask))

# rewind!
rewind!(x, rng) = nothing
rewind(x, rng) = nothing
function rewind!(m::MaskedMatrix, rng = rand)
    m.w .= rng(size(m)...)
    m.w .= (m.w .* m.mask)
    m
end
rewind(m::MaskedMatrix, rng = rand) = rewind!(copy(m), rng)

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
