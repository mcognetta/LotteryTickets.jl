
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


Base.copy(m::MaskedMatrix) = MaskedMatrix(copy(m.w), copy(m.mask))
Base.copyto!(m::MaskedMatrix, x::AbstractArray) = (m.w .= x; m)
function Base.copyto!(dest::MaskedMatrix, B::Base.Broadcast.Broadcasted)
    copyto!(dest.w, B)
    dest
end

@inline function Base.copyto!(dest::MaskedMatrix, B::Base.Broadcast.Broadcasted{<:AbstractGPUArrayStyle})
    copyto!(dest.w, B)
    dest
end



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
    # x̄r = ArrayInterface.restructure(x, x̄) # address some cases where Zygote's
                                            # output are not mutable, see #1510
                                            
    # x̄r = copyto!(similar(x̄), x̄)
    x̄r = copy(x̄)
    x.w .= (x.w .* x.mask)
    x.w .-= Flux.Optimise.apply!(opt, x, x̄r.w .* x̄r.mask)
end

# other
Base.Matrix(m::MaskedMatrix) = m.w .* m.mask
sparsify(m::MaskedMatrix) = SparseMatrixCSC(Base.Matrix(m))

# prune the bottom k

# workaround since partialsortperm isn't yet supproted by CUDA.jl
partialsortperm_cuda(c, k) = sortperm(c)[k]

prune!(x, p::Float64, zerooutweights::Bool) = nothing
function prune!(m::MaskedMatrix, p::Float64, zerooutweights::Bool=false)
    # @assert 0 <= p < 1, "p must be in [0, 1)"
    indices = findall(view(m.mask, :))
    k = round(Int, p * length(indices))
    before = sum(m.mask)
    # println("K = $k")
    # bot_k = partialsortperm(view(m.w, :)[indices], 1:k)
    bot_k = partialsortperm_cuda(abs.(reshape(m.w[indices], :)), 1:k)
    view(m.mask, :)[indices[bot_k]] .= zero(Bool)
    zerooutweights || (m.w .= (m.w .* m.mask))
    after = sum(m.mask)
    println("K = $k, before/after $before -> $after")
    m
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
Base.BroadcastStyle(::Type{<:MaskedMatrix}) = Broadcast.ArrayStyle{MaskedMatrix}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedMatrix{T, A}}}, ::Type{T}) where {T, A} =
    similar(MaskedMatrix{T, A}, axes(bc))

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedMatrix{T}}}, ::Type{T}, dims) where {T} = similar(MaskedMatrix{T}, dims)
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedMatrix}}, ::Type{T}, dims) where {T} = similar(MaskedMatrix{T}, dims)
    # MaskedMatrix{T}(undef, dims)

# function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedMatrix}}, ::Type{ElType}) where ElType
#     # Scan the inputs for the ArrayAndChar:
#     M = find_mm(bc)
#     # Use the char field of A to create the output
#     MaskedMatrix(similar(Array{ElType}, axes(bc)), M.mask)
# end

Base.show(io::IO, mime::MIME"text/plain", m::MaskedMatrix) = (println("MASKED MATRIX"); Base.show(io, mime, (m.w .* m.mask)))
Base.show(io::IO, m::MaskedMatrix) = Base.show(io, (m.w .* m.mask) .+ 10)



Adapt.adapt_storage(::Type{<:MaskedMatrix}, xs) = MaskedMatrix(xs)
Adapt.adapt_structure(::Type{<:MaskedMatrix}, xs) = MaskedMatrix(xs)
find_mm(bc::Base.Broadcast.Broadcasted) = find_mm(bc.args)
find_mm(args::Tuple) = find_mm(find_mm(args[1]), Base.tail(args))
find_mm(x) = x
find_mm(::Tuple{}) = nothing
find_mm(a::MaskedMatrix, rest) = a
find_mm(::Any, rest) = find_mm(rest)