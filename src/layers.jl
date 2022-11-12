# wrappers for some common Flux layers

Flux.create_bias(m::MaskedMatrix, bias::Bool, dims::Integer...) = Flux.create_bias(m.w, bias, dims...)

function MaskedDense(d::Pair{<:Integer, <:Integer}, σ=identity; bias=true, init=Flux.glorot_uniform)
    temp = Dense(d, σ; bias=bias, init=init)
    Dense(MaskedMatrix(temp.weight), bias, temp.σ)
end

# Flux.trainable(m::MaskedMatrix) = m
Flux.params(m::MaskedMatrix) = m

Flux.adapt_storage(to::Flux.FluxCUDAAdaptor, x::MaskedMatrix) = MaskedMatrix(CUDA.cu(x.w), CUDA.cu(x.mask))

Adapt.adapt_structure(to, m::MaskedMatrix) = MaskedMatrix(adapt(to, m.w), adapt(to, m.mask))
# Flux.trainable(m::MaskedMatrix) = (m.w,)
Flux.Functors.functor(m::MaskedMatrix) = (), _->m