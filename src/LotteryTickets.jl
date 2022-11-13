module LotteryTickets

using Adapt, LinearAlgebra, SparseArrays, CUDA, ChainRulesCore, ChainRules, ArrayInterfaceCore, Flux, OneHotArrays

import Base: copy, size, length, trues, getindex, setindex!, vec, similar
import Base: +, -, *, ==
import Base: show
import Flux: create_bias, @functor, adapt_storage, FluxCUDAAdaptor

export MaskedMatrix, prune!, sparsify, rewind!, rewind
export MaskedDense



# Write your package code here.
include("core.jl")
include("rrules.jl")
include("layers.jl")



end
