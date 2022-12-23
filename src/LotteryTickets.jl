module LotteryTickets

using Adapt, LinearAlgebra, SparseArrays, CUDA, ChainRulesCore, ChainRules, ArrayInterface, ArrayInterfaceCore, Flux, OneHotArrays, GPUArraysCore

import Base: copy, copyto!, size, length, trues, getindex, setindex!, vec, similar
import Base: +, -, *, ==
import Base: show
using Base.Broadcast: BroadcastStyle, Broadcasted
import Flux: create_bias, @functor, adapt_storage, FluxCUDAAdaptor

export MaskedMatrix, prune!, sparsify, rewind!, rewind
export MaskedDense



# Write your package code here.
include("core.jl")
include("rrules.jl")
include("layers.jl")



end
