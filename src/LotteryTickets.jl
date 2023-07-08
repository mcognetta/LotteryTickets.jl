module LotteryTickets

# using Adapt, LinearAlgebra, SparseArrays, CUDA, ChainRulesCore, ChainRules, ArrayInterface, ArrayInterfaceCore, Flux, OneHotArrays, GPUArraysCore

# import Base: copy, copyto!, size, length, trues, getindex, setindex!, vec, similar
# import Base: +, -, *, ==
# import Base: show
# using Base.Broadcast: BroadcastStyle, Broadcasted
# import Flux: create_bias, @functor, adapt_storage, FluxCUDAAdaptor

using Flux, Zygote, CUDA
using Flux: @functor
using Zygote: @adjoint
using CUDA: CuArray

export MaskedDense, MaskedConv, MaskedRNNCell, prunableweights, prunableweightmasks, prunableweightorigins
export Pruner, PruneGroup, prune!, pruneandrewind!, rewind!



# Write your package code here.
# include("core.jl")
# include("rrules.jl")
# include("layers.jl")

include("layers.jl")
include("pruner.jl")


end
