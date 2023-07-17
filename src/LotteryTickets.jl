module LotteryTickets

using Flux, Zygote, CUDA
using Flux: @functor
using Zygote: @adjoint
using CUDA: CuArray
using SparseArrays

export AbstractPrunableLayer,
    PrunableDense,
    PrunableRNNCell,
    PrunableLSTMCell,
    PrunableGRUCell,
    PrunableGRUv3Cell,
    PrunableRNN,
    PrunableLSTM,
    PrunableGRU,
    PrunableGRUv3,
    PrunableConv,
    PrunableMultiHeadAttention
export sparsify,
    checkpoint!,
    rewind!,
    applymask!,
    prunableweights,
    prunableweightmasks,
    prunableweightorigins
export AbstractPruneGroup, Pruner, MagnitudePruneGroup, prune!, pruneandrewind!, rewind!

include("layers.jl")
include("pruner.jl")

end
