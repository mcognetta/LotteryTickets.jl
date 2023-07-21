module LotteryTickets

using Flux, Zygote, CUDA
using Flux: @functor
using Zygote: @adjoint
using CUDA: CuArray
using SparseArrays
using Functors

export AbstractPrunableLayer,
    PrunableDense,
    PrunableBilinear,
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
    sparsify,
    applymask!,
    prunableweights,
    prunableweightmasks,
    prunableweightorigins
export AbstractPruneGroup, Pruner, MagnitudePruneGroup, prune!, pruneandrewind!, rewind!

include("layers.jl")
include("pruner.jl")

end
