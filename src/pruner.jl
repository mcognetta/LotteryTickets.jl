abstract type AbstractPruneGroup end

pruneandrewind!(g::AbstractPruneGroup) = (prune!(g); rewind!(g); g)

######################################
#
# Magnitude Prune Group
#
#####################################

mutable struct MagnitudePruneGroup <: AbstractPruneGroup
    layers::Vector{Union{AbstractPrunableLayer,Flux.Recur{<:AbstractPrunableRecurrentCell}}}
    p::Float64
end

updatepruningvalue!(g::MagnitudePruneGroup, p::Float64) = (g.p = p; g)

function _gpu_prune!(g::MagnitudePruneGroup)

    cpu_group = MagnitudePruneGroup(cpu.(g.layers), g.p)

    _cpu_prune!(cpu_group)
    for (cm, gm) in zip(cpu_group.layers, g.layers)
        copyto!(gm, gpu(cm))
    end
    g
end

function _cpu_prune!(g::MagnitudePruneGroup)
    L = 0

    v = Vector{Any}()

    weightsandmasks = collect(
        Iterators.flatten(
            zip(LotteryTickets.prunableweights(l), LotteryTickets.prunableweightmasks(l)) for l in g.layers
        ),
    )

    for (idx, (weight, mask)) in enumerate(weightsandmasks)
        indices = findall(view(mask, :))
        L += length(indices)
        append!(v, collect(zip(weight[indices], Iterators.repeated(idx), indices)))
    end

    k = min(L, round(Int, g.p * L, RoundUp))

    indices = sortperm(v, by = x -> abs(x[1]))[1:k]

    for (_, m_idx, idx) in v[indices]
        weightsandmasks[m_idx][1][idx] = 0.0
        weightsandmasks[m_idx][2][idx] = false
    end
    g
end

function prune!(g::MagnitudePruneGroup)
    if all(
        mask isa CuArray for l in g.layers for mask in LotteryTickets.prunableweightmasks(l)
    )
        _gpu_prune!(g)
    else
        _cpu_prune!(g)
    end
    g
end

function rewind!(g::MagnitudePruneGroup)
    for layer in g.layers
        rewind!(layer)
        applymask!(layer)
    end
    g
end

function pruneandrewind!(g::MagnitudePruneGroup)
    prune!(g)
    rewind!(g)
    g
end

######################################
#
# Identity Prune Group
#
#####################################

struct IdentityPruneGroup <: AbstractPruneGroup
    layers::Vector{Union{AbstractPrunableLayer,Flux.Recur{<:AbstractPrunableRecurrentCell}}}
end


prune!(g::IdentityPruneGroup) = g

function rewind!(g::IdentityPruneGroup)
    for layer in g.layers
        rewind!(layer)
        applymask!(layer)
    end
    g
end

pruneandrewind!(g::IdentityPruneGroup) = (prune!(g); rewind!(g); g)

######################################
#
# Pruner
#
######################################

struct Pruner
    groups::Vector{AbstractPruneGroup}
end

function prune!(p::Pruner)
    for g in p.groups
        prune!(g)
    end
    p
end

function pruneandrewind!(p::Pruner)
    for g in p.groups
        pruneandrewind!(g)
    end
    p
end