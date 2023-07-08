struct Pruner
    groups::Vector{PruneGroup}
end

function prune!(p::Pruner)
    for g in p.groups
        prune!(g)
    end
end

struct PruneGroup
    layers::Vector{<:AbstractPrunableLayer}
    p::Float64
end

function _gpu_prune!(g::PruneGroup, zerooutweights::Bool = false)

    cpu_group = PruneGroup(cpu.(g.layers), g.p)

    _cpu_prune!(cpu_group, zerooutweights)
    for (cm, gm) in zip(cpu_group.layers, g.layers)
        copyto!(gm, gpu(cm))
    end
    g
end

function _cpu_prune!(g::PruneGroup, zerooutweights::Bool = false)
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
        zerooutweights && (weightsandmasks[m_idx][1][idx] = 0.0)
        weightsandmasks[m_idx][2][idx] = false
    end
    g
end

function prune!(g::PruneGroup, zerooutweights::Bool = false)
    if all(
        mask isa CuArray for l in g.layers for mask in LotteryTickets.prunableweightmasks(l)
    )
        _gpu_prune!(g, zerooutweights)
    else
        _cpu_prune!(g, zerooutweights)
    end
    g
end

function rewind!(g, zerooutweights::Bool = false)
    for layer in g.layers
        for (weight, orig, mask) in zip(
            prunableweights(layer),
            prunableweightmasks(layer),
            prunableweightmasks(layer),
        )
            weight .*= orig
            zerooutweights && weight .*= mask
        end
    end
    g
end

function pruneandrewind!(g::PruneGroup, zerooutweights::Bool = false)
    prune!(g, zerooutweights)
    rewind!(g, zerooutweights)
    g
end
