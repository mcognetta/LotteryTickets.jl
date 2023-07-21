# LotteryTickets.jl: A Library for Sparsifying Flux Models

<p align="center">
    <img width="400px" src="white_background_logo.png#gh-light-mode-only">
    <img width="400px" src="white_background_logo_darkmode.png#gh-dark-mode-only">
</p>

## Background

The *Lottery Ticket Hypothesis* roughly states that deep, dense neural networks contain sparse subnetworks that account for most of the performance of the fully-parameterized model. In some cases, subnetworks with fewer than 10% of the original number of parameters can be found that outperform the original model. Such subnetworks are called *winning tickets*.

Finding winning tickets is challenging, but one common technique is to iteratively train a model, prune the bottom X% of parameters (by magnitude), reset the remaining weights to their original initialization, and train the model again. This process is repeated until the desired parameter count is reached or the model performance begins to degrade.

## LotteryTickets.jl

This library provides the basic tools for iterative pruning to find winning tickets. In particular, it provides prunable versions of all layers provided by Flux as well implementations of pruning strategies. The library also allows for easily producing a sparse representation of a model, after training and pruning has been completed.

### Layers
All (trainable) Flux layers are supported by LotteryTickets.jl, with identical construction signatures. For example, the `Dense` layer has an analagous `PrunableDense` type defined by LotteryTickets. This type implements all of the constructors that `Dense` implements, as well as a way to construct a `PruneableDense` layer from a preexisting `Dense` specification.

```julia
julia> using Flux, LotteryTickets

julia> d = Flux.Dense(3 => 5, relu)

# prunable dense construction
julia> p = LotteryTickets.PrunableDense(3 => 5, relu)

# wrapping an existing dense layer
julia> p = LotteryTickets.PrunableDense(d)
```

In general, a Flux layer `Layer` has a prunable counterpart `PrunableLayer`. For example:
- `Dense -> PrunableDense`
- `LSTM -> PrunableLSTM`
- `Conv -> Prunable Conv`

These layers can be used as normal in Flux models.

```julia
  m = Chain(
      	# a prunable dense layer
      	PrunableDense(1024 => 256, relu),
      	# all flux layers are supported
      	PrunableLSTM(256 => 256),
      	PrunableDense(256 => 64, relu),
      	# mixing prunable and non-prunable is ok!
      	Dense(64 => 10, relu),
  ) |> device # cpu and gpu is supported
```

Prunable layers can be converted to a sparse representation after pruning using the `sparsify` method. This applies to nested models as well (like `Chain`), but only prunable layers are converted.

### Pruners

Prunable layers enable pruning, but `LotteryTickets.jl` implements the actual pruning procedures through `PruneGroups`. Roughly, a prune group is a collection of layers (and their prunable weights) as well as a pruning strategy. Layers in a pruning group are pruned collectively via the specified strategy.

The basic pruner type is *magnitude pruning*, implemented by `MagnitudePruneGroup`, which takes a percentage `p` and prunes the bottom `p%` of parameters from the group by magnitude.

Given a prunable model, a `MagnitudePruneGroup` can be specified by selecting the layers from the model to prune together and the pruning percentage.

```julia

julia> group1 = MagnitudePruneGroup([m[1], m[2]], 0.2)

julia> group2 = MagnitudePruneGroup([m[3]], 0.1)
```

Pruning groups are controlled by a `Pruner`, which activates all of the pruning groups' execution at the same time.

```julia

julia> pruner = Pruner([group1, group2])

```

The layers within the prune group can be pruned, reset (to their original weights), or both.

```julia
julia> pruneandrewind!(p)
```

### Example usage

```julia
using Flux, LotteryTickets

function main(config)
  m = Chain(
      	# a prunable dense layer
      	PrunableDense(1024 => 256, relu),
      	# all flux layers are supported
      	PrunableLSTM(256 => 256),
      	PrunableDense(256 => 64, relu),
      	# mixing prunable and non-prunable is ok!
      	Dense(64 => 10, relu),
  ) |> config.device
  
  # pruning groups. layers in a group are
  # pruned collectively
  g1 = MagnitudePruneGroup([m[1], m[2]], 0.2)
  g2 = MagnitudePruneGroup([m[3]], 0.1)

  # the pruner controller
  p = Pruner([g1, g2])
  
  for _ in 1:config.pruning_rounds
    
    # run a full training job to convergence
    train_model!(m, config)
    
    # prune and reset the model for the next
    # training round
    pruneandrewind!(p)
  end
  
  # convert the model to a sparse representation
  return sparsify(m)
end
```