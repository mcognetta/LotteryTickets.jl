# LotteryTickets.jl: A Library for Sparsifying Flux Models

<p align="center">
    <img width="400px" src="white_background_logo.png#gh-light-mode-only">
    <img width="400px" src="white_background_logo_darkmode.png#gh-dark-mode-only">
</p>

This library provides the basic tools for iterative pruning to find winning tickets. In particular, it provides prunable versions of all layers provided by Flux as well implementations of pruning strategies. The library also allows for easily producing a sparse representation of a model, after training and pruning has been completed.

## Background

The *Lottery Ticket Hypothesis* roughly states that deep, dense neural networks contain sparse subnetworks that account for most of the performance of the fully-parameterized model. For example, in some common architectures/tasks, subnetworks with fewer than 10% of the original number of parameters can be found that outperform the original model. Such subnetworks are called *winning tickets*.

Finding winning tickets is challenging, but one common technique is to iteratively train a model, prune the bottom X% of parameters (by magnitude), reset the remaining weights to their original initialization, and train the model again. This process is repeated until the desired parameter count is reached or the model performance begins to degrade.

## General Implementation Notes

This library revolves around wrapping Flux layers in a layer that captures and masks gradients. These layers contain the underlying layer, a set of masks, and a set copies of the original underlying weights. The prunable weights (usually dense matrices in the wrapped layer) have a corresponding mask (a boolean matrix of the same shape). If an index in the mask is `false`, it signals that the corresponding weight has been pruned. The underlying matrix weight will be set to zero, but this is not enough to ensure that the weight is pruned --- the gradients for that weight must also be set to zero during the backwards pass. The wrapped layers also include a copy of the original weight matrix (usually the initialization weights), so that the weights can be reset between training and pruning rounds.

We use the the word *rewind* and *reset* interchangeably to mean resetting the weights of the underlying layer to their original initialization. In the API, this is always called `rewind`.

## Layers
All (trainable) Flux layers are supported by LotteryTickets.jl, with identical construction and calling signatures. For example, the `Dense` layer has an analogous `PrunableDense` type defined by LotteryTickets.jl. This type implements all of the constructors that `Dense` implements, as well as a way to construct a `PruneableDense` layer from a preexisting `Dense` specification. It also implements all of the calling conventions that `Dense` implements (that is, it accepts any input type that `Dense` would accept, and acts on it in the same way).

```julia
julia> using Flux, LotteryTickets

julia> d = Flux.Dense(3 => 5, relu)

# prunable dense construction
julia> p = LotteryTickets.PrunableDense(3 => 5, relu)

# wrapping an existing dense layer
julia> p = LotteryTickets.PrunableDense(d)

# the calling signature is identical to `Dense`
julia> p(rand(Float32, 3))
```

In general, a Flux layer `Layer` has a prunable counterpart `PrunableLayer`. For example:
- `Dense -> PrunableDense`
- `LSTM -> PrunableLSTM`
- `Conv -> PrunableConv`

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

### Interface

When wrapping a new layer type, there is a not-so-small interface to implement (reducing the interface surface is a top priority for this project), but implementing each part is usually straightforward. It is best to follow the examples in `layers.jl` if you are stuck. The most basic one is `PrunableDense`, but you may also be interested in `PrunableRNNCell` for layers that have multiple outputs or `PrunableBilinear` for layers that have multiple inputs.

Suppose we have a new layer `CustomLayer` defined as:

```julia
struct CustomLayer{W1, W2, ...}
  w1::W1
  w2::W2
  ...
end

Flux.@functor CustomLayer
```
#### Constructors

For constructors, you should first implement a struct for holding the base layer and mask and original weight information:

```julia
PrunableCustomLayer{C<:CustomLayer,W1,W2...,M1,M2,...} <: AbstractPrunableLayer
  c::C
  orig_w1::W1
  orig_w2::W2
  ...
  mask1::M1
  mask2::M2
  ...
end
```

where `c` is an instance of your `CustomLayer` and `orig_w1, ..., orig_wn` are the same type as the weights in your layer (these will store copies of the original weights) while `mask1, ..., maskn` are boolean matrices of the same shape as the corresponding weight.

You should make this a functor by including the line:

`Flux.@functor PrubableCustomLayer`

And you should mark the wrapped type as the only trainable parameter:

`Flux.trainable(c::PrunableCustomLayer) = (; c = c.c)`

Then, you should implement:

- `PrunableCustomLayer(l::CustomLayer)`: construct a `PrunableCustomLayer` from a `CustomLayer`, which should copy all the original weights into `orig_w1, ..., orig_wn` and construct the appropriate sized masks.

You should also implement all the same construction signatures for `PrunableCustomLayer` as for `CustomLayer`, but usually this is as simple as defining: 

`PrunableCustomLayer(<args>) = PrunableCustomLayer(CustomLayer(<args>))`

#### Callers

For each caller method for `CustomLayer`, you should implement an analogous method for the `PrunableCustomLayer` type.

This is usually as easy as `(c::PrunableCustomLayer)(x) = c.c(x)`. You may worry that the mask is not applied properly in this call, but the remaining API will properly mask out the weights when pruning is applied so that you should not have to worry about it during inference.

#### Gradients

The most complicated part is correctly defining the gradient masking for a new layer. In short, we want to capture the pullback of the wrapped type (the `CustomLayer` instance), mask its gradient, then pass that along as the gradient for the `PrunableCustomLayer`. For each call signature, you should implement a custom `Zygote.@adjoint`. In it, you want to compute the forward pass of the wrapped `CustomLayer`, but make a new pullback function that captures the `CustomLayer` gradient and applies the mask. Assume `CustomLayer` has a single-input call signature. Then you would define:

```julia
Zygote.@adjoint function (c::PrunableCustomLayer)(x)
  # capture the pullback from the wrapped type
  out, pb = Zygote._pullback(c.c, x)

  # define a new pullback
  function masked_pb(y)
    # capture the gradients from the wrapped pullback
    grad, val = pb(y)

    # mask the gradients and merge them back into grad
    # this allows for weights that aren't masked (e.g., biases)
    # to retain their original gradient information, and only weights
    # that are masked are updated
    masked_gradients = (; w1 = grad.w1 .* c.mask1, w2 = grad.w2 .* c.mask2, ...)

    # return the updated gradients
    return merge(grad, (; c = masked_gradients)), val
  
  end
  return out, masked_pb
end
```

See `layers.jl` for examples of this function.

#### Other

A few other methods are required, but are easy to implement:

- `prunableweights` - return the weights that can be pruned from the *wrapped* layer (e.g., `w1`, `w2`, ... from the wrapped `CustomLayer`)
- `prunableweightmasks` - return the masks (e.g., `mask1`, `mask2`, ... from the `PrunableCustomLayer`)
- `prunableweightorigins`- return the original weights (e.g., `orig_w1`, `orig_w2`, ... from `PrunableCustomLayer`)

**NOTE:** It is important that these return the analogous weights in the same relative orders.

For `PrunableCustomLayer` it would be sufficient to define:

```julia
prunableweights(c::PrunableCustomLayer) = (c.c.w1, c.c.w2, ...)
prunableweightmasks(c::PrunableCustomLayer) = (c.mask1, c.mask2, ...)
prunableweightorigins(c::PrunableCustomLayer) = (c.orig_w1, c.orig_w2, ...)
```

These are used to implement the following methods:

- `applymask!`: mask out the weights in the wrapped layer
- `checkpoint!`: update the original weights (for example, if you don't want to start all the way over at the initial weights each time you prune and rewind)
- `rewind!`: reset the weights to their original values (ignoring pruned weights)

If you would like to implement special behavior for any of these, you can reimplement `_applymask!`, `_checkpoint!`, or `_rewind!` (note the leading underscores).

You should also supply a `_sparsify` (note the leading underscore) method, which produces a `CustomLayer` (*not* `PrunableCustomLayer`) but with the weights converted to a sparse representation (after weight masking is applied).

For this example, it might look like:

```julia
using SparseArrays

_sparsify(c::PrunableCustomLayer) = (applymask!(c); CustomLayer(sparse(c.c.w1), sparse(c.c.w2), ...))
```

## Pruners

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

### Interface

A prune group should implement the following interface:

- `prune!(g::AbstractPruneGroup)`: prune the layers in `g` by modifying the masks of each layer, and return `g`

The remaining API is:

- `rewind!(g::AbstractPruneGroup)`: rewind all layers to their original weights (default behavior just calls `rewind!` on each layer)
- `pruneandrewind!(g::AbstractPruneGroup)`: prune then rewind all layers to their original weights (default behavior just calls `prune!(g)` then `rewind!(g)`)

These are defined for the `AbstractPruneGroup` type, but can be specified for your specific type if different functionality is desired.

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

## Rough Edges

A small rough edge is with convolutional layers. `Conv` is implemented as a dense *tensor*, but `SparseArrays` only supports vectors and arrays. Thus, when calling `sparsify` on a `PrunableConv` layer, will just result in a regular (dense) `Conv` layer, but with the pruned weights masked out.

A major rough edge is the handling of layers that you don't want to prune. When resetting the network between pruning rounds, one may still want to reset these layers as well. The ideal way to do this is to wrap it in a prunable layer, but not prune it. To handle this, we have an `IdentityPruner` type, that implements the same interface as other pruners, but simply doesn't prune any weights. This means it can still rewind the weights of its prunable layers.

In our example above, you may want to replace the model definition with:

```julia
  m = Chain(
      	PrunableDense(1024 => 256, relu),
      	PrunableLSTM(256 => 256),
      	PrunableDense(256 => 64, relu),
      	PrunableDense(64 => 10, relu),
  ) |> config.device
```

and add another group `g3 = IdentityPruneGroup([m[4]])` to the `Pruner`.

The second rough edge is also related to this issue. When calling `sparsify`, layers are converted to a sparse representation regardless of the level of sparsity of the underlying weights. In the pessimal case, layers that were not pruned at all (e.g., those that are in an `IdentityPruner` group) would be converted to a sparse representation despite being fully dense. As of right now, this library does not have a built-in way to handle this case, so it must be handled specially by the user. However, this should only happen at the very end of training and pruning, so it can be handled in a one-off fashion as a postprocessing step.

Finally, there is currently no check that a layer exists only in one pruning group. The burden is on the caller to ensure that layers appear only in the desired groups.

## Masked Dense Matrices vs Sparse Matrices

One may ask why we use dense matrices with weight masks to simulate pruned weights, rather than a sparse matrix representation. The main reason is that, until a matrix is *very* sparse, masking + dense matrix-matrix multiply is substantially faster than sparse matrix multiplication.

The second reason is that, if gradient updates are not handled correctly, it is easy to accidently make a sparse matrix very dense (for example, by doing an element-wise addition). Using dense matrices + masking removes this issue at the cost of slightly more complex code.

## Related Projects

There are some other great projects in the sparsification space. In no particular order:

- [OpenLTH](https://github.com/facebookresearch/open_lth): PyTorch library for lottery-ticket style pruning by the original authors of the [Lottery Ticket paper](https://arxiv.org/abs/1803.03635)
- [JaxPruner](https://github.com/google-research/jaxpruner): A Jax library for pruning neural models
- [TinyNets.jl](https://github.com/AInnervate/TinyNets.jl): A Julia library for iterative pruning

## TODO
Contributions are welcome. Here are some areas under active development:

- [ ] Minimizing prunable layer API surface / automatic prunable wrapping
- [ ] Pruning Schedulers
- [ ] Sparse Model Zoo
- [ ] Better handling of unwrapping/sparsifying layers
- [ ] Lux support
