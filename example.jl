# # Simple multi-layer perceptron


# In this example, we create a simple [multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP) that classifies handwritten digits
# using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). A MLP consists of at least *three layers* of stacked perceptrons: Input, hidden, and output. Each neuron of an MLP has parameters 
# (weights and bias) and uses an [activation function](https://en.wikipedia.org/wiki/Activation_function) to compute its output. 


# ![mlp](../mlp_mnist/docs/mlp.svg)

# Source: http://d2l.ai/chapter_multilayer-perceptrons/mlp.html



# To run this example, we need the following packages:

using Revise, Optimisers
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA
using MLDatasets
using LotteryTickets

# We set default values for learning rate, batch size, epochs, and the usage of a GPU (if available) for the model:

@kwdef mutable struct Args
    η::Float64 =  0.0005        ## learning rate
    batchsize::Int = 32         ## batch size
    epochs::Int = 10             ## number of epochs
    use_cuda::Bool = true       ## use gpu (if cuda available)
    use_conv_model::Bool = false ## MLP or CNN
    prune::Bool = true
end

# If a GPU is available on our local system, then Flux uses it for computing the loss and updating the weights and biases when training our model.


# ## Data

# We create the function `getdata` to load the MNIST train and test data from [MLDatasets](https://github.com/JuliaML/MLDatasets.jl) and reshape them so that they are in the shape that Flux expects. 

function getdata(args, device)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    ## Load dataset	
    if args.use_conv_model
        xtrain, ytrain = MLDatasets.CIFAR10(:train)[:]
        xtest, ytest = MLDatasets.CIFAR10(:test)[:]
    
        train4dim = reshape(xtrain, 28,28,1,:)    # insert trivial channel dim
        trainyhot = Flux.onehotbatch(ytrain, 0:9)  # make a 10×60000 OneHotMatrix
        train_loader = Flux.DataLoader((train4dim, trainyhot) |> device; batchsize=args.batchsize, shuffle=true)

        test4dim = reshape(xtest, 28,28,1,:)    # insert trivial channel dim
        testyhot = Flux.onehotbatch(ytest, 0:9)  # make a 10×60000 OneHotMatrix
        test_loader = Flux.DataLoader((test4dim, testyhot) |> device ; batchsize=args.batchsize)

        return train_loader, test_loader
    else
        xtrain, ytrain = MLDatasets.MNIST(:train)[:]
        xtest, ytest = MLDatasets.MNIST(:test)[:]
    
        ## Reshape input data to flatten each image into a linear array
        xtrain = Flux.flatten(xtrain)
        xtest = Flux.flatten(xtest)

        ## One-hot-encode the labels
        ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

        ## Create two DataLoader objects (mini-batch iterators)
        train_loader = DataLoader((xtrain, ytrain) |> device, batchsize=args.batchsize, shuffle=true)
        test_loader = DataLoader((xtest, ytest) |> device, batchsize=args.batchsize)

        return train_loader, test_loader
    end
end

# The function `getdata` performs the following tasks:

# * **Loads MNIST dataset:** Loads the train and test set tensors. The shape of train data is `28x28x60000` and test data is `28X28X10000`. 
# * **Reshapes the train and test data:**  Uses the [flatten](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.flatten) function to reshape the train data set into a `784x60000` array and test data set into a `784x10000`. Notice that we reshape the data so that we can pass these as arguments for the input layer of our model (a simple MLP expects a vector as an input).
# * **One-hot encodes the train and test labels:** Creates a batch of one-hot vectors so we can pass the labels of the data as arguments for the loss function. For this example, we use the [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) function and it expects data to be one-hot encoded. 
# * **Creates mini-batches of data:** Creates two DataLoader objects (train and test) that handle data mini-batches of size `1024 ` (as defined above). We create these two objects so that we can pass the entire data set through the loss function at once when training our model. Also, it shuffles the data points during each iteration (`shuffle=true`).

# ## Model

# As we mentioned above, a MLP consist of *three* layers that are fully connected. For this example, we define our model with the following layers and dimensions: 

# * **Input:** It has `784` perceptrons (the MNIST image size is `28x28`). We flatten the train and test data so that we can pass them as arguments to this layer.
# * **Hidden:** It has `32` perceptrons that use the [relu](https://fluxml.ai/Flux.jl/stable/models/nnlib/#NNlib.relu) activation function.
# * **Output:** It has `10` perceptrons that output the model's prediction or probability that a digit is 0 to 9. 


# We define the model with the `build_model` function: 


# function build_model(; imgsize=(28,28,1), nclasses=10)
#     return Chain( MaskedDense(Dense(prod(imgsize) => 512, relu)),
#                   MaskedDense(Dense(512 => 256, relu)),
#                   MaskedDense(Dense(256 => 128, relu)),
#                   Dense(128 => nclasses))
# end

function build_model(; imgsize = (28, 28, 1), nclasses=10)

    return Chain(PrunableDense(prod(imgsize) => 512, relu),
                 PrunableDense(512 => 256, relu), 
                 PrunableDense(256 => 128, relu),
                 Dense(128 => nclasses))
end

function build_old_model(; imgsize = (28, 28, 1), nclasses=10)

    return Chain(Dense(prod(imgsize) => 512, relu),
                 Dense(512 => 256, relu), 
                 Dense(256 => 128, relu),
                 Dense(128 => nclasses))
end

function build_conv_model(; imgsize=(28, 28, 1), nclasses = 10)
    return Chain(
        LotteryTickets.MaskedConv((5, 5), 1=>6, relu),
        MaxPool((2, 2)),
        LotteryTickets.MaskedConv((5, 5), 6=>16, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        MaskedDense(256 => 120, relu),
        MaskedDense(120 => 84, relu), 
        MaskedDense(84 => 10),
    )
end


# Note that we use the functions [Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) so that our model is *densely* (or fully) connected and [Chain](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain) to chain the computation of the three layers.

# ## Loss function

# Now, we define the loss function `loss_and_accuracy`. It expects the following arguments:
# * ADataLoader object.
# * The `build_model` function we defined above.
# * A device object (in case we have a GPU available).

function loss_and_accuracy(data_loader, model)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        # x, y = device(x), device(y)
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
        num +=  size(x)[end]
    end
    return ls / num, acc / num
end

# This function iterates through the `dataloader` object in mini-batches and uses the function 
# [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) to compute the difference between 
# the predicted and actual values (loss) and the accuracy. 


# ## Train function

# Now, we define the `train` function that calls the functions defined above and trains the model.

loss(m, x, y) = logitcrossentropy(m(x), y)

function train(model, opt, args, train, test; kws...)

    opt_state = Optimisers.setup(opt, model)
    
    ## Training
    for epoch in 1:args.epochs
        # for (x, y) in train
        #     # x, y = device(x), device(y) ## transfer data to device
        #     gs = gradient(() -> logitcrossentropy(model(x), y), ps) ## compute gradient
        #     Flux.Optimise.update!(opt, ps, gs) ## update parameters
        # end

        # for (x, y) in train
        #     loss, grad = Flux.withgradient(ps) do
        #         # Evaluate model and loss inside gradient context:
        #         # y_hat = model(x)
        #         # Flux.crossentropy(y_hat, y)
        #         logitcrossentropy(model(x), y)
        #     end
        #     Flux.update!(opt, ps, grad)
        #     # push!(losses, loss)  # logging, outside gradient context
        # end

        Flux.train!(loss, model, train, opt_state)

        ## Report on train and test
        train_loss, train_acc = loss_and_accuracy(train, model)
        test_loss, test_acc = loss_and_accuracy(test, model)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
    end

    model, opt
end

# ## Run the example 

# function main(;kws...) 
#     args = Args(; kws...) ## Collect options in a struct for convenience
#     device = args.use_cuda ? gpu : cpu
#     model = build_model() |> device
        
#     ## Optimizer
#     opt = ADAM(args.η)

#     model, opt = train(model, opt, args)

#     return model, opt, args
# end

function main(;kws...)
    CUDA.allowscalar(false)
    if Args().prune
        model, opt = schedule(5; kws...)
    else
        model, opt = train(; kws...)
    end
    return model, opt
end


# We call the `train` function:
function schedule(rounds; kws...)
    # models = Vector{Any}()
    args = Args(; kws...) ## Collect options in a struct for convenience
    device = args.use_cuda ? gpu : cpu
    # model = build_old_model() |> device

    if args.use_conv_model
        model = build_conv_model() |> device
        pruner = LotteryTickets.MagnitudePruneGroup([model[1].weight, model[3].weight, model[6].weight, model[7].weight])
    else
        model = build_model() |> device
        pruner = LotteryTickets.MagnitudePruneGroup([model[1], model[2], model[3]], 0.1)
    end

    # model = build_model(;imgsize = (32,32,3)) |> device
    # pruner = LotteryTickets.PruneGroup([model[1], model[2], model[3]], 0.15)

    ## Create test and train dataloaders
    train_loader, test_loader = getdata(args, device)
    ## Optimizer

    opt = Optimisers.Descent(args.η)
    # model, opt = train(model, opt, args)
    for r in 1:rounds-1
        @info "ROUND $r"
        train(model, opt, args, train_loader, test_loader)

        LotteryTickets.pruneandrewind!(pruner, true)

        CUDA.reclaim()
        # model = new
    end
    @info "ROUND $rounds (last round)"
    train(model, opt, args, train_loader, test_loader)

    test_loss, test_acc = loss_and_accuracy(test_loader, model)
    println("FINAL LOSS AND ACCURACY  test_loss = $test_loss, test_accuracy = $test_acc")
    model, opt
end


function train(; kws...)
    # models = Vector{Any}()
    args = Args(; kws...) ## Collect options in a struct for convenience
    device = args.use_cuda ? gpu : cpu
    # model = build_old_model() |> device

    model = build_old_model() |> device

    # model = build_model(;imgsize = (32,32,3)) |> device
    # pruner = LotteryTickets.PruneGroup([model[1], model[2], model[3]], 0.15)

    ## Create test and train dataloaders
    train_loader, test_loader = getdata(args, device)
    opt = Optimisers.Descent(args.η)
    train(model, opt, args, train_loader, test_loader)

    ## Optimizer

    # model, opt = train(model, opt, args)
    train(model, opt, args, train_loader, test_loader)
    test_loss, test_acc = loss_and_accuracy(test_loader, model)
    println("FINAL LOSS AND ACCURACY  test_loss = $test_loss, test_accuracy = $test_acc")
    model, opt
end


if abspath(PROGRAM_FILE) == @__FILE__
    model, opt = main()
end

# >**Note:** We can change hyperparameters by modifying train(η=0.01). 

# ## Resources
 
# * [3Blue1Brown Neural networks videos](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
# * [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)



