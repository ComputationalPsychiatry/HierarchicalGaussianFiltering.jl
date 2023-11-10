# # Variations of utility functions in the Hierarchical Gaussian Filtering package


# A lot of commonly used utility functions are collected here in an overview with examples. The following utility functions can be used:

# 1. [Getting Parameters](#Getting-Parameters)
# 2. [Getting States](#Getting-States)
# 3. [Setting Parameters](#Setting-Parameters)
# 4. [Giving Inputs](#Giving-Inputs)
# 5. [Getting History](#Getting-History)
# 6. [Plotting State Trajectories](#Plotting-State-Trajectories)
# 7. [Getting Predictions](#Getting-Predictions)
# 8. [Getting Surprise](#Getting-Surprise)
# 9. [Resetting an HGF-agent](#Resetting-an-HGF-agent)

# we start by defining an agent to use
using HierarchicalGaussianFiltering

# See which agent to choose
premade_agent("help")

# set agent
agent = premade_agent("hgf_binary_softmax_action")


# ### Getting Parameters

#Let us start by defining a premade agent:

#getting all parameters 
get_parameters(agent)

# getting couplings 
get_parameters(agent, ("xprob", "xvol", "volatility_coupling"))

# getting multiple parameters specify them in a vector
get_parameters(agent, [("xvol", "volatility"), ("xvol", "initial_precision")])


# ### Getting States

#getting all states from an agent model
get_states(agent)

#getting a single state
get_states(agent, ("xprob", "posterior_precision"))

#getting multiple states
get_states(agent, [("xprob", "posterior_precision"), ("xprob", "volatility_weighted_prediction_precision")])


# ### Setting Parameters

# you can set parameters before you initialize your agent, you can set them after and change them when you wish to.
# Let's try an initialize a new agent with parameters. We start by choosing the premade unit square sigmoid action agent whose parameter is sigmoid action precision.

agent_parameter = Dict("sigmoid_action_precision" => 3)

#We also specify our HGF and custom parameter settings:

hgf_parameters = Dict(
    ("u", "category_means") => Real[0.0, 1.0],
    ("u", "input_precision") => Inf,
    ("xprob", "volatility") => -2.5,
    ("xprob", "initial_mean") => 0,
    ("xprob", "initial_precision") => 1,
    ("xvol", "volatility") => -6.0,
    ("xvol", "initial_mean") => 1,
    ("xvol", "initial_precision") => 1,
    ("xbin", "xprob", "value_coupling") => 1.0,
    ("xprob", "xvol", "volatility_coupling") => 1.0,
)

hgf = premade_hgf("binary_3level", hgf_parameters)

# Define our agent with the HGF and agent parameter settings
agent = premade_agent("hgf_unit_square_sigmoid_action", hgf, agent_parameter)


# Changing a single parameter

set_parameters!(agent, ("xvol", "initial_precision"), 4)

# Changing multiple parameters

set_parameters!(
    agent,
    Dict(("xvol", "initial_precision") => 5, ("xbin", "xprob", "value_coupling") => 2.0),
)

# ###Giving Inputs


#give single input
give_inputs!(agent, 0)

#-

#reset the agent
reset!(agent)

# Giving multiple inputs
inputs = [
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
]
give_inputs!(agent, inputs)

# ### Getting History

#getting the action state from the agent
get_history(agent)

#-

# getting history of single state 
get_history(agent, ("xvol", "posterior_precision"))

#-

# getting history of multiple states:
get_history(agent, [("xbin", "prediction_mean"), ("xvol", "posterior_precision")])

# ### Plotting State Trajectories

using StatsPlots
using Plots
## Plotting single state:
plot_trajectory(agent, ("u", "input_value"))

#Adding state trajectory on top
plot_trajectory!(agent, ("xbin", "prediction"))

# Plotting more individual states:



## Plot posterior of xprob
plot_trajectory(agent, ("xprob", "posterior"))

#-

## Plot posterior of xvol
plot_trajectory(agent, ("xvol", "posterior"))

# ### Getting Predictions

# You can specify an HGF or an agent in the funciton. 

# get prediction of the last state
get_prediction(agent)

#specify another node to get predictions from:
get_prediction(agent, "xprob")

# ### Getting Purprise

#getting surprise of input node
get_surprise(agent, "u")

# ### Resetting an HGF-agent

# resetting the agent with reset()

reset!(agent)

# see that action state is cleared
get_history(agent)

# ### Overview of all ultility functions

# ```@autodocs
# Modules = [HierarchicalGaussianFiltering]
# Pages = ["ActionModels_variations/utils/get_history.jl","ActionModels_variations/utils/get_parameters.jl","ActionModels_variations/utils/get_states.jl","ActionModels_variations/utils/give_inputs.jl","ActionModels_variations/utils/reset.jl","ActionModels_variations/utils/set_parameters.jl",utils/get_prediction.jl", "utils/get_surprise.jl"]
# ```
