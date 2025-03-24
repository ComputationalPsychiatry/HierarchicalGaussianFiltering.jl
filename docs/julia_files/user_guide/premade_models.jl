# # Premade models

# The Hierarchical Gaussian Filtering package contains a set of premade HGF's and HGF-agents for you to use. We will provide an overview of how to work with the premade agent models, and lastly a total overview of the premade HGF models to use in the package.

# ## Premade HGF-agents

# 1. [HGF Gaussian Action Noise](#HGF-with-Gaussian-Action-Noise-agent)
# 2. [HGF Binary Softmax](#HGF-Binary-Softmax-agent)
# 3. [HGF unit square sigmoid](#HGF-unit-square-sigmoid-agent)
# 4. [HGF-Predict-Category](#HGF-Predict-Category-agent)


# ## HGF with Gaussian Action Noise agent

# This premade agent model can be found as "hgf_gaussian" in the package. The Action distribution is a gaussian distribution with mean of the target state from the chosen HGF, and the standard deviation consisting of the action precision parameter inversed.  #md

# - Default hgf: contionus_2level
# - Default Target state: (x, posterior mean)
# - Default Parameters: gaussian action precision = 1

# ## HGF Binary Softmax agent

# The action distribution is a Bernoulli distribution, and the parameter is action probability. Action probability is calculated using a softmax on the action precision parameter and the target value from the HGF. #md

# - Default hgf: binary_3level
# - Default target state; (xbin, prediction mean)
# - Default parameters: softmax action precision = 1

# ## HGF unit square sigmoid agent

# The action distribution is Bernoulli distribution with the parameter beinga a softmax of the target value and action precision.

# - Default hgf: binary_3level
# - Default target state; (xbin, prediction mean)
# - Default parameters: softmax action precision = 1 


# ## HGF Predict Category agent

# The action distribution is a categorical distribution. The action model takes the target node from the HGF, and takes out the prediction state. This state is a vector of values for each category. The vector is the only thing used in the categorical distribution 

# - Default hgf: categorical_3level
# - Default target state: Target categorical node xcat
# - Default parameters: none

# ## Using premade agents

# We will demonstrate how to work with a premade agent with basic functions from the ActionModels.jl package. 

# Getting a list of premade HGF agents
using HierarchicalGaussianFiltering

premade_agent("help")

# Define an agent with default parameter values and default HGF 
agent = premade_agent("hgf_binary_softmax")

# ## Utility functions for accessing parameters and states

# Get all parameters in an agent:
get_parameters(agent)

# Get specific parameter in agent:
get_parameters(agent, ("xvol", "initial_precision"))

# Get all states in an agent:
get_states(agent)

# Get specific state in an agent:
get_states(agent, ("xbin", "posterior_precision"))

# Set a parameter value
set_parameters!(agent, ("xvol", "initial_precision"), 0.4)

# Set multiple parameter values
set_parameters!(
    agent,
    Dict(("xvol", "initial_precision") => 1, ("xvol", "volatility") => 0),
)


# Let us move on to giving a set of inputs to the agent. 

# Define inputs
input = [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0]

# Give inputs and generate actions
actions = give_inputs!(agent, input)

# Get the history of a single state in the agent
get_history(agent, ("xbin", "prediction_mean"))

# We can plot the input and prediciton means with plot trajectory. Notice, when using plot_trajectory!() you can layer plots. 

using StatsPlots
using Plots

plot_trajectory(agent, ("u", "input_value"))

# Let's add prediction mean on top of the plot
plot_trajectory!(agent, ("xbin", "prediction_mean"))

