# # Fitting parameters in HGF agents


# - [Introduction To Fitting Models](#Introduction-To-Fitting-Models)

# - [Setting Priors and The Fit_model() Function](#Setting-Priors-and-The-Fit_model()-Function)

# - [Plotting Functions](#Plotting-Functions)

# - [Predictive Simulations](#Predictive-Simulations)



# ## Introduction To Fitting Models Function

# When you work with participants' data in HGF-agents and you have one or more target parameters in sight for investigation, you can recover them with model fitting. When you fit models for different groups of participant, you can idnetify group differences based on the parameter recovery.

# ## Setting Priors and The Fit_model() 

# Hierarchical Gaussian Filtering uses the fit_model() function from the ActionModels.jl package. 
# The fit_ model() function takes the following inputs:

# ![Image1](../images/fit_model_image.png)

# Let us run through the inputs to the function one by one. 

# - agent::Agent: a specified agent created with either premade agent or init\_agent.
# - param_priors::Dict: priors (written as distributions) for the parameters you wish to fit. e.g. priors = Dict("learning_rate" => Uniform(0, 1))
# - inputs:Array: array of inputs.
# - actions::Array: array of actions.
# - fixed_parameters::Dict = Dict(): fixed parameters if you wish to change the parameter settings of the parameters you dont fit
# - sampler = NUTS(): specify the type of sampler. See Turing documentation for more details on sampler types.
# - n_iterations = 1000: iterations pr. chain.
# - n_chains = 1: amount of chains.
# - verbose = true: set to false to hide warnings
# - show\_sample\_rejections = false: if set to true, get a message every time a sample is rejected.
# - impute\_missing\_actions = false : if true, include missing actions in the fitting process.

# We will run through an example of fitting an agent model to data.

# load packages
using ActionModels
using HierarchicalGaussianFiltering

# We will define a binary 3-level HGF and its parameters

hgf_parameters = Dict(
    ("u", "category_means") => Real[0.0, 1.0],
    ("u", "input_precision") => Inf,
    ("xprob", "volatility") => -2.5,
    ("xprob", "initial_mean") => 0,
    ("xprob", "initial_precision") => 1,
    ("xvol", "volatility") => -6.0,
    ("xvol", "initial_mean") => 1,
    ("xvol", "initial_precision") => 1,
    ("xbin", "xprob", "coupling_strength") => 1.0,
    ("xprob", "xvol", "coupling_strength") => 1.0,
)

hgf = premade_hgf("binary_3level", hgf_parameters, verbose = false)

# Create an agent
agent_parameters = Dict("action_noise" => 0.2);
agent =
    premade_agent("hgf_unit_square_sigmoid_action", hgf, agent_parameters, verbose = false);

# Define a set of inputs
inputs =
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0];

# Evolve agent and save actions
actions = give_inputs!(agent, inputs)


# We can  by plotting the actions our agent has produced.
using StatsPlots
using Plots
plot_trajectory(agent, ("u", "input_value"))
plot_trajectory!(agent, ("xbin", "prediction"))



# When defining the fixed parameters for fit_model() it overrites any previous parameter settings with the "newly" defined fixed parameters. If you dont state any fixed parameters it uses the current parameter values.

# We define a set of fixed parameters to use in this fitting process:

# Set fixed parameters. We choose to fit the evolution rate of the xprob node. 
fixed_parameters = Dict(
    "action_noise" => 0.2,
    ("u", "category_means") => Real[0.0, 1.0],
    ("u", "input_precision") => Inf,
    ("xprob", "initial_mean") => 0,
    ("xprob", "initial_precision") => 1,
    ("xvol", "initial_mean") => 1,
    ("xvol", "initial_precision") => 1,
    ("xbin", "xprob", "coupling_strength") => 1.0,
    ("xprob", "xvol", "coupling_strength") => 1.0,
    ("xvol", "volatility") => -6.0,
);

# As you can read from the fixed parameters, the evolution rate of xprob is not configured. We set the prior for the xprob evolution rate:
using Distributions
param_priors = Dict(("xprob", "volatility") => Normal(-3.0, 0.5));

# We can fit the evolution rate by inputting the variables:

# Fit the actions
fitted_model = fit_model(
    agent,
    param_priors,
    inputs,
    actions,
    fixed_parameters = fixed_parameters,
    verbose = true,
    n_iterations = 10,
)
set_parameters!(agent, hgf_parameters)

# ## Plotting Functions
plot(fitted_model)

# Plot the posterior
plot_parameter_distribution(fitted_model, param_priors)


# # Predictive Simulations with plot\_predictive\_distributions()

# Hierarical Gaussian Filtering uses the plot\_predictive\_distribution function from Action Models to produce and plot predictive prior/posterior simulations.

# For more information on predictive simulations check out ActionModels.jl documentation [LINK]

# We will provide a code example of prior and posterior predictive simulation. We can fit a different parameter, and start with a  prior predictive check.

# Set prior we wish to simulate over
param_priors = Dict(("xvol", "initial_precision") => Normal(1.0, 0.5));

# When we look at our predictive simulation plot we should aim to see actions in the plausible space they could be in.
# Prior predictive plot
plot_predictive_simulation(
    param_priors,
    agent,
    inputs,
    ("xbin", "prediction_mean"),
    n_simulations = 100,
)

# Let's fit our model

# Fit the actions where we use the default parameter values from the HGF. 
fitted_model =
    fit_model(agent, param_priors, inputs, actions, verbose = true, n_iterations = 10)
set_parameters!(agent, hgf_parameters)

# We can place our turing chain as a our posterior in the function, and get our posterior predictive simulation plot:
plot_predictive_simulation(
    fitted_model,
    agent,
    inputs,
    ("xbin", "prediction_mean"),
    n_simulations = 100,
)

# We can get the posterior 
get_posteriors(fitted_model)

# plot the chains
plot(fitted_model)

# plot the parameter distribution
plot_parameter_distribution(fitted_model, param_priors)
