# # Tutorial on 3-level binary

# This tutorial is a copy of the 3 level binary hgf tutorial in MATLAB

# First load packages
using ActionModels
using HierarchicalGaussianFiltering
using CSV
using DataFrames
using Plots
using StatsPlots
using Distributions

# Get the path for the HGF superfolder
hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))
# Add the path to the data files
data_path = hgf_path * "/docs/julia_files/tutorials/data/"

# Load the data 
inputs = CSV.read(data_path * "classic_binary_inputs.csv", DataFrame)[!, 1];

# Create an HGF
hgf_parameters = Dict(
    ("xprob", "volatility") => -2.5,
    ("xprob", "initial_mean") => 0,
    ("xprob", "initial_precision") => 1,
    ("xvol", "volatility") => -6.0,
    ("xvol", "initial_mean") => 1,
    ("xvol", "initial_precision") => 1,
    ("xbin", "xprob", "coupling_strength") => 1.0,
    ("xprob", "xvol", "coupling_strength") => 1.0,
);

hgf = premade_hgf("binary_3level", hgf_parameters, verbose = false);

# Create an agent
agent_parameters = Dict("action_noise" => 0.2);
agent = premade_agent("hgf_unit_square_sigmoid", hgf, agent_parameters, verbose = false);

# Evolve agent and save actions
actions = give_inputs!(agent, inputs);

# Plot the trajectory of the agent
plot_trajectory(agent, ("u", "input_value"))
plot_trajectory!(agent, ("xbin", "prediction"))

# -

plot_trajectory(agent, ("xprob", "posterior"))
plot_trajectory(agent, ("xvol", "posterior"))

# Set fixed parameters
fixed_parameters = Dict(
    "action_noise" => 0.2,
    ("xprob", "initial_mean") => 0,
    ("xprob", "initial_precision") => 1,
    ("xvol", "initial_mean") => 1,
    ("xvol", "initial_precision") => 1,
    ("xbin", "xprob", "coupling_strength") => 1.0,
    ("xprob", "xvol", "coupling_strength") => 1.0,
    ("xvol", "volatility") => -6.0,
);

# Set priors for parameter recovery
param_priors = Dict(("xprob", "volatility") => Normal(-3.0, 0.5));
#-
# # Prior predictive plot
# plot_predictive_simulation(
#     param_priors,
#     agent,
#     inputs,
#     ("xbin", "prediction_mean"),
#     n_simulations = 100,
# )
#-
# Get the actions from the MATLAB tutorial
actions = CSV.read(data_path * "classic_binary_actions.csv", DataFrame)[!, 1];
#-
# Fit the actions
#Create model
model = create_model(agent, param_priors, inputs, actions)

#Fit single chain with 10 iterations
fitted_model = fit_model(model; n_iterations = 10, n_chains = 1)
#-
#Plot the chains
plot(fitted_model.chains)
#-
# Plot the posterior
# plot_parameter_distribution(fitted_model, param_priors)
#-
# Posterior predictive plot
# plot_predictive_simulation(
#     fitted_model,
#     agent,
#     inputs,
#     ("xbin", "prediction_mean"),
#     n_simulations = 3,
# )
