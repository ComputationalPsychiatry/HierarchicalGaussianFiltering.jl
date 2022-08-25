# This tutorial is a copy of the 3 level binary hgf tutorial in MATLAB
#

# First load packages
using ActionModels
using HGF
using Turing
using CSV
using DataFrames
using Plots
using StatsPlots

# Get the path for the HGF superfolder
hgf_path = dirname(dirname(pathof(HGF)))
# Add the path to the data files
data_path = hgf_path * "/docs/tutorials/data/"

# Load the data 
inputs = CSV.read(data_path * "classic_binary_inputs.csv", DataFrame)[!, 1];

# Create an HGF
hgf_params = Dict(
    ("u", "category_means") => Real[0.0, 1.0],
    ("u", "input_precision") => Inf,
    ("x2", "evolution_rate") => -2.5,
    ("x2", "initial_mean") => 0,
    ("x2", "initial_precision") => 1,
    ("x3", "evolution_rate") => -6.0,
    ("x3", "initial_mean") => 1,
    ("x3", "initial_precision") => 1,
    ("u", "x1", "value_coupling") => 1.0,
    ("x1", "x2", "value_coupling") => 1.0,
    ("x2", "x3", "volatility_coupling") => 1.0,
);
my_hgf = premade_hgf("binary_3level", hgf_params);

# Create an agent
agent_params = Dict("sigmoid_action_precision" => 5);
my_agent = premade_agent("hgf_unit_square_sigmoid_action", my_hgf, agent_params);

# Evolve agent and save actions
actions = give_inputs!(my_agent, inputs);

# Plot the trajectory of the agent
trajectory_plot(my_agent, ("u", "input_value"))
trajectory_plot!(my_agent, ("x1", "prediction"))

trajectory_plot(my_agent, ("x2", "posterior"))
trajectory_plot(my_agent, ("x3", "posterior"))

# Set fixed parameters
fixed_params = Dict(
    "sigmoid_action_precision" => 5,
    ("u", "category_means") => Real[0.0, 1.0],
    ("u", "input_precision") => Inf,
    ("x2", "initial_mean") => 0,
    ("x2", "initial_precision") => 1,
    ("x3", "initial_mean") => 1,
    ("x3", "initial_precision") => 1,
    ("u", "x1", "value_coupling") => 1.0,
    ("x1", "x2", "value_coupling") => 1.0,
    ("x2", "x3", "volatility_coupling") => 1.0,
    ("x2", "evolution_rate") => -3.0,
    ("x3", "evolution_rate") => -6.0,
);

# Set priors for parameter recovery
param_priors = Dict(("x2", "evolution_rate") => Normal(-3.0, 0.5));

# Prior predictive plot
predictive_simulation_plot(param_priors, my_agent, inputs, ("x1", "prediction_mean"))

# Get the actions from the MATLAB tutorial
actions = CSV.read(data_path * "classic_binary_actions.csv", DataFrame)[!, 1];

# Fit the actions
chain = fit_model(
    my_agent,
    inputs,
    actions,
    param_priors,
    fixed_params,
    verbose = true,
)

#Plot the chains
plot(chain)

# Plot the posterior
parameter_distribution_plot(chain, param_priors)

# Posterior predictive plot
predictive_simulation_plot(chain, my_agent, inputs, ("x1", "prediction_mean"))
