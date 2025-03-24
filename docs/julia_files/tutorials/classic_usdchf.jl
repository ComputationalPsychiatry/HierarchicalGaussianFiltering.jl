# # Tutorial on 2-level continuous HGF

#This is a replication of the tutorial from the MATLAB toolbox, using an HGF to filter the exchange rates between USD and CHF

# First load packages
using ActionModels
using HierarchicalGaussianFiltering
using Plots
using StatsPlots
using Distributions

# Get the path for the HGF superfolder
hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))
# Add the path to the data files
data_path = hgf_path * "/docs/julia_files/tutorials/data/"

# Load the data
inputs = Float64[]
open(data_path * "classic_usdchf_inputs.dat") do f
    for ln in eachline(f)
        push!(inputs, parse(Float64, ln))
    end
end

#Create HGF
hgf = premade_hgf("continuous_2level", verbose = false);
agent = premade_agent("hgf_gaussian", hgf, verbose = false);

# Set parameters for parameter recover
parameters = Dict(
    ("x", "xvol", "coupling_strength") => 1.0,
    ("u", "input_noise") => -log(1e4),
    ("x", "volatility") => -13,
    ("xvol", "volatility") => -2,
    ("x", "initial_mean") => 1.04,
    ("x", "initial_precision") => 1 / (0.0001),
    ("xvol", "initial_mean") => 1.0,
    ("xvol", "initial_precision") => 1 / 0.1,
    "action_noise" => 0.01,
);

set_parameters!(agent, parameters)
reset!(agent)

# Evolve agent
actions = give_inputs!(agent, inputs);

# Plot trajectories
plot_trajectory(
    agent,
    "u",
    size = (1300, 500),
    xlims = (0, 615),
    markersize = 3,
    markercolor = "green2",
    title = "HGF trajectory",
    ylabel = "CHF-USD exchange rate",
    xlabel = "Trading days since 1 January 2010",
)
#-
plot_trajectory!(agent, ("x", "posterior"), color = "red")
plot_trajectory!(
    agent,
    "action",
    size = (1300, 500),
    xlims = (0, 614),
    markersize = 3,
    markercolor = "orange",
)
#-
plot_trajectory(
    agent,
    "xvol",
    color = "blue",
    size = (1300, 500),
    xlims = (0, 615),
    xlabel = "Trading days since 1 January 2010",
    title = "Volatility parent trajectory",
)
#-
# Set priors for fitting
fixed_parameters = Dict(
    ("x", "xvol", "coupling_strength") => 1.0,
    ("x", "initial_mean") => 0,
    ("x", "initial_precision") => 2000,
    ("xvol", "initial_mean") => 1.0,
    ("xvol", "initial_precision") => 600.0,
);

param_priors = Dict(
    ("u", "input_noise") => Normal(-6, 1),
    ("x", "volatility") => Normal(-4, 1),
    ("xvol", "volatility") => Normal(-4, 1),
    "action_noise" => LogNormal(log(0.01), 1),
);
#-
# # Prior predictive simulation plot
# plot_predictive_simulation(
#     param_priors,
#     agent,
#     inputs,
#     ("x", "posterior_mean");
#     n_simulations = 100,
# )
#-
# Do parameter recovery
model = create_model(agent, param_priors, inputs, actions)

#Fit single chain with 10 iterations
fitted_model = fit_model(model; n_iterations = 10, n_chains = 1)
#-
# Plot the chains
plot(fitted_model.chains)
#-
# Plot prior posterior distributions
# plot_parameter_distribution(fitted_model, param_priors)
#-
