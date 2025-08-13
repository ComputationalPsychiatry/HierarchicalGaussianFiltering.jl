# # Tutorial on 3-level binary

# This tutorial is a copy of the 3 level binary hgf tutorial in MATLAB

# First load packages
using ActionModels
using HierarchicalGaussianFiltering
using CSV
using DataFrames
using StatsPlots



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
action_model = ActionModel(HGFSigmoid(; HGF = hgf, action_noise = 0.2))
agent = init_agent(action_model);

# Evolve agent and save actions
actions = simulate!(agent, inputs);

# Plot the trajectory of the agent
plot(agent, ("u", "input_value"))
plot!(agent, ("xbin", "prediction"))

# -

plot(agent, ("xprob", "posterior"))
plot(agent, ("xvol", "posterior"))


# Set priors for parameter recovery
prior = (; xprob_volatility = Normal(-3.0, 0.5))

#-
# Get the actions from the MATLAB tutorial
actions = CSV.read(data_path * "classic_binary_actions.csv", DataFrame)[!, 1];
#-
# Fit the actions
#Create model
model =
    create_model(action_model, prior, inputs, actions, check_parameter_rejections = true)

#Fit model
posterior_chains = sample_posterior!(model, n_samples = 200, n_chains = 2)
#-
#Plot the chains
plot(posterior_chains)






# #Define nodes
# nodes = [
#     BinaryInput(name = "u"),
#     BinaryState(name = "xbin"),
#     ContinuousState(name = "xprob"),
#     ContinuousState(name = "xvol"),
# ]

# #Define edges
# edges = Dict(
#     ("u", "xbin") => ObservationCoupling(),
#     ("xbin", "xprob") => ProbabilityCoupling(),
#     ("xprob", "xvol") => VolatilityCoupling(),
# )

# #Initialise HGF
# hgf = init_hgf(nodes = nodes, edges = edges, node_defaults = NodeDefaults(
#     update_type = HierarchicalGaussianFiltering.UnboundedUpdate(),
# ))


# # Create an HGF
# hgf_parameters = Dict(
#     ("xprob", "volatility") => 1,
#     ("xprob", "initial_mean") => 0,
#     ("xprob", "initial_precision") => 1,
#     ("xvol", "volatility") => 1,
#     ("xvol", "initial_mean") => 1,
#     ("xvol", "initial_precision") => 1,
#     ("xbin", "xprob", "coupling_strength") => 1.0,
#     ("xprob", "xvol", "coupling_strength") => 1.0,
# );

# set_parameters!(hgf, hgf_parameters)
# reset!(hgf)


# multiple_inputs!(hgf, inputs)



# plt1 = plot(hgf, ("xbin", "prediction"))
# plot!(inputs, seriestype = :scatter)

# plt2 = plot(hgf, ("xprob", "posterior"))
# plt3 = plot(hgf, ("xvol", "posterior"))

# plot(plt1, plt2, plt3, layout = (3, 1), size = (400, 600), legend = false)