using ActionModels, HierarchicalGaussianFiltering
using CSV, DataFrames
using Plots, StatsPlots
using Distributions


# Get the path for the HGF superfolder
hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))
# Add the path to the data files
data_path = hgf_path * "/docs/julia_files/tutorials/data/"

#Load data
data = CSV.read(data_path * "classic_cannonball_data.csv", DataFrame)
inputs = data[(data.ID.==20).&(data.session.==1), :].outcome

#Create HGF
hgf = premade_hgf("JGET", verbose = false)
#Create agent
agent = premade_agent("hgf_gaussian", hgf)
#Set parameters
parameters = Dict(
    "action_noise" => 1,
    ("u", "input_noise") => 0,
    ("x", "initial_mean") => first(inputs) + 2,
    ("x", "initial_precision") => 0.001,
    ("x", "volatility") => -8,
    ("xvol", "volatility") => -8,
    ("xnoise", "volatility") => -7,
    ("xnoise_vol", "volatility") => -2,
    ("x", "xvol", "coupling_strength") => 1,
    ("xnoise", "xnoise_vol", "coupling_strength") => 1,
)
set_parameters!(agent, parameters)
reset!(agent)

#Simulate updates and actions
actions = give_inputs!(agent, inputs);
#Plot belief trajectories
plot_trajectory(agent, "u")
plot_trajectory!(agent, "x")
plot_trajectory(agent, "xvol")
plot_trajectory(agent, "xnoise")
plot_trajectory(agent, "xnoise_vol")
