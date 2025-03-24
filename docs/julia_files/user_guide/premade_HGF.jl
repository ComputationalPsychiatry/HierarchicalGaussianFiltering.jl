# # Premade HGF's in the Hierarchical Gaussian Filtering package

# For information on states and parameters of the nodes see section on HGF nodes [LINK]

# - [Continous 2-level HGF](#Continuous-2-level-HGF) 
# - [JGET HGF](#JGET-HGF)
# - [Binary 2-level HGF](#Binary-2-level-HGF)
# - [Binary 3-level HGF](#Binary-3-level-HGF)
# - [Categorical 3-level HGF](#Categorical-3-level-HGF)
# - [Categorical 3-level state transition HGF](#Categorical-3-level-state-transition)


# #Load data for examples

using HierarchicalGaussianFiltering #hide
using ActionModels #hide
using CSV #hide
using DataFrames #hide
using Plots #hide
using StatsPlots #hide

#CSV.read(pwd(), DataFrame)

hgf_path_continuous = dirname(dirname(pathof(HierarchicalGaussianFiltering))); #hide
hgf_path_continuous = hgf_path_continuous * "/docs/julia_files/tutorials/data/"; #hide

inputs_continuous = Float64[]; #hide
open(hgf_path_continuous * "classic_usdchf_inputs.dat") do f #hide
    for ln in eachline(f) #hide
        push!(inputs_continuous, parse(Float64, ln)) #hide
    end #hide
end #hide

hgf_path_binary = dirname(dirname(pathof(HierarchicalGaussianFiltering))); #hide

hgf_path_binary = hgf_path_binary * "/docs/julia_files/tutorials/data/"; #hide

inputs_binary = CSV.read(hgf_path_binary * "classic_binary_inputs.csv", DataFrame)[!, 1]; #hide



# ## Continuous 2-level HGF



# ![continuous 2-level graph](../images/HGF_structures/continuous.svg)


# The continuous 2-level HGF is structured with following nodes:

# - input node: continuous 
# - state nodes: 
#   - 1st level: continuous (value coupling to input node)
#   - 2nd level: continous (volatility coupling to 1st level)


#Create HGF and Agent
continuous_2_level = premade_hgf("continuous_2level");
agent_continuous_2_level =
    premade_agent("hgf_gaussian", continuous_2_level, verbose = false);

# Evolve agent plot trajetories
give_inputs!(agent_continuous_2_level, inputs_continuous);
plot_trajectory(
    agent_continuous_2_level,
    "xvol",
    color = "blue",
    size = (1300, 500),
    xlims = (0, 615),
    xlabel = "Trading days since 1 January 2010",
    title = "Volatility parent trajectory",
)


# ## JGET HGF

# - input node: continuous 
# - state nodes:
#   - 1st level: continuous (value coupling to input node)
#   - 2nd level: continous (volatility coupling to 1st level)
#   - 3rd level: continous (volatility coupling to input node)
#   - 4th level: continous (volatility coupling to 3rd level)


#Create HGF and Agent
JGET = premade_hgf("JGET");
agent_JGET = premade_agent("hgf_gaussian", JGET, verbose = false);

# Evolve agent plot trajetories
give_inputs!(agent_JGET, inputs_continuous);
plot_trajectory(
    agent_JGET,
    "xvol",
    color = "blue",
    size = (1300, 500),
    xlims = (0, 615),
    xlabel = "Trading days since 1 January 2010",
    title = "Volatility parent trajectory",
)

# ## Binary 2-level HGF

# ![continuous 2-level graph](../images/HGF_structures/binary.svg)


# - input node: binary
# - state nodes:
#   - 1st level: binary (value coupling to input node)
#   - 2nd level: continous (volatility coupliong to 1st level)

hgf_binary_2_level = premade_hgf("binary_2level", verbose = false);

# Create an agent
agent_binary_2_level =
    premade_agent("hgf_unit_square_sigmoid", hgf_binary_2_level, verbose = false);

# Evolve agent plot trajetories
give_inputs!(agent_binary_2_level, inputs_binary);
plot_trajectory(agent_binary_2_level, ("u", "input_value"))
plot_trajectory!(agent_binary_2_level, ("xbin", "prediction"))


#-

plot_trajectory(agent_binary_2_level, ("xprob", "posterior"))

# ## Binary 3-level HGF

# - input node: Binary
# - state nodes:
#   - 1st level: binary (value coupling to input node)
#   - 2nd level: continous (value coupling to 1st level)
#   - 3rd level: continous (volatility coupling to 2nd level)

hgf_binary_3_level = premade_hgf("binary_3level", verbose = false);

# Create an agent
agent_binary_3_level =
    premade_agent("hgf_unit_square_sigmoid", hgf_binary_3_level, verbose = false);

# Evolve agent plot trajetories
give_inputs!(agent_binary_3_level, inputs_binary);
plot_trajectory(agent_binary_3_level, ("u", "input_value"))
plot_trajectory!(agent_binary_3_level, ("xbin", "prediction"))

#-

plot_trajectory(agent_binary_3_level, ("xprob", "posterior"))
#-
plot_trajectory(agent_binary_3_level, ("xvol", "posterior"))



# ## Categorical 3-level HGF

# ![continuous 2-level graph](../images/HGF_structures/categorical.svg)


# The categorical 3-level HGF model takes an input from one of m categories and learns the probability of a category appearing.

# - input node: categorical
# - state nodes:
#   - 1st level: categorical (value coupling to input node)
#   - 2nd level: m binary (all value couplings to 1st level)
#   - 3rd level: continuous (shared volatility coupling to all m nodes in 2nd level)

# ## Categorical 3-level state transition HGF

# The categorical 3-level HGF model learns state transition probabilities between a set of categorical states.

# - input node: categorical input nodes
# - state nodes:
#   - 1st level: categorical state nodes (value coupling to input node)
#   - 2nd level: binary state nodes for each categorical state node (value coupling from each categorical state node to binary state nodes)
#   - 3rd level: continous (volatility coupling to all nodes in 2nd level)
