# # Creating an HGF Agent

# In this section we will build a binary 2-level HGF from scratch using the init_hgf() funciton. 

# When building an HGF we need to define the following:

# 1. [Input Nodes](#Defining-Input-Nodes)
# 2. [State Nodes](#Defining-State-Nodes)
# 3. [Edges](#Defining-Edges)


# A binary two level HGF is fairly simple. It consists of a binary input node, a binary state node and a continuous state node. 

# The continuous state node is a  value parent for the binary state node. The Binary input node has the binary state node as parent. Let's start with setting up the binary input node.

# ## Defining Input Nodes

# We can recall from the HGF nodes, that a binary input node's parameters are category means and input precision. We will set category means to [0,1] and the input precision to Inf. 

using HierarchicalGaussianFiltering
using ActionModels

nodes = [
    BinaryInput("Input_node"),
    BinaryState("binary_state_node"),
    ContinuousState(
        name = "continuous_state_node",
        volatility = -2,
        initial_mean = 0,
        initial_precision = 1,
    ),
]

# ## Defining State Nodes

# We are defining two state nodes. Let's start with the binary state node. The only parameter in this node is value coupling which is set when defining edges.

# The continuous state node have evolution rate, initial mean and initial precision parameters which we specify as well. 

# ## Defining Edges

# When defining the edges we start by sepcifying which node the perspective is from. So, when we specify the edges we start by specifying what the child in the relation is. 

# At the buttom of our hierarchy we have the binary input node. The Input node has binary state node as parent. 

edges = Dict(
    ("Input_node", "binary_state_node") => ObservationCoupling(),
    ("binary_state_node", "continuous_state_node") => ProbabilityCoupling(1),
);

# We are ready to initialize our HGF now.

Binary_2_level_hgf = init_hgf(nodes = nodes, edges = edges, verbose = false);
# We can access the states in our HGF:
get_states(Binary_2_level_hgf)
#-
# We can access the parameters in our HGF
get_parameters(Binary_2_level_hgf)


# # Creating an Agent and Action model

# Agents and aciton models are two sides of the same coin. The Hierarchical Gaussian Filtering package uses the Actionmodels.jl package for configuration of models, agents and fitting processes. An agent means nothing without an action model and vise versa. You can see more on action models in the documentation for ActionModel.jl
# The agent will have our Binary 2-level HGF as a substruct.

# In this example we would like to create an agent whose actions are distributed according to a Bernoulli distribution with action probability is the softmax of one of the nodes in the HGF.

# We initialize the action model and create it. In a softmax action model we need a parameter from the agent called softmax action precision which is used in the update step of the action model. 

using Distributions
function binary_softmax(agent, input)

    ##------- Staty by getting all information ---------


    ##Get HGF from the agents' substruct
    hgf = agent.substruct

    ##Take out the target state from the agents' settings. The target state will be specified in the agent
    target_state = agent.settings["target_state"]

    ##Take out the parameter from our agent
    action_noise = agent.parameters["action_noise"]

    ##Get the specified state out of the hgf
    target_value = get_states(hgf, target_state)

    ##--------------- Update step starts  -----------------

    ##Use sotmax to get the action probability 
    action_probability = 1 / (1 + exp(action_noise * target_value))

    ##---------------- Update step end  ------------------
    ##If the action probability is not between 0 and 1
    #if !(0 <= action_probability <= 1)
    ##Throw an error that will reject samples when fitted
    ##throw(
    ##RejectParameters(
    ## "With these parameters and inputs, the action probability became $action_probability, which should be between 0 and 1. Try other parameter settings",
    ##),
    ##)
    ##end

    ##---------------- Get action distribution  ------------------

    ##Create Bernoulli normal distribution with mean of the target value and a standard deviation from parameters
    distribution = Distributions.Bernoulli(action_probability)

    ##Return the action distribution
    return distribution
end

# ## Creating an agent using our action model and having our HGF as substruct


# We will create an agent with the init_agent() function. We need to specify an action model, substruct, parameters, states and settings. 

# Let's define our action model

action_model = binary_softmax;

# The parameter of the agent is just softmax action precision. We set this value to 1

parameters = Dict("action_noise" => 1);

# The states of the agent are empty, but the states from the HGF will be accessible.

states = Dict()

# In the settings we specify what our target state is. We want it to be the prediction mean of our binary state node.

settings = Dict("target_state" => ("binary_state_node", "prediction_mean"));

## Let's initialize our agent
agent = init_agent(
    action_model,
    substruct = Binary_2_level_hgf,
    parameters = parameters,
    states = states,
    settings = settings,
)

## Define inputs
Inputs = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0];

## Give Inputs and save actions
actions = give_inputs!(agent.substruct, Inputs)


# plot the input and the prediction state from our binary state node

using Plots
using StatsPlots

plot_trajectory(agent, ("Input_node", "input_value"))

plot_trajectory!(agent, ("binary_state_node", "prediction"))
