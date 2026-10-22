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

function hgf_softmax(attributes::ModelAttributes, hgf_observation::Int64)
    #Extract HGF
    hgf = attributes.submodel

    #Extract inverse temperature
    β = 1/load_parameters(attributes).action_noise

    #Update the HGF
    update_hgf!(hgf, hgf_observation)

    #Extract the predicted probability
    value = get_states(hgf, :binary_state_node_prediction_mean)

    #Calculate the action probability with a binary softmax (a softmax over the two options with values p and 1 - p)
    action_probability = logistic(β * (2 * value - 1))

    #Create Bernoulli distribution with mean of the target value and a standard deviation from parameters
    return Bernoulli(action_probability)
end

# ## Creating an agent using our action model and having our HGF as substruct


# We will create an agent with the init_agent() function. We need to specify an action model, substruct, parameters, states and settings. 

# Let's define our action model

am_function = hgf_softmax;

# The parameter of the agent is just softmax action noise. We set this value to 1

parameters = (action_noise = Parameter(1.0),)

# The states of the agent are empty, but the states from the HGF will be accessible.

states = (;)

## Let's initialize our agent

#Define observations and actions
observations = (; hgf_observation = Observation(Int64))
actions = (; report = Action(Bernoulli),)

#Create final action model
action_model = ActionModel(
    am_function,
    parameters = parameters,
    states = states,
    observations = observations,
    actions = actions,
    submodel = Binary_2_level_hgf,
)

## Create agent for simulation
agent = init_agent(action_model)


## Define inputs
inputs = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0];

## Give Inputs and save actions
actions = simulate!(agent, inputs)


# plot the input and the prediction state from our binary state node

using StatsPlots

plot(agent, "Input_node")

plot!(agent, "binary_state_node")
