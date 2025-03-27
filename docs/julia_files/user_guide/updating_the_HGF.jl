# # Updating the HGF 


# In this section we will describe the HGF update process. We update all nodes in an HGF based on an input. This input can either be missing, a single value, a vector of values, or a dictionary of input node names and corresponding values. 

# The update_hgf!() function call takes an hgf and the input as inputs. All HGF's follow the same update order described in this section. The update process is the following:

# We start by updating the predictions from the last timestep of all nodes. First the state nodes, then the input nodes. 

# We give the first input to the input node. Then in the input node we calculate the value prediction error (prediction from last timestep vs. new input). 

# (Early update state nodes) We update the input node's value parents posteriors, value prediction error and volatility prediction error

# We update the input node's volatility prediction errors

# (Late update state nodes) We update the remaining state nodes. We update their posterior, value prediction error and volatility prediciton error.

# The update_hgf!() function does not return anything but updates the HGF.

# # Example of updating the HGF

# We deinfe a premade HGF:
using HierarchicalGaussianFiltering

hgf = premade_hgf("help")

hgf = premade_hgf("binary_3level");

update_hgf!(hgf, [1, 0, 1]);

# As you can see, the states are updated but only with the last trial saved.

get_states(hgf)
