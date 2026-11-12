###################################
######## Update prediction ########
###################################

##### Superfunction #####
"""
    update_node_prediction!(node::AbstractInputNode)

Update the posterior of a single input node.
"""
function update_node_prediction!(node::ContinuousInputNode, stepsize::Real)

    #Update node prediction mean
    node.states.prediction_mean = calculate_prediction_mean(node)

    #Update prediction precision
    node.states.prediction_precision = calculate_prediction_precision(node)

    return nothing
end


##### Mean update #####
function calculate_prediction_mean(node::ContinuousInputNode)
    #Extract parents
    observation_parents = node.edges.observation_parents

    #Initialize prediction at the bias
    prediction_mean = node.parameters.bias

    #Sum the predictions of the parents
    for parent in observation_parents
        prediction_mean += parent.states.prediction_mean
    end

    return prediction_mean
end


##### Precision update #####
"""
    calculate_prediction_precision(node::AbstractInputNode)

Calculates an input node's prediction precision.
"""
function calculate_prediction_precision(node::ContinuousInputNode)

    #Extract noise parents
    noise_parents = node.edges.noise_parents

    #Initialize noise from input noise parameter
    predicted_noise = node.parameters.input_noise

    #Go through each noise parent
    for parent in noise_parents
        #Add its mean to the predicted noise
        predicted_noise +=
            parent.states.posterior_mean * node.parameters.coupling_strengths[parent.name]
    end

    #Calculate prediction precision
    prediction_precision = exp(-predicted_noise)

    return prediction_precision
end



###############################################
######## Update value prediction error ########
###############################################

##### Superfunction #####
"""
    update_node_value_prediction_error!(node::AbstractInputNode)

Update the value prediction error of a single input node.
"""
function update_node_value_prediction_error!(node::ContinuousInputNode)

    #Calculate value prediction error
    node.states.value_prediction_error = calculate_value_prediction_error(node)

    return nothing
end



"""
    calculate_value_prediction_error(node::ContinuousInputNode)

Calculate's an input node's value prediction error.
"""
function calculate_value_prediction_error(node::ContinuousInputNode)
    #For missing input
    if ismissing(node.states.input_value)
        #Set the prediction error to missing
        value_prediction_error = missing
    else
        #Get value prediction error between the prediction and the input value
        value_prediction_error = node.states.input_value - node.states.prediction_mean
    end

    return value_prediction_error
end


###################################################
######## Update precision prediction error ########
###################################################

##### Superfunction #####
"""
    update_node_precision_prediction_error!(node::AbstractInputNode)

Update the value prediction error of a single input node.
"""
function update_node_precision_prediction_error!(node::ContinuousInputNode)

    #Calculate volatility prediction error, only if there are volatility parents
    node.states.precision_prediction_error = calculate_precision_prediction_error(node)

    return nothing
end

"""
    calculate_precision_prediction_error(node::ContinuousInputNode)

Calculates an input node's noise prediction error (Weber et al., 2026, Eq. 72). It compares the input with the posterior of the node's observation parents, which have been updated with the input before this step: their posterior variances plus the squared distance between the input and the value they imply, scaled by the predicted noise precision.
"""
function calculate_precision_prediction_error(node::ContinuousInputNode)

    #If there are no noise parents
    if length(node.edges.noise_parents) == 0
        #Skip
        return missing
    end

    #For missing input
    if ismissing(node.states.input_value)
        #Set the prediction error to missing
        precision_prediction_error = missing
    else
        #Initialize the input implied by the observation parents' posteriors at the bias, and its variance at zero
        posterior_input_mean = node.parameters.bias
        posterior_input_variance = 0

        #For each observation parent
        for parent in node.edges.observation_parents
            #Add its posterior mean and its posterior variance
            posterior_input_mean += parent.states.posterior_mean
            posterior_input_variance += 1 / parent.states.posterior_precision
        end

        #Get the noise prediction error from the posterior variance and the squared distance between the input and the posterior mean
        precision_prediction_error =
            node.states.prediction_precision * (
                posterior_input_variance +
                (node.states.input_value - posterior_input_mean)^2
            ) - 1
    end

    return precision_prediction_error
end
