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

    #The prediction precision is the inverse of the predicted noise
    prediction_precision = 1 / exp(predicted_noise)

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

Calculates an input node's volatility prediction error.
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
        #Extract parents
        observation_parents = node.edges.observation_parents

        #Average the posterior precision of the observation parents 
        parents_average_posterior_precision = 0

        for parent in observation_parents
            parents_average_posterior_precision += parent.states.posterior_precision
        end

        parents_average_posterior_precision =
            parents_average_posterior_precision / length(observation_parents)

        #Get the noise prediction error using the average parent parents_posterior_precision 
        precision_prediction_error =
            node.states.prediction_precision / parents_average_posterior_precision +
            node.states.prediction_precision * node.states.value_prediction_error^2 - 1
    end

    return precision_prediction_error
end
