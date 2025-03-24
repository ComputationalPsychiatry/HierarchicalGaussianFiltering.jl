###################################
######## Update prediction ########
###################################

##### Superfunction #####
"""
    update_node_prediction!(node::BinaryStateNode)

Update the prediction of a single binary state node.
"""
function update_node_prediction!(node::BinaryStateNode, stepsize::Real)

    #Update prediction mean
    node.states.prediction_mean = calculate_prediction_mean(node)

    #Update prediction precision
    node.states.prediction_precision = calculate_prediction_precision(node)

    return nothing
end

##### Mean update #####
"""
    calculate_prediction_mean(node::BinaryStateNode)

Calculates a binary state node's prediction mean.
"""
function calculate_prediction_mean(node::BinaryStateNode)
    probability_parents = node.edges.probability_parents

    prediction_mean = 0

    for parent in probability_parents
        prediction_mean +=
            parent.states.prediction_mean * node.parameters.coupling_strengths[parent.name]
    end

    prediction_mean = 1 / (1 + exp(-prediction_mean))

    return prediction_mean
end

##### Precision update #####
"""
    calculate_prediction_precision(node::BinaryStateNode)

Calculates a binary state node's prediction precision.
"""
function calculate_prediction_precision(node::BinaryStateNode)
    1 / (node.states.prediction_mean * (1 - node.states.prediction_mean))
end

##################################
######## Update posterior ########
##################################

##### Superfunction #####
"""
    update_node_posterior!(node::AbstractStateNode; update_type::HGFUpdateType)

Update the posterior of a single continuous state node. This is the classic HGF update.
"""
function update_node_posterior!(node::BinaryStateNode, update_type::HGFUpdateType)
    #Update posterior precision
    node.states.posterior_precision = calculate_posterior_precision(node)

    #Update posterior mean
    node.states.posterior_mean = calculate_posterior_mean(node, update_type)

    return nothing
end

##### Precision update #####
"""
    calculate_posterior_precision(node::BinaryStateNode)

Calculates a binary node's posterior precision.
"""
function calculate_posterior_precision(node::BinaryStateNode)

    ## If the child is an observation child ##
    if length(node.edges.observation_children) > 0

        #Extract the observation child
        child = node.edges.observation_children[1]

        #Simple update with inifinte precision
        posterior_precision = Inf


        ## If the child is a category child ##
    elseif length(node.edges.category_children) > 0

        posterior_precision = Inf

    else
        @error "the binary state node $(node.name) has neither category nor observation children"
    end

    return posterior_precision
end

##### Mean update #####
"""
    calculate_posterior_mean(node::BinaryStateNode)

Calculates a node's posterior mean.
"""
function calculate_posterior_mean(node::BinaryStateNode, update_type::HGFUpdateType)

    ## If the child is an observation child ##
    if length(node.edges.observation_children) > 0

        #Extract the child
        child = node.edges.observation_children[1]

        #For missing inputs
        if ismissing(child.states.input_value)
            #Set the posterior to missing
            posterior_mean = missing
        else
            posterior_mean = child.states.input_value
        end

        ## If the child is a category child ##
    elseif length(node.edges.category_children) > 0

        #Extract the child
        child = node.edges.category_children[1]

        #Find the nodes' own category number
        category_number = findfirst(child.edges.category_parent_order .== node.name)

        #Find the corresponding value in the child
        posterior_mean = child.states.posterior[category_number]

    else
        @error "the binary state node $(node.name) has neither category nor observation children"
    end

    return posterior_mean
end


###############################################
######## Update value prediction error ########
###############################################

##### Superfunction #####
"""
    update_node_value_prediction_error!(node::AbstractStateNode)

Update the value prediction error of a single state node.
"""
function update_node_value_prediction_error!(node::BinaryStateNode)
    #Update value prediction error
    node.states.value_prediction_error = calculate_value_prediction_error(node)

    return nothing
end

"""
    calculate_value_prediction_error(node::AbstractNode)

Calculate's a state node's value prediction error.
"""
function calculate_value_prediction_error(node::BinaryStateNode)
    node.states.posterior_mean - node.states.prediction_mean
end

###################################################
######## Update precision prediction error ########
###################################################

##### Superfunction #####
"""
    update_node_precision_prediction_error!(node::BinaryStateNode)

There is no volatility prediction error update for binary state nodes.
"""
function update_node_precision_prediction_error!(node::BinaryStateNode)
    return nothing
end
