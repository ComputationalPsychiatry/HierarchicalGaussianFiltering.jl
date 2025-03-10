###################################
######## Update prediction ########
###################################

##### Superfunction #####
"""
    update_node_prediction!(node::ContinuousStateNode)

Update the prediction of a single state node.
"""
function update_node_prediction!(node::ContinuousStateNode, stepsize::Real)

    #Update prediction mean
    node.states.prediction_mean = calculate_prediction_mean(node, stepsize)

    #Update prediction precision
    node.states.prediction_precision, node.states.effective_prediction_precision =
        calculate_prediction_precision(node, stepsize)

    return nothing
end

##### Mean update #####
@doc raw"""
    calculate_prediction_mean(node::AbstractNode)

Calculates a node's prediction mean.

Uses the equation
`` \hat{\mu}_i=\mu_i+\sum_{j=1}^{j\;value\;parents} \mu_{j} \cdot \alpha_{i,j} ``
"""
function calculate_prediction_mean(node::ContinuousStateNode, stepsize::Real)
    #Get out drift parents
    drift_parents = node.edges.drift_parents

    #Initialize the total drift as the baseline drift
    predicted_drift = node.parameters.drift

    #For each drift parent
    for parent in drift_parents

        #Get out the coupling transform
        coupling_transform = node.parameters.coupling_transforms[parent.name]

        #Transform the parent's value
        drift_increment = transform_parent_value(
            parent.states.posterior_mean,
            coupling_transform;
            derivation_level = 0,
            child = node,
            parent = parent,
        )

        #Add the drift increment
        predicted_drift += drift_increment * node.parameters.coupling_strengths[parent.name]
    end

    #Multiply with stepsize
    predicted_drift = stepsize * predicted_drift

    #Add the drift to the posterior to get the prediction mean
    prediction_mean =
        node.parameters.autoconnection_strength * node.states.posterior_mean +
        predicted_drift

    return prediction_mean
end

##### Precision update #####
@doc raw"""
    calculate_prediction_precision(node::AbstractNode)

Calculates a node's prediction precision.

Uses the equation
`` \hat{\pi}_i^ =  ``
"""
function calculate_prediction_precision(node::ContinuousStateNode, stepsize::Real)
    #Extract volatility parents
    volatility_parents = node.edges.volatility_parents

    #Initialize the predicted volatility as the baseline volatility
    predicted_volatility = node.parameters.volatility

    #Add contributions from volatility parents
    for parent in volatility_parents
        predicted_volatility +=
            parent.states.posterior_mean * node.parameters.coupling_strengths[parent.name]
    end

    #Exponentiate and multiply with stepsize
    predicted_volatility = stepsize * exp(predicted_volatility)

    #Calculate prediction precision 
    prediction_precision = 1 / (1 / node.states.posterior_precision + predicted_volatility)

    #Calculate the volatility-weighted effective precision
    effective_prediction_precision = predicted_volatility * prediction_precision

    #If the posterior precision is negative
    if prediction_precision < 0
        #Throw an error
        throw(
            #Of the custom type where samples are rejected
            RejectParameters(
                "With these parameters and inputs, the prediction precision of node $(node.name) becomes negative after $(length(node.history.prediction_precision)) inputs",
            ),
        )
    end

    return prediction_precision, effective_prediction_precision
end


##################################
######## Update posterior ########
##################################

##### Superfunction #####
"""
    update_node_posterior!(node::AbstractStateNode; update_type::HGFUpdateType)

Update the posterior of a single continuous state node. This is the classic HGF update.
"""
function update_node_posterior!(node::ContinuousStateNode, update_type::ClassicUpdate)
    #Update posterior precision
    node.states.posterior_precision = calculate_posterior_precision(node, update_type)

    #Update posterior mean
    node.states.posterior_mean = calculate_posterior_mean(node, update_type)

    return nothing
end

"""
    update_node_posterior!(node::AbstractStateNode)

Update the posterior of a single continuous state node. This is the enahnced HGF update.
"""
function update_node_posterior!(node::ContinuousStateNode, update_type::EnhancedUpdate)
    #Update posterior mean
    node.states.posterior_mean = calculate_posterior_mean(node, update_type)

    #Update posterior precision
    node.states.posterior_precision = calculate_posterior_precision(node, update_type)

    return nothing
end

"""
    update_node_posterior!(node::AbstractStateNode; update_type::HGFUpdateType)

Update the posterior of a single continuous state node. This is the classic HGF update.
"""
function update_node_posterior!(node::ContinuousStateNode, update_type::UniversalUpdate)
    #Update posterior precision
    node.states.posterior_precision = calculate_posterior_precision(node, update_type)

    #Update posterior mean
    node.states.posterior_mean = calculate_posterior_mean(node, update_type)

    return nothing
end

##### Precision update #####
@doc raw"""
    calculate_posterior_precision(node::AbstractNode)

Calculates a node's posterior precision.

Uses the equation
`` \pi_i^{'} = \hat{\pi}_i +\underbrace{\sum_{j=1}^{j\;children} \alpha_{j,i} \cdot \hat{\pi}_{j}} _\text{sum \;of \;VAPE \; continuous \; value \;chidren} ``
"""
function calculate_posterior_precision(
    node::ContinuousStateNode,
    update_type::HGFUpdateType,
)

    #Initialize as the node's own prediction
    posterior_precision = node.states.prediction_precision

    #Add update terms from drift children
    for child in node.edges.drift_children
        posterior_precision += calculate_posterior_precision_increment(
            node,
            child,
            DriftCoupling(),
            update_type,
        )
    end

    #Add update terms from observation children
    for child in node.edges.observation_children
        posterior_precision +=
            calculate_posterior_precision_increment(node, child, ObservationCoupling())
    end

    #Add update terms from probability children
    for child in node.edges.probability_children
        posterior_precision +=
            calculate_posterior_precision_increment(node, child, ProbabilityCoupling())
    end

    #Add update terms from volatility children
    for child in node.edges.volatility_children
        posterior_precision +=
            calculate_posterior_precision_increment(node, child, VolatilityCoupling())
    end

    #Add update terms from noise children
    for child in node.edges.noise_children
        posterior_precision +=
            calculate_posterior_precision_increment(node, child, NoiseCoupling())
    end


    #If the posterior precision is negative
    if posterior_precision < 0
        #Throw an error
        throw(
            #Of the custom type where samples are rejected
            RejectParameters(
                "With these parameters and inputs, the posterior precision of node $(node.name) becomes negative after $(length(node.history.posterior_precision)) inputs",
            ),
        )
    end

    return posterior_precision
end

function calculate_posterior_precision(
    node::ContinuousStateNode,
    update_type::UniversalUpdate,
)

    #Initialize as the node's own prediction
    posterior_precision = node.states.prediction_precision

    #Get the update from the volatilty and noise children (universal!)
    posterior_precision = update_precision_VOPE(node, 
                            node.edges.volatility_children, 
                            update_type)

    #Add update terms from drift children
    for child in node.edges.drift_children
        posterior_precision += calculate_posterior_precision_increment(
            node,
            child,
            DriftCoupling(),
            update_type,
        )
    end

    #Add update terms from observation children
    for child in node.edges.observation_children
        posterior_precision +=
            calculate_posterior_precision_increment(node, child, ObservationCoupling())
    end

    #Add update terms from probability children
    for child in node.edges.probability_children
        posterior_precision +=
            calculate_posterior_precision_increment(node, child, ProbabilityCoupling())
    end

    #Add update terms from noise children
    for child in node.edges.noise_children
        posterior_precision +=
            calculate_posterior_precision_increment(node, child, NoiseCoupling())
    end
 
    #If the posterior precision is negative
    if posterior_precision < 0
        #Throw an error
        throw(
            #Of the custom type where samples are rejected
            RejectParameters(
                "With these parameters and inputs, the posterior precision of node $(node.name) becomes negative after $(length(node.history.posterior_precision)) inputs",
            ),
        )
    end

    return posterior_precision
end

## Drift coupling ##
function calculate_posterior_precision_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::DriftCoupling,
    update_type::HGFUpdateType,
)
    #Get out the coupling strength and coupling stransform
    coupling_strength = child.parameters.coupling_strengths[node.name]
    coupling_transform = child.parameters.coupling_transforms[node.name]

    #Calculate the increment
    child.states.prediction_precision * (
        coupling_strength^2 * transform_parent_value(
            node.states.posterior_mean,
            coupling_transform;
            derivation_level = 1,
            parent = node,
            child = child,
        ) -
        coupling_strength *
        transform_parent_value(
            node.states.posterior_mean,
            coupling_transform;
            derivation_level = 2,
            parent = node,
            child = child,
        ) *
        child.states.value_prediction_error
    )

end

## Observation coupling ##
function calculate_posterior_precision_increment(
    node::ContinuousStateNode,
    child::ContinuousInputNode,
    coupling_type::ObservationCoupling,
)

    #If missing input
    if ismissing(child.states.input_value)
        #No increment
        return 0
    else
        return child.states.prediction_precision
    end
end

## Probability coupling ##
function calculate_posterior_precision_increment(
    node::ContinuousStateNode,
    child::BinaryStateNode,
    coupling_type::ProbabilityCoupling,
)

    #If there is a missing posterior (due to a missing input)
    if ismissing(child.states.posterior_mean)
        #No update
        return 0
    else
        return child.parameters.coupling_strengths[node.name]^2 /
               child.states.prediction_precision
    end
end

@doc raw"""
    calculate_posterior_precision_vope(node::AbstractNode, child::AbstractNode)

Calculates the posterior precision update term for a single continuous volatility child to a state node.

Uses the equation
``   ``
"""
function calculate_posterior_precision_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::VolatilityCoupling,
)

    1 / 2 *
    (
        child.parameters.coupling_strengths[node.name] *
        child.states.effective_prediction_precision
    )^2 +
    child.states.precision_prediction_error *
    (
        child.parameters.coupling_strengths[node.name] *
        child.states.effective_prediction_precision
    )^2 -
    1 / 2 *
    child.parameters.coupling_strengths[node.name]^2 *
    child.states.effective_prediction_precision *
    child.states.precision_prediction_error
end

function calculate_posterior_precision_increment(
    node::ContinuousStateNode,
    child::ContinuousInputNode,
    coupling_type::NoiseCoupling,
)

    #If the input node child had a missing input
    if ismissing(child.states.input_value)
        #No increment
        return 0
    else
        update_term =
            1 / 2 * (child.parameters.coupling_strengths[node.name])^2 +
            child.states.precision_prediction_error *
            (child.parameters.coupling_strengths[node.name])^2 -
            1 / 2 *
            child.parameters.coupling_strengths[node.name]^2 *
            child.states.precision_prediction_error

        return update_term
    end
end

##### Mean update #####
@doc raw"""
    calculate_posterior_mean(node::AbstractNode)

Calculates a node's posterior mean.

Uses the equation
``   ``
"""
function calculate_posterior_mean(node::ContinuousStateNode, update_type::HGFUpdateType)

    #Initialize as the prediction
    posterior_mean = node.states.prediction_mean

    #Add update terms from drift children
    for child in node.edges.drift_children
        posterior_mean +=
            calculate_posterior_mean_increment(node, child, DriftCoupling(), update_type)
    end

    #Add update terms from observation children
    for child in node.edges.observation_children
        posterior_mean += calculate_posterior_mean_increment(
            node,
            child,
            ObservationCoupling(),
            update_type,
        )
    end

    #Add update terms from probability children
    for child in node.edges.probability_children
        posterior_mean += calculate_posterior_mean_increment(
            node,
            child,
            ProbabilityCoupling(),
            update_type,
        )
    end

    #Add update terms from volatility children
    for child in node.edges.volatility_children
        posterior_mean += calculate_posterior_mean_increment(
            node,
            child,
            VolatilityCoupling(),
            update_type,
        )
    end

    #Add update terms from noise children
    for child in node.edges.noise_children
        posterior_mean +=
            calculate_posterior_mean_increment(node, child, NoiseCoupling(), update_type)
    end

    return posterior_mean
end

## Classic drift coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::DriftCoupling,
    update_type::ClassicUpdate,
)
    (
        (
            child.parameters.coupling_strengths[node.name] *
            transform_parent_value(
                node.states.posterior_mean,
                child.parameters.coupling_transforms[node.name];
                derivation_level = 1,
                parent = node,
                child = child,
            ) *
            child.states.prediction_precision
        ) / node.states.posterior_precision
    ) * child.states.value_prediction_error
end

## Enhanced drift coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::DriftCoupling,
    update_type::EnhancedUpdate,
)
    (
        (
            child.parameters.coupling_strengths[node.name] *
            transform_parent_value(
                node.states.posterior_mean,
                child.parameters.coupling_transforms[node.name];
                derivation_level = 1,
                parent = node,
                child = child,
            ) *
            child.states.prediction_precision
        ) / node.states.prediction_precision
    ) * child.states.value_prediction_error

end

## Classic observation coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousInputNode,
    coupling_type::ObservationCoupling,
    update_type::ClassicUpdate,
)
    #For input node children with missing input
    if ismissing(child.states.input_value)
        #No update
        return 0
    else
        return (child.states.prediction_precision / node.states.posterior_precision) *
               child.states.value_prediction_error
    end
end

## Enhanced observation coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousInputNode,
    coupling_type::ObservationCoupling,
    update_type::EnhancedUpdate,
)
    #For input node children with missing input
    if ismissing(child.states.input_value)
        #No update
        return 0
    else
        return (child.states.prediction_precision / node.states.prediction_precision) *
               child.states.value_prediction_error
    end
end

## Classic probability coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::BinaryStateNode,
    coupling_type::ProbabilityCoupling,
    update_type::ClassicUpdate,
)
    #If the posterior is missing (due to missing inputs)
    if ismissing(child.states.posterior_mean)
        #No update
        return 0
    else
        return child.parameters.coupling_strengths[node.name] /
               node.states.posterior_precision * child.states.value_prediction_error
    end
end

## Enhanced Probability coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::BinaryStateNode,
    coupling_type::ProbabilityCoupling,
    update_type::EnhancedUpdate,
)
    #If the posterior is missing (due to missing inputs)
    if ismissing(child.states.posterior_mean)
        #No update
        return 0
    else
        return child.parameters.coupling_strengths[node.name] /
               node.states.prediction_precision * child.states.value_prediction_error
    end
end

## Classic Volatility coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::VolatilityCoupling,
    update_type::ClassicUpdate,
)
    1 / 2 * (
        child.parameters.coupling_strengths[node.name] *
        child.states.effective_prediction_precision
    ) / node.states.posterior_precision * child.states.precision_prediction_error
end

## Enhanced Volatility coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::VolatilityCoupling,
    update_type::EnhancedUpdate,
)
    1 / 2 * (
        child.parameters.coupling_strengths[node.name] *
        child.states.effective_prediction_precision
    ) / node.states.prediction_precision * child.states.precision_prediction_error
end

## Classic Noise coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousInputNode,
    coupling_type::NoiseCoupling,
    update_type::ClassicUpdate,
)
    #For input node children with missing input
    if ismissing(child.states.input_value)
        #No update
        return 0
    else
        update_term =
            1 / 2 * (child.parameters.coupling_strengths[node.name]) /
            node.states.posterior_precision * child.states.precision_prediction_error

        return update_term
    end
end

## Enhanced Noise coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousInputNode,
    coupling_type::NoiseCoupling,
    update_type::EnhancedUpdate,
)
    #For input node children with missing input
    if ismissing(child.states.input_value)
        #No update
        return 0
    else
        update_term =
            1 / 2 * (child.parameters.coupling_strengths[node.name]) /
            node.states.prediction_precision * child.states.precision_prediction_error

        return update_term
    end
end

###############################################
######## Update value prediction error ########
###############################################

##### Superfunction #####
"""
    update_node_value_prediction_error!(node::AbstractStateNode)

Update the value prediction error of a single state node.
"""
function update_node_value_prediction_error!(node::ContinuousStateNode)
    #Update value prediction error
    node.states.value_prediction_error = calculate_value_prediction_error(node)

    return nothing
end

@doc raw"""
    calculate_value_prediction_error(node::AbstractNode)

Calculate's a state node's value prediction error.

Uses the equation
`` \delta_n = \mu_n - \hat{\mu}_n  ``
"""
function calculate_value_prediction_error(node::ContinuousStateNode)
    node.states.posterior_mean - node.states.prediction_mean
end

###################################################
######## Update precision prediction error ########
###################################################

##### Superfunction #####
"""
    update_node_precision_prediction_error!(node::AbstractStateNode)

Update the volatility prediction error of a single state node.
"""
function update_node_precision_prediction_error!(node::ContinuousStateNode)

    #Update volatility prediction error, only if there are volatility parents
    node.states.precision_prediction_error = calculate_precision_prediction_error(node)

    return nothing
end

@doc raw"""
    calculate_precision_prediction_error(node::AbstractNode)

Calculates a state node's volatility prediction error.

Uses the equation
`` \Delta_n = \frac{\hat{\pi}_n}{\pi_n} + \hat{\pi}_n \cdot \delta_n^2-1  ``
"""
function calculate_precision_prediction_error(node::ContinuousStateNode)

    #If there are no volatility parents
    if length(node.edges.volatility_parents) == 0
        #Skip
        return missing
    end

    node.states.prediction_precision / node.states.posterior_precision +
    node.states.prediction_precision * node.states.value_prediction_error^2 - 1

end
