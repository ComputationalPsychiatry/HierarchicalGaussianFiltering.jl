###################################
######## Update prediction ########
###################################

#Minimum for the predicted variance, to avoid underflow
const MIN_PREDICTED_VARIANCE = 1e-128

##### Superfunction #####
"""
    update_node_prediction!(node::ContinuousStateNode)

Update the prediction of a single state node.
"""
function update_node_prediction!(node::ContinuousStateNode, stepsize::Real)

    #Store the posterior precision from the last timestep, before the posterior is updated.
    node.states.previous_posterior_precision = node.states.posterior_precision

    #Update prediction mean
    node.states.prediction_mean = calculate_prediction_mean(node, stepsize)

    #Update prediction precision
    node.states.prediction_precision, node.states.effective_prediction_precision =
        calculate_prediction_precision(node, stepsize)

    return nothing
end

##### Mean update #####
"""
    calculate_prediction_mean(node::AbstractNode)

Calculates a node's prediction mean.
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
"""
    calculate_prediction_precision(node::AbstractNode)

Calculates a node's prediction precision.

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

    #Add the log of the stepsize, to get the log of the predicted volatility. Use this to avoid over/underflow
    log_predicted_volatility = log(stepsize) + predicted_volatility

    #Exponentiate to get the predicted volatility
    predicted_volatility = exp(log_predicted_volatility)

    #Carried variance: the mean prediction multiplies the belief by λ, so its
    #variance is multiplied by λ².
    carried_variance =
        node.parameters.autoconnection_strength^2 / node.states.posterior_precision

    #Calculate prediction precision. The denominator is floored so that λ = 0
    #with Ω → 0 yields Inf cleanly instead of a NaN/overflow.
    prediction_precision =
        1 / max(carried_variance + predicted_volatility, MIN_PREDICTED_VARIANCE)

    #Calculate the volatility-weighted effective precision, which is the proportion of the
    #predicted variance that comes from the predicted volatility. 
    effective_prediction_precision =
        logistic(log_predicted_volatility - log(carried_variance))

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
    update_node_posterior!(node::AbstractStateNode, update_type::UnboundedUpdate)

Update the posterior of a single continuous state node. This is the unbounded HGF update (Mathys et al., 2026), which guarantees a positive posterior precision for volatility parents.
"""
function update_node_posterior!(node::ContinuousStateNode, update_type::UnboundedUpdate)

    #If the node has no volatility children, the unbounded update is identical to the classic update
    if length(node.edges.volatility_children) == 0
        update_node_posterior!(node, ClassicUpdate())
        return nothing
    end

    #Update posterior precision with the update terms from all children except the volatility children.
    #This gives the effective prior precision for the unbounded update.
    node.states.posterior_precision = calculate_posterior_precision(node, update_type)

    #Calculate the posterior mean with the update terms from all children except the volatility children.
    #This gives the effective prior mean for the unbounded update. It is not stored yet, since the previous posterior mean is still needed.
    prior_mean = calculate_posterior_mean(node, update_type)

    #Update posterior precision and mean jointly with the update terms from the volatility children
    node.states.posterior_precision, node.states.posterior_mean =
        calculate_unbounded_posterior(node, prior_mean)

    return nothing
end

##### Precision update #####
"""
    calculate_posterior_precision(node::AbstractNode)

Calculates a node's posterior precision.
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
        posterior_precision += calculate_posterior_precision_increment(
            node,
            child,
            VolatilityCoupling(),
            update_type,
        )
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

"""
    calculate_posterior_precision_vope(node::AbstractNode, child::AbstractNode)

Calculates the posterior precision update term for a single continuous volatility child to a state node.
"""
function calculate_posterior_precision_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::VolatilityCoupling,
    update_type::Union{ClassicUpdate,EnhancedUpdate},
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
    child::ContinuousStateNode,
    coupling_type::VolatilityCoupling,
    update_type::UnboundedUpdate,
)
    #The volatility children are handled jointly by the unbounded update, so there is no separate increment
    return 0
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
"""
    calculate_posterior_mean(node::AbstractNode)

Calculates a node's posterior mean.
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

## Classic and unbounded drift coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::DriftCoupling,
    update_type::Union{ClassicUpdate,UnboundedUpdate},
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

## Classic and unbounded observation coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousInputNode,
    coupling_type::ObservationCoupling,
    update_type::Union{ClassicUpdate,UnboundedUpdate},
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

## Classic and unbounded probability coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::BinaryStateNode,
    coupling_type::ProbabilityCoupling,
    update_type::Union{ClassicUpdate,UnboundedUpdate},
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

## Unbounded Volatility coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    coupling_type::VolatilityCoupling,
    update_type::UnboundedUpdate,
)
    #The volatility children are handled jointly by the unbounded update, so there is no separate increment
    return 0
end

## Classic and unbounded Noise coupling ##
function calculate_posterior_mean_increment(
    node::ContinuousStateNode,
    child::ContinuousInputNode,
    coupling_type::NoiseCoupling,
    update_type::Union{ClassicUpdate,UnboundedUpdate},
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

##### Unbounded update #####
"""
    calculate_unbounded_posterior(node::ContinuousStateNode, prior_mean::Real)

Calculates the posterior precision and mean of a continuous state node with volatility children, using the unbounded HGF update (Mathys et al., 2026).
The node's variational energy is approximated with quadratic expansions: a first expansion at the prior mean, which uses only the concave part of the variational energy and therefore always has a positive precision, and a second expansion for each volatility child, at the approximate secondary mode of the variational energy which that child can create, found with the Lambert W function.
The expansions are weighted with a softmax over the variational energy at their means, and the posterior is the moment-matched Gaussian of the resulting mixture, so that the posterior precision is positive by construction.
The effective prior, which includes the update terms from all children that are not volatility children, is given by the node's stored posterior precision and the passed prior mean.
"""
function calculate_unbounded_posterior(node::ContinuousStateNode, prior_mean::Real)

    #Extract the effective prior precision, which has been stored as the posterior precision
    prior_precision = node.states.posterior_precision

    #Make the first expansion, at the prior mean
    first_expansion_precision, first_expansion_mean =
        calculate_unbounded_first_expansion(node, prior_precision, prior_mean)

    #Collect the precisions and means of the expansions, starting with the first expansion
    expansion_precisions = Real[first_expansion_precision]
    expansion_means = Real[first_expansion_mean]

    #For each volatility child
    for child in node.edges.volatility_children

        #Make a second expansion, at the approximate secondary mode which that child can create
        second_expansion =
            calculate_unbounded_second_expansion(node, child, prior_precision, prior_mean)

        #If the expansion could be made
        if !isnothing(second_expansion)
            #Add it to the list of expansions
            push!(expansion_precisions, second_expansion[1])
            push!(expansion_means, second_expansion[2])
        end
    end

    #Calculate the variational energy at the mean of each expansion
    variational_energies = [
        calculate_variational_energy(node, expansion_mean, prior_precision, prior_mean)
        for expansion_mean in expansion_means
    ]

    #Weight the expansions with a softmax over their variational energies, so that expansions at points of higher variational energy get higher weights.
    #The maximum is subtracted first, so that the exponentials cannot overflow.
    expansion_weights = exp.(variational_energies .- maximum(variational_energies))
    expansion_weights = expansion_weights ./ sum(expansion_weights)

    #Gaussian mixture moment matching: the posterior mean is the weighted mean of the expansion means
    posterior_mean = sum(expansion_weights .* expansion_means)

    #The posterior variance is the weighted mean of the expansion variances, plus the variance between the expansion means
    posterior_variance = sum(
        expansion_weights .*
        (1 ./ expansion_precisions .+ (expansion_means .- posterior_mean) .^ 2),
    )

    #The posterior precision is the inverse of the posterior variance
    posterior_precision = 1 / posterior_variance

    return posterior_precision, posterior_mean
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

"""
    calculate_value_prediction_error(node::AbstractNode)

Calculate's a state node's value prediction error.
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

"""
    calculate_precision_prediction_error(node::AbstractNode)

Calculates a state node's volatility prediction error.
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
