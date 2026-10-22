###################################
######## Update prediction ########
###################################

#Floor on the total predicted variance (carried + volatility): at λ = 0 the
#carried term vanishes, so without it the precision can diverge as Ω → 0; at
#the default λ = 1 the sum already exceeds it, so it never binds.
const MIN_PREDICTED_VARIANCE = 1e-128

##### Superfunction #####
"""
    update_node_prediction!(node::ContinuousStateNode)

Update the prediction of a single state node.
"""
function update_node_prediction!(node::ContinuousStateNode, stepsize::Real)

    #Store the posterior precision from the last timestep, before the posterior is updated.
    #The unbounded HGF update of volatility parents needs it, since it gives the variance
    #that is carried over from the last posterior into this node's prediction.
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

    #Add the log of the stepsize, to get the log of the predicted volatility. This is kept,
    #since the exponentiated version can over- or underflow.
    log_predicted_volatility = log(stepsize) + predicted_volatility

    #Exponentiate to get the predicted volatility
    predicted_volatility = exp(log_predicted_volatility)

    #Carried variance: the mean prediction multiplies the belief by λ, so its
    #variance is multiplied by λ² (unchanged at λ = 1, zero at λ = 0). Only the
    #node's own variance takes the factor; the drift is added after, unscaled.
    carried_variance =
        node.parameters.autoconnection_strength^2 / node.states.posterior_precision

    #Calculate prediction precision. The denominator is floored so that λ = 0
    #with Ω → 0 yields Inf cleanly instead of a NaN/overflow; at λ = 1 the
    #denominator already exceeds the floor, so it never binds.
    prediction_precision =
        1 / max(carried_variance + predicted_volatility, MIN_PREDICTED_VARIANCE)

    #Calculate the volatility-weighted effective precision, which is the proportion of the
    #predicted variance that comes from the predicted volatility. It is calculated as a
    #logistic function in log-space, so that it stays finite (saturating at 0 or 1) when the
    #predicted volatility under- or overflows, instead of becoming NaN.
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

"""
    calculate_unbounded_first_expansion(node::ContinuousStateNode, prior_precision::Real, prior_mean::Real)

Makes the first quadratic expansion of the unbounded HGF update, at the prior mean. The precision uses only the concave part of the variational energy, so it is always positive.
Returns the precision and mean of the expansion.
"""
function calculate_unbounded_first_expansion(
    node::ContinuousStateNode,
    prior_precision::Real,
    prior_mean::Real,
)

    #Initialize the precision as the prior precision, and the mean increment as zero
    expansion_precision = prior_precision
    mean_increment = 0

    #For each volatility child
    for child in node.edges.volatility_children

        #Get the coupling strength
        coupling_strength = child.parameters.coupling_strengths[node.name]

        #Get the child's effective prediction precision and precision prediction error given the prior mean
        effective_prediction_precision, precision_prediction_error, _ =
            calculate_volatility_coupling_terms(node, child, prior_mean)

        #Add the concave part of the curvature of the variational energy
        expansion_precision +=
            1 / 2 *
            coupling_strength^2 *
            effective_prediction_precision *
            (1 - effective_prediction_precision)

        #Add the gradient of the variational energy
        mean_increment +=
            1 / 2 *
            coupling_strength *
            effective_prediction_precision *
            precision_prediction_error
    end

    #The mean is the prior mean plus the precision-weighted gradient
    expansion_mean = prior_mean + mean_increment / expansion_precision

    return expansion_precision, expansion_mean
end

"""
    calculate_unbounded_second_expansion(node::ContinuousStateNode, child::ContinuousStateNode, prior_precision::Real, prior_mean::Real)

Makes the second quadratic expansion of the unbounded HGF update for a single volatility child, at the approximate secondary mode of the variational energy which that child can create. The full variational energy, with the contributions from all volatility children, is expanded at this point.
Returns the precision and mean of the expansion, or nothing if the expansion is not finite.
"""
function calculate_unbounded_second_expansion(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    prior_precision::Real,
    prior_mean::Real,
)

    #Find the expansion point
    expansion_point =
        calculate_unbounded_expansion_point(node, child, prior_precision, prior_mean)

    #If the expansion point is not finite, the expansion cannot be made
    if !isfinite(expansion_point)
        return nothing
    end

    #Initialize the precision as the prior precision, both with the full curvature and with only its concave part, and the mean increment as zero
    expansion_precision = prior_precision
    concave_expansion_precision = prior_precision
    mean_increment = 0

    #For each volatility child
    for volatility_child in node.edges.volatility_children

        #Get the coupling strength
        coupling_strength = volatility_child.parameters.coupling_strengths[node.name]

        #Get the child's effective prediction precision and precision prediction error given the expansion point
        effective_prediction_precision, precision_prediction_error, _ =
            calculate_volatility_coupling_terms(node, volatility_child, expansion_point)

        #Add the full curvature of the variational energy
        expansion_precision +=
            1 / 2 *
            coupling_strength^2 *
            effective_prediction_precision *
            (
                effective_prediction_precision +
                (2 * effective_prediction_precision - 1) * precision_prediction_error
            )

        #Add the concave part of the curvature of the variational energy
        concave_expansion_precision +=
            1 / 2 *
            coupling_strength^2 *
            effective_prediction_precision *
            (1 - effective_prediction_precision)

        #Add the gradient of the variational energy
        mean_increment +=
            1 / 2 *
            coupling_strength *
            effective_prediction_precision *
            precision_prediction_error
    end

    #If the precision is not positive (which can happen if the expansion point is far from the actual mode), use only the concave part of the curvature instead
    if expansion_precision <= 0
        expansion_precision = concave_expansion_precision
    end

    #The mean is the expansion point plus the precision-weighted gradient, which includes the gradient of the prior
    expansion_mean =
        expansion_point +
        (mean_increment - prior_precision * (expansion_point - prior_mean)) /
        expansion_precision

    #If the expansion is not finite, it cannot be used
    if !(isfinite(expansion_precision) && isfinite(expansion_mean))
        return nothing
    end

    return expansion_precision, expansion_mean
end

"""
    calculate_unbounded_expansion_point(node::ContinuousStateNode, child::ContinuousStateNode, prior_precision::Real, prior_mean::Real)

Finds the expansion point for the second expansion of the unbounded HGF update: the approximate secondary mode of the variational energy created by a volatility child, in the limit where the variance carried over from the child's previous posterior goes to zero.
The mode is found in closed form with the Lambert W function, in the canonical variable y = log(t) + κx + ω, which is the log of the child's predicted volatility.
"""
function calculate_unbounded_expansion_point(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    prior_precision::Real,
    prior_mean::Real,
)

    #Get the coupling strength
    coupling_strength = child.parameters.coupling_strengths[node.name]

    #The total posterior uncertainty about the child: its posterior variance plus its squared value prediction error
    total_uncertainty =
        1 / child.states.posterior_precision + child.states.value_prediction_error^2

    #The log of the child's predicted volatility given the prior mean, which is the prior mean in the canonical variable
    log_predicted_volatility = calculate_log_predicted_volatility(node, child, prior_mean)

    #The prior precision in the canonical variable
    canonical_prior_precision = prior_precision / coupling_strength^2

    #The mode equation is solved by the Lambert W function. Its argument is kept in log-space, since it can be far above the largest representable number.
    log_lambert_argument =
        log(total_uncertainty) - log(2 * canonical_prior_precision) +
        1 / (2 * canonical_prior_precision) - log_predicted_volatility

    #The mode in the canonical variable
    canonical_mode =
        log_predicted_volatility - 1 / (2 * canonical_prior_precision) +
        lambert_w0_from_log(log_lambert_argument)

    #Convert back to the node's own variable
    expansion_point =
        prior_mean + (canonical_mode - log_predicted_volatility) / coupling_strength

    return expansion_point
end

"""
    calculate_volatility_coupling_terms(node::ContinuousStateNode, child::ContinuousStateNode, node_value::Real)

For a volatility child, calculates the terms of the unbounded HGF update which depend on the node's value: the child's effective prediction precision (the proportion of its predicted variance that comes from the predicted volatility), its precision prediction error, and its contribution to the node's variational energy.
These are calculated in log-space, so that they stay finite when the predicted volatility under- or overflows.
"""
function calculate_volatility_coupling_terms(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    node_value::Real,
)

    #The log of the variance carried over from the child's previous posterior into its prediction (scaled by the autoconnection strength).
    #The variance is floored, so that it can be log-transformed when the autoconnection strength is zero.
    log_carried_variance = log(
        max(
            child.parameters.autoconnection_strength^2 /
            child.states.previous_posterior_precision,
            MIN_PREDICTED_VARIANCE,
        ),
    )

    #The log of the child's predicted volatility given the node's value
    log_predicted_volatility = calculate_log_predicted_volatility(node, child, node_value)

    #The log of the child's total predicted variance
    log_predicted_variance = logaddexp(log_carried_variance, log_predicted_volatility)

    #The total posterior uncertainty about the child: its posterior variance plus its squared value prediction error
    total_uncertainty =
        1 / child.states.posterior_precision + child.states.value_prediction_error^2

    #The effective prediction precision, as a logistic function so that it saturates at 0 or 1
    effective_prediction_precision =
        logistic(log_predicted_volatility - log_carried_variance)

    #The precision prediction error
    precision_prediction_error = total_uncertainty * exp(-log_predicted_variance) - 1

    #The child's contribution to the node's variational energy
    variational_energy =
        -1 / 2 * log_predicted_variance -
        1 / 2 * total_uncertainty * exp(-log_predicted_variance)

    return effective_prediction_precision, precision_prediction_error, variational_energy
end

"""
    calculate_log_predicted_volatility(node::ContinuousStateNode, child::ContinuousStateNode, node_value::Real)

Calculates the log of a volatility child's predicted volatility (including the stepsize) as a function of the node's value.
The part which does not depend on the node is recovered from the child's prediction, which was made with the node's previous posterior mean.
"""
function calculate_log_predicted_volatility(
    node::ContinuousStateNode,
    child::ContinuousStateNode,
    node_value::Real,
)

    #Get the coupling strength
    coupling_strength = child.parameters.coupling_strengths[node.name]

    #The log of the predicted volatility at the node's previous posterior mean, recovered from the child's prediction (the effective prediction precision is the predicted volatility times the prediction precision)
    log_predicted_volatility =
        log(child.states.effective_prediction_precision) -
        log(child.states.prediction_precision)

    #Move from the node's previous posterior mean to the given value
    log_predicted_volatility +=
        coupling_strength * (node_value - node.states.posterior_mean)

    return log_predicted_volatility
end

"""
    calculate_variational_energy(node::ContinuousStateNode, node_value::Real, prior_precision::Real, prior_mean::Real)

Calculates the variational energy of a continuous state node with volatility children, at a given value of the node (up to a constant). This is the quadratic prior term plus the contributions from each volatility child.
"""
function calculate_variational_energy(
    node::ContinuousStateNode,
    node_value::Real,
    prior_precision::Real,
    prior_mean::Real,
)

    #Initialize with the prior term
    variational_energy = -1 / 2 * prior_precision * (node_value - prior_mean)^2

    #Add the contribution from each volatility child
    for child in node.edges.volatility_children
        variational_energy +=
            calculate_volatility_coupling_terms(node, child, node_value)[3]
    end

    return variational_energy
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
