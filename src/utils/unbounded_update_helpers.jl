##### Unbounded update helpers #####
"""
    calculate_unbounded_first_expansion(node::ContinuousStateNode, prior_precision::Real, prior_mean::Real)

Makes the first quadratic expansion of the unbounded HGF update, at the prior mean. The precision uses only the concave part of the variational energy, so it is always positive.
Returns the precision and mean of the expansion.
This function is only used in the unbounded HGF update.
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
This function is only used in the unbounded HGF update.
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
This function is only used in the unbounded HGF update.
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
This function is only used in the unbounded HGF update.
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
This function is only used in the unbounded HGF update.
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
This function is only used in the unbounded HGF update.
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

##### Lambert W function #####
"""
    lambert_w0_from_log(log_argument::Real)

The principal branch W₀ of the Lambert W function, which solves w * exp(w) = z for z ≥ 0, given the log of its argument z.
Taking the log of the argument means that z itself never has to be represented, so it can be far above the largest representable number.
Uses Halley's method, which converges to machine precision in a few iterations. The iteration is written in terms of the residual w - exp(log(z) - w), so that no intermediate value can overflow.
This function is only used in the unbounded HGF update.
"""
function lambert_w0_from_log(log_argument::Real)

    #For very small arguments, W₀(z) = z to machine precision
    if log_argument < -36
        return exp(log_argument)
    end

    #Initial guess log(z) - log(log(1 + z)), which is close to the solution for all arguments
    w = log_argument - log(log1pexp(log_argument))

    #Halley iterations
    for _ = 1:8

        #The residual, which is zero at the solution
        residual = w - exp(log_argument - w)

        #Halley step
        w -= 2 * residual * (1 + w) / (2 * (1 + w)^2 - residual * (2 + w))
    end

    return w
end
