"""
    reset!(hgf::HGF)

Reset an HGF to its initial state.
"""
function ActionModels.reset!(hgf::HGF)

    #Reset the timesteps for the HGF
    hgf.timesteps = [0]

    #Go through each node
    for node in hgf.ordered_nodes.all_nodes

        #Reset its state
        reset_state!(node)

        #For each state in the history
        for state_name in fieldnames(typeof(node.history))

            #Empty the history
            empty!(getfield(node.history, state_name))

            #Add the new current state as the first state in the history
            push!(getfield(node.history, state_name), getfield(node.states, state_name))
        end
    end
end


function reset_state!(node::ContinuousStateNode)

    node.states.posterior_mean = node.parameters.initial_mean
    node.states.posterior_precision = node.parameters.initial_precision

    node.states.prev_posterior_precision = node.parameters.initial_precision

    node.states.value_prediction_error = missing
    node.states.precision_prediction_error = missing

    node.states.prediction_mean = missing
    node.states.prediction_precision = missing
    node.states.effective_prediction_precision = missing

    return nothing
end

function reset_state!(node::ContinuousInputNode)

    node.states.input_value = missing

    node.states.value_prediction_error = missing
    node.states.precision_prediction_error = missing

    node.states.prediction_mean = missing
    node.states.prediction_precision = missing

    return nothing
end

function reset_state!(node::BinaryStateNode)

    node.states.posterior_mean = missing
    node.states.posterior_precision = missing

    node.states.value_prediction_error = missing

    node.states.prediction_mean = missing
    node.states.prediction_precision = missing

    return nothing
end

function reset_state!(node::BinaryInputNode)

    node.states.input_value = missing

    return nothing
end

function reset_state!(node::CategoricalStateNode)

    node.states.posterior .= missing
    node.states.value_prediction_error .= missing
    node.states.prediction .= 1/length(node.states.prediction)
    node.states.parent_predictions .= 1/length(node.states.parent_predictions)

    return nothing
end

function reset_state!(node::CategoricalInputNode)

    node.states.input_value = missing

    return nothing
end
