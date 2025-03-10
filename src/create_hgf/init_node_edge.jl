### Function for initializing a node ###
function init_node(node_info::ContinuousState)
    ContinuousStateNode(
        name = node_info.name,
        parameters = ContinuousStateNodeParameters(
            volatility = node_info.volatility,
            drift = node_info.drift,
            initial_mean = node_info.initial_mean,
            initial_precision = node_info.initial_precision,
            autoconnection_strength = node_info.autoconnection_strength,
        ),
    )
end

function init_node(node_info::ContinuousInput)
    ContinuousInputNode(
        name = node_info.name,
        parameters = ContinuousInputNodeParameters(input_noise = node_info.input_noise),
    )
end

function init_node(node_info::BinaryState)
    BinaryStateNode(name = node_info.name)
end

function init_node(node_info::BinaryInput)
    BinaryInputNode(name = node_info.name)
end

function init_node(node_info::CategoricalState)
    CategoricalStateNode(name = node_info.name)
end

function init_node(node_info::CategoricalInput)
    CategoricalInputNode(name = node_info.name)
end


### Function for initializing an edge ###
function init_edge!(
    child_node::AbstractNode,
    parent_node::AbstractStateNode,
    coupling_type::CouplingType,
    node_defaults::NodeDefaults,
)

    #Get correct field for storing parents
    if coupling_type isa DriftCoupling
        parents_field = :drift_parents
        children_field = :drift_children

    elseif coupling_type isa ObservationCoupling
        parents_field = :observation_parents
        children_field = :observation_children

    elseif coupling_type isa CategoryCoupling
        parents_field = :category_parents
        children_field = :category_children

    elseif coupling_type isa ProbabilityCoupling
        parents_field = :probability_parents
        children_field = :probability_children

    elseif coupling_type isa VolatilityCoupling
        parents_field = :volatility_parents
        children_field = :volatility_children

    elseif coupling_type isa NoiseCoupling
        parents_field = :noise_parents
        children_field = :noise_children
    end

    #Add the parent to the child node
    push!(getfield(child_node.edges, parents_field), parent_node)

    #Add the child node to the parent node
    push!(getfield(parent_node.edges, children_field), child_node)

    #If the coupling type has a coupling strength
    if hasproperty(coupling_type, :strength)
        #If the user has not specified a coupling strength
        if isnothing(coupling_type.strength)
            #Use the defaults coupling strength
            coupling_strength = node_defaults.coupling_strength

            #Otherwise
        else
            #Use the specified coupling strength 
            coupling_strength = coupling_type.strength
        end

        #And set it as a parameter for the child
        child_node.parameters.coupling_strengths[parent_node.name] = coupling_strength
    end

    #If the coupling can have a transformation
    if hasproperty(coupling_type, :transform)
        #Save that transformation in the node
        child_node.parameters.coupling_transforms[parent_node.name] =
            coupling_type.transform
    end

    #If the enhanced HGF update is the defaults, and if it is a precision coupling (volatility or noise)
    if node_defaults.update_type isa EnhancedUpdate && coupling_type isa PrecisionCoupling
        #Set the node to use the enhanced HGF update
        parent_node.update_type = node_defaults.update_type
    end
    #If the enhanced HGF update is the defaults, and if it is a precision coupling (volatility or noise)
    if node_defaults.update_type isa UniversalUpdate && coupling_type isa PrecisionCoupling
        #Set the node to use the enhanced HGF update
        parent_node.update_type = node_defaults.update_type
    end
end
