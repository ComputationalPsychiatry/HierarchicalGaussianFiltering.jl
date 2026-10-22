
"""
    premade_categorical_state_transitions(config::Dict; verbose::Bool = true)

The categorical state transition 3 level HGF model, learns state transition probabilities between a set of n categorical states.
It has one categorical input node u, with a categorical value parent xcat_n for each of the n categories, representing which category was transitioned from.
Each categorical node then has a binary parent xbin_n_m, representing the category m which the transition was towards.
Each binary node xbin_n_m has a continuous parent xprob_n_m. 
Finally, all of these continuous nodes share a continuous volatility parent xvol. 
Setting parameter values for xbin and xprob sets that parameter value for each of the xbin_n_m and xprob_n_m nodes.

This HGF has five shared parameters: 
"xprob_volatility"
"xprob_initial_precisions"
"xprob_initial_means"
"coupling_strengths_xbin_xprob"
"coupling_strengths_xprob_xvol"

# Config defaults:
    - "n_categories": 4
    - ("xprob", "volatility"): -2
    - ("xvol", "volatility"): -2
    - ("xbin", "xprob", "coupling_strength"): 1
    - ("xprob", "xvol", "coupling_strength"): 1
    - ("xprob", "initial_mean"): 0
    - ("xprob", "initial_precision"): 1
    - ("xvol", "initial_mean"): 0
    - ("xvol", "initial_precision"): 1
"""
function premade_categorical_state_transitions(config::Dict; verbose::Bool = true)

    #Defaults
    defaults = Dict(
        "n_categories_from" => 4,
        "n_categories_to" => 4,
        "include_volatility_parent" => true,
        ("xprob", "volatility") => -2,
        ("xprob", "drift") => 0,
        ("xprob", "autoconnection_strength") => 1,
        ("xprob", "initial_mean") => 0,
        ("xprob", "initial_precision") => 1,
        ("xvol", "volatility") => -2,
        ("xvol", "drift") => 0,
        ("xvol", "autoconnection_strength") => 1,
        ("xvol", "initial_mean") => 0,
        ("xvol", "initial_precision") => 1,
        ("xbin", "xprob", "coupling_strength") => 1,
        ("xprob", "xvol", "coupling_strength") => 1,
        "update_type" => UnboundedUpdate(),
        "save_history" => true,
    )

    #Merge to overwrite defaults
    config = merge(defaults, config)


    ##Prepare node names
    #Empty lists for node names
    input_node_names = Vector{String}()
    observation_parent_names = Vector{String}()
    category_parent_names = Vector{String}()
    probability_parent_names = Vector{String}()

    #Empty lists for grouped parameters
    grouped_parameters_xprob_initial_precision = []
    grouped_parameters_xprob_initial_mean = []
    grouped_parameters_xprob_volatility = []
    grouped_parameters_xprob_drift = []
    grouped_parameters_xprob_autoconnection_strength = []
    grouped_parameters_xbin_xprob_coupling_strength = []
    grouped_parameters_xprob_xvol_coupling_strength = []

    #Go through each category that the transition may have been from
    for category_from = 1:config["n_categories_from"]
        #One input node and its state node parent for each                             
        push!(input_node_names, "u" * string(category_from))
        push!(observation_parent_names, "xcat_" * string(category_from))
        #Go through each category that the transition may have been to
        for category_to = 1:config["n_categories_to"]
            #Each categorical state node has a binary parent for each
            push!(
                category_parent_names,
                "xbin_" * string(category_from) * "_" * string(category_to),
            )
            #And each binary parent has a continuous parent of its own
            push!(
                probability_parent_names,
                "xprob_" * string(category_from) * "_" * string(category_to),
            )
        end
    end

    ##List of nodes
    nodes = Vector{AbstractNodeInfo}()

    #For each categorical input node
    for node_name in input_node_names
        #Add it to the list
        push!(nodes, CategoricalInput(node_name))
    end

    #For each categorical state node
    for node_name in observation_parent_names
        #Add it to the list
        push!(nodes, CategoricalState(node_name))
    end

    #For each categorical node binary parent
    for node_name in category_parent_names
        #Add it to the list                                    
        push!(nodes, BinaryState(node_name))
    end

    #For each binary node continuous parent
    for node_name in probability_parent_names
        #Add it to the list, with parameter settings from the config
        push!(
            nodes,
            ContinuousState(
                name = node_name,
                volatility = config[("xprob", "volatility")],
                drift = config[("xprob", "drift")],
                autoconnection_strength = config[("xprob", "autoconnection_strength")],
                initial_mean = config[("xprob", "initial_mean")],
                initial_precision = config[("xprob", "initial_precision")],
            ),
        )
        #Add the grouped parameter name to grouped parameters vector
        push!(grouped_parameters_xprob_initial_precision, (node_name, "initial_precision"))
        push!(grouped_parameters_xprob_initial_mean, (node_name, "initial_mean"))
        push!(grouped_parameters_xprob_volatility, (node_name, "volatility"))
        push!(grouped_parameters_xprob_drift, (node_name, "drift"))
        push!(
            grouped_parameters_xprob_autoconnection_strength,
            (node_name, "autoconnection_strength"),
        )
    end

    #If volatility parent is included
    if config["include_volatility_parent"]
        #Add the shared volatility parent of the continuous nodes
        push!(
            nodes,
            ContinuousState(
                name = "xvol",
                volatility = config[("xvol", "volatility")],
                drift = config[("xvol", "drift")],
                autoconnection_strength = config[("xvol", "autoconnection_strength")],
                initial_mean = config[("xvol", "initial_mean")],
                initial_precision = config[("xvol", "initial_precision")],
            ),
        )

    end


    ##Create edges
    #Initialize list                                                     
    edges = Dict{Tuple{String,String},CouplingType}()

    #For each categorical input node and its corresponding state node parent
    for (input_node_name, observation_parent_name) in
        zip(input_node_names, observation_parent_names)

        #Add their connection
        edges[(input_node_name, observation_parent_name)] = ObservationCoupling()
    end

    #For each categorical state node
    for observation_parent_name in observation_parent_names
        #Get the category it represents transitions from
        (observation_parent_prefix, child_category_from) =
            split(observation_parent_name, "_")

        #For each potential parent node
        for category_parent_name in category_parent_names
            #Get the category it represents transitions to and from
            (category_parent_prefix, parent_category_from, parent_category_to) =
                split(category_parent_name, "_")

            #If these match
            if parent_category_from == child_category_from
                #Add their connection
                edges[(observation_parent_name, category_parent_name)] = CategoryCoupling()
            end
        end
    end

    #For each set of category parent and probability parent
    for (category_parent_name, probability_parent_name) in
        zip(category_parent_names, probability_parent_names)

        #Connect them
        edges[(category_parent_name, probability_parent_name)] =
            ProbabilityCoupling(config[("xbin", "xprob", "coupling_strength")])

        #If there is a volatility parent
        if config["include_volatility_parent"]
            #Connect the probability parents to the shared volatility parent
            edges[(probability_parent_name, "xvol")] =
                VolatilityCoupling(config[("xprob", "xvol", "coupling_strength")])
        end

        #Add the parameters as grouped parameters for shared parameters
        push!(
            grouped_parameters_xbin_xprob_coupling_strength,
            (category_parent_name, probability_parent_name, "coupling_strength"),
        )

        #If volatility parent is included
        if config["include_volatility_parent"]
            push!(
                grouped_parameters_xprob_xvol_coupling_strength,
                (probability_parent_name, "xvol", "coupling_strength"),
            )
        end

    end

    #Create dictionary with shared parameter information

    parameter_groups = [
        ParameterGroup(
            "xprob_volatility",
            grouped_parameters_xprob_volatility,
            config[("xprob", "volatility")],
        ),
        ParameterGroup(
            "xprob_initial_precision",
            grouped_parameters_xprob_initial_precision,
            config[("xprob", "initial_precision")],
        ),
        ParameterGroup(
            "xprob_initial_mean",
            grouped_parameters_xprob_initial_mean,
            config[("xprob", "initial_mean")],
        ),
        ParameterGroup(
            "xprob_drift",
            grouped_parameters_xprob_drift,
            config[("xprob", "drift")],
        ),
        ParameterGroup(
            "xprob_autoconnection_strength",
            grouped_parameters_xprob_autoconnection_strength,
            config[("xprob", "autoconnection_strength")],
        ),
        ParameterGroup(
            "xbin_xprob_coupling_strength",
            grouped_parameters_xbin_xprob_coupling_strength,
            config[("xbin", "xprob", "coupling_strength")],
        ),
    ]

    #If volatility parent is included
    if config["include_volatility_parent"]
        push!(
            parameter_groups,
            ParameterGroup(
                "xprob_xvol_coupling_strength",
                grouped_parameters_xprob_xvol_coupling_strength,
                config[("xprob", "xvol", "coupling_strength")],
            ),
        )
    end

    #Initialize the HGF
    init_hgf(
        nodes = nodes,
        edges = edges,
        parameter_groups = parameter_groups,
        verbose = false,
        node_defaults = NodeDefaults(update_type = config["update_type"]),
        save_history = config["save_history"],
    )
end
