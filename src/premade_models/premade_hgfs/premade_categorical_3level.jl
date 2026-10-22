

"""
    premade_categorical_3level(config::Dict; verbose::Bool = true)

The categorical 3 level HGF model, which takes an input from one of n categories and learns the probability of a category appearing.
It has one categorical input node u, with a categorical value parent xcat.
The categorical node has a binary value parent xbin_n for each category n, each of which has a continuous value parent xprob_n.
Finally, all of these continuous nodes share a continuous volatility parent xvol. 
Setting parameter values for xbin and xprob sets that parameter value for each of the xbin_n and xprob_n nodes.

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
function premade_categorical_3level(config::Dict; verbose::Bool = true)

    #Defaults
    defaults = Dict(
        "n_categories" => 4,
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


    ##Prep category node parent names
    #Vector for category node binary parent names
    category_parent_names = Vector{String}()
    #Vector for binary node continuous parent names
    probability_parent_names = Vector{String}()

    #Empty lists for grouped parameters
    grouped_parameters_xprob_initial_precision = []
    grouped_parameters_xprob_initial_mean = []
    grouped_parameters_xprob_volatility = []
    grouped_parameters_xprob_drift = []
    grouped_parameters_xprob_autoconnection_strength = []
    grouped_parameters_xbin_xprob_coupling_strength = []
    grouped_parameters_xprob_xvol_coupling_strength = []

    #Populate the category node vectors with node names
    for category_number = 1:config["n_categories"]
        push!(category_parent_names, "xbin_" * string(category_number))
        push!(probability_parent_names, "xprob_" * string(category_number))
    end

    #Initialize list of nodes
    nodes = [CategoricalInput("u"), CategoricalState("xcat")]

    #Add category node binary parents
    for node_name in category_parent_names
        push!(nodes, BinaryState(node_name))
    end

    #Add binary node continuous parents
    for node_name in probability_parent_names
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

    #Add volatility parent
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

    ##List of edges
    #Set the input node coupling
    edges = Dict{Tuple{String,String},CouplingType}(("u", "xcat") => ObservationCoupling())

    #For each set of categroy parents and their probability parents
    for (category_parent_name, probability_parent_name) in
        zip(category_parent_names, probability_parent_names)

        #Connect the binary category parents to the categorical state node
        edges[("xcat", category_parent_name)] = CategoryCoupling()

        #Connect each category parent to its probability parent
        edges[(category_parent_name, probability_parent_name)] =
            ProbabilityCoupling(config[("xbin", "xprob", "coupling_strength")])

        #Connect the probability parents to the shared volatility parent
        edges[(probability_parent_name, "xvol")] =
            VolatilityCoupling(config[("xprob", "xvol", "coupling_strength")])

        #Add the coupling strengths to the lists of grouped parameters
        push!(
            grouped_parameters_xbin_xprob_coupling_strength,
            (category_parent_name, probability_parent_name, "coupling_strength"),
        )
        push!(
            grouped_parameters_xprob_xvol_coupling_strength,
            (probability_parent_name, "xvol", "coupling_strength"),
        )
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
        ParameterGroup(
            "xprob_xvol_coupling_strength",
            grouped_parameters_xprob_xvol_coupling_strength,
            config[("xprob", "xvol", "coupling_strength")],
        ),
    ]

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
