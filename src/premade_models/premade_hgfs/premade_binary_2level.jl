
"""
    premade_binary_2level(config::Dict; verbose::Bool = true)

The standard binary 2 level HGF model, which takes a binary input, and learns the probability of either outcome.
It has one binary input node u, with a binary value parent xbin, which in turn has a continuous value parent xprob.

# Config defaults:
"""
function premade_binary_2level(config::Dict; verbose::Bool = true)

    #Defaults
    spec_defaults = Dict(
        ("xprob", "volatility") => -2,
        ("xprob", "drift") => 0,
        ("xprob", "autoconnection_strength") => 1,
        ("xprob", "initial_mean") => 0,
        ("xprob", "initial_precision") => 1,
        ("xbin", "xprob", "coupling_strength") => 1,
        "update_type" => UnboundedUpdate(),
        "save_history" => true,
    )

    #Merge to overwrite defaults
    config = merge(spec_defaults, config)

    #List of nodes
    nodes = [
        BinaryInput("u"),
        BinaryState("xbin"),
        ContinuousState(
            name = "xprob",
            volatility = config[("xprob", "volatility")],
            drift = config[("xprob", "drift")],
            autoconnection_strength = config[("xprob", "autoconnection_strength")],
            initial_mean = config[("xprob", "initial_mean")],
            initial_precision = config[("xprob", "initial_precision")],
        ),
    ]

    #List of edges
    edges = Dict(
        ("u", "xbin") => ObservationCoupling(),
        ("xbin", "xprob") =>
            ProbabilityCoupling(config[("xbin", "xprob", "coupling_strength")]),
    )

    #Initialize the HGF
    init_hgf(
        nodes = nodes,
        edges = edges,
        verbose = false,
        node_defaults = NodeDefaults(update_type = config["update_type"]),
        save_history = config["save_history"],
    )
end
