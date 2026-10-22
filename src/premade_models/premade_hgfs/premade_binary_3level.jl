
"""
    premade_binary_3level(config::Dict; verbose::Bool = true)

The standard binary 3 level HGF model, which takes a binary input, and learns the probability of either outcome.
It has one binary input node u, with a binary value parent xbin, which in turn has a continuous value parent xprob. This then has a continunous volatility parent xvol.

This HGF has five shared parameters: 
"xprob_volatility"
"xprob_initial_precisions"
"xprob_initial_means"
"coupling_strengths_xbin_xprob"
"coupling_strengths_xprob_xvol"

# Config defaults:
 - ("xprob", "volatility"): -2
 - ("xvol", "volatility"): -2
 - ("xbin", "xprob", "coupling_strength"): 1
 - ("xprob", "xvol", "coupling_strength"): 1
 - ("xprob", "initial_mean"): 0
 - ("xprob", "initial_precision"): 1
 - ("xvol", "initial_mean"): 0
 - ("xvol", "initial_precision"): 1
"""
function premade_binary_3level(config::Dict; verbose::Bool = true)

    #Defaults
    defaults = Dict(
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
        ContinuousState(
            name = "xvol",
            volatility = config[("xvol", "volatility")],
            drift = config[("xvol", "drift")],
            autoconnection_strength = config[("xvol", "autoconnection_strength")],
            initial_mean = config[("xvol", "initial_mean")],
            initial_precision = config[("xvol", "initial_precision")],
        ),
    ]

    #List of child-parent relations
    edges = Dict(
        ("u", "xbin") => ObservationCoupling(),
        ("xbin", "xprob") =>
            ProbabilityCoupling(config[("xbin", "xprob", "coupling_strength")]),
        ("xprob", "xvol") =>
            VolatilityCoupling(config[("xprob", "xvol", "coupling_strength")]),
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
