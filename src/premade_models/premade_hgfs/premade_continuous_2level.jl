"""
    premade_continuous_2level(config::Dict; verbose::Bool = true)

The standard 2 level continuous HGF, which filters a continuous input.
It has a continous input node u, with a single value parent x, which in turn has a single volatility parent xvol.

# Config defaults:
 - ("u", "input_noise"): -2
 - ("x", "volatility"): -2
 - ("xvol", "volatility"): -2
 - ("u", "x", "coupling_strength"): 1
 - ("x", "xvol", "coupling_strength"): 1
 - ("x", "initial_mean"): 0
 - ("x", "initial_precision"): 1
 - ("xvol", "initial_mean"): 0
 - ("xvol", "initial_precision"): 1
"""
function premade_continuous_2level(config::Dict; verbose::Bool = true)

    #Defaults
    spec_defaults = Dict(
        ("u", "input_noise") => -2,
        ("u", "bias") => 0,
        ("x", "volatility") => -2,
        ("x", "drift") => 0,
        ("x", "autoconnection_strength") => 1,
        ("x", "initial_mean") => 0,
        ("x", "initial_precision") => 1,
        ("xvol", "volatility") => -2,
        ("xvol", "drift") => 0,
        ("xvol", "autoconnection_strength") => 1,
        ("xvol", "initial_mean") => 0,
        ("xvol", "initial_precision") => 1,
        ("x", "xvol", "coupling_strength") => 1,
        "update_type" => UnboundedUpdate(),
        "save_history" => true,
    )

    #Merge to overwrite defaults
    config = merge(spec_defaults, config)

    #List of nodes
    nodes = [
        ContinuousInput(
            name = "u",
            input_noise = config[("u", "input_noise")],
            bias = config[("u", "bias")],
        ),
        ContinuousState(
            name = "x",
            volatility = config[("x", "volatility")],
            drift = config[("x", "drift")],
            autoconnection_strength = config[("x", "autoconnection_strength")],
            initial_mean = config[("x", "initial_mean")],
            initial_precision = config[("x", "initial_precision")],
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
        ("u", "x") => ObservationCoupling(),
        ("x", "xvol") => VolatilityCoupling(config[("x", "xvol", "coupling_strength")]),
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
