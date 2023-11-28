"""
premade_JGET(config::Dict; verbose::Bool = true)

The HGF used in the JGET model. It has a single continuous input node u, with a value parent x, and a volatility parent xnoise. x has volatility parent xvol, and xnoise has a volatility parent xnoise_vol.

# Config defaults:
 - ("u", "input_noise"): -2
 - ("x", "volatility"): -2
 - ("xvol", "volatility"): -2
 - ("xnoise", "volatility"): -2
 - ("xnoise_vol", "volatility"): -2
 - ("u", "x", "coupling_strength"): 1
 - ("u", "xnoise", "coupling_strength"): 1
 - ("x", "xvol", "coupling_strength"): 1
 - ("xnoise", "xnoise_vol", "coupling_strength"): 1
 - ("x", "initial_mean"): 0
 - ("x", "initial_precision"): 1
 - ("xvol", "initial_mean"): 0
 - ("xvol", "initial_precision"): 1
 - ("xnoise", "initial_mean"): 0
 - ("xnoise", "initial_precision"): 1
 - ("xnoise_vol", "initial_mean"): 0
 - ("xnoise_vol", "initial_precision"): 1
"""
function premade_JGET(config::Dict; verbose::Bool = true)

    #Defaults
    spec_defaults = Dict(
        ("u", "input_noise") => -2,
        ("x", "volatility") => -2,
        ("x", "drift") => 0,
        ("x", "autoregression_target") => 0,
        ("x", "autoregression_strength") => 0,
        ("x", "initial_mean") => 0,
        ("x", "initial_precision") => 1,
        ("xvol", "volatility") => -2,
        ("xvol", "drift") => 0,
        ("xvol", "autoregression_target") => 0,
        ("xvol", "autoregression_strength") => 0,
        ("xvol", "initial_mean") => 0,
        ("xvol", "initial_precision") => 1,
        ("xnoise", "volatility") => -2,
        ("xnoise", "drift") => 0,
        ("xnoise", "autoregression_target") => 0,
        ("xnoise", "autoregression_strength") => 0,
        ("xnoise", "initial_mean") => 0,
        ("xnoise", "initial_precision") => 1,
        ("xnoise_vol", "volatility") => -2,
        ("xnoise_vol", "drift") => 0,
        ("xnoise_vol", "autoregression_target") => 0,
        ("xnoise_vol", "autoregression_strength") => 0,
        ("xnoise_vol", "initial_mean") => 0,
        ("xnoise_vol", "initial_precision") => 1,
        ("u", "xnoise", "coupling_strength") => 1,
        ("x", "xvol", "coupling_strength") => 1,
        ("xnoise", "xnoise_vol", "coupling_strength") => 1,
        "update_type" => EnhancedUpdate(),
    )

    #Warn the user about used defaults and misspecified keys
    if verbose
        warn_premade_defaults(spec_defaults, config)
    end

    #Merge to overwrite defaults
    config = merge(spec_defaults, config)

    #List of nodes
    nodes = [
        ContinuousInput(name = "u", input_noise = config[("u", "input_noise")]),
        ContinuousState(
            name = "x",
            volatility = config[("x", "volatility")],
            drift = config[("x", "drift")],
            autoregression_target = config[("x", "autoregression_target")],
            autoregression_strength = config[("x", "autoregression_strength")],
            initial_mean = config[("x", "initial_mean")],
            initial_precision = config[("x", "initial_precision")],
        ),
        ContinuousState(
            name = "xvol",
            volatility = config[("xvol", "volatility")],
            drift = config[("xvol", "drift")],
            autoregression_target = config[("xvol", "autoregression_target")],
            autoregression_strength = config[("xvol", "autoregression_strength")],
            initial_mean = config[("xvol", "initial_mean")],
            initial_precision = config[("xvol", "initial_precision")],
        ),
        ContinuousState(
            name = "xnoise",
            volatility = config[("xnoise", "volatility")],
            drift = config[("xnoise", "drift")],
            autoregression_target = config[("xnoise", "autoregression_target")],
            autoregression_strength = config[("xnoise", "autoregression_strength")],
            initial_mean = config[("xnoise", "initial_mean")],
            initial_precision = config[("xnoise", "initial_precision")],
        ),
        ContinuousState(
            name = "xnoise_vol",
            volatility = config[("xnoise_vol", "volatility")],
            drift = config[("xnoise_vol", "drift")],
            autoregression_target = config[("xnoise_vol", "autoregression_target")],
            autoregression_strength = config[("xnoise_vol", "autoregression_strength")],
            initial_mean = config[("xnoise_vol", "initial_mean")],
            initial_precision = config[("xnoise_vol", "initial_precision")],
        ),
    ]

    #List of child-parent relations
    edges = Dict(
        ("u", "x") => ObservationCoupling(),
        ("u", "xnoise") => NoiseCoupling(config[("u", "xnoise", "coupling_strength")]),
        ("x", "xvol") => VolatilityCoupling(config[("x", "xvol", "coupling_strength")]),
        ("xnoise", "xnoise_vol") =>
            VolatilityCoupling(config[("xnoise", "xnoise_vol", "coupling_strength")]),
    )

    #Initialize the HGF
    init_hgf(
        nodes = nodes,
        edges = edges,
        verbose = false,
        node_defaults = NodeDefaults(update_type = config["update_type"]),
    )
end
