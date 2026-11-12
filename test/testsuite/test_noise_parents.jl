using HierarchicalGaussianFiltering
using Test

@testset "Noise parents" begin

    ### Setup ###
    #Inputs whose spread changes over time, so that the noise parent has something to track
    inputs = [sin(k / 5) + (1 + 0.8 * sin(k / 17)) * sin(1.7 * k) for k = 1:200]

    #Parameters of an input with a value parent x and a noise parent q
    parameters = (
        input_noise = -1.0,
        bias = 0.3,
        x_volatility = -2.0,
        q_volatility = -3.0,
        noise_coupling = 1.4,
        x_initial_mean = 0.0,
        x_initial_precision = 1.0,
        q_initial_mean = 0.5,
        q_initial_precision = 2.0,
    )

    #Create an HGF with an input u, its value parent x and its noise parent q
    function create_noise_hgf(update_type)
        nodes = [
            ContinuousInput(name = "u", input_noise = parameters.input_noise),
            ContinuousState(
                name = "x",
                volatility = parameters.x_volatility,
                initial_mean = parameters.x_initial_mean,
                initial_precision = parameters.x_initial_precision,
            ),
            ContinuousState(
                name = "q",
                volatility = parameters.q_volatility,
                initial_mean = parameters.q_initial_mean,
                initial_precision = parameters.q_initial_precision,
            ),
        ]
        edges = Dict(
            ("u", "x") => ObservationCoupling(),
            ("u", "q") => NoiseCoupling(strength = parameters.noise_coupling),
        )
        hgf = init_hgf(
            nodes = nodes,
            edges = edges,
            node_defaults = NodeDefaults(update_type = update_type),
            verbose = false,
        )

        #Set the bias of the input, so that the noise prediction error is tested with a bias
        set_parameters!(hgf, ("u", "bias"), parameters.bias)

        return hgf
    end

    #The same model, written directly from the equations of Weber et al. (2026):
    #the predictions of x and q, the noise precision of the input (Eq. 66), the update of the value parent,
    #the noise prediction error (Eq. 72), and the update of the noise parent (Eqs. 74 and 78)
    function reference_noise_filter(inputs)
        μ_x, π_x = parameters.x_initial_mean, parameters.x_initial_precision
        μ_q, π_q = parameters.q_initial_mean, parameters.q_initial_precision
        κ = parameters.noise_coupling
        trajectory = (μ_x = Float64[], π_x = Float64[], μ_q = Float64[], π_q = Float64[])

        for u in inputs
            #Predictions of the two parents, which have no parents themselves
            μ̂_x, π̂_x = μ_x, 1 / (1 / π_x + exp(parameters.x_volatility))
            μ̂_q, π̂_q = μ_q, 1 / (1 / π_q + exp(parameters.q_volatility))

            #Prediction of the input, whose noise is set by the noise parent
            μ̂_u = parameters.bias + μ̂_x
            π̂_u = exp(-(parameters.input_noise + κ * μ̂_q))

            #Update of the value parent
            π_x = π̂_x + π̂_u
            μ_x = μ̂_x + π̂_u / π_x * (u - μ̂_u)

            #Noise prediction error, against the updated value parent
            ϵ = π̂_u * (1 / π_x + (u - parameters.bias - μ_x)^2) - 1

            #Update of the noise parent
            π_q = π̂_q + 1 / 2 * κ^2 * (1 + ϵ)
            μ_q = μ̂_q + 1 / 2 * κ * ϵ / π_q

            push!(trajectory.μ_x, μ_x)
            push!(trajectory.π_x, π_x)
            push!(trajectory.μ_q, μ_q)
            push!(trajectory.π_q, π_q)
        end

        return trajectory
    end

    @testset "Follows Weber et al. (2026) under $(typeof(update_type))" for update_type in [
        ClassicUpdate(),
        EnhancedUpdate(),
        UnboundedUpdate(),
    ]

        #Run the HGF and the reference
        hgf = create_noise_hgf(update_type)
        multiple_inputs!(hgf, inputs)
        reference = reference_noise_filter(inputs)

        #The noise parent takes the classic update whatever the default update type
        @test hgf.state_nodes["q"].update_type isa ClassicUpdate

        #The trajectories match the reference (the first history entry is the prior)
        @test hgf.state_nodes["x"].history.posterior_mean[2:end] ≈ reference.μ_x rtol =
            1e-12
        @test hgf.state_nodes["x"].history.posterior_precision[2:end] ≈ reference.π_x rtol =
            1e-12
        @test hgf.state_nodes["q"].history.posterior_mean[2:end] ≈ reference.μ_q rtol =
            1e-12
        @test hgf.state_nodes["q"].history.posterior_precision[2:end] ≈ reference.π_q rtol =
            1e-12
    end

    @testset "Noise parent that is also a volatility parent" begin

        #q is a noise parent of u and a volatility parent of x
        nodes = [
            ContinuousInput(name = "u"),
            ContinuousState(name = "x"),
            ContinuousState(name = "q"),
        ]
        edges = Dict(
            ("u", "x") => ObservationCoupling(),
            ("u", "q") => NoiseCoupling(),
            ("x", "q") => VolatilityCoupling(),
        )

        #It takes the default update type from its volatility coupling
        for update_type in [ClassicUpdate(), EnhancedUpdate(), UnboundedUpdate()]
            hgf = init_hgf(
                nodes = nodes,
                edges = edges,
                node_defaults = NodeDefaults(update_type = update_type),
                verbose = false,
            )
            @test hgf.state_nodes["q"].update_type isa typeof(update_type)

            #And its updates stay finite
            multiple_inputs!(hgf, inputs[1:50])
            @test all(isfinite, hgf.state_nodes["q"].history.posterior_mean)
            @test all(>(0), hgf.state_nodes["q"].history.posterior_precision)
        end
    end

    @testset "Missing input" begin

        #With a missing input, the noise parent keeps its prediction
        hgf = create_noise_hgf(ClassicUpdate())
        multiple_inputs!(hgf, [0.2, missing, 0.4])
        q = hgf.state_nodes["q"].history
        @test ismissing(hgf.input_nodes["u"].history.precision_prediction_error[3])
        @test q.posterior_mean[3] == q.prediction_mean[3]
        @test q.posterior_precision[3] == q.prediction_precision[3]
    end
end
