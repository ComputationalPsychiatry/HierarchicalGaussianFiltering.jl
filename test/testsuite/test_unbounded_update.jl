using HierarchicalGaussianFiltering
using Test
using CSV
using DataFrames

@testset "Unbounded update" begin

    ### Setup ###
    #Get the path for the HGF superfolder
    hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))
    #Add the path to the data files
    data_path = hgf_path * "/test/testsuite/data/"

    @testset "Default update type" begin

        #Create a premade HGF with default settings
        hgf = premade_hgf("continuous_2level", verbose = false)

        #Volatility parents use the unbounded update by default, other nodes the classic update
        @test hgf.state_nodes["xvol"].update_type isa UnboundedUpdate
        @test hgf.state_nodes["x"].update_type isa ClassicUpdate

        #The same holds for custom HGFs
        nodes = [
            ContinuousInput(name = "u"),
            ContinuousState(name = "x"),
            ContinuousState(name = "xvol"),
        ]
        edges =
            Dict(("u", "x") => ObservationCoupling(), ("x", "xvol") => VolatilityCoupling())
        hgf = init_hgf(nodes = nodes, edges = edges, verbose = false)
        @test hgf.state_nodes["xvol"].update_type isa UnboundedUpdate
        @test hgf.state_nodes["x"].update_type isa ClassicUpdate

        #Other update types can still be chosen
        hgf = premade_hgf(
            "continuous_2level",
            Dict("update_type" => EnhancedUpdate()),
            verbose = false,
        )
        @test hgf.state_nodes["xvol"].update_type isa EnhancedUpdate
    end

    @testset "Previous posterior precision" begin

        #Create a premade HGF and give it inputs
        hgf = premade_hgf("continuous_2level", verbose = false)
        multiple_inputs!(hgf, [0.01, 0.05, 0.01, missing, 0.02])

        #Before the first input, there is no previous posterior precision
        @test ismissing(hgf.state_nodes["x"].history.previous_posterior_precision[1])

        #At each timestep, the stored previous posterior precision is the posterior precision from the last timestep
        for node_name in ["x", "xvol"]
            @test hgf.state_nodes[node_name].history.previous_posterior_precision[2:end] ==
                  hgf.state_nodes[node_name].history.posterior_precision[1:(end-1)]
        end

        #Resetting the HGF removes it again
        reset!(hgf)
        @test ismissing(hgf.state_nodes["x"].states.previous_posterior_precision)
    end

    @testset "Canonical continuous 2level" begin

        #The target trajectory was generated with the reference implementation of the unbounded HGF
        #(UnboundedHGF.jl, accompanying Mathys et al., 2026), on the canonical input series
        input_trajectory_path = data_path * "canonical_continuous2level_inputs.dat"
        canonical_trajectory_path =
            data_path * "canonical_continuous2level_unbounded_states.csv"

        #Import the input trajectory
        input_trajectory = Float64[]
        open(input_trajectory_path) do f
            for ln in eachline(f)
                push!(input_trajectory, parse(Float64, ln))
            end
        end

        #Import the target trajectory
        target_outputs = CSV.read(canonical_trajectory_path, DataFrame)

        #Set parameters and starting states (as in the canonical test), with the unbounded update
        parameters = Dict(
            ("u", "input_noise") => log(1e-4),
            ("x", "volatility") => -13,
            ("x", "initial_mean") => 1.04,
            ("x", "initial_precision") => 1e4,
            ("xvol", "volatility") => -2,
            ("xvol", "initial_mean") => 1.0,
            ("xvol", "initial_precision") => 10,
            ("x", "xvol", "coupling_strength") => 1.0,
            "update_type" => UnboundedUpdate(),
        )

        #Create HGF
        test_hgf = premade_hgf("continuous_2level", parameters, verbose = false)

        #Give inputs
        multiple_inputs!(test_hgf, input_trajectory)

        #Construct result output dataframe
        result_outputs = DataFrame(
            x_mean = test_hgf.state_nodes["x"].history.posterior_mean,
            x_precision = test_hgf.state_nodes["x"].history.posterior_precision,
            xvol_mean = test_hgf.state_nodes["xvol"].history.posterior_mean,
            xvol_precision = test_hgf.state_nodes["xvol"].history.posterior_precision,
        )

        #Test if the values are approximately the same
        @test nrow(result_outputs) == nrow(target_outputs)
        for i = 1:nrow(result_outputs)
            @test result_outputs.x_mean[i] ≈ target_outputs.mu1[i]
            @test result_outputs.x_precision[i] ≈ target_outputs.pi1[i]
            @test result_outputs.xvol_mean[i] ≈ target_outputs.mu2[i]
            @test result_outputs.xvol_precision[i] ≈ target_outputs.pi2[i]
        end
    end

    @testset "Robustness where the classic update fails" begin

        #An input series with large regime changes, under high volatility and meta-volatility
        inputs =
            vcat(zeros(60), fill(30.0, 60), fill(-40.0, 60), zeros(60), fill(80.0, 60)) .+
            0.5 .* sin.(1:300)
        config = Dict(("x", "volatility") => 2, ("xvol", "volatility") => 2)

        #The classic update produces a negative posterior precision, and rejects the parameters
        hgf_classic = premade_hgf(
            "continuous_2level",
            merge(config, Dict("update_type" => ClassicUpdate())),
            verbose = false,
        )
        @test_throws RejectParameters multiple_inputs!(hgf_classic, inputs)

        #The unbounded update completes, with positive and finite posterior precisions and finite posterior means
        hgf_unbounded = premade_hgf(
            "continuous_2level",
            merge(config, Dict("update_type" => UnboundedUpdate())),
            verbose = false,
        )
        multiple_inputs!(hgf_unbounded, inputs)
        xvol = hgf_unbounded.state_nodes["xvol"]
        @test length(xvol.history.posterior_mean) == length(inputs) + 1
        @test all(x -> isfinite(x) && x > 0, xvol.history.posterior_precision)
        @test all(isfinite, xvol.history.posterior_mean)
    end

    @testset "Multiple volatility children" begin

        #A volatility parent with two volatility children, which have different coupling strengths
        nodes = [
            ContinuousInput(name = "u1"),
            ContinuousInput(name = "u2"),
            ContinuousState(name = "x1"),
            ContinuousState(name = "x2"),
            ContinuousState(name = "xvol"),
        ]
        edges = Dict(
            ("u1", "x1") => ObservationCoupling(),
            ("u2", "x2") => ObservationCoupling(),
            ("x1", "xvol") => VolatilityCoupling(),
            ("x2", "xvol") => VolatilityCoupling(2),
        )
        hgf = init_hgf(nodes = nodes, edges = edges, verbose = false)
        @test hgf.state_nodes["xvol"].update_type isa UnboundedUpdate

        #Give inputs with regime changes to both input nodes
        inputs = [0.0 0.0; 0.1 -0.1; 20.0 0.2; 20.5 -30.0; 21.0 -30.2; -5.0 10.0; -5.2 10.1]
        multiple_inputs!(hgf, inputs)

        #The posterior precisions are positive and finite
        @test all(
            x -> isfinite(x) && x > 0,
            hgf.state_nodes["xvol"].history.posterior_precision,
        )
        @test all(isfinite, hgf.state_nodes["xvol"].history.posterior_mean)
    end
end
