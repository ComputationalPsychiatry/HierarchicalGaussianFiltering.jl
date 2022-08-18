using HGF
using Test

@testset "Initialization" begin
    #Parameter values to be used for all nodes unless other values are given
    node_defaults = (;
        evolution_rate = 3,
        category_means = [0, 1],
        input_precision = Inf,
        initial_mean = 1,
        initial_precision = 2,
        value_coupling = 1,
    )

    #List of input nodes to create
    input_nodes = [(name = "u1", params = (; evolution_rate = 2)), "u2"]

    #List of state nodes to create
    state_nodes = [
        "x1",
        "x2",
        "x3",
        (name = "x4", params = (; evolution_rate = 2)),
        (
            name = "x5",
            params = (; evolution_rate = 2, initial_mean = 4, initial_precision = 3),
        ),
    ]

    #List of child-parent relations
    edges = [
        (child_node = "u1", value_parents = "x1"),
        (child_node = "u2", value_parents = "x2", volatility_parents = ["x3"]),
        (
            child_node = "x1",
            value_parents = (name = "x3", value_coupling = 2),
            volatility_parents = [(name = "x4", volatility_coupling = 2), "x5"],
        ),
    ]

    #Initialize an HGF
    test_hgf = HGF.init_hgf(node_defaults, input_nodes, state_nodes, edges, verbose = false)

    @testset "Check if inputs were placed the right places" begin
        @test test_hgf.input_nodes["u1"].params.evolution_rate == 2
        @test test_hgf.input_nodes["u2"].params.evolution_rate == 3

        @test test_hgf.state_nodes["x1"].params.evolution_rate == 3
        @test test_hgf.state_nodes["x2"].params.evolution_rate == 3
        @test test_hgf.state_nodes["x3"].params.evolution_rate == 3
        @test test_hgf.state_nodes["x4"].params.evolution_rate == 2
        @test test_hgf.state_nodes["x5"].params.evolution_rate == 2

        @test test_hgf.input_nodes["u1"].params.value_coupling["x1"] == 1
        @test test_hgf.input_nodes["u2"].params.value_coupling["x2"] == 1
        @test test_hgf.input_nodes["u2"].params.volatility_coupling["x3"] == 1
        @test test_hgf.state_nodes["x1"].params.value_coupling["x3"] == 2
        @test test_hgf.state_nodes["x1"].params.volatility_coupling["x4"] == 2
        @test test_hgf.state_nodes["x1"].params.volatility_coupling["x5"] == 1

        @test test_hgf.state_nodes["x1"].states.posterior_mean == 1
        @test test_hgf.state_nodes["x1"].states.posterior_precision == 2
        @test test_hgf.state_nodes["x2"].states.posterior_mean == 1
        @test test_hgf.state_nodes["x2"].states.posterior_precision == 2
        @test test_hgf.state_nodes["x3"].states.posterior_mean == 1
        @test test_hgf.state_nodes["x3"].states.posterior_precision == 2
        @test test_hgf.state_nodes["x4"].states.posterior_mean == 1
        @test test_hgf.state_nodes["x4"].states.posterior_precision == 2
        @test test_hgf.state_nodes["x5"].states.posterior_mean == 4
        @test test_hgf.state_nodes["x5"].states.posterior_precision == 3
    end

    @testset "check warnings for unspecified output" begin
        @test_logs (
            :warn,
            "node parameter volatility_coupling is not specified in node_defaults. Using 1 as default.",
        ) HGF.init_hgf(node_defaults, input_nodes, state_nodes, edges)
    end
end
