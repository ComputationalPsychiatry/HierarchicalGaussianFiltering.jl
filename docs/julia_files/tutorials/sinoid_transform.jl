### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 5ae02fee-48f1-11ef-209c-539e778f577d
using HierarchicalGaussianFiltering, Distributions, StatsPlots

# ╔═╡ ff1abe62-b477-4c34-927d-4cc758981d60
nodes = [
    ContinuousInput(name = "u"),
    ContinuousState(name = "x1"),
    ContinuousState(name = "x2"),
]

# ╔═╡ b53c65b6-24fb-4df0-85e8-c94e973f92d1
sine_transform = NonlinearTransform(
    function (x, parameters::Dict)
        sin(x)
    end, #base function
    function (x, parameters::Dict)
        cos(x)
    end, #first derivative
    function (x, parameters::Dict)
        -sin(x)
    end, #second derivative
    Dict(), #no parameters
)

# ╔═╡ c1e30a10-e9f9-4348-92b5-85fdc0be00b0
begin
    edges_linear =
        Dict(("u", "x1") => ObservationCoupling(), ("x1", "x2") => DriftCoupling())

    edges_nonlinear = Dict(
        ("u", "x1") => ObservationCoupling(),
        ("x1", "x2") => DriftCoupling(1, sine_transform),
    )

end

# ╔═╡ ec6ceba5-d202-4345-bce8-fc0dada2a019
begin
    hgf_linear = init_hgf(nodes = nodes, edges = edges_linear, verbose = false)
    hgf_nonlinear = init_hgf(nodes = nodes, edges = edges_nonlinear, verbose = false)
end

# ╔═╡ d2e95dab-7312-4835-b780-91571ba83239
parameters = Dict(
    ("x1", "autoconnection_strength") => 0,
    ("x1", "volatility") => -4,
    ("x2", "volatility") => -4,
    ("u", "input_noise") => log(0.25),
)

# ╔═╡ e319c480-cb6d-41fd-a4ec-892a12b95a06
begin
    set_parameters!(hgf_linear, parameters)

    set_parameters!(hgf_nonlinear, parameters)
end

# ╔═╡ cb170251-4dce-4e75-b457-40f2fb959a5b
begin

    sample_rate = 20

    inputs = sin.(collect(0:1/sample_rate:35))

    inputs = rand(Normal(0, 0.25), length(inputs)) + inputs
    plot(inputs)
end

# ╔═╡ 847373ac-647d-43a3-93a1-92af9772dce0
begin
    reset!(hgf_linear)
    give_inputs!(hgf_linear, inputs)
    reset!(hgf_nonlinear)
    give_inputs!(hgf_nonlinear, inputs)
end

# ╔═╡ c7f92940-18c8-4212-8808-8b29d9b870a1
plot_settings = (; label = "", title = "")

# ╔═╡ 5835d516-f822-4128-900b-16d09b550cb7

plot(
    plot_trajectory(hgf_linear, "u"; plot_settings..., title = "inputs"),
    plot_trajectory(
        hgf_linear,
        ("u", "prediction");
        plot_settings...,
        title = "u prediction",
    ),
    plot_trajectory(hgf_linear, ("x1"); plot_settings..., title = "x1 posterior"),
    plot_trajectory(hgf_linear, ("x2"); plot_settings..., title = "x2 posterior"),
)

# ╔═╡ f1ad8c91-7642-4478-8091-bbcecc9f2bcf

plot(
    plot_trajectory(hgf_nonlinear, "u"; plot_settings..., title = "inputs"),
    plot_trajectory(
        hgf_nonlinear,
        ("u", "prediction");
        plot_settings...,
        title = "u prediction",
    ),
    plot_trajectory(hgf_nonlinear, ("x1"); plot_settings..., title = "x1 posterior"),
    plot_trajectory(hgf_nonlinear, ("x2"); plot_settings..., title = "x2 posterior"),
)

# ╔═╡ 50330bc3-e74d-4cfc-908a-a56acbcdd2bc
begin
    μ₁_linear = get_history(hgf_linear, ("x1", "posterior_mean"))

    #Band mu1 between -1 and 1 (the noise makes it occasionally jump over)
    μ₁_linear = min.(μ₁_linear, 1.0)
    μ₁_linear = max.(μ₁_linear, -1.0)

    μ₂_linear = get_history(hgf_linear, ("x2", "posterior_mean"))



    μ₁_nonlinear = get_history(hgf_nonlinear, ("x1", "posterior_mean"))

    #Band mu1 between -1 and 1 (the noise makes it occasionally jump over)
    μ₁_nonlinear = min.(μ₁_nonlinear, 1.0)
    μ₁_nonlinear = max.(μ₁_nonlinear, -1.0)

    μ₂_nonlinear = get_history(hgf_nonlinear, ("x2", "posterior_mean"))
end

# ╔═╡ fa51e10d-3ab3-487c-90bc-9362956f563a
#Plot sine transformed x2 against x1 - should be equal
plot(
    plot(sin.(μ₂_linear) - μ₁_linear, title = "linear"),
    plot(sin.(μ₂_nonlinear) - μ₁_nonlinear, title = "nonlinear"),
)

# ╔═╡ c52a0167-e780-47fc-901c-44cf84b973d4
#Plot asine transformed x1 against x2 - should be equal
plot(
    plot(asin.(μ₁_linear) - μ₂_linear, title = "linear"),
    plot(asin.(μ₁_nonlinear) - μ₂_nonlinear, title = "nonlinear"),
)
#The linear fares worse

# ╔═╡ af36ec40-ca4f-4aff-ae35-439adc5cae89
begin
    linear_plt = plot(asin.(μ₁_linear), label = "μ₁ asin")
    plot!(μ₂_linear, label = "μ₂", title = "linear")

    nonlinear_plt = plot(asin.(μ₁_nonlinear), label = "μ₁ asin")
    plot!(μ₂_nonlinear, label = "μ₂", title = "nonlinear")

    plot(linear_plt, nonlinear_plt)
end
