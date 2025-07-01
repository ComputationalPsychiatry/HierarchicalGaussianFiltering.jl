export HGFSigmoid

Base.@kwdef struct HGFSigmoid <: ActionModels.AbstractPremadeModel
    action_noise::Float64 = 1.0
    target_state::Symbol = :xbin_prediction_mean
    HGF::Union{HGF,String} = "continuous_2level"
end

function ActionModels.ActionModel(config::HGFSigmoid)

    #Extract hgf
    if config.HGF isa String
        #If the HGF is a string, we assume it is a name of a premade HGF
        hgf = premade_hgf(config.HGF)
    else
        hgf = config.HGF
    end
    #Extract target state
    target_state = config.target_state

    #Create action model function
    am_function = function hgf_sigmoid(
        attributes::ModelAttributes,
        hgf_observation::R,
    ) where {R<:Real}

        #Extract HGF
        hgf = attributes.submodel

        #Extract inverse temperature
        β⁻¹ = 1/load_parameters(attributes).action_noise

        #Update the HGF
        update_hgf!(hgf, hgf_observation)

        #Extract specified belief state
        value = get_states(hgf, target_state)

        #Calculate the sigmoid action probability
        action_probability = value^β⁻¹ / (value^β⁻¹ + (1 - value)^β⁻¹)

        #If the probability mean becomes a NaN
        if isnan(action_probability)
            #Throw an error that will reject samples when fitted
            throw(
                RejectParameters(
                    "With these parameters and inputs, the mean of the gaussian action became $action_probability, which is invalid. Try other parameter settings",
                ),
            )
        end

        #Create normal distribution with mean of the target value and a standard deviation from parameters
        return Bernoulli(action_probability)
    end

    parameters = (action_noise = Parameter(config.action_noise),)

    observations = (; hgf_observation = Observation(Real))

    actions = (; report = Action(Bernoulli),)

    return ActionModel(
        am_function,
        parameters = parameters,
        observations = observations,
        actions = actions,
        submodel = hgf,
    )

end
