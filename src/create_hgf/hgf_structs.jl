#####################################
######## Abstract node types ########
#####################################

#Top-level node type
abstract type AbstractNode end

#Input and state node subtypes
abstract type AbstractStateNode <: AbstractNode end
abstract type AbstractInputNode <: AbstractNode end

#Variable type subtypes
abstract type AbstractContinuousStateNode <: AbstractStateNode end
abstract type AbstractContinuousInputNode <: AbstractInputNode end
abstract type AbstractBinaryStateNode <: AbstractStateNode end
abstract type AbstractBinaryInputNode <: AbstractInputNode end
abstract type AbstractCategoricalStateNode <: AbstractStateNode end
abstract type AbstractCategoricalInputNode <: AbstractInputNode end

#Abstract type for node information
abstract type AbstractNodeInfo end
abstract type AbstractInputNodeInfo <: AbstractNodeInfo end
abstract type AbstractStateNodeInfo <: AbstractNodeInfo end

##################################
######## HGF update types ########
##################################

#Supertype for HGF update types
abstract type HGFUpdateType end

#Classic and enhance dupdate types
struct ClassicUpdate <: HGFUpdateType end
struct EnhancedUpdate <: HGFUpdateType end
struct UniversalUpdate <: HGFUpdateType end

################################
######## Coupling types ########
################################

#Types for specifying nonlinear transformations
abstract type CouplingTransform end

Base.@kwdef mutable struct LinearTransform <: CouplingTransform
    parameters::Dict = Dict()
end

Base.@kwdef mutable struct NonlinearTransform <: CouplingTransform
    base_function::Function
    first_derivation::Function
    second_derivation::Function
    parameters::Dict = Dict()
end

#Supertypes for coupling types
abstract type CouplingType end
abstract type ValueCoupling <: CouplingType end
abstract type PrecisionCoupling <: CouplingType end

#Concrete value coupling types
Base.@kwdef mutable struct DriftCoupling <: ValueCoupling
    strength::Union{Nothing,Real} = nothing
    transform::CouplingTransform = LinearTransform()
end
Base.@kwdef mutable struct ProbabilityCoupling <: ValueCoupling
    strength::Union{Nothing,Real} = nothing
end
Base.@kwdef mutable struct CategoryCoupling <: ValueCoupling end
Base.@kwdef mutable struct ObservationCoupling <: ValueCoupling end

#Concrete precision coupling types
Base.@kwdef mutable struct VolatilityCoupling <: PrecisionCoupling
    strength::Union{Nothing,Real} = nothing
end
Base.@kwdef mutable struct NoiseCoupling <: PrecisionCoupling
    strength::Union{Nothing,Real} = nothing
end

############################
######## HGF Struct ########
############################
"""
"""
Base.@kwdef mutable struct OrderedNodes
    all_nodes::Vector{AbstractNode} = AbstractNode[]
    input_nodes::Vector{AbstractInputNode} = []
    all_state_nodes::Vector{AbstractStateNode} = []
    early_update_state_nodes::Vector{AbstractStateNode} = []
    late_update_state_nodes::Vector{AbstractStateNode} = []
end

"""
"""
Base.@kwdef mutable struct HGF
    all_nodes::Dict{String,AbstractNode}
    input_nodes::Dict{String,AbstractInputNode}
    state_nodes::Dict{String,AbstractStateNode}
    ordered_nodes::OrderedNodes = OrderedNodes()
    parameter_groups::Dict = Dict()
    save_history::Bool = true
    timesteps::Vector{Real} = [0]
end

##################################
######## HGF Info Structs ########
##################################
Base.@kwdef struct NodeDefaults
    input_noise::Real = -2
    bias::Real = 0
    volatility::Real = -2
    drift::Real = 0
    autoconnection_strength::Real = 1
    initial_mean::Real = 0
    initial_precision::Real = 1
    coupling_strength::Real = 1
    update_type::HGFUpdateType = EnhancedUpdate()
end

Base.@kwdef mutable struct ContinuousState <: AbstractStateNodeInfo
    name::String
    volatility::Union{Real,Nothing} = nothing
    drift::Union{Real,Nothing} = nothing
    autoconnection_strength::Union{Real,Nothing} = nothing
    initial_mean::Union{Real,Nothing} = nothing
    initial_precision::Union{Real,Nothing} = nothing
end

Base.@kwdef mutable struct ContinuousInput <: AbstractInputNodeInfo
    name::String
    input_noise::Union{Real,Nothing} = nothing
    bias::Union{Real,Nothing} = nothing
end

Base.@kwdef mutable struct BinaryState <: AbstractStateNodeInfo
    name::String
end

Base.@kwdef mutable struct BinaryInput <: AbstractInputNodeInfo
    name::String
end

Base.@kwdef mutable struct CategoricalState <: AbstractStateNodeInfo
    name::String
end

Base.@kwdef mutable struct CategoricalInput <: AbstractInputNodeInfo
    name::String
end



#######################################
######## Continuous State Node ########
#######################################
Base.@kwdef mutable struct ContinuousStateNodeEdges
    #Possible parent types
    drift_parents::Vector{<:AbstractContinuousStateNode} = Vector{ContinuousStateNode}()
    volatility_parents::Vector{<:AbstractContinuousStateNode} =
        Vector{ContinuousStateNode}()

    #Possible children types
    drift_children::Vector{<:AbstractContinuousStateNode} = Vector{ContinuousStateNode}()
    volatility_children::Vector{<:AbstractContinuousStateNode} =
        Vector{ContinuousStateNode}()
    probability_children::Vector{<:AbstractBinaryStateNode} = Vector{BinaryStateNode}()
    observation_children::Vector{<:AbstractContinuousInputNode} =
        Vector{ContinuousInputNode}()
    noise_children::Vector{<:AbstractContinuousInputNode} = Vector{ContinuousInputNode}()
end

"""
Configuration of continuous state nodes' parameters 
"""
Base.@kwdef mutable struct ContinuousStateNodeParameters
    volatility::Real = 0
    drift::Real = 0
    autoconnection_strength::Real = 1
    initial_mean::Real = 0
    initial_precision::Real = 0
    coupling_strengths::Dict{String,Real} = Dict{String,Real}()
    coupling_transforms::Dict{String,CouplingTransform} = Dict{String,Real}()
end

"""
Configurations of the continuous state node states
"""
Base.@kwdef mutable struct ContinuousStateNodeState
    posterior_mean::Union{Real} = 0
    posterior_precision::Union{Real} = 1
    value_prediction_error::Union{Real,Missing} = missing
    precision_prediction_error::Union{Real,Missing} = missing
    prediction_mean::Union{Real,Missing} = missing
    prediction_precision::Union{Real,Missing} = missing
    effective_prediction_precision::Union{Real,Missing} = missing
end

"""
Configuration of continuous state node history
"""
Base.@kwdef mutable struct ContinuousStateNodeHistory
    posterior_mean::Vector{Real} = []
    posterior_precision::Vector{Real} = []
    value_prediction_error::Vector{Union{Real,Missing}} = []
    precision_prediction_error::Vector{Union{Real,Missing}} = []
    prediction_mean::Vector{Union{Real,Missing}} = []
    prediction_precision::Vector{Union{Real,Missing}} = []
    effective_prediction_precision::Vector{Union{Real,Missing}} = []
end

"""
"""
Base.@kwdef mutable struct ContinuousStateNode <: AbstractContinuousStateNode
    name::String
    edges::ContinuousStateNodeEdges = ContinuousStateNodeEdges()
    parameters::ContinuousStateNodeParameters = ContinuousStateNodeParameters()
    states::ContinuousStateNodeState = ContinuousStateNodeState()
    history::ContinuousStateNodeHistory = ContinuousStateNodeHistory()
    update_type::HGFUpdateType = ClassicUpdate()
end


#######################################
######## Continuous Input Node ########
#######################################
Base.@kwdef mutable struct ContinuousInputNodeEdges
    #Possible parents
    observation_parents::Vector{<:AbstractContinuousStateNode} =
        Vector{ContinuousStateNode}()
    noise_parents::Vector{<:AbstractContinuousStateNode} = Vector{ContinuousStateNode}()


end

"""
Configuration of continuous input node parameters
"""
Base.@kwdef mutable struct ContinuousInputNodeParameters
    input_noise::Real = 0
    bias::Real = 0
    coupling_strengths::Dict{String,Real} = Dict{String,Real}()
    coupling_transforms::Dict{String,CouplingTransform} = Dict{String,Real}()
end

"""
Configuration of continuous input node states
"""
Base.@kwdef mutable struct ContinuousInputNodeState
    input_value::Union{Real,Missing} = missing
    value_prediction_error::Union{Real,Missing} = missing
    precision_prediction_error::Union{Real,Missing} = missing
    prediction_mean::Union{Real,Missing} = missing
    prediction_precision::Union{Real,Missing} = missing
end

"""
Configuration of continuous input node history
"""
Base.@kwdef mutable struct ContinuousInputNodeHistory
    input_value::Vector{Union{Real,Missing}} = []
    value_prediction_error::Vector{Union{Real,Missing}} = []
    precision_prediction_error::Vector{Union{Real,Missing}} = []
    prediction_mean::Vector{Union{Real,Missing}} = []
    prediction_precision::Vector{Union{Real,Missing}} = []
end

"""
"""
Base.@kwdef mutable struct ContinuousInputNode <: AbstractContinuousInputNode
    name::String
    edges::ContinuousInputNodeEdges = ContinuousInputNodeEdges()
    parameters::ContinuousInputNodeParameters = ContinuousInputNodeParameters()
    states::ContinuousInputNodeState = ContinuousInputNodeState()
    history::ContinuousInputNodeHistory = ContinuousInputNodeHistory()
end

###################################
######## Binary State Node ########
###################################
Base.@kwdef mutable struct BinaryStateNodeEdges
    #Possible parent types
    probability_parents::Vector{<:AbstractContinuousStateNode} =
        Vector{ContinuousStateNode}()

    #Possible children types
    category_children::Vector{<:AbstractCategoricalStateNode} =
        Vector{CategoricalStateNode}()
    observation_children::Vector{<:AbstractBinaryInputNode} = Vector{BinaryInputNode}()
end

"""
 Configure parameters of binary state node
"""
Base.@kwdef mutable struct BinaryStateNodeParameters
    coupling_strengths::Dict{String,Real} = Dict{String,Real}()
end

"""
Overview of the states of the binary state node
"""
Base.@kwdef mutable struct BinaryStateNodeState
    posterior_mean::Union{Real,Missing} = missing
    posterior_precision::Union{Real,Missing} = missing
    value_prediction_error::Union{Real,Missing} = missing
    prediction_mean::Union{Real,Missing} = missing
    prediction_precision::Union{Real,Missing} = missing
end

"""
Overview of the history of the binary state node
"""
Base.@kwdef mutable struct BinaryStateNodeHistory
    posterior_mean::Vector{Union{Real,Missing}} = []
    posterior_precision::Vector{Union{Real,Missing}} = []
    value_prediction_error::Vector{Union{Real,Missing}} = []
    prediction_mean::Vector{Union{Real,Missing}} = []
    prediction_precision::Vector{Union{Real,Missing}} = []
end

"""
Overview of edge posibilities 
"""
Base.@kwdef mutable struct BinaryStateNode <: AbstractBinaryStateNode
    name::String
    edges::BinaryStateNodeEdges = BinaryStateNodeEdges()
    parameters::BinaryStateNodeParameters = BinaryStateNodeParameters()
    states::BinaryStateNodeState = BinaryStateNodeState()
    history::BinaryStateNodeHistory = BinaryStateNodeHistory()
    update_type::HGFUpdateType = ClassicUpdate()
end



###################################
######## Binary Input Node ########
###################################
Base.@kwdef mutable struct BinaryInputNodeEdges
    observation_parents::Vector{<:AbstractBinaryStateNode} = Vector{BinaryStateNode}()
end

"""
Configuration of parameters in binary input node. Default category mean set to [0,1]
"""
Base.@kwdef mutable struct BinaryInputNodeParameters
    coupling_strengths::Dict{String,Real} = Dict{String,Real}()
end

"""
Configuration of states of binary input node
"""
Base.@kwdef mutable struct BinaryInputNodeState
    input_value::Union{Real,Missing} = missing
end

"""
Configuration of history of binary input node
"""
Base.@kwdef mutable struct BinaryInputNodeHistory
    input_value::Vector{Union{Real,Missing}} = [missing]
end

"""
"""
Base.@kwdef mutable struct BinaryInputNode <: AbstractBinaryInputNode
    name::String
    edges::BinaryInputNodeEdges = BinaryInputNodeEdges()
    parameters::BinaryInputNodeParameters = BinaryInputNodeParameters()
    states::BinaryInputNodeState = BinaryInputNodeState()
    history::BinaryInputNodeHistory = BinaryInputNodeHistory()
end



########################################
######## Categorical State Node ########
########################################
Base.@kwdef mutable struct CategoricalStateNodeEdges
    #Possible parents
    category_parents::Vector{<:AbstractBinaryStateNode} = Vector{BinaryStateNode}()
    #The order of the category parents
    category_parent_order::Vector{String} = []

    #Possible children
    observation_children::Vector{<:AbstractCategoricalInputNode} =
        Vector{CategoricalInputNode}()
end

Base.@kwdef mutable struct CategoricalStateNodeParameters
    coupling_strengths::Dict{String,Real} = Dict{String,Real}()
end

"""
Configuration of states in categorical state node
"""
Base.@kwdef mutable struct CategoricalStateNodeState
    posterior::Vector{Union{Real,Missing}} = []
    value_prediction_error::Vector{Union{Real,Missing}} = []
    prediction::Vector{Union{Real,Missing}} = []
    parent_predictions::Vector{Union{Real,Missing}} = []
end

"""
Configuration of history in categorical state node
"""
Base.@kwdef mutable struct CategoricalStateNodeHistory
    posterior::Vector{Vector{Union{Real,Missing}}} = []
    value_prediction_error::Vector{Vector{Union{Real,Missing}}} = []
    prediction::Vector{Vector{Union{Real,Missing}}} = []
    parent_predictions::Vector{Vector{Union{Real,Missing}}} = []
end

"""
Configuration of edges in categorical state node
"""
Base.@kwdef mutable struct CategoricalStateNode <: AbstractCategoricalStateNode
    name::String
    edges::CategoricalStateNodeEdges = CategoricalStateNodeEdges()
    parameters::CategoricalStateNodeParameters = CategoricalStateNodeParameters()
    states::CategoricalStateNodeState = CategoricalStateNodeState()
    history::CategoricalStateNodeHistory = CategoricalStateNodeHistory()
    update_type::HGFUpdateType = ClassicUpdate()
end



########################################
######## Categorical Input Node ########
########################################
Base.@kwdef mutable struct CategoricalInputNodeEdges
    observation_parents::Vector{<:AbstractCategoricalStateNode} =
        Vector{CategoricalStateNode}()
end

Base.@kwdef mutable struct CategoricalInputNodeParameters
    coupling_strengths::Dict{String,Real} = Dict{String,Real}()
end

"""
Configuration of states of categorical input node
"""
Base.@kwdef mutable struct CategoricalInputNodeState
    input_value::Union{Real,Missing} = missing
end

"""
History of categorical input node
"""
Base.@kwdef mutable struct CategoricalInputNodeHistory
    input_value::Vector{Union{Real,Missing}} = [missing]
end

"""
"""
Base.@kwdef mutable struct CategoricalInputNode <: AbstractCategoricalInputNode
    name::String
    edges::CategoricalInputNodeEdges = CategoricalInputNodeEdges()
    parameters::CategoricalInputNodeParameters = CategoricalInputNodeParameters()
    states::CategoricalInputNodeState = CategoricalInputNodeState()
    history::CategoricalInputNodeHistory = CategoricalInputNodeHistory()
end
