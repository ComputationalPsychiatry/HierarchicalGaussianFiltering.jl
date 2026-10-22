module HierarchicalGaussianFiltering

#Load packages
using Reexport
@reexport using ActionModels
using RecipesBase
#Numerically stable versions of functions involving exp and log
using LogExpFunctions: logaddexp, log1pexp, logistic, logit

import ActionModels: Agent

#Export functions
export init_node, init_hgf, premade_hgf, check_hgf, update_hgf!, multiple_inputs!
export get_prediction, get_surprise
export ParameterGroup
export EnhancedUpdate, ClassicUpdate, UnboundedUpdate
export NodeDefaults
export ContinuousState,
    ContinuousInput, BinaryState, BinaryInput, CategoricalState, CategoricalInput
export DriftCoupling,
    ObservationCoupling,
    CategoryCoupling,
    ProbabilityCoupling,
    VolatilityCoupling,
    NoiseCoupling,
    LinearTransform,
    NonlinearTransform


const node_attribute_separator = "_"

#Types for HGFs
include("create_hgf/hgf_structs.jl")

#Extending ActionModels functions
include("ActionModels_extensions/initialize_attributes.jl")
include("ActionModels_extensions/manipulate_attributes.jl")
include("ActionModels_extensions/plot_trajectory.jl")
include("ActionModels_extensions/simulation.jl")
include("ActionModels_extensions/manipulate_hgf/set_parameters.jl")
include("ActionModels_extensions/manipulate_hgf/get_parameters.jl")
include("ActionModels_extensions/manipulate_hgf/get_states.jl")
include("ActionModels_extensions/manipulate_hgf/reset.jl")

#Functions for updating the HGF
include("update_hgf/update_hgf.jl")
include("update_hgf/multiple_inputs.jl")
include("update_hgf/nonlinear_transforms.jl")
include("update_hgf/node_updates/continuous_input_node.jl")
include("update_hgf/node_updates/continuous_state_node.jl")
include("update_hgf/node_updates/binary_input_node.jl")
include("update_hgf/node_updates/binary_state_node.jl")
include("update_hgf/node_updates/categorical_input_node.jl")
include("update_hgf/node_updates/categorical_state_node.jl")

#Functions for creating HGFs
include("create_hgf/check_hgf.jl")
include("create_hgf/init_hgf.jl")
include("create_hgf/init_node_edge.jl")
include("create_hgf/create_premade_hgf.jl")

#Functions for premade agents
include("premade_models/premade_action_models/premade_gaussian.jl")
include("premade_models/premade_action_models/premade_predict_category.jl")
include("premade_models/premade_action_models/premade_sigmoid.jl")
include("premade_models/premade_action_models/premade_softmax.jl")

include("premade_models/premade_hgfs/premade_binary_2level.jl")
include("premade_models/premade_hgfs/premade_binary_3level.jl")
include("premade_models/premade_hgfs/premade_categorical_3level.jl")
include("premade_models/premade_hgfs/premade_categorical_transitions_3level.jl")
include("premade_models/premade_hgfs/premade_continuous_2level.jl")
include("premade_models/premade_hgfs/premade_JGET.jl")

#Utility functions for HGFs
include("utils/get_prediction.jl")
include("utils/get_surprise.jl")
include("utils/pretty_printing.jl")
include("utils/set_save_history.jl")
include("utils/lambert_w.jl")

end
