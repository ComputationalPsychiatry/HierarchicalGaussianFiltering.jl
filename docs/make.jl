using HierarchicalGaussianFiltering
using Documenter
using Literate
using Glob

## SET FOLDER NAMES ##
if haskey(ENV, "GITHUB_WORKSPACE")
    project_dir = ENV["GITHUB_WORKSPACE"]
else
    project_dir = pwd()
end
julia_files_folder = joinpath(project_dir, "docs", "julia_files")
markdown_files_folder = joinpath(project_dir, "docs", "src")
generated_files_folder = joinpath(markdown_files_folder, "generated")
theory_folder = joinpath(markdown_files_folder, "theory")


## GENERATE MARKDOWNS ##
#Remove old markdowns
for markdown_file in glob( "*.md", generated_files_folder)
    rm(markdown_file)
end

#Create markdowns from julia files
for julia_file in [glob( "*/*.jl", julia_files_folder); glob( "*.jl", julia_files_folder)]

    Literate.markdown(julia_file,
                      generated_files_folder, 
                      execute=true, 
                      documenter=true,
                      codefence =  "```julia" => "```"
                      )
end


## GENERATE AND DEPLOY DOCS ##
DocMeta.setdocmeta!(
    HierarchicalGaussianFiltering,
    :DocTestSetup,
    :(using HierarchicalGaussianFiltering);
    recursive = true,
)

#Create documentation
makedocs(;
    modules = [HierarchicalGaussianFiltering],
    authors = "Peter Thestrup Waade ptw@cas.au.dk, Christoph Mathys chmathys@cas.au.dk and contributors",
    repo = "https://github.com/ComputationalPsychiatry/HierarchicalGaussianFiltering.jl/blob/{commit}{path}#{line}",
    sitename = "HierarchicalGaussianFiltering.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ComputationalPsychiatry.github.io/HierarchicalGaussianFiltering.jl",
        assets = String[],
        size_threshold = 1_500_000, ##MAKE THIS SMALLER?
    ),
    doctest=true,
    pages = [
        "Introduction to Hierarchical Gaussian Filtering" => joinpath(".", "generated", "index.md"),
        # "Theory" => [
        #     "./theory" * "/genmodel.md",
        #     "./theory" * "/node.md",
        #     "./theory" * "/vape.md",
        #     "./theory" * "/vope.md",
        # ],
        "Using the package" => [
            
            "The HGF Nodes" => joinpath(".", "generated", "the_HGF_nodes.md"),
            "Building an HGF" => joinpath(".", "generated", "building_an_HGF.md"),
            "Updating the HGF" => joinpath(".", "generated", "updating_the_HGF.md"),
            "List Of Premade Agent Models" => joinpath(".", "generated", "premade_models.md"),
            "List Of Premade HGF's" => joinpath(".", "generated", "premade_HGF.md"),
            "Fitting an HGF-agent model to data" => joinpath(".", "generated", "fitting_hgf_models.md"),
            "Utility Functions" => joinpath(".", "generated", "utility_functions.md"),
        ],
        "Tutorials" => [
            "classic binary" => joinpath(".", "generated", "classic_binary.md"),
            "classic continouous" => joinpath(".", "generated", "classic_usdchf.md"),
            "classic JGET" => joinpath(".", "generated", "classic_JGET.md"),
        ],
        "All Functions" => joinpath(".", "generated", "all_functions.md"),
    ],
)

deploydocs(;
    repo = "github.com/ComputationalPsychiatry/HierarchicalGaussianFiltering.jl",
    devbranch = "main",
)