using HierarchicalGaussianFiltering
using Test
using Glob

#Get the root path of the package
hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))

@testset "All tests" begin

    @testset "Unit tests" begin

        #Get the path to the testing folder
        test_path = hgf_path * "/test/"

        @testset "quick tests" begin
            # Test the quick tests that are used as pre-commit tests
            include(test_path * "quicktests.jl")
        end

        # List the julia filenames in the testsuite
        filenames = glob("*.jl", test_path * "testsuite")

        # For each file
        for filename in filenames
            #Run it
            include(filename)
        end
    end

    @testset "Documentation tests" begin

        #Set up path for the documentation folder
        documentation_path = joinpath(hgf_path, "docs", "src")

        @testset "Sourcefiles" begin

            # List the julia filenames in the documentation source files folder
            filenames = [glob("*/*.jl", joinpath(documentation_path, "julia_files"));glob("*.jl", joinpath(documentation_path, "julia_files"))] 

            for filename in filenames
                @testset "$filename" begin
                    include(filename)
                end
            end
        end

        # @testset "Tutorials" begin

        #     # List the julia filenames in the tutorials folder
        #     filenames = glob("*.jl", joinpath(documentation_path, "tutorials"))

        #     for filename in filenames
        #         @testset "$filename" begin
        #             include(filename)
        #         end
        #     end
        # end
    end
end
