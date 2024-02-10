# https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
ENV["GKSwstype"] = "100"


using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CSV, DataFrames, DelimitedFiles
using MLJ
using GenerativeTopographicMapping
using CairoMakie, MintsMakieRecipes
using JSON
using ArgParse
using Random

set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)




function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--datapath", "-d"
            help = "Path to directory where dataset is stored"
            arg_type = String
            default = "/Users/johnwaczak/gitrepos/RobotTeamUnsupervised/data/robot-team/unsupervised"
        "-k"
            help = "k² is total number of latent nodes"
            arg_type = Int
            default = 32
        "--m_max", "-m"
            help = "Max value of m to use in parameter sweep. There are m² rbf centers."
            arg_type = Int
            default = 32
        "-s"
            help = "Scale factor for rbf variance"
            arg_type = Float64
            default = 0.5
        "-a"
            help = "Regularization factor"
            arg_type = Float64
            default = 0.1
        "--refs_only"
            help = "Label for output files"
            arg_type = Bool
            default = true
    end


    parsed_args = parse_args(ARGS, s; as_symbols=true)

    println(parsed_args)

    @assert ispath(parsed_args[:datapath]) "datapath does not exist"

    return parsed_args
end



function main()
    # seed reproducible pseudo-random number generator
    @info "Setting random number seed"
    flush(stdout)
    flush(stderr)

    rng = Xoshiro(42)

    # parse args making sure that supplied target does exist
    parsed_args = parse_commandline()

    println(parsed_args)

    datapath = parsed_args[:datapath]
    k = parsed_args[:k]
    m_max = parsed_args[:m_max]
    s = parsed_args[:s]
    α = parsed_args[:a]
    refs_only = parsed_args[:refs_only]

    @info "Loading datasets..."
    flush(stdout)
    flush(stderr)


    X = CSV.read(joinpath(datapath, "data", "df_features.csv"), DataFrame)

    println("nrow: ", nrow(X), "\tncol: ", ncol(X))


    folder_name = "all_features"

    if refs_only
        X = X[:, 1:462]
        folder_name = "refs_only"
    end

    for m ∈ 2:1:m_max
        @info "m=$(m)"
        flush(stdout)
  	flush(stderr)


        # let's set up the path for saving results
        outpath_base = joinpath(datapath, "models", folder_name, "k=$(k)__m=$(m)__s=$(s)__α=$(α)")

        if !ispath(outpath_base)
            @info "\tCreating save directory at $(outpath_base)"
	    flush(stdout)
    	    flush(stderr)


            mkpath(outpath_base)
        end


        @info "\tInitializing GTM"
        flush(stdout)
        flush(stderr)

        gtm = GTM(k=k, m=m, s=s, α=α, tol=1e-5, nepochs=20)
        mach = machine(gtm, X)

        @info "\tFitting GTM"
        flush(stdout)
        flush(stderr)

        fit!(mach)

        rpt = report(mach)
        Ξ = rpt[:Ξ]
        Rs = predict_responsibility(mach, X)

        # contourf(Ξ[:,1], Ξ[:,2],  log10.(Rs[1,:] .+ eps(Float64)))
        # contourf(Ξ[:,1], Ξ[:,2],  Rs[1,:])
        @info "\tCreating output files for GTM means and mode class labels"
        flush(stdout)
        flush(stderr)


        df_res = DataFrame(MLJ.transform(mach, X))
        df_res.mode_class = get.(MLJ.predict(mach, X))
        CSV.write(joinpath(outpath_base, "fitres.csv"), df_res)

        @info "\tComputing Responsibility matrix"
        flush(stdout)
        flush(stderr)


        Rs = predict_responsibility(mach, X)
        writedlm(joinpath(outpath_base, "responsibility.csv"), Rs, ',')

        @info "\tSaving report"
        flush(stdout)
        flush(stderr)


        rpt = report(mach)
        open(joinpath(outpath_base, "gtm_report.json"), "w") do f
            JSON.print(f, rpt)
        end

        @info "\tGenerating Plots"
        flush(stdout)
        flush(stderr)


        llhs = rpt[:llhs]
        Ξ = rpt[:Ξ]

        fig = Figure();
        ax = Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
        lines!(ax, 1:length(llhs), llhs, linewidth=5)

        save(joinpath(outpath_base, "training-llhs.png"), fig)
        save(joinpath(outpath_base, "training-llhs.pdf"), fig)

        fig = Figure();
        ax = Axis(
            fig[1,1],
            xlabel="ξ₁",
            ylabel="ξ₂",
            title="GTM Means"
        )
        scatter!(ax, df_res.ξ₁, df_res.ξ₂, color=df_res.mode_class)

        save(joinpath(outpath_base, "latent-means.png"), fig)
        save(joinpath(outpath_base, "latent-means.pdf"), fig)

        @info "\tSaving machine"
        flush(stdout)
        flush(stderr)

        MLJ.save(joinpath(outpath_base, "gtm.jls"), mach)
    end
end


main()
