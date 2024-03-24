using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CSV, DataFrames, DelimitedFiles
using MLJ
using GenerativeTopographicMapping
using JSON
using ArgParse
using Random


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--datapath", "-d"
            help = "Path to directory where dataset is stored"
            arg_type = String
            default = "/Users/johnwaczak/gitrepos/robot-team/RobotTeamUnsupervised/data/robot-team"
        "--modelspath", "-m"
            help = "Path to directory where model output is stored"
            arg_type = String
            default = "/Users/johnwaczak/gitrepos/robot-team/RobotTeamUnsupervised/models"
        # "-k"
        #     help = "k² is total number of latent nodes"
        #     arg_type = Int
        #     default = 32
        # "--m_max", "-m"
        #     help = "Max value of m to use in parameter sweep. There are m² rbf centers."
        #     arg_type = Int
        #     default = 4
        "-s"
            help = "Index for scale factor for rbf variance"
            arg_type = Int
            default = 1
        "-a"
            help = "Regularization factor"
            arg_type = Float64
            default = 0.1
    end


    parsed_args = parse_args(ARGS, s; as_symbols=true)
    @assert ispath(parsed_args[:datapath]) "datapath does not exist"
    @assert ispath(parsed_args[:modelspath]) "modelspath does not exist"

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

    # load in config file
    include("./utils/config.jl")

    datapath = parsed_args[:datapath]
    # k = parsed_args[:k]
    k = 32
    ms = 2:12
    # m_max = parsed_args[:m_max]

    svals = [0.1,0.25,0.5,1.0,1.5,2.0,2.5,3.0]
    s = svals[parsed_args[:s]]
    α = parsed_args[:a]

    @info "Loading datasets..."
    flush(stdout)
    flush(stderr)

    # load in data
    X1 = CSV.read(joinpath(datapath, "df_features_unsup.csv"), DataFrame);
    X2 = CSV.read(joinpath(datapath, "df_features_sup.csv"), DataFrame);
    Y1 = CSV.read(joinpath(datapath, "df_targets_unsup.csv"), DataFrame);
    Y2 = CSV.read(joinpath(datapath, "df_targets_sup.csv"), DataFrame);

    # join into single dataset
    is_sup = vcat([false for _ ∈ 1:nrow(X1)], [true for _ in 1:nrow(X2)])
    idx_900 = findfirst(wavelengths .≥ 900)
    X = vcat(X1[:, 1:idx_900], X2[:, 1:idx_900])

    println("nrow: ", nrow(X), "\tncol: ", ncol(X))

    models_path = parsed_args[:modelspath]

    for m ∈ ms
        @info "m=$(m)"
        flush(stdout)
  	    flush(stderr)


        # let's set up the path for saving results
        outpath_base = joinpath(models_path, "param-search", "m=$(m)__s=$(s)__α=$(α)")

        if !ispath(outpath_base)
            @info "\tCreating save directory at $(outpath_base)"
	          flush(stdout)
    	      flush(stderr)

            mkpath(outpath_base)
        end


        @info "\tInitializing GTM"
        flush(stdout)
        flush(stderr)

        gtm = GTM(k=k, m=m, s=s, α=α, tol=1e-5, nepochs=250)
        mach = machine(gtm, X)

        @info "\tFitting GTM"
        flush(stdout)
        flush(stderr)

        fit!(mach)

        rpt = report(mach)
        @info "\tSaving report"
        flush(stdout)
        flush(stderr)

        rpt_out = Dict()
        rpt_out[:k] = k
        rpt_out[:m] = m
        rpt_out[:s] = s
        rpt_out[:α] = α
        rpt_out[:converged] = rpt[:converged]
        rpt_out[:llhs] = rpt[:llhs]
        rpt_out[:AIC] = rpt[:AIC]
        rpt_out[:BIC] = rpt[:BIC]

        open(joinpath(outpath_base, "gtm_report.json"), "w") do f
            JSON.print(f, rpt_out)
        end
    end
end


main()
