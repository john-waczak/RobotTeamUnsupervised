using CSV, DataFrames, DelimitedFiles
using MLJ
using Pkg
Pkg.add(["Clustering", "MLJClusteringInterface"])
using Clustering: silhouettes

using GenerativeTopographicMapping
using CairoMakie, MintsMakieRecipes

using BenchmarkTools

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

# include("./utils.jl")


figures_path = joinpath("./figures", "iris")
if !ispath(figures_path)
    mkpath(figures_path)
end

models_path = joinpath("./models", "iris")
if !ispath(figures_path)
    mkpath(figures_path)
end




df = CSV.read(download("https://ncsa.osn.xsede.org/ees230012-bucket01/mintsML/toy-datasets/iris/iris.csv"), DataFrame)

X = df[:, 1:4]
y = df[:,5]
target_labels = unique(y)
column_labels = uppercasefirst.(replace.(names(df)[1:4], "."=>" "))
y = [findfirst(y[i] .== target_labels) for i in axes(y,1)]



# KMeans -- uses Centroids
KMeans = @load KMeans pkg=Clustering
kmeans = KMeans()
mach = machine(kmeans, X)
fit!(mach)


# get the distance of each record to each of the cluster centroids
X̃ = MLJ.transform(mach, X)

# get the predicted label for each record
ŷ = MLJ.predict(mach, X)

# get the
rpt = report(mach)
fp = fitted_params(mach)

assignments = rpt[:assignments]
centers = fp[:centers]  # D × K where D = number of reatures and K = number of clusters

Δ = pairwise(mach.model.metric, Matrix(X), dims=1)
s = silhouettes(assignments, Δ)
s̄ = mean(s)

k = mach.model.k

fig = Figure();
ax = Axis(fig[1,1])



# write the centers to a file
centers


# SOM


# GTM









# set up code for fitting in parallel, producing silhouette plots, saving centroids
# we can then have an analysis script which plots the mean silhouette scores for all values of k
# we can also visualize the centroids
