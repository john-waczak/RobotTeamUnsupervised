using CSV, DataFrames, DelimitedFiles
using MLJ
#using Pkg
#Pkg.add(["Clustering", "MLJClusteringInterface"])
#using Clustering: silhouettes
#using SelfOrganizingMaps
#using Distances
using GenerativeTopographicMapping
using CairoMakie, MintsMakieRecipes
using JSON

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


figures_path = joinpath("./figures", "iris")
if !ispath(figures_path)
    mkpath(figures_path)
end

models_path = joinpath("./models", "iris")
if !ispath(figures_path)
    mkpath(figures_path)
end


data_path = "./data"


df = CSV.read(joinpath(data_path, "toy-datasets", "iris", "df_iris.csv"), DataFrame)

X = df[:, 1:4]
y = df[:,5]
target_labels = unique(y)
column_labels = uppercasefirst.(replace.(names(df)[1:4], "."=>" "))
y = [findfirst(y[i] .== target_labels) for i in axes(y,1)]



gtm = GTM(k=6, m=2, tol=1e-5, nepochs=100)
mach = machine(gtm, X)
fit!(mach)

df_res = DataFrame(MLJ.transform(mach, X))
df_res.mode_class = get.(MLJ.predict(mach, X))



# N × K
Rs = predict_responsibility(mach, X)


rpt = report(mach)
llhs = rpt[:llhs]
Ξ = rpt[:Ξ]

rpt


fig = Figure();
ax = Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 1:length(llhs), llhs, linewidth=5)
fig




fig = Figure();
ax = Axis(
    fig[1,1],
    xlabel="ξ₁",
    ylabel="ξ₂",
    title="GTM Means"
)
scatter!(ax, df_res.ξ₁, df_res.ξ₂, color=df_res.mode_class)

fig

rpt
JSON.print(rpt)

# open(joinpath(path_to_use, "$(savename)-occam__$(suffix).json"), "w") do f
#     JSON.print(f, res_dict)
# end








# KMeans -- uses Centroids
# KMeans = @load KMeans pkg=Clustering
# kmeans = KMeans()
# mach = machine(kmeans, X)
# fit!(mach)


# X̃ = MLJ.transform(mach, X)  # distance to all cluster centers
# ŷ = MLJ.predict(mach, X)    # cluster identification

# # get the
# rpt = report(mach)
# fp = fitted_params(mach)
# assignments = rpt[:assignments]
# centers = fp[:centers]  # D × K where D = number of reatures and K = number of clusters

# df_centers = DataFrame(centers', names(X))

# Δ = pairwise(mach.model.metric, Matrix(X), dims=1)
# s = silhouettes(assignments, Δ)


# s̄ = mean(s)

# k = mach.model.k

# fig = Figure();
# ax = Axis(fig[1,1])
# write the centers to a file


# SOM


# GTM



