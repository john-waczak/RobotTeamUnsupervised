using CSV, DataFrames, DelimitedFiles
using MLJ
using GenerativeTopographicMapping
using CairoMakie
using Random

Random.seed!(42)

using JSON
using ArgParse
using Random
using ProgressMeter
using LaTeXStrings
using MultivariateStats
using Distances
using LinearAlgebra

include("utils/makie-defaults.jl")
include("utils/config.jl")

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





datapath="./data/robot-team/unsupervised"
@assert ispath(datapath)

figures_path = joinpath("./figures", "robot-team")
if !ispath(figures_path)
    mkpath(figures_path)
end

X = CSV.read(joinpath(datapath, "data", "df_features.csv"), DataFrame);
Y = CSV.read(joinpath(datapath, "data", "df_targets.csv"), DataFrame);


# square topology
k = 32
m = 5
nepochs = 250

gtm = GTM(k=k, m=m, tol=1e-5, nepochs=nepochs)
mach = machine(gtm, X)
fit!(mach)

# get fit results
gtm_mdl = fitted_params(mach)[:gtm]
rpt = report(mach)
M = gtm_mdl.M                          # RBF centers
Ξ = rpt[:Ξ]                            # Latent Points
Ψ = rpt[:W] * rpt[:Φ]'                 # Projected Node Means
llhs = rpt[:llhs]

# compute responsabilities and projections
Rs = predict_responsibility(mach, X)
mean_proj = DataFrame(MLJ.transform(mach, X))
mode_proj = DataFrame(DataModes(gtm_mdl, Matrix(X)), [:ξ₁, :ξ₂] )
class_id = get.(MLJ.predict(mach, X))


# compute PCA as well
pca = MultivariateStats.fit(PCA, Matrix(X)', maxoutdim=3, pratio=0.99999);
U = MultivariateStats.predict(pca, Matrix(X)')[1:2,:]'


# plot log-likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 3:length(llhs), llhs[3:end], linewidth=5)
fig
save(joinpath(figures_path, "square-llhs.pdf"), fig)

# set up 2-dimensional color map
fig = Figure();
axl = Axis(fig[1,1], xlabel="u₁", ylabel="u₂", title="PCA", aspect=AxisAspect(1.0))
axr = Axis(fig[1,2], xlabel="ξ₁", ylabel="ξ₂", title="GTM ⟨ξ⟩", aspect=AxisAspect(1.0))
scatter!(axl, U[:,1], U[:,2], markersize=5, alpha=0.7)
scatter!(axr, mean_proj.ξ₁, mean_proj.ξ₂, markersize=5, alpha=0.7, color=class_id)
fig

save(joinpath(figures_path, "square-means.pdf"), fig)

# plot some exemplar means
sample_coords = JSON.parsefile("data/robot-team/sample-coords.json")

# add class info to dict
for (type, info) ∈ sample_coords
    x_type = info["x"]
    y_type = info["y"]
    idx_type = argmin((Y.x .- x_type).^2 .+ (Y.y .- y_type).^2 )
    info["idx_data"] = idx_type
    info["idx_class"] = class_id[idx_type]
end


idx_900 = findfirst(wavelengths .≥ 900)

# ALGAE
idx_algae = sample_coords["algae"]["idx_data"]
idx_sort = sortperm(Rs[idx_algae, :], rev=true)

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="Scaled Reflectance");

ls = []
l_1 = lines!(ax, wavelengths[1:idx_900], Vector(X[idx_algae, 1:idx_900]), linewidth=2)
push!(ls, l_1)
for i ∈ 1:3
    l_2 = lines!(ax, wavelengths[1:idx_900], Ψ[1:idx_900, idx_sort[i]], linewidth=2, color=(mints_colors[2], 1/i))
    if i == 1
        push!(ls, l_2)
    end
end

xlims!(ax, wavelengths[1], 900)
axislegend(ax, ls, ["Algae Spectrum", "GTM Class Signatures"])
fig

save(joinpath(figures_path, "square-algae-spec.pdf"), fig)


# RHODAMINE
idx_plume = sample_coords["plume"]["idx_data"]
idx_sort = sortperm(Rs[idx_plume, :], rev=true)

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="Scaled Reflectance");

ls = []
l_1 = lines!(ax, wavelengths[1:idx_900], Vector(X[idx_plume, 1:idx_900]), linewidth=2)
push!(ls, l_1)
for i ∈ 1:3
    l_2 = lines!(ax, wavelengths[1:idx_900], Ψ[1:idx_900, idx_sort[i]], linewidth=2, color=(mints_colors[2], 1/i))
    if i == 1
        push!(ls, l_2)
    end
end

axislegend(ax, ls, ["Rhodamine Spectrum", "GTM Class Signatures"])
xlims!(ax, wavelengths[1], 900)
fig

save(joinpath(figures_path, "square-rhodamine-spec.pdf"), fig)
save(joinpath(figures_path, "square-rhodamine-spec.png"), fig)

# WATER
idx_water= sample_coords["water"]["idx_data"]
idx_sort = sortperm(Rs[idx_water, :], rev=true)

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="Scaled Reflectance");

ls = []
l_1 = lines!(ax, wavelengths[1:idx_900], Vector(X[idx_water, 1:idx_900]), linewidth=2)
push!(ls, l_1)
for i ∈ 1:3
    l_2 = lines!(ax, wavelengths[1:idx_900], Ψ[1:idx_900, idx_sort[i]], linewidth=2, color=(mints_colors[2], 1/i))
    if i == 1
        push!(ls, l_2)
    end
end

axislegend(ax, ls, ["Water Spectrum", "GTM Class Signatures"])
xlims!(ax, wavelengths[1], 900)
fig

save(joinpath(figures_path, "square-water-spec.pdf"), fig)
save(joinpath(figures_path, "square-water-spec.png"), fig)


# GRASS
idx_grass = sample_coords["grass"]["idx_data"]
idx_sort = sortperm(Rs[idx_grass, :], rev=true)

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="Scaled Reflectance");

ls = []
l_1 = lines!(ax, wavelengths[1:idx_900], Vector(X[idx_grass, 1:idx_900]), linewidth=2)
push!(ls, l_1)
for i ∈ 1:3
    l_2 = lines!(ax, wavelengths[1:idx_900], Ψ[1:idx_900, idx_sort[i]], linewidth=2, color=(mints_colors[2], 1/i))
    if i == 1
        push!(ls, l_2)
    end
end

axislegend(ax, ls, ["Grass Spectrum", "GTM Class Signatures"], position=:lt)
xlims!(ax, wavelengths[1], 900)
fig

save(joinpath(figures_path, "square-grass-spec.pdf"), fig)
save(joinpath(figures_path, "square-grass-spec.png"), fig)


# plot location in latent space
fig = Figure();
ax = Axis(fig[1,1], xlabel="ξ₁", ylabel="ξ₂");
scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=5, alpha=0.7, color=class_id)
s_a = scatter!(ax, mean_proj.ξ₁[sample_coords["algae"]["idx_data"]], mean_proj.ξ₂[sample_coords["algae"]["idx_data"]], marker=:circle, color=:green, markersize=15)
s_p = scatter!(ax, mean_proj.ξ₁[sample_coords["plume"]["idx_data"]], mean_proj.ξ₂[sample_coords["plume"]["idx_data"]], marker=:circle, color=:red, markersize=15)
s_w = scatter!(ax, mean_proj.ξ₁[sample_coords["water"]["idx_data"]], mean_proj.ξ₂[sample_coords["water"]["idx_data"]], marker=:circle, color=:blue, markersize=15)
s_g = scatter!(ax, mean_proj.ξ₁[sample_coords["grass"]["idx_data"]], mean_proj.ξ₂[sample_coords["grass"]["idx_data"]], marker=:circle, color=:brown, markersize=15)

leg = Legend(fig[1,2], [s_a, s_p, s_w, s_g], ["Algae", "Rhodamine", "Water", "Grass"])

fig

save(joinpath(figures_path, "square-means-labeled.png"), fig)
save(joinpath(figures_path, "square-means-labeled.pdf"), fig)



# Generate final analysis dataframe keeping only those unique pairs of GTM means
res_df = DataFrame();
res_df.ξ₁_mean = mean_proj.ξ₁;
res_df.ξ₂_mean = mean_proj.ξ₂;
res_df.ξ₁_mode = mode_proj.ξ₁;
res_df.ξ₂_mode = mode_proj.ξ₂;
res_df = hcat(res_df, Y);

gdf_mean = groupby(res_df, [:ξ₁_mode, :ξ₂_mode]);
df = combine(gdf_mean, first, renamecols=false);

# NDWI

fig = Figure();
ax = Axis(fig[2,1], xlabel="ξ₁", ylabel="ξ₂");
s = scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=9, alpha=0.85, color=Vector(Y.NDWI1))

s_a = scatter!(ax, mean_proj.ξ₁[sample_coords["algae"]["idx_data"]], mean_proj.ξ₂[sample_coords["algae"]["idx_data"]], marker=:circle, color=:green, markersize=15)
s_p = scatter!(ax, mean_proj.ξ₁[sample_coords["plume"]["idx_data"]], mean_proj.ξ₂[sample_coords["plume"]["idx_data"]], marker=:circle, color=:red, markersize=15)
s_w = scatter!(ax, mean_proj.ξ₁[sample_coords["water"]["idx_data"]], mean_proj.ξ₂[sample_coords["water"]["idx_data"]], marker=:circle, color=:blue, markersize=15)
s_g = scatter!(ax, mean_proj.ξ₁[sample_coords["grass"]["idx_data"]], mean_proj.ξ₂[sample_coords["grass"]["idx_data"]], marker=:circle, color=:brown, markersize=15)

fig[1,1] = Legend(fig, [s_a, s_p, s_w, s_g], ["Algae", "Rhodamine", "Water", "Grass"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

cb = Colorbar(fig[2,2], s, label="NDWI")
fig

save(joinpath(figures_path, "square-ndwi.png"), fig)
save(joinpath(figures_path, "square-ndwi.pdf"), fig)


names(Y)


# NDVI
fig = Figure();
ax = Axis(fig[2,1], xlabel="ξ₁", ylabel="ξ₂");
s = scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=9, alpha=0.85, color=Vector(Y.NDVI))

s_a = scatter!(ax, mean_proj.ξ₁[sample_coords["algae"]["idx_data"]], mean_proj.ξ₂[sample_coords["algae"]["idx_data"]], marker=:circle, color=:green, markersize=15)
s_p = scatter!(ax, mean_proj.ξ₁[sample_coords["plume"]["idx_data"]], mean_proj.ξ₂[sample_coords["plume"]["idx_data"]], marker=:circle, color=:red, markersize=15)
s_w = scatter!(ax, mean_proj.ξ₁[sample_coords["water"]["idx_data"]], mean_proj.ξ₂[sample_coords["water"]["idx_data"]], marker=:circle, color=:blue, markersize=15)
s_g = scatter!(ax, mean_proj.ξ₁[sample_coords["grass"]["idx_data"]], mean_proj.ξ₂[sample_coords["grass"]["idx_data"]], marker=:circle, color=:brown, markersize=15)

fig[1,1] = Legend(fig, [s_a, s_p, s_w, s_g], ["Algae", "Rhodamine", "Water", "Grass"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

cb = Colorbar(fig[2,2], s, label="NDVI")
fig

save(joinpath(figures_path, "square-ndvi.png"), fig)
save(joinpath(figures_path, "square-ndvi.pdf"), fig)


# update dict to save exemplar
sample_coords["algae"]["Ψ"] = Ψ[:, sample_coords["algae"]["idx_class"]]
sample_coords["plume"]["Ψ"] = Ψ[:, sample_coords["plume"]["idx_class"]]
sample_coords["water"]["Ψ"] = Ψ[:, sample_coords["water"]["idx_class"]]
sample_coords["grass"]["Ψ"] = Ψ[:, sample_coords["grass"]["idx_class"]]

models_path = "./models"
if !ispath(models_path)
    mkpath(models_path)
end

open(joinpath(models_path, "exemplar-spectra.json"), "w") do f
    JSON.print(f, sample_coords)
end



