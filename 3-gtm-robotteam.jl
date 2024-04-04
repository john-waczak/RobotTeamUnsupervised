using CSV, DataFrames, DelimitedFiles, Tables
using MLJ
using GenerativeTopographicMapping
using CairoMakie
using Random
using HDF5

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
include("utils/viz.jl")

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



# get map of exemplar points
h5path= "/Users/johnwaczak/data/robot-team/processed/hsi/12-09/Dye_1/Dye_1-6.h5"
rgb_image = get_h5_rgb(h5path)


blue = colorant"#0366fc"
darkblue = colorant"#013582"

red = colorant"#d10f0f"
darkred = colorant"#a10d0d"

green = colorant"#28a10d"
darkgreen = colorant"#157000"

brown = colorant"#703e00"
darkbrown = colorant"#543105"

tan = colorant"#f0ba78"
darktan = colorant"#ba8a50"



datapath="./data/robot-team"
@assert ispath(datapath)

figures_path = joinpath("./figures", "robot-team")
if !ispath(figures_path)
    mkpath(figures_path)
end

X1 = CSV.read(joinpath(datapath, "df_features_unsup.csv"), DataFrame);
X2 = CSV.read(joinpath(datapath, "df_features_sup.csv"), DataFrame);
Y1 = CSV.read(joinpath(datapath, "df_targets_unsup.csv"), DataFrame);
Y2 = CSV.read(joinpath(datapath, "df_targets_sup.csv"), DataFrame);


is_sup = vcat([false for _ ∈ 1:nrow(X1)], [true for _ in 1:nrow(X2)])
idx_900 = findfirst(wavelengths .≥ 900)
X = vcat(X1[:, 1:idx_900], X2[:, 1:idx_900])


# square topology
k = 32
m = 14
α = 0.1
s = 1.0
nepochs = 300
gtm = GTM(k=k, m=m, s=s, α=α, tol=1e-5, nepochs=nepochs)
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
mean_proj_2 = DataFrame(MLJ.transform(mach, X2))
mode_proj = DataFrame(DataModes(gtm_mdl, Matrix(X)), [:ξ₁, :ξ₂] )
class_id = get.(MLJ.predict(mach, X))

class_id1 = get.(MLJ.predict(mach, X1))

# plot some exemplar means
sample_coords = JSON.parsefile("data/robot-team/sample-coords.json")

# add class info to dict
for (type, info) ∈ sample_coords
    x_type = info["x"]
    y_type = info["y"]
    idx_type = argmin((Y1.x .- x_type).^2 .+ (Y1.y .- y_type).^2 )
    info["idx_data"] = idx_type
    info["idx_class"] = class_id1[idx_type]
end

# add in pixel coords for points in image
sample_coords["plume"]["img_idxs"] = CartesianIndex(80, 235)
sample_coords["water"]["img_idxs"] = CartesianIndex(400, 320)
sample_coords["algae"]["img_idxs"] = CartesianIndex(285, 100)
sample_coords["grass"]["img_idxs"] = CartesianIndex(400, 80)





# compute PCA as well
pca = MultivariateStats.fit(PCA, Matrix(X)', maxoutdim=3, pratio=0.99999);
U = MultivariateStats.predict(pca, Matrix(X)')[1:2,:]'


# plot log-likelihoods
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 3:length(llhs), llhs[3:end], linewidth=5)
fig
save(joinpath(figures_path, "square-llhs.pdf"), fig)

# set up 2-dimensional color map
fig = Figure();
axl = CairoMakie.Axis(fig[1,1], xlabel="u₁", ylabel="u₂", title="PCA", aspect=AxisAspect(1.0), xticklabelsize=16,)
axr = CairoMakie.Axis(fig[1,2], xlabel="ξ₁", ylabel="ξ₂", title="GTM ⟨ξ⟩", aspect=AxisAspect(1.0))
scatter!(axl, U[:,1], U[:,2], markersize=5, alpha=0.7)
scatter!(axr, mean_proj.ξ₁, mean_proj.ξ₂, markersize=5, alpha=0.7, color=class_id)
fig

save(joinpath(figures_path, "square-means.pdf"), fig)




# ALGAE
idx_algae = sample_coords["algae"]["idx_data"]
idx_sort = sortperm(Rs[idx_algae, :], rev=true)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="λ (nm)", ylabel="Scaled Reflectance");
ax_inset = CairoMakie.Axis(fig[2,1],
                           width = Relative(0.333),
                           height = Relative(0.333),
                           halign = 0.1,
                           valign = 0.9,
                           )
hidedecorations!(ax_inset)
hidespines!(ax_inset)

ls = []
l_1 = lines!(ax, wavelengths[1:idx_900], Vector(X[idx_algae, 1:idx_900]), linewidth=2)
push!(ls, l_1)
for i ∈ 1:3
    if Rs[idx_algae, idx_sort[i]] > 0
        l_2 = lines!(ax, wavelengths[1:idx_900], Ψ[1:idx_900, idx_sort[i]], linewidth=2, color=(mints_colors[2], 1/i))
    end
    if i == 1
        push!(ls, l_2)
    end
end

fig[1,1] = Legend(fig, ls, ["Algae Spectrum", "GTM Node Signatures"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, wavelengths[1], 900)
ylims!(ax, 0, 1)

heatmap!(ax_inset, rgb_image)
scatter!(ax_inset, [sample_coords["algae"]["img_idxs"][1]], [sample_coords["algae"]["img_idxs"][2]], markersize=7, color=:white, marker = :xcross)

fig
save(joinpath(figures_path, "square-algae-spec.pdf"), fig)
save(joinpath(figures_path, "square-algae-spec.png"), fig)


# RHODAMINE
idx_plume = sample_coords["plume"]["idx_data"]
idx_sort = sortperm(Rs[idx_plume, :], rev=true)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="λ (nm)", ylabel="Scaled Reflectance");
ax_inset = CairoMakie.Axis(fig[2,1],
                           width = Relative(0.333),
                           height = Relative(0.333),
                           halign = 0.9,
                           valign = 0.9,
                           )
hidedecorations!(ax_inset)
hidespines!(ax_inset)

ls = []
l_1 = lines!(ax, wavelengths[1:idx_900], Vector(X[idx_plume, 1:idx_900]), linewidth=2)
push!(ls, l_1)
for i ∈ 1:3
    if Rs[idx_plume, idx_sort[i]] > 0
        l_2 = lines!(ax, wavelengths[1:idx_900], Ψ[1:idx_900, idx_sort[i]], linewidth=2, color=(mints_colors[2], 1/i))
    end

    if i == 1
        push!(ls, l_2)
    end
end


fig[1,1] = Legend(fig, ls, ["Rhodamine Spectrum", "GTM Node Signatures"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, wavelengths[1], 900)
ylims!(ax, 0, 1)

heatmap!(ax_inset, rgb_image)
scatter!(ax_inset, [sample_coords["plume"]["img_idxs"][1]], [sample_coords["plume"]["img_idxs"][2]], markersize=7, color=:white, marker = :xcross)


fig

save(joinpath(figures_path, "square-rhodamine-spec.pdf"), fig)
save(joinpath(figures_path, "square-rhodamine-spec.png"), fig)


# WATER
idx_water= sample_coords["water"]["idx_data"]
idx_sort = sortperm(Rs[idx_water, :], rev=true)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="λ (nm)", ylabel="Scaled Reflectance");
ax_inset = CairoMakie.Axis(fig[2,1],
                           width = Relative(0.333),
                           height = Relative(0.333),
                           halign = 0.9,
                           valign = 0.9,
                           )
hidedecorations!(ax_inset)
hidespines!(ax_inset)

ls = []
l_1 = lines!(ax, wavelengths[1:idx_900], Vector(X[idx_water, 1:idx_900]), linewidth=2)
push!(ls, l_1)
for i ∈ 1:3
    if Rs[idx_water, idx_sort[i]] > 0
        l_2 = lines!(ax, wavelengths[1:idx_900], Ψ[1:idx_900, idx_sort[i]], linewidth=2, color=(mints_colors[2], 1/i))
    end

    if i == 1
        push!(ls, l_2)
    end
end

fig[1,1] = Legend(fig, ls, ["Water Spectrum", "GTM Node Signatures"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, wavelengths[1], 900)
ylims!(ax, 0, 1)

heatmap!(ax_inset, rgb_image)
scatter!(ax_inset, [sample_coords["water"]["img_idxs"][1]], [sample_coords["water"]["img_idxs"][2]], markersize=7, color=:white, marker = :xcross)

fig

save(joinpath(figures_path, "square-water-spec.pdf"), fig)
save(joinpath(figures_path, "square-water-spec.png"), fig)


# GRASS
idx_grass = sample_coords["grass"]["idx_data"]
idx_sort = sortperm(Rs[idx_grass, :], rev=true)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="λ (nm)", ylabel="Scaled Reflectance");
ax_inset = CairoMakie.Axis(fig[2,1],
                           width = Relative(0.333),
                           height = Relative(0.333),
                           halign = 0.1,
                           valign = 0.9,
                           )
hidedecorations!(ax_inset)
hidespines!(ax_inset)

ls = []
l_1 = lines!(ax, wavelengths[1:idx_900], Vector(X[idx_grass, 1:idx_900]), linewidth=2)
push!(ls, l_1)
for i ∈ 1:3
    if Rs[idx_grass, idx_sort[i]] > 0
        l_2 = lines!(ax, wavelengths[1:idx_900], Ψ[1:idx_900, idx_sort[i]], linewidth=2, color=(mints_colors[2], 1/i))
    end
    if i == 1
        push!(ls, l_2)
    end
end

fig[1,1] = Legend(fig, ls, ["Grass Spectrum", "GTM Node Signatures"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, wavelengths[1], 900)
ylims!(ax, 0, 1)
heatmap!(ax_inset, rgb_image)
scatter!(ax_inset, [sample_coords["grass"]["img_idxs"][1]], [sample_coords["grass"]["img_idxs"][2]], markersize=7, color=:white, marker = :xcross)

fig

save(joinpath(figures_path, "square-grass-spec.pdf"), fig)
save(joinpath(figures_path, "square-grass-spec.png"), fig)


# plot location in latent space
stroke_width = 1.5

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="ξ₁", ylabel="ξ₂");
scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=5, alpha=0.7, color=class_id)

s_a = scatter!(ax, mean_proj.ξ₁[sample_coords["algae"]["idx_data"]], mean_proj.ξ₂[sample_coords["algae"]["idx_data"]], marker=:circle, color=green, markersize=15, strokewidth=stroke_width, strokecolor=darkgreen, )
s_p = scatter!(ax, mean_proj.ξ₁[sample_coords["plume"]["idx_data"]], mean_proj.ξ₂[sample_coords["plume"]["idx_data"]], marker=:circle, color=red, markersize=15, strokewidth=stroke_width, strokecolor=darkred)
s_w = scatter!(ax, mean_proj.ξ₁[sample_coords["water"]["idx_data"]], mean_proj.ξ₂[sample_coords["water"]["idx_data"]], marker=:circle, color=blue, markersize=15, strokewidth=stroke_width, strokecolor=darkblue)
s_g = scatter!(ax, mean_proj.ξ₁[sample_coords["grass"]["idx_data"]], mean_proj.ξ₂[sample_coords["grass"]["idx_data"]], marker=:circle, color=brown, markersize=15, strokewidth=stroke_width, strokecolor=darkbrown)

fig[1,1] = Legend(fig, [s_a, s_p, s_w, s_g,], ["Algae", "Rhodamine", "Water", "Grass",], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig

save(joinpath(figures_path, "square-means-labeled.png"), fig)
save(joinpath(figures_path, "square-means-labeled.pdf"), fig)


# NDWI

is_sup
is_unsup = [i ? false : true for i ∈ is_sup]

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="ξ₁", ylabel="ξ₂");
s = scatter!(ax, mean_proj.ξ₁[is_unsup], mean_proj.ξ₂[is_unsup], markersize=9, alpha=0.85, color=Vector(Y1.NDWI1))

s_a = scatter!(ax, mean_proj.ξ₁[sample_coords["algae"]["idx_data"]], mean_proj.ξ₂[sample_coords["algae"]["idx_data"]], marker=:circle, color=green, markersize=15, strokewidth=stroke_width, strokecolor=darkgreen, )
s_p = scatter!(ax, mean_proj.ξ₁[sample_coords["plume"]["idx_data"]], mean_proj.ξ₂[sample_coords["plume"]["idx_data"]], marker=:circle, color=red, markersize=15, strokewidth=stroke_width, strokecolor=darkred)
s_w = scatter!(ax, mean_proj.ξ₁[sample_coords["water"]["idx_data"]], mean_proj.ξ₂[sample_coords["water"]["idx_data"]], marker=:circle, color=blue, markersize=15, strokewidth=stroke_width, strokecolor=darkblue)
s_g = scatter!(ax, mean_proj.ξ₁[sample_coords["grass"]["idx_data"]], mean_proj.ξ₂[sample_coords["grass"]["idx_data"]], marker=:circle, color=brown, markersize=15, strokewidth=stroke_width, strokecolor=darkbrown)

fig[1,1] = Legend(fig, [s_a, s_p, s_w, s_g,], ["Algae", "Rhodamine", "Water", "Grass",], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
cb = Colorbar(fig[2,2], s, label="NDWI")
fig

save(joinpath(figures_path, "square-ndwi.png"), fig)
save(joinpath(figures_path, "square-ndwi.pdf"), fig)




# NDVI
fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="ξ₁", ylabel="ξ₂");
s = scatter!(ax, mean_proj.ξ₁[is_unsup], mean_proj.ξ₂[is_unsup], markersize=9, alpha=0.85, color=Vector(Y1.NDVI))

s_a = scatter!(ax, mean_proj.ξ₁[sample_coords["algae"]["idx_data"]], mean_proj.ξ₂[sample_coords["algae"]["idx_data"]], marker=:circle, color=green, markersize=15, strokewidth=stroke_width, strokecolor=darkgreen, )
s_p = scatter!(ax, mean_proj.ξ₁[sample_coords["plume"]["idx_data"]], mean_proj.ξ₂[sample_coords["plume"]["idx_data"]], marker=:circle, color=red, markersize=15, strokewidth=stroke_width, strokecolor=darkred)
s_w = scatter!(ax, mean_proj.ξ₁[sample_coords["water"]["idx_data"]], mean_proj.ξ₂[sample_coords["water"]["idx_data"]], marker=:circle, color=blue, markersize=15, strokewidth=stroke_width, strokecolor=darkblue)
s_g = scatter!(ax, mean_proj.ξ₁[sample_coords["grass"]["idx_data"]], mean_proj.ξ₂[sample_coords["grass"]["idx_data"]], marker=:circle, color=brown, markersize=15, strokewidth=stroke_width, strokecolor=darkbrown)

fig[1,1] = Legend(fig, [s_a, s_p, s_w, s_g,], ["Algae", "Rhodamine", "Water", "Grass",], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
cb = Colorbar(fig[2,2], s, label="NDVI")
fig

save(joinpath(figures_path, "square-ndvi.png"), fig)
save(joinpath(figures_path, "square-ndvi.pdf"), fig)



# colored class map with identified pixels
function map_color(ξ₁, ξ₂)
    red = (ξ₁ + 1.0) / 2.0
    blue = (ξ₂ + 1.0) / 2.0

    red = (red < 0.0) ? 0.0 : red
    blue = (blue < 0.0) ? 0.0 : blue

    RGBA(red, 0.0, blue, 1.0)
end


fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="ξ₁", ylabel="ξ₂");
mean_proj = DataFrame(MLJ.transform(mach, X))
gtm_colors = map_color.(mean_proj.ξ₁, mean_proj.ξ₂)

s = scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=9, alpha=0.85, color=gtm_colors)

s_a = scatter!(ax, mean_proj.ξ₁[sample_coords["algae"]["idx_data"]], mean_proj.ξ₂[sample_coords["algae"]["idx_data"]], marker=:circle, color=green, markersize=15, strokewidth=stroke_width, strokecolor=darkgreen, )
s_p = scatter!(ax, mean_proj.ξ₁[sample_coords["plume"]["idx_data"]], mean_proj.ξ₂[sample_coords["plume"]["idx_data"]], marker=:circle, color=red, markersize=15, strokewidth=stroke_width, strokecolor=darkred)
s_w = scatter!(ax, mean_proj.ξ₁[sample_coords["water"]["idx_data"]], mean_proj.ξ₂[sample_coords["water"]["idx_data"]], marker=:circle, color=blue, markersize=15, strokewidth=stroke_width, strokecolor=darkblue)
s_g = scatter!(ax, mean_proj.ξ₁[sample_coords["grass"]["idx_data"]], mean_proj.ξ₂[sample_coords["grass"]["idx_data"]], marker=:circle, color=brown, markersize=15, strokewidth=stroke_width, strokecolor=darkbrown)

fig[1,1] = Legend(fig, [s_a, s_p, s_w, s_g,], ["Algae", "Rhodamine", "Water", "Grass",], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
fig

save(joinpath(figures_path, "gtm-classes-latent.png"), fig)
save(joinpath(figures_path, "gtm-classes-latent.pdf"), fig)




# update dict to save exemplar
sample_coords["algae"]["Ψ"] = Ψ[:, sample_coords["algae"]["idx_class"]]
sample_coords["plume"]["Ψ"] = Ψ[:, sample_coords["plume"]["idx_class"]]
sample_coords["water"]["Ψ"] = Ψ[:, sample_coords["water"]["idx_class"]]
sample_coords["grass"]["Ψ"] = Ψ[:, sample_coords["grass"]["idx_class"]]
# sample_coords["road"]["Ψ"] = Ψ[:, sample_coords["road"]["idx_class"]]

models_path = "./models"
if !ispath(models_path)
    mkpath(models_path)
end

open(joinpath(models_path, "exemplar-spectra.json"), "w") do f
    JSON.print(f, sample_coords)
end



# now let's figure out how to make a map of the class label
hsipath = "/Users/johnwaczak/data/robot-team/processed/hsi"

f_list_1 = [
    "Scotty_1-1.h5",
    "Scotty_1-2.h5",
    "Scotty_1-3.h5",
    "Scotty_1-4.h5",
    "Scotty_1-5.h5",
    "Scotty_1-6.h5",
    "Scotty_1-7.h5",
    "Scotty_1-8.h5",
    "Scotty_1-9.h5",
    "Scotty_1-10.h5",
    "Scotty_1-11.h5",
    "Scotty_1-12.h5",
    "Scotty_1-13.h5",
    "Scotty_1-14.h5",
    "Scotty_1-15.h5",
    "Scotty_1-17.h5",
    "Scotty_1-18.h5",
    "Scotty_1-19.h5",
    "Scotty_1-20.h5",
    "Scotty_1-21.h5",
    "Scotty_1-22.h5",
    "Scotty_1-23.h5",
    "Scotty_1-24.h5",
    "Scotty_1-25.h5",
]

f_list_2 = [
    "Scotty_2-1.h5",
    "Scotty_2-2.h5",
    "Scotty_2-3.h5",
    "Scotty_2-4.h5",
    "Scotty_2-5.h5",
    "Scotty_2-6.h5",
    "Scotty_2-7.h5",
    "Scotty_2-8.h5",
    "Scotty_2-11.h5",
    "Scotty_2-12.h5",
    "Scotty_2-13.h5",
    "Scotty_2-14.h5",
    "Scotty_2-15.h5",
    "Scotty_2-17.h5",
    "Scotty_2-18.h5",
    "Scotty_2-19.h5",
    "Scotty_2-24.h5",
    "Scotty_2-26.h5",
]

f_list_1 = joinpath.(hsipath, "11-23", "Scotty_1", f_list_1)
f_list_2 = joinpath.(hsipath, "11-23", "Scotty_2", f_list_2)
@assert all(ispath.(f_list_1))
@assert all(ispath.(f_list_2))




function in_water(Datacube, varnames; threshold=0.25)
    idx_ndwi = findfirst(varnames .== "NDWI1")
    return findall(Datacube[idx_ndwi,:,:] .> threshold)
end



function get_data_for_map(h5path, Δx = 0.1,)
    # Δx = 0.1
    # h5path = f_list_1[1]
    h5 = h5open(h5path, "r")

    # extract data
    varnames = read(h5["data-Δx_$(Δx)/varnames"])
    Data = read(h5["data-Δx_$(Δx)/Data"])[:, :, :]
    IsInbounds = read(h5["data-Δx_$(Δx)/IsInbounds"])
    Longitudes = read(h5["data-Δx_$(Δx)/Longitudes"])
    Latitudes = read(h5["data-Δx_$(Δx)/Latitudes"])
    xs = read(h5["data-Δx_$(Δx)/X"])
    ys = read(h5["data-Δx_$(Δx)/Y"])

    # close the file
    close(h5)

    # get water pixels
    ij_inbounds = in_water(Data, varnames)

    if any([occursin(piece, split(h5path, "/")[end]) for piece in ["1-19", "1-20", "1-21", "1-22", "1-23"]])
        ij_good = findall(Latitudes .> 33.70152)
        ij_inbounds = [ij for ij ∈ ij_inbounds if ij ∈ ij_good]
    end

    if any([occursin(piece, split(h5path, "/")[end]) for piece in ["1-3", "2-3"]])
        ij_good = findall(Longitudes .< -97.7152)
        ij_inbounds = [ij for ij ∈ ij_inbounds if ij ∈ ij_good]
    end

    if occursin("2-19", h5path)
        ij_good = findall(Latitudes .> 33.7016)
        ij_inbounds = [ij for ij ∈ ij_inbounds if ij ∈ ij_good]
    end

    if any([occursin(piece, split(h5path, "/")[end]) for piece in ["1-25", "2-26"]])
        ij_good = findall(Longitudes .> -97.713775)
        ij_inbounds = [ij for ij ∈ ij_inbounds if ij ∈ ij_good]
    end

    # create matrices for X,Y coordinates
    X = zeros(size(Data,2), size(Data,3))
    Y = zeros(size(Data,2), size(Data,3))

    # fill with values
    for x_i ∈ axes(Data,2)
        for y_j ∈ axes(Data,3)
            X[x_i, y_j] = xs[x_i]
            Y[x_i, y_j] = ys[y_j]
        end
    end

    # keep only the non-nan pixels
    Data = Data[1:idx_900, :, :]

    # scale so that peak value is always 1.0
    for i ∈ axes(Data, 2), j ∈ axes(Data, 3)
        R_max = maximum(Data[:, i, j])
        Data[:,i, j] .= Data[:, i,j] ./ R_max
    end

    # set up output array of colors
    class_color = zeros(RGBA, size(Data, 2), size(Data,3))

    # form table
    df_pred = Tables.table(Data[:, ij_inbounds]', header=varnames[1:idx_900])
    mean_proj =  MLJ.transform(mach, df_pred)
    class_color[ij_inbounds] .= map_color.(mean_proj.ξ₁, mean_proj.ξ₂)

    return class_color, X, Y, Longitudes, Latitudes
end



ϕ_scale = 33.70093
m_per_deg = 111412.84*cosd(ϕ_scale) - 93.5*cosd(3*ϕ_scale) + 0.118 * cosd(5*ϕ_scale)
λ_scale_l = -97.7166
λ_scale_r = λ_scale_l + 30/m_per_deg

w= -97.717472
n= 33.703572
s= 33.700797
e= -97.712413

satmap = get_background_satmap(w,e,s,n)




lon_min, lon_max = (-97.7168, -97.7125)
lat_min, lat_max = (33.70075, 33.7035)

CairoMakie.activate!()
fig = Figure(size=(800,600));
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="Longitude", xtickformat = x -> string.(round.(x .+ lon_min, digits=6)),#  xticklabelfont = 13,
    ylabel="Latitude",  ytickformat = y -> string.(round.(y .+ lat_min, digits=6)),#  yticklabelfont = 13,
    title="GTM Class Map for Water",
    titlealign=:left
);
bg = heatmap!(
    ax,
    (satmap.w - lon_min)..(satmap.e - lon_min),
    (satmap.s - lat_min)..(satmap.n - lat_min),
    satmap.img
)

# add 30 meter scale bar
lines!(ax, [λ_scale_l - lon_min, λ_scale_r - lon_min], [ϕ_scale - lat_min, ϕ_scale - lat_min], color=:white, linewidth=5)
scatter!(ax, [λ_scale_l - lon_min + 0.00003,], [ϕ_scale - lat_min + 0.000075], color=:white, marker=:utriangle, markersize=15)
text!(ax, λ_scale_l - lon_min, ϕ_scale -lat_min - 0.0001, text = "30 m", color=:white, fontsize=12, font=:bold)
text!(ax, [λ_scale_l - lon_min,], [ϕ_scale - lat_min + 0.000125], text="N", color=:white, fontsize=12, font=:bold)

xlims!(ax, 0, -97.7125 - lon_min)
ylims!(ax, 0, 33.7035 - lat_min)

@showprogress for h5path ∈ vcat(f_list_1, f_list_2)
    class_color, X, Y, Lon, Lat = get_data_for_map(h5path)
    lon_l, lon_h = extrema(Lon)
    lat_l, lat_h = extrema(Lat)
    h = heatmap!(
        ax,
        (lon_l-lon_min)..(lon_h-lon_min),
        (lat_l-lat_min)..(lat_h-lat_min),
        class_color,
    )
end


fig


save(joinpath(figures_path, "gtm-class-map.png"), fig)




