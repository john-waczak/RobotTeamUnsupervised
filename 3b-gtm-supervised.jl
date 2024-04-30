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

figures_path = joinpath("./figures", "robot-team-sup")
if !ispath(figures_path)
    mkpath(figures_path)
end

df_X = CSV.read(joinpath(datapath, "df_features_sup.csv"), DataFrame);
df_Y = CSV.read(joinpath(datapath, "df_targets_sup.csv"), DataFrame);
df_lat_lon = CSV.read(joinpath(datapath, "df_lat_lon_sup.csv"), DataFrame);


idx_900 = findfirst(wavelengths .≥ 900)
X = df_X[:, 1:idx_900]

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
mode_proj = DataFrame(DataModes(gtm_mdl, Matrix(X)), [:ξ₁, :ξ₂] )
class_id = get.(MLJ.predict(mach, X))


# compute PCA as well
pca = MultivariateStats.fit(PCA, Matrix(X)', maxoutdim=3, pratio=0.99999);
U = MultivariateStats.predict(pca, Matrix(X)')[1:2,:]'


# plot log-likelihoods
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 3:length(llhs), llhs[3:end], linewidth=5)
fig

save(joinpath(figures_path, "square-llhs.pdf"), fig)



#
names(Y)




# set up 2-dimensional color map
fig = Figure();
axl = CairoMakie.Axis(fig[1,1], xlabel="u₁", ylabel="u₂", title="PCA", aspect=AxisAspect(1.0), xticklabelsize=16,)
axr = CairoMakie.Axis(fig[1,2], xlabel="ξ₁", ylabel="ξ₂", title="GTM ⟨ξ⟩", aspect=AxisAspect(1.0))
scatter!(axl, U[:,1], U[:,2], markersize=5, alpha=0.7)
scatter!(axr, mean_proj.ξ₁, mean_proj.ξ₂, markersize=5, alpha=0.7, color=class_id)
fig

save(joinpath(figures_path, "square-means.pdf"), fig)




# colored class map with identified pixels
function map_color(ξ₁, ξ₂)
    red = (ξ₁ + 1.0) / 2.0
    blue = (ξ₂ + 1.0) / 2.0

    red = (red < 0.0) ? 0.0 : red
    blue = (blue < 0.0) ? 0.0 : blue

    RGBA(red, 0.0, blue, 1.0)
end


fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="ξ₁", ylabel="ξ₂");
mean_proj = DataFrame(MLJ.transform(mach, X))
gtm_colors = map_color.(mean_proj.ξ₁, mean_proj.ξ₂)
s = scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=9, alpha=0.85, color=gtm_colors)
fig

idx_bl = argmin((mean_proj.ξ₁ .- (-1)) .^2 .+ (mean_proj.ξ₂ .- (-1)) .^2)    # A
idx_tl = argmin((mean_proj.ξ₁ .- (-1)) .^2 .+ (mean_proj.ξ₂ .- (1)) .^2)    # A
idx_tr = argmin((mean_proj.ξ₁ .- (1)) .^2 .+ (mean_proj.ξ₂ .- (1)) .^2)    # A
idx_br = argmin((mean_proj.ξ₁ .- (1)) .^2 .+ (mean_proj.ξ₂ .- (-1)) .^2)    # A
idx_c = argmin((mean_proj.ξ₁ .- (0)) .^2 .+ (mean_proj.ξ₂ .- (0)) .^2)    # A

idx_points = [idx_bl, idx_tl, idx_tr, idx_br, idx_c]

scatter!(ax, mean_proj.ξ₁[idx_points], mean_proj.ξ₂[idx_points], marker=[:circle, :rect, :diamond, :cross, :xcross], color=:cyan, strokecolor=:grey, strokewidth=1.5, markersize=15)

fig


save(joinpath(figures_path, "gtm-classes-latent.png"), fig)
save(joinpath(figures_path, "gtm-classes-latent.pdf"), fig)


# Plot the signatures corresponding to the identified points

Ψ_bl = Ψ[1:idx_900, sortperm(Rs[idx_bl, :], rev=true)[1]]
Ψ_tl = Ψ[1:idx_900, sortperm(Rs[idx_tl, :], rev=true)[1]]
Ψ_tr = Ψ[1:idx_900, sortperm(Rs[idx_tr, :], rev=true)[1]]
Ψ_br = Ψ[1:idx_900, sortperm(Rs[idx_br, :], rev=true)[1]]
Ψ_c = Ψ[1:idx_900, sortperm(Rs[idx_c, :], rev=true)[1]]

fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="λ (nm)", ylabel="Scaled Reflectance", title="GTM Classes")
l_bl = lines!(ax, wavelengths[1:idx_900], Ψ_bl, linewidth=2, color=gtm_colors[idx_bl])
l_tl = lines!(ax, wavelengths[1:idx_900], Ψ_tl, linewidth=2, color=gtm_colors[idx_tl])
l_tr = lines!(ax, wavelengths[1:idx_900], Ψ_tr, linewidth=2, color=gtm_colors[idx_tr])
l_br = lines!(ax, wavelengths[1:idx_900], Ψ_br, linewidth=2, color=gtm_colors[idx_br])
l_c = lines!(ax, wavelengths[1:idx_900], Ψ_c, linewidth=2, color=gtm_colors[idx_c])
fig[1,2] = Legend(fig, [l_bl, l_tl, l_tr, l_br, l_c], ["●", "■", "◆", "✚", "✖"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=13, height=-5)

xlims!(ax, wavelengths[1], 900)
ylims!(ax, 0, 1)

fig

save(joinpath(figures_path, "gtm-classes-selected.png"), fig)
save(joinpath(figures_path, "gtm-classes-selected.pdf"), fig)




# plot with in latent space with color by the Supervised boat values



# CDOM
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="ξ₁", ylabel="ξ₂");
mean_proj = DataFrame(MLJ.transform(mach, X))
target_color = Vector(df_Y.CDOM)
clims = (20.1, 21.6)
cm = cgrad(:roma, rev=true)

s = scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=9, alpha=0.85, color=target_color, colormap=cm, colorrange=clims)
cb = Colorbar(fig[1,2], label="CDOM", colorrange=clims, colormap=cm, lowclip = cm[1], highclip = cm[end])
fig
save(joinpath(figures_path, "CDOM.png"), fig)
save(joinpath(figures_path, "CDOM.pdf"), fig)


names(df_Y)

# Temperature
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="ξ₁", ylabel="ξ₂");
mean_proj = DataFrame(MLJ.transform(mach, X))
target_color = Vector(df_Y.Temp3488)
clims = (13.25, 13.95)
cm = cgrad(:roma, rev=true)

s = scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=9, alpha=0.85, color=target_color, colormap=cm, colorrange=clims)
cb = Colorbar(fig[1,2], label="Temperature", colorrange=clims, colormap=cm, lowclip = cm[1], highclip = cm[end])
fig
save(joinpath(figures_path, "temperature.png"), fig)
save(joinpath(figures_path, "Temperature.pdf"), fig)


# Na
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="ξ₁", ylabel="ξ₂");
mean_proj = DataFrame(MLJ.transform(mach, X))
target_color = Vector(df_Y.Na)
clims = (200, 380)
cm = cgrad(:roma, rev=true)

s = scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=9, alpha=0.85, color=target_color, colormap=cm, colorrange=clims)
cb = Colorbar(fig[1,2], label="Na⁺", colorrange=clims, colormap=cm, lowclip = cm[1], highclip = cm[end])
fig
save(joinpath(figures_path, "Na.png"), fig)
save(joinpath(figures_path, "Na.pdf"), fig)




# CO
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="ξ₁", ylabel="ξ₂");
mean_proj = DataFrame(MLJ.transform(mach, X))
target_color = Vector(df_Y.CO)
clims = (25.7, 27.3)
cm = cgrad(:roma, rev=true)

s = scatter!(ax, mean_proj.ξ₁, mean_proj.ξ₂, markersize=9, alpha=0.85, color=target_color, colormap=cm, colorrange=clims)
cb = Colorbar(fig[1,2], label="Crude Oil", colorrange=clims, colormap=cm, lowclip = cm[1], highclip = cm[end])
fig

save(joinpath(figures_path, "CO.png"), fig)
save(joinpath(figures_path, "CO.pdf"), fig)











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


# add in the selected points
scatter!(ax, df_lat_lon.longitudes[idx_points] .- lon_min, df_lat_lon.latitudes[idx_points] .- lat_min, marker=[:circle, :rect, :diamond, :cross, :xcross], color=:cyan, strokecolor=:grey, strokewidth=1.5, markersize=15)

fig


# add in points

save(joinpath(figures_path, "gtm-class-map.png"), fig)


