using CSV, DataFrames, DelimitedFiles
using CairoMakie
using Makie.Colors
using JSON
using ProgressMeter
using LinearAlgebra, Statistics

using Random
Random.seed!(42)

include("utils/makie-defaults.jl")

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

include("utils/viz.jl")
include("utils/config.jl")

figures_path = joinpath("./figures", "maps")
if !ispath(figures_path)
    mkpath(figures_path)
end

# see https://en.wikipedia.org/wiki/Geographic_coordinate_system#Latitude_and_longitude
# for detail
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
ax = CairoMakie.Axis(fig[1,1]);
bg = heatmap!(ax, (satmap.w - lon_min)..(satmap.e - lon_min), (satmap.s - lat_min)..(satmap.n - lat_min), satmap.img);

# add 30 meter scale bar
lines!(ax, [λ_scale_l - lon_min, λ_scale_r - lon_min], [ϕ_scale - lat_min, ϕ_scale - lat_min], color=:white, linewidth=5)
text!(ax, λ_scale_l - lon_min, ϕ_scale - lat_min - 0.0001, text = "30 m", color=:white, fontsize=12, font=:bold)

# add North arrow
scatter!(ax, [λ_scale_l - lon_min + 0.00003,], [ϕ_scale - lat_min + 0.000075], color=:white, marker=:utriangle, markersize=15)
text!(ax, [λ_scale_l - lon_min,], [ϕ_scale - lat_min + 0.000125], text="N", color=:white, fontsize=12, font=:bold)

xlims!(ax, 0, -97.7125 - lon_min)
ylims!(ax, 0, 33.7035 - lat_min)

fig



# generate two file lists for dye plume visualization

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
]

f_list_1 = joinpath.(hsipath, "11-23", "Scotty_1", f_list_1)
f_list_2 = joinpath.(hsipath, "11-23", "Scotty_2", f_list_2)
@assert all(ispath.(f_list_1))
@assert all(ispath.(f_list_2))


f_list_dye = [
    joinpath(hsipath, "11-23", "Scotty_4", "Scotty_4-1.h5"),
    joinpath(hsipath, "11-23", "Scotty_5", "Scotty_5-1.h5"),
    joinpath(hsipath, "11-23", "Scotty_5", "Scotty_5-2.h5"),
    joinpath(hsipath, "12-09", "Dye_1", "Dye_1-6.h5"),
    joinpath(hsipath, "12-09", "Dye_2", "Dye_2-5.h5"),
    joinpath(hsipath, "12-10", "Dye_1", "Dye_1-6.h5"),
    joinpath(hsipath, "12-10", "Dye_2", "Dye_2-1.h5"),
]

@assert all(ispath.(f_list_dye))


function in_water(Datacube, varnames; threshold=0.25)
    idx_ndwi = findfirst(varnames .== "NDWI1")
    return findall(Datacube[idx_ndwi,:,:] .> threshold)
end

function get_data_for_pred(h5path, Δx = 0.1,)
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

    ij_inbounds= in_water(Data, varnames)
    # ij_inbounds = findall(IsInbounds)

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
    Data = Data[:, ij_inbounds]

    # scale so that peak value is always 1.0
    for j ∈ axes(Data, 2)
        # use R vals with λ < 900 nm
        R_max = maximum(Data[1:idx_900, j])
        Data[1:462, j] .= Data[1:462, j] ./ R_max
    end


    X = X[ij_inbounds]
    Y = Y[ij_inbounds]
    Longitudes = Longitudes[ij_inbounds]
    Latitudes = Latitudes[ij_inbounds]

    return Data[1:idx_900, :], X, Y, Longitudes, Latitudes
end


function get_data_for_heatmap(h5path, Δx = 0.1,)
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

    ij_inbounds = in_water(Data, varnames)
    # ij_inbounds = findall(IsInbounds)

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

    return Data, X, Y, Longitudes, Latitudes, ij_inbounds, IsInbounds
end


exemplar_spectra = JSON.parsefile("./models/exemplar-spectra.json")


# now we have all of our files that we will
idx_900 = findfirst(wavelengths .≥ 900)
Ro_plume = Float64.(exemplar_spectra["plume"]["R"][1:idx_900])
Ro_algae = Float64.(exemplar_spectra["algae"]["R"][1:idx_900])


R_plume = Float64.(exemplar_spectra["plume"]["Ψ"][1:idx_900])
R_algae = Float64.(exemplar_spectra["algae"]["Ψ"][1:idx_900])


function cosine_distance(Rᵢ, Rⱼ)
    1 - dot(Rᵢ, Rⱼ)/(norm(Rᵢ) * norm(Rⱼ))
end

function MSE(Rᵢ, Rⱼ)
    mean((Rᵢ .- Rⱼ).^2)
end

function RMSE(Rᵢ, Rⱼ)
    sqrt(MSE(Rᵢ, Rⱼ))
end

function NS3(Rᵢ, Rⱼ)
    sqrt(MSE(Rᵢ, Rⱼ) + cosine_distance(Rᵢ, Rⱼ)^2)
end


# plume 1 plot
rmse_thresh = 0.265
cos_thresh = 0.265
ns3_thresh = 0.4

c_low = colorant"#f73a2d"
c_high = colorant"#450a06"
cmap = cgrad([c_high, c_low])
clims = (0.05, ns3_thresh)


R, X, Y, Lon, Lat, ij_inbounds, IsInbounds = get_data_for_heatmap(f_list_dye[1]);
rmse_plume = zeros(size(R,2), size(R,3));
ns3_plume = zeros(size(R,2), size(R,3));
cos_plume = zeros(size(R,2), size(R,3));



# compute distribution of ns3 for this datacube
# using Statistics

# R, X, Y, Lon, Lat, ij_inbounds, IsInbounds = get_data_for_heatmap(f_list_dye[1]);
# R1, X1, Y1, Lon1, Lat1, ij_inbounds1, IsInbounds1 = get_data_for_heatmap(f_list_dye[2]);
# R2, X2, Y2, Lon2, Lat2, ij_inbounds2, IsInbounds2 = get_data_for_heatmap(f_list_dye[3]);



# idx_notnan = findall(.!(isnan.(R[1,:,:])))
# R[:, idx_notnan]
# ns3_plume_vals_1 = [NS3(R[:, idx], R_plume) for idx in idx_notnan]

# idx_notnan = findall(.!(isnan.(R1[1,:,:])))
# R1[:, idx_notnan]
# ns3_plume_vals_2 = [NS3(R1[:, idx], R_plume) for idx in idx_notnan]

# idx_notnan = findall(.!(isnan.(R2[1,:,:])))
# R2[:, idx_notnan]
# ns3_plume_vals_3 = [NS3(R2[:, idx], R_plume) for idx in idx_notnan]

# ns3_res = vcat(ns3_plume_vals_1, ns3_plume_vals_2, ns3_plume_vals_3)

# quantile(ns3_plume_vals_1, 0.15) # ~0.4

# length(ns3_res)
# density(ns3_res)
# hist(ns3_res, bins=30)

# flist_to_use = vcat(f_list_1[1:7], "/Users/johnwaczak/data/robot-team/processed/hsi/12-09/Dye_2/Dye_2-1.h5", f_list_dye[1])
# R, X, Y, Lon, Lat, ij_inbounds, IsInbounds = get_data_for_heatmap(flist_to_use[1]);
# idx_notnan = findall(.!(isnan.(R[1,:,:])))
# R[:, idx_notnan]
# ns3_1 = [NS3(R[:, idx], R_algae) for idx in idx_notnan]

# quantile(ns3_1[ns3_1 .> 0.2], 0.03)


for i ∈ axes(rmse_plume,1), j ∈ axes(rmse_plume,2)
    rmse = RMSE(R[:,i,j], R_plume)
    ns3 = NS3(R[:,i,j], R_plume)
    sam = RMSE(R[:,i,j], R_plume)

    if rmse ≤ rmse_thresh
        rmse_plume[i,j] = rmse
    else
        rmse_plume[i,j] = NaN
    end

    if ns3 ≤ ns3_thresh
        ns3_plume[i,j] = ns3
    else
        ns3_plume[i,j] = NaN
    end

    if sam ≤ cos_thresh
        cos_plume[i,j] = sam
    else
        cos_plume[i,j] = NaN
    end
end


idx_good = findall(ns3_plume .≤ ns3_thresh)
tot_area = length(idx_good) * 0.1 * 0.1

lon_min, lon_max = (-97.7168, -97.7145)
lat_min, lat_max = (33.7015, 33.7035)

lon_l, lon_h = extrema(Lon)
lat_l, lat_h = extrema(Lat)


fig = Figure(; size=(800, 600));
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="Longitude", xtickformat = x -> string.(round.(x .+ lon_min, digits=6)),#  xticklabelfont = 13,
    ylabel="Latitude",  ytickformat = y -> string.(round.(y .+ lat_min, digits=6)),#  yticklabelfont = 13,
    title="Rohodamine Dye Plume",
    titlealign=:left
);
bg = heatmap!(
    ax,
    (satmap.w - lon_min)..(satmap.e - lon_min),
    (satmap.s - lat_min)..(satmap.n - lat_min),
    satmap.img
)

# h = heatmap!(ax, (lon_l-lon_min)..(lon_h-lon_min), (lat_l-lat_min)..(lat_h-lat_min), ns3_plume, colormap=map, colorrange=clims, lowclip=c_high)
h = heatmap!(ax, (lon_l-lon_min)..(lon_h-lon_min), (lat_l-lat_min)..(lat_h-lat_min), ns3_plume, colormap=cmap, colorrange=clims, lowclip=c_high, high_clip=c_low)

Δϕ = 33.7015 - 33.70075
xlims!(ax, -97.7168 - lon_min, -97.7145 - lon_min)
ylims!(ax, 33.7015 - lat_min, 33.7035 - lat_min)

# add 30 meter scale bar
lines!(ax, [λ_scale_l - lon_min, λ_scale_r - lon_min], [ϕ_scale + Δϕ - lat_min,  ϕ_scale + Δϕ - lat_min], color=:white, linewidth=5)
text!(ax, λ_scale_l - lon_min, ϕ_scale + Δϕ - 0.0001 - lat_min, text = "30 m", color=:white, fontsize=12, font=:bold)

# add North arrow
scatter!(ax, [λ_scale_l - lon_min + 0.00003,], [ϕ_scale - lat_min + Δϕ + 0.000075], color=:white, marker=:utriangle, markersize=15)
text!(ax, [λ_scale_l - lon_min,], [ϕ_scale + Δϕ - lat_min + 0.000125], text="N", color=:white, fontsize=12, font=:bold)

cb = Colorbar(fig[1,2], colormap=cmap, colorrange=clims, lowclip=c_high, highclip=c_low, label="Normalized Spectral Similarity Score")
text!(fig.scene, 0.625, 0.905, text="Total Area = $(round(tot_area, digits=1)) m²", space=:relative, )

fig

save(joinpath(figures_path, "plume-map-1.png"), fig)




# plot for plume 2
R1, X1, Y1, Lon1, Lat1, ij_inbounds1, IsInbounds1 = get_data_for_heatmap(f_list_dye[2]);
R2, X2, Y2, Lon2, Lat2, ij_inbounds2, IsInbounds2 = get_data_for_heatmap(f_list_dye[3]);


ns3_plume1 = zeros(size(R1,2), size(R1,3));
ns3_plume2 = zeros(size(R2,2), size(R2,3));

for i ∈ axes(ns3_plume1,1), j ∈ axes(ns3_plume1,2)
    ns3 = NS3(R1[:,i,j], R_plume)
    ns3_plume1[i,j] = ns3
    if ns3 ≤ ns3_thresh
        ns3_plume1[i,j] = ns3
    else
        ns3_plume1[i,j] = NaN
    end
end

for i ∈ axes(ns3_plume2,1), j ∈ axes(ns3_plume2,2)
    ns3 = NS3(R2[:,i,j], R_plume)
    ns3_plume2[i,j] = ns3
    if ns3≤ ns3_thresh
        ns3_plume2[i,j] = ns3
    else
        ns3_plume2[i,j] = NaN
    end
end

idx_good1 = findall(ns3_plume1 .≤ ns3_thresh)
idx_good2 = findall(ns3_plume2 .≤ ns3_thresh)
tot_area = (length(idx_good1) + length(idx_good2)) * 0.1 * 0.1

lon_min, lon_max = (-97.7168, -97.7145)
lat_min, lat_max = (33.7015, 33.7035)

lon_l1, lon_h1 = extrema(Lon1)
lat_l1, lat_h1 = extrema(Lat1)
lon_l2, lon_h2 = extrema(Lon2)
lat_l2, lat_h2 = extrema(Lat2)



fig = Figure(; size=(800, 600));
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="Longitude", xtickformat = x -> string.(round.(x .+ lon_min, digits=6)),#  xticklabelfont = 13,
    ylabel="Latitude",  ytickformat = y -> string.(round.(y .+ lat_min, digits=6)),#  yticklabelfont = 13,
    title="Rohodamine Dye Plume",
    titlealign=:left
);
bg = heatmap!(
    ax,
    (satmap.w - lon_min)..(satmap.e - lon_min),
    (satmap.s - lat_min)..(satmap.n - lat_min),
    satmap.img
)

heatmap!(ax, (lon_l1-lon_min)..(lon_h1-lon_min), (lat_l1-lat_min)..(lat_h1-lat_min), ns3_plume1, colormap=cmap, colorrange=clims, lowclip=c_high, highclip=c_low)
heatmap!(ax, (lon_l2-lon_min)..(lon_h2-lon_min), (lat_l2-lat_min)..(lat_h2-lat_min), ns3_plume2, colormap=cmap, colorrange=clims, lowclip=c_high, highclip=c_low)

Δϕ = 33.7015 - 33.70075
xlims!(ax, -97.7168 - lon_min, -97.7145 - lon_min)
ylims!(ax, 33.7015 - lat_min, 33.7035 - lat_min)

# add 30 meter scale bar
lines!(ax, [λ_scale_l - lon_min, λ_scale_r - lon_min], [ϕ_scale + Δϕ - lat_min,  ϕ_scale + Δϕ - lat_min], color=:white, linewidth=5)
text!(ax, λ_scale_l - lon_min, ϕ_scale + Δϕ - 0.0001 - lat_min, text = "30 m", color=:white, fontsize=12, font=:bold)

# add North arrow
scatter!(ax, [λ_scale_l - lon_min + 0.00003,], [ϕ_scale - lat_min + Δϕ + 0.000075], color=:white, marker=:utriangle, markersize=15)
text!(ax, [λ_scale_l - lon_min,], [ϕ_scale + Δϕ - lat_min + 0.000125], text="N", color=:white, fontsize=12, font=:bold)


cb = Colorbar(fig[1,2], colormap=cmap, colorrange=clims, lowclip=c_high, highclip=c_low, label="Normalized Spectral Similarity Score")
text!(fig.scene, 0.625, 0.905, text="Total Area = $(round(tot_area, digits=1)) m²", space=:relative, )

fig

save(joinpath(figures_path, "plume-map-2.png"), fig)







# plot for algae
c_low = colorant"#b5faaf"
c_high = colorant"#11ab03"
cmap = cgrad([c_high, c_low])
ns3_thresh = 0.4275
clims = (0.25, ns3_thresh)

# lon_min, lon_max = (-97.7168, -97.7125)
# lat_min, lat_max = (33.70075, 33.7035)

#lon_min, lon_max = (-97.7168, -97.7145)
lon_min, lon_max = (-97.7165, -97.7150)
# lat_min, lat_max = (33.7015, 33.7035)
lat_min, lat_max = (33.7015, 33.7026)


fig = Figure(; size=(800, 600));
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="Longitude", xtickformat = x -> string.(round.(x .+ lon_min, digits=6)),#  xticklabelfont = 13,
    ylabel="Latitude",  ytickformat = y -> string.(round.(y .+ lat_min, digits=6)),#  yticklabelfont = 13,
    title="Algae Distribution",
    titlealign=:left
);
bg = heatmap!(
    ax,
    (satmap.w - lon_min)..(satmap.e - lon_min),
    (satmap.s - lat_min)..(satmap.n - lat_min),
    satmap.img
)

Δλ = 0.00025
Δϕ = 33.7015 - 33.70075
xlims!(ax, 0, lon_max - lon_min)
ylims!(ax, 0, lat_max - lat_min)

# add 30 meter scale bar
lines!(ax, [λ_scale_l - lon_min + Δλ, λ_scale_r - lon_min + Δλ], [ϕ_scale + Δϕ - lat_min,  ϕ_scale + Δϕ - lat_min], color=:white, linewidth=5)
text!(ax, λ_scale_l - lon_min + Δλ, ϕ_scale + Δϕ - 0.00005 - lat_min, text = "30 m", color=:white, fontsize=12, font=:bold)

# add North arrow
scatter!(ax, [λ_scale_l - lon_min + Δλ + 0.0000115,], [ϕ_scale - lat_min + Δϕ + 0.00003], color=:white, marker=:utriangle, markersize=15)
text!(ax, [λ_scale_l - lon_min + Δλ,], [ϕ_scale + Δϕ - lat_min + 0.00005], text="N", color=:white, fontsize=12, font=:bold)


λ_l, λ_r = [-97.71575,-97.715621]
ϕ_b, ϕ_t = [ϕ_scale + Δϕ - 0.00005, 33.701875]

# scatter!(ax, [λ_l, λ_l, λ_r, λ_r] .- lon_min, [ϕ_b, ϕ_t, ϕ_b, ϕ_t] .- lat_min, markersize=15, color=:white)
# fig

# flist_to_use = vcat(f_list_1[1:7], f_list_2[1:7], f_list_dye)
# flist_to_use = vcat(f_list_1[1:7], f_list_dye[end])
flist_to_use = vcat(f_list_1[1:7], "/Users/johnwaczak/data/robot-team/processed/hsi/12-09/Dye_2/Dye_2-1.h5", f_list_dye[1])
f_list_dye


n_good = 0

@showprogress for h5path ∈ flist_to_use
    R, X, Y, Lon, Lat, ij_inbounds, IsInbounds = get_data_for_heatmap(h5path)


    ns3_algae= zeros(size(R,2), size(R,3));
    ns3_algae[:,:] .= NaN;
    for i ∈ axes(ns3_algae, 1), j ∈ axes(ns3_algae,2)
        ns3 = NS3(R[:, i, j], R_algae)
        if ns3 ≥ clims[1] && ns3 ≤ clims[2]
            ns3_algae[i, j] = ns3
        end


        # check if we're in the bounding box to ignore the Camper
        if λ_l ≤ Lon[i,j] && Lon[i,j] ≤ λ_r && ϕ_b ≤ Lat[i,j] && Lat[i,j] ≤ ϕ_t
            ns3_algae[i,j] = NaN
        end
    end

    n_good += length(findall(ns3_algae .≥ clims[1] .&& ns3_algae .≤ clims[2]))

    lon_l, lon_h = extrema(Lon)
    lat_l, lat_h = extrema(Lat)

    h = heatmap!(
        ax,
        (lon_l-lon_min)..(lon_h-lon_min),
        (lat_l-lat_min)..(lat_h-lat_min),
        ns3_algae,
        colormap=cmap,
        colorrange=clims,
        highclip=c_low,
        lowclip=c_high
    )
end

# add total area to top of plot
tot_area = (n_good) * 0.1 * 0.1
text!(fig.scene, 0.625, 0.905, text="Total Area = $(round(tot_area, digits=1)) m²", space=:relative, )

cb =  Colorbar(fig[1,2], colormap=cmap, colorrange=clims, highclip=c_low, lowclip=c_high, label="Normalized Spectral Similarity Score")

fig

save(joinpath(figures_path, "algae.png"), fig)





# class map (w/ land):



# class map (w/o land):
