using CSV, DataFrames, DelimitedFiles
using CairoMakie
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

exemplar_spectra = JSON.parsefile("./models/exemplar-spectra.json")


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


CairoMakie.activate!()
fig = Figure(size=(800,600));
ax = CairoMakie.Axis(fig[1,1]);
bg = heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
# add 30 meter scale bar
lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=:white, linewidth=5)
text!(ax, λ_scale_l, ϕ_scale - 0.0001, text = "30 m", color=:white, fontsize=12, font=:bold)
xlims!(ax, -97.7168, -97.7125)
ylims!(ax, 33.70075, 33.7035)
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
    joinpath(hsipath, "12-09", "Dye_1", "Dye_1-6.h5"),
    joinpath(hsipath, "12-09", "Dye_2", "Dye_2-5.h5"),
    joinpath(hsipath, "12-10", "Dye_1", "Dye_1-6.h5"),
    joinpath(hsipath, "12-10", "Dye_2", "Dye_2-1.h5"),
]

@assert all(ispath.(f_list_dye))


function in_water(Datacube, varnames; threshold=0.3)
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


# now we have all of our files that we will
idx_900 = findfirst(wavelengths .≥ 900)
Ro_plume = Float64.(exemplar_spectra["plume"]["R"][1:idx_900])
Ro_algae = Float64.(exemplar_spectra["algae"]["R"][1:idx_900])


R_plume = Float64.(exemplar_spectra["plume"]["Ψ"][1:idx_900])
R_algae = Float64.(exemplar_spectra["algae"]["Ψ"][1:idx_900])


function cosine_distance(Rᵢ, Rⱼ)
    1 - dot(Rᵢ, Rⱼ)/(dot(Rᵢ, Rᵢ) * dot(Rⱼ, Rⱼ))
end


function MSE(Rᵢ, Rⱼ)
    mean((Rᵢ .- Rⱼ).^2)
end

function RMSE(Rᵢ, Rⱼ)
    sqrt(MSE(Rᵢ, Rⱼ))
end



# plume 1 plot
cmap = cgrad(:amp, rev=true)
clims = (0.045, 0.25)

fig = Figure(size=(800,600));
ax = CairoMakie.Axis(fig[1,1], xlabel="Longtiude", ylabel="Latitude", title="Rohodamine Dye Plume", titlealign=:left);
bg = heatmap!(ax, satmap.w..satmap.e, satmap.s..satmap.n, satmap.img);
# add 30 meter scale bar
lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=:white, linewidth=5)
text!(ax, λ_scale_l, ϕ_scale - 0.0001, text = "30 m", color=:white, fontsize=12, font=:bold)
cb = Colorbar(fig[1,2], colormap=cmap, colorrange=clims, label="RMSE")
xlims!(ax, -97.7168, -97.7125)
ylims!(ax, 33.70075, 33.7035)

@showprogress for h5path ∈ vcat(f_list_1, f_list_2)
    R, X, Y, Lon, Lat = get_data_for_pred(h5path)

    if any([occursin(piece, split(h5path, "/")[end]) for piece in ["1-19", "1-20", "1-21", "1-22", "1-23"]])
        idx_use = findall(Lat .> 33.70152)
        R = R[:, idx_use]
        X = X[idx_use]
        Y = Y[idx_use]
        Lon = Lon[idx_use]
        Lat = Lat[idx_use]
    end

    if any([occursin(piece, split(h5path, "/")[end]) for piece in ["1-3", "2-3"]])
        idx_use = findall(Lon .< -97.7152)
        R = R[:, idx_use]
        X = X[idx_use]
        Y = Y[idx_use]
        Lon = Lon[idx_use]
        Lat = Lat[idx_use]
    end

    rmse_plume = [RMSE(R_col, R_plume) for R_col ∈ eachcol(R)]
    s = scatter!(ax, Lon, Lat, color=rmse_plume, markersize=1.5, colormap=cmap, colorrange=clims, highclip=:white)
end


R, X, Y, Lon, Lat = get_data_for_pred(f_list_dye[1])
rmse_plume = [RMSE(R_col, R_plume) for R_col ∈ eachcol(R)]
s = scatter!(ax, Lon, Lat, color=rmse_plume, markersize=1.5, colormap=cmap, colorrange=clims, highclip=:white)

# compute total area:
rmse_max = 0.20
idx_good = findall(rmse_plume .≥ rmse_max)
tot_area = length(idx_good) * 0.1 * 0.1

text!(fig.scene, 0.625, 0.905, text="Total Area = $(tot_area) m²", space=:relative, )

fig

save(joinpath(figures_path, "plume-map-1.png"), fig)
