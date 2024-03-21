using HDF5
using Statistics
using CairoMakie
using GLMakie
using GeometryBasics
using ProgressMeter
using CSV, DataFrames
using JSON

include("utils/makie-defaults.jl")
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


h5_basepath = "/Users/johnwaczak/data/robot-team/processed/hsi"
@assert ispath(h5_basepath)



# generate file lists
files_dict = Dict(
    "11-23" => Dict(
        "no-dye" => [
            joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-2.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-7.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-13.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-14.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_2", "Scotty_2-1.h5"),
        ],
        "dye" => [
            joinpath(h5_basepath, "11-23", "Scotty_4", "Scotty_4-1.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_5", "Scotty_5-1.h5"),
        ],
    ),
    "12-09" => Dict(
        "no-dye" => [
            joinpath(h5_basepath, "12-09", "NoDye_1", "NoDye_1-4.h5"),
            joinpath(h5_basepath, "12-09", "NoDye_2", "NoDye_2-2.h5"),
        ],
        "dye" => [
            joinpath(h5_basepath, "12-09", "Dye_1", "Dye_1-6.h5"),
            joinpath(h5_basepath, "12-09", "Dye_2", "Dye_2-5.h5"),
        ],
    ),
    "12-10" => Dict(
        "no-dye" => [
            joinpath(h5_basepath, "12-10", "NoDye_1", "NoDye_1-1.h5"),
            joinpath(h5_basepath, "12-10", "NoDye_2", "NoDye_2-20.h5"),
        ],
        "dye" => [
            joinpath(h5_basepath, "12-10", "Dye_1", "Dye_1-6.h5"),
            joinpath(h5_basepath, "12-10", "Dye_2", "Dye_2-1.h5"),
        ],
    ),
)

# make sure files exist
for (day, collections) in files_dict
    for (collection , files) in collections
        for f in files
            @assert ispath(f)
        end
    end
end



# set up savepath for figures
figs_path = "./figures/datacubes"
if !ispath(figs_path)
    mkpath(figs_path)
end
@assert ispath(figs_path)


# visualize each of the datacubes with dye to find the best one
h5path = files_dict["11-23"]["no-dye"][1]
fig1, fig2 = vis_rectified_cube(h5path; azimuth=-π/3, elevation=5π/16)

GLMakie.activate!()
save(joinpath(figs_path, "11-23_Scotty_1-2.png"), fig1)

CairoMakie.activate!()
save(joinpath(figs_path, "11-23_Scotty_1-2__colorbar.svg"), fig2)

# ---

h5path = files_dict["11-23"]["no-dye"][2]
fig1, fig2 = vis_rectified_cube(h5path; azimuth=-π/3, elevation=π/4)

GLMakie.activate!()
save(joinpath(figs_path, "11-23_Scotty_2-1.png"), fig1)

CairoMakie.activate!()
save(joinpath(figs_path, "11-23_Scotty_2-1__colorbar.svg"), fig2)

# ---

h5path = files_dict["11-23"]["dye"][1]
fig1, fig2 = vis_rectified_cube(h5path; azimuth=0)

GLMakie.activate!()
save(joinpath(figs_path, "11-23_Scotty_4-1.png"), fig1)

CairoMakie.activate!()
save(joinpath(figs_path, "11-23_Scotty_4-1__colorbar.svg"), fig2)

# ---

h5path = files_dict["11-23"]["dye"][2]

GLMakie.activate!()
fig1, fig2 = vis_rectified_cube(h5path; azimuth=-π/3, elevation=π/4)
save(joinpath(figs_path, "11-23_Scotty_5-1.png"), fig1)

CairoMakie.activate!()
save(joinpath(figs_path, "11-23_Scotty_5-1__colorbar.svg"), fig2)

# ---

h5path = files_dict["12-09"]["dye"][1]

h5 = h5open(h5path, "r")
Δx = 0.1
xs = h5["data-Δx_$(Δx)/X"][:];
ys = h5["data-Δx_$(Δx)/Y"][:];
close(h5)

GLMakie.activate!()
fig1, fig2 = vis_rectified_cube(h5path; azimuth=3π/4, elevation=3π/16, ibounds=(100, length(xs)))

save(joinpath(figs_path, "12-09_Dye_1-6.png"), fig1)

CairoMakie.activate!()
save(joinpath(figs_path, "12-09_Dye_1-6__colorbar.svg"), fig2)


h5 = h5open(h5path, "r")
Δx = 0.1
xs = h5["data-Δx_$(Δx)/X"][:];
ys = h5["data-Δx_$(Δx)/Y"][:];
Data = h5["data-Δx_$(Δx)/Data"][:, :, :];
λs = h5["data-Δx_$(Δx)/λs"][:];

close(h5)
# get x,y coords for important points and save for later
important_coords = Dict(
    "plume" => Dict(
        "x" => xs[80],
        "y" => ys[225],
        "R" => Data[1:462, 80, 225] ./ maximum(Data[1:462, 80, 225]),
        "λs" => λs,
    ),
    "water" => Dict(
        "x" => xs[400],
        "y" => ys[320],
        "R" => Data[1:462, 400, 320] ./ maximum(Data[1:462, 400, 320]),
        "λs" => λs,
    ),
    "algae" => Dict(
        "x" => xs[186],
        "y" => ys[45],
        "R" => Data[1:462, 186, 45] ./ maximum(Data[1:462, 186, 45]),
        "λs" => λs,
    ),
    "grass" => Dict(
        "x" => xs[400],
        "y" => xs[80],
        "R" => Data[1:462, 400, 80] ./ maximum(Data[1:462, 400, 80]),
        "λs" => λs,
    ),
)

# visuzlie the spectra

CairoMakie.activate!()
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="λ (nm)", ylabel="Scaled Reflectance",);
lines!(ax, important_coords["algae"]["λs"], important_coords["algae"]["R"], linewidth=2, label="Algae")
lines!(ax, important_coords["plume"]["λs"], important_coords["plume"]["R"], linewidth=2, label="Rhodamine")
lines!(ax, important_coords["water"]["λs"], important_coords["water"]["R"], linewidth=2, label="Water")
lines!(ax, important_coords["grass"]["λs"], important_coords["grass"]["R"], linewidth=2, label="Grass", color=:brown)
axislegend(ax, position=:lt, labelsize=13)
xlims!(ax, important_coords["algae"]["λs"][1], important_coords["algae"]["λs"][end])
fig

save(joinpath(figs_path, "sample-spectra.pdf"), fig)
save(joinpath(figs_path, "sample-spectra.png"), fig)


# write the dict to a json file
open("data/robot-team/sample-coords.json", "w") do f
    JSON.print(f, important_coords)
end

# ---

h5path = files_dict["12-09"]["dye"][2]

h5 = h5open(h5path, "r")
Δx = 0.1
xs = h5["data-Δx_$(Δx)/X"][:];
ys = h5["data-Δx_$(Δx)/Y"][:];
close(h5)

GLMakie.activate!()
fig1, fig2 = vis_rectified_cube(h5path; azimuth=3π/4, elevation=1π/8,)
save(joinpath(figs_path, "12-09_Dye_2-5.png"), fig1)

CairoMakie.activate!()
save(joinpath(figs_path, "12-09_Dye_2-5__colorbar.svg"), fig2)


# ---

h5path = files_dict["12-10"]["dye"][1]

GLMakie.activate!()
fig1, fig2 = vis_rectified_cube(h5path; azimuth=3π/4, elevation=3π/16,)

fig1

save(joinpath(figs_path, "12-10_Dye_1-6.png"), fig1)

CairoMakie.activate!()
save(joinpath(figs_path, "12-10_Dye_1-6__colorbar.svg"), fig2)



# ---

h5path = files_dict["12-10"]["dye"][2]

h5 = h5open(h5path, "r")
Δx = 0.1
xs = h5["data-Δx_$(Δx)/X"][:];
ys = h5["data-Δx_$(Δx)/Y"][:];
close(h5)

GLMakie.activate!()
fig1, fig2 = vis_rectified_cube(h5path; azimuth=-π/3, elevation=2π/16,)
save(joinpath(figs_path, "12-10_Dye_2-1.png"), fig1)

CairoMakie.activate!()
save(joinpath(figs_path, "12-10_Dye_2-1__colorbar.svg"), fig2)





# generate dataset using these selected datacubes

function get_h5_data(h5path, Δx = 0.1, skip_size = 5)
    # open file in read mode
    h5 = h5open(h5path, "r")

    # extract data
    varnames = read(h5["data-Δx_$(Δx)/varnames"])
    Data = read(h5["data-Δx_$(Δx)/Data"])[:, :, :]
    IsInbounds = read(h5["data-Δx_$(Δx)/IsInbounds"])
    Longitudes = read(h5["data-Δx_$(Δx)/Longitudes"])
    Latitudes = read(h5["data-Δx_$(Δx)/Latitudes"])
    xs = read(h5["data-Δx_$(Δx)/X"])
    ys = read(h5["data-Δx_$(Δx)/Y"])

    # close file
    close(h5)

    # generate indices for sampling along grid in x-y space at
    # a spacing given by Δx * skip_size
    IsSkipped = zeros(Bool, size(Data,2), size(Data,3))
    IsSkipped[1:skip_size:end, 1:skip_size:end] .= true

    # only keep pixels within boundary and at skip locations
    ij_inbounds = findall(IsInbounds .&& IsSkipped)

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
        R_max = maximum(Data[1:462, j])
        Data[1:462, j] .= Data[1:462, j] ./ R_max
    end

    X = X[ij_inbounds]
    Y = Y[ij_inbounds]
    Longitudes = Longitudes[ij_inbounds]
    Latitudes = Latitudes[ij_inbounds]

    df_h5 = DataFrame(Data', varnames)
    df_h5.x = X
    df_h5.y = Y
    df_h5.longitude = Longitudes
    df_h5.latitude = Latitudes

    return df_h5
end




dfs = []

# loop over files and produce dataframes
for (day, collections) in files_dict
    for (collection , files) in collections
        for f in files
            println("Working on $(f)")
            df = get_h5_data(f)
            push!(dfs, df)
        end
    end
end


df_out = vcat(dfs...);

println(nrow(df_out))


out_path = abspath("data/robot-team/unsupervised/data")
if !ispath(out_path)
    mkpath(out_path)
end


# feature_names = ["R_"*lpad(i, 3, "0") for i ∈ 1:462]
# target_names = ["roll", "pitch", "heading", "altitude", "view_angle", "solar_azimuth", "solar_elevation", "solar_zenith", "Σrad", "Σdownwelling", "x", "y", "longitude", "latitude"]

# df_features = df_out[:, feature_names];
# df_targets = df_out[:, target_names];


df_features = df_out[:, 1:462];
df_targets = df_out[:, 463:end];



CSV.write(joinpath(out_path, "df_features.csv"), df_features)
CSV.write(joinpath(out_path, "df_targets.csv"), df_targets)


