using HDF5
using CSV, DataFrames
using ProgressMeter

include("./config.jl")


hsi_to_use = [
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
    #----------------
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
    "Scotty_2-20.h5",
    "Scotty_2-22.h5",
    "Scotty_2-24.h5",
    #----------------
    "Scotty_3-1.h5",
    #----------------
    "Scotty_4-1.h5",
    #----------------
    "Scotty_5-1.h5",
    "Scotty_5-2.h5",
    "Scotty_5-2.h5",
]


hsipath = "/Users/johnwaczak/data/robot-team/processed/hsi"


hsi_list = []

for (root, dirs, files) ∈ walkdir(hsipath)
    for file ∈ files
        if file ∈ hsi_to_use
            println(file)
            push!(hsi_list, joinpath(root, file))
        end
    end
end



function get_h5_data(h5path, Δx = 0.1, skip_size = 5)
    # open file in read mode
    h5 = h5open(hsi_list[1], "r")

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

    Data = Data[:, ij_inbounds]
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

@showprogress for h5_path ∈ hsi_list
    df_h5 = get_h5_data(h5_path);
    push!(dfs, df_h5)
end

df_out = vcat(dfs...)

out_path = abspath("data/robot-team/unsupervised/data")
if !ispath(out_path)
    mkpath(out_path)
end


feature_names = ["R_"*lpad(i, 3, "0") for i ∈ 1:462]
target_names = ["roll", "pitch", "heading", "altitude", "view_angle", "solar_azimuth", "solar_elevation", "solar_zenith", "Σrad", "Σdownwelling", "x", "y", "longitude", "latitude"]


df_features = df_out[:, feature_names]
df_targets = df_out[:, target_names]


CSV.write(joinpath(out_path, "df_features.csv"), df_features)
CSV.write(joinpath(out_path, "df_targets.csv"), df_targets)
