using CSV, DataFrames
#using MLJ
#using GenerativeTopographicMapping
using CairoMakie, MintsMakieRecipes

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


include("config.jl")


datapath = "/Users/johnwaczak/data/robot-team/finalized/Full/df_full.csv"
@assert ispath(datapath)

df = CSV.read(datapath, DataFrame);

target_names = keys(targets_dict)


# there are 526 total features
n_features = 526
df_features = df[:, 1:n_features];
df_targets = df[:, n_features+1:end];


targets_to_keep = [
    :Temp3488,
    :SpCond,
    :Ca,
    :HDO,
    :Cl,
    :Na,
    :pH,
    :bg,
    :bgm,
    :CDOM,
    :Chl,
    :OB,
    :ChlRed,
    :CO,
    :Turb3489,
    :RefFuel
]

df_targets = df_targets[:, targets_to_keep];


nλs = length(wavelengths)

# remove spectral/vegetation indices from features
reflectance_names = [Symbol("R_"*lpad(i, 3, "0")) for i ∈ 1:nλs]
features_to_keep = [Symbol("R_"*lpad(i, 3, "0")) for i ∈ 1:nλs]
push!(features_to_keep, :roll)
push!(features_to_keep, :pitch)
push!(features_to_keep, :heading)
push!(features_to_keep, :altitude)
push!(features_to_keep, :view_angle)
push!(features_to_keep, :solar_azimuth)
push!(features_to_keep, :solar_elevation)
push!(features_to_keep, :solar_zenith)
push!(features_to_keep, :Σrad)
push!(features_to_keep, :Σdownwelling)

df_features = df_features[:, features_to_keep];


# let's decide if we need to trim the wavelengths at all
# as λ's past about 900 nm look pretty noisy
df_fi_cdom = CSV.read("/Users/johnwaczak/data/robot-team/supervised/Full/CDOM/models/RandomForestRegressor/default/importance_ranking__vanilla.csv", DataFrame)
df_fi_temp = CSV.read("/Users/johnwaczak/data/robot-team/supervised/Full/Temp3488/models/RandomForestRegressor/default/importance_ranking__vanilla.csv", DataFrame)
df_fi_ca = CSV.read("/Users/johnwaczak/data/robot-team/supervised/Full/Ca/models/RandomForestRegressor/default/importance_ranking__vanilla.csv", DataFrame)


fi_cdom_idx = [findfirst(Symbol.(df_fi_cdom[:,1]) .== name) for name in reflectance_names]
fi_cdom = df_fi_cdom[fi_cdom_idx, 2]

fi_temp_idx = [findfirst(Symbol.(df_fi_temp[:,1]) .== name) for name in reflectance_names]
fi_temp= df_fi_cdom[fi_temp_idx, 2]

fi_ca_idx = [findfirst(Symbol.(df_fi_ca[:,1]) .== name) for name in reflectance_names]
fi_ca= df_fi_cdom[fi_ca_idx, 2]

# set all negative values to zero for convenience
fi_cdom[fi_cdom .< 0.0] .= 0.0
fi_temp[fi_temp.< 0.0] .= 0.0
fi_ca[fi_ca.< 0.0] .= 0.0

fig = Figure()
ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="Permuation Importance")
line1 = lines!(ax, wavelengths, fi_cdom, linewidth=3, alpha=0.85)
line2 = lines!(ax, wavelengths, fi_temp, linewidth=3, alpha=0.85)
line3 = lines!(ax, wavelengths, fi_ca, linewidth=3, alpha=0.85)
axislegend(ax, [line1,line2,line3], ["CDOM", "Temperature", "Ca⁺⁺"])
ylims!(ax, 0, min(maximum(fi_cdom), maximum(fi_temp), maximum(fi_ca)))
xlims!(ax, extrema(wavelengths)...)
fig

outpath = joinpath("./figures", "data")
if !ispath(outpath)
    mkpath(outpath)
end

save(joinpath(outpath, "fi-comparison.png"), fig)
save(joinpath(outpath, "fi-comparison.pdf"), fig)

# save the data
data_path = joinpath("./data", "robot-team", "unsupervised")
if !ispath(data_path)
    mkpath(data_path)
end

CSV.write(joinpath(data_path, "df_features.csv"), df_features)
CSV.write(joinpath(data_path, "df_targets.csv"), df_targets)




