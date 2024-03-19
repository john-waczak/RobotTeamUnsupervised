using CSV, DataFrames, DelimitedFiles
using MLJ
using GenerativeTopographicMapping
using CairoMakie
using JSON


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




# square topology
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
save(joinpath(figures_path, "square-llhs.pdf"), fig)



fig = Figure();
ax = Axis(
    fig[1,1],
    xlabel="ξ₁",
    ylabel="ξ₂",
    title="GTM Means"
)
scatter!(ax, df_res.ξ₁, df_res.ξ₂, color=df_res.mode_class)

fig

save(joinpath(figures_path, "square-means.pdf"), fig)




# cylindrical topology
gtm = GTM(k=10, m=4, tol=1e-5, nepochs=100, topology=:cylinder)
mach = machine(gtm, X)
fit!(mach)

df_res = DataFrame(MLJ.transform(mach, X))
df_res.mode_class = get.(MLJ.predict(mach, X))

Rs = predict_responsibility(mach, X);
rpt = report(mach);
llhs = rpt[:llhs];
Ξ = rpt[:Ξ];

fig = Figure();
ax = Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 1:length(llhs), llhs, linewidth=5)
fig

save(joinpath(figures_path, "cylinder-llhs.pdf"), fig)



fig = Figure();
ax = Axis(
    fig[1,1],
    xlabel="ξ₁",
    ylabel="ξ₂",
    title="GTM Means"
)
scatter!(ax, df_res.ξ₁, df_res.ξ₂, color=df_res.mode_class)

fig

save(joinpath(figures_path, "cylinder-means.pdf"), fig)


# torus
gtm = GTM(k=10, m=4, tol=1e-8, nepochs=100, topology=:torus)
mach = machine(gtm, X)
fit!(mach)

df_res = DataFrame(MLJ.transform(mach, X))
df_res.mode_class = get.(MLJ.predict(mach, X))

Rs = predict_responsibility(mach, X);
rpt = report(mach);
llhs = rpt[:llhs];
Ξ = rpt[:Ξ];

fig = Figure();
ax = Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 1:length(llhs), llhs, linewidth=5)
fig

save(joinpath(figures_path, "torus-llhs.pdf"), fig)



fig = Figure();
ax = Axis(
    fig[1,1],
    xlabel="ξ₁",
    ylabel="ξ₂",
    title="GTM Means"
)
scatter!(ax, df_res.ξ₁, df_res.ξ₂, color=df_res.mode_class)

fig

df_res.ξ₁

save(joinpath(figures_path, "cylinder-means.pdf"), fig)


