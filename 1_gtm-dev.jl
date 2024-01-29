using CSV, DataFrames
using CairoMakie, MintsMakieRecipes
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


include("gtm.jl")




# load iris datset that we can use to validate the code
df = CSV.read("data/iris.csv", DataFrame)

X = Matrix(df[:, 1:4])
y = df[:,5]

target_labels = unique(y)
column_labels = uppercasefirst.(replace.(names(df)[1:4], "."=>" "))

y = [findfirst(y[i] .== target_labels) for i in axes(y,1)]

# visualize the dataset
fig = Figure()
ax = Axis(
    fig[1,1],
    xlabel=column_labels[1],
    ylabel=column_labels[2],
    title="Iris Datset"
)

idx1 = y .== 1
idx2 = y .== 2
idx3 = y .== 3

sc1 = scatter!(ax, X[idx1,1], X[idx1,2], color=mints_colors[1])
sc2 = scatter!(ax, X[idx2,1], X[idx2,2], color=mints_colors[2])
sc3 = scatter!(ax, X[idx3,1], X[idx3,2], color=mints_colors[3])

axislegend(ax, [sc1, sc2, sc3], target_labels)

fig



k = 8
m = 5
s = 2

gtm = GTM(k, m, s, X)

gtm = GTM(k, m, s, X)
llhs, R = fit!(gtm, X, α = 1, niter=50, tol=0.001)

lines(1:length(llhs), llhs)




means = DataMeans(gtm, X)

# modes = DataModes(gtm, X)



fig = Figure()
ax = Axis(
    fig[1,1],
    xlabel="ξ₁",
    ylabel="ξ₂",
    title="GTM Means"
)

idx1 = y .== 1
idx2 = y .== 2
idx3 = y .== 3

sc1 = scatter!(ax, means[idx1,1], means[idx1,2], color=mints_colors[1])
sc2 = scatter!(ax, means[idx2,1], means[idx2,2], color=mints_colors[2])
sc3 = scatter!(ax, means[idx3,1], means[idx3,2], color=mints_colors[3])

axislegend(ax, [sc1, sc2, sc3], target_labels)

fig



modes = DataModes(gtm, X)

fig = Figure()
ax = Axis(
    fig[1,1],
    xlabel="ξ₁",
    ylabel="ξ₂",
    title="GTM Modes"
)

idx1 = y .== 1
idx2 = y .== 2
idx3 = y .== 3

sc1 = scatter!(ax, modes[idx1,1], modes[idx1,2], color=mints_colors[1])
sc2 = scatter!(ax, modes[idx2,1], modes[idx2,2], color=mints_colors[2])
sc3 = scatter!(ax, modes[idx3,1], modes[idx3,2], color=mints_colors[3])

axislegend(ax, [sc1, sc2, sc3], target_labels)

fig

