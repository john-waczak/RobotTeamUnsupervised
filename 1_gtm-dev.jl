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



gtm = GTM(10, 4, 2, X)

getLatentMeans(gtm)

D = getNodeDistances(gtm, X)
@benchmark getNodeDistances(gtm, X)
@benchmark getNodeDistances!(D, gtm, X)


R = Responsabilities(gtm, X)

R2 = zeros(size(R));
Rtmp = sum(R2, dims=1)
Rtmp = maximum(R2, dims=1)
Responsabilities!(R2, Rtmp, D, gtm, X)

R2 == R

@benchmark Responsabilities(gtm, X)
@benchmark Responsabilities!(R2, Rtmp, D, gtm, X)

@assert R2 == R

diagm(sum(R, dims=2)[:])

G = getGMatrix(R)
size(G)
size(R)
G2 = zeros(size(G))
getGMatrix!(G2,R)

@benchmark getGMatrix(R)
@benchmark getGMatrix!(G2, R)

diag(G) == sum(R, dims=2)[:]
diag(G2) == sum(R, dims=2)[:]


G2[end,end]
