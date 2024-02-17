using CSV, DataFrames, DelimitedFiles
using MLJ
using GenerativeTopographicMapping
using CairoMakie, MintsMakieRecipes
using JSON
using ArgParse
using Random
using ProgressMeter
using LaTeXStrings

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


datapath="./data/robot-team/unsupervised"
@assert ispath(datapath)

X = CSV.read(joinpath(datapath, "data", "df_features.csv"), DataFrame)
# limit to reflectances only
X = X[:, 1:462]

model_path = joinpath(datapath, "models", "refs_only")
@assert ispath(model_path)



json_files = []

for (root, dirs, files) ∈ walkdir(model_path)
    for file ∈ files
        if endswith(file, ".json")
            push!(json_files, joinpath(root, file))
        end
    end
end

println("There are ", length(json_files), " many files")


# generate dataframe summarizing the results
nfiles = length(json_files)
df = DataFrame(
    :k => zeros(Int, nfiles),
    :m => zeros(Int, nfiles),
    :llh_max => zeros(nfiles),
    :BIC => zeros(nfiles),
    :AIC => zeros(nfiles),
    :α => zeros(nfiles),
    :s => zeros(nfiles),
    :converged => Bool[false for _ in 1:nfiles],
)

@showprogress for i ∈ axes(json_files, 1)
    res = JSON.parsefile(json_files[i])
    df.k[i] = res["k"]
    df.m[i] = res["m"]
    df.llh_max[i] = res["llhs"][end]
    df.BIC[i] = res["BIC"]
    df.AIC[i] = res["AIC"]
    df.α[i] = res["α"]
    df.s[i] = res["s"]
    df.converged[i] = res["converged"]
end


# save the CSV to the data directory
summary_path = joinpath(datapath, "models", "refs_only", "summary")
if !ispath(summary_path)
    mkpath(summary_path)
end

CSV.write(joinpath(summary_path, "fit-summary.csv"), df)


# Make sure all the fits actually converged
@assert all(df.converged)


# find the global optima for BIC and AIC
idx_bic_global = argmin(df.BIC)
df[idx_bic_global,:]

idx_aic_global = argmin(df.AIC)
df[idx_aic_global,:]

k_best = df.k[idx_bic_global]
m_best = df.m[idx_bic_global]
s_best = df.s[idx_bic_global]
α_best = df.α[idx_bic_global]


# the hyperparameters we varied were k, m, s, and α. Lets group the dataframe by s and α so we can plot the AIC and BIC for each
gdf = groupby(df, [:s, :α])

# get the k and m corresponding to the lowest BIC/AIC for each pair of α and s
bic_aics = []
for (key, subdf) in pairs(gdf)
    res = Dict()
    res[:s] = key[:s]
    res[:α] = key[:α]

    idx_bic = argmin(subdf.BIC)
    idx_aic = argmin(subdf.AIC)

    res[:kb] = subdf.k[idx_bic]
    res[:mb] = subdf.m[idx_bic]
    res[:BIC] = subdf.BIC[idx_bic]

    res[:ka] = subdf.k[idx_aic]
    res[:ma] = subdf.m[idx_aic]
    res[:AIC] = subdf.BIC[idx_aic]

    push!(bic_aics, DataFrame(res))
end

df_best = vcat(bic_aics...)


# visualize the dependence of the BIC and AIC on α and s
savepath = joinpath("./figures", "hp-comparison")
if !ispath(savepath)
    mkpath(savepath)
end

fig = Figure();
ax = Axis(fig[1,1], xlabel="α", ylabel="s");
s = scatter!(ax, df_best.α, df_best.s, color = df_best.BIC, markersize=50)

idx_bic = argmin(df_best.BIC)
ab = df_best.α[idx_bic]
sb = df_best.s[idx_bic]

sl = scatter!(ax, Point2f(ab, sb), marker = :star5, markersize=25, color=:white, strokecolor=:gray, strokewidth=1)
cb = Colorbar(fig[1,2], s, label="Bayesian Information Criterion", ticks = ([extrema(df_best.BIC)...], ["Low", "High"]))

save(joinpath(savepath, "bic-comparison.png"), fig)
save(joinpath(savepath, "bic-comparison.pdf"), fig)

fig


fig = Figure();
ax = Axis(fig[1,1], xlabel="α", ylabel="s");
s = scatter!(ax, df_best.α, df_best.s, color = df_best.AIC, markersize=50)

idx_aic = argmin(df_best.AIC)
aa = df_best.α[idx_aic]
sa = df_best.s[idx_aic]

sl = scatter!(ax, Point2f(aa, sa), marker = :star5, markersize=25, color=:white, strokecolor=:gray, strokewidth=1)
cb = Colorbar(fig[1,2], s, label="Akaike Information Criterion", ticks = ([extrema(df_best.AIC)...], ["Low", "High"]))

save(joinpath(savepath, "aic-comparison.png"), fig)
save(joinpath(savepath, "aic-comparison.pdf"), fig)

fig


subdf = gdf[(s=s_best, α=α_best,)]


idx_bic = argmin(subdf.BIC)
idx_aic = argmin(subdf.AIC)
k_b = subdf.k[idx_bic]
m_b = subdf.m[idx_bic]
k_a = subdf.k[idx_aic]
m_a = subdf.m[idx_aic]

unique(subdf.k)
k_b

fig = Figure();
ax = Axis(fig[1,1], xlabel="k (k² Latent Nodes)", ylabel="m (m² RBF Centers)", title="GTM Hyperparameter Results\ns=$(s_best), α=$(α_best)");
#h = contourf!(ax, subdf.k, subdf.m, subdf.BIC; levels=15)
h = heatmap!(ax, subdf.k, subdf.m, subdf.BIC)
c = contour!(ax, subdf.k, subdf.m, subdf.AIC; levels=10, color=:white)
s = scatter!(ax, Point2f(k_b, m_b), marker = :star5, markersize=25, color=:white, strokecolor=:gray, strokewidth=1)
s2 = scatter!(ax, Point2f(k_a, m_a), marker = :star6, markersize=25, color=:white, strokecolor=:gray, strokewidth=1)
cb = Colorbar(fig[1,2], h; label="Bayesian Information Criterion", ticks = ([extrema(subdf.BIC)...], ["Low", "High"]))
xlims!(ax, extrema(subdf.k))
ylims!(ax, extrema(subdf.m))

fig

k_b*k_b
m_b

# since the number of parameters don't actually depend on k (they only depend on m since Ψ = WΦ')

"let's instead plot"


# let's do these K values and see what happens "just" when we vary m, s, and α
4
8
16
32



df_fixed = df[df.α .== 0.1 .&& df.s .== 0.1, :]
gdf = groupby(df_fixed, :k)

fig = Figure();
ax = Axis(fig[1,1], xlabel="m", ylabel="Bayesian Information Criterion");
for k ∈ [4,]
    df_k = gdf[(k=k,)]
    df_k = df_k[df_k.m .≤ 32, :]
    sort!(df_k, :m)

    lines!(ax, df_k.m, df_k.BIC, label="k = $(k)")
end

axislegend(ax; position=:lt)
fig

unique(df.s)
