using CSV, DataFrames, DelimitedFiles
using MLJ
using GenerativeTopographicMapping
using CairoMakie

using JSON
using ArgParse
using Random
using ProgressMeter
using LaTeXStrings
using MultivariateStats
using Distances

include("utils/makie-defaults.jl")
include("utils/config.jl")

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

X = CSV.read(joinpath(datapath, "data", "df_features.csv"), DataFrame);
Y = CSV.read(joinpath(datapath, "data", "df_targets.csv"), DataFrame);

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


idx_slow = findall(df.converged .== false)
df[idx_slow, :]


# find the global optima for BIC and AIC
idx_bic_global = argmin(df.BIC)
df[idx_bic_global,:]

idx_aic_global = argmin(df.AIC)
df[idx_aic_global,:]

k_best = df.k[idx_bic_global]
m_best = df.m[idx_bic_global]
s_best = df.s[idx_bic_global]
α_best = df.α[idx_bic_global]


k_best
m_best
s_best
α_best



unique(df.s)
unique(df.α)

savepath = joinpath("./figures", "hp-comparison")
if !ispath(savepath)
    mkpath(savepath)
end


# visualize BIC vs m for fixed k, α, s

fig = Figure();
ax = Axis(fig[1,1], xlabel="m", ylabel="Bayesian Information Criterion");
xlims!(ax, minimum(df.m), maximum(df.m))

df_fixed = df[df.α .== α_best .&& df.s .== s_best, :];
gdf = groupby(df_fixed, :k);

for k ∈ [4,8,16,32]
    df_k = gdf[(k=k,)]
    sort!(df_k, :m)
    lines!(ax, df_k.m, df_k.BIC, label="k = $(k)", linewidth=3)
end

axislegend(ax; position=:lt)
fig

save(joinpath(savepath, "bic-vs-m.png"), fig)
save(joinpath(savepath, "bic-vs-m.svg"), fig)
save(joinpath(savepath, "bic-vs-m.pdf"), fig)


# visualize BIC vs s for fixed m, k, α
fig = Figure();
ax = Axis(fig[1,1], xlabel="scale factor", ylabel="Bayesian Information Criterion");
xlims!(ax, minimum(df.s), maximum(df.s))

df_fixed = df[df.k .== k_best .&& df.m .== m_best .&& df.α .== α_best, :]
lines!(ax, df_fixed.s, df_fixed.BIC, linewidth=3)
fig

save(joinpath(savepath, "bic-vs-s.png"), fig)
save(joinpath(savepath, "bic-vs-s.svg"), fig)
save(joinpath(savepath, "bic-vs-s.pdf"), fig)


# visualize BIC vs α for fixed m, k, s
fig = Figure();
ax = Axis(fig[1,1], xlabel="α", ylabel="Bayesian Information Criterion");
xlims!(ax, minimum(df.α), maximum(df.α))

df_fixed = df[df.k .== k_best .&& df.m .== m_best .&& df.s .== s_best, :]
lines!(ax, df_fixed.α, df_fixed.BIC, linewidth=3)
fig

save(joinpath(savepath, "bic-vs-alpha.png"), fig)
save(joinpath(savepath, "bic-vs-alpha.svg"), fig)
save(joinpath(savepath, "bic-vs-alpha.pdf"), fig)



# add code to generate summary table
# with:
# parameter name, values searched, optimal value




# make final output directory
savepath = joinpath("./figures", "model-final")
if !ispath(savepath)
    mkpath(savepath)
end


Xdata = X
# Xdata = Xall

# headers = names(X)
# Xdata = copy(Matrix(X))

# for i ∈ axes(Xdata, 1)
#     Xdata[i, :] .= Xdata[i, :] ./ sum(Xdata[i,:])
# end

# Xdata = DataFrame(Xdata, headers)

m = 4
s = 3
k = round(Int, 10*(m-1)/(2*s) + 1)

k = 32

gtm = GTM(k=k, m=m, s=s, α=1.0, tol=1e-5, nepochs=250)
mach = machine(gtm, Xdata)
fit!(mach)



# Make a square plot of the latent points and rbf means
gtm_mdl = fitted_params(mach)[:gtm]
rpt = report(mach)

keys(rpt)

M = gtm_mdl.M                          # RBF centers
Ξ = rpt[:Ξ]                            # Latent Points
Ψ = rpt[:W] * rpt[:Φ]'                 # Projected Node Means
llhs = rpt[:llhs]

Rs = predict_responsibility(mach, Xdata)
mean_proj = DataFrame(MLJ.transform(mach, Xdata))
mode_proj = DataFrame(DataModes(gtm_mdl, Matrix(Xdata)), [:ξ₁, :ξ₂] )
class_id = MLJ.predict(mach, Xdata)

# compute PCA as well
pca = MultivariateStats.fit(PCA, Matrix(Xdata)', maxoutdim=3, pratio=0.99999);
U = MultivariateStats.predict(pca, Matrix(Xdata)')[1:2,:]'




# set up 2-dimensional color map
fig = Figure();
axl = Axis(fig[1,1], xlabel="u₁", ylabel="u₂", title="PCA", aspect=AxisAspect(1.0))
axr = Axis(fig[1,2], xlabel="ξ₁", ylabel="ξ₂", title="GTM ⟨ξ⟩", aspect=AxisAspect(1.0))
scatter!(axl, U[:,1], U[:,2], markersize=5, alpha=0.7)
scatter!(axr, mean_proj.ξ₁, mean_proj.ξ₂, markersize=5, alpha=0.7)
fig

names(Xrest)
names(Y)



# color_col = Y.CDOM[:]
# clims = color_clims["CDOM"]["11-23"]
# cm = cgrad(:roma, rev=true)

# fig = Figure();
# axl = Axis(fig[1,1], xlabel="u₁", ylabel="u₂", title="PCA", aspect=AxisAspect(1.0))
# axr = Axis(fig[1,2], xlabel="ξ₁", ylabel="ξ₂", title="GTM ⟨ξ⟩", aspect=AxisAspect(1.0))
# h1 = scatter!(axl, U[:,1], U[:,2], color=color_col, colorrange=clims, colormap=cm, markersize=5, alpha=0.7)
# h2 = scatter!(axr, mean_proj.ξ₁, mean_proj.ξ₂, color=color_col, colorrange=clims, colormap=cm, markersize=5, alpha=0.7)
# cb = Colorbar(fig[1,3], colorrange=clims, colormap=cm)

# fig


# Rs[1,:]
# scatter(Ξ[:,1], Ξ[:,2], color=Rs[1,:])


# plot log-likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 3:length(llhs), llhs[3:end], linewidth=5)
fig


fig = Figure();
ax = Axis(fig[1,1]);
#xlims!(ax, wavelengths[1], 900)
# ylims!(ax, 0, 0.05)
size(Ψ)

for i ∈ [1, 32, 1024-31, 1024]
    lines!(ax, wavelengths, Ψ[:,i])
end

fig


size(Rs)
size(Ψ)

X̂ = Ψ * Rs'
Xmat = Matrix(Xdata)
Σs = svdvals(Xmat)


using LinearAlgebra


Δ² = colwise(sqeuclidean, X̂, Xmat')

rms_error = sqrt(mean(Δ²))

Rs[1,:]


fig = Figure();
ax = Axis(fig[1,1], xlabel="wavelength", ylabel="Reflectance");
l1 = lines!(ax, wavelengths, Xmat[1,:])
l2 = lines!(ax, wavelengths, X̂[:,1])
axislegend(ax, [l1, l2], ["original", "reconstructed"])

fig

size(Xmat)


fig

μ = Rs' * Matrix(X)

sum(Rs, dims=1)

size(Rs)
size(μ)


for i ∈ axes(Rs,2)
    μ[i,:] .= μ[i,:] ./ sum(Rs[:,i])
end

fig

fig = Figure();
gl = fig[1,1] = GridLayout();
ax = Axis(gl[1,1], xlabel="λ (nm)", ylabel="Reflectance");
ax2 = Axis(gl[1,2], xlabel="λ (nm)")

idx_900 = findfirst(wavelengths .≥ 900)
xlims!(ax, wavelengths[1], wavelengths[idx_900])
xlims!(ax2, wavelengths[idx_900], wavelengths[end])

colsize!(gl, 1, Relative(0.7))

#for i ∈ [1, 32, 1024-31, 1024]
for i ∈ [1, k, k^2-k, k^2]
    lines!(ax, wavelengths[1:idx_900], μ[i,1:idx_900])
    lines!(ax2, wavelengths[idx_900:end], μ[i,idx_900:end])
end

fig


mean(sum(Rs .> 0, dims=2))
