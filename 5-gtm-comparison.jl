using CSV, DataFrames, DelimitedFiles
using MLJ
using GenerativeTopographicMapping
using CairoMakie

using JSON
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


model_path = joinpath("./models", "param-search")
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
data_path = joinpath("./data", "robot-team")
@assert ispath(data_path)

CSV.write(joinpath(data_path, "fit-summary.csv"), df)


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

k_best_a = df.k[idx_aic_global]
m_best_a = df.m[idx_aic_global]
s_best_a = df.s[idx_aic_global]
α_best_a = df.α[idx_aic_global]




savepath = joinpath("./figures", "hp-comparison")
if !ispath(savepath)
    mkpath(savepath)
end

df_fixed = df[df.α .== α_best, :];

# BIC vs s & m for fixed α
mvals = sort(unique(df.m))
mvals = 2:2:20
svals = sort(unique(df.s))
αvals = sort(unique(df.α))

fig = Figure();
gl = fig[1,1] = GridLayout();
ax = Axis(
    gl[1:2,1],
    xlabel="m", xticks=(mvals, string.(mvals)),
    ylabel="s", yticks=(0:0.5:3, string.(0:0.5:3)),
    title="k=$(k_best), α=$(α_best)",
    titlefont=:regular,
    titlesize=17,
    titlealign=:left,
)

h = heatmap!(ax, df_fixed.m, df_fixed.s, df_fixed.BIC)
s = scatter!(ax, [m_best], [s_best], marker=:star5, color=:white, strokewidth=1, strokecolor=:gray, markersize=15)
lab = Label(gl[1,2], "×1e8", fontsize=12, rotation=π/2,)
cb = Colorbar(gl[2,2], h, label="Bayesian Information Criterion", tickformat= x -> string.(x ./ 1e8), labelsize=16, ticklabelsize=13)
rowgap!(gl, 1)

fig

save(joinpath(savepath, "bic_m-vs-s.png"), fig)
save(joinpath(savepath, "bic_m-vs-s.pdf"), fig)


# AIC vs s & m for fixed α
fig = Figure();
gl = fig[1,1] = GridLayout();
ax = Axis(
    gl[1:2,1],
    xlabel="m", xticks=(mvals, string.(mvals)),
    ylabel="s", yticks=(0:0.5:3, string.(0:0.5:3)),
    # title=L"$k=%$(k_best)$, $\alpha=%$(α_best)$",
    title="k=$(k_best), α=$(α_best)",
    titlefont=:regular,
    titlesize=17,
    titlealign=:left,
)

h = heatmap!(ax, df_fixed.m, df_fixed.s, df_fixed.AIC)
s = scatter!(ax, [m_best], [s_best], marker=:star5, color=:white, strokewidth=1, strokecolor=:gray, markersize=15)
lab = Label(gl[1,2], "×1e8", fontsize=12, rotation=π/2,)
cb = Colorbar(gl[2,2], h, label="Akaike Information Criterion", tickformat= x -> string.(x ./ 1e8), labelsize=16, ticklabelsize=13)
rowgap!(gl, 1)

fig

save(joinpath(savepath, "aic_m-vs-s.png"), fig)
save(joinpath(savepath, "aic_m-vs-s.pdf"), fig)

# BIC vs α & m for fixed s
df_fixed = df[df.s .== s_best, :];

fig = Figure();
gl = fig[1,1] = GridLayout();
ax = Axis(
    gl[1:2,1],
    xlabel="m", xticks=(mvals, string.(mvals)),
    ylabel="log10(α)", #, yscale=log10,
    title="k=$(k_best), s=$(s_best)",
    titlefont=:regular,
    titlesize=17,
    titlealign=:left,
)

h = heatmap!(ax, df_fixed.m, log10.(df_fixed.α), df_fixed.BIC)
s = scatter!(ax, [m_best], [log10(α_best)], marker=:star5, color=:white, strokewidth=1, strokecolor=:gray, markersize=15)
lab = Label(gl[1,2], "×1e8", fontsize=12, rotation=π/2,)
cb = Colorbar(gl[2,2], h, label="Bayesian Information Criterion", tickformat= x -> string.(x ./ 1e8), labelsize=16, ticklabelsize=13)
rowgap!(gl, 1)

fig

save(joinpath(savepath, "bic_m-vs-α.png"), fig)
save(joinpath(savepath, "bic_m-vs-α.pdf"), fig)

# AIC vs α & m for fixed s
fig = Figure();
gl = fig[1,1] = GridLayout();
ax = Axis(
    gl[1:2,1],
    xlabel="m", xticks=(mvals, string.(mvals)),
    ylabel="log10(α)", #, yscale=log10,
    title="k=$(k_best), s=$(s_best)",
    titlefont=:regular,
    titlesize=17,
    titlealign=:left,
)

h = heatmap!(ax, df_fixed.m, log10.(df_fixed.α), df_fixed.AIC)
s = scatter!(ax, [m_best], [log10(α_best)], marker=:star5, color=:white, strokewidth=1, strokecolor=:gray, markersize=15)
lab = Label(gl[1,2], "×1e8", fontsize=12, rotation=π/2,)
cb = Colorbar(gl[2,2], h, label="Akaike Information Criterion", tickformat= x -> string.(x ./ 1e8), labelsize=16, ticklabelsize=13)
rowgap!(gl, 1)

fig

save(joinpath(savepath, "aic_m-vs-α.png"), fig)
save(joinpath(savepath, "aic_m-vs-α.pdf"), fig)


# add code to generate summary table
# with:
# parameter name, values searched, optimal value

sort!(df, :BIC)

function generate_tex_table(df)
    # m, α, s, k, BIC, AIC
    out = "\\begin{table}[H]\n"
    out = out * "  \\caption{This is a table caption.\\label{tab:fit-results}}\n"
    out = out * "  \\begin{adjustwidth}{-\\extralength}{0cm}\n"
    out = out * "  \\newcolumntype{C}{>{\\centering\\arraybackslash}X}\n"
    out = out * "  \\begin{tabularx}{\\fulllength}{CCCCCC}\n"
    out = out * "    \\toprule\n"
    out = out * "    \\textbf{k} & \\textbf{\$\alpha\$} & \\textbf{s} & \\textbf{k} & \\textbf{BIC} & \\textbf{AIC} \\\\\n"
    out = out * "    \\midrule\n"

    for i ∈ 1:25
        row = df[i, :]

        ki = row.k
        mi = row.m
        αi = row.α
        si = row.s
        bic_i = string(round(row.BIC, sigdigits=4))
        aic_i = string(round(row.AIC, sigdigits=4))

        out = out * "    $(mi) & $(αi) & $(si) & $(ki) & $(bic_i) & $(aic_i)\\\\\n"
    end

    out = out * "    \\bottomrule\n"
    out = out * "  \\end{tabularx}\n"
    out = out * "  \\end{adjustwidth}\n"
    out = out * "\\end{table}\n"

    return out
end

tex_table = generate_tex_table(df)

open("./models/hp-search-table.tex", "w") do f
    println(f, tex_table)
end


