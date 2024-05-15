using CairoMakie
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

figures_path = joinpath("./figures", "extras")
if !ispath(figures_path)
    mkpath(figures_path)
end


k = 15
m = 5



ξ₁ = range(-1, stop=1.0, length=k)
ξ₂ = range(-1, stop=1.0, length=k)

μ₁ = range(-1, stop=1.0, length=m)
μ₂ = range(-1, stop=1.0, length=m)



s = 0.01
function rbf_val(x,y)
    out = 0.0
    for i ∈ 1:m, j ∈ 1:m
        out += exp(-((x-μ₁[i])^2 + (y - μ₂[j])^2)/(2*s))
    end
    out
end

cm = cgrad([CairoMakie.RGBA(0,0,0,0), mints_colors[2]]) #CairoMakie.RGBA(1,0,0, 1)])



mints_colors[2]

fig =Figure();
ax = Axis(
    fig[1,1],
    xlabel="ξ₁", ylabel="ξ₂",
    aspect=DataAspect(),
    xticklabelsvisible=false, xticksvisible=false, xminorticksvisible=false, xgridvisible=false, xminorgridvisible=false, xlabelsize=30,
    yticklabelsvisible=false, yticksvisible=false, yminorticksvisible=false, ygridvisible=false, yminorgridvisible=false, ylabelsize=30,
);

hidespines!(ax)

pts_x = [(Point(ξ₁[i], -1), Point(ξ₁[i], 1)) for i ∈ 1:k]
pts_y = [(Point(-1, ξ₂[i]), Point(1, ξ₂[i])) for i ∈ 1:k]
pts_joined = [Point(ξ₁[i], ξ₂[j]) for i ∈ 1:k for j ∈ 1:k]

pts_rbf = [Point(μ₁[i], μ₂[j]) for i ∈ 1:m for j ∈ 1:m]

linesegments!(ax, pts_x, color=:lightgray, linewidth=2)
linesegments!(ax, pts_y, color=:lightgray, linewidth=2)
heatmap!(ax, -1:0.005:1, -1:0.005:1, rbf_val, colormap=cm)
scatter!(ax, pts_joined, color=mints_colors[1], markersize=12)
fig

save(joinpath(figures_path, "gtm-latent.svg"), fig)



# Create 3d plot of the grid with spheres on the node points

fz(x,y) = x*exp(-x^2 -y^2)

xs = range(-2, stop=2, length=k)
ys = range(-2, stop=2, length=k)
zs = [fz(xs[i], ys[j]) for i∈1:k, j∈1:k]

pts = [Point3f(xs[i], ys[j], zs[i,j]) for i in 1:k for j ∈ 1:k]

fig = Figure();
ax = Axis3(fig[1,1]);
hidedecorations!(ax)
hidespines!(ax)


#surface!(ax, xs, ys, fz, color=[CairoMakie.RGBA(0,0,0,0.5) for i ∈ 1:k for j ∈ 1:k]);
wireframe!(ax, xs, ys, zs, color=:lightgray);
scatter!(ax, pts, color=mints_colors[3], markersize=10)

save(joinpath(figures_path, "gtm-projected.svg"), fig)

fig
