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


k = 32
m = 14
s = 0.1

ξ₁ = range(-1, stop=1.0, length=k)
ξ₁ = range(-1, stop=1.0, length=k)



# create figure with latent space grid and RBF points as well as
# the "3d" projection of means in the "data" space
