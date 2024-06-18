using CondaPkg
CondaPkg.add("contextily")
using PythonCall
using Images
using HDF5
using ProgressMeter
export get_background_satmap


ctx = pyimport("contextily")

struct SatMap
    w
    e
    s
    n
    img
end


"""
    get_background_satmap(w::Float64, e::Float64, s::Float64, n::Float64; out_name::String="Scotty)


Grab Esri World Imagery tiles for region with corners (w,n), (e,s) in longitude and latitude.
Saves resulting image to `outpath`

**Note:** result images are saved in Web-Mercator projection by default. See `WebMercatorfromLLA` and `LLAfromWebMercator` from `Geodesy.jl` for conversion details.
"""
function get_background_satmap(w::Float64, e::Float64, s::Float64, n::Float64; out_name::String="Scotty")
    # ctx = pyimport("contextily")
    tiff, ext = ctx.bounds2raster(w, s, e, n, out_name*".tiff", source=ctx.providers["Esri"]["WorldImagery"], ll=true)
    warped_tiff, warped_ext = ctx.warp_tiles(tiff, ext, "EPSG:4326")

    warped_ext = pyconvert(Vector{Float64}, warped_ext)
    tiff_array = permutedims(pyconvert(Array{Int}, warped_tiff)./255, (3,1,2))
    tiff_img = colorview(RGBA, tiff_array)

    tiff_img = rotr90(tiff_img)

    return SatMap(
        warped_ext[1],
        warped_ext[2],
        warped_ext[3],
        warped_ext[4],
        tiff_img
    )
end



# metric bounds
metric_bounds = Dict(
    "mNDWI" => (-1, 1),
    "NDVI" => (-1 ,1),
    "SR" => (0, 30),
    "EVI" => (-1, 1),
    "AVRI" => (-1, 1),
    "NDVI_705" => (-1,1),
    "MSR_705" => (0, 30),
    "MNDVI" => (-1, 1),
    "VOG1" => (0, 20),
    "VOG2" => (0, 20),
    "VOG3" => (0, 20),
    "PRI" => (-1,1),
    "SIPI" => (0, 2),
    "PSRI" => (-1, 1),
    "CRI1" => (0, 15),
    "CRI2" => (0, 15),
    "ARI1" => (0, 0.2),
    "ARI2" => (0, 0.2),
    "WBI" => (0.5, 1.5),
    "MCRI" => (0, 15),
    "TCARI" => (0, 15)
)



function getRGB(h5::HDF5.File; λred=630.0, λgreen=532.0, λblue=465.0, Δx=0.10, α=10.0, β=0.0)
    λs = h5["data-Δx_$(Δx)/λs"][:]

    λred=630.0
    λgreen=532.0
    λblue=465.0

    idx_r = argmin(abs.(λs .- λred))
    idx_g = argmin(abs.(λs .- λgreen))
    idx_b = argmin(abs.(λs .- λblue))

    Rr = h5["data-Δx_$(Δx)/Data"][idx_r, :, :]
    Rg = h5["data-Δx_$(Δx)/Data"][idx_g, :, :]
    Rb = h5["data-Δx_$(Δx)/Data"][idx_b, :, :]

    ij_pixels = findall(h5["data-Δx_$(Δx)/IsInbounds"][:,:])
    img = zeros(4, size(Rr)...)

    Threads.@threads for ij ∈ ij_pixels
        img[1, ij] = Rr[ij]
        img[2, ij] = Rg[ij]
        img[3, ij] = Rb[ij]
        img[4, ij] = 1.0
    end

    return img
end


function gamma_correct(img, γ=1/2)
    # see https://en.wikipedia.org/wiki/Gamma_correction
    img_out = copy(img)

    img_out[1:3,:,:] .= img_out[1:3,:,:] .^ γ

    return img_out
end

function brighten(img, α=0.13, β=0.0)
    # see : https://www.satmapper.hu/en/rgb-images/
    img_out = copy(img)
    for i ∈ 1:3
        img_out[i,:,:] .= clamp.(α .* img_out[i,:,:] .+ β, 0, 1)
    end

    return img_out
end

function get_h5_rgb(h5path)
    h5 = h5open(h5path)
    img = getRGB(h5)
    close(h5)
    img_out = brighten(gamma_correct(img, .75), 3.5, 0)
    return colorview(RGBA, img_out)
end


function vis_rectified_cube(h5path; size=(800,600), colormap=:jet, azimuth=3π / 4, elevation=3π / 16, colorbar=false, ibounds=(1,1), jbounds=(1,1))

    println("Loading data")

    h5 = h5open(h5path, "r")

    Δx = 0.1
    Data = h5["data-Δx_$(Δx)/Data"][:, :, :];
    IsInbounds = h5["data-Δx_$(Δx)/IsInbounds"][:, :];
    xs = h5["data-Δx_$(Δx)/X"][:];
    ys = h5["data-Δx_$(Δx)/Y"][:];
    λs = h5["data-Δx_$(Δx)/λs"][:];

    close(h5)

    nλs = length(λs)
    # ibounds = (1,1)
    # jbounds=(1,1)
    offset = eps(1.0)
    colormap=:jet

    ij_pixels = findall(IsInbounds)
    Ref_img = get_h5_rgb(h5path)

    # reshape array for plotting
    println("Reshaping for plotting")
    data = PermutedDimsArray(Data[1:nλs, :, :], (2, 3, 1));
    data = log10.(data .+ offset);

    # find all good pixels
    println("Finding good pixels")
    idx_not_nan_or_inf = findall(.!(isnan.(data)) .&& .!(isinf.(data)))
    Rmin = quantile(data[idx_not_nan_or_inf], 0.1)
    Rmax = quantile(data[idx_not_nan_or_inf], 0.99)

    # chop image to bounds if needed
    if ibounds != (1, 1)
        imin, imax = ibounds
        data = data[imin:imax, :, :]
        Ref_img = Ref_img[imin:imax, :]
    end

    if jbounds != (1, 1)
        jmin, jmax = jbounds
        data = data[:, jmin:jmax, :]
        Ref_img = Ref_img[:, jmin:jmax]
    end

    # construct figure
    println("Making figure")
    fig = Figure(; size=size);
    ax = Axis3(
        fig[1, 1],
        perspectiveness=0.5,
        elevation=elevation,
        azimuth=azimuth,
        tellheight=true,
        tellwidth=true,
        # aspect=:data,
    )

    hidedecorations!(ax)
    hidespines!(ax)

    @showprogress for k in 1:length(λs)
        mr = Rect3f(Vec3f(0,0,(k-1)*Δx), Vec3f(length(xs)*Δx, length(ys)*Δx, Δx))
        mesh!(
            ax,
            mr;
            color = data[:,:,k],
            interpolate=false,
            colormap = colormap,
            colorrange= (Rmin, Rmax),
            shading= NoShading,
            )
    end


    # add color image to top
    println("Adding color image to top")
    mr = Rect3f(Vec3f(0,0,462*Δx), Vec3f(length(xs)*Δx, length(ys)*Δx, Δx))
    mesh!(
        ax,
        mr;
        color= Ref_img,
        interpolate=true,
        shading= NoShading,
    )

    if colorbar
        println("Adding colorbar")
        Colorbar(fig[1, 2], limits=(Rmin, Rmax), colormap=colormap, label="log10(Reflectance)", height=Relative(0.5), tellwidth=true)
    end

    fig2 = Figure();
    Colorbar(fig2[1, 1], limits=(Rmin, Rmax), colormap=colormap, label="log10(Reflectance)", height=Relative(0.5), tellwidth=true)

    return fig, fig2
end
