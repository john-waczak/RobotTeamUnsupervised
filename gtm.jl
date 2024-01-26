using LinearAlgebra
using Statistics, MultivariateStats
using Distances

function makeCoords(k)
    ξ = range(-1.0, 1.0, length=k)

    ξ1 = vcat([ξ[i] for i in 1:length(ξ), j in 1:length(ξ)]...)
    ξ2 = vcat([ξ[j] for i in 1:length(ξ), j in 1:length(ξ)]...)
    Ξ = hcat(ξ1, ξ2)
end


function getΦMatrix(X, M, σ²)
    n_nodes = size(X, 1)
    n_rbf_centers = size(M,1)
    Φ = zeros(n_nodes, n_rbf_centers + 1)
    Φdistances = pairwise(sqeuclidean, X, M, dims=1)

    Φ[:, 1:end-1] .= exp.(-1.0 .* Φdistances ./ (2σ²))
    Φ[:, end] .= 1.0

    # set the last column to ones to allow for a bias term
    Φ[:,end] .= 1
    return Φ
end



mutable struct GTM{T <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray}
    Ξ::T
    M::T2
    σ²::Float64
    Φ::T3
    W::T4
    β⁻¹::Float64
    α::Float64
    tol::Float64
    niter::Int
    nrepeats::Int
end



function GTM(k, m, s, X; α=0.1, tol=0.0001, niter=200, nrepeats=4)

    N = size(X,1)  # number of records
    D = size(X,2)  # number of features

    # node locations
    Ξ = makeCoords(k)  # size K×2
    # rbf locations
    M = makeCoords(m)  # size M×2

    # set the rbf variance to the distance
    # between rbf centers scaled by s
    σ = s * abs(M[2,1]-M[1,1])

    # initialize the Φ matrix: size K×(M+1)
    Φ = zeros(k*k, m*m + 1)
    Δ² = pairwise(sqeuclidean, Ξ , M, dims=1)

    Φ[:, 1:end-1] .= exp.(-Δ² ./ (2*σ^2))  # rbf's
    Φ[:, end] .= 1.0                       # constant bias

    # initialize hyperparameters with PCA
    pca = fit(PCA, X'; maxoutdim=3, pratio=0.99999)
    pca_vecs = projection(pca)
    pca_var = principalvars(pca)
    U = pca_vecs[:, 1:2]  # size (N×2)

    # now we want W*Φ' ≈ UΞ'
    # use matrix right-division with /
    # (UΞ') = WΦ' --> W = (UΞ')/Φ'
    # W = (U*Ξ')/(Φ')
    W = (U*Ξ') * pinv(Φ')  # <- identical, but faster...

    # set to variance of 3rd principal component
    # or half mean distance between projected latent
    # vector means
    β⁻¹ = max(pca_var[3],mean(pairwise(sqeuclidean, W*Φ', dims=2))/2)

    return GTM(Ξ, M, σ^2, Φ, W, β⁻¹, α, tol, niter, nrepeats)
end



function getLatentMeans(gtm)
    return gtm.W*gtm.Φ'
end


function getNodeDistances(gtm, X)
    return pairwise(sqeuclidean, getLatentMeans(gtm), X', dims=2)
end


function getNodeDistances!(D, gtm, X)
    pairwise!(D, sqeuclidean, getLatentMeans(gtm), X', dims=2)
end



function Responsabilities(gtm, X)
    D = getNodeDistances(gtm, X)

    # use exp-normalize trick
    # to prevent overflow
    R = -(1.0/(2.0*gtm.β⁻¹)) .* D
    for j in axes(R, 2)
        R[:,j] .= exp.(R[:,j] .- maximum(R[:,j]))
        R[:,j] .= R[:,j] ./ sum(R[:,j])
    end

    return R
end



# in place version for less allocations
function Responsabilities!(R, R_tmp, D, gtm, X)
    getNodeDistances!(D, gtm, X)

    R .= -(1.0/(2.0*gtm.β⁻¹)) .* D
    maximum!(R_tmp, R)

    #Threads.@threads for j in axes(R, 2)
    for j in axes(R, 2)
        R[:,j] .= exp.(R[:,j] .- R_tmp[j])
    end

    sum!(R_tmp, R)
    # Threads.@threads for j in axes(R,2)
    for j in axes(R,2)
        R[:,j] .= R[:,j] ./ R_tmp[j]
    end
end


function getGMatrix(R)
    return diagm(sum(R, dims=2)[:])
end


function getGMatrix!(G,R)
    G_diag = @view G[diagind(G)]
    sum!(G_diag, R)
end
