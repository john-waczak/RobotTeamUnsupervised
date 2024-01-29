using LinearAlgebra
using Statistics, MultivariateStats
using Distances


mutable struct GTM{T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray}
    Ξ::T1
    M::T2
    Φ::T3
    W::T4
    β⁻¹::Float64
end


function GTM(k, m, s, X)
    # 1. define grid parameters
    n_records, n_features = size(X)
    n_nodes = k*k
    n_rbf_centers = m*m

    # 2. create grid of K latent nodes
    ξ = range(-1.0, stop=1.0, length=k)
    Ξ = hcat([ξ[i] for i in axes(ξ,1) for j in axes(ξ,1)],[ξ[j] for i in axes(ξ,1) for j in axes(ξ,1)])

    # 3. create grid of M rbf centers (means)
    μ = range(-1.0, stop=1.0, length=m)
    M = hcat([μ[i] for i in axes(μ,1) for j in axes(μ,1)],[μ[j] for i in axes(μ,1) for j in axes(μ,1)])

    # 4. initialize rbf width
    σ = s * abs(μ[2]-μ[1])

    # 5. create rbf activation matrix Φ
    Φ = ones(n_nodes, n_rbf_centers+1)
    let
        Δ² = pairwise(sqeuclidean, Ξ, M, dims=1)
        Φ[:, 1:end-1] .= exp.(-Δ² ./ (2*σ^2) )
    end

    # 6. perform PCA on data
    pca = fit(PCA, X', maxoutdim=3, pratio=0.99999)
    pca_vecs = projection(pca)
    pca_vars = principalvars(pca)

    # 7. create matrix U from first two principal axes of data cov. matrix
    U = pca_vecs[:, 1:2]
    # scale by square root of variance for some reason
    U[:,1] .= U[:,1] .* sqrt(pca_vars[1])
    U[:,2] .= U[:,2] .* sqrt(pca_vars[2])

    # 8. Initialize parameter matrix W using U and Φ

    Ξnorm = Ξ
    Ξnorm[:,1] = (Ξnorm[:,1] .-  mean(Ξnorm[:,1])) ./ std(Ξnorm[:,1])
    Ξnorm[:,2] = (Ξnorm[:,2] .-  mean(Ξnorm[:,2])) ./ std(Ξnorm[:,2])

    scatter(Ξnorm[:,1], Ξnorm[:,2])

    W = U*Ξnorm' * pinv(Φ')
    # W = rand(n_features, n_rbf_centers+1)

    # 9. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'

    # 10. Set noise variance parameter to largest between
    #     - 3rd principal component variance
    #     - 1/2 the average distance between data manifold centers (from Y)

    β⁻¹ = max(pca_vars[3], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)
    pca_vars[3]

    # 11. return final GTM object

    return GTM(Ξ, M, Φ, W, β⁻¹)
end




function exp_normalize(Λ)
    maxes = maximum(Λ, dims=1)
    res = zeros(size(Λ))
    for j in axes(Λ, 2)
        res[:,j] .= exp.(Λ[:,j] .- maxes[j])
    end
    return res
end


function getPMatrix(Δ², β⁻¹)
    # use exp-normalize trick
    return exp_normalize(-(1/(2*β⁻¹)) .* Δ²)
end


function Responsabilities(P)
    R = zeros(size(P))
    for j in axes(R,2)
        R[:,j] .= P[:,j] ./ sum(P[:,j])
    end
    return R
end



# function Responsabilities(Δ², β⁻¹)
#     R = getPMatrix(Δ², β⁻¹)
#     for j in axes(R,2)
#         R[:,j] .= R[:,j] ./ sum(R[:,j])
#     end
#     return R
# end


function getUpdateW(R, G, Φ, X, β⁻¹, α)
    # W is the solution of
    # (Φ'GΦ + (αβ⁻¹)I)W' = Φ'RX
    if α > 0
        return ((Φ'*G*Φ + α*I)\(Φ'*R*X))'
    else
        return ((Φ'*G*Φ)\(Φ'*R*X))'
    end
end



function getUpdateβ⁻¹(R, Δ², X)
    n_records, n_features = size(X)

    return sum(R .* Δ²)/(n_records*n_features)
end


function loglikelihood(P, β⁻¹, X, Ξ)
    N, D = size(X)
    K = size(Ξ,1)

    prexp = (1/(2* β⁻¹* π))^(D/2)

    return sum(log.((prexp/K) .* sum(P, dims=1)))
end




function fit!(gtm, X; α = 0.1, niter=100, tol=0.001, nconverged=5)
    # 1. create distance matrix Δ² between manifold points and data matrix
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    # 2. Until convergence, i.e. log-likelihood < tol

    llhs = Float64[]
    llh_prev = 0.0
    nclose = 0
    converged = false  # a flag to tell us if we converged successfully
    for i in 1:niter
        # expectation
        P = getPMatrix(Δ², gtm.β⁻¹)
        R = Responsabilities(P)
        G = diagm(sum(R, dims=2)[:])

        # (maximization)
        gtm.W = getUpdateW(R, G, gtm.Φ, X, gtm.β⁻¹, α)
        Ψ = gtm.W * gtm.Φ'
        Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
        gtm.β⁻¹ = getUpdateβ⁻¹(R, Δ², X)

        # compute log-likelihood
        l = loglikelihood(P, gtm.β⁻¹, X, gtm.Ξ)
        push!(llhs, l)

        llh_diff = abs(l - llh_prev)
        llh_prev = l

        if llh_diff <= tol
            # increment the number of "close" difference
            nclose += 1
        end

        if nclose == nconverged
            converged = true
            break
        end
    end

    # update responsabilities after final pass
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    P = getPMatrix(Δ², gtm.β⁻¹)
    R = Responsabilities(P)

    return converged,llhs, R
end



function DataMeans(gtm)
    return R'*gtm.Ξ
end


function DataModes(gtm)
    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return gtm.Ξ[idx,:]
end



