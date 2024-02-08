using Distances

function silhouettes(mach, X)
    Xdata = Matrix(X)

    # form lookup table for cluster identity
    cluster_idx = zeros(Bool, nrow(X), mach.model.k)
    for j in 1:mach.model.k
        cluster_idx[:,j] .= (assignments .== j)
    end

    Δ = pairwise(mach.model.metric, Xdata, dims=1)

    # consider the ith record
    s = zeros(nrow(X))

    for i ∈ 1:nrow(X)
        lᵢ = assignments[i]
        ab = [mean(Δ[i,Not(i)][assignments[Not(i)] .== j]) for j ∈ 1:mach.model.k]
        aᵢ = ab[lᵢ]
        bᵢ = minimum(ab[Not(lᵢ)])
        sᵢ = (bᵢ - aᵢ)/max(aᵢ, bᵢ)
    end


end
