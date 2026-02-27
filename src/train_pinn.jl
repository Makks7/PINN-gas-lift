# train_pinn.jl
# Physics-Informed Neural Network for Gas-Lift Optimization
# Final production version for STSE paper

using Flux, Zygote, CSV, DataFrames, Statistics, Random, StatsBase
using LinearAlgebra, Plots, JLD2, Printf, Dates

Random.seed!(123)

const PROJECT_ROOT = dirname(@__DIR__)
const DATA_PATH = joinpath(PROJECT_ROOT, "data", "niger_delta_gaslift_daily.csv")
const MODELS_DIR = joinpath(PROJECT_ROOT, "models")
const RESULTS_DIR = joinpath(PROJECT_ROOT, "results", "figures")

# Clean start
isfile(joinpath(MODELS_DIR, "pinn_best.jld2")) && rm(joinpath(MODELS_DIR, "pinn_best.jld2"))

function load_data(path=DATA_PATH)
    println("Loading: $path")
    raw = CSV.read(path, DataFrame)
    df = filter(r -> r.qgi_Mscfd > 5.0 && r.qo_STBd > 5.0, raw)
    
    # Feature engineering
    df[!, :drawdown] = df.Pres_psi .- df.Pbh_psi
    df[!, :qgi_ratio] = df.qgi_Mscfd ./ (df.qo_STBd .+ 1.0)
    
    X_cols = [:qgi_Mscfd, :Pres_psi, :GOR_scfSTB, :water_cut_frac,
              :API_gravity, :visc_cp, :depth_ft, :perm_mD,
              :drawdown, :qgi_ratio]
    y_cols = [:qo_STBd, :Pwh_psi, :Pbh_psi, :qw_STBd]

    X = Float32.(Matrix(df[:, X_cols]))
    y = Float32.(Matrix(df[:, y_cols]))

    # Standardization
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1) .+ 1f-8
    y_mean = mean(y, dims=1)
    y_std = std(y, dims=1) .+ 1f-8

    X_n = (X .- X_mean) ./ X_std
    y_n = (y .- y_mean) ./ y_std

    n = nrow(df)
    idx = randperm(n)
    n_tr = round(Int, 0.70 * n)
    n_va = round(Int, 0.15 * n)
    
    i_tr = idx[1:n_tr]
    i_va = idx[n_tr+1:n_tr+n_va]
    i_te = idx[n_tr+n_va+1:end]
    
    np = (X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)

    return (
        Xt = X_n[i_tr, :]',   yt = y_n[i_tr, :]',
        Xv = X_n[i_va, :]',   yv = y_n[i_va, :]',
        Xte = X_n[i_te, :]',  yte = y_n[i_te, :]',
        Xraw = X[i_tr, :]',
        norm = np,
        df = df
    )
end

function build_model()
    model = Chain(
        Dense(10, 128, tanh),
        Dense(128, 128, tanh),
        Dense(128, 64, tanh),
        Dense(64, 64, tanh),
        Dense(64, 4)
    )
    println("  Parameters: $(sum(length, Flux.trainable(model)))")
    return model
end

denorm(y, y_mean, y_std) = y .* y_std .+ y_mean

function physics_losses(m, Xn, yn, Xraw, np)
    yhat = m(Xn)
    
    qo_p = denorm(yhat[1:1,:], np.y_mean[1], np.y_std[1])
    Pwh_p = denorm(yhat[2:2,:], np.y_mean[2], np.y_std[2])
    Pbh_p = denorm(yhat[3:3,:], np.y_mean[3], np.y_std[3])
    qw_p = denorm(yhat[4:4,:], np.y_mean[4], np.y_std[4])
    
    qgi = Xraw[1:1,:]
    Pres = Xraw[2:2,:]
    fw_in = Xraw[4:4,:]

    # L1: Pressure ordering (Pbh > Pwh + 50)
    L1 = mean(max.(0.0f0, Pwh_p .- Pbh_p .+ 50.0f0) .^ 2) ./ 10000.0f0

    # L2: Darcy consistency for qo
    draw = max.(Pres .- Pbh_p, 10.0f0)
    L2 = mean((qo_p ./ (sqrt.(Xraw[8:8,:]) .* draw .+ 1f-6) .- 
              mean(qo_p ./ (sqrt.(Xraw[8:8,:]) .* draw .+ 1f-6))) .^ 2)

    # L3: Water cut consistency
    ql = qo_p .+ qw_p .+ 1f-6
    fw_p = qw_p ./ ql
    L3 = mean((fw_p .- fw_in) .^ 2)

    return L1, L2, L3
end

function total_loss(m, Xn, yn, Xraw, np; λ_phys=0.05f0)
    yhat = m(Xn)
    L_data = mean((yhat .- yn) .^ 2)
    
    L1, L2, L3 = physics_losses(m, Xn, yn, Xraw, np)
    L_phys = L1 + L2 + L3
    
    return L_data + λ_phys * L_phys, L_data, L_phys
end

function train!(model, data; epochs=800, batch=256, lr=1e-3, patience=200)
    mkpath(MODELS_DIR)
    opt_state = Flux.setup(Adam(lr), model)
    n = size(data.Xt, 2)

    hist = Dict(
        :total => Float32[],
        :data => Float32[],
        :phys => Float32[],
        :val_mape_qo => Float32[],
        :val_mape_Pwh => Float32[],
        :val_mape_Pbh => Float32[],
        :val_mape_qw => Float32[],
        :val_r2_Pwh => Float32[]
    )

    best_mape = Inf32
    best_ep = 0
    no_improve = 0
    model_path = joinpath(MODELS_DIR, "pinn_best.jld2")

    println("\n" * "="^70)
    println("  PINN Training | epochs=$epochs | patience=$patience")
    println("="^70)
    @printf("  %-6s %-10s %-10s %-10s %-8s %-8s %-8s %-8s %-8s\n",
            "Epoch", "Total", "Data", "Physics", "MAPEqo", "MAPEPwh", "MAPEPbh", "MAPEqw", "R2Pwh")

    for ep in 1:epochs
        # Cosine annealing
        frac = Float32(ep / epochs)
        lr_t = lr * (0.5f0 + 0.5f0 * cos(frac * Float32(pi)))
        opt_state = Flux.setup(Adam(lr_t), model)

        # Training
        perm = randperm(n)
        for s in 1:batch:n
            idx = perm[s:min(s+batch-1, n)]
            Xb = data.Xt[:, idx]
            yb = data.yt[:, idx]
            Xbr = data.Xraw[:, idx]
            
            loss, grads = Flux.withgradient(model) do m
                total_loss(m, Xb, yb, Xbr, data.norm)[1]
            end
            Flux.update!(opt_state, model, grads[1])
        end

        # Validation
        yv = model(data.Xv)
        l_tot, l_dat, l_phys = total_loss(model, data.Xt, data.yt, data.Xraw, data.norm)
        
        # Per-output metrics
        mapes = Float32[]
        for i in 1:4
            yt = denorm(data.yv[i:i,:], data.norm.y_mean[i], data.norm.y_std[i])
            yp = denorm(yv[i:i,:], data.norm.y_mean[i], data.norm.y_std[i])
            mape = mean(abs.((yt .- yp) ./ (abs.(yt) .+ 1f-6))) * 100.0f0
            push!(mapes, mape)
        end
        
        # Pwh R²
        Pwh_t = denorm(data.yv[2:2,:], data.norm.y_mean[2], data.norm.y_std[2])
        Pwh_p = denorm(yv[2:2,:], data.norm.y_mean[2], data.norm.y_std[2])
        r2_Pwh = 1.0f0 - sum((Pwh_t .- Pwh_p).^2) / (sum((Pwh_t .- mean(Pwh_t)).^2) + 1f-8)

        push!(hist[:total], l_tot)
        push!(hist[:data], l_dat)
        push!(hist[:phys], l_phys)
        push!(hist[:val_mape_qo], mapes[1])
        push!(hist[:val_mape_Pwh], mapes[2])
        push!(hist[:val_mape_Pbh], mapes[3])
        push!(hist[:val_mape_qw], mapes[4])
        push!(hist[:val_r2_Pwh], r2_Pwh)

        status = ""
        if mapes[2] < best_mape  # Based on Pwh MAPE
            best_mape = mapes[2]
            best_ep = ep
            no_improve = 0
            JLD2.jldsave(model_path; model=model, norm=data.norm, epoch=ep, 
                        mape=mapes[2], r2=r2_Pwh)
            status = "*"
        else
            no_improve += 1
        end

        if ep <= 5 || ep % 50 == 0 || status == "*"
            @printf("  %-6d %-10.5f %-10.5f %-10.5f %-8.2f %-8.2f %-8.2f %-8.2f %-8.3f %s\n",
                    ep, l_tot, l_dat, l_phys, mapes[1], mapes[2], mapes[3], mapes[4], r2_Pwh, status)
        end

        if no_improve >= patience
            println("  Early stop at epoch $ep")
            break
        end
    end

    println("\n  Best: epoch $best_ep, Pwh MAPE=$(round(best_mape, digits=2))%")
    loaded = JLD2.load(model_path)
    return loaded["model"], hist, data.norm
end

function make_plots(hist, model, data, np)
    mkpath(RESULTS_DIR)
    ENV["GKSwstype"] = "nul"
    gr(dpi=300, size=(1200, 400))
    
    ep = 1:length(hist[:total])

    # Plot 1: Loss curves
    p1 = plot(ep, hist[:total], label="Total Loss", lw=2.5, color=:steelblue,
              yscale=:log10, framestyle=:box, legend=:topright)
    plot!(p1, ep, hist[:data], label="Data Loss", lw=2, ls=:dash, color=:seagreen)
    plot!(p1, ep, hist[:phys], label="Physics Loss", lw=2, ls=:dot, color=:coral)
    xlabel!(p1, "Epoch", fontsize=12)
    ylabel!(p1, "Loss (log scale)", fontsize=12)
    title!(p1, "Training Convergence", fontsize=14)

    # Plot 2: Per-output MAPE
    p2 = plot(ep, hist[:val_mape_qo], lw=2, label="Oil Rate", color=:steelblue)
    plot!(p2, ep, hist[:val_mape_Pwh], lw=2.5, label="Wellhead P", color=:darkorange)
    plot!(p2, ep, hist[:val_mape_Pbh], lw=2, label="Bottomhole P", color=:forestgreen)
    plot!(p2, ep, hist[:val_mape_qw], lw=2, label="Water Rate", color=:crimson)
    hline!(p2, [5.0], lw=2, color=:red, ls=:dash, label="5% target")
    xlabel!(p2, "Epoch", fontsize=12)
    ylabel!(p2, "Validation MAPE (%)", fontsize=12)
    title!(p2, "Prediction Accuracy by Output", fontsize=14)
    ylims!(p2, 0, 20)

    # Plot 3: Pwh R² evolution
    p3 = plot(ep, hist[:val_r2_Pwh], lw=2.5, color=:darkorange, 
              fill=(0, 0.15, :darkorange), label="Pwh R²")
    hline!(p3, [0.85], lw=2, color=:green, ls=:dash, label="Target")
    xlabel!(p3, "Epoch", fontsize=12)
    ylabel!(p3, "R²", fontsize=12)
    title!(p3, "Wellhead Pressure R²", fontsize=14)
    ylims!(p3, 0, 1)

    # Combined training curves
    plt = plot(p1, p2, p3, layout=(1, 3), size=(1500, 450), margin=10Plots.mm)
    savefig(plt, joinpath(RESULTS_DIR, "01_training_curves.png"))
    println("  Saved: 01_training_curves.png")

    # Parity plots for all outputs
    yhat = model(data.Xte)
    names = ["Oil Rate (STB/d)", "Wellhead Pressure (psi)", 
             "Bottomhole Pressure (psi)", "Water Rate (STB/d)"]
    cols = [:steelblue, :darkorange, :forestgreen, :crimson]
    units = ["STB/d", "psi", "psi", "STB/d"]
    
    ps = []
    results = []
    
    for i in 1:4
        yt = vec(denorm(data.yte[i:i,:], np.y_mean[i], np.y_std[i]))
        yp = vec(denorm(yhat[i:i,:], np.y_mean[i], np.y_std[i]))
        
        mape = mean(abs.((yt .- yp) ./ (abs.(yt) .+ 1e-8))) * 100
        r2 = 1.0 - sum((yt .- yp).^2) / (sum((yt .- mean(yt)).^2) + 1e-10)
        rmse = sqrt(mean((yt .- yp).^2))
        
        push!(results, (name=names[i], mape=mape, r2=r2, rmse=rmse))
        
        lims = (min(minimum(yt), minimum(yp)), max(maximum(yt), maximum(yp)))
        
        p = scatter(yt, yp, ms=3, alpha=0.25, color=cols[i], label=false,
                    markerstrokewidth=0, framestyle=:box)
        plot!(p, [lims[1], lims[2]], [lims[1], lims[2]], 
              color=:black, lw=1.5, ls=:dash, label="1:1 Line")
        xlabel!(p, "Measured $(units[i])", fontsize=11)
        ylabel!(p, "Predicted $(units[i])", fontsize=11)
        title!(p, "$(names[i])\nMAPE=$(round(mape,digits=2))%, R²=$(round(r2,digits=4))", fontsize=12)
        push!(ps, p)
    end
    
    parity = plot(ps..., layout=(2, 2), size=(1100, 1000), margin=12Plots.mm)
    savefig(parity, joinpath(RESULTS_DIR, "02_parity_plots.png"))
    println("  Saved: 02_parity_plots.png")
    
    # Print summary table
    println("\n  Final Test Results:")
    println("  " * "-"^70)
    @printf("  %-20s %10s %10s %10s\n", "Output", "MAPE (%)", "R²", "RMSE")
    println("  " * "-"^70)
    for r in results
        @printf("  %-20s %10.2f %10.4f %10.2f\n", r.name, r.mape, r.r2, r.rmse)
    end
    println("  " * "-"^70)
    
    return results
end

function main()
    println("\n" * "="^70)
    println("  PINN-MPC | Niger Delta Gas-Lift | Final Training")
    println("="^70)
    
    data = load_data()
    model = build_model()
    model, hist, np = train!(model, data; epochs=800, lr=1e-3, patience=200)
    results = make_plots(hist, model, data, np)
    
    println("\n  Training complete! All plots saved to results/figures/")
    return model, data, hist, results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end