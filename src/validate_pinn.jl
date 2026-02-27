# validate_pinn.jl
# Physics consistency validation

using Flux, JLD2, CSV, DataFrames, Statistics, Plots, Printf

const PROJECT_ROOT = dirname(@__DIR__)
const DATA_PATH = joinpath(PROJECT_ROOT, "data", "niger_delta_gaslift_daily.csv")
const MODEL_PATH = joinpath(PROJECT_ROOT, "models", "pinn_best.jld2")
const RESULTS_DIR = joinpath(PROJECT_ROOT, "results", "figures")

denorm(y, y_mean, y_std) = y .* y_std .+ y_mean

function validate_model()
    println("="^70)
    println("  PINN Physics Validation")
    println("="^70)
    
    isfile(MODEL_PATH) || error("Model not found. Run train_pinn.jl first.")
    
    loaded = JLD2.load(MODEL_PATH)
    model = loaded["model"]
    np = loaded["norm"]
    
    # Load and prepare data
    df = CSV.read(DATA_PATH, DataFrame)
    df = filter(r -> r.qgi_Mscfd > 5.0 && r.qo_STBd > 5.0, df)
    
    df[!, :drawdown] = df.Pres_psi .- df.Pbh_psi
    df[!, :qgi_ratio] = df.qgi_Mscfd ./ (df.qo_STBd .+ 1.0)
    
    X_cols = [:qgi_Mscfd, :Pres_psi, :GOR_scfSTB, :water_cut_frac,
              :API_gravity, :visc_cp, :depth_ft, :perm_mD,
              :drawdown, :qgi_ratio]
    
    X = Float32.(Matrix(df[:, X_cols]))
    X_n = (X .- np.X_mean) ./ np.X_std
    
    yhat = model(X_n')
    
    # Denormalize predictions
    qo_p = vec(denorm(yhat[1,:], np.y_mean[1], np.y_std[1]))
    Pwh_p = vec(denorm(yhat[2,:], np.y_mean[2], np.y_std[2]))
    Pbh_p = vec(denorm(yhat[3,:], np.y_mean[3], np.y_std[3]))
    qw_p = vec(denorm(yhat[4,:], np.y_mean[4], np.y_std[4]))
    
    qo_t = df.qo_STBd
    Pwh_t = df.Pwh_psi
    Pbh_t = df.Pbh_psi
    qw_t = df.qw_STBd
    
    # Physics checks
    mkpath(RESULTS_DIR)
    ENV["GKSwstype"] = "nul"
    gr(dpi=300)
    
    # 1. Pressure ordering
    pressure_drop = Pbh_p .- Pwh_p
    p1 = histogram(pressure_drop, bins=50, color=:steelblue, alpha=0.7,
                   xlabel="Pbh - Pwh (psi)", ylabel="Count",
                   title="Pressure Drop Distribution (Min: 50 psi)", framestyle=:box)
    vline!(p1, [50], lw=3, color=:red, ls=:dash, label="Minimum")
    
    violations = sum(pressure_drop .< 50)
    println("\n  Physics Constraints:")
    @printf("    Pressure drop < 50 psi: %d (%.2f%%)\n", violations, 100*violations/length(pressure_drop))
    
    # 2. Darcy consistency
    Pres = df.Pres_psi
    k = df.perm_mD
    drawdown = Pres .- Pbh_p
    darcy_ratio = qo_p ./ (sqrt.(k) .* max.(drawdown, 1.0))
    
    p2 = scatter(drawdown, darcy_ratio, ms=2, alpha=0.3, color=:forestgreen,
                 xlabel="Drawdown (Pres - Pbh) [psi]", 
                 ylabel="qo / (sqrt(k) * drawdown)",
                 title="Darcy Consistency Check", framestyle=:box)
    
    # 3. Water cut comparison
    fw_input = df.water_cut_frac
    ql_p = qo_p .+ qw_p
    fw_pred = qw_p ./ ql_p
    
    p3 = scatter(fw_input, fw_pred, ms=2, alpha=0.3, color=:coral,
                 xlabel="Input Water Cut", ylabel="Predicted Water Cut",
                 title="Water Cut Consistency", framestyle=:box)
    plot!(p3, [0, 1], [0, 1], color=:black, lw=2, ls=:dash, label="1:1")
    
    # 4. Pwh vs qgi correlation
    qgi = df.qgi_Mscfd
    p4 = scatter(qgi, Pwh_p, ms=2, alpha=0.3, color=:darkorange,
                 xlabel="Gas Injection (Mscf/d)", ylabel="Wellhead Pressure (psi)",
                 title="Pwh vs Gas Injection", framestyle=:box)
    
    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 1000), margin=12Plots.mm)
    savefig(plt, joinpath(RESULTS_DIR, "03_physics_validation.png"))
    println("  Saved: 03_physics_validation.png")
    
    # Summary metrics
    println("\n  Validation Metrics (Full Dataset):")
    println("  " * "-"^60)
    @printf("  %-20s %10s %10s %10s\n", "Output", "MAPE (%)", "R²", "RMSE")
    println("  " * "-"^60)
    
    names = ["qo (STBd)", "Pwh (psi)", "Pbh (psi)", "qw (STBd)"]
    truths = [qo_t, Pwh_t, Pbh_t, qw_t]
    preds = [qo_p, Pwh_p, Pbh_p, qw_p]
    
    for i in 1:4
        mape = mean(abs.((truths[i] .- preds[i]) ./ (abs.(truths[i]) .+ 1e-8))) * 100
        r2 = 1.0 - sum((truths[i] .- preds[i]).^2) / sum((truths[i] .- mean(truths[i])).^2)
        rmse = sqrt(mean((truths[i] .- preds[i]).^2))
        @printf("  %-20s %10.2f %10.4f %10.2f\n", names[i], mape, r2, rmse)
    end
    
    println("\n  Validation complete!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_model()
end