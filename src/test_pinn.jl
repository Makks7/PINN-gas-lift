# test_pinn.jl
# SIMPLIFIED - Use same approach as validation

using Flux, JLD2, CSV, DataFrames, Statistics, Plots, Printf, Dates

const PROJECT_ROOT = dirname(@__DIR__)
const DATA_PATH = joinpath(PROJECT_ROOT, "data", "niger_delta_gaslift_daily.csv")
const MODEL_PATH = joinpath(PROJECT_ROOT, "models", "pinn_best.jld2")
const RESULTS_DIR = joinpath(PROJECT_ROOT, "results", "figures")

function test_model()
    println("="^70)
    println("  PINN Test Evaluation")
    println("="^70)
    
    isfile(MODEL_PATH) || error("Model not found!")
    
    loaded = JLD2.load(MODEL_PATH)
    model = loaded["model"]
    np = loaded["norm"]
    
    # Load EXACT same data processing as validation
    df = CSV.read(DATA_PATH, DataFrame)
    df = filter(r -> r.qgi_Mscfd > 5.0 && r.qo_STBd > 5.0, df)
    
    # Add features EXACTLY as in training
    df[!, :drawdown] = df.Pres_psi .- df.Pbh_psi
    df[!, :qgi_ratio] = df.qgi_Mscfd ./ (df.qo_STBd .+ 1.0)
    
    X_cols = [:qgi_Mscfd, :Pres_psi, :GOR_scfSTB, :water_cut_frac,
              :API_gravity, :visc_cp, :depth_ft, :perm_mD,
              :drawdown, :qgi_ratio]
    
    X = Float32.(Matrix(df[:, X_cols]))
    
    # Normalize using saved params
    X_n = (X .- np.X_mean) ./ np.X_std
    
    # Predict on ALL data (like validation does)
    yhat_n = model(X_n')
    
    # Denormalize
    yhat = similar(yhat_n)
    for i in 1:4
        yhat[i:i, :] = yhat_n[i:i, :] .* np.y_std[1,i] .+ np.y_mean[1,i]
    end
    
    # Ground truth
    y_cols = [:qo_STBd, :Pwh_psi, :Pbh_psi, :qw_STBd]
    y_true = Matrix{Float32}(df[:, y_cols])'
    
    # Metrics on full dataset (consistent with validation)
    names = ["Oil Rate (STB/d)", "Wellhead Pressure (psi)", 
             "Bottomhole Pressure (psi)", "Water Rate (STB/d)"]
    
    println("\n  Full Dataset Results (consistent with validation):")
    println("  " * "-"^70)
    @printf("  %-22s %8s %10s %10s %10s\n", "Output", "MAPE(%)", "RMSE", "MAE", "R²")
    println("  " * "-"^70)
    
    for i in 1:4
        yt = y_true[i, :]
        yp = yhat[i, :]
        
        mape = mean(abs.((yt .- yp) ./ (abs.(yt) .+ 1e-8))) * 100
        rmse = sqrt(mean((yt .- yp).^2))
        mae = mean(abs.(yt .- yp))
        r2 = 1.0 - sum((yt .- yp).^2) / sum((yt .- mean(yt)).^2)
        
        @printf("  %-22s %8.2f %10.2f %10.2f %10.4f\n", names[i], mape, rmse, mae, r2)
    end
    
    # Time series for ND-001
    mkpath(RESULTS_DIR)
    ENV["GKSwstype"] = "nul"
    gr(dpi=300)
    
    mask = df.well_id .== "ND-001"
    if sum(mask) > 50
        dates_sub = df.date[mask]
        yt_qo = y_true[1, mask]
        yp_qo = yhat[1, mask]
        yt_Pwh = y_true[2, mask]
        yp_Pwh = yhat[2, mask]
        
        p1 = plot(dates_sub, yt_qo, lw=2, label="Measured", color=:steelblue)
        plot!(p1, dates_sub, yp_qo, lw=2, label="Predicted", color=:coral, ls=:dash)
        title!(p1, "ND-001 Oil Rate")
        
        p2 = plot(dates_sub, yt_Pwh, lw=2, label="Measured", color=:steelblue)
        plot!(p2, dates_sub, yp_Pwh, lw=2, label="Predicted", color=:coral, ls=:dash)
        title!(p2, "ND-001 Wellhead Pressure")
        
        plt = plot(p1, p2, layout=(2,1), size=(1200,800))
        savefig(plt, joinpath(RESULTS_DIR, "04_test_timeseries.png"))
        println("\n  Saved: 04_test_timeseries.png")
    end
    
    println("\n  Done!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_model()
end