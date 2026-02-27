# data_generation.jl 
# Niger Delta Gas-Lift Dataset (2020-2024)

using CSV, DataFrames, Random, Dates, Statistics, Printf

Random.seed!(42)

# Path handling
const PROJECT_ROOT = @__DIR__
const DATA_DIR = joinpath(PROJECT_ROOT, "..", "data")

const WELLS = [
    (id="ND-001", k=285.0, h=87.0,  D=9210.0,  API=28.5, mu=1.85, Pres=2845.0, Pb=1650.0, GOR=625.0,  fw_i=0.42, fw_f=0.78, qi=485.0,  qgi_opt=125.0, Di=0.185, b=0.52),
    (id="ND-002", k=425.0, h=94.0,  D=8820.0,  API=31.2, mu=1.45, Pres=2650.0, Pb=1480.0, GOR=485.0,  fw_i=0.55, fw_f=0.88, qi=685.0,  qgi_opt=165.0, Di=0.235, b=0.42),
    (id="ND-003", k=195.0, h=79.0,  D=9790.0,  API=35.8, mu=0.95, Pres=3125.0, Pb=1820.0, GOR=895.0,  fw_i=0.25, fw_f=0.62, qi=385.0,  qgi_opt=95.0,  Di=0.155, b=0.58),
    (id="ND-004", k=565.0, h=107.0, D=8395.0,  API=24.8, mu=2.65, Pres=2480.0, Pb=1320.0, GOR=325.0,  fw_i=0.68, fw_f=0.94, qi=825.0,  qgi_opt=195.0, Di=0.265, b=0.38),
    (id="ND-005", k=340.0, h=89.0,  D=9495.0,  API=30.5, mu=1.68, Pres=2785.0, Pb=1585.0, GOR=585.0,  fw_i=0.48, fw_f=0.82, qi=545.0,  qgi_opt=135.0, Di=0.195, b=0.48),
    (id="ND-006", k=455.0, h=96.0,  D=8905.0,  API=29.1, mu=1.95, Pres=2620.0, Pb=1425.0, GOR=425.0,  fw_i=0.72, fw_f=0.96, qi=625.0,  qgi_opt=155.0, Di=0.295, b=0.40),
    (id="ND-007", k=245.0, h=83.0,  D=10095.0, API=33.2, mu=1.15, Pres=3250.0, Pb=1925.0, GOR=745.0,  fw_i=0.22, fw_f=0.58, qi=425.0,  qgi_opt=105.0, Di=0.165, b=0.55),
    (id="ND-008", k=380.0, h=92.0,  D=9305.0,  API=31.8, mu=1.52, Pres=2850.0, Pb=1620.0, GOR=535.0,  fw_i=0.52, fw_f=0.85, qi=595.0,  qgi_opt=145.0, Di=0.205, b=0.46),
]

arps_q(qi, Di, b, t) = b < 0.01 ? qi * exp(-Di * t) : qi / (1.0 + b * Di * t)^(1.0 / b)
water_cut(fw_i, fw_f, t; lam=2.2) = fw_i + (fw_f - fw_i) * (1.0 - exp(-lam * t))

function glpc_factor(qgi, qgi_opt)
    qgi <= 0.5 && return 0.18
    r = qgi / qgi_opt
    eta = r * exp(1.0 - r)
    over = max(0.0, r - 1.0)
    return clamp(eta * (1.0 - 0.12 * over - 0.045 * over^2), 0.04, 1.02)
end

function oil_fvf(API, GOR, T=185.0)
    gamma_o = 141.5 / (API + 131.5)
    term = GOR * sqrt(gamma_o / 0.75) + 1.25 * T
    return 0.972 + 1.47e-4 * term^1.175
end

# FIXED: More realistic Pwh calculation with proper gas lift physics
function compute_pressures(D, qgi, qo, qw, API, Pres, GOR, t_yr)
    ql = qo + qw
    rho_o = 141.5 / (API + 131.5) * 62.4
    fw = ql > 1.0 ? qw / ql : 0.0
    rho_l = fw * 64.2 + (1.0 - fw) * rho_o
    
    # Gas-liquid ratio
    glr = ql > 0.5 ? (GOR + qgi * 1000.0 / ql) : GOR
    
    # Modified Hagedorn-Brown holdup
    Ek = 1.0 + 0.00018 * glr^0.75 * (1.0 + 0.002 * t_yr)
    rho_m = rho_l / Ek
    
    # CRITICAL FIX: Pwh now properly depends on qgi with wider range
    base_pwh = 85.0 + 15.0 * rand()  # Base separator pressure varies 85-100 psi
    
    # Gas lift increases Pwh significantly
    gas_lift_effect = 8.0 * log(max(qgi, 1.0))
    
    # Friction/flow effect
    flow_effect = 0.02 * ql
    
    # Seasonal variation
    month = Dates.month(Date(2020, 1, 1) + Day(round(Int, t_yr * 365.25)))
    seasonal = month in 5:10 ? -5.0 : 3.0
    
    Pwh = base_pwh + gas_lift_effect + flow_effect + seasonal + randn() * 3.0
    
    # Wider realistic range for Niger Delta: 80-200 psi
    Pwh = clamp(Pwh, 80.0, 200.0)
    
    # Hydrostatic + friction
    dP_hydro = rho_m * D / 144.0
    dP_fric = 0.015 * ql^1.68 / (D^0.22 + 50.0)
    
    Pbh_raw = Pwh + dP_hydro + dP_fric
    Pbh = clamp(Pbh_raw, Pwh + 95.0, Pres * 0.94)
    
    return Pwh, Pbh, rho_m
end

function make_qgi_schedule(qgi_opt, Di, b, n_days)
    qgi = zeros(n_days)
    current = qgi_opt * 0.78
    campaigns = Set([1, 91, 182, 273, 365, 456, 547, 638, 730, 821, 912, 1003, 1095, 1186, 1277, 1462])
    
    for i in 1:n_days
        t_yr = (i - 1) / 365.25
        qgi_t = qgi_opt * max(0.55, arps_q(1.0, Di * 0.85, b, t_yr))
        
        if i in campaigns
            current = qgi_t * (0.82 + 0.28 * rand())
        end
        
        current += randn() * 2.2
        
        month = Dates.month(Date(2020, 1, 1) + Day(i - 1))
        if month in 5:10
            current *= (0.88 + 0.04 * rand())
        end
        
        qgi[i] = clamp(current, 15.0, qgi_opt * 2.4)
        
        if rand() < 0.025
            qgi[i] = 0.0
        end
        
        if i % 180 == 0 && rand() < 0.3
            qgi[i:min(i+2, n_days)] .= 0.0
        end
    end
    
    return qgi
end

function generate_data(; verbose=true)
    verbose && println("="^70, "\n  Niger Delta Gas-Lift Dataset Generator (2020-2024)\n", "="^70)

    dates = collect(Date(2020, 1, 1):Day(1):Date(2024, 12, 31))
    n_days = length(dates)
    rows = []

    for well in WELLS
        verbose && print("  Processing $(well.id) ... ")
        qgi_sched = make_qgi_schedule(well.qgi_opt, well.Di, well.b, n_days)
        Bo_ref = oil_fvf(well.API, well.GOR)

        for (i, date) in enumerate(dates)
            t_yr = (i - 1) / 365.25
            qgi = qgi_sched[i]

            Pres = well.Pres * (1.0 - 0.085 * t_yr / 5.0) + randn() * 15.0
            Pb = well.Pb * (1.0 - 0.045 * t_yr / 5.0) + randn() * 10.0
            
            fw = clamp(water_cut(well.fw_i, well.fw_f, t_yr) + randn() * 0.012, 0.0, 0.98)
            
            if Pres > Pb
                GOR = well.GOR * (1.0 + randn() * 0.03)
            else
                GOR = well.GOR * (1.0 + 0.25 * (Pb - Pres) / Pb) * (1.0 + randn() * 0.04)
            end
            GOR = max(GOR, 80.0)
            
            Bo = oil_fvf(well.API, GOR)
            
            qi_t = arps_q(well.qi, well.Di, well.b, t_yr)
            eta = glpc_factor(qgi, well.qgi_opt)
            
            qo = max(qi_t * eta * (1.0 - fw) * (1.0 + randn() * 0.028), 1.5)
            qw = max(fw > 0.0 ? qo * fw / (1.0 - fw + 1e-9) : 0.0, 0.0) * (1.0 + randn() * 0.025)

            Pwh, Pbh, rho_m = compute_pressures(well.D, qgi, qo, qw, well.API, Pres, GOR, t_yr)
            
            qg_tot = GOR * qo / 1000.0 + max(qgi, 0.0)
            eff = qgi > 5.0 ? qo / qgi : 0.0

            push!(rows, (
                date = date,
                well_id = well.id,
                qgi_Mscfd = round(max(qgi, 0.0), digits=2),
                qo_STBd = round(qo, digits=1),
                qw_STBd = round(qw, digits=1),
                Pres_psi = round(Pres, digits=1),
                Pwh_psi = round(Pwh, digits=1),
                Pbh_psi = round(Pbh, digits=1),
                GOR_scfSTB = round(GOR, digits=1),
                water_cut_frac = round(fw, digits=4),
                API_gravity = round(well.API + randn() * 0.22, digits=2),
                perm_mD = round(well.k + randn() * 6.0, digits=1),
                visc_cp = round(well.mu * (1.0 + randn() * 0.035), digits=3),
                depth_ft = round(well.D + randn() * 12.0, digits=1),
                mix_dens_lbft3 = round(rho_m, digits=2),
                glift_eff = round(eff, digits=3),
                qg_total_Mscfd = round(qg_tot, digits=3),
            ))
        end
        verbose && println("done ($(n_days) days)")
    end

    df = DataFrame(rows)
    mkpath(DATA_DIR)
    CSV.write(joinpath(DATA_DIR, "niger_delta_gaslift_daily.csv"), df)

    # Monthly aggregation
    df[!, :ym] = Dates.format.(df.date, "yyyy-mm")
    gdf = groupby(df, [:well_id, :ym])
    df_mo = combine(gdf,
        :qgi_Mscfd => mean => :qgi_Mscfd,
        :qo_STBd => mean => :qo_STBd,
        :qw_STBd => mean => :qw_STBd,
        :water_cut_frac => mean => :water_cut_frac,
        :Pres_psi => mean => :Pres_psi,
        :GOR_scfSTB => mean => :GOR_scfSTB,
        :Pwh_psi => mean => :Pwh_psi,
        :Pbh_psi => mean => :Pbh_psi,
        :glift_eff => mean => :glift_eff,
        :mix_dens_lbft3 => mean => :mix_dens_lbft3,
        :depth_ft => first => :depth_ft,
        :perm_mD => first => :perm_mD,
        :API_gravity => first => :API_gravity,
        :visc_cp => first => :visc_cp,
    )
    CSV.write(joinpath(DATA_DIR, "niger_delta_gaslift_monthly.csv"), df_mo)

    if verbose
        println("\n  Summary Statistics:")
        @printf("  %-8s  %-14s  %-14s  %-10s  %-10s  %-10s  %-10s\n",
                "Well", "qo mean", "qgi mean", "WC 2020", "WC 2024", "GOR 2024", "Pwh range")
        println("  " * "-"^80)
        for well in WELLS
            wdf = filter(r -> r.well_id == well.id && r.qgi_Mscfd > 5.0, df)
            wc_s = mean(filter(r -> r.well_id == well.id && year(r.date) == 2020, df).water_cut_frac)
            wc_e = mean(filter(r -> r.well_id == well.id && year(r.date) == 2024, df).water_cut_frac)
            gor_e = mean(filter(r -> r.well_id == well.id && year(r.date) == 2024, df).GOR_scfSTB)
            pwh_min = minimum(wdf.Pwh_psi)
            pwh_max = maximum(wdf.Pwh_psi)
            @printf("  %-8s  %-14.0f  %-14.1f  %-10.2f  %-10.2f  %-10.0f  %-3.0f-%-3.0f\n",
                    well.id, mean(wdf.qo_STBd), mean(wdf.qgi_Mscfd), wc_s, wc_e, gor_e, pwh_min, pwh_max)
        end
        println()
        @printf("  Daily rows   : %d\n", nrow(df))
        @printf("  Monthly rows : %d\n", nrow(df_mo))
        @printf("  Cumul. oil   : %.3f MMSTB\n", sum(df.qo_STBd) / 1e6)
        
        violations = sum(df.Pbh_psi .>= df.Pres_psi)
        @printf("  Pbh >= Pres violations : %d (must be 0)\n", violations)
        
        pwh_std = std(df.Pwh_psi)
        @printf("  Pwh std dev  : %.2f psi (should be >10 for good learning)\n", pwh_std)
        
        println("\n  Files saved -> data/")
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_data()
end