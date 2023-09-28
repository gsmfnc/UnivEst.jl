"""
    loss_invalg(p)

Loss function to estimate parameter of the gradient inversion algorithm.
"""
function loss_invalg(p)
    global SUPPENV

    vloss = 0.0;
    u0 = vcat(SUPPENV.u0, zeros(SUPPENV.m, 1), zeros(SUPPENV.n, 1));
    sol = get_sol(SUPPENV.dynamics, vec(u0), p, SUPPENV.t0, SUPPENV.tf,
        SUPPENV.ts, SUPPENV.tolerances);
    N = size(sol, 2);
    for i = 1:1:N
        vloss = vloss + sum(abs.(sol[1:SUPPENV.n, i] -
            sol[(SUPPENV.n + SUPPENV.m + 1):end, i]));
    end

    return 1 / N * vloss, sol;
end

"""
    loss_inv(p)

Loss function to estimate inverse of observability map.
"""
function loss_inv(p)
    global SUPPENV

    N = size(SUPPENV.data, 2);
    vloss = 0;
    for i = 1:1:N
        zeta = SUPPENV.data[:, i];
        vloss = vloss + sum(abs.(SUPPENV.N3(SUPPENV.obs_map(zeta, 0.0), p) -
            zeta));
    end

    return 1 / N * vloss;
end
"""
    loss_pre(p)

Loss function to estimate parameters using g(O_m(zeta))=L_f^m h(zeta).
"""
function loss_pre(p)
    global SUPPENV

    N = size(SUPPENV.data, 2);
    vloss = 0;
    for i = 1:1:N
        zeta = SUPPENV.data[:, i];
        vloss = vloss + sum(abs.(SUPPENV.g(SUPPENV.obs_map(zeta, 0.0), p, 0.0) -
            SUPPENV.Lfmh(zeta, 0.0)));
    end

    return 1 / N * vloss;
end
"""
    loss_gain(p)

Loss function to estimate parameters in time-varying gain.
"""
function loss_gain(p)
    global SUPPENV

    N = size(SUPPENV.ics, 1);
    vloss = 0.0;
    sol = 0;
    for i = 1:1:N
        u0 = vcat(SUPPENV.ics[i, :], zeros(SUPPENV.n, 1));
        sol = get_sol(SUPPENV.f, vec(u0), p, SUPPENV.t0, SUPPENV.tf, SUPPENV.ts,
            SUPPENV.tolerances);

        if SUPPENV.hgo_type == UnivEst.MIN_CASCADE
            x = sol[1:SUPPENV.n, :];
            z = sol[(SUPPENV.n + 1):(2 * SUPPENV.n), :];

            gain_func = get_gain_func(SUPPENV.gain_type);
            for j = SUPPENV.d_samples:1:size(z, 2)
                hx2 = get_mincascade_estimates(SUPPENV.n,
                    gain_func(p, (j - 1) * SUPPENV.ts), SUPPENV.coeffs,
                    SUPPENV.S, z[:, j]);
                vloss = vloss + sum(abs.(sol[1:SUPPENV.n, j] - hx2));
            end
        else
            for j = SUPPENV.d_samples:1:size(sol, 2)
                vloss = vloss + sum(abs.(sol[1:SUPPENV.n, j] -
                    sol[(SUPPENV.n + 1):(2 * SUPPENV.n), j]));
            end
        end
    end
    factor = (size(sol, 2) - SUPPENV.d_samples) * N;

    if length(p) == 4
        vloss = factor^-1 * vloss + 10^2 * p[1]^-2 + p[2]^2 + 10^2 * p[3]^-2 +
            0.1^2 * p[4]^2;
    end
    if length(p) == 3
        vloss = factor^-1 * vloss + 10^2 * p[1]^-2 + p[2]^-2 + 0.1^2 * p[3]^2;
    end
    if length(p) == 2
        vloss = factor^-1 * vloss + 10^2 * p[1]^-2 + p[2]^2;
    end

    return vloss, sol;
end

"""
    loss_sys(p)

Loss function to estimate nonlinear systems.
"""
function loss_sys(p)
    global SUPPENV

    hu0 = p[1:SUPPENV.n];
    hp = p[(SUPPENV.n + 1):end];

    sol = get_sol(SUPPENV.f, hu0, hp, SUPPENV.t0, SUPPENV.tf_tr, SUPPENV.ts,
        SUPPENV.tolerances);

    vloss = 0;
    if SUPPENV.obs_map(hu0, hp, 0.0) == 0
        for i = SUPPENV.d_samples:1:size(sol, 2)
            vloss = vloss + abs(SUPPENV.data[i] -
                SUPPENV.h(sol[:, i], hp, (i - 1) * SUPPENV.ts));
        end
    else
        for i = 1:1:SUPPENV.d_samples
            vloss = vloss + abs(SUPPENV.data[i, 1] -
                SUPPENV.h(sol[:, i], hp, (i - 1) * SUPPENV.ts));
        end
        for i = SUPPENV.d_samples:1:size(sol, 2)
            vloss = vloss + sum(abs.(SUPPENV.data[i, :] -
                SUPPENV.obs_map(sol[:, i], hp, (i - 1) * SUPPENV.ts)) ./
                SUPPENV.mxs);
        end
    end

    factor = size(sol, 2) - SUPPENV.d_samples;

    return factor^-1 * vloss, sol;
end

"""
    batch_loss_sys(p)

Batch loss function to estimate nonlinear systems.
"""
function batch_loss_sys(p)
    global SUPPENV

    hu0 = p[1:SUPPENV.n];
    hp = p[(SUPPENV.n + 1):end];

    sol = get_sol(SUPPENV.f, hu0, hp, SUPPENV.t0, SUPPENV.tf_tr, SUPPENV.ts,
        SUPPENV.tolerances);

    d_samples = Int(round((SUPPENV.dtime - SUPPENV.t0) / SUPPENV.ts)) + 1;

    if SUPPENV.batch_indexes[1] == 0
        for_indxs = d_samples:1:size(sol, 2);
    else
        for_indxs = SUPPENV.batch_indexes;
    end

    vloss = 0;
    if SUPPENV.obs_map(hu0, hp, 0.0) == 0
        for i = for_indxs
            vloss = vloss + abs(SUPPENV.data[i] -
                SUPPENV.h(sol[:, i], hp, (i - 1) * SUPPENV.ts));
        end
    else
        for i = for_indxs
            vloss = vloss + sum(abs.(SUPPENV.data[i, :] -
                SUPPENV.obs_map(sol[:, i], hp, (i - 1) * SUPPENV.ts)));
        end
    end

    factor = size(sol, 2) - d_samples;

    return factor^-1 * vloss, sol;
end

"""
    loss_sysobs(p)

Loss function to estimate systems in observability canonical form.
"""
function loss_sysobs(p)
    global SUPPENV

    hu0 = p[1:SUPPENV.n];
    hp = p[(SUPPENV.n + 1):end];

    sol = get_sol(SUPPENV.f, hu0, hp, SUPPENV.t0, SUPPENV.tf_tr, SUPPENV.ts,
        SUPPENV.tolerances);

    if SUPPENV.obs_map(hu0, hp, 0.0) == 0
        vloss = 0;
        for i = SUPPENV.d_samples:1:size(sol, 2)
            vloss = vloss + abs(SUPPENV.data[i] - sol[1, i]);
        end
    else
        vloss = 0;
        for i = 1:1:SUPPENV.d_samples
            vloss = vloss + abs(SUPPENV.data[i, 1] - sol[1, i]);
        end
        for i = SUPPENV.d_samples:1:size(sol, 2)
            vloss = vloss + sum(abs.(SUPPENV.data[i, :] -
                SUPPENV.obs_map(sol[:, i], hp, (i - 1) * SUPPENV.ts)) ./
                SUPPENV.mxs);
        end
    end

    factor = size(sol, 2) - SUPPENV.d_samples;

    return factor^-1 * vloss, sol;
end

"""
    loss_fd(p)

Loss function to estimate forward kinematics.
"""
function loss_fd(p)
    global SUPPENV

    N = Int(round((SUPPENV.tf_train - SUPPENV.fd_kin.t0) /
        SUPPENV.fd_kin.ts)) + 1;

    vloss = 0;
    for i = 1:1:N
        vloss = vloss + sum(abs.(SUPPENV.data[i, :] -
            SUPPENV.fd_kin.forward_kinematics(0, p,
                (i - 1) * SUPPENV.fd_kin.ts)));
    end

    return vloss;
end

"""
    loss_freq(p)

Loss function to estimate periodical signals.
"""
function loss_freq(p)
    global SUPPENV

    n = Int(floor(length(p) - 1) / 3);

    freqs = p[1:n];
    phases = p[(n + 1):(2 * n)];
    amps = p[(2 * n + 1):(3 * n)];
    bias = p[end];

    d_samples = Int(round(SUPPENV.t0 / SUPPENV.ts)) + 1;
    N = Int(round(SUPPENV.tf_tr / SUPPENV.ts));

    vloss = 0;
    for i = d_samples:1:N
        vloss = vloss + abs(SUPPENV.y_samples[i] -
            SUPPENV.s.s(bias, amps, freqs, phases, (i - 1) * SUPPENV.ts));
    end

    return vloss;
end
