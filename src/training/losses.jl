"""
    loss_fd(p)

Loss function to estimate forward kinematics.
"""
function loss_fd(p)
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
