# System CJ_3 from Table 4.1 in "Sprott, Julien C. Elegant chaos: algebraically
# simple chaotic flows. World Scientific, 2010."

## Bode diagrams

mag1, phase1, w1 = bode_hgo(UnivEst.CLASSICALHGO, 2, [0.1]);
mag2, phase2, w2 = bode_hgo(UnivEst.M_CASCADE, 2, [0.01, 0.1, 0.1]);
mag3, phase3, w3 = bode_hgo(UnivEst.CASCADE, 2, [0.1]);

include("../examples/high_gain_observers/cascades/bode_plots.jl")
p1, p2, p3 = bode_diagram(hgo, mcascade, cascade);
p4 = plot(p1, p2, p3, layout = (3, 1));

## System definition and time derivatives estimation

phi(u, p, t) = p[1] * u[2] + p[2] * u[1] - sinh(u[1]);
u0 = [-1.22, -0.04, 1.43];
p = [-5.0, 2.0];

cj3 = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
d(t) = 0.1 * sin(1e03 * t);
times = cj3.t0:cj3.ts:(cj3.tf - cj3.ts);

# Standard high-gain observer

yll = 10;

eps = [0.005];
xi, hxi = estimate_t_derivatives(cj3, UnivEst.CLASSICALHGO, 2, eps, d);
tildexi = (xi - hxi)';
p51 = plot(times, tildexi, ylim = (-yll, yll));

using DelimitedFiles
writedlm("tildexi_hgo_eps005.csv", tildexi, ',');

eps = [0.01];
xi, hxi = estimate_t_derivatives(cj3, UnivEst.CLASSICALHGO, 2, eps, d);
tildexi = (xi - hxi)';
p52 = plot(times, tildexi, ylim = (-yll, yll));

writedlm("tildexi_hgo_eps01.csv", tildexi, ',');

eps = [0.025];
xi, hxi = estimate_t_derivatives(cj3, UnivEst.CLASSICALHGO, 2, eps, d);
tildexi = (xi - hxi)';
p53 = plot(times, tildexi, ylim = (-yll, yll));

writedlm("tildexi_hgo_eps025.csv", tildexi, ',');

# m-cascade of 2nd-order high-gain observers

eps = [0.001, 0.005, 0.01];
xi1, z1 = estimate_t_derivatives(cj3, UnivEst.M_CASCADE, 2, eps, d);
xi1e = [xi1[1, :] xi1[2, :] xi1[2, :] xi1[3, :] xi1[3, :]];
tildexi1 = (xi1e' - z1)'; tildexi1 = tildexi1[:, [1, 3, 5]];
p6 = plot(times, tildexi1, ylim = (-yll, yll));

writedlm("tildexi_mcasc_001_005_01.csv", tildexi1, ',');

# Cascade of high-gain observers

eps = [0.005];
xi2, z2 = estimate_t_derivatives(cj3, UnivEst.CASCADE, 2, eps, d);
xi2e = [xi1[1, :] xi1[2, :] xi1[2, :] xi1[3, :]];
tildexi2 = (xi2e' - z2)'; tildexi2 = tildexi2[:, [1, 3, 4]];
p7 = plot(times, tildexi2, ylim = (-yll, yll));

writedlm("tildexi_casc_005.csv", tildexi2, ',');

# Observers matrices eigenvalues

A1, B1 = get_hgo_matrices(UnivEst.CLASSICALHGO, 2, [], [0.01]);
A2, B2 = get_hgo_matrices(UnivEst.M_CASCADE, 2, [], [0.001, 0.005, 0.01]);
A3, B3 = get_hgo_matrices(UnivEst.CASCADE, 2, [], [0.005]);

using LinearAlgebra
eigen(A1)
eigen(A2)
eigen(A3)
