## System definition and data collection

phi(u, p, t) = p[1] * u[3] + u[2]^2 + p[2] * u[1];
u0 = [5.0, 2.0, 0.0];
p = [-2.02, -1.0];

cj3 = init_system_obs(phi, u0, p, t0 = 0.0, tf = 50.0, ts = 1e-02,
    reltol = 1e-10, abstol = 1e-10);
times = cj3.t0:cj3.ts:(cj3.tf - cj3.ts);
d(t) = 0.1 * sin(1e03 * t);

include("../examples/high_gain_observers/compute_errors.jl")

eps1 = [0.02];
xi, hxi = estimate_t_derivatives(cj3, UnivEst.CLASSICALHGO, 2, eps1, d);
tildexi = compute_errors(UnivEst.CLASSICALHGO, xi, hxi);
p1 = plot(times, tildexi[:, 1:3]);

eps2 = [0.015, 0.002, 0.001];
xi1, z1 = estimate_t_derivatives(cj3, UnivEst.M_CASCADE, 2, eps2, d);
tildexi1 = compute_errors(UnivEst.M_CASCADE, xi1, z1);
p2 = plot(times, tildexi1[:, [1, 3, 4]]);

eps3 = [0.0075];
xi2, z2 = estimate_t_derivatives(cj3, UnivEst.CASCADE, 2, eps3, d);
tildexi2 = compute_errors(UnivEst.CASCADE, xi2, z2);
p3 = plot(times, tildexi2[:, [1, 3, 4]]);

yll = 1.5;
plot(p1, p2, p3, layout = (3, 1), ylim = (-yll, yll))

print(sum(abs.(tildexi[1001:end, 1])), ';',
      sum(abs.(tildexi[1001:end, 2])), ';',
      sum(abs.(tildexi[1001:end, 3])), '\n');
print(sum(abs.(tildexi1[1001:end, 1])), ';',
      sum(abs.(tildexi1[1001:end, 3])), ';',
      sum(abs.(tildexi1[1001:end, 4])), '\n');
print(sum(abs.(tildexi2[1001:end, 1])), ';',
      sum(abs.(tildexi2[1001:end, 3])), ';',
      sum(abs.(tildexi2[1001:end, 4])), '\n');

print(sum(abs.(tildexi[1001:end, 1])), ';',
      sum(abs.(tildexi[1001:end, 2])), ';',
      sum(abs.(tildexi[1001:end, 3])), ';',
      sum(abs.(tildexi[1001:end, 4])), ';',
      sum(abs.(tildexi[1001:end, 5])), '\n');
print(sum(abs.(tildexi1[1001:end, 1])), ';',
      sum(abs.(tildexi1[1001:end, 3])), ';',
      sum(abs.(tildexi1[1001:end, 5])), ';',
      sum(abs.(tildexi1[1001:end, 7])), ';',
      sum(abs.(tildexi1[1001:end, 9])), '\n');
print(sum(abs.(tildexi2[1001:end, 1])), ';',
      sum(abs.(tildexi2[1001:end, 3])), ';',
      sum(abs.(tildexi2[1001:end, 5])), ';',
      sum(abs.(tildexi2[1001:end, 7])), ';',
      sum(abs.(tildexi2[1001:end, 8])), '\n');

A1, B1 = get_hgo_matrices(UnivEst.CLASSICALHGO, 4, [], eps1);
A2, B2 = get_hgo_matrices(UnivEst.M_CASCADE, 4, [], eps2);
A3, B3 = get_hgo_matrices(UnivEst.CASCADE, 4, [], eps3);

using LinearAlgebra
eigen(A1)
eigen(A2)
eigen(A3)
