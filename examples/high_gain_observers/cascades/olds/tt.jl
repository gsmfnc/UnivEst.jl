## System definition and data collection

phi(u, p, t) = p[1] * u[3] + u[2]^2 + p[2] * u[1];
u0 = [5.0, 2.0, 0.0];
p = [-2.02, -1.0];

cj3 = init_system_obs(phi, u0, p, t0 = 0.0, tf = 50.0, ts = 1e-02,
    reltol = 1e-10, abstol = 1e-10);
times = cj3.t0:cj3.ts:(cj3.tf - cj3.ts);
lorenz(u, p, t) = 20 * [
    u[2] - u[1]
    -u[1] * u[3]
    u[1] * u[2] - 1
];
d(u) = 1e-01 * (u[1] + u[2] + u[3]);
lorenz_u0 = [1.0, 0.0, 1.0];

eps = [0.1];
xi, hxi, dx = estimate_t_derivatives(cj3, UnivEst.CLASSICALHGO, 2, eps, d,
    lorenz, lorenz_u0);
tildexi = (xi - hxi)';
p1 = plot(times, tildexi);

eps = [0.01, 0.05, 0.1];
xi1, z1, dx1 = estimate_t_derivatives(cj3, UnivEst.M_CASCADE, 2, eps, d,
    lorenz, lorenz_u0);
xi1e = [xi1[1, :] xi1[2, :] xi1[2, :] xi1[3, :] xi1[3, :]];
tildexi1 = (xi1e' - z1)'; tildexi1 = tildexi1[:, [1, 2, 5]];
p2 = plot(times, tildexi1);

eps = [0.1];
xi2, z2, dx2 = estimate_t_derivatives(cj3, UnivEst.CASCADE, 2, eps, d,
    lorenz, lorenz_u0);
xi2e = [xi1[1, :] xi1[2, :] xi1[2, :] xi1[3, :]];
tildexi2 = (xi2e' - z2)'; tildexi2 = tildexi2[:, [1, 2, 4]];
p3 = plot(times, tildexi2);

dst_vec = zeros(length(times), 1);
for i = 1:1:length(times)
    dst_vec[i] = d(dx[:, i]);
end
p4 = plot(times, dst_vec);
plot!(times, xi[1, :]);

yll = 5;
plot(p1, p2, p3, p4, layout = (4, 1), ylim = (-yll, yll))

p11 = plot(times, xi[1, :]); plot!(times, hxi[1, :]);
p12 = plot(times, xi[2, :]); plot!(times, hxi[2, :]);
p13 = plot(times, xi[3, :]); plot!(times, hxi[3, :]);
p21 = plot(times, xi1[1, :]); plot!(times, z1[1, :]);
p22 = plot(times, xi1[2, :]); plot!(times, z1[3, :]);
p23 = plot(times, xi1[3, :]); plot!(times, z1[5, :]);
p31 = plot(times, xi2[1, :]); plot!(times, z2[1, :]);
p32 = plot(times, xi2[2, :]); plot!(times, z2[3, :]);
p33 = plot(times, xi2[3, :]); plot!(times, z2[4, :]);
yll = 3;
plot(p11, p12, p13, p21, p22, p23, p31, p32, p33,
    layout = (3, 3), ylim = (-yll, yll))

starting_indx = 101;
print(sum(abs.(tildexi[starting_indx:end, 1])), ';',
      sum(abs.(tildexi[starting_indx:end, 2])), ';',
      sum(abs.(tildexi[starting_indx:end, 3])), '\n',
      sum(abs.(tildexi1[starting_indx:end, 1])), ';',
      sum(abs.(tildexi1[starting_indx:end, 2])), ';',
      sum(abs.(tildexi1[starting_indx:end, 3])), '\n',
      sum(abs.(tildexi2[starting_indx:end, 1])), ';',
      sum(abs.(tildexi2[starting_indx:end, 2])), ';',
      sum(abs.(tildexi2[starting_indx:end, 3])), '\n');
print(maximum(abs.(tildexi[starting_indx:end, 1])), ';',
      maximum(abs.(tildexi[starting_indx:end, 2])), ';',
      maximum(abs.(tildexi[starting_indx:end, 3])), '\n',
      maximum(abs.(tildexi1[starting_indx:end, 1])), ';',
      maximum(abs.(tildexi1[starting_indx:end, 2])), ';',
      maximum(abs.(tildexi1[starting_indx:end, 3])), '\n',
      maximum(abs.(tildexi2[starting_indx:end, 1])), ';',
      maximum(abs.(tildexi2[starting_indx:end, 2])), ';',
      maximum(abs.(tildexi2[starting_indx:end, 3])), '\n');
