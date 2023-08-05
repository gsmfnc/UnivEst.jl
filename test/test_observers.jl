mag1 = 0; mag2 = 0; mag3 = 0;
phase1 = 0; phase2 = 0; phase3 = 0;
w1 = 0; w2 = 0; w3 = 0;
for i = 2:1:4
    eps1 = vec(zeros(1, 1) .+ 1);
    eps2 = vec(zeros(i + 1, 1) .+ 1);
    eps3 = vec(zeros(1, 1) .+ 1);

    mag1, phase1, w1 = bode_hgo(UnivEst.CLASSICALHGO, i, eps1);
    mag2, phase2, w2 = bode_hgo(UnivEst.M_CASCADE, i, eps2);
    mag3, phase3, w3 = bode_hgo(UnivEst.CASCADE, i, eps3);
end

phi(u, p, t) = p[1] * u[2] + p[2] * u[1] - sinh(u[1]);
u0 = [-1.22, -0.04, 1.43];
p = [-5.0, 2.0];
cj3 = init_system_obs(phi, u0, p, t0 = 0.0, tf = 50.0, ts = 1e-02);

xi = 0; hxi = 0;

eps = [0.01];
for i = 1:1:4
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.CLASSICALHGO, i, eps);
    epsm = vec(zeros(i + 1, 1) .+ 0.01);
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.M_CASCADE, i, epsm);
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.CASCADE, i, eps);
end

d(t) = 0.1 * sin(1e03 * t);
for i = 1:1:4
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.CLASSICALHGO, i, eps, d);
    epsm = vec(zeros(i + 1, 1) .+ 0.01);
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.M_CASCADE, i, epsm, d);
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.CASCADE, i, eps, d);
end

lorenz(u, t) = 20 * [
    u[2] - u[1]
    -u[1] * u[3]
    u[1] * u[2] - 1
];
d(u) = 1e-01 * (u[1] + u[2] + u[3]);
lorenz_u0 = [1.0, 0.0, 1.0];
for i = 1:1:4
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.CLASSICALHGO, i, eps, d,
        lorenz, lorenz_u0);
    epsm = vec(zeros(i + 1, 1) .+ 0.01);
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.M_CASCADE, i, epsm, d, lorenz,
        lorenz_u0);
    xi, hxi = estimate_t_derivatives(cj3, UnivEst.CASCADE, i, eps, d, lorenz,
        lorenz_u0);
end

A1 = 0; A2 = 0; A3 = 0;
B1 = 0; B2 = 0; B3 = 0;

for i = 2:1:4
    eps1 = vec(zeros(1, 1) .+ 1);
    eps2 = vec(zeros(i + 1, 1) .+ 1);
    eps3 = vec(zeros(1, 1) .+ 1);

    A1, B1 = get_hgo_matrices(UnivEst.CLASSICALHGO, i, [], eps1);
    A2, B2 = get_hgo_matrices(UnivEst.M_CASCADE, i, [], eps2);
    A3, B3 = get_hgo_matrices(UnivEst.CASCADE, i, [], eps3);
end
