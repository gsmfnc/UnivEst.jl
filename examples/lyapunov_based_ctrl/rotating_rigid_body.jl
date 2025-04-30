# Rotating rigid body (Example 4.10 in "Nonlinear systems - Third Edition - 
# Hassan K. Khalil")

u0 = [0.0, 0.0, 0.0];

h(u, p, t) = [
    u[1]
    u[2]
    u[3]
];
p = [0.5, 0.3, 0.6];
inp(t) = [
    sin(2.4944 * t)
    sin(5.9451 * t)
    sin(7.1838 * t)
];
f(u, p, t) = [
    1 / p[1] * (p[2] - p[3]) * u[2] * u[3]
    1 / p[2] * (p[3] - p[1]) * u[3] * u[1]
    1 / p[3] * (p[1] - p[2]) * u[1] * u[2]
] + inp(t);

rot_rigid_body = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0,
    ts = 1e-02);
sol, y = get_sys_solution(rot_rigid_body);

# Noise signal
noise_samples = randn(size(y, 1), 3) * 0.05;
samples = y + noise_samples;

p1 = plot(0.0:1e-02:(50.0-1e-02), y[:, 1], label = L"x_1");
p1 = plot!(0.0:1e-02:(50.0-1e-02), noise_samples[:, 1],
    label = L"N_1");
p2 = plot(0.0:1e-02:(50.0-1e-02), y[:, 2], label = L"x_2");
p2 = plot!(0.0:1e-02:(50.0-1e-02), noise_samples[:, 2],
    label = L"N_2");
p3 = plot(0.0:1e-02:(50.0-1e-02), y[:, 3], label = L"x_3");
p3 = plot!(0.0:1e-02:(50.0-1e-02), noise_samples[:, 3],
    label = L"N_3", xlabel = L"t\ [s]");

plot(p1, p2, p3, layout = (3, 1))

############################ IDENTIFICATION
n = 3;
N = 64;
Np1 = 2 * N * n;
Np2 = 2 * N * n;
sigma(x) = tanh(x) + 1;
W1(p) = reshape(p[1:(N * n)], n, :);
V1(p) = reshape(p[(N * n + 1):(N * n + N * n)], :, n);
W2(p) = reshape(p[1:(N * n)], n, :);
V2(p) = reshape(p[(N * n + 1):(N * n + N * n)], :, n);
lin(p) = reshape(p[(Np1 + Np2 + 1):end], n, n);

est_f(u, p, t) = (W1(p[1:Np1]) * sigma.(V1(p[1:Np1]) *
    W2(p[(Np1 + 1):(Np1 + Np2)]) * sigma.(V2(p[(Np1 + 1):(Np1 + Np2)]) *
    u))) + lin(p) * u + inp(t);
estp0 = vec(randn(Np1 + Np2 + n * n, 1));
est_sys = init_system(est_f, h, u0, p = estp0, t0 = 0.0, tf = 50.0,
    ts = 1e-02);

tfs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.5, 15.0, 20.0];
hu0, hp, times, estps = sys_training(est_sys, samples, tfs, 100,
    save = true, callback = false, estu0 = u0);

tfs = [25.0, 30.0];
hu0, hp, times, estps = sys_training(est_sys, samples, tfs, 100,
    save = true, callback = false, estu0 = u0, estp0 = vec(hp));

tfs = [35.0, 40.0, 45.0, 50.0];
hu0, hp, times, estps = sys_training(est_sys, samples, tfs, 100,
    save = true, callback = false, estu0 = u0, estp0 = vec(hp));

est_sys = init_system(est_f, h, u0, p = vec(hp), t0 = 0.0, tf = 50.0,
    ts = 1e-02);
hsol, hy = get_sys_solution(est_sys);
p1 = plot(samples[:, 1]);
p1 = plot!(hsol[1, :]);
p2 = plot(samples[:, 2]);
p2 = plot!(hsol[2, :]);
p3 = plot(samples[:, 3]);
p3 = plot!(hsol[3, :]);
plot(p1, p2, p3, layout = (3, 1))

############################ CONTROLLER DESIGN
n = 3;
N = 64;
Np1 = 2 * N * n;
Np2 = 2 * N * n;
sigma(x) = tanh(x) + 1;
W1(p) = reshape(p[1:(N * n)], n, :);
V1(p) = reshape(p[(N * n + 1):(N * n + N * n)], :, n);
W2(p) = reshape(p[1:(N * n)], n, :);
V2(p) = reshape(p[(N * n + 1):(N * n + N * n)], :, n);
lin(p) = reshape(p[(Np1 + Np2 + 1):end], n, n);

include("examples/lyapunov_based_ctrl/estimates_rotating_rigid_body.jl")

# parametric controller
u(x, p) = [
    p[1] * x[1] + p[2] * x[2] + p[3] * x[3]
    p[4] * x[1] + p[5] * x[2] + p[6] * x[3]
    p[7] * x[1] + p[8] * x[2] + p[9] * x[3]
];

dyn_noctrl(x, p) = (W1(p[1:Np1]) * sigma.(V1(p[1:Np1]) *
    W2(p[(Np1 + 1):(Np1 + Np2)]) * sigma.(V2(p[(Np1 + 1):(Np1 + Np2)]) *
    x))) + lin(p) * x;
f(x, p, t) = dyn_noctrl(x, hp) + u(x, p);

# initialize controlled system
sys_to_ctrl = init_controlled_system(f, u);

# Generate test points
test_points = zeros(n, 2000);
radius = 0.1;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.1;
    end
end

estp0 = randn(9);
#estp1 = ctrl_training(sys_to_ctrl, estp0, test_points, 1000,
#    alpha = 1e00, gamma = 1e-02);
estp1 = [
 -0.49346610764412635
  0.038339050064759524
 -0.03607824922518825
  0.06872289058564086
 -0.5924461373865579
  0.16423837613021647
 -0.09577200568941331
  0.3276769801492247
 -1.2560306003357713
];

#estp2 = ctrl_training(sys_to_ctrl, estp0, test_points, 1000,
#    alpha = 1e01, gamma = 1e-02);
estp2 = [
 -0.22817922947082564
  0.048017208140689015
 -0.04686977985625742
  0.11872610112018847
 -0.49731873067289883
  0.23564332968235446
 -0.2512460748322884
  0.48556866848572683
 -1.0666469932946618
];

# alpha = 5e00
estp = [
 -2.5317602368565644
  0.26668493465377624
 -0.2536309658456564
  0.07357210267874616
 -0.6817032958798577
  0.02189251201989707
  0.20707940716595954
  0.03484854171187378
 -1.902860170016643
];

############################ TEST ON REAL DYNAMICS
dist_dyn(x) = [
    0.24*x[1]+0.37*x[2]-0.53*x[3]+0.71*x[1]^2-0.84*x[2]^2-0.90*x[3]^2
   -1.28*x[1]+0.62*x[2]+0.51*x[3]-0.45*x[1]^2-0.51*x[2]^2-2.40*x[3]^2
   -0.46*x[1]+2.46*x[2]-0.75*x[3]+0.38*x[1]^2-0.26*x[2]^2+0.48*x[3]^2
];
params = [0.5, 0.3, 0.6];
real_dyn_noctrl(u, p) = [
    1 / p[1] * (p[2] - p[3]) * u[2] * u[3]
    1 / p[2] * (p[3] - p[1]) * u[3] * u[1]
    1 / p[3] * (p[1] - p[2]) * u[1] * u[2]
];
real_f(x, p, t) = real_dyn_noctrl(x, params) + u(x, p) + p[10] * dist_dyn(x);
real_sys_to_ctrl = init_controlled_system(real_f, u);

#### "Nominal" response
u0e = [-0.237, 1.01, -1.589];
estp = vec(vcat(estp1, [0.0]));
t_sol1, sol1, u_vals1 = get_controlled_sys_solution(real_sys_to_ctrl, estp, u0e,
    tf = 15.0);
estp = vec(vcat(estp2, [0.0]));
t_sol2, sol2, u_vals2 = get_controlled_sys_solution(real_sys_to_ctrl, estp, u0e,
    tf = 15.0);

p1 = plot(t_sol1, sol1', label = [L"x_1" L"x_2" L"x_3"],
    ylabel = L"\alpha = 1");
p2 = plot(t_sol1, u_vals1',
    label = [L"u_1(x;p_{\star,1})" L"u_2(x;p_{\star,1})" L"u_3(x;p_{\star,1})"],
    ylim = (-1, 2.5));
p3 = plot(t_sol2, sol2', label = [L"x_1" L"x_2" L"x_3"], xlabel = L"t\ [s]",
    ylabel = L"\alpha = 10");
p4 = plot(t_sol2, u_vals2',
    label = [L"u_1(x;p_{\star,2})" L"u_2(x;p_{\star,2})" L"u_3(x;p_{\star,2})"],
    xlabel = L"t\ [s]", ylim = (-1, 2.5));
plot(p1, p2, p3, p4, layout = (2, 2))

### Time response with parasitic dynamics
u0e = [-0.237, 1.01, -1.589];
amp1 = 0.3;
estp = vec(vcat(estp1, [amp1]));
t_sol1, sol1, u_vals1 = get_controlled_sys_solution(real_sys_to_ctrl, estp, u0e,
    tf = 50.0);
estp = vec(vcat(estp2, [amp1]));
t_sol2, sol2, u_vals2 = get_controlled_sys_solution(real_sys_to_ctrl, estp, u0e,
    tf = 50.0);

p1 = plot(t_sol1, sol1', label = [L"x_1" L"x_2" L"x_3"],
    ylabel = L"\alpha = 1");
p2 = plot(t_sol1, u_vals1',
    label = [L"u_1(x;p_{\star,1})" L"u_2(x;p_{\star,1})" L"u_3(x;p_{\star,1})"],
    ylim = (-1, 2.5));
p3 = plot(t_sol2, sol2', label = [L"x_1" L"x_2" L"x_3"], xlabel = L"t\ [s]",
    ylabel = L"\alpha = 10");
p4 = plot(t_sol2, u_vals2',
    label = [L"u_1(x;p_{\star,2})" L"u_2(x;p_{\star,2})" L"u_3(x;p_{\star,2})"],
    xlabel = L"t\ [s]", ylim = (-1, 2.5));
plot(p1, p2, p3, p4, layout = (2, 2))

0.3 * [0.24 0.37 -0.53 0.71 -0.84 -0.90
-1.28 0.62 0.51 -0.45 -0.51 -2.40
-0.46 2.46 -0.75 0.38 -0.26 0.48]
