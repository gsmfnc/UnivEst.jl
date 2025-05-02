# Chua's circuit (Table (3.4)) in "Sprott, Julien C. Elegant chaos:
# algebraically simple chaotic flows. World Scientific, 2010."

include("../../src/UnivEst.jl")

p_chua = [0.3, 1.0, -1.0, 1.0, 1.0, -1.0];
chua_dyn(u, p, t) = [
    p[1] * u[2] + p[2] * u[1] + p[3] * u[1]^3
    p[4] * u[1] + p[5] * u[3]
    p[6] * u[2] + sin(t)
];
chua_h(u, p, t) = u[1];
u0 = [0., -3., 1.];

chua = init_system(chua_dyn, chua_h, u0, p = p_chua, t0 = 0.0, tf = 50.0,
    ts = 1e-02);
sol, y = get_sys_solution(chua);

# Noise signal
noise_samples = randn(length(y), 1) * 0.05;
samples = y + noise_samples;
plot(0.0:1e-02:(50.0-1e-02), y, label = L"x_1")
plot!(0.0:1e-02:(50.0-1e-02), noise_samples,
    label = L"N_1", xlabel = L"t\ [s]")

############################ Identification
n = 3;
est_phi(u, p, t) = - u[1]^3 * p[3] * p[5] * p[6] + u[2] * p[5] * p[6] +
    p[1] * u[2] * p[4] - u[1] * p[2] * p[5] * p[6] + 3 * u[1]^2 * p[3] * u[3] +
    p[2] * u[3] + 6 * u[1] * u[2]^2 * p[3] + p[1] * p[5] * sin(t);
estp0 = vec(randn(6, 1));
est_duff = init_system_obs(est_phi, vec(zeros(n, 1)), p = estp0, t0 = 0.0,
    tf = 50.0, ts = 1e-02);

tfs = [1.0, 1.5];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 300,
    save = true, callback = true);
tfs = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 100,
    save = true, callback = true, estu0 = hu0, estp0 = hp);
tfs = [12.5, 15.0, 17.5, 20.0];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 100,
    save = true, callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-03));

############################ Controller design (with desired dynamics)
n = 3;
est_phi(u, p, t) = - u[1]^3 * p[3] * p[5] * p[6] + u[2] * p[5] * p[6] +
    p[1] * u[2] * p[4] - u[1] * p[2] * p[5] * p[6] + 3 * u[1]^2 * p[3] * u[3] +
    p[2] * u[3] + 6 * u[1] * u[2]^2 * p[3];

hp = [
  0.33251423830100524
  1.2169380271885124
 -0.881900587556232
  0.5780986442154339
  0.9156237828541423
 -1.0656548889935966
];

# parametric controller
u(x, p) = [
    0.0
    0.0
    p[1] * x[1] + p[2] * x[2] + p[3] * x[3] +
        p[4] * x[1]^2 + p[5] * x[2]^2 + p[6] * x[3]^2 +
        p[7] * x[1] * x[2] + p[8] * x[1] * x[3] + p[9] * x[2] * x[3] +
        p[10] * x[1]^3 + p[11] * x[2]^3 + p[12] * x[3]^3 +
        p[13] * x[1]^2 * x[2] + p[14] * x[1] * x[2]^2 +
        p[15] * x[1]^2 * x[3] + p[16] * x[1] * x[3]^2 +
        p[17] * x[2]^2 * x[3] + p[18] * x[2] * x[3]^2
];

# system dynamics
dyn_noctrl(x, p) = [
    x[2]
    x[3]
    est_phi(x, hp, 0.0)
];
f(x, p, t) = dyn_noctrl(x, p) + hp[1] * hp[5] * u(x, p);

# initialize controlled system
sys_to_ctrl = init_controlled_system(f, u);

using MatrixEquations
A = [
    0.  1.  0.
    0.  0.  1.
    -2. -2. -2.
];
f_des(x, p, t) = A * x;
Q = [
    1.  0.  0.
    0.  1.  0.
    0.  0.  1.
];
P = lyapc(A', Q, Q);

# Generate test points
test_points = zeros(n, 500);
radius = 0.1;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.1;
    end
end

# random initial guess
estp0 = vec(randn(18));

# training
# estp = ctrl_training(sys_to_ctrl, f_des, estp0, test_points, 2000, P);

estp = [
 -10.471200776678943
  -3.995302402223473
 -10.563413911599932
   0.0028352208271858843
  -0.0003410315652666398
   0.005772468026096798
  -0.0018492158456964109
   0.0012820322476743465
   0.00423114170756239
   2.827080074063784
   0.00021135659432088005
   0.002595736061714209
  -0.0003495602799284084
  17.379049086711973
   8.690571932150194
   0.0002000192933888996
  -0.0019772372682263202
   0.0009414746239598413
];

dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp, P);
plot(dV_values)

############################ Numerical simulation

eps = 0.001;
sat_val = 5.;
full_u(x, p) = [
    0.
    0.
    max(-sat_val, min(p[1] * x[1] + p[2] * x[2] + p[3] * x[3] +
        p[4] * x[1]^2 + p[5] * x[2]^2 + p[6] * x[3]^2 +
        p[7] * x[1] * x[2] + p[8] * x[1] * x[3] + p[9] * x[2] * x[3] +
        p[10] * x[1]^3 + p[11] * x[2]^3 + p[12] * x[3]^3 +
        p[13] * x[1]^2 * x[2] + p[14] * x[1] * x[2]^2 +
        p[15] * x[1]^2 * x[3] + p[16] * x[1] * x[3]^2 +
        p[17] * x[2]^2 * x[3] + p[18] * x[2] * x[3]^2, sat_val))
    0.
    0.
    0.
];
full_dyn(x, p, t) = [
    0.3 * x[2] + x[1] - x[1]^3
    x[1] + x[3]
    - x[2]
    x[5]                        + eps^-1 * 0.600 * (x[1] - x[4])
    x[6]                        + eps^-2 * 0.110 * (x[1] - x[4])
    est_phi(x[4:6], hp, 0.0)    + eps^-3 * 0.006 * (x[1] - x[4])
];
full_f(x, p, t) = full_dyn(x, p, t) + full_u(x[4:6], p);

control_sys = init_controlled_system(full_f, full_u);
desired_sys = init_controlled_system(f_des, u);

u0 = vec(randn(2 * n, 1) * 0.8);
t_sol, sol, u_vals = get_controlled_sys_solution(control_sys, estp, u0,
    tf = 30.0);
t_sol2, sol2, u_vals2 = get_controlled_sys_solution(desired_sys, estp, u0[1:n],
    tf = 30.0);

p1 = plot(t_sol, sol[1, :], label = L"x_1");
p1 = plot!(t_sol, sol2[1, :], label = L"\chi_{1,des}");
#p1 = plot!(t_sol, sol[4, :], label = L"\hat x_1");

p2 = plot(t_sol, sol[2, :], label = L"x_2");

p3 = plot(t_sol, sol[3, :], label = L"x_3");

p4 = plot(t_sol, u_vals[3, :], label = L"u(\hat\chi;p_{\star})",
    xlabel = L"t\ [s]");
plot(p1, p2, p3, p4, layout = (4, 1), legendfontsize = 12)
