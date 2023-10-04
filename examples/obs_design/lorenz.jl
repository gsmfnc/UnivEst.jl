##### Lorenz system ############################################################
f(u, p, t) = [
    u[2] - u[1]
    - u[1] * u[3]
    u[1] * u[2] + p[1]
];
h(u, p, t) = u[1];

# Estimated Lorenz
u0 = [0.9933, 0.0007, 1.0015];
p = [-0.9994];
lor_e = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
lor_sol_e, lor_y_e = get_sys_solution(lor_e);
samples = lor_y_e;

# Real Lorenz
u0 = [1.0, 0.0, 1.0];
p = [-1.0];
lor = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
lor_sol, lor_y = get_sys_solution(lor);
samples_r = lor_y;

##### Pre-training of g ########################################################
N = 32;
n = 3;
sigma(x) = tanh(x) + 1;
W(p) = reshape(p[1:N], 1, :);
V(p) = reshape(p[N + 1:(N + N * n)], :, n);
lin(p) = p[(N + N * n + 1):end]';
g(u, p, t) = (W(p) * sigma.(V(p) * u))[1] + lin(p) * u;
obs_map(u, t) = [
    u[1]
    u[2] - u[1]
    - u[1] * u[3] - u[2] + u[1]
];

Lf3h(u, t) = - u[1] * (p[1] + u[1]^2) + u[1] * u[3] +
    (u[2] - u[1]) * (1 - u[3]);
estp = pretraining(g, Lf3h, obs_map, 1000, N, n, Nsamples = 500);

##### Training of g ############################################################
lorobs = init_system_obs(g, obs_map(u0, 0.0), p = estp, t0 = lor.t0,
    tf = lor.tf, ts = lor.ts);

tfs = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0];
hu0, hp, times, estps = sysobs_training(lorobs, samples, tfs, 300,
    save = true, callback = true, estp0 = estp, estu0 = obs_map(u0, 0.0),
    opt = Adam(1e-03));

tfs = [25.0];
hu0, hp, times, estps = sysobs_training(lorobs, samples, tfs, 100,
    save = true, callback = true, estp0 = hp, estu0 = hu0,
    opt = Adam(1e-04));

tfs = [30.0, 35.0, 40.0, 45.0];
hu0, hp, times, estps = sysobs_training(lorobs, samples, tfs, 100,
    save = true, callback = true, estp0 = hp, estu0 = hu0,
    opt = Adam(1e-05));

writedlm("lor_hp.csv", hp, ",")
writedlm("lor_hu0.csv", hu0, ",")

##### Gain tuning ##############################################################
lorobs = init_system_obs(g, hu0, p = hp, t0 = lor.t0, tf = lor.tf,
    ts = lor.ts);

#x, y = get_sys_solution(lorobs, u0_arg = [0.9933, -0.9925, -0.0021]);
#plot(x')

d(t) = 0.1 * sin(100 * t);
ics = [
    0.8862 -1.1288 -0.1280
    0.9933 -0.9925 -0.0021
];

#estp = gain_training(lorobs, 10.0, 1000, d, ics, callback = true,
    estW0 = [100.0, 0.0, 10.0, 10.0]);
estp = [50.88, -0.00, 47.71, 0.82]

gain_plot(estp, lorobs.t0, lorobs.ts, lorobs.tf)
gvals = gain_plot(estp, lorobs.t0, lorobs.ts, lorobs.tf, get_vals = 1);
writedlm("lorobs_gain.csv", gvals, ",")

x, hx = test_timevarying_hgo(lorobs, estp, d,
    gain_type = UnivEst.TIMEVARYING_GAIN, hgo_type = UnivEst.CLASSICALHGO);

plot(x)
plot!(hx)

##### Gradient-like inversion algorithm ########################################
K(u, t) = [
    1   -1  -u[3]+1
    0   1   -1
    0   0   -u[1]
];
d(t) = 0.1 * sin(100 * t);
grad_p = gradient_inversion_algorithm(lor_e, lorobs, K, obs_map, d, estp,
        10.0, 300, callback = true, opt = Adam(1e00))
grad_p = [95.46, 1.0, 53.42, -2.54, 43.17, 3.36]

x, hxi, hx = test_timevarying_hgo(lor, lorobs, grad_p[3:end], grad_p[1:2], d);

p1 = plot(x[:, 1]);
p1 = plot!(hx[:, 1]);
p2 = plot(x[:, 2]);
p2 = plot!(hx[:, 2]);
p3 = plot(x[:, 3]);
p3 = plot!(hx[:, 3]);
plot(p1, p2, p3, layout = (3, 1))

writedlm("lor_x.csv", x, ",")
writedlm("lor_hx.csv", hx, ",")

gvals = gain_plot(grad_p[3:end], lorobs.t0, lorobs.ts, lorobs.tf, get_vals = 1);

writedlm("lor_gain.csv", gvals, ",")

##### Estimation of inverse of observability map ###############################
N = 64;
n = 3;
W3(p) = reshape(p[1:(N * n)], n, :);
V3(p) = reshape(p[(N * n + 1):(N * n + N * n)], :, n);
lin3(p) = reshape(p[(N * n + N * n + 1):end]', n, :);
N3(u, p) = (W3(p) * sigma.(V3(p) * u)) + lin3(p) * u;

dummy_data = randn(n, 1000) * 2;
estp = inverse_training(N3, obs_map, 5000, N, n, data = dummy_data,
    opt = Adam(1e-03));

hlor_sol = zeros(size(lor_sol));
for i = 1:1:size(hlor_sol, 2)
    hlor_sol[:, i] = N3(hx[i, :], estp);
end

p1 = plot(lor_sol[1, :]);
p1 = plot!(hlor_sol[1, :]);
p2 = plot(lor_sol[2, :]);
p2 = plot!(hlor_sol[2, :]);
p3 = plot(lor_sol[3, :]);
p3 = plot!(hlor_sol[3, :]);
plot(p1, p2, p3, layout = (3, 1))

writedlm("lor_n3.csv", estp, ",")
writedlm("lor_x1.csv", lor_sol, ",")
writedlm("lor_hx1.csv", hlor_sol, ",")
