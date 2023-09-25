u0 = [0.9933, 0.0007, 1.0015];
p = [-0.9994];
f(u, p, t) = [
    u[2] - u[1]
    - u[1] * u[3]
    u[1] * u[2] + p[1]
];
h(u, p, t) = u[1];
lor = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
lor_sol, lor_y = get_sys_solution(lor);
samples = lor_y;

# Pre-training of g
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

# Training of g
lorobs = init_system_obs(g, obs_map(u0), p = estp, t0 = lor.t0, tf = lor.tf,
    ts = lor.ts);

tfs = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0];
hu0, hp, times, estps = sysobs_training(lorobs, samples, tfs, 300,
    save = true, callback = true, estp0 = estp, estu0 = obs_map(u0),
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

lorobs = init_system_obs(g, hu0, p = hp, t0 = lor.t0, tf = lor.tf,
    ts = lor.ts);

#x, y = get_sys_solution(lorobs, u0_arg = [0.9933, -0.9925, -0.0021]);
#plot(x')

d(t) = 0.1 * sin(100 * t);
ics = [
    0.8862 -1.1288 -0.1280
    0.9933 -0.9925 -0.0021
];
#estp = gain_training(lorobs, 10.0, 1000, d, ics, hgo_type = UnivEst.MIN_CASCADE,
#    S = [10.0, 10.0, 10.0], gain_type = UnivEst.DECREASING_GAIN,
#    callback = true);
estp = [53.41418738455853, 44.25187590816433, -1.3922708187327921];

gain_plot(estp, lorobs.t0, lorobs.ts, lorobs.tf,
    gain_type = UnivEst.DECREASING_GAIN)

x, hx = test_timevarying_hgo(lorobs, estp, d,
    gain_type = UnivEst.DECREASING_GAIN, hgo_type = UnivEst.MIN_CASCADE,
    S = [10.0, 10.0, 10.0]);
plot(x)
plot!(x - hx)

# Estimation of inverse of observability map
N = 32;
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
