##### Sprott system ############################################################
f(u, p, t) = [
    p[1] - u[2]
    u[3] + p[2]
    u[1] * u[2] - u[3]
];
h(u, p, t) = u[1];

# Estimated Sprott Q
u0 = [1.9999, 0.0009, 0.0011];
p = [0.8996, 0.4004];
sprott_e = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
sprott_sol_e, sprott_y_e = get_sys_solution(sprott_e);
samples = sprott_y_e;

# Real Sprott
u0 = [2.0, 0.0, 0.0];
p = [0.9, 0.4];
sprott = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
sprott_sol, sprott_y = get_sys_solution(sprott);
samples_r = sprott_y;

##### Pre-training of g ########################################################
g(u, p, t) =
    p[1] * u[1] + p[2] * u[2] + p[3] * u[3] +
    p[4] * u[1]^2 + p[5] * u[2]^2 + p[6] * u[3]^2 +
    p[7] * u[1] * u[2] + p[8] * u[2] * u[3] + p[9] * u[1] * u[3] + p[10];
obs_map(u, t) = [
    u[1]
    0.8996 - u[2]
    - u[3] - 0.4004
];

estp0 = randn(10) / 3;
Lf3h(u, t) = u[3] - u[1] * u[2];
estp = pretraining(g, Lf3h, obs_map, 1000, 10, n, Nsamples = 1000,
    estp0 = estp0);

##### Training of g ############################################################
sprottobs = init_system_obs(g, obs_map(u0, 0.0), p = estp, t0 = sprott.t0,
    tf = sprott.tf, ts = sprott.ts);

tfs = [30.0, 40.0, 50.0];
hu0, hp, times, estps = sysobs_training(sprottobs, samples, tfs, 100,
    save = true, callback = true, estp0 = estp, estu0 = obs_map(u0, 0.0),
    opt = Adam(1e-04), fixed_ic = true);

tfs = [50.0];
hu0, hp, times, estps = sysobs_training(sprottobs, samples, tfs, 100,
    save = true, callback = true, estp0 = hp, estu0 = obs_map(u0, 0.0),
    opt = Adam(1e-04), fixed_ic = true);

writedlm("sprott_hp.csv", hp, ",")

##### Gain tuning ##############################################################
hp = [
-0.899474058976291
0.00014005644203599974
-1.000072332873841
-9.794385624926087e-5
-0.00023828343121783634
0.00015261121994325285
0.9996516457721881
-0.0005655318581493434
2.5466983000665065e-5
-0.400045204827963
];
sprottobs = init_system_obs(g, obs_map(u0, 0.0), p = hp, t0 = sprott.t0,
    tf = sprott.tf, ts = sprott.ts);

#x, y = get_sys_solution(sprottobs, u0_arg = [-2.0, 0.1, 1.0]);
#plot(x')

d(t) = 0.1 * sin(100 * t);
ics = [
    2.0 0.8994 -0.4004
    1.5 0.5 1.0
    -2.0 0.1 1.0
];

#estp = gain_training(sprottobs, 10.0, 1000, d, ics, callback = true);
estp = [50.14, 0.04, 48.36, 1.94];

gain_plot(estp, sprottobs.t0, sprottobs.ts, sprottobs.tf)
gvals = gain_plot(estp, sprottobs.t0, sprottobs.ts, sprottobs.tf, get_vals = 1);
writedlm("sprottobs_gain.csv", gvals, ",")

x, hx = test_timevarying_hgo(sprottobs, estp, d,
    gain_type = UnivEst.TIMEVARYING_GAIN, hgo_type = UnivEst.CLASSICALHGO);

plot(x)
plot!(hx)

##### Estimation of inverse of observability map ###############################
N = 64;
n = 3;
sigm(x) = tanh(x) + 1;
W3(p) = reshape(p[1:(N * n)], n, :);
V3(p) = reshape(p[(N * n + 1):(N * n + N * n)], :, n);
lin3(p) = reshape(p[(N * n + N * n + 1):end]', n, :);
N3(u, p) = (W3(p) * sigm.(V3(p) * u)) + lin3(p) * u;

dummy_data = randn(n, 1000) * 4;
estp = inverse_training(N3, obs_map, 1000, N, n, data = dummy_data,
    opt = Adam(1e-03));

hsprott_sol = zeros(size(sprott_sol));
for i = 1:1:size(hsprott_sol, 2)
    hsprott_sol[:, i] = N3(hx[i, :], estp);
end

p1 = plot(sprott_sol[1, :]);
p1 = plot!(hsprott_sol[1, :]);
p2 = plot(sprott_sol[2, :]);
p2 = plot!(hsprott_sol[2, :]);
p3 = plot(sprott_sol[3, :]);
p3 = plot!(hsprott_sol[3, :]);
plot(p1, p2, p3, layout = (3, 1))

writedlm("sprott_n3.csv", estp, ",")
writedlm("sprott_x.csv", sprott_sol, ",")
writedlm("sprott_hx.csv", hsprott_sol, ",")
