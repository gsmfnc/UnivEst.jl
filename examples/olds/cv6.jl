phi(u, p, t) = p[2] * atan(u[3] + u[1]) - u[3] + (p[1] - 1) * u[2] - u[1];
u0 = [0.7, 6.0, 0.0];
p = [0.2, 2];
t0 = 0.0;
tf = 35.0;
ts = 1e-02;
opt = ADAM(1e-02);

env = init_env(phi, u0, p, t0, tf, ts, opt, reltol = 1e-6, abstol = 1e-6);

epsilon = [1e-02];
data1 = estimate_time_derivatives(env, UnivEst.CLASSICALHGO, epsilon, 2,
    Ts = 1e-04);

env = set_env_parameter(env, "data", data1);
env = set_env_parameter(env, "M", 3);

tfs = [5.0];
estp = randn(length(env.u0) + length(env.p)) * 1e-03;
hp = training_routine(env, tfs, estp, 500)

epsilon = [1e-02, 1e-02, 1e-02];
data = estimate_time_derivatives(env, UnivEst.CASCADE1, epsilon, 2,
   Ts = 1e-04);
data2 = extract_estimates(data, UnivEst.CASCADE1);

env = set_env_parameter(env, "data", data2[1]);

tfs = [5.0, 10.0, 15.0];
estp = randn(length(env.u0) + length(env.p)) * 1e-03;
hp = training_routine(env, tfs, estp, 500)

epsilon = [1e-02];
data = estimate_time_derivatives(env, UnivEst.CASCADE2, epsilon, 2,
   Ts = 1e-04);
data3 = extract_estimates(data, UnivEst.CASCADE2);

env = set_env_parameter(env, "data", data3[1]);

tfs = [5.0, 10.0, 15.0];
estp = randn(length(env.u0) + length(env.p)) * 1e-03;
hp = training_routine(env, tfs, estp, 500)

################################################################################

f(u, p, t) = [
    u[2];
    u[3];
    p[2] * atan(u[3] + u[1]) - u[3] + (p[1] - 1) * u[2] - u[1];
];
O(u, p, t) = [
    u[1]
    u[2]
    u[3]
];
u0 = [0.7, 6.0, 0.0];
p = [0.2, 2];
t0 = 0.0;
tf = 35.0;
ts = 1e-02;
opt = ADAM(1e-02);

env = init_env(f, O, u0, p, t0, tf, ts, opt, reltol = 1e-6, abstol = 1e-6);

tfs = [5.0, 10.0];
estp = randn(length(env.u0) + length(env.p)) * 1e-03;
hp = training_routine(env, tfs, estp, 500)

################################################################################

f(u, p, t) = [
    u[2];
    u[3];
    p[2] * atan(u[3] + u[1]) - u[3] + (p[1] - 1) * u[2] - u[1];
];
O(u, p, t) = [
    u[1]
    u[2]
    u[3]
];
u0 = [0.7, 6.0, 0.0];
p = [0.2, 2];
t0 = 0.0;
tf = 35.0;
ts = 1e-02;
opt = ADAM(1e-02);

prob = ODEProblem(f, u0, (t0, tf), saveat = range(t0, tf,
    length = Int(round((tf - t0) / ts))));
sol = solve(prob, p = p, abstol = 1e-06, reltol = 1e-06);
y_samples = sol[1, :];

env = init_env(f, O, y_samples, t0, tf, ts, opt, reltol = 1e-6, abstol = 1e-6);

epsilon = [1e-02];
data1 = estimate_time_derivatives(env, UnivEst.CLASSICALHGO, epsilon, 2,
    Ts = 1e-04);
