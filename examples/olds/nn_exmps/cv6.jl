phi(u, p, t) = p[2] * atan(u[3] + u[1]) - u[3] + (p[1] - 1) * u[2] - u[1];
u0 = [0.7, 6.0, 0.0];
p = [0.2, 2];
t0 = 0.0;
tf = 50.0;
ts = 1e-01;
opt = ADAM(1e-02);
env = init_env(phi, u0, p, t0, tf, ts, opt);

epsilon = [1e-02];
data = estimate_time_derivatives(env, UnivEst.CLASSICALHGO, epsilon, 2,
    Ts = 1e-03);

N = 3;
n = 3;
l = N + N * n;

sigma(x) = tanh(x) + 1;
W(p) = reshape(p[1:N], 1, :);
V(p) = reshape(p[N + 1:end], :, n);
hphi(u, p, t) = W(p) * sigma.(V(p) * u);

function add_loss(env, u, p, u0)
    alpha = 0.1;
    beta = 0.1;

    return alpha * maximum(sum(abs, W(p), dims = 1)) +
        beta * maximum(sum(abs, V(p), dims = 1));
end

env = set_env_parameter(env, "phi", hphi);
env = set_env_parameter(env, "data", data);
env = set_env_parameter(env, "M", 3);
env = set_env_parameter(env, "d_time", 1.0);
env = set_env_parameter(env, "tolerances", [1e-6, 1e-6]);

estp = randn(n + l);
tfs = [10.0, 20.0];
env = set_env_parameter(env, "add_loss", add_loss);
hp = training_routine(env, tfs, estp, 1000)
