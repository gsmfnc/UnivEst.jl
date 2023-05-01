phi(u, p, t) = - p[1] * sinh(u[1]) + sin(t);
u0 = [1.0, 0.0];
p = [0.2];
t0 = 0.0;
tf = 50.0;
ts = 1e-01;
opt = ADAM(1e-02);
env = init_env(phi, u0, p, t0, tf, ts, opt);

epsilon = [1e-02];
data = estimate_time_derivatives(env, UnivEst.CLASSICALHGO, epsilon, 2,
    Ts = 1e-03);

N = 2;
n = 2;
l = N + N * n;

sigma(x) = tanh(x);
W(p) = p[1:N];
V(p) = reshape(p[N + 1:end], :, n);
hphi(u, p, t) = W(p)' * sigma.(V(p) * u) + sin(t);

function add_loss(env, u, p, u0)
    alpha = 0.1;
    beta = 0.1;
    gamma = 1.0;
    theta = 0.0;

    lyap_term = 0;
    for i = 1:1:size(u, 2)
        du = [u[2, i], W(p)' * sigma.(V(p) * u[:, i])];
        lyap_term = lyap_term + gamma * (tanh(u[:, i]' * du) + 1);
    end

    return alpha * maximum(sum(abs, W(p), dims = 1)) +
        beta * maximum(sum(abs, V(p), dims = 1)) + lyap_term +
        theta * sum(abs, data[1:2, 10:size(u, 2)] .- u[:, 10:end]);
end

function add_loss2(env, u, p, u0)
    alpha = 0.1;
    beta = 0.1;

    return alpha * maximum(sum(abs, W(p), dims = 1)) +
        beta * maximum(sum(abs, V(p), dims = 1));
end

env = set_env_parameter(env, "phi", hphi);
env = set_env_parameter(env, "data", data);
env = set_env_parameter(env, "M", 2);
env = set_env_parameter(env, "d_time", 1.0);
env = set_env_parameter(env, "tolerances", [1e-6, 1e-6]);

estp = randn(n + l);

tfs = [5.0, 10.0];
env = set_env_parameter(env, "add_loss", add_loss2);
hp = training_routine(env, tfs, estp, 1000)

tfs = [5.0, 10.0];
env = set_env_parameter(env, "add_loss", add_loss);
hp2 = training_routine(env, tfs, estp, 1000)
