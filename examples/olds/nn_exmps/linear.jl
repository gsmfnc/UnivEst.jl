phi(u, p, t) = - 3.45 * u[1];
u0 = [-2.0, 0.0];
p = [0.0];
t0 = 0.0;
tf = 50.0;
ts = 1e-02;
opt = ADAM(1e-02);
env = init_env(phi, u0, p, t0, tf, ts, opt);

epsilon = [1e-02];
data = estimate_time_derivatives(env, UnivEst.CLASSICALHGO, epsilon, 1,
    Ts = 1e-04);

N = 3;
n = 2;
l = N + N * n;

sigma(x) = tanh(x);
W(p) = p[1:N];
V(p) = reshape(p[N + 1:end], :, n);
hphi(u, p, t) = W(p)' * sigma.(V(p) * u);

function add_loss(env, u, p, u0)
    alpha = 1e-01;
    beta = 1e-01;
    gamma = 2.0;

    lyap_term = 0;
    for i = 1:1:size(u, 2)
        du = [u[2, i], W(p)' * sigma.(V(p) * u[:, i])];
        lyap_term = lyap_term + gamma * (tanh(u[:, i]' * du) + 1);
    end

    return alpha * maximum(sum(abs, W(p), dims = 1)) +
        beta * maximum(sum(abs, V(p), dims = 1)) + lyap_term;
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
env = set_env_parameter(env, "add_loss", add_loss2);

tfs = [10.0];
#estp = randn(n + l);
hp = training_routine(env, tfs, estp, 1000)

tmpnn = 5000;
est_sol = get_sys_solution(env, hp[(n + 1):end], hp[1:n]);
times = env.t0:env.ts:(env.tf - env.ts);
yy = 5;
using Plots
p1 = plot(times, data[1, 1:tmpnn]);
p1 = plot!(times, est_sol[1, :], ylim = (-yy, yy));
p2 = plot(times, data[2, 1:tmpnn]);
p2 = plot!(times, est_sol[2, :], ylim = (-yy, yy));
plot(p1, p2, layout = (2, 1))

t0 = 0.0;
tf = 100.0;
ts = 1e-02;
epsilon = 1e-02;
#hu0 = randn(3);
est_phi(u, p, t) = hphi(u, p, t);
x, hx = test_hgo(epsilon, phi, p, est_phi, hp[(n + 1):end], u0, hp[1:n], t0, tf,
    ts);

hphi2(u, p, t) = 0;
x, hx2 = test_hgo(epsilon, phi, p, hphi2, [0.0], u0, hp[1:n], t0, tf, ts);

using Plots
times = t0:ts:(tf - ts);
yy = 0.04;
p1 = plot(times, (x .- hx)', ylim = (-yy, yy))
p2 = plot(times, (x .- hx2)', ylim = (-yy, yy))
plot(p1, p2, layout = (2, 1))

function test_add_loss(u, p, u0)
    gamma = 1.4;

    lyap_term = 0;
    arr = zeros(1, size(u, 2));
    for i = 1:1:size(u, 2)
        du = [u[2, i], u[3, i], W(p)' * sigma.(V(p) * u[:, i])];
        lyap_term = lyap_term + gamma * (tanh(u[:, i]' * du) + 1);

        arr[i] = gamma * (tanh(u[:, i]' * du) + 1);
    end

    return lyap_term, arr;
end
