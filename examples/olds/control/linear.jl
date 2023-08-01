n = 2;
N = 3;

random_data = zeros(n, 50000);
radius = 0.01;
for i = 1:1:size(random_data, 2)
    tmp = randn(n);
    nz = 1 / sqrt(sum(tmp.^2));
    random_data[:, i] = tmp * nz * radius;
    if i % 100 == 0
        radius = radius + 0.01;
    end
end

sigma(x) = tanh(x);
W(p) = reshape(p[1:N], 1, :);
V(p) = reshape(p[N + 1:end], :, n);

u(x, p) = (W(p) * sigma.(V(p) * x))[1];
f(x, p) = [x[2], - x[1] + u(x, p)];

alpha = 1e-01;
beta = 1e01;
gamma = 1e-01;

#xs = -20:0.1:20;
#plot(xs, tanh.(alpha * (xs .+ beta)))

env = init_ctrl_env(alpha, beta, gamma, random_data, f, u);

p0 = randn(N + N * n);
estp = ctrl_training_routine(env, p0)

dV_values = get_lyapunov_derivative_values(env, random_data, estp);
u0 = vec(randn(n, 1));
tf = 100.0;
t_sol, sol, u_vals = get_ctrl_sol(env, estp, u0, tf);

using Plots
p1 = plot(dV_values);
p2 = plot(t_sol, sol');
p3 = plot(t_sol, u_vals');
plot(p1, p2, p3, layout = (3, 1))
