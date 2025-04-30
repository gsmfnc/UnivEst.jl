# Define system
n = 3;

# Generate test points
test_points = zeros(n, 2000);
radius = 0.01;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.01;
    end
end
test_points[3, :] = test_points[3, :] * pi;

# define system and control laws
function single_hidden_layer(x, p, n)
    return p[1:n_nodes]' * (tanh.(
        reshape(p[(n_nodes + 1):(n_nodes + n_nodes * n)], n_nodes, n) * x)) +
        p[(n_nodes + n_nodes * n + 1):end]' * x;
end
function single_hidden_layer_bias(x, p, n)
    return p[1:n_nodes]' * (tanh.(
        reshape(p[(n_nodes + 1):(n_nodes + n_nodes * n)], n_nodes, n) * x +
        p[(n_nodes + n_nodes * n + 1):(n_nodes + n_nodes * n + n_nodes)])) +
        p[(n_nodes + n_nodes * n + n_nodes + 1):end]' * x;
end

n_nodes = 3;
#v1(x, p) = single_hidden_layer_bias(x,
#    p[1:(n_nodes + n_nodes * n + n_nodes + n)], 3);
#v2(x, p) = single_hidden_layer(- x[3] + atan(0 - x[2], 0 - x[1]),
#    p[(n_nodes + n_nodes * n + n_nodes + n + 1):end], 1);
v1(x, p) = - norm(x[1:2]) * cos(x[3] - atan(x[2], x[1]));
v2(x, p) = - x[3] + atan(0 - x[2], 0 - x[1]);
u(x, p) = [
    v1(x, p) * cos(x[3])
    v1(x, p) * sin(x[3])
    v2(x, p)
];
dyn_noctrl(x, p) = [
    0
    0
    0
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p);

sys_to_ctrl = init_controlled_system(f, u);

#estp0 = randn(n_nodes * 2 + n_nodes * n * 2 + n_nodes + 2 * n);
estp0 = randn(n_nodes + n_nodes * n + n_nodes + n + n_nodes + n_nodes * 1 + 1);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, alpha = 1e-01);

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp)
plot_lyapunov_derivative_values_3d(test_points, dV_values)

u0 = vec(randn(n, 1) * 0.5);
u0[3] = u0[3] * pi;
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);

v_vals = zeros(2, size(sol, 2));
for i = 1:1:size(sol, 2)
    v_vals[1, i] = v1(sol[:, i], estp);
    v_vals[2, i] = v2(sol[:, i], estp);
end

p1 = plot(t_sol, sol[1, :], label = ["x"]);
p2 = plot(t_sol, sol[2, :], label = ["y"]);
p3 = plot(t_sol, rad2deg.(sol[3, :]), label = ["theta"]);
p4 = plot(t_sol, v_vals[1, :], label = ["v1 velocity"]);
plot!(t_sol, v_vals[2, :], label = ["v2 velocity"]);
plot(p1, p2, p3, p4, layout = (2, 2))

GC.gc()

analysis_points = zeros(n, 100);
radius = 0.1;
for i = 1:1:size(analysis_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    analysis_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.01;
    end
end
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, analysis_points, estp)
plot_lyapunov_derivative_values_3d(test_points, dV_values)
