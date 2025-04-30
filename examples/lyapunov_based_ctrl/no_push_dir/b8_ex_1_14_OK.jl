# Define system
n = 2;

M = 1;
theta = 30; # [deg]
k1 = 0.8;
k2 = 0.7;
k3 = 0.65;
g = 9.81;

n_nodes = 3;
indx_nn = n_nodes + n_nodes * n + n_nodes + n + 1;
function single_hidden_layer_bias(x, p, n)
    return p[1:n_nodes]' * (tanh.(
        reshape(p[(n_nodes + 1):(n_nodes + n_nodes * n)], n_nodes, n) * x +
        p[(n_nodes + n_nodes * n + 1):(n_nodes + n_nodes * n + n_nodes)])) +
        p[(n_nodes + n_nodes * n + n_nodes + 1):end]' * [x; 1.];
end

u(x, p) = [
    0.
    single_hidden_layer_bias(x, p, n)
];
dyn_noctrl(x, p) = [
    x[2]
    1 / M * (-k1 * sign(x[2]) - k2 * x[2] - k3 * x[2]^2) - g * sind(theta)
];
f(x, p, t) = dyn_noctrl(x, p) + 1 / M * u(x, p);

sys_to_ctrl = init_controlled_system(f, u);

# Generate test points
test_points = zeros(n, 3000);
radius = 0.01;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 10 == 0
        radius = radius + 0.01;
    end
end

P = [
    1.  1.
    0.  1.
];
estp0 = randn(indx_nn);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 300, P,
    alpha = 5e00);

# alpha = 5e00
estp = [
 -9.94729013736369
  9.414705000218389
 -8.745280609044809
  8.048568512138432
 -8.792885641345077
  8.01530440134019
  7.26988844005492
 -9.734892073302769
  7.267928062030757
 -0.0030185666949201603
  0.008163641696058556
 -0.0034045495248735285
 -6.549949608980645
 -7.931775036355846
  4.84363574082976
];

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp, P);
plot(dV_values)
plot_lyapunov_derivative_values_2d(test_points, dV_values)

u0 = vec(randn(n, 1));
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);

p1 = plot(t_sol, sol[1, :]);
p2 = plot(t_sol, sol[2, :]);
plot(p1, p2, layout = (2, 1))
sol

GC.gc()

# Generate quiv points
quiv_points = zeros(n, 300);
radius = 0.01;
for i = 1:1:size(quiv_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    quiv_points[:, i] = temp * nz * radius;
    if i % 1 == 0
        radius = radius + 0.01;
    end
end
quivplot = plot_vector_fields(sys_to_ctrl.f, quiv_points, estp);
quivplot
