# Define system
n = 2;
m = 2;

u(x, p) = [
    0
    p[1] * x[1] + p[2] * x[2] + p[3] * x[1]^2 + p[4] * x[2]^2 +
        p[5] * x[1] * x[2]
];
dyn_noctrl(x, p) = [
    - x[1] - x[2]
    2 * x[1] - x[2]^3
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p);

sys_to_ctrl = init_controlled_system(f, u);

# Generate test points
test_points = zeros(n, 2000);
radius = 0.1;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.1;
    end
end

estp0 = randn(5);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000,
    alpha = 5e00);

# alpha = 5e00
estp = [-1.11242, -2.89606, 0.470396, -0.687578, 0.401577];

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp);
u0 = vec(randn(n, 1) * 2);
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 100.0);

p1 = plot_lyapunov_derivative_values_2d(test_points, dV_values);
p2 = plot(t_sol, sol');
p3 = plot(t_sol, u_vals');
plot(p1, p2, p3, layout = (3, 1))

# Generate (less) test points
quiv_points = zeros(n, 5000);
radius = 0.1;
for i = 1:1:size(quiv_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    quiv_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.1;
    end
end

quivplot = plot_vector_fields(sys_to_ctrl.f, quiv_points, estp);
plot(quivplot, ylim = (-3, 3), xlim = (-3, 3))
quivplot

GC.gc()
