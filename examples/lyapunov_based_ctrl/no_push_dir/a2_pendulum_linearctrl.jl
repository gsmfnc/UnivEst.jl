# Define system
n = 2;
m = 2;

a = 0.6;
b = 0.15;

# define pendulum dynamics with x[1] substituted by x[1] - 1.0 (to stabilize
# [1.0, 0.0] rather than [0.0, 0.0] that is already stable)
u(x, p) = [
    0
    p[1] * x[1] + p[2] * x[2] + 0.0 * p[3]
];
dyn_noctrl(x, p) = [
    x[2]
    - a * sin(x[1]) - b * x[2]
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
        radius = radius + 0.06;
    end
end

P = [
    1.  1.
    0.  1.
];
estp0 = randn(3);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 300, P,
    alpha = 5e00);

# alpha = 5e00
estp = [-0.2793645816939249, -1.6260614697831375, 0.0]

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp);

# define actual pendulum dynamics
u(x, p) = [
    0
    p[1] * (x[1] - 1.0) + p[2] * x[2] + p[3]
];
dyn_noctrl(x, p) = [
    x[2]
    - a * sin(x[1]) - b * x[2]
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p);

u0 = vec(randn(n, 1) * pi);
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 50.0);

p1 = plot_lyapunov_derivative_values_2d(test_points, dV_values);
p2 = plot(t_sol, sol');
p3 = plot(t_sol, u_vals');
plot(p1, p2, p3, layout = (3, 1))

GC.gc()
