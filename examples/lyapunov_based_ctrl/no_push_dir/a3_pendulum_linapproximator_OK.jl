# Define system
n = 2;

g = 9.81;
l = 1;
a = g / l;

m = 1;
k = 0.75;
b = k / m;

u(x, p) = [
    0
    p[1] * x[1] + p[2] * x[2] + p[3] * x[1]^2 + p[4] * x[2]^2 +
        p[5] * x[1] * x[2]
];
B = [
    0.
    1 / (m * l^2)
];
dyn_noctrl(x, p) = [
    x[2]
    - a * sin(x[1]) - b * x[2]
];
f(x, p, t) = dyn_noctrl(x, p) + b .* u(x, p);
sys_to_ctrl = init_controlled_system(f, u);

# generate test points
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
estp0 = randn(5);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, P,
    alpha = 5e00);

# alpha = 5e00
# estp = [
#  -7.683930455453602
# -10.197307183354546
#   0.5843170641150458
#   0.9665505387435173
#   1.5247570082645903
# ]

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp, P);

f(x, p, t) = dyn_noctrl(x, p) + b .* u(x .- [1.0, 0.0], p) +
    b .* [0., b^-1 * a * sin(1)];
sys_to_ctrl = init_controlled_system(f, u);

u0 = vec(randn(n, 1)) * 2;
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);

p1 = plot_lyapunov_derivative_values_2d(test_points, dV_values);
p2 = plot(t_sol, sol');
p3 = plot(t_sol, u_vals');
plot(p1, p2, p3, layout = (3, 1))

GC.gc()
