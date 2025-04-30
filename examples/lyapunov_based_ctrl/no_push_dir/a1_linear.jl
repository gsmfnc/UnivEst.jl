# Define system
n = 2;
m = 2;

T = [
    -0.168  0.206
    -0.647  0.189
];
A = [
    0   1
    -1  1
];
Abar = T * A * inv(T);

u(x, p) = [
    p[1] * x[1] + p[2] * x[2]
    p[3] * x[1] + p[4] * x[2]
];
dyn_noctrl(x, p) = [
    -0.141318 * x[1] + 0.355087 * x[2]
    -3.270430 * x[1] + 1.141320 * x[2]
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p);

sys_to_ctrl = init_controlled_system(f, u);

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

estp0 = randn(4);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000,
    alpha = 5e00);

# alpha = 5e00
estp = [-2.9615794156909447, 1.7373645835878124, 0.8886307514764181,
    -4.081719130769423]

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp);
u0 = vec(randn(n, 1));
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 50.0);

p1 = plot_lyapunov_derivative_values_2d(test_points, dV_values);
p2 = plot(t_sol, sol');
p3 = plot(t_sol, u_vals');
plot(p1, p2, p3, layout = (3, 1))

GC.gc()
