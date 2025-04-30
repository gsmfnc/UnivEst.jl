# Define system
n = 3;

j1 = 0.5;
j2 = 0.3;
j3 = 0.6;

u(x, p) = [
    p[1] * x[1] + p[2] * x[2] + p[3] * x[3]
    p[4] * x[1] + p[5] * x[2] + p[6] * x[3]
    p[7] * x[1] + p[8] * x[2] + p[9] * x[3]
];
dyn_noctrl(x, p) = [
    1 / j1 * (j2 - j3) * x[2] * x[3]
    1 / j2 * (j3 - j1) * x[3] * x[1]
    1 / j3 * (j1 - j2) * x[1] * x[2]
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

estp0 = randn(9);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000,
    alpha = 5e00);

# alpha = 5e00
estp = [
    -0.9170943765894541
    -0.17465152234210488
     0.8827454399604804
     0.6154340114620048
    -1.3878150379142271
     1.1069502729185767
    -0.3542491244622418
    -1.0738260338201853
    -2.8147757366513018
];

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp);
p0 = -[1., 0., 0., 0., 1., 0., 0., 0., 1.];
dV_values_std = get_lyapunov_derivative_values(sys_to_ctrl, test_points, p0)

plot(dV_values)
plot!(dV_values_std)

u0 = vec(randn(n, 1) * 2);
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);
t_sol_std, sol_std, u_vals_std =
    get_controlled_sys_solution(sys_to_ctrl, p0, u0, tf = 10.0);

p1 = plot(t_sol_std, sol_std');
p2 = plot(t_sol, sol');
p3 = plot(t_sol_std, u_vals_std');
p4 = plot(t_sol, u_vals');
plot(p1, p2, p3, p4, layout = (2, 2))

GC.gc()

# Robustness test
j1 = 0.5;
j2 = 0.3;
j3 = 0.6;

u(x, p) = [
    p[1] * x[1] + p[2] * x[2] + p[3] * x[3]
    p[4] * x[1] + p[5] * x[2] + p[6] * x[3]
    p[7] * x[1] + p[8] * x[2] + p[9] * x[3]
];
dyn_noctrl(x, p) = [
    1 / j1 * (j2 - j3) * x[2] * x[3]
    1 / j2 * (j3 - j1) * x[3] * x[1]
    1 / j3 * (j1 - j2) * x[1] * x[2]
];
d(x, t) = [
    0.5 * (x[1]^2 + x[2]^2 + x[3]^2)
    0.5 * (x[1]^2 + x[2]^2 + x[3]^2)
    0.5 * (x[1]^2 + x[2]^2 + x[3]^2)
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p) + d(x, t);

sys_to_ctrl = init_controlled_system(f, u);

t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);
t_sol_std, sol_std, u_vals_std =
    get_controlled_sys_solution(sys_to_ctrl, p0, u0, tf = 2.1);

p1 = plot(t_sol_std, sol_std');
p2 = plot(t_sol, sol');
p3 = plot(t_sol_std, u_vals_std');
p4 = plot(t_sol, u_vals');
plot(p1, p2, p3, p4, layout = (2, 2))
