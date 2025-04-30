# Define system
n = 6;

g = 9.81;
m = 1;
J = 0.01;

rt1(x, p) = p[1:4]' * [x[1] + x[3], x[2] + x[4], x[5], x[6]] + p[5];
rt2(x, p) = p[6:9]' * [x[1] + x[3], x[2] + x[4], x[5], x[6]] + p[10];
u(x, p) = [
    0
    (rt1(x, p) + rt2(x, p)) / m * sind(x[3])
    0
    (rt1(x, p) + rt2(x, p)) / m * cosd(x[3])
    0
    (rt2(x, p) - rt1(x, p)) / J
];
dyn_noctrl(x, p) = [
    x[2]        # x position
    0.0         # x velocity
    x[4]        # z position
    - g         # z velocity
    x[6]        # pitch angle
    0.0         # pitch rate
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p);

sys_to_ctrl = init_controlled_system(f, u);

# Generate test points
test_points = zeros(n, 2000);
radius = 0.001;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 10 == 0
        radius = radius + 0.005;
    end
end

estp0 = randn(10);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000,
    alpha = 5e00);

# alpha = 5e00
estp = [
]:

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp)
plot(dV_values)

u0 = vec(randn(n, 1) * 0.01);
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);

rt_vals = zeros(2, size(sol, 2));
for i = 1:1:size(sol, 2)
    rt_vals[1, i] = rt1(sol[:, i], estp);
    rt_vals[2, i] = rt2(sol[:, i], estp);
end

p1 = plot(t_sol, sol[1, :], label = ["x"]);
p2 = plot(t_sol, sol[3, :], label = ["z"]);
p3 = plot(t_sol, sol[5, :], label = ["pitch"]);
p4 = plot(t_sol, sol[2, :], label = ["x velocity"]);
p5 = plot(t_sol, sol[4, :], label = ["z velocity"]);
p6 = plot(t_sol, sol[6, :], label = ["pitch rate"]);
#p7 = plot(t_sol, rt_vals');
plot(p1, p4, p2, p5, p3, p6, layout = (3, 2))

GC.gc()
