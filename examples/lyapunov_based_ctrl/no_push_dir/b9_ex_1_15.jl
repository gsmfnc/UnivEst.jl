# Define system
n = 4;

m = 0.5; # mass of the pendulum
M = 2; # mass of the cart
L = 0.5; # distance from center of gravity to pivot
I = 0.015; # inertia of pendulum wrt center of gravity
k = 0.7; # friction coefficient
g = 9.81;

n_nodes = 2;
indx_nn = n_nodes + n_nodes * n + n_nodes + n;
function single_hidden_layer_bias(x, p, n)
    return p[1:n_nodes]' * (tanh.(
        reshape(p[(n_nodes + 1):(n_nodes + n_nodes * n)], n_nodes, n) * x +
        p[(n_nodes + n_nodes * n + 1):(n_nodes + n_nodes * n + n_nodes)])) +
        p[(n_nodes + n_nodes * n + n_nodes + 1):end]' * x;
end

indx_nn = 10;
function monomials(x, p)
    return p[1:4]' * x + p[5:7]' * (x[1] .* x[2:4]) +
        p[8:9] * (x[2] .* x[3:4]) + p[10] x[3] * x[4]
end

u(x, p) = [
    0.
    single_hidden_layer_bias(x, p, n)
    0.
    single_hidden_layer_bias(x, p, n)
];
g_u(x) = [
    0.
    - 1 / Delta(x[1]) * m * L * cos(x[1]) * (abs(x[1]) < pi/2)
    0.
    1 / Delta(x[1]) * (I + m * L^2)
];
Delta(x) = (I + m * L^2) * (m + M) - m^2 * L^2 * cos(x)^2;
dyn_noctrl(x) = [
    x[2] * (abs(x[1]) < pi/2)        # angular rotation of pendulum
    1 / Delta(x[1]) * ((m + M) * m * g * L * sin(x[1]) - m * L * cos(x[1]) *
        (m * L * x[2]^2 * sin(x[1]) - k * x[4])) * (abs(x[1]) < pi/2)
    x[4]        # displacement of the pivot
    1 / Delta(x[1]) * (- m^2 * L^2 * g * sin(x[1]) * cos(x[1]) +
        (I + m * L^2) * (m * L * x[2]^2 * sin(x[1]) - k * x[4]))
];
f(x, p, t) = dyn_noctrl(x) + g_u(x) .* u(x, p);

sys_to_ctrl = init_controlled_system(f, u);

# Generate test points
test_points = zeros(n, 1000);
radius = 0.01;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 10 == 0
        radius = radius + 0.01;
    end
end
mults = [pi / 4, pi / 12, 5., 1.];
test_points[1, :] = test_points[1, :] * mults[1];
test_points[2, :] = test_points[2, :] * mults[2];
test_points[3, :] = test_points[3, :] * mults[3];
test_points[4, :] = test_points[4, :] * mults[4];

P = [
    1/mults[1]  1/mults[1]  0.          0.
    0.          1/mults[2]  0.          0.
    0.          0.          1/mults[3]  1/mults[3]
    0.          0.          0.          1/mults[4]
];
estp0 = randn(indx_nn);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 300, P,
    alpha = 5e00);
estp = ctrl_training(sys_to_ctrl, estp, test_points, 300, P,
    alpha = 5e00);

# alpha = 5e00

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp, P);
plot(dV_values)

u0 = vec(randn(n, 1));
u0 = u0 ./ maximum(abs.(u0)) ./ 2 .* mults .* 0.1;
u0 = [pi/32, pi/1000, 0., -1.475];
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);

p1 = plot(t_sol, sol[1, :]);
p2 = plot(t_sol, sol[2, :]);
p3 = plot(t_sol, sol[3, :]);
p4 = plot(t_sol, sol[4, :]);
plot(p1, p2, p3, p4, layout = (4, 1))
sol

GC.gc()
