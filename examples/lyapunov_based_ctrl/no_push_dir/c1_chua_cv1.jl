# Define system
n = 3;

alpha = 0.3;
u(x, p) = [
    0.
    0.
    p[1] * x[1] + p[2] * x[2] + p[3] * x[3] +
        p[4] * x[1]^2 + p[5] * x[2]^2 + p[6] * x[3]^2 +
        p[7] * x[1] * x[2] + p[8] * x[1] * x[3] + p[9] * x[2] * x[3] +
        p[10] * x[1]^3 + p[11] * x[2]^3 + p[12] * x[3]^3 +
        p[13] * x[1]^2 * x[2] + p[14] * x[1] * x[2]^2 +
        p[15] * x[1]^2 * x[3] + p[16] * x[1] * x[3]^2 +
        p[17] * x[2]^2 * x[3] + p[18] * x[2] * x[3]^2
];
dyn_noctrl(x, p) = [
    x[2]
    x[3]
    (1 - 3 * x[1]^2) * x[3] - 6 * x[1] * x[2]^2 + (alpha - 1) * x[2] -
        x[1]^3 + x[1]
];
f(x, p, t) = dyn_noctrl(x, p) + alpha .* u(x, p);

sys_to_ctrl = init_controlled_system(f, u);

# Generate test points
test_points = zeros(n, 500);
radius = 0.1;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.1;
    end
end

using MatrixEquations
A = [
    0.  1.  0.
    0.  0.  1.
    -2. -2. -2.
];
f_des(x, p, t) = A * x;
Q = [
    1.  0.  0.
    0.  1.  0.
    0.  0.  1.
];
P = lyapc(A', Q, Q);

estp0 = randn(18);
estp = ctrl_training(sys_to_ctrl, f_des, estp0, test_points, 1000, P,
    alpha = 1e02);
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp, P);
plot(dV_values)

# alpha = 5e00
estp = [-9.999677998722706, -4.333341526563112, -10.000003991731074,
    0.0007042243212405241, 0.00013775249481807615, -0.00026130996738419357,
    0.0006919484720856878, 0.0006503061854191607, 3.3376288864806966e-5,
    3.3321880153305177, 5.116040161138338e-5, 0.00020581705195875052,
    -8.90613741655838e-5, 20.00040719708594, 10.001598221239979,
    0.0001811313879405876, -0.0001979193632982055, -0.0007873616222140173]

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp, P);
u0 = vec(randn(n, 1));
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 50.0);

p1 = plot_lyapunov_derivative_values_2d(test_points, dV_values);
p2 = plot(t_sol, sol');
p3 = plot(t_sol, u_vals');
plot(p1, p2, p3, layout = (3, 1))

GC.gc()
