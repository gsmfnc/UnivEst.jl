# Define system
n = 2;

u(x, p) = [
    0.0
    p[1] * x[1] + p[2] * x[2]
];
dyn_noctrl(x, p) = [
    x[2]
    - x[1]
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

P = [
    1.  1.
    0.  1.
];
estp0 = vec([1.7373401895859977 0.8599224071037785]);
using LinearAlgebra

######################### alpha = 0.1 ##########################################
# estp = ctrl_training(sys_to_ctrl, estp0, test_points, 5000, P, alpha = 0.1);
# estp = ctrl_training(sys_to_ctrl, estp, test_points, 1000, P, alpha = 0.1);
# estp = ctrl_training(sys_to_ctrl, estp, test_points, 1000, P, alpha = 0.1);
estp = [-33.31340373963575, -33.44453446034039];

A = [
    0           1
    -1+estp[1]  estp[2]
];
eigen(A)

dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp);
plot_lyapunov_derivative_values_2d(test_points, dV_values)

u0 = vec(randn(n, 1));
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);

p1 = plot(t_sol, sol[1, :]);
p2 = plot(t_sol, sol[2, :]);
plot(p1, p2, layout = (2, 1))

################################################################################

#estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, P, alpha = 0.1);
estp = [-7.258004376465809, -8.192495734110116];

A = [
    0           1
    -1+estp[1]  estp[2]
];
eigen(A)

#estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, P, alpha = 1.0);
estp = [-3.415357786862305, -4.381947045644152];

A = [
    0           1
    -1+estp[1]  estp[2]
];
eigen(A)

#estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, P, alpha = 10.0);
estp = [-1.0514997962522863, -2.2625760751956427];

A = [
    0           1
    -1+estp[1]  estp[2]
];
eigen(A)

u0 = vec(randn(n, 1));
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 10.0);

p1 = plot(t_sol, sol[1, :]);
p2 = plot(t_sol, sol[2, :]);
pnew = plot(p1, p2, layout = (2, 1))
sol

################################################################################

using ControlSystems
A = [
    0   1
    -1  0
];
B = [
    0
    1
];
C = [
    1   0
    0   1
];
D = [
    0
    0
];

sys1 = ss(A, B, C, D);

estp = [-1.0514997962522863, -2.2625760751956427];
K1 = [estp[1], estp[2]]';
fdbk1 = ss(0, zeros(1, 2), 0, K1);
L1 = series(sys1, fdbk1);

estp = [-7.258004376465809, -8.192495734110116];
K2 = [estp[1], estp[2]]';
fdbk2 = ss(0, zeros(1, 2), 0, K2);
L2 = series(sys1, fdbk2);

plotlyjs()
nyquistplot(tf(L1))
nyquistplot!(tf(L2))
