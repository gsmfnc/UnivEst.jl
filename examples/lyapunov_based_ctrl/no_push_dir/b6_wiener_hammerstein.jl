# Define system
n = 2;
m = 2;

G1_zpk = DSP.Chebyshev1(3, 0.5);
mlt1 = 100;
m = length(G1_zpk.p) - length(G1_zpk.z);
G1 = ControlSystems.zpk(G1_zpk.z * mlt1, G1_zpk.p * mlt1, G1_zpk.k * mlt1^m);

G2_zpk = DSP.Chebyshev2(3, 40);
mlt2 = 100;
m = length(G2_zpk.p) - length(G2_zpk.z);
G2 = ControlSystems.zpk(G2_zpk.z * mlt2, G2_zpk.p * mlt2, G2_zpk.k * mlt2^m);

ControlSystems.bodeplot(G1, hz = true)
ControlSystems.bodeplot!(G2, hz = true)

a = 0.6;
b = 0.15;

u(x, p) = [
    0
    p[1] * x[1] + p[2] * x[2] + p[3] * x[1]^2 + p[4] * x[2]^2 +
        p[5] * x[1] * x[2]
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

estp0 = randn(5);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000,
    alpha = 8e00);

# alpha = 8e00
estp = [-1.02320437046919, -4.164354252391252, -0.003418812921382694,
    -0.02075270556600821, -0.01264028928028057]

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
