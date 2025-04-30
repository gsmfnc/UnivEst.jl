# Define system
n = 3;

Rf = 281.3;
Lf = 156;
Ra = 2.581;
La = 0.0281;
J = 0.0221;
c1 = 1.25;
c2 = 0.516;
c3 = 0.002953;

n_nodes = 4;

function single_hidden_layer_bias(x, p, n)
    return p[1:n_nodes]' * (tanh.(
        reshape(p[(n_nodes + 1):(n_nodes + n_nodes * n)], n_nodes, n) * x +
        p[(n_nodes + n_nodes * n + 1):(n_nodes + n_nodes * n + n_nodes)])) +
        p[(n_nodes + n_nodes * n + n_nodes + 1):end]' * x;
end

# Va = 240
u(x, p) = [
    1 / Lf * ( -240. )
    1 / La * ( 240. )
    0.
];

indx_nn = n_nodes + n_nodes * n + n_nodes + n;
u(x, p) = [
    1 / Lf * (single_hidden_layer_bias(x, p[1:indx_nn], n) - 240.)
    1 / La * (single_hidden_layer_bias(x, p[(indx_nn + 1):end], n) + 240.)
    0.
];
dyn_noctrl(x, p) = [
    - Rf / Lf * x[1]                            #i_f
    - Ra / La * x[2] - c1 / La * x[1] * x[3]    #i_a
    - c3 / J * x[3] + c2 / J * x[1] * x[2]      #omega
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p);

sys_to_ctrl = init_controlled_system(f, u);

# u0 = vec(zeros(n, 1));
# p0 = vec(zeros(1, 1));
# t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, p0, u0,
#     tf = 10.0);
# p1 = plot(t_sol, sol[1, :]);
# p2 = plot(t_sol, sol[2, :]);
# p3 = plot(t_sol, sol[3, :]);
# plot(p1, p2, p3, layout = (3, 1))

# Generate test points
test_points = zeros(n, 3000);
radius = 0.0001;
for i = 1:1:1000
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 10 == 0
        radius = radius + 0.0001;
    end
end
radius = 0.1;
for i = 1001:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.01;
    end
end
mults = [1., 100., 300.];
test_points[1, :] = test_points[2, :] * mults[1];
test_points[2, :] = test_points[2, :] * mults[2];
test_points[3, :] = test_points[3, :] * mults[3];

P = [
    1 / mults[1]    0.              0.
    0.              1 / mults[2]    0.
    0.              0.              1 / mults[3]
];
estp0 = randn(indx_nn * 2);
estp = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, P,
    alpha = 5e00);

# alpha = 5e00
estp = [-4.4444064167518516, 5.807950501630677, 2.6834366210954204, 
-5.267532729654429, 5.294244238619143, -4.810947419396803, -5.5188249100007765,
9.093350130818653, 7.395651986536144, -5.404487019781638, -4.189674627586864,
9.84565061967658, 0.04283256451869488, -0.07562183160110314,
-0.0903050247338298, 0.04032285245828334, 0.05704941235846687,
-0.05836107913821883, -0.06187211055585157, 0.06393747449009224,
-1.6148593189377893, 0.03652312369888478, 0.2896386387080523,
-5.4403232058801585, 5.061330205141915, -4.656390755284868, 0.8457660740355236,
8.280179848664764, -7.1670121558251685, 4.2618189705959955, -2.475539736645273,
9.093990958636349, -6.500848500858857, 4.069147397278513, -1.3470735025331888,
0.040592325937708555, -0.0509359867923161, 0.0816497919201552,
0.9173591170974384, 0.06153174608141412, -0.05633173538838013,
0.0590099069748886, -0.5859564227284361, -0.8440436000201691,
-0.04260866181345608, 0.004068729281581627];

# Plot
dV_values = get_lyapunov_derivative_values(sys_to_ctrl, test_points, estp);
plot(dV_values)
plot_lyapunov_derivative_values_3d(test_points, dV_values)

u0 = vec(max.(-1, min.(randn(n, 1), 1)) .* mults);
t_sol, sol, u_vals = get_controlled_sys_solution(sys_to_ctrl, estp, u0,
    tf = 50.0);

p1 = plot(t_sol, sol[1, :]);
p2 = plot(t_sol, sol[2, :]);
p3 = plot(t_sol, sol[3, :]);
pufix = plot(p1, p2, p3, layout = (3, 1))
sol

GC.gc()
