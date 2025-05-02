include("../../src/UnivEst.jl")

################# SYSTEM DEFINITION
# state dimension
n = 2;

# parametric controller
u(x, p) = [
    0.0
    p[1] * x[1] + p[2] * x[2]
];

# system dynamics
dyn_noctrl(x, p) = [
    x[2]
    - x[1]
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p);

# initialize controlled system
sys_to_ctrl = init_controlled_system(f, u);

################# CONTROLLER DESIGN
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

# random initial guess
estp0 = vec([1.73 0.85]);

# define P for Lyapunov candidate of the form V=x'Px
P = [
    1.  1.
    0.  1.
];

# training with various alphas
#estp1 = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, P, alpha = 1.0,
#    gamma = 0.1);
#estp2 = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, P, alpha = 5.0,
#    gamma = 0.1);
#estp3 = ctrl_training(sys_to_ctrl, estp0, test_points, 1000, P, alpha = 10.0,
#    gamma = 0.1);

estp1 = [-2.1694528537189743, -2.856874834280935];
estp2 = [-1.0270646179820149, -1.6449328404020613];
estp3 = [-0.709319971895083, -1.261144824313184];

################# BODE PLOTS
# check robustness by computing open loop function and looking at bode plots
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
base_sys = ss(A, B, C, D);

K1 = [estp1[1], estp1[2]]';
fdbk1 = ss(0, zeros(1, 2), 0, K1);
L1 = series(base_sys, fdbk1);

K2 = [estp2[1], estp2[2]]';
fdbk2 = ss(0, zeros(1, 2), 0, K2);
L2 = series(base_sys, fdbk2);

K3 = [estp3[1], estp3[2]]';
fdbk3 = ss(0, zeros(1, 2), 0, K3);
L3 = series(base_sys, fdbk3);

using LaTeXStrings
ws = 1e-02:5e-03:100;
bodeplot(tf(L1), ws, label=L"\alpha=1")
bodeplot!(tf(L2), ws, label=L"\alpha=5")
bodeplot!(tf(L3), ws, label=L"\alpha=10", title="")

margin(minreal(tf(L1)), vec(ws))
margin(minreal(tf(L2)), vec(ws))
margin(minreal(tf(L3)), vec(ws))

################# TIME RESPONSE
# closed-loop eigenvalues
using LinearAlgebra
F1 = eigen(A + B * estp1');
F2 = eigen(A + B * estp2');
F3 = eigen(A + B * estp3');

scatter([F1.values[1].re F1.values[2].re], [F1.values[1].im F1.values[2].im],
    markercolor = [:black :black], label = [L"\alpha=1" ""])
scatter!([F2.values[1].re F2.values[2].re], [F2.values[1].im F2.values[2].im],
    markercolor = [:blue :blue], label = [L"\alpha=5" ""])
scatter!([F3.values[1].re F3.values[2].re], [F3.values[1].im F3.values[2].im],
    markercolor = [:green :green], label = [L"\alpha=10" ""],
    xlim = (-1.5, 0), ylim = (-1.2, 1.2),
    xlabel = "Real part", ylabel = "Imaginary part")

# re-definition of system with possible uncertainties
dist_dyn(x) = [
    -2.4 * x[1]^2 - 0.51 * x[2]^2 + 0.241 * x[1] * x[2]
    0.689 * x[1]^2 - 0.277 * x[2]^2 - 0.415 * x[1] * x[2]
];
unc_f(x, p, t) = dyn_noctrl(x, p) + p[3] * u(x, p) + p[4] * dist_dyn(x);
unc_sys_to_ctrl = init_controlled_system(unc_f, u);

######### "nominal" time response
u0 = vec([2.63, -1.05]);
amp1 = 1.;
amp2 = 0.;
estp = vec(vcat(estp1, [amp1, amp2]));
t_sol1, sol1, u_vals1 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 10.0);
estp = vec(vcat(estp2, [amp1, amp2]));
t_sol2, sol2, u_vals2 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 10.0);
estp = vec(vcat(estp3, [amp1, amp2]));
t_sol3, sol3, u_vals3 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 10.0);

p1 = plot(t_sol1, sol1', label = [L"x_1" L"x_2"], ylabel = L"\alpha = 1");
p2 = plot(t_sol2, sol2', label = [L"x_1" L"x_2"], ylabel = L"\alpha = 5");
p3 = plot(t_sol3, sol3', label = [L"x_1" L"x_2"], xlabel = L"t\ [s]",
    ylabel = L"\alpha = 10");
p4 = plot(t_sol1, u_vals1[2, :], label = L"u(x;p_{\star,1})");
p5 = plot(t_sol2, u_vals2[2, :], label = L"u(x;p_{\star,2})");
p6 = plot(t_sol3, u_vals3[2, :], label = L"u(x;p_{\star,3})",
    xlabel = L"t\ [s]");
plot(p1, p4, p2, p5, p3, p6, layout = (3, 2))

######### time response with halved B matrix
u0 = vec([2.63, -1.05]);
amp1 = 0.5;
amp2 = 0.;
estp = vec(vcat(estp1, [amp1, amp2]));
t_sol1, sol1, u_vals1 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 15.0);
estp = vec(vcat(estp2, [amp1, amp2]));
t_sol2, sol2, u_vals2 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 15.0);
estp = vec(vcat(estp3, [amp1, amp2]));
t_sol3, sol3, u_vals3 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 15.0);

p1 = plot(t_sol1, sol1', label = [L"x_1" L"x_2"], ylabel = L"\alpha = 1");
p2 = plot(t_sol2, sol2', label = [L"x_1" L"x_2"], ylabel = L"\alpha = 5");
p3 = plot(t_sol3, sol3', label = [L"x_1" L"x_2"], xlabel = L"t\ [s]",
    ylabel = L"\alpha = 10");
p4 = plot(t_sol1, u_vals1[2, :], label = L"u(x;p_{\star,1})");
p5 = plot(t_sol2, u_vals2[2, :], label = L"u(x;p_{\star,2})");
p6 = plot(t_sol3, u_vals3[2, :], label = L"u(x;p_{\star,3})",
    xlabel = L"t\ [s]");
plot(p1, p4, p2, p5, p3, p6, layout = (3, 2))

######### time response with parasitic dynamics
u0 = vec([2.63, -1.05]);
amp1 = 1.;
amp2 = 1.1;
estp = vec(vcat(estp1, [amp1, amp2]));
t_sol1, sol1, u_vals1 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 10.0);
estp = vec(vcat(estp2, [amp1, amp2]));
t_sol2, sol2, u_vals2 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 3.58);
estp = vec(vcat(estp3, [amp1, amp2]));
t_sol3, sol3, u_vals3 = get_controlled_sys_solution(unc_sys_to_ctrl, estp, u0,
    tf = 2.44);

p1 = plot(t_sol1, sol1', label = [L"x_1" L"x_2"], ylabel = L"\alpha = 1",
    xlim = (0, 10), ylim = (-2, 2));
p2 = plot(t_sol2, sol2', label = [L"x_1" L"x_2"], ylabel = L"\alpha = 5",
    xlim = (0, 10), ylim = (-2, 2));
p3 = plot(t_sol3, sol3', label = [L"x_1" L"x_2"], xlabel = L"t\ [s]",
    ylabel = L"\alpha = 10", xlim = (0, 10), ylim = (-2, 2));
p4 = plot(t_sol1, u_vals1[2, :], label = L"u(x;p_{\star,1})",
    xlim = (0, 10), ylim = (-2, 2));
p5 = plot(t_sol2, u_vals2[2, :], label = L"u(x;p_{\star,2})",
    xlim = (0, 10), ylim = (-2, 2));
p6 = plot(t_sol3, u_vals3[2, :], label = L"u(x;p_{\star,3})",
    xlabel = L"t\ [s]", xlim = (0, 10), ylim = (-2, 2));
plot(p1, p4, p2, p5, p3, p6, layout = (3, 2))
