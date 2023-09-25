# Duffing oscillator (Equation (2.6)) in "Sprott, Julien C. Elegant chaos:
# algebraically simple chaotic flows. World Scientific, 2010."

N = 16;
n = 2;
sigma(x) = tanh(x) + 1;
W(p) = reshape(p[1:N], 1, :);
V(p) = reshape(p[N + 1:(N + N * n)], :, n);
lin(p) = p[(N + N * n + 1):end]';
phi(u, p, t) = (W(p) * sigma.(V(p) * u))[1] + lin(p) * u + sin(t);
fq1 = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 10.0, ts = 1e-02);

#x, y = get_sys_solution(fq1, u0_arg = [0.0, 0.4]);
#plot(x')
#x, y = get_sys_solution(fq1, u0_arg = [-0.4, 0.2]);
#plot(x')

d(t) = 0.1 * sin(100 * t);
ics = [
    0.18813 -0.78586
    0.0 0.4
    -0.4 0.2
];

# time-varying gain for classical hgo
#estp = gain_training(fq1, 10.0, 1000, d, ics, callback = true);
estp = [63.53, -0.00, 54.26, 0.87];

gain_plot(estp, fq1.t0, fq1.ts, fq1.tf)

x, hx = test_timevarying_hgo(fq1, estp, d,
    gain_type = UnivEst.TIMEVARYING_GAIN, hgo_type = UnivEst.CLASSICALHGO);

plot(x)
plot!(hx)
