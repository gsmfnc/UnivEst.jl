# System CS_1 from Table 6.4 in "Sprott, Julien C. Elegant chaos: algebraically
# simple chaotic flows. World Scientific, 2010."

phi(u, p, t) = - u[4] + p[1] * u[3] + p[2] * u[2] + p[3] * (u[1]^2 - 1);
u0 = [-0.9232, 0.0723, -0.0446, -0.2339];
p = [-5.1756, -2.6617, 4.4530];

cs1 = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);

#x, y = get_sys_solution(cs1, u0_arg = [-0.8, 0.2, 0.1, 0.0]);
#plot(x')
#x, y = get_sys_solution(cs1, u0_arg = [-1.2, 0.0, -1.0, 1.0]);
#plot(x')

d(t) = 0.1 * sin(100 * t);
ics = [
    -0.9232 0.0723 -0.0446 -0.2339
    -0.8 0.2 0.1 0.0
    -1.2 0.0 -1.0 1.0
];

# time-varying gain for classical hgo
#estp = gain_training(cs1, 10.0, 1000, d, ics, callback = true);
estp = [53.28, -0.028, 39.23, -0.53];

gain_plot(estp, cs1.t0, cs1.ts, cs1.tf)
gvals = gain_plot(estp, cs1.t0, cs1.ts, cs1.tf, get_vals = 1);

x, hx = test_timevarying_hgo(cs1, estp, d,
    gain_type = UnivEst.TIMEVARYING_GAIN, hgo_type = UnivEst.CLASSICALHGO);

plot(x)
plot!(hx)

writedlm("cs1_x.csv", x, ",")
writedlm("cs1_hx.csv", hx, ",")
writedlm("cs1_gain.csv", gvals, ",")
