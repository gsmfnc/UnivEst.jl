# Duffing oscillator (Equation (2.6)) in "Sprott, Julien C. Elegant chaos:
# algebraically simple chaotic flows. World Scientific, 2010."

phi(u, p, t) =
    - p[1] * u[2] - p[2] * u[1] - p[3] * u[1]^3 + p[4] * sin(p[5] * t);
u0 = [-0.8969, 3.9593];
p = [0.99, -0.9989, 1.003, -0.9925, -0.8003];

duff = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);

d(t) = 0.1 * sin(10 * t);
ics = [
    -0.8969 3.9593
    -0.9100 4.0500
    -0.9000 4.0000
];

# time-varying gain for classical hgo
estp = gain_training(duff, 30.0, 1000, d, ics, callback = true);

x, hx = test_timevarying_hgo(duff, estp, d,
    gain_type = UnivEst.TIMEVARYING_GAIN, hgo_type = UnivEst.CLASSICALHGO);

plot(x)
plot!(hx)

# Time-varying gain for minimum-order cascade

estp = gain_training(duff, 10.0, 1000, d, ics, hgo_type = UnivEst.MIN_CASCADE,
    S = [10.0, 10.0], gain_type = UnivEst.DECREASING_GAIN, callback = true);

x, hx = test_timevarying_hgo(duff, estp, d,
    gain_type = UnivEst.DECREASING_GAIN, hgo_type = UnivEst.MIN_CASCADE,
    S = [10.0, 10.0]);

plot(x)
plot!(hx)
