# Duffing oscillator (Equation (2.6)) in "Sprott, Julien C. Elegant chaos:
# algebraically simple chaotic flows. World Scientific, 2010."

phi(u, p, t) =
    - p[1] * u[2] - p[2] * u[1] - p[3] * u[1]^3 + p[4] * sin(p[5] * t);
u0 = [-0.8969, 3.9593];
p = [0.99, -0.9989, 1.003, -0.9925, -0.8003];

duff = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);

d(t) = 0.1 * sin(100 * t);
ics = [
    -0.8969 3.9593
    -0.9100 4.0500
    -0.9000 4.0000
];

# Gain tuning
#estp = gain_training(duff, 10.0, 1000, d, ics, hgo_type = UnivEst.MIN_CASCADE,
#    S = [10.0, 10.0, 10.0], gain_type = UnivEst.DECREASING_GAIN,
#    callback = true);
estp = [57.910828695645705, 41.83779488117897, 0.961574968646006]

gain_plot(estp, duff.t0, duff.ts, duff.tf,
    gain_type = UnivEst.DECREASING_GAIN)
gvals = gain_plot(estp, duff.t0, duff.ts, duff.tf,
    gain_type = UnivEst.DECREASING_GAIN, get_vals = 1);

x, hx = test_timevarying_hgo(duff, estp, d,
    gain_type = UnivEst.DECREASING_GAIN, hgo_type = UnivEst.MIN_CASCADE,
    S = [10.0, 10.0, 10.0]);
plot(x)
plot!(x - hx)

writedlm("duff_x.csv", x, ",")
writedlm("duff_hx.csv", hx, ",")
writedlm("duff_gain.csv", gvals, ",")
