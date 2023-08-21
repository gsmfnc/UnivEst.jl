# Signal definition and sampling
bias = 3.4;
amps = [2.0, 3.5];
puls = [3.34, 1.23];
phases = [0.0, 0.12];

sig = init_periodical_signal(tf = 20.0);
samples = get_periodical_signal_samples(sig, bias, amps, phases, puls);

# Training
hbias, hamps, hphases, hpuls = periodical_signal_training(sig, samples, 5.0,
    100, nu = 2, window_size = 0.3);

est_samples = get_periodical_signal_samples(sig, hbias, hamps, hphases, hpuls);

plot(samples)
plot!(est_samples)
