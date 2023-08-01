s(t) = 2 * sin(3.34 * t) + 3.5 * sin(1.23 * t + 0.12);
t0 = 0.0;
tf = 20.0;
ts = 1e-02;
times = t0:ts:(tf - ts);
nu = 2;

signal_samples = s.(times);

env = init_env(signal_samples, t0, tf, ts, ADAM(1e-01));

F = 0.3;
tf_t = 10.0;
estp = randn(3 * nu + 1) * 1e-02;
hfreqs, hphases, hamps, hbias = freq_training_routine(env, F, tf_t, estp, 100)

p1, fft_freqs, fft_amps = fft_plot(signal_samples, ts, tf);

no_peaks = 2;
estpuls, estamps, estphases = find_peaks_infos(fft_amps, fft_freqs, no_peaks);

nu = 3;
estp = randn(3 * nu + 1) * 1e-02;
hfreqs, hphases, hamps, hbias = freq_training_routine(env, F, tf_t, estp, 100)

nu = 2;
estp = randn(3 * nu + 1) * 1e-02;
hfreqs, hphases, hamps, hbias = freq_training_routine(env, F, tf_t, estp,
    [100, 50], adam_param = [1e-01, 1e-03], window_size = 5)
