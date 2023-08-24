#; Signal definition and sampling
bias = 2.0;
amps = [0.5, 1.0];
puls = [1.0, 2.0];
phases = [0.0, 0.0];

sig = init_periodical_signal(tf = 20.0, ts = 1e-01);
samples0 = get_periodical_signal_samples(sig, bias, amps, phases, puls);

#; Noise
using Random
rng = MersenneTwister(1234);
noise = randn(rng, length(samples0)) * 5e-01;
samples = samples0 + noise;

plot(samples)
plot!(noise)

#; Training
hpuls, hphases, hamps, hbias, estps, times =
    periodical_signal_training(sig, samples, 20.0, varying_iters = [200, 100],
    nu = 2, window_size = 0.5, max_window_number = 30,
    varying_adam_p = [0.1, 0.001], save = true);

est_samples =
    get_periodical_signal_samples(sig, hbias, hamps, hphases, hpuls);

p2 = plot(samples0);
p2 = plot!(est_samples);
plot(p2)

writedlm("realtime_samples.csv", samples, ",")
writedlm("realtime_estsamples.csv", est_samples, ",")
writedlm("realtime_noise.csv", noise, ",")
writedlm("realtime_hpuls.csv", hpuls, ",")
writedlm("realtime_hphases.csv", hphases, ",")
writedlm("realtime_hamps.csv", hamps, ",")
writedlm("realtime_hbias.csv", hbias, ",")
writedlm("realtime_estps.csv", estps, ",")
writedlm("realtime_times.csv", times, ",")
