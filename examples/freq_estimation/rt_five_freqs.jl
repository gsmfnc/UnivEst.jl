#; Signal definition and sampling
bias = 0.5;
amps = [1.2, 0.7, 1.0, 1.3, 0.85];
puls = [1.0, 1.5, 2.3, 3.1, 5.0];
phases = [0.06, -0.08, -0.11, 0.05, 0.07];

sig = init_periodical_signal(tf = 60.0, ts = 1e-01);
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
    periodical_signal_training(sig, samples, 60.0, varying_iters = [150, 100],
    nu = 7, window_size = 0.5, max_window_number = 20,
    varying_adam_p = [0.1, 0.001], save = true);

est_samples =
    get_periodical_signal_samples(sig, hbias, hamps, hphases, hpuls);

p2 = plot(samples);
p2 = plot!(est_samples);
plot(p2)

writedlm("rt_fivefreqs_samples.csv", samples, ",")
writedlm("rt_fivefreqs_estsamples.csv", est_samples, ",")
writedlm("rt_fivefreqs_noise.csv", noise, ",")
writedlm("rt_fivefreqs_hpuls.csv", hpuls, ",")
writedlm("rt_fivefreqs_hphases.csv", hphases, ",")
writedlm("rt_fivefreqs_hamps.csv", hamps, ",")
writedlm("rt_fivefreqs_hbias.csv", hbias, ",")
writedlm("rt_fivefreqs_estps.csv", estps, ",")
writedlm("rt_fivefreqs_times.csv", times, ",")
