#; Signal definition and sampling
bias = 0.0;
amps = [2.0, 4.0];
puls = [1.0, 1.2];
phases = [0.0, 0.0];

sig = init_periodical_signal(tf = 20.0, ts = 1e-02);
samples = get_periodical_signal_samples(sig, bias, amps, phases, puls);

#; Noise
using Random
rng = MersenneTwister(1234);
noise = randn(rng, length(samples)) * 5e-01;
samples = samples + noise;

plot(samples)
plot!(noise)

#; Training
hpuls, hphases, hamps, hbias =
    periodical_signal_training(sig, samples, 5.0, varying_iters = [300, 100],
    nu = 2, window_size = 0.5);

hpuls2, hphases2, hamps2, hbias2 =
    periodical_signal_training(sig, samples, 20.0, its = 300,
    nu = 2, window_size = 5.0,
    hpuls = hpuls, hamps = hamps, hbias = hbias, hphases = hphases);

est_samples =
    get_periodical_signal_samples(sig, hbias2, hamps2, hphases2, hpuls2);

p1 = plot(samples);
p1 = plot!(est_samples);
plot(p1)

writedlm("close_freq_samples.csv", samples, ",")
writedlm("close_freq_estsamples.csv", est_samples, ",")
writedlm("close_freq_noise.csv", noise, ",")
writedlm("close_freq_hpuls.csv", hpuls, ",")
writedlm("close_freq_hphases.csv", hphases, ",")
writedlm("close_freq_hamps.csv", hamps, ",")
writedlm("close_freq_hbias.csv", hbias, ",")
writedlm("close_freq_hpuls2.csv", hpuls2, ",")
writedlm("close_freq_hphases2.csv", hphases2, ",")
writedlm("close_freq_hamps2.csv", hamps2, ",")
writedlm("close_freq_hbias2.csv", hbias2, ",")
