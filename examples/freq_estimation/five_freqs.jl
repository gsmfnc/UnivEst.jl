#; Signal definition and sampling
bias = 0.5;
amps = [1.2, 0.7, 1.0, 1.3, 0.85];
puls = 2 * pi * [2, 3, 5, 7, 10];
phases = [0.06, -0.08, -0.11, 0.05, 0.07];

sig = init_periodical_signal(tf = 60.0, ts = 1e-02);
samples = get_periodical_signal_samples(sig, bias, amps, phases, puls);

#; Noise
using Random
rng = MersenneTwister(1234);
noise = randn(rng, length(samples)) * 5e-01;
samples = samples + noise;

plot(samples)
plot!(noise)

#; Analysis through FFT
p1, fft_freqs, fft_amps = fft_plot(vec(samples), sig.ts, sig.tf);
plot(p1)

no_peaks = 10;
fft_hpuls, fft_hamps, fft_hphases, fft_hbias =
    find_peaks_infos(fft_amps, fft_freqs, no_peaks);

F = maximum(fft_hpuls[6:end] / (2 * pi))^-1 / 2; #0.06826666666666667

#; Training
hpuls, hphases, hamps, hbias =
    periodical_signal_training(sig, samples, 5.0, varying_iters = [300, 100],
    nu = no_peaks, window_size = F,
    hpuls = fft_hpuls, hamps = fft_hamps, hbias = fft_hbias);

hpuls2, hphases2, hamps2, hbias2 =
    periodical_signal_training(sig, samples, 60.0, its = 300,
    nu = no_peaks, window_size = 20.0,
    hpuls = hpuls, hamps = hamps, hbias = hbias, hphases = hphases);

est_samples =
    get_periodical_signal_samples(sig, hbias2, hamps2, hphases2, hpuls2);

p2 = plot(samples);
p2 = plot!(samples - est_samples);
p2 = plot!(noise);
plot(p2)

writedlm("5freq_samples.csv", samples, ",")
writedlm("5freq_estsamples.csv", est_samples, ",")
writedlm("5freq_noise.csv", noise, ",")
writedlm("5freq_hpuls.csv", hpuls, ",")
writedlm("5freq_hphases.csv", hphases, ",")
writedlm("5freq_hamps.csv", hamps, ",")
writedlm("5freq_hbias.csv", hbias, ",")
writedlm("5freq_hpuls2.csv", hpuls2, ",")
writedlm("5freq_hphases2.csv", hphases2, ",")
writedlm("5freq_hamps2.csv", hamps2, ",")
writedlm("5freq_hbias2.csv", hbias2, ",")
writedlm("5freq_fft_freqs.csv", fft_freqs, ",")
writedlm("5freq_fft_amps.csv", fft_amps, ",")
