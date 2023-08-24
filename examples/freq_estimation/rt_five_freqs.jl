#; Signal definition and sampling
bias = 0.5;
amps = [1.2, 0.7, 1.0, 1.3, 0.85];
puls = 2 * pi * [2, 3, 5, 7, 10];
phases = [0.06, -0.08, -0.11, 0.05, 0.07];

sig = init_periodical_signal(tf = 60.0, ts = 1e-02);
samples0 = get_periodical_signal_samples(sig, bias, amps, phases, puls);

#; Noise
using Random
rng = MersenneTwister(1234);
noise = randn(rng, length(samples0)) * 5e-01;
samples = samples0 + noise;

plot(samples)
plot!(noise)

#; Analysis through FFT
@time begin
p1, fft_freqs, fft_amps = fft_plot(vec(samples[1:500]), sig.ts, 5.0);
plot(p1)

fft_hpuls1, fft_hamps1, fft_hphases1, fft_hbias =
    find_peaks_infos(fft_amps, fft_freqs, 10);

no_peaks = 10;
for i = 10:-1:2
    if fft_hamps1[i - 1] / fft_hamps1[i] < 0.3
        no_peaks = i - 1;
        break
    end
end
fft_hpuls = fft_hpuls1[(no_peaks + 1):end]
fft_hamps = fft_hamps1[(no_peaks + 1):end]
fft_hphases = fft_hphases1[(no_peaks + 1):end]

F = maximum(fft_hpuls / (2 * pi))^-1 * 4; #0.4096
end

#; Training
hpuls, hphases, hamps, hbias, estps, times =
    periodical_signal_training(sig, samples, 20.0, varying_iters = [200, 100],
    nu = no_peaks, window_size = F, max_window_number = 20,
    varying_adam_p = [0.01, 0.001], save = true, t0 = 5.0,
    hpuls = fft_hpuls, hamps = fft_hamps, hbias = fft_hbias);

[fft_hpuls fft_hamps]
[hpuls hamps]
[puls amps]

writedlm("rt_fivefreqs_samples.csv", samples, ",")
writedlm("rt_fivefreqs_noise.csv", noise, ",")
writedlm("rt_fivefreqs_hpuls.csv", hpuls, ",")
writedlm("rt_fivefreqs_hphases.csv", hphases, ",")
writedlm("rt_fivefreqs_hamps.csv", hamps, ",")
writedlm("rt_fivefreqs_hbias.csv", hbias, ",")
writedlm("rt_fivefreqs_estps.csv", estps, ",")
writedlm("rt_fivefreqs_times.csv", times, ",")
writedlm("rt_fivefreqs_fft_freqs.csv", fft_freqs, ",")
writedlm("rt_fivefreqs_fft_amps.csv", fft_amps, ",")
writedlm("rt_fivefreqs_fft_hpuls.csv", fft_hpuls, ",")
writedlm("rt_fivefreqs_fft_hamps.csv", fft_hamps, ",")
writedlm("rt_fivefreqs_fft_hphases.csv", fft_hphases, ",")
writedlm("rt_fivefreqs_fft_hbias.csv", fft_hbias, ",")
