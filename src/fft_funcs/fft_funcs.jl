################################################################################
##########################EXPORT FUNCTIONS######################################
################################################################################
"""
    fft_plot(x::Vector{Float64}, ts::Float64, tfinal::Float64)

FFT of the samples 'x' with sampling time ts and considering the timespan 0:tf.
"""
function fft_plot(x::Vector{Float64}, ts::Float64, tfinal::Float64)
    Fs = Int(round(1 / ts));
    L = Int(round(tfinal * Fs));
    t = 0:ts:(tfinal - ts);

    n = nextpow(2, L);

    Y = fft(x);

    P2 = abs.(Y / L);
    P1 = P2[1:(Int(round(n/2)) + 1)];
    P1[2:(end - 1)] = 2 * P1[2:(end - 1)];

    freqs = 0:Fs/n:Fs/2-Fs/n;
    amps = P1[1:Int(round(n / 2))];

    p1 = plot(freqs, amps);

    p1, freqs, amps
end

"""
    find_peaks_infos(fft_amps::Vector{Float64},
        fft_freqs::StepRangeLen{Float64, Base.TwicePrecision{Float64},
        Base.TwicePrecision{Float64}}, n::Int)

Given amplitudes and frequencies of a FFT plot (see fft_plot), returns the
frequency and the amplitudes of the larger 'n' peaks.
"""
function find_peaks_infos(fft_amps::Vector{Float64},
        fft_freqs::StepRangeLen{Float64, Base.TwicePrecision{Float64},
        Base.TwicePrecision{Float64}}, n::Int)
    peaks_indx = findlocalmaxima(fft_amps);

    tmp = sortperm(fft_amps[peaks_indx])[(end - n + 1):end];
    tmp2 = peaks_indx[tmp];
    freqs = 2 * pi * fft_freqs[tmp2];
    amps = fft_amps[tmp2];
    bias = fft_amps[1];
    phases = zeros(length(freqs));

    return freqs, amps, phases, bias
end

################################################################################
########################NOT EXPORTED############################################
################################################################################
"""
    findlocalmaxima(signal::Vector{Float64})

Finds local maxima of a vector of values.
"""
function findlocalmaxima(signal::Vector{Float64})
    inds = Int[]
    if signal[2] > signal[3]
        push!(inds, 2)
    end
    if length(signal) > 1
        for i = 3:(length(signal) - 1)
            if signal[i - 1] < signal[i] > signal[i + 1]
                push!(inds, i)
            end
        end
    end
    inds
end
