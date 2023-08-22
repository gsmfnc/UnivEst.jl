using DifferentialEquations, DiffEqFlux, Optimization, Plots, FFTW, DSP
using Optim, ControlSystems, ComponentArrays, OptimizationOptimJL
using OptimizationFlux, DelimitedFiles

#export system_obs, forward_kinematics, periodical_signal
include("structs/structs.jl")

#export init_system_obs, init_forward_kinematics, init_periodical_signal
include("structs/init_structs.jl")

#export evaluate_forward_kinematics
include("structs/utils_structs.jl")

#export get_sys_solution, get_periodical_signal_samples
include("system_solution/system_solution.jl")

#export bode_hgo, estimate_t_derivatives, get_hgo_matrices, test_hgo
include("observers/observers.jl")

#export fd_kin_training, periodical_signal_training, find_infos_from_estp
include("training/training.jl")

include("training/losses.jl")

export fft_plot, find_peaks_infos
include("fft_funcs/fft_funcs.jl")

struct tmp
    CLASSICALHGO
    M_CASCADE
    CASCADE
    MIN_CASCADE
    INCREASING_GAIN
    DECREASING_GAIN
    TIMEVARYING_GAIN
end

CLASSICALHGO = 0;
M_CASCADE = 1;
CASCADE = 2;
MIN_CASCADE = 3;

INCREASING_GAIN = 0;
DECREASING_GAIN = 1;
TIMEVARYING_GAIN = 2;

UnivEst = tmp(CLASSICALHGO, M_CASCADE, CASCADE, MIN_CASCADE, INCREASING_GAIN,
    DECREASING_GAIN, TIMEVARYING_GAIN);
