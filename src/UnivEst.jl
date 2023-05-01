module UnivEst

export DiffEqFlux, ADAM, LBFGS, BFGS
export init_env, get_sys_solution, set_env_parameter, init_gain_env
export set_gain_env_parameter, init_ctrl_env, set_ctrl_env_parameter
export get_lyapunov_derivative_values, get_ctrl_sol
export estimate_time_derivatives, extract_estimates, test_hgo
export test_timevarying_hgo, plot_gain
export training_routine, freq_training_routine, alg_training_routine
export gain_training_routine, ctrl_training_routine
export fft_plot, find_peaks_infos

using DifferentialEquations, DiffEqFlux, Lux, Optimization, Plots, FFTW, DSP
using Optim
include("functions.jl")
include("observers_functions.jl")
include("training_functions.jl")
include("kwargs_training_functions.jl")
include("kwargs_functions.jl")

CLASSICALHGO = 0;
CASCADE1 = 1;
CASCADE2 = 2;

INCREASING_GAIN = 0;
DECREASING_GAIN = 1;
TIMEVARYING_GAIN = 2;

end
