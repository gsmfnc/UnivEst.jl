using DifferentialEquations, DiffEqFlux, Lux, Optimization, Plots, FFTW, DSP
using Optim, ControlSystems

#export system_obs 
include("structs/structs.jl")

#export init_system_obs
include("structs/init_structs.jl")

#export get_sys_solution
include("system_solution/system_solution.jl")

#export bode_hgo, estimate_t_derivatives, get_hgo_matrices
include("observers/observers.jl")

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
MIN_CASCADE = 2;
INCREASING_GAIN = 0;
DECREASING_GAIN = 1;
TIMEVARYING_GAIN = 2;

UnivEst = tmp(CLASSICALHGO, M_CASCADE, CASCADE, MIN_CASCADE, INCREASING_GAIN,
    DECREASING_GAIN, TIMEVARYING_GAIN);
