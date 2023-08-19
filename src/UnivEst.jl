using DifferentialEquations, DiffEqFlux, Optimization, Plots, FFTW, DSP
using Optim, ControlSystems, ComponentArrays, OptimizationOptimJL
using OptimizationFlux

#export system_obs, forward_kinematics 
include("structs/structs.jl")

#export init_system_obs, init_forward_kinematics
include("structs/init_structs.jl")

#export evaluate_forward_kinematics
include("structs/utils_structs.jl")

#export get_sys_solution
include("system_solution/system_solution.jl")

#export bode_hgo, estimate_t_derivatives, get_hgo_matrices, test_hgo
include("observers/observers.jl")

#export fd_kin_training
include("training/training.jl")

include("training/losses.jl")

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
