# UnivEst.jl

Software versions:

-   Julia version 1.8.3 (2022-11-14)
    (Julia version 1.9.2 (2023-07-05))
-   ComponentArrays v0.13.7
-   ControlSystems v1.5.2
-   DSP v0.7.7
-   DiffEqFlux v1.52.0
-   DifferentialEquations v7.6.0
-   FFTW v1.5.0
-   Optim v1.7.4
-   Optimization v3.9.4
-   OptimizationFlux v0.1.2
-   OptimizationOptimJL v0.1.5
-   Plots v1.36.4

# Features

Implements:
- [Cascades of high-gain observers](https://github.com/gsmfnc/UnivEst.jl/tree/main/examples/high_gain_observers)
- [Off-line estimation of the Denavit-Hartenberg table parameters](https://github.com/gsmfnc/UnivEst.jl/tree/main/examples/dh_table)
- [Estimation of the frequency content in a periodic signal](https://github.com/gsmfnc/UnivEst.jl/tree/main/examples/freq_estimation)
- [Parameters estimation for observable continuous-time systems](https://github.com/gsmfnc/UnivEst.jl/tree/main/examples/sys_est)
- [Neural high-gain observers](https://github.com/gsmfnc/UnivEst.jl/tree/main/examples/obs_design)
- [Data-driven Lyapunov-based synthesis of feedback controllers](https://github.com/gsmfnc/UnivEst.jl/tree/main/examples/lyapunov_based_ctrl)

References:
1.  Gismondi, F., Possieri, C., & Tornambe, A. (2022). Design of neural high-gain observers for autonomous nonlinear systems using universal differential equations. International Journal of Dynamics and Control, 10(6), 1794-1806.
1.  Gismondi, F. Data-driven and adaptive approaches for system identification, observer design and controller synthesis. PhD Thesis.

# How to use

Clone this repository and run `include("src/UnivEst.jl")` in your Julia console.
In case you are missing some of the required packages, you can run the script
install.jl in the src folder.
