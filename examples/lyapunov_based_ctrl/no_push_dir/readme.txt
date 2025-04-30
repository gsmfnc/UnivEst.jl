TODOs:
    black box identification of duffing
    using one of the systems used as an example for neural HGOs paper/thesis,
        consider it rewritten in obs form, synthesize a controller using that
        re-formulation, then simulate (real system, neural HGO, controller with
        neural HGO states) and see what happens
    stabilize a7_...

--------------------------------------------------------------------------------
a0_linear_OK.jl
    dx1 = x2
    dx2 = - x1

    experiment showing that alpha has effects on robustness
    --> stabilized

a1_linear.jl
    A = [0, 1; -1, 1]
    T = [-0.168, 0.206; -0.647, 0.189]
    --> stabilized

a2_pendulum_linearctrl.jl
    pendulum with friction (linear controller)
    --> how to stabilize [1.0, 0.0] equilibrium?
    LINEAR CONTROLLER MAY NOT BE ABLE TO STABILIZE THE PENDULUM SO FOCUS ON THE
    NEXT FILE!

a3_pendulum_linapproximator_OK.jl
    pendulum with friction (vector of monomials)
    --> origin stabilized + bias to stabilize other points

a4_duffing.jl
    duffing oscillator (vector of monomials)
    --> stabilized

a5_ex4_27.jl
    dx1 = - x[1] + x[2]^2
    dx2 = - x[2]

    vector of monomials --> stabilized

a6_4_10_exercises_4_3_1.jl
    dx1 = - x[1] + x[1] * x[2]
    dx2 = - x[2]

    vector of monomials --> stabilized

a7_4_10_exercises_4_3_2.jl
    - x[2] - x[1] * (1 - x[1]^2 - x[2]^2)
    x[1] - x[2] * (1 - x[1]^2 - x[2]^2)

    vector of monomials

a8_4_10_exercises_4_3_3.jl
    x[2] * (1 - x[1]^2)
    - (x[1] + x[2]) * (1 - x[1]^2)

    vector of monomials --> stabilized

a9_4_10_exercises_4_3_4.jl
    - x[1] - x[2]
    2 * x[1] - x[2]^3

    vector of monomials --> stabilized

b1_4_10_exercises_rotating_rigid_spacecraft.jl
    full-state linear feedback --> stabilized
    proven that it finds a more robust controller

b2_quadrotor.jl
b3_quadrotor_linear.jl
b4_unicycle.jl

b5_double_pendulum.jl
b6_wiener_hammerstein.jl
b7_separately_excited_dc_motor.jl
b8_ex_1_14_OK.jl
    dx1 = x[2]
    dx2 = 1 / M * (-k1 * sign(x[2]) - k2 * x[2] - k3 * x[2]^2) - g * sind(theta)
    
    cart + hill

b9_ex_1_15.jl
    cart + pendulum

c1_chua_cv1.jl
    stabilized considering second loss function

c2_chua_cv1.jl
    use HGO+ctrl law of c1_chua_cv1.jl to regulate x_1 --> success
