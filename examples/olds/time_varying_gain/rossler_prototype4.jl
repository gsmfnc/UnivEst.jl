phi(u, p, t) = - p[2] * u[3] - u[2] - u[1] * p[2] + p[1] * u[1]^2 - p[1] * u[1];
u0 = [0.29984, 0.10035, -0.30002];
hu0 = randn(3, 5);
hu0 = zeros(3, 1);
hp = [0.49687, 0.49528];
t0 = 0.0;
tf = 20.0;
ts = 1e-01;
opt = ADAM(1e-01);

env = init_gain_env(phi, u0, hp, hu0, t0, tf, ts, opt);

hgo_type = UnivEst.CLASSICALHGO;
gain_type = UnivEst.TIMEVARYING_GAIN;

function add_loss(p)
    gamma1 = 1.0;
    gamma2 = 1.0;
    gamma3 = 1.0;
    gamma4 = 1.0;

    return gamma1 * p[1]^-1 + gamma2 * p[2] + gamma3 * p[3]^-1 + gamma4 * p[4]
end
env = set_gain_env_parameter(env, "add_loss", add_loss);

d(t) = 0.01 * sin(50 * t);
env = set_gain_env_parameter(env, "disturbance", d);

W0 = [10.0, 1.0, 5.0, 10.0];
W = gain_training_routine(env, hgo_type, gain_type, W0, 100)

t0s = 0.0;
tfs = 20.0;
tss = 1e-02;
x, hx = test_timevarying_hgo(env, W, u0, [0.0, 0.0, 0.0], t0s, tfs, tss,
    hgo_type, gain_type);

W = [7.58716, 1.05122, 6.52158, 2.72667];

using Plots
times = t0s:tss:(tfs - tss);
p1 = plot(times, (x - hx)');
p2 = plot_gain(W, gain_type);
p3 = plot_gain(W0, gain_type);
plot(p1, p2, p3, layout = (3, 1))
