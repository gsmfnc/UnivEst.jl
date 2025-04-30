# System CS_1 from Table 6.4 in "Sprott, Julien C. Elegant chaos: algebraically
# simple chaotic flows. World Scientific, 2010."

phi(u, p, t) = - u[4] + p[1] * u[3] + p[2] * u[2] + p[3] * (u[1]^2 - 1);
u0 = [-0.9, 0.0, 0.0, 0.0];
p = [-5.2, -2.7, 4.5];
obs_map(u, p, t) = [u[1]; u[2]; u[3]];

cs1 = init_system_obs(phi, obs_map, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
cs1_sol, cs1_y = get_sys_solution(cs1);

# Noise signal
noise_samples = randn(length(cs1_y), 1) * 0.01;
samples = cs1_y + noise_samples;
plot(samples)
plot!(noise_samples)

# Time derivatives
sol = estimate_t_derivatives(samples, UnivEst.M_CASCADE, 3,
    [0.01, 0.02, 0.03, 0.04], cs1.t0, cs1.ts, cs1.tf);
hcs1_sol = sol[[1, 3, 5, 7], :];

dataset = hcat(hcs1_sol[1, :], hcs1_sol[2, :], hcs1_sol[3, :]);

tfs = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0];
hu0, hp, times, estps = sysobs_training(cs1, dataset, tfs, 300, save = true,
    dtime = 0.5,
    callback = true, estu0 = [0.01, -0.01, 0.01, -0.01],
    estp0 = [0.01, -0.01, 0.01, -0.01]);

cs1_2 = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
tfs = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0];
hu02, hp2, times, estps = sysobs_training(cs1_2, samples, tfs, 300, save = true,
    callback = true, estu0 = [0.01, -0.01, 0.01, -0.01],
    estp0 = [0.01, -0.01, 0.01, -0.01]);

cs1_hsol, cs1_hy = get_sys_solution(cs1, u0_arg = hu0, p_arg = hp);
plot(cs1_y - cs1_hy)

writedlm("cs1_estps.csv", estps5, ",")
writedlm("cs1_noise.csv", noise_samples, ",")
writedlm("cs1_y.csv", cs1_y, ",")
writedlm("cs1_hy.csv", cs1_hy, ",")
