phi(u, p, t) = - 2.02 * u[3] + u[2]^2 - u[1];
u0 = [5.0, 0.0, 1.0];
p = [0.0];

jd0 = init_system_obs(phi, u0, p);
jd0 = init_system_obs(phi, u0, p, t0 = 0.0, tf = 10.0, ts = 1e-02);
