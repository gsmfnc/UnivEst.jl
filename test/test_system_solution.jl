phi(u, p, t) = p[1] * u[3] + p[2] * u[2]^2 + p[3] * u[1];
u0 = [5.0, 0.0, 1.0];
p = [-2.02, 1, -1];

jd0 = init_system_obs(phi, u0, p);
sol1, out1 = get_sys_solution(jd0);
jd0 = init_system_obs(phi, u0, p, t0 = 0.0, tf = 10.0, ts = 1e-02);
sol2, out2 = get_sys_solution(jd0);
