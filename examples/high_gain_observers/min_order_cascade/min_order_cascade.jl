# System JD_0 from Table 3.2 in "Sprott, Julien C. Elegant chaos: algebraically
# simple chaotic flows. World Scientific, 2010."

phi(u, p, t) = - 2.02 * u[3] + u[2]^2 - u[1];
u0 = [5.0, 0.0, 1.0];
p = [0.0];
jd0 = init_system_obs(phi, u0, p, t0 = 0.0, tf = 10.0, ts = 1e-02);

eps = 0.1;
u, z, hu = test_hgo(jd0, UnivEst.MIN_CASCADE, eps, phi, p,
    coeffs = [2.0, 2.0, 1.0], S = [10.0, 10.0, 10.0]);

using DelimitedFiles
writedlm("u_mincascade.csv", u, ',');
writedlm("z_mincascade.csv", z, ',');
writedlm("hu_mincascade.csv", hu, ',');
