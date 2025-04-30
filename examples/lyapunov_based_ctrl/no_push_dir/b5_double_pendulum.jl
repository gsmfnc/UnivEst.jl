# alpha[1]*((m2+m1)*L1^2+2*cos(q[2])*m2*L1*L2+m2*L2^2)+
#   alpha[2]*m2*(cos(q[2])*L1*L2+L2^2)-omega[2]^2*sin(q[2])*m2*L1*L2-
#   2*omega[1]*omega[2]*sin(q[2])*m2*L1*L2+cos(q[2]+q[1])*g*m2*L2+
#   cos(q[1])*g*(m1+m2)*L1
#
# alpha[1]*m2*(cos(q[2])*L1*L2+L2^2)+alpha[2]*m2*L2^2+
#   omega[1]^2*sin(q[2])*m2*L1*L2+cos(q[2]+q[1])*g*m2*L2
#
#inverse_M
# [1/(sin(q[2])^2*L1^2*m2+L1^2*m1) -(L2+cos(q[2])*L1)/(sin(q[2])^2*L1^2*L2*m2+
#   L1^2*L2*m1);
# -(L2+cos(q[2])*L1)/(sin(q[2])^2*L1^2*L2*m2+L1^2*L2*m1) ((L2^2+
#	2*cos(q[2])*L1*L2+L1^2)*m2+L1^2*m1)/(sin(q[2])^2*L1^2*L2^2*m2^2+
#	L1^2*L2^2*m1*m2)]
#

# Define system
n = 4;

L1 = 0.5;
L2 = 0.5;
m1 = 1.;
m2 = 1.;
g = 9.81;

# just some tests
u(x, p) = [
	0.
    p[2] * (-tanh(10. * (x[1] + 1.0)) * 0.1 - x[2])
	0.
    p[4] * (-tanh(10. * x[3]) * 0.1 - x[4])
];
invM(x) = [
	1	0	0	0;
	0 1/(sin(x[2])^2*L1^2*m2+L1^2*m1) 0 -(L2+cos(x[2])*L1)/(sin(x[2])^2*
		L1^2*L2*m2+L1^2*L2*m1);
	0	0	1	0;
 	0 -(L2+cos(x[2])*L1)/(sin(x[2])^2*L1^2*L2*m2+L1^2*L2*m1) 0 ((L2^2+
		2*cos(x[2])*L1*L2+L1^2)*m2+L1^2*m1)/(sin(x[2])^2*L1^2*L2^2*m2^2+
		L1^2*L2^2*m1*m2)
];
dyn_noctrl(x) = [
	x[2]
	-x[4]^2*sin(x[2])*m2*L1*L2-
	   2*x[2]*x[4]*sin(x[2])*m2*L1*L2+cos(x[2]+x[1])*g*m2*L2+
	   cos(x[1])*g*(m1+m2)*L1
	x[4]
	x[2]^2*sin(x[2])*m2*L1*L2+cos(x[2]+x[1])*g*m2*L2
];
u_eq = deg2rad.([90.0, 0.0, 0.0, 0.0]);
f(x, p, t) = invM(x .+ u_eq) * (dyn_noctrl(x .+ u_eq) + u(x .+ u_eq, p)) .*
	[1.0, 1.0, 0.0, 0.];
sys_to_ctrl = init_controlled_system(f, u);

u0 = vec(randn(n, 1));
p0 = [0., 10., 0., 1.];
t_sol_std, sol_std, u_vals_std =
    get_controlled_sys_solution(sys_to_ctrl, p0, u0, tf = 50.0);

p1 = plot(t_sol_std, rad2deg.(sol_std[1, :]), ylabel = "theta1");
p2 = plot(t_sol_std, rad2deg.(sol_std[3, :]), ylabel = "theta2");
p3 = plot(t_sol_std, sol_std[2, :], ylabel = "v1");
p4 = plot(t_sol_std, sol_std[4, :], ylabel = "v2");
plot(p1, p2, p3, p4, layout = (2, 2))

# consider only velocities equations, find stabilizing controller and then use
# tanh(10. * theta_des) * vmax as the desidered velocity
