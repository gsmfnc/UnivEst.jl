include("../examples/dh_table/quanser_data.jl")

# Inputs
q1(t) = cos(t);
q2(t) = sin(2 * t);
q3(t) = 0.5 * cos(3 * t);
q4(t) = sin(t);

# Forward kinematics
forward_kin(u, p, t) = [
    -cos(q1(t))*(-sin(p[end]+q2(t))*(p[3]*sin(p[end]-q3(t))+
        ((p[2])/(cos(p[end]))))-p[3]*cos(p[end]+q2(t))*cos(p[end]-q3(t)))
    -sin(q1(t))*(-sin(p[end]+q2(t))*(p[3]*sin(p[end]-q3(t))+
        ((p[2])/(cos(p[end]))))-p[3]*cos(p[end]+q2(t))*cos(p[end]-q3(t)))
    cos(p[end]+q2(t))*(p[3]*sin(p[end]-q3(t))+((p[2])/(cos(p[end]))))-
        p[3]*sin(p[end]+q2(t))*cos(p[end]-q3(t))+p[1]
];

# Training
fd_kin = init_forward_kinematics(forward_kin, t0 = 0.0, tf = 10.0, ts = 1e-02);

tfs = [1.0, 2.0, 3.0];
estp = vec(randn(4, 1) * 1e-2);
hp = fd_kin_training(fd_kin, samples, tfs, 500, estp);

# Comparison with nominal Denavit-Hartenberg table
nominal_p = [0.14, 0.35, 0.25, deg2rad(8.13)];
nominal_traj = evaluate_forward_kinematics(fd_kin, nominal_p);
estim_traj = evaluate_forward_kinematics(fd_kin, hp);

## Plots
using LaTeXStrings
labs = [L"y_1(t)-f_1(u(t),p_2,p_3,p_4)",
        L"y_2(t)-f_2(u(t),p_2,p_3,p_4)",
        L"y_3(t)-f_3(u(t),p_2,p_3,p_4)-p_1",
        L"y_1(t)-f_1(u(t),\hat p_2,\hat p_3,\hat p_4)",
        L"y_2(t)-f_2(u(t),\hat p_2,\hat p_3,\hat p_4)",
        L"y_3(t)-f_3(u(t),\hat p_2,\hat p_3,\hat p_4)-\hat p_1"];
times = fd_kin.t0:fd_kin.ts:(fd_kin.tf - fd_kin.ts);
p1 = plot(times, samples[:, 1] - nominal_traj[1, :], label = labs[1]);
p1 = plot!(times, samples[:, 1] - estim_traj[1, :], label = labs[4]);
p1 = plot!(legend = :outerright);
p1 = xlabel!("t");
p1 = plot!(grid = :false);

p2 = plot(times, samples[:, 2] - nominal_traj[2, :], label = labs[2]);
p2 = plot!(times, samples[:, 2] - estim_traj[2, :], label = labs[5]);
p2 = plot!(legend = :outerright);
p2 = xlabel!("t");
p2 = plot!(grid = :false);

p3 = plot(times, samples[:, 3] - nominal_traj[3, :], label = labs[3]);
p3 = plot!(times, samples[:, 3] - estim_traj[3, :], label = labs[6]);
p3 = plot!(legend = :outerright);
p3 = xlabel!("t");
p3 = plot!(grid = :false);

plot(p1, p2, p3, layout = (3, 1), grid = :true)

######
using DelimitedFiles
writedlm("nominal_traj.csv", nominal_traj, ',');
writedlm("estim_traj.csv", estim_traj, ',');
writedlm("samples.csv", samples', ',');
