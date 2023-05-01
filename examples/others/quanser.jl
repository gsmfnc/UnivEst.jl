include("quanser_data.jl")

q1(t) = cos(t);
q2(t) = sin(2 * t);
q3(t) = 0.5 * cos(3 * t);
q4(t) = sin(t);
forward_kin(u, p, t) = [
    -cos(q1(t))*(-sin(p[end]+q2(t))*(p[3]*sin(p[end]-q3(t))+
        ((p[2])/(cos(p[end]))))-p[3]*cos(p[end]+q2(t))*cos(p[end]-q3(t)))
    -sin(q1(t))*(-sin(p[end]+q2(t))*(p[3]*sin(p[end]-q3(t))+
        ((p[2])/(cos(p[end]))))-p[3]*cos(p[end]+q2(t))*cos(p[end]-q3(t)))
    cos(p[end]+q2(t))*(p[3]*sin(p[end]-q3(t))+((p[2])/(cos(p[end]))))-
        p[3]*sin(p[end]+q2(t))*cos(p[end]-q3(t))+p[1]
];

t0 = 0.0;
tf = 10.0;
ts = 1e-02;
opt = ADAM(1e-02);

env = init_env(forward_kin, samples, t0, tf, ts, opt);

tfs = [1.0, 2.0, 3.0];
estp = vcat(randn(4) * 1e-03);
hp = alg_training_routine(env, tfs, estp, 500)

N = Int(round(tf / ts));
traj_nominal = zeros(3, N);
traj_estimated = zeros(3, N);
nominal_p = [0.14, 0.35, 0.25, deg2rad(8.13)];
for i = 1:1:N
    traj_nominal[:, i] = forward_kin(0, nominal_p, (i - 1) * ts);
    traj_estimated[:, i] = forward_kin(0, hp, (i - 1) * ts);
end

using Plots, LaTeXStrings
labs = [L"y_1(t)-f_1(u(t),p_2,p_3,p_4)",
        L"y_2(t)-f_2(u(t),p_2,p_3,p_4)",
        L"y_3(t)-f_3(u(t),p_2,p_3,p_4)-p_1",
        L"y_1(t)-f_1(u(t),\hat p_2,\hat p_3,\hat p_4)",
        L"y_2(t)-f_2(u(t),\hat p_2,\hat p_3,\hat p_4)",
        L"y_3(t)-f_3(u(t),\hat p_2,\hat p_3,\hat p_4)-\hat p_1"];
times = t0:ts:(tf - ts);
p1 = plot(times, env.data[:, 1] - traj_nominal[1, :],
    label = labs[1])
p1 = plot!(times, env.data[:, 1] - traj_estimated[1, :],
    label = labs[4])
p1 = plot!(legend = :outerright)
p1 = xlabel!("t")
p1 = plot!(grid = :false)

p2 = plot(times, env.data[:, 2] - traj_nominal[2, :],
    label = labs[2])
p2 = plot!(times, env.data[:, 2] - traj_estimated[2, :],
    label = labs[5])
p2 = plot!(legend = :outerright)
p2 = xlabel!("t")
p2 = plot!(grid = :false)

p3 = plot(times, env.data[:, 3] - traj_nominal[3, :],
    label = labs[3])
p3 = plot!(times, env.data[:, 3] - traj_estimated[3, :],
    label = labs[6])
p3 = plot!(legend = :outerright)
p3 = xlabel!("t")
p3 = plot!(grid = :false)

plot(p1, p2, p3, layout = (3, 1), grid = :false)
