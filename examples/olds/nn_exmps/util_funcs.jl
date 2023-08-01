##########################SOLUTION COMPARISON
est_sol = get_sys_solution(env, hp[(n + 1):end], hp[1:n]);
times = env.t0:env.ts:(env.tf - env.ts);
yy = 50;
using Plots
p11 = plot(times, data[1, :]);
p11 = plot!(times, est_sol[1, :], ylim = (-yy, yy));
p21 = plot(times, data[2, :]);
p21 = plot!(times, est_sol[2, :], ylim = (-yy, yy));
p31 = plot(times, data[3, :]);
p31 = plot!(times, est_sol[3, :], ylim = (-yy, yy));
plot(p11, p21, p31, layout = (1, 3))

est_sol = get_sys_solution(env, hp[(n + 1):end], hp[1:n]);
est_sol2 = get_sys_solution(env, hp2[(n + 1):end], hp2[1:n]);
times = env.t0:env.ts:(env.tf - env.ts);
yy = 2.5;
using Plots
p11 = plot(times, data[1, :]);
p11 = plot!(times, est_sol[1, :], ylim = (-yy, yy));
p21 = plot(times, data[2, :]);
p21 = plot!(times, est_sol[2, :], ylim = (-yy, yy));
p31 = plot(times, data[3, :]);
p31 = plot!(times, est_sol[3, :], ylim = (-yy, yy));
p12 = plot(times, data[1, :]);
p12 = plot!(times, est_sol2[1, :], ylim = (-yy, yy));
p22 = plot(times, data[2, :]);
p22 = plot!(times, est_sol2[2, :], ylim = (-yy, yy));
p32 = plot(times, data[3, :]);
p32 = plot!(times, est_sol2[3, :], ylim = (-yy, yy));
plot(p11, p21, p31, p12, p22, p32, layout = (2, 3))

est_sol = get_sys_solution(env, hp[(n + 1):end], hp[1:n]);
est_sol2 = get_sys_solution(env, hp2[(n + 1):end], hp2[1:n]);
times = env.t0:env.ts:(env.tf - env.ts);
yy = 20;
using Plots
p11 = plot(times, data[1, :]);
p11 = plot!(times, est_sol[1, :], ylim = (-yy, yy));
p21 = plot(times, data[2, :]);
p21 = plot!(times, est_sol[2, :], ylim = (-yy, yy));
p12 = plot(times, data[1, :]);
p12 = plot!(times, est_sol2[1, :], ylim = (-yy, yy));
p22 = plot(times, data[2, :]);
p22 = plot!(times, est_sol2[2, :], ylim = (-yy, yy));
plot(p11, p21, p12, p22, layout = (2, 2))

##########################FUNCTION ESTIMATION
DISCARDED_SAMPLES = 10;
function evaluate_functions(data, est_sol, p, hp, ts)
    real_phi = zeros(1, size(data, 2) - DISCARDED_SAMPLES);
    estimated_phi = zeros(1, size(data, 2) - DISCARDED_SAMPLES);

    for i = 11:1:size(data, 2)
        real_phi[i - DISCARDED_SAMPLES] = phi(data[:, i], p, i * ts);
        estimated_phi[i - DISCARDED_SAMPLES] = hphi(est_sol[:, i], hp, i * ts);
    end

    return real_phi, estimated_phi;
end

real_phi, estimated_phi = evaluate_functions(data, est_sol2, p, hp2[4:end], ts);
plot(real_phi')
plot!(estimated_phi')

##########################HGO TEST
t0 = 0.0;
tf = 100.0;
ts = 1e-02;
epsilon = 1e-02;
hu0 = randn(3);
est_phi(u, p, t) = hphi(u, p, t);
x, hx = test_hgo(epsilon, phi, p, est_phi, hp[(n + 1):end], u0, hp[1:n], t0, tf,
    ts);

hphi2(u, p, t) = 0;
x, hx2 = test_hgo(epsilon, phi, p, hphi2, [0.0], u0, hp[1:n], t0, tf, ts);

using Plots
times = t0:ts:(tf - ts);
yy = 0.04;
p1 = plot(times, (x .- hx)', ylim = (-yy, yy))
p2 = plot(times, (x .- hx2)', ylim = (-yy, yy))
plot(p1, p2, layout = (2, 1))

##########################
function test_add_loss(u, p, u0)
    gamma = 1.4;

    lyap_term = 0;
    arr = zeros(1, size(u, 2));
    for i = 1:1:size(u, 2)
        du = [u[2, i], u[3, i], W(p)' * sigma.(V(p) * u[:, i])];
        lyap_term = lyap_term + gamma * (tanh(u[:, i]' * du) + 1);

        arr[i] = gamma * (tanh(u[:, i]' * du) + 1);
    end

    return lyap_term, arr;
end

##########################ROBUSTNESS TO IC
u0_d = 1.0 * randn(3);
env = set_env_parameter(env, "phi", phi);
sol1 = get_sys_solution(env, p, u0 + u0_d);
env = set_env_parameter(env, "phi", hphi);
sol2 = get_sys_solution(env, hp2[(n + 1):end], u0 + u0_d);

times = env.t0:env.ts:(env.tf - env.ts);
using Plots
p1 = plot(times, sol1[1, :]);
p1 = plot!(times, sol2[1, :]);
p2 = plot(times, sol1[2, :]);
p2 = plot!(times, sol2[2, :]);
p3 = plot(times, sol1[3, :]);
p3 = plot!(times, sol2[3, :]);
plot(p1, p2, p3, layout = (3, 1))
#plot!(ylim = (-5, 5))

##########################WEIGHTS INIT
random_data = zeros(3, 10000);
radius = 10.0;
for i = 1:1:10000
    tmp = randn(3);
    nz = 1 / sqrt(sum(tmp.^2));
    random_data[:, i] = tmp * nz * radius;
    if i % 100 == 0
        radius = radius + 0.1;
    end
end

function loss(p)
    l = 0;
    for i = 1:1:size(random_data, 2)
        x = random_data[:, i];
        l = l + tanh(x[1] * x[2] + x[2] * x[3] + x[3] *
            (W(p) * sigma.(V(p) * x))[1]) + 1;
    end
    alpha = 1e02;
    beta = 1e01;

    l = l + alpha * maximum(sum(abs, W(p), dims = 1));
    l = l + beta * maximum(sum(abs, V(p), dims = 1));

    return l;
end

using Optim
p0 = randn(l)
res = Optim.optimize(loss, p0);
estp = res.minimizer

dV_values = zeros(size(random_data, 2));
for i = 1:1:length(dV_values)
    x = random_data[:, i];
    dx = [x[2], x[3], (W(estp) * sigma.(V(estp) * x))[1]];
    dV_values[i] = x' * dx;
end
using Plots
plot(dV_values)

#dV_values = zeros(size(random_data, 2));
#for i = 1:1:length(dV_values)
#    x = random_data[:, i];
#    dx = [x[2], x[3], phi(x, p, 0.0)];
#    dV_values[i] = x' * dx;
#end
#plot(dV_values)

dV_values = zeros(size(random_data, 2));
for i = 1:1:length(dV_values)
    x = random_data[:, i];
    dx = [x[2], x[3], (W(estp[4:end]) * sigma.(V(estp[4:end]) * x))[1]];
    dV_values[i] = x' * dx;
end
using Plots
plot(dV_values)
