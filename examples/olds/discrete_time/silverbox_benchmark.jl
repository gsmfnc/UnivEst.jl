using CSV, DataFrames

data = CSV.read("examples/discrete_time/SNLS80mV.csv", DataFrame);
array_data = Matrix(data);

input_data = array_data[:, 1];
output_data = array_data[:, 2];

tmp_u = input_data[40650:127400];
tmp_y = output_data[40650:127400];
validation_u = tmp_u[1:Int(floor(size(tmp_u, 1) * 0.25))];
validation_y = tmp_y[1:Int(floor(size(tmp_y, 1) * 0.25))];
training_u = tmp_u[(Int(floor(size(tmp_u, 1) * 0.25)) + 1):end];
training_y = tmp_y[(Int(floor(size(tmp_y, 1) * 0.25)) + 1):end];
test_u = input_data[10:40575];
test_y = output_data[10:40575];
training_y = reshape(training_y, size(training_y, 1), 1);
validation_y = reshape(validation_y, size(validation_y, 1), 1);
test_y = reshape(test_y, size(test_y, 1), 1);

ts = 1;
f(u, p, k) = [
    u[1] + ts * u[2]
    u[2] + ts * (- p[1] * u[1] - p[2] * u[2] - p[3] * training_y[k]^2 * u[1] +
        p[4] * training_u[k])
];
h(u, p) = u[1];
n = 2;
l = 4;
opt = ADAM(1e-02);

env = init_dt_env(f, h, n, opt, training_y);
estp = randn(n + l) * 1e-03;

N = 10;
env = set_dt_env_parameter(env, "y_samples", reshape(training_y[1:N], N, 1));
hp = dt_training_routine(env, estp, 100);

Ns = [20, 30, 50, 100];
for i = 1:1:length(Ns)
    N = Ns[i];
    env = set_dt_env_parameter(env, "y_samples",
        reshape(training_y[1:N], N, 1));
    hp = dt_training_routine(env, hp, 100);
end

est_sol = get_dt_sys_sol(env, hp[(n + 1):end], hp[1:n], N);
plot(training_y[1:N])
plot!(est_sol[1, :])

#[0.03188812368657232, -0.011212483431586267, 0.4653669827852573,
#    0.5283817194640894, 1.2992087818129188, 0.41669539531921873]

Ns = [1000, 2000, 3000, 4000, 5000];
for i = 1:1:length(Ns)
    N = Ns[i];
    env = set_dt_env_parameter(env, "y_samples",
        reshape(training_y[1:N], N, 1));
    hp = dt_training_routine(env, hp, 100);
end

est_sol = get_dt_sys_sol(env, hp[(n + 1):end], hp[1:n], N);
plot(training_y[1:N])
plot!(training_y[1:N] - est_sol[1, :])

#[0.03313566503916129, -0.011817976170249506, 0.4604614873641791,
#    0.5308164624528258, 1.9775156778886793, 0.4254577251499293]

Ns = [10000, 20000];
for i = 1:1:length(Ns)
    N = Ns[i];
    env = set_dt_env_parameter(env, "y_samples",
        reshape(training_y[1:N], N, 1));
    hp = dt_training_routine(env, hp, 100);
end

#[0.03405936010583936, -0.012255777583653083, 0.4596310405702068,
#    0.5299543303412912, 2.077766578074066, 0.4229882356783704]

env = set_dt_env_parameter(env, "y_samples", training_y);
hp = dt_training_routine(env, hp, 100);

################################################################################

valf(u, p, k) = [
    u[1] + ts * u[2]
    u[2] + ts * (- p[1] * u[1] - p[2] * u[2] - p[3] * validation_y[k]^2 * u[1] +
        p[4] * validation_u[k])
];
testf(u, p, k) = [
    u[1] + ts * u[2]
    u[2] + ts * (- p[1] * u[1] - p[2] * u[2] - p[3] * test_y[k]^2 * u[1] +
        p[4] * test_u[k])
];

N = size(validation_y, 1);
val_sol = get_dt_sys_sol(valf, hp[(n + 1):end], hp[1:n], N);
p1 = plot(validation_y[1:N]);
p1 = plot!(validation_y[1:N] - val_sol[1, :]);
sqrt(sum((validation_y[1:N] - val_sol[1, :]).^2) / N) * 1e03

N = size(test_y, 1);
test_sol = get_dt_sys_sol(testf, hp[(n + 1):end], hp[1:n], N);
p2 = plot(test_y[1:N]);
p2 = plot!(test_y[1:N] - test_sol[1, :]);
sqrt(sum((test_y[1:N] - test_sol[1, :]).^2) / N) * 1e03
sqrt(sum((test_y[1:32000] - test_sol[1, 1:32000]).^2) / N) * 1e03

plot(p1, p2, layout = (2, 1))
