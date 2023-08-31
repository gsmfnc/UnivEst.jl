################################################################################
########################EXPORTED FUNCTIONS######################################
################################################################################
"""
    evaluate_forward_kinematics(f::forward_kinematics, p::Vector{Float64})

Returns the trajectory in the space of the forward kinematics given in
'f.forward_kinematics', from time 'f.t0' to time 'f.tf' with sampling time
'f.ts', considering the parameters 'p'.
"""
function evaluate_forward_kinematics(f::forward_kinematics, p::Vector{Float64})
    tf = f.tf;
    ts = f.ts;
    t0 = f.t0;

    N = Int(round(tf / ts));
    traj = zeros(length(fd_kin.forward_kinematics(0, p, 0.0)), N);
    for i = 1:1:N
        traj[:, i] = f.forward_kinematics(0, p, (i - 1) * ts);
    end

    return traj;
end

################################################################################
########################NOT EXPORTED############################################
################################################################################
"""
    set_sys_training_env_val(env::sys_training_env, param::String,
        new_value::Any)

Sets the 'param' field of 'env' to 'new_value' and returns a new
sys_training_env object.
"""
function set_sys_training_env_val(env::sys_training_env, param::String,
        new_value::Any)
    labels = ["f", "h", "obs_map", "n", "t0", "tf_tr", "ts", "tolerances",
        "d_samples", "data", "batch_indexes", "mxs"];

    arguments_array = [env.f, env.h, env.obs_map, env.n, env.t0, env.tf_tr,
        env.ts, env.tolerances, env.d_samples, env.data, env.batch_indexes,
        env.mxs];
    for i = 1:1:length(labels)
        if labels[i] == param
            arguments_array[i] = new_value;
        end
    end

    a = arguments_array;
    new_env = sys_training_env(a[1], a[2], a[3], a[4], a[5], a[6], a[7],
        a[8], a[9], a[10], a[11], a[12]);
    return new_env;
end

"""
    set_sysobs_training_env_val(env::sys_training_env, param::String,
        new_value::Any)

Sets the 'param' field of 'env' to 'new_value' and returns a new
sysobs_training_env object.
"""
function set_sysobs_training_env_val(env::sysobs_training_env, param::String,
        new_value::Any)
    labels = ["f", "obs_map", "n", "t0", "tf_tr", "ts", "tolerances",
        "d_samples", "data"];

    arguments_array = [env.f, env.obs_map, env.n, env.t0, env.tf_tr, env.ts,
        env.tolerances, env.d_samples, env.data];
    for i = 1:1:length(labels)
        if labels[i] == param
            arguments_array[i] = new_value;
        end
    end

    a = arguments_array;
    new_env = sysobs_training_env(a[1], a[2], a[3], a[4], a[5], a[6], a[7],
        a[8], a[9]);
    return new_env;
end

"""
    set_fk_training_val(env::fk_training_env, param::String,
        new_value::Any)

Sets the 'param' field of 'env' to 'new_value' and returns a new fk_training_env
object.
"""
function set_fk_training_val(env::fk_training_env, param::String,
        new_value::Any)
    labels = ["fd_kin", "tf_train", "data"];

    arguments_array = [env.fd_kin, env.tf_train, env.data];
    for i = 1:1:length(labels)
        if labels[i] == param
            arguments_array[i] = new_value;
        end
    end

    a = arguments_array;
    new_env = fk_training_env(a[1], a[2], a[3]);
    return new_env;
end

"""
    set_freq_training_val(env::freq_training_env, param::String,
        new_value::Any)

Sets the 'param' field of 'env' to 'new_value' and returns a new
freq_training_env object.
"""
function set_freq_training_val(env::freq_training_env, param::String,
        new_value::Any)
    labels = ["s", "t0", "ts", "tf_tr", "y_samples"];

    arguments_array = [env.s, env.t0, env.ts, env.tf_tr, env.y_samples];
    for i = 1:1:length(labels)
        if labels[i] == param
            arguments_array[i] = new_value;
        end
    end

    a = arguments_array;
    new_env = freq_training_env(a[1], a[2], a[3], a[4], a[5]);
    return new_env;
end
