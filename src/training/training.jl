################################################################################
#######################EXPORTED FUNCTIONS#######################################
################################################################################
"""
    fd_kin_training(fd_kin::forward_kinematics, data::Matrix{Float64},
        tfs::Vector{Float64}, its::Int, estp::Vector{Float64};
        opt::Function = Adam(1e-02))
"""
function fd_kin_training(fd_kin::forward_kinematics, data::Matrix{Float64},
        tfs::Vector{Float64}, its::Int, estp::Vector{Float64};
        opt = Adam(1e-02))
    global SUPPENV
    SUPPENV = fk_training_env(fd_kin, 0.0, data);

    @time begin
    for i = 1:1:length(tfs)
        SUPPENV = set_fk_training_val(SUPPENV, "tf_train", tfs[i]);

        pinit = ComponentArray(estp);
        adtype = Optimization.AutoZygote();
        optf = Optimization.OptimizationFunction((x, p) -> loss_fd(x),
            adtype);
        optprob = Optimization.OptimizationProblem(optf, pinit);

        @time begin
        res = Optimization.solve(optprob,
                                opt,
                                maxiters = its);
        end
        estp = res.u;
        println(estp)
    end
    end
    return estp;
end
