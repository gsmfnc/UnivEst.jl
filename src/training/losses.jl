"""
    loss_fd(p)

Loss function to estimate forward kinematics.
"""
function loss_fd(p)
    N = Int(round((SUPPENV.tf_train - SUPPENV.fd_kin.t0) /
        SUPPENV.fd_kin.ts)) + 1;

    vloss = 0;
    for i = 1:1:N
        vloss = vloss + sum(abs.(SUPPENV.data[i, :] -
            SUPPENV.fd_kin.forward_kinematics(0, p,
                (i - 1) * SUPPENV.fd_kin.ts)));
    end

    return vloss;
end
