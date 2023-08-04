function compute_errors(hgo_type, xi, hxi)
    N = size(xi, 2);
    xitot = zeros(size(xi, 1), N);# + 2, N);
    for i = 1:1:N
        xitot[1, i] = xi[1, i];
        xitot[2, i] = xi[2, i];
        xitot[3, i] = xi[3, i];
        #xitot[4, i] = -2.02 * xi[3, i] + xi[2, i]^2 - xi[1, i];
        #xitot[5, i] = -2.02 * xitot[4, i] + 2 * xi[2, i] * xi[3, i] - xi[2, i];
    end

    if hgo_type == UnivEst.CLASSICALHGO
        return (xitot - hxi)'
    end

    if hgo_type == UnivEst.M_CASCADE
        xitote = [xitot[1, :] xitot[2, :] xitot[2, :] xitot[3, :]];
        xitote = [xitote xitot[3, :]]';# xitot[4, :] xitot[4, :] xitot[5, :]];
        #xitote = [xitote xitot[5, :]]';
        return (xitote - hxi)'
    end

    if hgo_type == UnivEst.CASCADE
        xitote = [xitot[1, :] xitot[2, :] xitot[2, :] xitot[3, :]]'; #];
        #xitote = [xitote xitot[3, :] xitot[4, :] xitot[4, :] xitot[5, :]]';
        return (xitote - hxi)'
    end
end
