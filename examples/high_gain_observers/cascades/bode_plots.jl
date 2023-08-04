using LaTeXStrings

struct bode_infos
    mag
    phase
    w
end

hgo = bode_infos(mag1, phase1, w1);
mcascade = bode_infos(mag2, phase2, w2);
cascade = bode_infos(mag3, phase3, w3);

function bode_diagram(hgo, mcascade, cascade)
    mag1 = hgo.mag;
    phase1 = hgo.phase;
    w1 = hgo.w;
    mag2 = mcascade.mag;
    phase2 = mcascade.phase;
    w2 = mcascade.w;
    mag3 = cascade.mag;
    phase = cascade.phase;
    w3 = cascade.w;

    p1 = plot(w1, 20 * log10.(mag1[1, :, :])',
        xaxis = :log,
        label = L"W_{\hat\xi_1,y}",
        linecolor = :black);
    plot!(w2, 20 * log10.(mag2[1, :, :])',
        xaxis = :log,
        label = L"W_{z_{1,1},y}",
        linecolor = :blue);
    plot!(w3, 20 * log10.(mag3[1, :, :])',
        xaxis = :log,
        label = L"W_{\bar z_{1,1},y}",
        linecolor = :green);

    p2 = plot(w1, 20 * log10.(mag1[2, :, :])',
        xaxis = :log,
        label = L"W_{\hat\xi_2,y}",
        linecolor = :black);
    plot!(w2, 20 * log10.(mag2[2, :, :])',
        xaxis = :log,
        label = L"W_{z_{1,2},y}",
        linecolor = :blue);
    plot!(w2, 20 * log10.(mag2[3, :, :])',
        xaxis = :log,
        label = L"W_{z_{2,1},y}",
        linecolor = :green);
    plot!(w3, 20 * log10.(mag3[2, :, :])',
        xaxis = :log,
        label = L"W_{\bar z_{1,2},y}",
        linestyle = :dashdot,
        linecolor = :blue);
    plot!(w3, 20 * log10.(mag3[3, :, :])',
        xaxis = :log,
        label = L"W_{\bar z_{2,1},y}",
        linecolor = :green,
        linestyle = :dashdot,
        legend = :topleft);

    p3 = plot(w1, 20 * log10.(mag1[3, :, :])',
        xaxis = :log,
        label = L"W_{\hat\xi_3,y}",
        linecolor = :black);
    plot!(w2, 20 * log10.(mag2[4, :, :])',
        xaxis = :log,
        label = L"W_{z_{2,2},y}",
        linecolor = :blue);
    plot!(w2, 20 * log10.(mag2[5, :, :])',
        xaxis = :log,
        label = L"W_{z_{3},y}",
        linecolor = :green);
    plot!(w3, 20 * log10.(mag3[4, :, :])',
        xaxis = :log,
        label = L"W_{\bar z_{2,2},y}",
        linestyle = :dashdot,
        linecolor = :blue,
        legend = :topleft);

    return p1, p2, p3;
end
