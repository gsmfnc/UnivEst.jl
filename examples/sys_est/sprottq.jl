#additional example where h(x) depends on some unknown parameters

u0 = [2.0, 0.0, 0.0];
p = [0.9, 0.4];

y(u) = u[2];
sys(u, p, t) = [
    p[1] - u[2]
    u[3] + p[2]
    u[1] * u[2] - u[3]
];
O(u, p, t) = [
    u[2]
    u[3] + p[2]
    u[1] * u[2] - u[3]
];
