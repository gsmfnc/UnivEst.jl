# Linear systems

A dynamical system is linear if it is described by a set of equations of the
following form
```math
    \dot x=Ax+Bu, \\
    y=Cx+Du,
```
where
$x\in R^n$ is the state vector, $A\in R^{n\times n}$,
$u\in R^m$ is the input vector, $B\in R^{n\times m}$,
$y\in R^p$ is the output vector, $C\in R^{p\times n}$
and $D\in R^{p\times m}$.
In order to simplify the exposition, assume that $p=1$, i.e. the output is
scalar, and $D=0$.
A linear system is said to be in \emph{observability canonical form} if
```math
    A=
    \left[
    \begin{array}{ccc}
        0_{(n-1)\times1} & \vline & I_{(n-1)\times(n-1)} \\
        \hline
        \cdots & k^\top & \cdots
    \end{array}
    \right],\quad
    C=\left[\begin{array}{cccc}1&0&\cdots&0\end{array}\right],
```
where $k\in R^n$ is a vector of constants.
