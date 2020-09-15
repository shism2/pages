---
toc: false
layout: post
title: An unusual ODE-solver
tags: [NCP, wormnet]
image: "images/wormnet/tw.png"

---

This is the third part in a series explaining our recent paper titled XXX.
In the previous part, we have defined an ODE-based neuron model, together with a non-linear synapse model.
Here, we will derive an ODE-solver specifically created to deal with the issues of our model.

## Stiff ordinary differential equations
 
One might say that this is enough to build a recurrent neural network (RNN). People familiar with David Duvenaud's research group's work [^1] might argue that we could simply put the ODE system into an off-the-shelf ODE-solver to obtain a solution at the required timesteps. 
Packages like torchdiffeq [^2] provide a set of powerful ODE-solvers like the Kunge-Kutta [^3] or the more sophisticated Dormand-Prince method [^4], which is even the default in Matlab [^5].
However, there is one issue with the differential equations: They realize a phenomenon known as stiff equations [^6].
For instance, if we simulate the unremarkable and linear ODE 

$$ \frac{d x}{d t} = \begin{pmatrix} 1 & 1\\ -20.5 & -21.5 \end{pmatrix} x $$

with a simple fixed-step solver, we might get something like 

![award]({{ site.baseurl }}/images/wormnet/diffeq/seq/explicit.gif "The explicit Euler method applied to the linear ODE above")

Such a stiffness usually occurs when there are two antagonistic forces applied to a single variable. One force tries to increase the variable, while the other one wants to decrease it. In the analytical solution, both forces converge to a stable equilibrium. However, in a discrete-time numerical approximation, the two forces push the variable above and below the equilibrium. If the magnitude of this "pushing around" increases, the resulting numerical simulation becomes unstable, as we observed above. 

Dynamic stepsize solvers, like the Dormand-Prince method, deal with this issue by making the simulation grid finer if such divergence is detected. However, this solution comes at a very high computational cost, which we want to avoid.

Implicit ODE-solving methods provide a more elegant solution.
Standard explicit methods like the explicit Euler, Runge-Kutta, and the Dormand-Prince approach all simulate a differential equation like

$$ x(t+T) = x(t) + T \cdot \text{solver\_magic}(x(t)) $$

In the simplest case, i.e., the explicit Euler, $$ \text{solver\_magic} $$ is simply the right-hand-side of the given ODE $$ \frac{d x}{d t} = f(x) $$.
The idea of implicit methods is define the numerical solution in an implcit equation,

$$ x(t+T) = x(t) + T \cdot \text{solver\_magic}(x(t+t)) $$

The simplest case is the implicit Euler, which is defined as

$$ x(t+T) = x(t) + T \cdot f(x(t+t)) $$

If our ODE is linear, like the one illustrated above, we can analytically solve this equation and get

$$ x(t+T) = (T\cdot A-E)^{-1} x(t) $$

which behaves smoothly and stable when run with the same stepsize as the figure above.

![award]({{ site.baseurl }}/images/wormnet/diffeq/euler_implicit.png "Comparision of the explicit and implicit Euler method")

There is one catch though: implicit methods require solving a possible non-linear equation. We have two options for solving this equation: Analytically or numerically.
Solving the equation analytically as we have done in the implicit Euler method for the linear ODE above is not possible for our model due to the non-linear transcendental sigmoid function.
Solving the equation numerically is also not a legitimate option, as we would have to solve it every timestep, resulting in a high computational cost. 

## A hybrid solution

Our solution to this dilemma is a hybrid implicit/explicit ODE-solver. In our ODE model, the stiff parts, i.e., the terms that make the variables over- and undershoot the equilibrium, occur only linearly in the equation. 
Consequently, we can apply the explicit Euler method for computing the non-linear synapse activations and use the super-stable implicit Euler method for remaining linear ODE.
Mathematically, we discretize the ODE 

$$ \frac{d x}{d t} = f(x) $$

by the approach 

$$ x(t+T) = x(t) + T*f(x \mapsto x(t)/x(t+t)) $$

where we substitute $$x$$ by $$x(t)$$ in the non-linear and $$x$$ by $$ x(t+T) $$ in the linear occruances of $$x$$ in $$f$$.
After an extensive amount of re-writing, we can solve the ODE for (x+T):

From the equation of our hybrid solver, we can infer one important stability property. If we want to guarantee that we never divide by zero, we have to make sure that Gleak, Gsyn is non-negative, and Cm positive. This will be important later when we train these parameters.

Now that we have overcome the stability issue, we need to talk about precision. Just because we have some stable discretization does not imply we accurately approximate the underlying ODE. Especially, Euler methods are known not to be the most precise solver choices.
Our approach to boost precision is to stack multiple ODE-solver steps into a single RNN computation step. 

![award]({{ site.baseurl }}/images/wormnet/stacked_rnn.png "We sequentially stack ODE-solver step into a single RNN-step to increase numerical precision")

This does not guarantee that we always approximate the system in all conditions with perfect precision, but that's a tradeoff we can live with.

## Conclusion

We showed how to obtain a stable and decently accurate discretization of our neural network model. In the next part, we will put this solution into a deep learning framework and learn its parameters from training data.

## References

[^1]: Rubanova et al. NeurIPS 2019. [Latent ordinary differential equations for irregularly-sampled time series](https://papers.nips.cc/paper/8773-latent-ordinary-differential-equations-for-irregularly-sampled-time-series.pdf)
[^2]: [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
[^3]: [Wolfram Mathworld - Runge-Kutta methods](https://mathworld.wolfram.com/Runge-KuttaMethod.html)
[^4]: Dormand and Prince 1980, *A family of embedded Runge-Kutta formulae*, J. Comp. Appl. Math., Vol. 6
[^5]: [Matlab's ode45](https://www.mathworks.com/help/matlab/ref/ode45.html)
[^6]: [mathworks.com - Stiff Differential Equations](https://www.mathworks.com/company/newsletters/articles/stiff-differential-equations.html)