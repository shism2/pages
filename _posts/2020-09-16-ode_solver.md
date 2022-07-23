---
toc: false
layout: post
title: Neural Circuit Policies (Part 3) - An Unusual ODE-solver
tags: [NCP, wormnet]
image: "images/thumb_sized/ode.png"

---

This is the third part in a series explaining our recent paper titled [Neural circuit policies enabling auditable autonomy](https://rdcu.be/b8sEo).
Here, we will derive an ODE-solver specifically created to deal with the numerical issues of our LTC neural network model.

- [Part 1: Introduction]({{ site.baseurl }}/2020/09/14/wormnet1.html)
- [Part 2: A Hundred-Year-Old Neuron Model]({{ site.baseurl }}/2020/09/15/the_model.html)
- **Part 3: An Unusual ODE-solver**
- [Part 4: Training RNNs is Difficult]({{ site.baseurl }}/2021/02/18/rnn.html)
- [Part 5: Sparsity in Machine Learning]({{ site.baseurl }}/2021/03/08/sparsity.html)
- [Part 6: Pros and Cons of Neural Circuit Policies]({{ site.baseurl }}/2021/06/29/procon.html)

## Stiff ordinary differential equations
 
In the previous part, we have defined an ODE-based neuron model, together with a non-linear synapse model.
One might say that this is already enough to build a recurrent neural network (RNN). People familiar with David Duvenaud's research group's work [^1] might argue that we could simply put the ODE system into an off-the-shelf ODE-solver to obtain a solution at the required timesteps. 
Packages like torchdiffeq [^2] provide a set of powerful ODE-solvers like the Kunge-Kutta [^3] or the more sophisticated Dormand-Prince method [^4], which is even the default in Matlab [^5].
However, there is one issue with the differential equations: They realize a phenomenon known as stiff equations [^6].
For instance, if we simulate the unremarkable linear ODE 

$$ \frac{d x}{d t} = \begin{pmatrix} 0 & 1\\ -20.5 & -21.5 \end{pmatrix} x $$

with a simple fixed-step solver, we might get something like 

![explosion]({{ site.baseurl }}/images/wormnet/diffeq/seq/explicit.gif "The explicit Euler method applied to the linear ODE above")

Note that the explosion happends despite a relatively fine (small) time-step of the ODE solver.
Such a stiffness usually occurs when there are two antagonistic forces applied to a single variable. One force tries to increase the variable, while the other one wants to decrease it. In the analytical solution, both forces converge to a stable equilibrium. However, in a discrete-time numerical approximation, the two forces push the variable above and below the equilibrium. If the magnitude of this "pushing around" increases, the resulting numerical simulation becomes unstable, as we observed above. 

Dynamic stepsize solvers, like the Dormand-Prince method, deal with this issue by making the simulation grid finer if such divergence is detected. However, this solution comes at a very high computational cost, which we want to avoid.

**Implicit ODE-solving methods** provide a more elegant solution.
Standard explicit methods like the explicit Euler, Runge-Kutta, and the Dormand-Prince approach all simulate a differential equation like

$$ x(t+\Delta) = x(t) + \Delta \cdot \text{solver\_magic}(x(t)) $$

In the simplest case, i.e., the explicit Euler, $$ \text{solver\_magic} $$ is simply the right-hand-size of the given ODE $$ \frac{d x}{d t} = f(x) $$ evaluated at $$x=x(t)$$:

$$ x(t+\Delta) = x(t) + \Delta \cdot f(x(t)) $$

The idea of implicit methods is define the numerical solution in an implcit equation,

$$ x(t+\Delta) = x(t) + \Delta \cdot \text{solver\_magic}(x(t+t)) $$

The simplest case is the implicit Euler, which is defined as

$$ x(t+\Delta) = x(t) + \Delta \cdot f(x(t+\Delta)) $$

If our ODE is linear, like the one illustrated above, we can analytically solve this equation and get

$$ x(t+\Delta) = (\begin{pmatrix}1 & 0 \\0 & 1 \end{pmatrix} - \Delta\cdot \begin{pmatrix} 0 & 1\\ -20.5 & -21.5 \end{pmatrix})^{-1} x(t) $$

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

$$ x(t+\Delta) = x(t) + \Delta \cdot f(x \mapsto x(t)/x(t+\Delta)) $$

where we substitute $$x$$ by $$x(t)$$ in the non-linear and $$x$$ by $$ x(t+\Delta) $$ in the linear occruances of $$x$$ in $$f$$.
After an extensive amount of re-writing, we can solve the ODE for (x+\Delta):

From the equation of our hybrid solver, we can infer one important stability property. If we want to guarantee that we never divide by zero, we have to make sure that Gleak, Gsyn is non-negative, and Cm positive. This will be important later when we train these parameters.

Now that we have overcome the stability issue, we need to talk about precision. Just because we have some stable discretization does not imply we accurately approximate the underlying ODE. Especially, Euler methods are known not to be the most precise solver choices.
Our approach to boost precision is to stack multiple ODE-solver steps into a single RNN computation step. 

![award]({{ site.baseurl }}/images/wormnet/stacked_rnn.png "We sequentially stack ODE-solver step into a single RNN-step to increase numerical precision")

This does not guarantee that we always approximate the system in all conditions with perfect precision, but that's a tradeoff we can live with.

## Conclusion

We showed how to obtain a stable and decently accurate discretization of our neural network model. 
In the [next part (Training RNN is Difficult)]({{ site.baseurl }}/2021/02/18/rnn.html), we will discuss issues arising the training of RNNs.

## References

[^1]: Rubanova et al. NeurIPS 2019. [Latent ordinary differential equations for irregularly-sampled time series](https://papers.nips.cc/paper/8773-latent-ordinary-differential-equations-for-irregularly-sampled-time-series.pdf)
[^2]: The [torchdiffeq](https://github.com/rtqichen/torchdiffeq) package
[^3]: [Wolfram Mathworld - Runge-Kutta methods](https://mathworld.wolfram.com/Runge-KuttaMethod.html)
[^4]: Dormand and Prince 1980, *A family of embedded Runge-Kutta formulae*, J. Comp. Appl. Math., Vol. 6
[^5]: [Matlab's ode45](https://www.mathworks.com/help/matlab/ref/ode45.html)
[^6]: [mathworks.com - Stiff Differential Equations](https://www.mathworks.com/company/newsletters/articles/stiff-differential-equations.html)
