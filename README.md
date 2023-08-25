# Neural_SVEs
Deep learning model to learn the dynamics of stochastic Volterra equations (SVEs) which are a generalization of stochastic differential equations (SDEs).

# Files
The main file to perform neural SVEs is `neural_sve.py`. The configurations therefore can be customized in `configurations.yaml`.

You have the choice to generate and learn data from the disturbed pendulum, Ornstein-Uhlenbeck, or Rough Heston SVE. For other SVEs, simply insert their coefficients after line 110 of `neural_sve.py` and adapt `configurations.yaml`.

As benchmark algorithms, neural SDEs (see \[Kid22\]) and DeepONet (see \[LJP21\]) are implemented in `nsde.py` and `deepONet.py`:  
- `nsde.py` is basically the same file as `neural_sve.py` just witout the ability of the neural structure to learn the dynamics of the kernels in the SVE. It uses the same `configuratons.yaml` file for the configs (note that the variable `hidden_states_kernels` simply gets ignored)
- `deepONet.py` uses `configurations_deepONet.yaml` for its configs. To run `deepONet.py` you must have installed the [`deepxde`](https://github.com/lululxvi/deepxde) package

Note that `nsde.py` and `deepONet.py` are only implemented for dimension `dim=1`.


### References
\[Kid22\] Patrick Kidger, On neural differential equations, 2022.

\[LJP21\] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Karniadakis, Learning nonlinear
operators via deeponet based on the universal approximation theorem of operators, Nature
Machine Intelligence 3 (2021), 218–229.
