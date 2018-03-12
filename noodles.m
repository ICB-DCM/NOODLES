function results = noodles(objfun, x0, options)
% noodles is the function to call for optimization with Noodles.
%
% Input:
% objfun      : objective function to be minimized
% x0          : starting point
% options     : NoodleOptions, containing all optimization options
%
% Output:
% results     : NoodleResults, containing all relevant optimization results

% handle options
if nargin < 3
    options = struct();
end

% initialize problem
problem = noodles.NoodleProblem(objfun, x0, options);

% run optimization
results = problem.run_optimization();

end