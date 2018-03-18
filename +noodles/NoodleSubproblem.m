classdef (Abstract) NoodleSubproblem < handle
    % The NoodleSubproblem class administers the local search for the next
    % evaluation point. To customize the subproblem solution, derive from
    % this class and implement the various methods used in
    % NoodleProblem.run_optimization(), at least the abstract ones.
    
    properties ( GetAccess = 'public', SetAccess = 'protected' )
        % subproblem options, as defined in get_options
        options;
        
        % problem dimension
        dim;
        
        % boundaries: lower bound
        lb;
        
        % boundaries: upper bound
        ub;
        
        % current parameter
        x;
        
        % current function value
        fval;
        
        % current gradient
        grad;
        
        % current hessian
        hess;
        
        % current gradient norm
        gradnorm;
        
        % step proposed for current subproblem
        step;
        
        % norm of step
        stepnorm;
    end
    
    methods
        
        function this = NoodleSubproblem()
            % Constructor, called implicitly. Does nothing.
        end
        
        function init(this, noodle_problem)
            % Initialize/clear the subproblem, pull required data from the
            % NoodeProblem.
            %
            % Input:
            % noodle_problem    : super NoodleProblem instance
            
            this.dim        = noodle_problem.dim;
            this.lb         = noodle_problem.options.lb;
            this.ub         = noodle_problem.options.ub;
            
            % All values initialized empty for efficiency reasons.
            this.fval       = [];
            this.grad       = [];
            this.hess       = [];
            this.gradnorm   = [];
            this.step       = [];
            this.stepnorm   = [];
        end
        
        function update(this, state)
            % Update values after state has changed.
            %
            % Input:
            % state   : NoodleState from the super NoodleProblem
            
            this.x          = state.x;
            this.fval       = state.fval;
            this.grad       = state.grad;
            this.hess       = state.hess;
            this.gradnorm   = state.gradnorm;
        end
        
        function accept_step = evaluate(this, fval_new)
            % Determine whether the beforehand computed step should be
            % accepted. Specialized subproblems might define more complex
            % rules on accepting a new step, in order to reduce the number
            % of derivative computations.
            %
            % Input:
            % fval_new  : objfun(x+step)
            
            accept_step = fval_new < this.fval;
        end
        
    end
    
    methods (Abstract)
        
        % Solve the subproblem and update the step variable so that x+step
        % indicates the predicted best next evaluation point.
        solve(this)
        
        % Update internal state according to whether the last step was
        % accepted or not. Only here relevant variables should be changed,
        % not in evaluate(). The reason for this subdivision is that
        % numerical issues can occur when in the outer routine the
        % derivatives are computed (e.g. they might contain nans or infs).
        % So, a step is accepted only if this was recommended in evaluate()
        % and all derivatives could be computed.
        handle_accept_step(this, accept_step)
        
    end
    
    methods (Static, Abstract)
        
        % Create an options struct filled with default values, overwrite
        % with user input and check for validity.
        options = get_options(options_in)
        
    end
end

