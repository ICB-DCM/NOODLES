classdef SubproblemArc < noodles.NoodleSubproblem
    % Adaptive Regularization using Cubics, based on [Adaptive cubic
    % regularization methods for unconstrained optimization. Part 1:
    % motivation, convergence and numerical results. Cartis, Gould, Toint.
    % 2007-2009]
    
    properties ( GetAccess = 'public', SetAccess = 'private' )
        sigma;
        ratio;
        hvp_fun;
    end
    
    methods
        
        function this = SubproblemCr(options_in)
            if nargin < 1
                options_in = struct();
            end
            
            this.options = noodles.SubproblemArc.get_options(options_in);
        end
        
        function init(this, noodle_problem)
            init@noodles.NoodleSubproblem(this, noodle_problem);
            this.sigma = this.options.sigma0;
            this.hvp_fun = noodle_problem.hvp_fun;
        end
        
        function update(this, state)
            % update fval, grad, hess
            update@noodles.NoodleSubproblem(this, state);
        end
        
        function solve(this)
            % minimize m(s) = g'*s + 1/2*s'*H's + 1/3*sigma*|s|^3
            
            % question: is it possible to solve the problem
            % sufficiently accurate without using the full hessian, but
            % only hessian-vector products? and is this more reliable and
            % faster than using e.g. sr1 approximations for the hessian?
            
            % TODO
        end
        
        function accept_step = evaluate(this, fval_new)
            
            % compute prediction ratio
            fval_diff = this.fval - fval_new;
            m = this.fval ...
                + this.grad'*this.step ...
                + 1/2*this.step'*this.hess*this.step ...
                + 1/6*this.sigma * norm(this.step, 2)^3;
            
            pred_diff = this.fval - m;
            this.ratio = fval_diff / pred_diff;
            
            % accept anyway
            accept_step = fval_new < this.fval;
        end
        
        function handle_accept_step(this, accept_step)
            if ~accept_step
                this.sigma = 2*this.sigma;
            else
                if this.ratio >= this.options.eta_2
                    this.sigma = max([0.5*this.sigma, 1e-10]);
                elseif this.ratio <= this.options.eta_1
                    this.sigma = 2*this.sigma;
                end
            end
        end
        
    end
    
    methods (Static)
        
        function options = get_options(options_in)
            options = struct();
            options.epsilon = 1e-5;
            options.sigma0  = 1;
            options.theta   = 1e-4;
            options.eta_1   = 0.1;
            options.eta_2   = 0.9;
            
            % fill from input
            cell_fieldnames = fieldnames(options);
            cell_fieldnames_in = fieldnames(options_in);
            
            for jf = 1:length(cell_fieldnames_in)
                fieldname = cell_fieldnames_in{jf};
                if ~any(strcmp(cell_fieldnames,fieldname))
                    error(['Options field ' fieldname ' does not exist.']);
                end
                options.(fieldname) = options_in.(fieldname);
            end
            
        end
        
    end
end

