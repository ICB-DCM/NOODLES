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
        
        function this = SubproblemArc(options_in)
            if nargin < 1
                options_in = struct();
            end
            
            this.options = noodles.SubproblemArc.get_options(options_in);
            error("This class is not implemented yet.");
        end
        
        function init(this, noodle_problem)
            init@noodles.NoodleSubproblem(this, noodle_problem);
            this.sigma = this.options.sigma0;
            this.hvp_fun = noodle_problem.hvp_fun;
        end
        
        function solve(this)
            this.step = SubproblemArc.arc_glrt(this.grad, this.hess, this.sigma);
            this.stepnorm = norm(step, 2);
        end
        
        function accept_step = evaluate(this, fval_new)
            
            % compute prediction ratio
            fval_diff = this.fval - fval_new;
            m = this.fval ...
                + this.grad'*this.step ...
                + 1/2*this.step'*this.hess*this.step ...
                + 1/3*this.sigma * norm(this.step, 2)^3;
            
            pred_diff = this.fval - m;
            this.ratio = fval_diff / pred_diff;
            
            % accept anyway
            accept_step = fval_new < this.fval;
        end
        
        function handle_accept_step(this, accept_step)
            if ~accept_step
                this.sigma = this.options.gamma_1*this.sigma;
            else
                if this.ratio >= this.options.eta_2
                    this.sigma = max(min(this.sigma,this.gradnorm),this.options.sigma_min);
%                     this.sigma = max([this.options.gamma_2*this.sigma, this.options.sigma_min]);
                elseif this.ratio <= this.options.eta_1
                    this.sigma = this.options.gamma_1*this.sigma;
                end
            end
        end
        
    end
    
    methods (Static)
        
        function options = get_options(options_in)
            options = struct();
            options.sigma0      = 10;           % initial regularization
            options.eta_1       = 0.25;         % threshold for bad model
            options.eta_2       = 0.75;         % threshold for good model
            options.gamma_1     = 2;            % factor for bad model
            options.gamma_2     = 0.5;          % factor for good model
            options.sigma_min   = sqrt(eps);    % minimum regularization
            
            
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
        
        function [s] = arc_glrt(grad, hess, sigma)
           s = arg_newton(A, b, sigma); 
        end
        
        function [s, lambda] = arc_newton(A, b, sigma)
            
        end
        
    end
end

