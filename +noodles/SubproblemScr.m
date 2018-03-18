classdef SubproblemScr < noodles.NoodleSubproblem
    % Cubic regularization with exact solution of the separable subproblem.
    
    properties ( GetAccess = 'public', SetAccess = 'private' )
        % regularization
        sigma;
        
        % prediction ratio
        ratio;
        
        % for Schur decomposition
        Q;
        D;
        b;
        
        % space for solution of rotated problem
        y;
    end
    
    methods
        
        function this = SubproblemScr(options_in)
            if nargin < 1
                options_in = struct();
            end
            
            this.options = noodles.SubproblemScr.get_options(options_in);
        end
        
        function init(this, noodle_problem)
            init@noodles.NoodleSubproblem(this, noodle_problem);
            this.sigma = this.options.sigma0;
            this.y  = nan(this.dim,1);
        end
        
        function update(this, state)
            % update fval, grad, hess
            update@noodles.NoodleSubproblem(this, state);
            
            % update Q, D, b
            [this.Q,this.D]   = schur(this.hess);
            this.b  = this.Q'*this.grad;
        end
        
        function solve(this)
            % minimize m(s) = g'*s + 1/2*s'*H's + 1/3*sigma*|s|^3
            % solve exactly by separating the problem, using the
            % full hessian and a variable inf-norm
            
            % solve rotated trust-region subproblem
            for j = 1:this.dim
                c0 = 0;
                c1 = this.b(j);
                c2 = this.D(j,j)/2;
                c4 = this.sigma/6;
                this.y(j) = noodles.Utils.min_poly3(c0,c1,c2,0,c4,-inf,inf);
            end
            % compute step
            this.step = this.Q*this.y;
            this.stepnorm = norm(this.y, inf);
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
            accept_step = isnan(this.fval) || fval_new < this.fval;
        end
        
        function handle_accept_step(this, accept_step)
            if ~accept_step
                this.sigma = this.options.gamma_1*this.sigma;
            else
                if this.ratio >= this.options.eta_2
                    this.sigma = max(min(this.sigma,this.gradnorm),1e-16);
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
            options.epsilon     = 1e-5;
            options.sigma0      = 1;     % initial regularization
            options.eta_1       = 0.1;   % threshold for bad model
            options.eta_2       = 0.9;   % threshold for good model
            options.gamma_1     = 2;     % factor for bad model
            options.gamma_2     = 0.5;   % factor for good model
            options.sigma_min   = 1e-10; % minimum regularization
            
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

