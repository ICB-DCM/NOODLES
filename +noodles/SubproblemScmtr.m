classdef SubproblemScmtr < noodles.NoodleSubproblem
    % Method for unconstrained optimization via a trust-region approach using a
    % separable cubic model. Based on [Separable cubic modeling and a
    % trust-region strategy for unconstrained minimization with impact in lobal
    % optimization. Martínez, Raydan. 2015].
    
    properties ( GetAccess = 'public', SetAccess = 'private' )
        Q;
        D;
        b;
        delta;
        rho;
        y; % space for solution of rotated problem
        
        ratio;
        flag_initial;
    end
    
    methods
        
        function this = SubproblemScmtr(options_in)
            if nargin < 1
                options_in = struct();
            end
            
            this.options = noodles.SubproblemScmtr.get_options(options_in);
        end
        
        function init(this, noodle_problem)
            init@noodles.NoodleSubproblem(this, noodle_problem);
            this.Q      = nan(this.dim,this.dim);
            this.D      = nan(this.dim,this.dim);
            this.b      = nan(this.dim,1);
            this.delta  = this.options.delta0;
            this.rho    = this.options.rho0 * ones(this.dim, 1);
            this.y      = nan(this.dim,1);
            this.flag_initial = true;
        end
        
        function update(this, state)
            hess_prev = this.hess;
            
            % update fval, grad, hess
            update@noodles.NoodleSubproblem(this, state);
            
            % update Q, D, b
            [this.Q,this.D]   = schur(this.hess);
            this.b  = this.Q'*this.grad;
            
            % update rho inspired by a third-order secant equation
            if ~this.flag_initial
                den = this.Q'*this.step;
                for j = 1:this.dim
                    den_j = den(j);
                    if -this.options.roundoff < den_j && den_j <= 0
                        den(j) = -this.options.roundoff;
                    elseif 0 < den_j && den_j < this.options.roundoff
                        den(j) = this.options.roundoff;
                    end
                end
                this.rho = diag((this.D - this.Q'*hess_prev*this.Q))./den;
                
                % keep rho within bounds
                this.rho = min(max(this.rho, this.options.rho_min), this.options.rho_max);
            end
            
            this.flag_initial = false;
        end
        
        function solve(this)
            % solve rotated trust-region subproblem
            for j = 1:this.dim
                c0 = 0;
                c1 = this.b(j);
                c2 = this.D(j,j)/2;
                c3 = this.rho(j)/6;
                this.y(j) = noodles.Utils.min_poly3(c0,c1,c2,c3,0,-this.delta,this.delta);
            end
            
            % compute step
            this.step = this.Q*this.y;
            this.stepnorm = norm(this.y, inf);
        end
        
        function accept_step = evaluate(this, fval_new)
            
            % compute prediction ratio
            fval_diff = this.fval - fval_new;
            q = this.fval + this.grad'*this.step + 1/2*this.step'*this.hess*this.step;
            pred_diff = this.fval - q;
            this.ratio = fval_diff / pred_diff;
            
            if fval_new < this.fval
                accept_step = true;
            else
                accept_step = false;
            end
            
        end
        
        function handle_accept_step(this, accept_step)
            if ~accept_step
                this.delta = this.options.gamma_d * this.delta;
            else % step was accepted
                if this.ratio >= this.options.eta_v
                    this.delta = this.options.gamma * this.delta;
                elseif this.ratio >= this.options.eta_s
                    this.delta = this.delta;
                else
                    this.delta = this.options.gamma_d * this.delta;
                end
            end
        end
        
    end
    
    methods (Static)
        
        function options = get_options(options_in)
            options = struct();
            options.rho0        = 1;
            options.rho_max     = 1e3;
            options.rho_min     = -options.rho_max;
            options.delta0      = 2;
            options.delta_min   = 0.1;
            options.delta_max   = 1e5;
            options.eta_v       = 0.9; % 0.9
            options.eta_s       = 0.1; % 0.1
            options.gamma       = 2;    % 2
            options.gamma_d     = 0.5;  % 0.5
            options.tol         = 1e-8;
            options.roundoff    = sqrt(eps);
            
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

