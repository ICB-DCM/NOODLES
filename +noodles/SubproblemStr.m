classdef SubproblemStr < noodles.NoodleSubproblem
    % Solve the subproblem by a simple separable trust region strategy.

    properties ( GetAccess = 'public', SetAccess = 'private' )
        tr_radius;
        ratio;
        
        % for Schur decomposition
        Q;
        D;
        b;
        
        % space for solution of rotated problem
        y;
    end
    
    methods
        
        function this = SubproblemStr(options_in)
            if nargin < 1
                options_in = struct();
            end
            
            this.options = noodles.SubproblemStr.get_options(options_in);
        end
        
        function init(this, noodle_problem)
            init@noodles.NoodleSubproblem(this, noodle_problem);
            this.tr_radius = 10;
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
            this.stepnorm = norm(this.step, 2);
        end
        
        function accept_step = evaluate(this, fval_new)
            
            % compute prediction ratio
            fval_diff = this.fval - fval_new;
            q = this.fval + this.grad'*this.step + 1/2*this.step'*this.hess*this.step;
            pred_diff = this.fval - q;
            this.ratio = fval_diff / pred_diff;
            
            accept_step = fval_new < this.fval;
        end
        
        function handle_accept_step(this, accept_step)
            if ~accept_step
                this.tr_radius = this.options.gamma_1 * this.tr_radius;
            else
                if this.ratio >= this.options.eta_2 && this.stepnorm >= 0.9 * this.tr_radius
                    this.tr_radius = this.options.gamma_2 * this.tr_radius;
                elseif this.ratio <= this.options.eta_1
                    this.tr_radius = min([this.options.gamma_1 * this.stepnorm, ...
                        this.options.gamma_1 * this.tr_radius]);
                end
            end
        end

    end
    
    methods (Static)
       
        function options = get_options(options_in)
            options = struct();
            options.eta_2 = 0.75;  % threshold for good model
            options.eta_1 = 0.25;  % threshold for bad model
            options.gamma_2 = 2;   % factor for good model
            options.gamma_1 = 0.5; % factor for bad model
            
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
        
        function s = solve_trust(grad, hess, tr_radius)
            s = trust(grad, hess, tr_radius);
        end
        
    end
end

