classdef SubproblemCr < noodles.NoodleSubproblem
    % Cubic regularization with exact solution of the subproblem.
    
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
        
        function this = SubproblemCr(options_in)
            if nargin < 1
                options_in = struct();
            end
            
            this.options = noodles.SubproblemCr.get_options(options_in);
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
            % version 1: solve exactly by separating the problem, using the
            % full hessian
            
            % solve rotated trust-region subproblem
            for j = 1:this.dim
                c0 = 0;
                c1 = this.b(j);
                c2 = this.D(j,j)/2;
                c4 = this.sigma/6;
                this.y(j) = noodles.SubproblemCr.min_poly(c0,c1,c2,0,c4,-inf,inf);
            end
            % compute step
            this.step = this.Q*this.y;
            this.stepnorm = norm(this.step, 2);
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
                this.sigma = this.options.gamma_1*this.sigma;
            else
                if this.ratio >= this.options.eta_2
                    this.sigma = max([this.options.gamma_2*this.sigma, this.options.sigma_min]);
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
        
        function z = min_poly(c0, c1, c2, c3, c4, DeltaNeg, DeltaPos)
            % compute the minimum of the function h in [DeltaNeg,DeltaPos]
            
            h = @(z) c0 + c1*z + c2*z^2 + c3*z^3 + c4*abs(z)^3;
            zs = [DeltaNeg,DeltaPos];
            argmin = @(zs, fun) noodles.SubproblemCr.argmin(zs, fun);
            if c2 == 0 && c3 == 0 && c4 == 0
                z = argmin(zs,h);
            elseif c2 ~= 0 && c3 == 0 && c4 == 0
                zcrt = -c1/(2*c2);
                if DeltaNeg < zcrt && zcrt < DeltaPos
                    zs = [zs zcrt];
                end
                z = argmin(zs,h);
            elseif c3 ~= 0 && c4 == 0
                xi = 4*c2^2-12*c3*c1;
                if xi < 0
                    z = argmin(zs,h);
                else
                    zcrt1 = (-2*c2-sqrt(xi))/(6*c3);
                    zcrt2 = (-2*c2+sqrt(xi))/(6*c3);
                    if DeltaNeg < zcrt1 && zcrt1 < DeltaPos
                        zs = [zs zcrt1];
                    end
                    if DeltaNeg < zcrt2 && zcrt2 < DeltaPos
                        zs = [zs zcrt2];
                    end
                    z = argmin(zs,h);
                end
            elseif c4 ~= 0
                zneg = noodles.SubproblemCr.min_poly(c0,c1,c2,c3-c4,0,DeltaNeg,0);
                zpos = noodles.SubproblemCr.min_poly(c0,c1,c2,c3+c4,0,0,DeltaPos);
                z = argmin([zneg,zpos],h);
            end
            
        end
        
        function z = argmin(zs, fun)
            % value z in zs such that fun(z) is minimal among all z in zs
            
            n = size(zs,2);
            z = zs(1);
            fval = fun(z);
            for j=2:n
                z_new = zs(j);
                fval_new = fun(z_new);
                if ~isfinite(fval) || (fval_new < fval && isfinite(fval_new))
                    z = z_new;
                    fval = fval_new;
                end
            end
            
        end
        
    end
end

