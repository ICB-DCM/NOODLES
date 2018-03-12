classdef SubproblemScmcr < noodles.NoodleSubproblem
    % Method for unconstrained optimization via a cubic-regularization approach
    % using a separable cubic model. Based on [Cubic-regularization counterpart
    % of a variable-norm trust-region method for unconstrained minimization.
    % Martínez, Raydan. 2015].
    
    properties ( GetAccess = 'public', SetAccess = 'private' )
        Q;
        D;
        b;
        delta;
        rho;
        sigma;
        y; % space for solution of rotated problem
        
        sufficient_decrease;
        flag_initial;
    end
    
    methods
        
        function this = SubproblemScmcr(options_in)
            if nargin < 1
                options_in = struct();
            end
            
            this.options = noodles.SubproblemScmcr.get_options(options_in);
        end
        
        function init(this, noodle_problem)
            init@noodles.NoodleSubproblem(this, noodle_problem);
            this.Q      = nan(this.dim,this.dim);
            this.D      = nan(this.dim,this.dim);
            this.b      = nan(this.dim,1);
            this.delta  = this.options.delta0;
            this.sigma  = this.options.sigma0;
            if length(this.options.rho0) == 1
                this.rho    = this.options.rho0 * ones(this.dim, 1);
            else
                this.rho    = this.options.rho0(:);
            end
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
                c4 = this.sigma/6;
                this.y(j) = noodles.SubproblemScmcr.min_poly(c0,c1,c2,c3,c4,-this.delta,this.delta);
            end
            
            % compute step
            this.step = this.Q*this.y;
            this.stepnorm = norm(this.step, 2);
        end
        
        function accept_step = evaluate(this, fval_new)
            
            this.sufficient_decrease = fval_new < this.fval - this.options.alpha * sum(abs(this.y).^3);
            
            accept_step = fval_new < this.fval;
        end
        
        function handle_accept_step(this, accept_step)
            
            if ~accept_step
                
                this.sigma = this.options.eta_s * this.sigma;
                this.sigma = max([this.sigma, this.options.sigma_min]);
                
            else % step was accepted
                
                if true || this.sufficient_decrease
                    this.sigma = this.options.eta_v * this.sigma;
                    this.sigma = max([this.sigma, this.options.sigma_min]);
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
            options.delta0      = 2; % 2
            options.eta_v       = 0.5;
            options.eta_s       = 10; % 10
            options.sigma0      = 0; % 0
            options.sigma_min   = 0.1;
            options.alpha       = 1e-4;
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
        
        function z = min_poly(c0, c1, c2, c3, c4, DeltaNeg, DeltaPos)
            % compute the minimum of the function h in [DeltaNeg,DeltaPos]
            
            h = @(z) c0 + c1*z + c2*z^2 + c3*z^3 + c4*abs(z)^3;
            zs = [DeltaNeg,DeltaPos];
            argmin = @(zs, fun) noodles.SubproblemScmcr.argmin(zs, fun);
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
                zneg = noodles.SubproblemScmcr.min_poly(c0,c1,c2,c3-c4,0,DeltaNeg,0);
                zpos = noodles.SubproblemScmcr.min_poly(c0,c1,c2,c3+c4,0,0,DeltaPos);
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
                if fval_new < fval
                    z = z_new;
                    fval = fval_new;
                end
            end
            
        end
        
    end
end

