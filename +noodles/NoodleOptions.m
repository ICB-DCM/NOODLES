classdef NoodleOptions
    % The NoodleOptions class contains all options that can be made for the
    % algorithm run.
    
    properties
        % contains an instance of a NoodleSubproblem class, to be used for
        % solving the subproblem
        subproblem = noodles.SubproblemTr();
        
        % boundaries: lower bound
        lb  = -inf;
        
        % boundaries: upper bound
        ub  = inf;
        
        % tolerances
        tol_grad        = 1e-6;
        tol_step        = 1e-6;
        tol_fvaldiff    = 1e-6;
        iter_max        = Inf;
        feval_max       = Inf;
        
        % [fval, grad, hess] = derivative_fun(problem, x)
        % use problem.objfun to compute values
        % implemented: objective (use third output), sr1, dfp, bfgs
        derivative_fun   = @noodles.NoodleProblem.objective;
        
        % hessian-vector product function,
        % hvp_fcn: v -> H*v
        hvp_fun = []
        
        % textual output
        % 0: no output
        % 1: output
        verbosity       = 1;
        
    end
    
    methods
        
        function options = NoodleOptions(options_in)
            % Constructor for NoodleOptions.
            %
            % Input:
            % options_in: NoodleOptions instance or struct having as fields
            %             a subset of the NoodleOptions properties.
            %             Validity of all fields is checked.
            
            if isa(options_in, 'noodles.NoodleOptions')
                options = options_in;
            elseif isa(options_in, 'struct')
                fields_in = fields(options_in);
                for jf = 1:length(fields_in)
                    if ~isprop(options, fields_in{jf})
                        error(['Cannot assign options property ' fields_in{jf} ' to NoodleOptions.']);
                    end
                    options.(fields_in{jf}) = options_in.(fields_in{jf});            
                end
            else
                error('Invalid input for NoodleOptions');
            end
        end
        
    end
end

