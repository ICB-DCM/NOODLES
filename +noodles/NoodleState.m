classdef NoodleState < handle
    % The NoodleState class contains all information about the current,
    % evolving, state of the problem.
    
    properties
        x;
        fval;
        grad;
        hess;
        gradnorm;
        fvaldiff;
        iter_count;
        feval_count;
        
        % extra information required by any of the adaptable components can
        % be put into this struct
        meta
    end
    
    methods
        function this = NoodleState(dim)
            this.x = nan(dim,1);
            this.fval = nan;
            this.grad = nan(dim,1);
            this.hess = nan(dim,dim);
            this.gradnorm = nan;
            this.fvaldiff = inf;
            this.iter_count = 0;
            this.feval_count = 0;
            this.meta = struct();
        end

    end
end

