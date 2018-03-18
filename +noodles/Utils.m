classdef Utils
    %The Utils class contains utility functions likely to be used by
    %several subproblem solvers.
    
    methods (Static)
        
        function z = min_poly3(c0, c1, c2, c3, c4, DeltaNeg, DeltaPos)
            % compute the minimum of the function 
            % h(z) = c0 + c1*z + c2*z^2 + c3*z^3 + c4*abs(z)^3
            % in [DeltaNeg,DeltaPos]
            
            h = @(z) c0 + c1*z + c2*z^2 + c3*z^3 + c4*abs(z)^3;
            zs = [DeltaNeg,DeltaPos];
            argmin = @(zs, fun) noodles.Utils.argmin(zs, fun);
            
            if c2 == 0 && c3 == 0 && c4 == 0
                % polynomial of degree <= 1
                z = argmin(zs,h);
            
            elseif c2 ~= 0 && c3 == 0 && c4 == 0
                % polynomial of degree 2
                % critical point
                zcrt = -c1/(2*c2);
                if DeltaNeg < zcrt && zcrt < DeltaPos
                    zs = [zs zcrt];
                end
                z = argmin(zs,h);
                
            elseif c3 ~= 0 && c4 == 0
                % polynomial of degree 3
                % discriminant
                xi = c2^2/(3*c3)^2 - c1/(3*c3);
                if xi < 0
                    z = argmin(zs,h);
                else
                    zcrt1 = -c2/(3*c3)-sqrt(xi);
                    zcrt2 = -c2/(3*c3)+sqrt(xi);
                    if DeltaNeg < zcrt1 && zcrt1 < DeltaPos
                        zs = [zs zcrt1];
                    end
                    if DeltaNeg < zcrt2 && zcrt2 < DeltaPos
                        zs = [zs zcrt2];
                    end
                    z = argmin(zs,h);
                end
                
            elseif c4 ~= 0
                % third order absolute value term, consider <>=0
                zneg = noodles.Utils.min_poly3(c0,c1,c2,c3-c4,0,DeltaNeg,0);
                zpos = noodles.Utils.min_poly3(c0,c1,c2,c3+c4,0,0,DeltaPos);
                z = argmin([zneg,zpos],h);
            end
            
        end
        
        function z = argmin(zs, fun)
            % value z in zs(dim,n) such that fun(z) is minimal among all z
            % in zs
            
            n = size(zs,2);
            z = zs(1);
            fval = fun(z);
            for j=2:n
                z_new = zs(j);
                fval_new = fun(z_new);
                if isnan(fval) || fval_new < fval 
                    z = z_new;
                    fval = fval_new;
                end
            end
            
        end

    end
end

