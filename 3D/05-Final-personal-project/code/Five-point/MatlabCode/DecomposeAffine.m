function [sx,sy,alpha,w] = DecomposeAffine(A)

	alpha = atan(A(2, 1) / A(1, 1));
	sx = A(1, 1) / cos(alpha);
	sy = (cos(alpha) * A(2,2) - sin(alpha) * A(1,2)) / (cos(alpha)*cos(alpha) + sin(alpha)*sin(alpha));
    w = (A(1,2) + sin(alpha)*sy) / cos(alpha);
    
    %w = A(2, 2) * sin(alpha) + A(1, 2) * cos(alpha);
	%sy = -(A(1, 2) - sin(alpha) * w) / sin(alpha);

end

