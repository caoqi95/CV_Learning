%-----------------------------------------------------------------%
% Copyright 2014-2016, Daniel Barath  barath.daniel@sztaki.mta.hu %
%-----------------------------------------------------------------%

function [normedPts1, normedPts2, T1, T2] = NormalizePoints(pts1, pts2)
				
	N			= size(pts1,1);
    
	massPoint1	= mean(pts1);
	massPoint2	= mean(pts2);
    
    normedPts1  = pts1 - repmat(massPoint1, N, 1);
    normedPts2  = pts2 - repmat(massPoint2, N, 1);
    
    avgDist1    = 0;
    avgDist2    = 0;
    
	for i = 1 : N
		avgDist1		= avgDist1 + norm(normedPts1(i,:));
		avgDist2		= avgDist2 + norm(normedPts2(i,:));
    end;
    
	avgDist1	= avgDist1 / N;
	avgDist2	= avgDist2 / N;
	avgRatio1	= sqrt(2) / avgDist1;
	avgRatio2	= sqrt(2) / avgDist2;
	
    normedPts1  = normedPts1 * avgRatio1;
    normedPts2  = normedPts2 * avgRatio2;
    normedPts1(:,3)	= 1;
    normedPts2(:,3)	= 1;
    		
	T1			= [avgRatio1, 0, 0;
					0, avgRatio1, 0;
					0, 0, 1] * [1, 0, -massPoint1(1);
					0, 1, -massPoint1(2);
					0, 0, 1];
	
	T2			= [avgRatio2, 0, 0;
					0, avgRatio2, 0;
					0, 0, 1] * [1, 0, -massPoint2(1);
					0, 1, -massPoint2(2);
					0, 0, 1];
end

