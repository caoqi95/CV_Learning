%x,y: coordinates on the second image (second: after multiplying the homography)
function A=GetAffineFromHomography(H,x1,y1,x2,y2)
	s = H(3,:) * [x1;y1;1];
    	
	a11=(H(1,1) - H(3,1)*x2)/s;
	a12=(H(1,2) - H(3,2)*x2)/s;
	a21=(H(2,1) - H(3,1)*y2)/s;
	a22=(H(2,2) - H(3,2)*y2)/s;

	A=[a11,a12;a21,a22];

end

