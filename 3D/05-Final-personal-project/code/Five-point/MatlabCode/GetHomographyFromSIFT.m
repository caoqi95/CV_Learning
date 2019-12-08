function Hom = GetHomographyFromSIFT(pts1, pts2, rotations)
        
    Hom = [];
    if size(pts1, 1) ~= 3
        'Three correspondences are required!'
        return;
    end

    % Get the null space of the homography
    A = zeros(2 * size(pts1, 1), 9); % 6x9
    for i = 1 : size(pts1, 1)
        x1 = pts1(i,1);
        y1 = pts1(i,2);
        x2 = pts2(i,1);
        y2 = pts2(i,2);

        A(2*i-1, :) = [-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2];
        A(2*i, :) = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2];
    end
    
    A = A / norm(A);    % norm(A) is the largest singular value of A
    N = null(A);
    N = N / norm(N);
    
    n11 = N(1, 1);
    n12 = N(2, 1);
    n13 = N(3, 1);
    n14 = N(4, 1);
    n15 = N(5, 1);
    n16 = N(6, 1);
    n17 = N(7, 1);
    n18 = N(8, 1);
    n19 = N(9, 1);
    
    n21 = N(1, 2);
    n22 = N(2, 2);
    n23 = N(3, 2);
    n24 = N(4, 2);
    n25 = N(5, 2);
    n26 = N(6, 2);
    n27 = N(7, 2);
    n28 = N(8, 2);
    n29 = N(9, 2);
    
    n31 = N(1, 3);
    n32 = N(2, 3);
    n33 = N(3, 3);
    n34 = N(4, 3);
    n35 = N(5, 3);
    n36 = N(6, 3);
    n37 = N(7, 3);
    n38 = N(8, 3);
    n39 = N(9, 3);
    
    % Estimate the homography parameters
    p = perms([1 2 3]);
    best_err = 1e10;
    best_hom = [];
    for i = 1 : size(p, 1)
        m = p(i, 1);
        n = p(i, 2);
        a1 = rotations(m);
        a2 = rotations(n);
    
        x11 = pts1(m,1);
        y11 = pts1(m,2);
        x21 = pts1(n,1);
        y21 = pts1(n,2);
        x12 = pts2(m,1);
        y12 = pts2(m,2);
        x22 = pts2(n,1);
        y22 = pts2(n,2);

        x = (cos(a2) * cos(a1) * n24 * n37 * y12-cos(a2) * cos(a1) * n24 * n37 * y22-cos(a2) * cos(a1) * n27 * n34 * y12+cos(a2) * cos(a1) * n27 * n34 * y22+cos(a2) * n21 * n37 * y22 * sin(a1)-cos(a2) * n24 * n37 * x12 * sin(a1)-cos(a2) * n27 * n31 * y22 * sin(a1)+cos(a2) * n27 * n34 * x12 * sin(a1)-cos(a1) * n21 * n37 * y12 * sin(a2)+cos(a1) * n24 * n37 * x22 * sin(a2)+cos(a1) * n27 * n31 * y12 * sin(a2)-cos(a1) * n27 * n34 * x22 * sin(a2)+n21 * n37 * x12 * sin(a2) * sin(a1)-n21 * n37 * x22 * sin(a2) * sin(a1)-n27 * n31 * x12 * sin(a2) * sin(a1)+n27 * n31 * x22 * sin(a2) * sin(a1)-cos(a2) * n21 * n34 * sin(a1)+cos(a2) * n24 * n31 * sin(a1)+cos(a1) * n21 * n34 * sin(a2)-cos(a1) * n24 * n31 * sin(a2))/(-cos(a2) * n11 * n24 * sin(a1)+cos(a2) * n14 * n21 * sin(a1)+cos(a1) * n11 * n24 * sin(a2)-cos(a1) * n14 * n21 * sin(a2)+n11 * n27 * x12 * sin(a1) * sin(a2)+n17 * n21 * x22 * sin(a1) * sin(a2)+cos(a2) * cos(a1) * n14 * n27 * y12-cos(a2) * cos(a1) * n14 * n27 * y22-cos(a2) * cos(a1) * n17 * n24 * y12+cos(a2) * cos(a1) * n17 * n24 * y22+cos(a2) * n11 * n27 * y22 * sin(a1)-cos(a2) * n14 * n27 * x12 * sin(a1)-cos(a2) * n17 * n21 * y22 * sin(a1)+cos(a2) * n17 * n24 * x12 * sin(a1)-n11 * n27 * x22 * sin(a1) * sin(a2)-n17 * n21 * x12 * sin(a1) * sin(a2)-cos(a1) * n11 * n27 * y12 * sin(a2)+cos(a1) * n14 * n27 * x22 * sin(a2)+cos(a1) * n17 * n21 * y12 * sin(a2)-cos(a1) * n17 * n24 * x22 * sin(a2));
        y = -(cos(a1) * n14 * n37 * x22 * sin(a2) + cos(a1) * n17 * n31 * y12 * sin(a2)-cos(a1) * n17 * n34 * x22 * sin(a2)-n11 * n37 * x22 * sin(a1) * sin(a2)-n17 * n31 * x12 * sin(a1) * sin(a2)+n11 * n37 * x12 * sin(a1) * sin(a2)+n17 * n31 * x22 * sin(a1) * sin(a2)+cos(a2) * cos(a1) * n14 * n37 * y12-cos(a2) * cos(a1) * n14 * n37 * y22-cos(a2) * cos(a1) * n17 * n34 * y12+cos(a2) * cos(a1) * n17 * n34 * y22+cos(a2) * n11 * n37 * y22 * sin(a1)-cos(a2) * n14 * n37 * x12 * sin(a1)-cos(a2) * n17 * n31 * y22 * sin(a1)+cos(a2) * n17 * n34 * x12 * sin(a1)-cos(a1) * n11 * n37 * y12 * sin(a2)-cos(a2) * n11 * n34 * sin(a1)+cos(a2) * n14 * n31 * sin(a1)+cos(a1) * n11 * n34 * sin(a2)-cos(a1) * n14 * n31 * sin(a2))/(-cos(a2) * n11 * n24 * sin(a1)+cos(a2) * n14 * n21 * sin(a1)+cos(a1) * n11 * n24 * sin(a2)-cos(a1) * n14 * n21 * sin(a2)+n11 * n27 * x12 * sin(a1) * sin(a2)+n17 * n21 * x22 * sin(a1) * sin(a2)+cos(a2) * cos(a1) * n14 * n27 * y12-cos(a2) * cos(a1) * n14 * n27 * y22-cos(a2) * cos(a1) * n17 * n24 * y12+cos(a2) * cos(a1) * n17 * n24 * y22+cos(a2) * n11 * n27 * y22 * sin(a1)-cos(a2) * n14 * n27 * x12 * sin(a1)-cos(a2) * n17 * n21 * y22 * sin(a1)+cos(a2) * n17 * n24 * x12 * sin(a1)-n11 * n27 * x22 * sin(a1) * sin(a2)-n17 * n21 * x12 * sin(a1) * sin(a2)-cos(a1) * n11 * n27 * y12 * sin(a2)+cos(a1) * n14 * n27 * x22 * sin(a2)+cos(a1) * n17 * n21 * y12 * sin(a2)-cos(a1) * n17 * n24 * x22 * sin(a2)); 

        h = x*N(:,1) + y*N(:,2) + N(:,3);

        Hom = double(transpose(reshape(h, 3, 3)));

        k = p(i, 3);
        A = GetAffineFromHomography(Hom, pts1(k, 1), pts1(k, 2), pts2(k, 1), pts2(k, 2));

        [sx, sy, angle, w] = DecomposeAffine(A);
        err = abs(angle - rotations(k));
     
        if best_err > err
            best_err = err;
            best_hom = Hom;
        end
    end
    
    best_hom = best_hom / best_hom(3,3);
end