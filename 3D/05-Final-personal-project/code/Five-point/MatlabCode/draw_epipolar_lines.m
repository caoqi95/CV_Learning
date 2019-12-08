% Draw Epipolar Lines Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Draw the epipolar lines given the fundamental matrix, left right images
% and left right datapoints

% You do not need to modify anything in this function, although you can if
% you want to.

function [] = draw_epipolar_lines( F_matrix, ImgLeft, ImgRight, PtsLeft, PtsRight)
    Pul=[1 1 1];
    Pbl=[1 size(ImgLeft,1) 1];
    Pur=[size(ImgLeft,2) 1 1];
    Pbr=[size(ImgLeft,2) size(ImgLeft,1) 1];

    lL = cross(Pul,Pbl);
    lR = cross(Pur,Pbr);
    figure
    imshow(ImgRight)
    for i = 1:size(PtsLeft,1)
        e = F_matrix*[PtsLeft(i,:) 1]';
        PL = cross(e,lL);
        PR = cross(e,lR);
        x = [PL(1)/PL(3) PR(1)/PR(3)];
        y = [PL(2)/PL(3) PR(2)/PR(3)];
        line(x,y)
        
    end
    hold on
    plot(PtsRight(:,1),PtsRight(:,2),'go','MarkerFaceColor','r','MarkerSize',5)
    hold off
    figure
    imshow(ImgLeft)
    for i = 1:size(PtsRight,1)
        e = F_matrix'*[PtsRight(i,:) 1]';
        PL = cross(e,lL);
        PR = cross(e,lR);
        x = [PL(1)/PL(3) PR(1)/PR(3)];
        y = [PL(2)/PL(3) PR(2)/PR(3)];
        line(x,y)
    end
    hold on
    plot(PtsLeft(:,1),PtsLeft(:,2),'go','MarkerFaceColor','r','MarkerSize',5)
    hold off
end

