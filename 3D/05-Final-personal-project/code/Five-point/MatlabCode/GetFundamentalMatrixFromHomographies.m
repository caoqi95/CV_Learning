function Fund = GetFundamentalMatrixFromHomographies(H, pts1, pts2, img_size)
    
    %% Computing the null space implied by the homography and a point correspondence    
    A = [];
    % generat points
    % 3x5
    hallu_pts1 = [0, 0, 1; img_size(1), 0, 1; 0, img_size(2), 1; img_size(1), img_size(2), 1; img_size(1) / 2, img_size(2) / 2, 1]';
    % 3x5
    hallu_pts2 = H * hallu_pts1;

    for i = 1 : 5
       hallu_pts2(:,i) = hallu_pts2(:,i) / hallu_pts2(3,i); % 除以最后一行
    end
    % pts1, pts2 size: 5x2
    hallu_pts1 = [hallu_pts1'; pts1, repmat(1,5,1)]'; % 3x10
    hallu_pts2 = [hallu_pts2'; pts2, repmat(1,5,1)]'; % 3x10
    
    % get A(D) matrix
    for i = 1 : size(hallu_pts1, 2) % 1-10
       pt1 = hallu_pts1(:,i); % 3x1
       pt2 = hallu_pts2(:,i);

       u1 = pt1(1);
       v1 = pt1(2);
       u2 = pt2(1);
       v2 = pt2(2);
       A = [A; u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, 1];

    end 
    % size of A: 10x9
    f = null(A);
    
    Fund = [];
    
    pts1h = [pts1(1:5,:) repmat(1, 5, 1)]'; %3x5
    pts2h = [pts2(1:5,:) repmat(1, 5, 1)]'; %3x5
    
    if size(f,2) == 1
        Fund = reshape(f, [3 3])';
        if ~signs_OK(Fund, pts1h, pts2h)
            Fund = [];
        end
    elseif size(f,2) == 2
        FF{1} = reshape(f(:,end-1), [3 3])';
        FF{2} = reshape(f(:,end  ), [3 3])';
        a = vgg_singF_from_FF(FF);
        
        for i = 1 : length(a)

            if imag(a(i)) ~= 0
                continue;
            end

            Fi = a(i) * FF{1} + (1 - a(i)) * FF{2};

            if signs_OK(Fi, pts1h, pts2h)
                Fund = cat(3, Fund, Fi);
            end
        end
    end
   
return

function OK = signs_OK(F, x1, x2)
    [u,s,v] = svd(F');
    e1 = v(:,3);
    l1 = vgg_contreps(e1)*x1;
    s = sum( (F*x2) .* l1 );
    OK = all(s>0) | all(s<0);
return

