% Copyright (C) 2018 Hungarian Academy of Sciences, Institute for Computer Science and Control (MTA SZTAKI).
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%
%     * Redistributions in binary form must reproduce the above
%       copyright notice, this list of conditions and the following
%       disclaimer in the documentation and/or other materials provided
%       with the distribution.
%
%     * Neither the name of MTA SZTAKI or Hungarian Academy of Sciences, Institute for Computer Science and Control nor the
%       names of its contributors may be used to endorse or promote products
%       derived from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
% Please contact the author of this code if you have any questions.
% If you use this code please cite the following paper: "Daniel Barath, 
% Five-point Fundamental Matrix Estimation for Uncalibrated Cameras,
% Conference on Computer Vision and Pattern Recognition, 2018"
% Author: Daniel Barath (barath.daniel@sztaki.mta.hu)

%% Loading the data
% Available tests: 'barrsmith', 'booksh', 'Kyoto', and 'johnssonb'
test = 'barrsmith';
% data contains correspondences, rotations
data = dlmread(strcat('D:/GitHub/five-point-fundamental/MatlabCode/data/', test, '_kps.txt'));
% num of features
n = size(data, 1);
maxvalues = max(data(:,1:2));
%% RANSAC
max_iterations = inf;
iterations = 2000;
threshold = 3.0;
best_inliers = [];
confidence = 0.99;

for i = 1 : iterations
    indices = randperm(n, 5); % 1-n 行中随机选择 5 个数,size(1,5)
    pts1 = data(indices, 1:2); % size (5,2)
    pts2 = data(indices, 3:4); % size (5,2)
    alphas = data(indices, 5); % size (5,1)
    
    % Estimate a homography from the first three correspondences
    [normedPts1, normedPts2, T1, T2] = NormalizePoints(pts1, pts2);
    % normedPts1(1:3, :) - size (3, 3)
    % alphas(1:3, :) - size (3, 1)
    H = GetHomographyFromSIFT(normedPts1(1:3, :), normedPts2(1:3, :), alphas(1:3, :));
    H = inv(T2) * H * T1;    
    
    % Check if the sstwo addition points lie on the same plane
    proj1 = H * [pts1(4:5,:), repmat(1, 2, 1)]';
    is_too_close = 0;
    for j = 1 : 2
        dist = norm(proj1(1:2,j) / proj1(3,j) - pts2(j + 3,:)');
        if dist < threshold
            is_too_close = 1;
            break;
        end
    end
    
    if is_too_close
        continue;
    end
    
    % Estimate a fundamental matrix from the homography and point correspondences
    F = GetFundamentalMatrixFromHomographies(H, pts1, pts2, maxvalues);
 
    if F == 0
        continue;
    end
    
    if size(F, 1) ~= 3
        continue;
    end    
    
    % Get the inliers
    for fi = 1 : size(F, 3)
        %disp(size(F,3));
        inliers = [];

        Fi = F(:,:,fi);

        % Estimate the symmetric epipolar distance for each correspondences
        for j = 1 : n 
            l1 = [data(j, 3:4), 1] * Fi;
            l2 = Fi * [data(j, 1:2), 1]';

            l1 = l1 / sqrt(l1(1)^2 + l1(2)^2);

            l2 = l2 / sqrt(l2(1)^2 + l2(2)^2);
            %disp(size(l2));
            %disp(size(l2(1)));
            dist = abs(l1 * [data(j, 1:2), 1]') + abs([data(j, 3:4), 1] * l2) * 0.5;

            if dist < threshold
                inliers = [inliers j];
            end
        end
        
        if length(best_inliers) < length(inliers)
            % Update inliers of the so-far-the-best model
            best_inliers = inliers;
            
            % Update max iteration number
            max_iterations = log(1 - confidence) / log(1 - (length(best_inliers) / n)^5);
        end
    end
    
    if i > max_iterations
        break;
    end    
end

%% Visualization
fprintf('Number of loaded points = %d\n', n);
fprintf('Number of found inliers = %d\n', length(best_inliers));
fprintf('Number of iterations = %d\n', i);

close all;
img1 = imread(strcat('D:/GitHub/five-point-fundamental/MatlabCode/data/', test, 'A.png'));
img2 = imread(strcat('D:/GitHub/five-point-fundamental/MatlabCode/data/', test, 'B.png'));


image = [img1 img2];
figure(1)
imshow(image)
hold on;
colormap hsv;
for i = 1 : length(best_inliers)
    color = rand(1,3);
    %color = colormap(i);
    %disp(size(data(best_inliers(i),1)));
    plot([data(best_inliers(i), 1), size(img1, 2) + data(best_inliers(i), 3)], [data(best_inliers(i), 2) data(best_inliers(i), 4)], 'Color', color, 'LineWidth',2)
    
    scatter(data(best_inliers(i), 1), data(best_inliers(i), 2), 80, color, 'filled');
    scatter(size(img1, 2) + data(best_inliers(i), 3), data(best_inliers(i), 4), 80, color,'filled');
end
hold off;

figure(2)
imshow(image)
hold on;
colormap hsv;
for i = 1 : length(best_inliers)
    color = rand(1,3);
    %color = colormap(i);
    %disp(size(data(best_inliers(i),1)));
    %plot([data(best_inliers(i), 1), size(img1, 2) + data(best_inliers(i), 3)], [data(best_inliers(i), 2) data(best_inliers(i), 4)], 'Color', color, 'LineWidth',2)
    
    scatter(data(best_inliers(i), 1), data(best_inliers(i), 2), 80, color, 'filled');
    scatter(size(img1, 2) + data(best_inliers(i), 3), data(best_inliers(i), 4), 80, color,'filled');
end
hold off;