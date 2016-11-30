function PCA(k)
    Data = dlmread('dorothea_train.data',' ');
    %Data = csvread('train2.csv');
    x=zeros(800,100000);
    for i=1:size(Data,1)
        for j=1:size(Data(i,:),2)
            if Data(i,j)>0
                x(i,Data(i,j))=1;
            end
        end
    end
    disp(size(x));
    x=sparse(x);
    [N,M]=size(x);
    nm=mean(x);
    x=x-repmat(nm,N,1);
    covariance= x * x';
    k=500;
    [eigvectors,eigvalues]=eigs(covariance,k);
    eigvalues=diag(eigvalues);
    M=x'*eigvectors;
    final_x=x*M;
    TestLabel = csvread('train_labels.csv');
    TrainPos = final_x(TestLabel(:,1)== 1, :);
    TrainNeg = final_x(TestLabel(:,1)== -1, :);
    MeanPos = mean(TrainPos)';
    MeanNeg = mean(TrainNeg)';
    SigmaPos = cov(TrainPos);
    SigmaNeg = cov(TrainNeg);
    gTrain = [];
    for i = 1:size(final_x,1)
        inputvector = transpose(final_x(i,1:k));
      PosProb = (-0.5 * (inputvector-MeanPos)' * inv(SigmaPos) * (inputvector-MeanPos)) - (0.5 *log(abs(det(SigmaPos)))) + log(abs(size(TrainPos,1)/(size(TrainPos,1)+size(TrainNeg,1)))); 
%        NegProb = (-0.5 * (inputvector-MeanNeg)' * inv(SigmaNeg) * (inputvector-MeanNeg)) - (0.5 *log(abs(det(SigmaNeg)))) + log(abs(size(TrainNeg,1)/(size(TrainPos,1)+size(TrainNeg,1))));
        if PosProb >= NegProb
            gTrain(i,1)=1;
        else
            gTrain(i,1)=-1;
        end
    end
    accuracy = mean(double(gTrain == TestLabel) * 100);
    error = mean(double(gTrain ~= TestLabel) * 100);
    disp(accuracy);
    disp(error);
end