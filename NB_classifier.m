function NB_classifier(n)
    D = csvread('train.csv',1,0);
%    TestData = csvread('train_new.csv',1,0);
    Ds = D(randperm(size(D, 1)), :);
    TrainData=Ds(1:n/2,:);
    TestData=Ds(n/2+1:n,:);
    TrainPos = TrainData(TrainData(:, 38)== 1, :);
    TrainNeg = TrainData(TrainData(:, 38)== 0, :);
    Trainlabel = TrainData(:,38);
    xtrain = TrainData(:,1:37);
    TrainPos = TrainPos(:,1:37);
    TrainNeg = TrainNeg(:,1:37);
    MeanPos = mean(TrainPos)';
    MeanNeg = mean(TrainNeg)';
    SigmaPos = cov(TrainPos);
    SigmaNeg = cov(TrainNeg);
    Testlabel = TestData(:,38);
    gTrain = [];
    for i = 1:size(TrainData,1)
        inputvector = transpose(TrainData(i,1:37));
        PosProb = (-0.5 * (inputvector-MeanPos)' * inv(SigmaPos) * (inputvector-MeanPos)) - (0.5 *log(abs(det(SigmaPos)))) + log(abs(size(TrainPos,1)/50000)); 
        NegProb = (-0.5 * (inputvector-MeanNeg)' * inv(SigmaNeg) * (inputvector-MeanNeg)) - (0.5 *log(abs(det(SigmaNeg)))) + log(abs(size(TrainNeg,1)/50000));
        if PosProb >= NegProb
            gTrain(i,1)=1;
        else
            gTrain(i,1)=0;
        end
    end

    accuracy = mean(double(gTrain == Testlabel) * 100);
    error = mean(double(gTrain ~= Testlabel) * 100);
    disp(accuracy);
    disp(error);
end