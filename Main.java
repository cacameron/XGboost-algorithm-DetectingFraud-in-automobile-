package Main;

import java.nio.file.*;
import org.apache.commons.csv.*;
import java.io.*;
import java.util.*;
import ml.dmlc.xgboost4j.java.*;
import smile.classifcation.LogisticRegression;

public class Main {

    public static void main(String[] args) throws Exception {

        // 1. LOAD DATA (CSV)
        List<double[]> Xlist = new ArrayList<>();
        List<Integer> ylist = new ArrayList<>();

        Reader reader = Files.newBufferedReader(Paths.get("fraud_oracle.csv"));
        CSVParser parser = CSVFormat.DEFAULT
                .withFirstRecordAsHeader()
                .parse(reader);


        for (CSVRecord r : parser) {
            double[] row = new double[104];
            for (int i = 0; i < 104; i++) {
                String v = r.get(i);
                row[i] = (v == null || v.isEmpty()) ? Double.NaN : Double.parseDouble(v);
            }
            Xlist.add(row);
            ylist.add(Integer.parseInt(r.get("label")));
        }

        int n = Xlist.size();
        double[][] X = Xlist.toArray(new double[n][]);
        int[] y = ylist.stream().mapToInt(i->i).toArray();


        // 2. PREPROCESS: Fill missing values with mean
        int d = X[0].length;
        double[] mean = new double[d];
        int[] count = new int[d];

        for (double[] row : X) {
            for (int j = 0; j < d; j++) {
                if (!Double.isNaN(row[j])) {
                    mean[j] += row[j];
                    count[j]++;
                }
            }
        }
        for (int j = 0; j < d; j++) mean[j] /= count[j];

        for (double[] row : X)
            for (int j = 0; j < d; j++)
                if (Double.isNaN(row[j])) row[j] = mean[j];


        // 3. TRAIN TEST SPLIT (60/40)
        int[] idx = Metrics.splitIndex(n, 0.6);
        int trainSize = (int)(n * 0.6);

        double[][] Xtrain = new double[trainSize][d];
        double[][] Xtest  = new double[n - trainSize][d];
        int[] ytrain = new int[trainSize];
        int[] ytest  = new int[n - trainSize];

        for (int i = 0; i < trainSize; i++) {
            Xtrain[i] = X[idx[i]];
            ytrain[i] = y[idx[i]];
        }
        for (int i = trainSize; i < n; i++) {
            Xtest[i - trainSize] = X[idx[i]];
            ytest[i - trainSize] = y[idx[i]];
        }

        // 4A. LOGISTIC REGRESSION (Smile)
        LogisticRegression logModel = LogisticRegression.fit(Xtrain, ytrain);

        int[] logPred = new int[Xtest.length];
        float[] logProb = new float[Xtest.length];

        for (int i = 0; i < Xtest.length; i++) {
            logPred[i] = logModel.predict(Xtest[i]);
            logProb[i] = (float) logModel.predictProba(Xtest[i])[1];
        }

        Metrics.evaluate("Logistic Regression", ytest, logPred, logProb);

        // 4B. XGBOOST
        DMatrix dtrain = new DMatrix(Xtrain, trainSize, d, Float.NaN);
        dtrain.setLabel(ytrain);

        DMatrix dtest = new DMatrix(Xtest, Xtest.length, d, Float.NaN);

        Map<String,Object> params = new HashMap<>();
        params.put("objective", "binary:logistic");
        params.put("eta", 0.01);
        params.put("max_depth", 5);
        params.put("gamma", 0.3);
        params.put("min_child_weight", 2);
        params.put("eval_metric", "auc");

        Booster booster = XGBoost.train(dtrain, params, 400, null, null, null);

        float[][] xgbOutput = booster.predict(dtest);
        float[] xgbProb = xgbOutput[0];

        int[] xgbPred = new int[xgbProb.length];
        for (int i = 0; i < xgbProb.length; i++)
            xgbPred[i] = (xgbProb[i] > 0.5 ? 1 : 0);

        Metrics.evaluate("XGBoost", ytest, xgbPred, xgbProb);
    }
}
