package Main;

import java.util.*;

public class Metrics {
	/** Train-test split index shuffle */
    public static int[] splitIndex(int size, double trainRatio) {
        int[] idx = new int[size];
        for (int i = 0; i < size; i++) idx[i] = i;
        List<Integer> list = new ArrayList<>();
        for (int i : idx) list.add(i);
        Collections.shuffle(list);
        return list.stream().mapToInt(i -> i).toArray();
    }

    /** Accuracy */
    public static double accuracy(int[] yTrue, int[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++)
            if (yTrue[i] == yPred[i]) correct++;
        return correct / (double) yTrue.length;
    }

    /** Recall for black users (label = 1) */
    public static double recallBlack(int[] yTrue, int[] yPred) {
        int tp = 0, fn = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1) {
                if (yPred[i] == 1) tp++;
                else fn++;
            }
        }
        return tp / (double)(tp + fn);
    }

    /** AUC (rank-based) */
    public static double auc(int[] yTrue, float[] prob) {
        List<double[]> pairs = new ArrayList<>();
        int n = yTrue.length;

        for (int i = 0; i < n; i++)
            pairs.add(new double[]{prob[i], yTrue[i]});

        pairs.sort(Comparator.comparingDouble(a -> a[0]));

        int nPos = 0, nNeg = 0;
        for (int y : yTrue) {
            if (y == 1) nPos++; else nNeg++;
        }

        double sumRankPos = 0;
        int rank = 1;
        for (double[] p : pairs) {
            if (p[1] == 1) sumRankPos += rank;
            rank++;
        }

        return (sumRankPos - nPos * (nPos + 1) / 2.0) / (nPos * nNeg);
    }

    /** Confusion Matrix */
    public static void confusionMatrix(int[] yTrue, int[] yPred) {
        int TP=0, FP=0, FN=0, TN=0;

        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1 && yPred[i] == 1) TP++;
            else if (yTrue[i] == 0 && yPred[i] == 1) FP++;
            else if (yTrue[i] == 1 && yPred[i] == 0) FN++;
            else TN++;
        }

        System.out.println("\nConfusion Matrix:");
        System.out.println("TP=" + TP + "  FP=" + FP);
        System.out.println("FN=" + FN + "  TN=" + TN);
    }

    /** Summary print for any model */
    public static void evaluate(String name, int[] yTrue, int[] yPred, float[] prob) {
        System.out.println("\n========== " + name + " PERFORMANCE ==========");

        System.out.printf("Accuracy:\t%.4f%n", accuracy(yTrue, yPred));
        System.out.printf("Recall (Black):\t%.4f%n", recallBlack(yTrue, yPred));
        System.out.printf("AUC:\t\t%.4f%n", auc(yTrue, prob));

        confusionMatrix(yTrue, yPred);
    }
}