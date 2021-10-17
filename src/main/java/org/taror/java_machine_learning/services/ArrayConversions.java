package org.taror.java_machine_learning.services;

public class ArrayConversions {

    public static int[] takeLayersFromIndividual(double[] individual) {

        int[] layers = new int[(int) (individual[1] + 2)];

        layers[0] = (int) individual[0];
        for (int i = 1; i < individual[1] + 2; i++) {
            layers[i] = (int) individual[2];
        }
        layers[(int) (individual[1] + 1)] = 1;

        return layers;
    }

    public static double[] stringArrToDoubleArr(String arr) {

        String[] items = arr.replaceAll("\\[", "").replaceAll("\\]", "").replaceAll("\\s", "").split(",");

        double[] results = new double[items.length];

        for (int i = 0; i < items.length; i++) {
            results[i] = Double.parseDouble(items[i]);
        }

        return results;
    }

    public static double[][][] weightStringToDoubleArr(int[] layers, String arr) {

        double[][] layer = new double[layers.length][];
        for (int i = 0; i < layer.length; ++i) {

            if (i == layer.length - 1) {
                layer[i] = new double[layers[i]];
            } else {
                layer[i] = new double[layers[i] + 1];
            }
        }

        double[][][] weigh = new double[layer.length - 1][][];
        String[] str1 = arr.split("]], \\[\\[");

        for (int i = 0; i < weigh.length; i++) {

            weigh[i] = new double[layer[i].length][];
            String[] str2 = str1[i].replaceAll("\\[\\[\\[", "").replaceAll("]]]", "").split("], \\[");

            for (int j = 0; j < weigh[i].length; j++) {

                if (i == weigh.length - 1) {
                    weigh[i][j] = new double[layer[i + 1].length];
                } else {
                    weigh[i][j] = new double[layer[i + 1].length - 1];
                }

                String[] str3 = str2[j].split(",");

                for (int k = 0; k < str3.length; k++) {
                    weigh[i][j][k] = Double.parseDouble(str3[k]);
                }
            }
        }

        return weigh;
    }
}
