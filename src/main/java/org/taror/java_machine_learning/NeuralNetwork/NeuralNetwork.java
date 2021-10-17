package org.taror.java_machine_learning.NeuralNetwork;

import java.util.Random;

public class NeuralNetwork {

    Properties properties;

    private final double[][] layer;
    private final double[][] sumOfWeighs;
    private final double[][] transmittedError;
    private double[][][] weigh;
    private final double[][][] oldWeigh;
    private final double[][][] gradient;

    private double error;

    public NeuralNetwork(Properties properties) {

        this.properties = properties;

        layer = new double[properties.getLayer().length][];
        sumOfWeighs = new double[layer.length][];
        transmittedError = new double[layer.length][];

        for (int i = 0; i < layer.length; ++i) {

            if (i == layer.length - 1) {
                layer[i] = new double[properties.getLayer()[i]];
            } else {
                layer[i] = new double[properties.getLayer()[i] + 1];
            }
            sumOfWeighs[i] = new double[layer[i].length];
            transmittedError[i] = new double[layer[i].length];
        }

        weigh = new double[layer.length - 1][][];
        oldWeigh = new double[weigh.length][][];
        gradient = new double[weigh.length][][];

        for (int i = 0; i < weigh.length; i++) {

            weigh[i] = new double[layer[i].length][];
            oldWeigh[i] = new double[weigh[i].length][];
            gradient[i] = new double[weigh[i].length][];

            for (int j = 0; j < weigh[i].length; j++) {

                if (i == weigh.length - 1) {
                    weigh[i][j] = new double[layer[i + 1].length];
                } else {
                    weigh[i][j] = new double[layer[i + 1].length - 1];
                }
                oldWeigh[i][j] = new double[weigh[i][j].length];
                gradient[i][j] = new double[weigh[i][j].length];
            }
        }

        initializeWeightsWithRandomValues();
    }

    public void learningIteration(double[] input, double[] answers) {

        updateValueInNeurons(input);
        backPropagationOfError(answers);
        gradientCalculation();
        weighUpdate();
    }

    public double getError() {
        return error;
    }

    public double[] getResult(double[] input) {

        updateValueInNeurons(input);
        return layer[layer.length - 1];
    }

    public void initializeWeightsWithRandomValues() {

        Random random = new Random(1);//-------------------------------------------

        for (int i = 0; i < weigh.length; i++) {
            for (int j = 0; j < weigh[i].length; j++) {
                for (int k = 0; k < weigh[i][j].length; k++) {
                    weigh[i][j][k] = random.nextDouble() * 2 - 1;
                }
            }
        }
    }

    private void updateValueInNeurons(double[] input) {

        for (int i = 0; i < layer[0].length - 1; i++) {
            layer[0][i] = input[i];
        }
        layer[0][layer[0].length - 1] = 1;

        int jCount;

        for (int i = 0; i < layer.length - 1; i++) {

            if (i == layer.length - 2) {
                jCount = layer[i + 1].length;
            } else {
                jCount = layer[i + 1].length - 1;
                layer[i + 1][jCount] = 1;
            }

            for(int j = 0; j < jCount; j++) {

                sumOfWeighs[i + 1][j] = 0;
                for(int k = 0; k < layer[i].length; k++) {
                    sumOfWeighs[i + 1][j] += layer[i][k] * weigh[i][k][j];
                }
                layer[i + 1][j] = activationFunction(properties.getBeta(), sumOfWeighs[i + 1][j]);
            }
        }
    }

    private void backPropagationOfError(double[] answers) {

        error = 0;

        for (int i = 0; i < transmittedError[transmittedError.length - 1].length; i++) {
            transmittedError[transmittedError.length - 1][i] = layer[layer.length - 1][i] - answers[i];
//            error += Math.pow(layer[layer.length - 1][i] - answers[i], 2);
            error += Math.abs(layer[layer.length - 1][i] - answers[i]);
        }

        error = error / answers.length;

        for (int i = transmittedError.length - 2; i > 0; i--) {
            for (int j = 0; j < transmittedError[i].length - 1; j++) {

                transmittedError[i][j] = 0;

                int counter;
                if (i == transmittedError.length - 2) {
                    counter = transmittedError[i + 1].length;
                } else {
                    counter = transmittedError[i + 1].length - 1;
                }

                for (int k = 0; k < counter; k++) {
                    transmittedError[i][j] += transmittedError[i + 1][k] * derivative(properties.getBeta(), sumOfWeighs[i + 1][k]) * weigh[i][j][k];
                }
            }
        }
    }

    private void gradientCalculation() {

        for (int i = 0; i < gradient.length; i++) {
            for(int j = 0; j < gradient[i].length; j++) {
                for (int k = 0; k < gradient[i][j].length; k++) {
                    gradient[i][j][k] = transmittedError[i + 1][k] * derivative(properties.getBeta(), sumOfWeighs[i + 1][k]) * layer[i][j];
                }
            }
        }
    }

    private void weighUpdate() {

        double deltaWeigh;

        for (int i = 0; i < weigh.length; i++) {
            for (int j = 0; j < weigh[i].length; j++) {
                for (int k = 0; k < weigh[i][j].length; k++) {
                    deltaWeigh = weigh[i][j][k] - oldWeigh[i][j][k];
                    oldWeigh[i][j][k] = weigh[i][j][k];
                    weigh[i][j][k] = (1 - properties.getLambda()) * weigh[i][j][k] - properties.getEpsilon() *
                            gradient[i][j][k] + properties.getAlpha() * deltaWeigh;
                }
            }
        }
    }

    private double activationFunction(double beta, double x) {

        if (properties.getActFun().equals("tanh")) {
            return ActivationFunction.tanh(beta, x);
        } else if (properties.getActFun().equals("sigmoid")) {
            return ActivationFunction.sigmoid(beta, x);
        } else {
            return 0; //Подумать что вернуть если ошибся в написании функции
        }
    }

    private double derivative(double beta, double x) {

        if (properties.getActFun().equals("tanh")) {
            return Derivative.tanh(beta, x);
        } else if (properties.getActFun().equals("sigmoid")) {
            return Derivative.sigmoid(beta, x);
        } else {
            return 0; //Подумать что вернуть если ошибся в написании функции
        }
    }

    public double[][][] getWeigh() {
        return weigh;
    }

    public void setWeigh(double[][][] weigh) {
        this.weigh = weigh;
    }
}
