package org.taror.java_machine_learning.libraries.neural_network;

import lombok.Getter;
import lombok.Setter;
import org.taror.java_machine_learning.libraries.neural_network.models.ActFun;
import org.taror.java_machine_learning.libraries.neural_network.models.LossFun;

import java.util.Random;

public class NeuralNetwork {

    @Getter
    private final double[][] layer;
    @Getter
    private final double[][] sumOfWeighs;
    @Getter
    private final double[][] transmittedError;
    @Getter
    @Setter
    private double[][][] weigh;
    private final double[][][] oldWeigh;
    @Getter
    private final double[][][] gradient;

    private final ActFun actFun;
    private final LossFun lossFun;

    private final double alpha;//момент
    private final double beta;//кооэф для функции активации (растягиваем или сжимаем график)
    private final double epsilon;//learning rate
    private final double lambda;//стимуляция нейронов

    @Getter
    private double error;

    public NeuralNetwork(ActFun actFun, LossFun lossFun, int[] layers, double alpha, double beta, double epsilon, double lambda) {

        this.actFun = actFun;
        this.lossFun = lossFun;
        this.alpha = alpha;
        this.beta = beta;
        this.epsilon = epsilon;
        this.lambda = lambda;

        layer = new double[layers.length][];
        sumOfWeighs = new double[layer.length][];
        transmittedError = new double[layer.length][];

        for (int i = 0; i < layer.length; ++i) {

            if (i == layer.length - 1) {
                layer[i] = new double[layers[i]];
            } else {
                layer[i] = new double[layers[i] + 1];
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
    }

    public void initializeWeightsWithRandomValues(int sid) {

        Random random;

        if (sid == 0) {
            random = new Random();
        } else {
            random = new Random(sid);
        }

        for (int i = 0; i < weigh.length; i++) {
            for (int j = 0; j < weigh[i].length; j++) {
                for (int k = 0; k < weigh[i][j].length; k++) {
                    weigh[i][j][k] = random.nextDouble() * 2 - 1;
                }
            }
        }
    }

    public void updateValueInNeurons(double[] input) {

        if (layer[0].length - 1 >= 0) System.arraycopy(input, 0, layer[0], 0, layer[0].length - 1);
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
                layer[i + 1][j] = activationFunction(beta, sumOfWeighs[i + 1][j]);
            }
        }

        if (lossFun.equals(LossFun.CROSS_ENTROPY)) {
            for(int j = 0; j < layer[layer.length - 1].length; j++) {
                layer[layer.length - 1][j] = ActivationFunction.softmax(sumOfWeighs[layer.length - 1], sumOfWeighs[layer.length - 1][j]);
            }
        }
    }

    public void backPropagationOfError(double[] result, double[] answers) {

        error = 0;

        for (int i = 0; i < transmittedError[transmittedError.length - 1].length; i++) {
            transmittedError[transmittedError.length - 1][i] = lossFunctionDerivative(result[i], answers[i]);
            error += lossFunction(layer[layer.length - 1][i], answers[i]);
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
                    transmittedError[i][j] += transmittedError[i + 1][k] * activationFunctionDerivative(beta, sumOfWeighs[i + 1][k]) * weigh[i][j][k];
                }
            }
        }
    }

    public void gradientCalculation() {

        for (int i = 0; i < gradient.length; i++) {
            for(int j = 0; j < gradient[i].length; j++) {
                for (int k = 0; k < gradient[i][j].length; k++) {
                    gradient[i][j][k] = transmittedError[i + 1][k] * activationFunctionDerivative(beta, sumOfWeighs[i + 1][k]) * layer[i][j];
                }
            }
        }
    }

    public void weighUpdate() {

        double deltaWeigh;

        for (int i = 0; i < weigh.length; i++) {
            for (int j = 0; j < weigh[i].length; j++) {
                for (int k = 0; k < weigh[i][j].length; k++) {
                    deltaWeigh = weigh[i][j][k] - oldWeigh[i][j][k];
                    oldWeigh[i][j][k] = weigh[i][j][k];
                    weigh[i][j][k] = (1 - lambda) * weigh[i][j][k] - epsilon * gradient[i][j][k] + alpha * deltaWeigh;
                }
            }
        }
    }

    public double[] getResult() {
        return layer[layer.length - 1];
    }

    private double activationFunction(double beta, double x) {

        if (actFun.equals(ActFun.TANH)) {
            return ActivationFunction.tanh(beta, x);
        } else if (actFun.equals(ActFun.SIGMOID)) {
            return ActivationFunction.sigmoid(beta, x);
        } else if (actFun.equals(ActFun.SOFTMAX)) {
            return ActivationFunction.sigmoid(beta, x);
        } else {
            return 0;
        }
    }

    private double activationFunctionDerivative(double beta, double x) {

        if (actFun.equals(ActFun.TANH)) {
            return ActivationFunctionDerivative.tanh(beta, x);
        } else if (actFun.equals(ActFun.SIGMOID)) {
            return ActivationFunctionDerivative.sigmoid(beta, x);
        } else if (actFun.equals(ActFun.SOFTMAX)) {
            return ActivationFunctionDerivative.sigmoid(beta, x);
        } else {
            return 0;
        }
    }


    private double lossFunction(double output, double answer) {

        if (lossFun.equals(LossFun.MSE)) {
            return LossFunction.mse(output, answer);
        } else if (lossFun.equals(LossFun.CROSS_ENTROPY)) {
            return LossFunction.crossEntropy(output, answer);
        } else {
            return 0;
        }
    }

    private double lossFunctionDerivative(double output, double answer) {

        if (lossFun.equals(LossFun.MSE)) {
            return LossFunctionDerivative.mse(output, answer);
        } else if (lossFun.equals(LossFun.CROSS_ENTROPY)) {
            return LossFunctionDerivative.crossEntropyPlusSoftmax(output, answer);
        } else {
            return 0;
        }
    }
}
