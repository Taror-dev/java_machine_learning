package org.taror.java_machine_learning.examples;

import org.taror.java_machine_learning.NeuralNetwork.NeuralNetwork;
import org.taror.java_machine_learning.NeuralNetwork.Properties;
import org.taror.java_machine_learning.services.RandomNumber;

public class NeuralNetworkXSinX {

    Properties properties;

    private int inputCount;
    private int iterationCount;
    private double error;

    private double[] input;
    private double[] answers;

    private double[][] learningInput;
    private double[][] learningAnswers;

    private double[][][] weigh;

    int counter;

    NeuralNetwork neuralNetwork;

    int countOfExamples;

    public NeuralNetworkXSinX(int inputCount, int iterationCount, int maxError, String actFun, int[] layers, double[] parameters) {

        double alpha = parameters[0];
        double beta = parameters[1];
        double epsilon = parameters[2];
        double lambda = parameters[3];

        setInputCount(inputCount);
        setIterationCount(iterationCount);

        setProperties(new Properties(actFun, layers, alpha, beta, epsilon, lambda));
    }

    public void setInputData() {

        input = new double[inputCount];
        answers = new double[inputCount];

        double x = 0;

        for (int i = 0; i < inputCount; i++) {

            input[i] = x;
            answers[i] = x * Math.sin(x) / 10;
            x += 0.25;
        }
    }

    private void generateLearningInputAndAnswers() {

        countOfExamples = (int) RandomNumber.randomInRange(1, 100);

        learningInput = new double[countOfExamples][inputCount];
        learningAnswers = new double[countOfExamples][inputCount];

        double x;

        for (int i = 0; i < countOfExamples; i++) {
            for (int j = 0; j < inputCount; j++) {

                x = input[(int) RandomNumber.randomInRange(0, inputCount)];
                learningInput[i][j] = x;
                learningAnswers[i][j] = x * Math.sin(x) / 10;
            }
        }
    }

    public void learningIteration(double[] input, double[] answers) {

        double[] in = new double[1];
        double[] out = new double[1];

        long end = System.currentTimeMillis() + 200;

        while(System.currentTimeMillis() < end) {

            for (int i = 0; i < input.length; i++) {

                in[0] = input[i];
                out[0] = answers[i];

                for (int j = 0; j < 5; j++) {
                    neuralNetwork.learningIteration(in, out);
                }
            }

            counter++;

            if (counter > iterationCount) {
                break;
            }
        }
    }

    public void run() {

        setInputData();
        generateLearningInputAndAnswers();

        neuralNetwork = new NeuralNetwork(properties);

        counter = 0;

        while (true) {

            int rnd = (int) RandomNumber.randomInRange(0, countOfExamples);

            learningIteration(learningInput[rnd], learningAnswers[rnd]);

            if (counter > iterationCount) {
                setWeigh(neuralNetwork.getWeigh());
                error = neuralNetwork.getError();
                break;
            }
        }
    }

    public void setProperties(Properties properties) {
        this.properties = properties;
    }

    public void setInputCount(int inputCount) {
        this.inputCount = inputCount;
    }

    public void setIterationCount(int iterationCount) {
        this.iterationCount = iterationCount;
    }

    public double[][][] getWeigh() {
        return weigh;
    }

    public void setWeigh(double[][][] weigh) {
        this.weigh = weigh;
    }

    public double getError() {
        return error;
    }
}
