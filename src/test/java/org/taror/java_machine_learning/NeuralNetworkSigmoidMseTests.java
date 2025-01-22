package org.taror.java_machine_learning;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.taror.java_machine_learning.libraries.neural_network.NeuralNetwork;
import org.taror.java_machine_learning.libraries.neural_network.models.ActFun;
import org.taror.java_machine_learning.libraries.neural_network.models.LossFun;

public class NeuralNetworkSigmoidMseTests {

    @Test
    void theSizeOfTheNeuralNetworkShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        double[][] layer = neuralNetwork.getLayer();
        double[][] result = new double[][] {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

        Assertions.assertArrayEquals(layer, result);
    }

    @Test
    void initializeWeightsShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);

        double[][][] weighs = neuralNetwork.getWeigh();
        double[][][] result = {{{0.4617563814065817, -0.17983837701559668, -0.5845703173805659}, {-0.33456588808097765, 0.9355118188482414, -0.9877656354684774}, {0.9274095940464153, 0.8797307775638197, 0.8943898353263877}}, {{0.8741642977919393, -0.2056513156305888, -0.3049639415937795, -0.41188593599192647}, {0.012967254652470173, -0.7680658239346845, 0.5410717601583555, 0.31978541738683997}, {-0.6865062188603075, -0.24359590935788944, -0.7204746341924977, 0.38989595920498377}, {0.6104555429474274, -0.9899496480150949, 0.04627031157666606, 0.4879689724746332}}, {{-0.7159545935681477, -0.03654339684880403, 0.08910961778734738, 0.154200522748553}, {-0.5901729084828768, 0.2467276212633316, -0.6305858194441027, -0.9786311780355423}, {-0.6779133532565955, -0.6438903067256505, 0.08079412956759291, 0.9476680198374003}, {-0.5091468928374088, -0.21095818741155647, -0.5647957501846304, -0.13598624021292172}, {-0.5336884104973332, 0.7798175033107542, -0.9233462133614492, 0.18475848772587744}}, {{0.31034720625883283, -0.7603219153916208, 0.3049534137919063}, {0.9686456013385898, -0.5865249852789853, -0.2507009994120597}, {-0.07330042486104849, -0.3327795776506606, -0.11357509936546184}, {0.008271133409147868, 0.9979616295358575, 0.2608097278457078}, {0.8191680711780485, 0.015319832139896183, -0.017097588876774816}}};

        Assertions.assertArrayEquals(weighs, result);
    }

    @Test
    void calculatedSumOfWeighsShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[][] sumOfWeighs = neuralNetwork.getSumOfWeighs();
        double[][] result = new double[][] {{0.0, 0.0, 0.0}, {0.9575482519011196, 1.3511186803649662, 0.009445295355018368, 0.0}, {0.7319538333551905, -1.6294872683002835, -0.18512954125048509, 0.6378915692794246, 0.0}, {-1.7743679623878217, 0.44640507963571724, -1.4182014816624, 0.21442939424728413, 0.0}, {1.425866354406599, -0.28128955939581635, 0.07277458481798582}};

        Assertions.assertArrayEquals(sumOfWeighs, result);
    }

    @Test
    void calculatedNeuronsWithSigmoidMseShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[][] layer = neuralNetwork.getLayer();
        double[][] result = new double[][] {{0.5, 0.6, 1.0}, {0.5239204319134367, 0.5337266752984148, 0.5002361323663203, 1.0}, {0.518290680438492, 0.45935271837506103, 0.4953718936503527, 0.5159418838963058, 1.0}, {0.45575681875981083, 0.5111582740583465, 0.46460426914105607, 0.5053605294602914, 1.0}, {0.5355863871961042, 0.4929682246597402, 0.5018193565908171}};

        Assertions.assertArrayEquals(layer, result);
    }

    @Test
    void backPropagationOfErrorWithSigmoidMseShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] answers = new double[] {1.25, 2, 1.15};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, answers);

        double[][] transmittedError = neuralNetwork.getTransmittedError();
        double[][] result = new double[][] {{0.0, 0.0, 0.0}, {3.0097086113043933E-6, -6.589070354466023E-5, 5.2779136381421365E-5, 0.0}, {-9.161787889468784E-4, 0.0011392174356674376, -0.0028244866157575094, -7.087997252274256E-4, 0.0}, {0.03636736130670971, 0.017886309288867697, 0.03135639837920636, -0.08392955423215846, 0.0}, {-1.4288272256077916, -3.0140635506805196, -1.2963612868183656}};

        Assertions.assertArrayEquals(transmittedError, result);
    }

    @Test
    void gradientCalculationWithSigmoidMseShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] answers = new double[] {1.25, 2, 1.15};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, answers);
        neuralNetwork.gradientCalculation();

        double[][][] gradient = neuralNetwork.getGradient();
        double[][][] result = new double[][][] {{{3.753525182477675E-8, -8.198862980139378E-7, 6.597390576235081E-7}, {4.5042302189732096E-8, -9.838635576167252E-7, 7.916868691482097E-7}, {7.50705036495535E-8, -1.6397725960278755E-6, 1.3194781152470161E-6}}, {{-1.1984061158668693E-5, 1.4822869091668613E-5, -3.699198654001938E-5, -9.274428711252794E-6}, {-1.2208367395463413E-5, 1.5100309430167545E-5, -3.7684367293296045E-5, -9.448018629987247E-6}, {-1.1442310776389711E-5, 1.4152787815333167E-5, -3.531973015013788E-5, -8.855169727739852E-6}, {-2.287381905473907E-5, 2.8292214215683955E-5, -7.060611552200598E-5, -1.7701979434894678E-5}}, {{4.675320228393226E-4, 2.3164226340589467E-4, 4.042571158109148E-4, -0.0010873726461168734}, {4.1436613414877097E-4, 2.0530082326779804E-4, 3.582865987732979E-4, -9.63720938331141E-4}, {4.4685778123610407E-4, 2.213990546304057E-4, 3.86380887287917E-4, -0.0010392890845631075}, {4.6541325505118246E-4, 2.3059250394108692E-4, 4.0242509808108116E-4, -0.0010824448764160824}, {9.02065270484459E-4, 4.469350350076085E-4, 7.799814487671264E-4, -0.002097997681912625}}, {{-0.016197476678431216, -0.03433520810536937, -0.014770441834402947}, {-0.018166429732366988, -0.03850897011773333, -0.016565925608524056}, {-0.016511912722643914, -0.03500174569194596, -0.015057175341966371}, {-0.017960379424288134, -0.03807218725651867, -0.016378028805160094}, {-0.03553973525290793, -0.07533668546924019, -0.03240860306730187}}};

        Assertions.assertArrayEquals(gradient, result);
    }

    @Test
    void weighUpdateWithSigmoidMseShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] answers = new double[] {1.25, 2, 1.15};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, answers);
        neuralNetwork.gradientCalculation();
        neuralNetwork.weighUpdate();

        double[][][] weighs = neuralNetwork.getWeigh();
        double[][][] result = {{{0.46175637765305655, -0.17983829502696688, -0.5845703833544716}, {-0.33456589258520786, 0.9355119172345971, -0.9877657146371643}, {0.927409586539365, 0.8797309415410792, 0.8943897033785762}}, {{0.8741654961980552, -0.205652797917498, -0.3049602423951255, -0.41188500854905535}, {0.012968475489209721, -0.7680673339656277, 0.5410755285950848, 0.319786362188703}, {-0.6865050746292298, -0.243597324636671, -0.7204711022194826, 0.3898968447219565}, {0.610457830329333, -0.9899524772365165, 0.04627737218821826, 0.48797074267257673}}, {{-0.7160013467704317, -0.036566561075144624, 0.08906919207576629, 0.1543092600131647}, {-0.5902143450962917, 0.2467070911810048, -0.63062164810398, -0.9785348059417094}, {-0.6779580390347192, -0.6439124466311136, 0.08075549147886411, 0.9477719487458567}, {-0.5091934341629138, -0.2109812466619506, -0.5648359926944384, -0.13587799572528012}, {-0.5337786170243816, 0.7797728098072535, -0.9234242115063259, 0.1849682874940687}}, {{0.311966953926676, -0.7568883945810838, 0.3064304579753466}, {0.9704622443118266, -0.582674088267212, -0.24904440685120732}, {-0.0716492335887841, -0.32927940308146597, -0.11206938183126522}, {0.010067171351576682, 1.0017688482615095, 0.2624475307262238}, {0.8227220447033392, 0.022853500686820204, -0.013856728570044632}}};

        Assertions.assertArrayEquals(weighs, result);
    }

    @Test
    void learningProcessShouldBeCorrect() {

        long maxNumberOfIterations = 30000;
        int numberOfInputs = 40;
        double error = 0.0;

        int[] layers = new int[]{1, 5, 1};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.9;//от 0 до 1
        double beta = 0.1;//от 0 до 1
        double epsilon = 1;//от 0 до 1
        double lambda = 0;// от 0 до 0,9

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);

        double[] inputs = new double[numberOfInputs];
        double[] answers = new double[numberOfInputs];

        double x = 0;

        for (int i = 0; i < numberOfInputs; i++) {

            inputs[i] = x;
            answers[i] = (1 + x * Math.sin(x*10)) / 2;
            x = x + 0.025;
        }

        int counter = 0;

        while (true) {

            long end = System.currentTimeMillis() + 200;
            while (System.currentTimeMillis() < end) {

                for (int i = 0; i < numberOfInputs; i++) {

                    neuralNetwork.updateValueInNeurons(new double[]{inputs[i]});
                    neuralNetwork.backPropagationOfError(neuralNetwork.getResult(), new double[]{answers[i]});
                    neuralNetwork.gradientCalculation();
                    neuralNetwork.weighUpdate();
                }

                counter++;

                error = 0.0;

                for (int j = 0; j < numberOfInputs; j++) {
                    neuralNetwork.updateValueInNeurons(new double[]{inputs[j]});
                    error += answers[j] - neuralNetwork.getResult()[0];
                }

                error = Math.abs(error / numberOfInputs);

                if (counter > maxNumberOfIterations) {
                    break;
                }
            }

            if (counter > maxNumberOfIterations) {
                break;
            }
        }

        Assertions.assertEquals(4.7992542475258637E-4, error);
    }
}
