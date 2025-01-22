package org.taror.java_machine_learning;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.taror.java_machine_learning.libraries.neural_network.NeuralNetwork;
import org.taror.java_machine_learning.libraries.neural_network.models.ActFun;
import org.taror.java_machine_learning.libraries.neural_network.models.LossFun;

class NeuralNetworkSigmoidCrossEntropyTests {

    @Test
    void theSizeOfTheNeuralNetworkShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.CROSS_ENTROPY;

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
        LossFun lossFun = LossFun.CROSS_ENTROPY;

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
        LossFun lossFun = LossFun.CROSS_ENTROPY;

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
    void calculatedNeuronsWithSigmoidSoftmaxCrossEntropyShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.CROSS_ENTROPY;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[][] layer = neuralNetwork.getLayer();
        double[][] result = new double[][] {{0.5, 0.6, 1.0}, {0.5239204319134367, 0.5337266752984148, 0.5002361323663203, 1.0}, {0.518290680438492, 0.45935271837506103, 0.4953718936503527, 0.5159418838963058, 1.0}, {0.45575681875981083, 0.5111582740583465, 0.46460426914105607, 0.5053605294602914, 1.0}, {0.6945308219061257, 0.1259746413288972, 0.1794945367649771}};

        Assertions.assertArrayEquals(layer, result);
    }

    @Test
    void backPropagationOfErrorWithSigmoidSoftmaxAndCrossEntropyShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.CROSS_ENTROPY;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] reward = new double[] {0.09, 0.005, -0.009};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, reward);

        double[][] transmittedError = neuralNetwork.getTransmittedError();
        double[][] result = new double[][] {{0.0, 0.0, 0.0}, {8.445728402543775E-7, -1.8339455396729618E-6, 3.3518723226684515E-6, 0.0}, {-6.71780963847457E-5, -4.957561400511887E-5, -1.5254532872042775E-4, -8.696748955754507E-5, 0.0}, {0.003804597278464346, 0.011610352228980513, -0.0026436457241130677, 0.004370988914584292, 0.0}, {0.6045308219061257, 0.12097464132889721, 0.1884945367649771}};

        Assertions.assertArrayEquals(transmittedError, result);
    }

    @Test
    void gradientCalculationWithSigmoidSoftmaxAndCrossEntropyShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.CROSS_ENTROPY;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] reward = new double[] {0.09, 0.005, -0.009};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, reward);
        neuralNetwork.gradientCalculation();

        double[][][] gradient = neuralNetwork.getGradient();
        double[][][] result = new double[][][] {{{1.0532997820535136E-8, -2.282001463624517E-8, 4.189839468858794E-8}, {1.2639597384642163E-8, -2.73840175634942E-8, 5.027807362630553E-8}, {2.1065995641070272E-8, -4.564002927249034E-8, 8.379678937717588E-8}}, {{-8.78721953957408E-7, -6.450505527124759E-7, -1.9978691756892926E-6, -1.1379431359673229E-6}, {-8.951690341309362E-7, -6.571239942700258E-7, -2.0352633870900647E-6, -1.159242071206032E-6}, {-8.38998528970628E-7, -6.158904559060213E-7, -1.9075536828573698E-6, -1.0865013817272666E-6}, {-1.6772049731830124E-6, -1.2311994597283668E-6, -3.8133064755515873E-6, -2.1719770153100965E-6}}, {{4.891119393260015E-5, 1.5036351132173545E-4, -3.4082759847971276E-5, 5.662956065573768E-5}, {4.3349206805920934E-5, 1.3326476874255758E-4, -3.0207003476590503E-5, 5.018987145512755E-5}, {4.674834349441322E-5, 1.4371444471343128E-4, -3.257562198964409E-5, 5.412540444463853E-5}, {4.868953753876745E-5, 1.496820919777398E-4, -3.392830314731953E-5, 5.637292606577903E-5}, {9.43701976103826E-5, 2.901142486191777E-4, -6.575993189600876E-5, 1.0926216270728058E-4}}, {{0.006853084623336513, 0.0013781028222056824, 0.0021476633247996023}, {0.007686140423686372, 0.0015456239623368863, 0.0024087316594631706}, {0.006986121198254766, 0.0014048554583437377, 0.0021893551743115643}, {0.007598961400313596, 0.001528092928539632, 0.002381410941643346}, {0.015036713311245418, 0.0030237678636508974, 0.004712301026331865}}};

        Assertions.assertArrayEquals(gradient, result);
    }

    @Test
    void weighUpdateWithSigmoidSoftmaxAndCrossEntropyShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.SIGMOID;
        LossFun lossFun = LossFun.CROSS_ENTROPY;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] reward = new double[] {0.09, 0.005, -0.009};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, reward);
        neuralNetwork.gradientCalculation();
        neuralNetwork.weighUpdate();

        double[][][] weighs = neuralNetwork.getWeigh();
        double[][][] result = {{{0.4617563803532819, -0.1798383747335952, -0.5845703215704053}, {-0.33456588934493736, 0.9355118215866431, -0.9877656404962848}, {0.9274095919398158, 0.8797307821278226, 0.8943898269467088}}, {{0.8741643856641347, -0.20565125112553356, -0.3049637418068619, -0.4118858221976129}, {0.012967344169373587, -0.7680657582222852, 0.5410719636846942, 0.3197855333110471}, {-0.6865061349604547, -0.24359584776884385, -0.7204744434371294, 0.38989606785512193}, {0.6104557106679248, -0.9899495248951489, 0.04627069290731361, 0.48796918967233477}}, {{-0.715959484687541, -0.03655843319993621, 0.08911302606333218, 0.15419485979248743}, {-0.5901772434035575, 0.24671429478645734, -0.6305827987437551, -0.978636197022688}, {-0.6779180280909449, -0.6439046781701219, 0.08079738712979187, 0.9476626072969558}, {-0.5091517617911627, -0.21097315562075425, -0.5647923573543157, -0.1359918775055283}, {-0.5336978475170941, 0.7797884918858924, -0.9233396373682595, 0.18474756150960672}}, {{0.3096618977964992, -0.7604597256738413, 0.30473864745942636}, {0.9678769872962213, -0.586679547675219, -0.25094187257800604}, {-0.07399903698087396, -0.332920063196495, -0.11379403488289301}, {0.007511237269116509, 0.9978088202430035, 0.26057158675154346}, {0.817664399846924, 0.015017455353531094, -0.017568818979408006}}};

        Assertions.assertArrayEquals(weighs, result);
    }
}
