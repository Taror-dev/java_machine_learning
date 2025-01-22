package org.taror.java_machine_learning;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.taror.java_machine_learning.libraries.neural_network.NeuralNetwork;
import org.taror.java_machine_learning.libraries.neural_network.models.ActFun;
import org.taror.java_machine_learning.libraries.neural_network.models.LossFun;

class NeuralNetworkTanhMseTests {

    @Test
    void theSizeOfTheNeuralNetworkShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.TANH;
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
        ActFun actFun = ActFun.TANH;
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
        ActFun actFun = ActFun.TANH;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[][] sumOfWeighs = neuralNetwork.getSumOfWeighs();
        double[][] result = new double[][] {{0.0, 0.0, 0.0}, {0.9575482519011196, 1.3511186803649662, 0.009445295355018368, 0.0}, {0.694999116747412, -1.112959779941464, 0.08914054761849291, 0.4919630705012664, 0.0}, {-0.5490239818631145, 0.7338253300676535, -0.8743130222979758, 0.3056911180402824, 0.0}, {0.8797459489395412, 0.07357658214995055, -0.03431255042873714}};

        Assertions.assertArrayEquals(sumOfWeighs, result);
    }

    @Test
    void calculatedNeuronsShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.TANH;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[][] layer = neuralNetwork.getLayer();
        double[][] result = new double[][] {{0.5, 0.6, 1.0}, {0.09546323633847538, 0.1342956620409206, 9.445292546189895E-4, 1.0}, {0.06938822708998257, -0.11083871001696907, 0.008913818664650085, 0.049156655879036545, 1.0}, {-0.05484730100507029, 0.0732510945651232, -0.08720919958958166, 0.030559593383172313, 1.0}, {0.08774833467420838, 0.007357525448597852, -0.0034312415769639404}};

        Assertions.assertArrayEquals(layer, result);
    }

    @Test
    void backPropagationOfErrorShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.TANH;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] answers = new double[] {0.09, 0.005, -0.009};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, answers);

        double[][] transmittedError = neuralNetwork.getTransmittedError();
        double[][] result = new double[][] {{0.0, 0.0, 0.0}, {-2.1130077687783735E-6, 1.4404525949856522E-5, -8.979235489735034E-6, 0.0}, {2.4287417427814668E-5, -7.33497990436301E-5, 1.4365010884126748E-4, 3.249913804493233E-5, 0.0}, {-1.5752135861963527E-4, -9.886043201007806E-4, -2.506363010722956E-4, 7.572962844409318E-4, 0.0}, {-0.004503330651583232, 0.004715050897195703, 0.011137516846072119}};

        Assertions.assertArrayEquals(transmittedError, result);
    }

    @Test
    void gradientCalculationShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.TANH;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] answers = new double[] {0.09, 0.005, -0.009};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, answers);
        neuralNetwork.gradientCalculation();

        double[][][] gradient = neuralNetwork.getGradient();
        double[][][] result = new double[][][] {{{-1.046875722031331E-7, 7.072367822570653E-7, -4.489613739520088E-7}, {-1.2562508664375971E-7, 8.486841387084783E-7, -5.387536487424105E-7}, {-2.093751444062662E-7, 1.4144735645141306E-6, -8.979227479040176E-7}}, {{2.3073922605229276E-7, -6.916185523488728E-7, 1.3712214683818112E-6, 3.094976152398243E-7}, {3.24599063577035E-7, -9.72954353214612E-7, 1.929005363364571E-6, 4.353946999172324E-7}, {2.282972561518914E-9, -6.8429898334324406E-9, 1.3567094948008056E-8, 3.062221259630666E-9}, {2.4170480166226662E-6, -7.244868065195939E-6, 1.4363869495477767E-5, 3.242060788118125E-6}}, {{-1.0897247504107186E-6, -6.822942588602967E-6, -1.7258940704307314E-6, 5.249837309012307E-6}, {1.740694216793513E-6, 1.0898767510804006E-5, 2.756892349251206E-6, -8.385935475412742E-6}, {-1.399892925199778E-7, -8.764955604825551E-7, -2.2171350131577813E-7, 6.744097630678063E-7}, {-7.719929850541183E-7, -4.8335727107631895E-6, -1.2226739961782533E-6, 3.7191387767464865E-6}, {-1.5704749870573358E-5, -9.832997433058755E-5, -2.4873010059654556E-5, 7.565890539621835E-5}}, {{2.4509372292825815E-5, -2.5859381654908876E-5, -6.108555469627288E-5}, {-3.273339461840841E-5, 3.453639424890265E-5, 8.158256945416592E-5}, {3.8970791651221195E-5, -4.1117355543122124E-5, -9.712824941112601E-5}, {-1.3656031155959827E-5, 1.4408223814718374E-5, 3.40353978937097E-5}, {-4.468656040259862E-4, 4.71479565649335E-4, 0.0011137385719422349}}};

        Assertions.assertArrayEquals(gradient, result);
    }

    @Test
    void weighUpdateShouldBeCorrect() {

        int[] layers = new int[] {2, 3, 4, 4, 3};
        ActFun actFun = ActFun.TANH;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0.1;
        double beta = 0.1;
        double epsilon = 0.1;
        double lambda = 0.1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);
        neuralNetwork.updateValueInNeurons(new double[] {0.5, 0.6});

        double[] answers = new double[] {0.09, 0.005, -0.009};
        double[] outputs = neuralNetwork.getResult();

        neuralNetwork.backPropagationOfError(outputs, answers);
        neuralNetwork.gradientCalculation();
        neuralNetwork.weighUpdate();

        double[][][] weighs = neuralNetwork.getWeigh();
        double[][][] result = {{{0.4617563918753389, -0.1798384477392749, -0.5845702724844285}, {-0.334565875518469, 0.9355117339798276, -0.9877655815931126}, {0.9274096149839297, 0.8797306361164632, 0.8943899251186626}}, {{0.8741642747180167, -0.2056512464687336, -0.30496407871592635, -0.411885966941688}, {0.012967222192563816, -0.7680657266392492, 0.5410715672578191, 0.31978537384737}, {-0.6865062190886048, -0.2435959086735905, -0.7204746355492072, 0.38989595889876166}, {0.6104553012426259, -0.9899489235282883, 0.04626887518971651, 0.4879686482685544}}, {{-0.7159544845956727, -0.03654271455454517, 0.08910979037675443, 0.1541999977648221}, {-0.5901730825522985, 0.24672653138658052, -0.6305860951333376, -0.978630339441995}, {-0.6779133392576663, -0.6438902190760944, 0.08079415173894304, 0.947667952396424}, {-0.5091468156381103, -0.2109577040542854, -0.5647956279172307, -0.1359866121267994}, {-0.5336868400223461, 0.7798273363081873, -0.9233437260604432, 0.1847509218353378}}, {{0.3103447553216036, -0.7603193294534553, 0.30495952234737594}, {0.9686488746780517, -0.5865284389184102, -0.25070915766900514}, {-0.07330432194021361, -0.33277546591510626, -0.11356538654052074}, {0.008272499012263464, 0.997960188713476, 0.26080632430591844}, {0.8192127577384511, 0.01527268418333125, -0.01720896273396904}}};

        Assertions.assertArrayEquals(weighs, result);
    }

    @Test
    void learningProcessShouldBeCorrect() {

        int numberOfInputs = 40;
        long maxNumberOfIterations = 52100;
        double error = 0;

        int[] layers = new int[]{1, 5, 1};
        ActFun actFun = ActFun.TANH;
        LossFun lossFun = LossFun.MSE;

        double alpha = 0;
        double beta = 0.1;
        double epsilon = 1;
        double lambda = 0;

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);

        double[] inputs = new double[numberOfInputs];
        double[] answers = new double[numberOfInputs];

        double x = 0;

        for (int i = 0; i < numberOfInputs; i++) {

            inputs[i] = x;
            answers[i] = x * Math.sin(x)/10;
            x = x + 0.25;
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

        Assertions.assertEquals(1.794310891081695E-4, error);
    }
}
