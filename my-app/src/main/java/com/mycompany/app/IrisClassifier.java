package com.mycompany.app;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

//imports from dl4j


public class IrisClassifier {

    private static final int CLASSES_COUNT = 3; //num classes that can be identified
    private static final int FEATURES_COUNT = 4; //num of collumns of data
    

    public static void main(String[] args) throws IOException, InterruptedException {

        DataSet allData;
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
            //input and inizialize the file, split by  ','
            
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
            allData = iterator.next();
            //this is how we will iterate through the file
        }

        allData.shuffle(42);
        //shuffling the data to get rid of class ordering
        
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);
        //normalizing the distribution of the data, which are numbers
        
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        //split the data into two groups, one for train at 65%, another for test data at 35%
        //this is done to make sure the data is classified correctly
        
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .iterations(1000)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .regularization(true)
                .learningRate(0.1).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3)
                        .build())
                //first layer, same amount of nodes as columns in train
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                //second layer, 3 nodes
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES_COUNT).build())
                //third layer, same amount of nodes as number of classes
                .backpropType(BackpropType.Standard).pretrain(false) //backprop algorithm for training
                .build(); //pretraining disabled
        
        		//configuring the network:
        			//iterations, or how many passes on the training set until the network converges to a good result
        			//activation, or function that runs inside a node to determine its output, we are using hyperbolic tangent function
        			//weightinit, or how many ways to set up initial weights for network, we use Gaussian dist.
        			//learning rate, or how much to change the model in response to estimated error each time the model weights are updated
        			//l2 regulatization, or "penalizing" the network for too large weights and prevents overfitting
        
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
	    model.init();
	    model.fit(trainingData);
	    //create, initialize, and run neural network
	    
	    INDArray output = model.output(testData.getFeatureMatrix());
	    Evaluation eval = new Evaluation(3);
	    eval.eval(testData.getLabels(), output);
	    //test the trained model by using the 35% of the dataset and verifying the results with evaluation metrics
	    
	    System.out.println(eval.stats());
    	}
    
	    
}
