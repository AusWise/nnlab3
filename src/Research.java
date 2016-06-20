import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import javax.imageio.ImageIO;
import net.vivin.digit.DigitImage;
import net.vivin.service.DigitImageLoadingService;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author auswise
 */
public class Research {

    private static final Random RANDOM = new Random();

    private double nabla, alpha, interval;

    int N = 784 + 1;
    int M = 784;
    
    private NeuralNetwork network;

    private List<double[][]> trainingSet, validationSet;

    public Research(double nabla, double alpha, int trainingSetSize, int validationSetSize, double interval) throws IOException {
        trainingSet = getSet(trainingSetSize);
        validationSet = getSet(validationSetSize);
        this.nabla = nabla;
        this.alpha = alpha;
        this.interval = interval;
        
    }

    public void research(int[] Ls) {
        
        for(double i=0.0;i<1.0;i+=0.1)
            learn(784 * 2, 0 , i);
        
        learn(784, 0, 0);
    }

    public void learn(int L, double lambda, double p) {
        int m = 3;
        double[] ts = new double[m];
        double[] es = new double[m];

        network = new Autocoder(L);
        for (int i = 0; i < m; i++) {
            ts[i] = network.learn(trainingSet, nabla, alpha, lambda, p);
//            System.out.println(ts[i]);
            es[i] = network.error(validationSet);
            network.resetWeights();
        }

        System.out.println(L + " " + avg(ts) + " " + avg(es));
    }

    public NeuralNetwork network(int L, double interval) {
        double[][] w_h = new double[L][N];
        double[][] w_o = new double[M][L];
        DoubleUnaryOperator[] f_o = new DoubleUnaryOperator[M];
        DoubleUnaryOperator[] f_h = new DoubleUnaryOperator[L];

        double b = interval / 2;
        double a = -b;

        DoubleUnaryOperator sigmoid = x -> 1.0D / (1 + Math.exp(-x));
        DoubleUnaryOperator ones = x -> 1.0D;
        for (int k = 0; k < M; k++) {
            f_o[k] = sigmoid;
            for (int j = 0; j < L; j++) {
                w_o[k][j] = RANDOM.nextDouble();
                w_o[k][j] = a + (b - a) * w_o[k][j];
            }
        }

        for (int j = 0; j < L; j++) {
            f_h[j] = sigmoid;
            for (int i = 0; i < N; i++) {
                w_h[j][i] = RANDOM.nextDouble();
                w_h[j][i] = a + (b - a) * w_h[j][i];
            }
        }

        f_h[0] = ones;

        return new NeuralNetwork(w_h, w_o, f_h, f_o);
    }

    public List<double[][]> getSet(int size) throws IOException {
        DigitImageLoadingService dils
                = new DigitImageLoadingService(
                        "/home/auswise/Documents/NetBeansProjects/nnlab3/train-labels.idx1-ubyte",
                        "/home/auswise/Documents/NetBeansProjects/nnlab3/train-images.idx3-ubyte");
        List<DigitImage> digitImages = dils.loadDigitImages();

        List<double[][]> trainingSet = new LinkedList<double[][]>();
        double[][] pattern;

        DigitImage image;
        for (int j = 0; j < size; j++) {
            pattern = new double[2][];
            image = digitImages.get(RANDOM.nextInt(6000));
            pattern[0] = new double[N];
            pattern[1] = new double[M];

            double[] data = image.getData();
            pattern[0][0] = 1;
            for (int i = 0; i < 784; i++) {
                pattern[0][i + 1] = data[i];
                pattern[1][i] = data[i];
            }

            trainingSet.add(pattern);
        }

        return trainingSet;
    }

    protected double error(NeuralNetwork network, Collection<double[][]> validation_set) {
        double sum = 0;
        double er;
        for (double[][] pattern : validation_set) {
            er = network.error(pattern);
            sum += er;
        }

        return sum / validation_set.size();
    }

    private double avg(double[] xs) {
        double sum = 0;
        for (double x : xs) {
            sum += x;
        }

        return sum / xs.length;
    }

    public NeuralNetwork getNetwork(){
        return network;
    }
    
    public void setNetwork(NeuralNetwork network){
        this.network = network;
    }
    
    public static void saveNetwork(File file, NeuralNetwork network) throws IOException{
        ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(file));
        stream.writeObject(network);
    }
    
    public void saveNetwork(File file) throws IOException{
        Research.saveNetwork(file, this.getNetwork());
    }
    
    public static NeuralNetwork _loadNetwork(File file) throws IOException, ClassNotFoundException{
        ObjectInputStream stream = new ObjectInputStream(new FileInputStream(file));
        return (NeuralNetwork) stream.readObject();
    } 
    
    public void loadNetwork(File file) throws IOException, ClassNotFoundException{
        this.setNetwork(Research._loadNetwork(file));
    } 
    
    public BufferedImage visualizeHiddenLayer(){
        return network.visualizeHiddenLayer();
    }
    
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        
//        
        int L = 784 * 2;
        double nabla = 0.5;
        double alpha = 0.0;
        double interval = 1.0;

        Research r = new Research(nabla, alpha, 100, 10, interval);
//
//        int[] Ls = new int[16];
//        for (int i = 1; i < Ls.length; i++) {
//            Ls[i - 1] = 100 * i;
//        }
//
//        r.research(Ls);
      r.learn(L, 0.0, 0.5);
      
      r.saveNetwork(new File("/home/auswise/Documents/NetBeansProjects/nnlab3/network.data"));
//        
      r.loadNetwork(new File("/home/auswise/Documents/NetBeansProjects/nnlab3/network.data"));
    
        
      BufferedImage image = r.visualizeHiddenLayer();
      ImageIO.write(image, "png", new File("/home/auswise/Documents/NetBeansProjects/nnlab3/test.png"));
      
    }

//    public static void main(String [] args) throws IOException{
//        DigitImageLoadingService dils = 
//                new DigitImageLoadingService(
//                        "/home/auswise/Documents/NetBeansProjects/nnlab3/train-labels.idx1-ubyte", 
//                        "/home/auswise/Documents/NetBeansProjects/nnlab3/train-images.idx3-ubyte");
//        List<DigitImage> digitImages = dils.loadDigitImages();
//        
//        DigitImage digitImage = digitImages.get(0);
//        
//        JFrame frame = new JFrame(){
//            
//            @Override
//            public void paint(Graphics g){
//                Graphics2D g2d = (Graphics2D)g;
//                int grey;
//                int x,y;
//                double data;
//                for(int i=0;i<digitImage.getData().length;i++){
//                    data = digitImage.getData()[i];
//                    x = i%28;
//                    y = i/28;
//                    grey = (int)(255.0*data);
////                    System.out.println(x + " " + y);
//                    g2d.setColor(new Color(grey,grey,grey));
//                    g2d.drawRect(x,y,1,1);
//                }
//            }
//        };
//        frame.setSize(300,300);
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setVisible(true);
//    }
//    public static void main(String [] args) throws IOException{
//        
//        List<double [][]> trainingSet = getSet(140);
//        List<double [][]> validationSet = getSet(60);
//        NeuralNetwork network = network(392, 1.0);
//        System.out.println(network.learn(trainingSet, 0.5, 0, 0));
//        
//        double [] xdata = trainingSet.get(0)[0];
//        double [] ydata = network.forward_propagation(xdata);
//        
//        JFrame frame = new JFrame(){
//          
//            @Override
//            public void paint(Graphics g){
//                Graphics2D g2d = (Graphics2D)g;
//                int grey;
//                int x,y;
//                for(int i=0;i<ydata.length;i++){
//                    x = i%28;
//                    y = i/28;
//                    grey = (int)(255.0*ydata[i]);
////                    System.out.println(x + " " + y);
//                    g2d.setColor(new Color(grey,grey,grey));
//                    g2d.drawRect(x,y,1,1);
//                    grey = (int)(255.0*xdata[i+1]);
//                    g2d.setColor(new Color(grey,grey,grey));
//                    g2d.drawRect(x+28,y,1,1);
//                }
//            }
//        };
//        frame.setSize(300,300);
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setVisible(true);
//    }
}
