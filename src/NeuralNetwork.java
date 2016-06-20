import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author auswise
 */
public class NeuralNetwork implements Serializable{
    private static final Random RANDOM = new Random();
    
    public double [][] w_h, w_o; 
    private double [] d_h, d_o, d_h_old, d_o_old;
    private boolean [] mask_o, mask_h, mask_i;
    private int N,L,M;
    private DoubleUnaryOperator [] f_h, f_o;
    
    public NeuralNetwork(int N, int L, int M){
        this.N = N;
        this.L = L;
        this.M = M;
   
        w_h = new double [L][N];
        w_o = new double [M][L];
        
        f_o = new DoubleUnaryOperator[M];
        f_h = new DoubleUnaryOperator[L];  
        
        this.d_h = new double [L];
        this.d_o = new double [M];
        this.d_h_old = new double [L];
        this.d_o_old = new double [M];
        
        
        DoubleUnaryOperator ones = new One();
        DoubleUnaryOperator sigmoid = new Sigmoid();
        this.resetWeights();
        
        for(int k=0;k<M;k++)
            f_o[k] = sigmoid;
            
        for(int j=0;j<L;j++)
            f_h[j] = sigmoid;
            
        f_h[0] = ones;
        
        mask_i = new boolean[N];
        mask_h = new boolean[L];
        mask_o = new boolean[M];
    }
    
    public NeuralNetwork(double [][] w_h, double [][] w_o, DoubleUnaryOperator [] f_h, DoubleUnaryOperator [] f_o){
        N = w_h[0].length;
        L = w_h.length;
        M = w_o.length;
        this.w_h = w_h;
        this.w_o = w_o;
        this.f_h = f_h;
        this.f_o = f_o;
        this.d_h = new double [L];
        this.d_o = new double [M];
        this.d_h_old = new double [L];
        this.d_o_old = new double [M];
        mask_i = new boolean[N];
        mask_h = new boolean[L];
        mask_o = new boolean[M];
    }
    
    public double [] forward_propagation(double [] x){
        double [] i = forward_propagation_hidden(x);
        double [] o = forward_propagation_out(i);
        return o;
    }
    
    private double [] forward_propagation_out(double [] i){
        double [] o = new double[M];
        DoubleUnaryOperator f;
        double net;
        for(int k=0;k<M;k++){
            net = dot(w_o[k], i);
//            if(Double.isInfinite(net))
//                System.out.print("ble");
                
            f = f_o[k];
            o[k] = f.applyAsDouble(net);
//            System.out.println(net);
        }
        
        return o;
    }
    
    private double [] forward_propagation_hidden(double [] x){
        double [] i = new double [L];
        DoubleUnaryOperator f;
        double net;
        for(int j=0;j<L;j++){
            net = dot(w_h[j], x);
            f = f_h[j];
            i[j] = f.applyAsDouble(net);
//            System.out.println(i[j]);
        }
        
        return i;
    }
    
    private double dot(double [] x, double [] y){
        if(x.length != y.length)
            throw new RuntimeException();
        
        int n = x.length;
        double dot = 0;
        double t = 0;
        for(int i=0;i<n;i++){
            
            if(  !(x[i]==0 && Double.isInfinite(y[i])) && !(y[i]==0 && Double.isInfinite(x[i]))){
                t = x[i]*y[i];
                dot += t;
            }
            
            if(Double.isNaN(dot))
                System.out.println(i + " " +t + " " + x[i] + " " + y[i]);
        }
//        System.out.println();
        
        return dot;
    }
   
    public int learn(List<double [][]> training_set, double nabla, double alpha, double lambda, double p){
        int T = 200;
        double eps = 0.01;
        
        boolean success;
        int t;
        double error;
        for(t=0;t<T;t++){
            success = true;
            for(int i=0;i<training_set.size();i++){
                this.backward_propagation(training_set.get(i), nabla, alpha, lambda, p);
            }
            
            error = this.error(training_set);
//            System.out.println(t);
            
            if(error<eps)
                break;
            
            Collections.shuffle(training_set);
        }
        
//        for(int j=0;j<L;j++)
//            for(int i=0;i<N;i++)
//                w_h[j][i] *= p;
//        
//        for(int k=0;k<M;k++)
//            for(int j=0;j<L;j++)
//                w_o[k][j] *= p;
        
        return t;
    }
    
    private void backward_propagation(double [][] pattern, double nabla, double alpha, double lambda, double p){
       double [] x = pattern[0];
       double [] y = pattern[1];
              
       randMask(p);
       
       x = mask(x, mask_i);
       double [] i = forward_propagation_hidden(x);
       i = mask(i, mask_h);
       double [] o = forward_propagation_out(i);
       o = mask(o, mask_o);
       
//       for(int j=0;j<i.length;j++)
//           System.out.print(i[j] + " ");
//       System.out.println();
//       
//       for(int k=0;k<o.length;k++)
//           System.out.print(o[k] + " ");
//       System.out.println();
//       
       d_h_old = d_h;
       d_o_old = d_o;
       
       this.d_h = new double [L];
       this.d_o = new double [M];
       
       DoubleUnaryOperator f;
       double net;
       for(int k=0;k<M;k++){
           if(mask_o[k]){
               net = dot(w_o[k], i);
               f = f_o[k];
               d_o[k] = (y[k]-o[k]) * derivative(f, net);
           }
       }
       
       for(int j=0;j<L;j++){
           if(mask_h[j]){
               net = dot(w_h[j], x);
               f = f_h[j];

               for(int k=0;k<M;k++){
                   d_h[j] += d_o[k]*w_o[k][j];
               }

               d_h[j] *= derivative(f,net);
           }
       }
       
       for(int k=0;k<M;k++)
           for(int j=0;j<L;j++){
               w_o[k][j] += nabla*(d_o[k] + alpha*d_o_old[k])*i[j] + lambda*w_o[k][j]*w_o[k][j];
//               System.out.println(w_o[k][j]);
           }
       
       for(int j=0;j<L;j++)
           for(int ii=0;ii<N;ii++){
               w_h[j][ii] += nabla*(d_h[j] + alpha*d_h_old[j])*x[ii]+ lambda*w_h[j][ii]*w_h[j][ii];
//               if(Double.isNaN(w_h[j][ii]))
//                    System.out.println(d_h[j] + " " + lambda);
           }
    }
    
   private double derivative(DoubleUnaryOperator f, double x){
       double h= 0.01;
       double result = (f.applyAsDouble(x+h) - f.applyAsDouble(x))/h;
//       System.out.println(result);
       return result;
   }
   
   public double error(double [][] pattern){
       double [] x = pattern[0];
       double [] y = pattern[1];
       
       double [] o = this.forward_propagation(x);
       double d;
       double sum = 0;
       for(int k=0;k<M;k++){
//           System.out.println(o[k]);
           d=o[k]-y[k];
           sum += d*d;
       }
       
       double error = 1.0D/2 * sum / y.length;
//       System.out.println(error);
       return error;
   }
   
   public double error(Collection<double [][]> patterns){
       double sum = 0.0;
       for(double [][] pattern : patterns){
           sum += error(pattern);
       }
       
       double error = sum/patterns.size();
//       System.out.println(error);
       return error;
   }

    public void resetWeights(){
//       double b = interval/2;
//       double a = -b;
       
       for(int k=0;k<M;k++)
            for(int j=0;j<L;j++){
                w_o[k][j] = RANDOM.nextDouble() - 0.5;
//                w_o[k][j] = a + (b-a)*w_o[k][j];
            }
        
        
        for(int j=0;j<L;j++)
            for(int i=0;i<N;i++){
                w_h[j][i] = RANDOM.nextDouble() - 0.5;
//                w_h[j][i] = a + (b-a)*w_h[j][i];
            }
    }
    
    private double[] mask(double[] x, boolean[] mask){
        double[] y = new double[x.length];
        
        for(int i=0;i<x.length;i++)
            y[i] = mask[i] ? x[i] : 0;
        
        return y;
    }
    
    private void randMask(double p){
        for(int i=0;i<N;i++)
            mask_i[i] = RANDOM.nextDouble()<=p;
        
        for(int j=0;j<L;j++)
            mask_h[j] = RANDOM.nextDouble()<=p;
        
        for(int k=0;k<M;k++)
            mask_o[k] = RANDOM.nextDouble()<=p;
    }
    
    public BufferedImage visualizeHiddenLayer(){
        int width = (int)Math.sqrt(L);
        System.out.println(width);
        System.out.println(L/width);
        System.out.println(L%width);
        
        int height = L/width + (L%width>0 ? 1 : 0);
        System.out.println(height);
        BufferedImage image = new BufferedImage(width*28, height*28, BufferedImage.TYPE_INT_ARGB);
        
        Graphics g = image.getGraphics();
        Graphics2D g2d = (Graphics2D)g;
        
        BufferedImage neuronImage;
        int j = 0;
        for(int x=0;x<width && j<L;x++){
            
            for(int y=0;y<height && j<L;y++){
                neuronImage = this.visualizeHiddenNeuron(j);
                g2d.drawImage(neuronImage, x*28, y*28, null);
                j++;
            }
        }
        return image;
        
    }
    
    public BufferedImage visualizeHiddenNeuron(int j){
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_ARGB);
        
        Graphics g = image.getGraphics();
        Graphics2D g2d = (Graphics2D)g;
        
        double [] w = normalize(w_h[j]);
        int x, y;
        int grey;
        for(int i=0;i<N;i++){
            x = i/28;
            y = i%28;
            grey = (int) (255.0 * w[i]);
//            System.out.println(grey);
            g2d.setColor(new Color(grey, grey, grey));
            g2d.fillRect(x, y, 1, 1);
        }
        
        return image;
    }
    
    private double[] normalize(double [] x){
        double norm = norm(x);
        double [] xprim = new double [x.length];
        
        double min = min(x);
        
        for(int i=0;i<x.length;i++)
            xprim[i] = x[i] - min;
        
        
        for(int i=0;i<x.length;i++)
            xprim[i]/=norm;
        
        return xprim;
    }
    
    private double norm(double [] x){
        return Math.sqrt(this.dot(x, x));
    }
    
    private double min(double [] x){
        double min = x[0];
        for(int i=0;i<x.length;i++)
            if(x[i]<min)
                min=x[i];
        
        return min;
    }
} 
