
import java.io.Serializable;
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
public class Sigmoid implements DoubleUnaryOperator, Serializable{
        
        @Override
        public double applyAsDouble(double x) {
            double y = 1.0D/(1+Math.exp(-x));
//            System.out.println(y);
            return y;
        }
        
        
    }