
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
public class One implements DoubleUnaryOperator, Serializable{

        @Override
        public double applyAsDouble(double x) {
            return 1;
        }
        
    }