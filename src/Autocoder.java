/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author auswise
 */
public class Autocoder extends NeuralNetwork {
    
    public Autocoder(int L){
        super(784 + 1, L, 784);
    }
}
