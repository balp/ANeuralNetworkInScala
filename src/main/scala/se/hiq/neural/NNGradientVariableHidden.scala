package se.hiq.neural

import breeze.linalg._
import breeze.numerics._
import breeze.stats._

/**
  * Created by balp on 2016-03-02.
  */
object NNGradientVariableHidden extends App {
  val alphas = Array(0.001,0.01,0.1,1,10,100,1000)
  val hiddenSize = 32
  var X : DenseMatrix[Double] = DenseMatrix( (0.0,0.0,1.0), (0.0,1.0,1.0), (1.0,0.0,1.0), (1.0,1.0,1.0) )
  var Y : DenseMatrix[Double] =  DenseMatrix((0.0),(1.0),(1.0),(0.0))

  var alpha = 0.0
  for ( alpha <- alphas)
  {
    println("Training with alpha " + alpha)
    var synapse_0 = DenseMatrix.rand(3,hiddenSize).mapValues( 2 * _ - 1)
    var synapse_1 = DenseMatrix.rand(hiddenSize,1).mapValues( 2 * _ - 1)

    //var layer_1: DenseMatrix[Double] = X
    //var layer_2: DenseMatrix[Double] = X
    var train = 0
    for (train <- 0 to 60000) {
      val layer_0 = X
      val layer_1 = sigmoid(layer_0 * synapse_0)
      val layer_2 = sigmoid(layer_1 * synapse_1)
      val layer_2_error = layer_2 - Y
      if (train % 10000 == 0) println("Error after " + train + " iterations: " + mean(abs(layer_2_error)))
      val layer_2_delta = layer_2_error :* sigmoid_output_to_derivative(layer_2)
      val layer_1_error = layer_2_delta * synapse_1.t
      val layer_1_delta = layer_1_error :* sigmoid_output_to_derivative(layer_1)
      synapse_1 :-= alpha * (layer_1.t * layer_2_delta)
      synapse_0 :-= alpha * (layer_0.t * layer_1_delta)
    }
    //println("Output after training")
    //println(layer_1)
  }
  def sigmoid( X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    1.0 :/ X.map( x => 1 + math.exp(-x))
  }
  def sigmoid_output_to_derivative(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    X :* ( X mapValues(1 - _))
  }
}
