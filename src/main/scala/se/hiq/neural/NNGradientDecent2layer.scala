package se.hiq.neural

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean

/**
  * Created by balp on 2016-03-02.
  */
object NNGradientDecent2layer extends App {
  val X = DenseMatrix( (0.0,1.0), (0.0,1.0), (1.0,0.0), (1.0,0.0) )
  val Y = DenseMatrix( (0.0), (0.0), (1.0), (1.0) )
  var synapse_0 = DenseMatrix.rand(2,1).mapValues( 2 * _ - 1)
  var layer_1 : DenseMatrix[Double] = X
  var train = 0
  for( train <- 0 to 60000 ) {
    val layer_0 = X
    layer_1 = sigmoid(layer_0 * synapse_0)
    val layer_1_error = layer_1 - Y
    if(train % 10000 == 0) println("Error after " + train + " iterations: " + mean(abs(layer_1_error)))
    val layer_1_delta = layer_1_error :* sigmoid_output_to_derivative(layer_1)
    synapse_0 :-= layer_0.t * layer_1_delta
  }
  println("Output after training")
  println(layer_1)

  def sigmoid( X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    1.0 :/ X.map( x => 1 + math.exp(-x))
  }
  def sigmoid_output_to_derivative(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    X :* ( X mapValues(1 - _))
  }
}
