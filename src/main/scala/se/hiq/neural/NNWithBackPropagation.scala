package se.hiq.neural

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean

/**
  * Created by balp on 2016-03-01.
  */
object NNWithBackPropagation extends App {
  var X : DenseMatrix[Double] = DenseMatrix( (0.0,0.0,1.0), (0.0,1.0,1.0), (1.0,0.0,1.0), (1.0,1.0,1.0) )
  var Y : DenseMatrix[Double] =  DenseMatrix((0.0),(1.0),(1.0),(0.0))
  var syn0 = DenseMatrix.rand(3,4).mapValues( 2 * _ - 1)
  var syn1 = DenseMatrix.rand(4,1).mapValues( 2 * _ - 1)

  println(X)
  println(Y)
  println("---- Before.....")
  println("  -- syn0")
  println(syn0)
  println("  -- syn1")
  println(syn1)
  println("---- train.....")
  var train = 0
  for( train <- 0 to 60000 ) {
    val l0 = X
    val l1 = sigmoid(l0 * syn0)
    val l2 = sigmoid(l1 * syn1)
    val l2_error = Y - l2
    if(train % 10000 == 0) println(mean(abs(l2_error)))
    val l2_delta = l2_error :* sigmoid_derivate(l2)
    val l1_error  = l2_delta * syn1.t
    val l1_delta  = l1_error :* sigmoid_derivate(l1)
    syn1 :+= l1.t * l2_delta
    syn0 :+= l0.t * l1_delta
  }
  println("  -- syn0")
  println(syn0)
  println("  -- syn1")
  println(syn1)


  def sigmoid_derivate(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    X :* ( X mapValues(1 - _))
  }
  def sigmoid(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    1.0 :/ X.map( x => math.exp(-x) + 1)
  }
}
