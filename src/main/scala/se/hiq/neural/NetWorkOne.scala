package se.hiq.neural

import breeze.linalg._

/**
  * Created by balp on 2016-03-01.
  */
object NetWorkOne extends App {
  var X : DenseMatrix[Double] = DenseMatrix( (0.0,0.0,1.0), (0.0,1.0,1.0), (1.0,0.0,1.0), (1.0,1.0,1.0) )
  var Y = Transpose(Vector(0.0,1.0,1.0,0.0))
  var syn0 = DenseMatrix.rand(3,4)
  var syn1 = DenseMatrix.rand(3,4)

  println(X)
  println(Y)
  println(syn0)
  println(syn1)
  var train = 0
  for( train <- 1 to 500 ) {
    //    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    var l0 = X
    val l1 = nonlin(l0 * syn0)
    //    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    val l2 = nonlin(l1 * syn1)
//    l2_delta = (y - l2)*(l2*(1-l2))
    var l2_error = Y - l2
    var l2_delta = nonlindelta(l2) * l2_error
//    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    var l1_error  = l2_delta * syn1.t
    var l1_delta  = l1_error * nonlindelta(l1)
//    syn1 += l1.T.dot(l2_delta)
    syn1 :+= l1.t * l2_delta
//    syn0 += X.T.dot(l1_delta)
    syn0 :+= l0.t * l1_delta
  }
  println(X)
  println(Y)
  println(syn0)
  println(syn1)


  def nonlindelta( X : Vector[Double]) : Vector[Double] = {
    // x*(1-x)
    X * ( 1 - X)
  }

  def nonlin(X : Vector[Double]) : Vector[Double] = {
    //1/(1+np.exp(-x))
    var n  = -X
    var e  = n.map( x => math.exp(x) )
    var tmp  = e + 1
    var tmp2  = 1.0 / tmp
    return tmp2
  }
}
