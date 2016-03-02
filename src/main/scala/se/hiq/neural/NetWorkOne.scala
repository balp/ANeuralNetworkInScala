package se.hiq.neural

import breeze.linalg._

/**
  * Created by balp on 2016-03-01.
  */
object NetWorkOne extends App {
  var X = DenseMatrix( (0,0,1), (0,1,1), (1,0,1), (1,1,1) )
  var Y = Transpose(Vector(0,1,1,0))
  var syn0 = DenseMatrix.rand(3,4)
  var syn1 = DenseMatrix.rand(3,4)

  println(X)
  println(Y)
  println(syn0)
  println(syn1)
  var train = 0
  for( train <- 1 to 60000 ) {
//    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
//    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
//    l2_delta = (y - l2)*(l2*(1-l2))
//    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
//    syn1 += l1.T.dot(l2_delta)
//    syn0 += X.T.dot(l1_delta)

  }

  def nonlin(X : DenseMatrix) : DenseMatrix = {
    //1/(1+np.exp(-x))
    var n = -X
    var e = n.map( f )
  }
}
