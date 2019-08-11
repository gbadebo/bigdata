import breeze.linalg._
import org.apache.spark.HashPartitioner
//spark-shell -i als.scala to run this code
//SPARK_SUBMIT_OPTS="-XX:MaxPermSize=4g" spark-shell -i als.scal


//val ratings = sc.textFile("hdfs://cshadoop1.utdallas.edu/hw4fall/ratings.dat").map(l => (l.split("::")(0),l.split("::")(1),l.split("::")(2))) //loads ratings from file

val ratings = sc.textFile("ratings.dat").map(l => (l.split("::")(0),l.split("::")(1),l.split("::")(2)))
//val ratings = sc.parallelize(theratings.take(100))

val itemCount = ratings.map(x=>x._2).distinct.count // counts unique items

val userCount = ratings.map(x=>x._1).distinct.count // counts unique user


val items = ratings.map(x=>x._2).distinct   // get distinct items

val users = ratings.map(x=>x._1).distinct  // get distinct user

val k= 5  // latent factor

val itemMatrix = items.map(x=> (x,DenseVector.zeros[Double](k)))   

var myitemMatrix = itemMatrix.map(x => (x._1,x._2(0 to k-1):=0.5)).partitionBy(new HashPartitioner(10)).persist  // generated a latent vector for each item

val userMatrix = users.map(x=> (x,DenseVector.zeros[Double](k)))

var myuserMatrix = userMatrix.map(x => (x._1,x._2(0 to k-1):=0.5)).partitionBy(new HashPartitioner(10)).persist // generate latent vector for each user
val ratingByItem = sc.broadcast(ratings.map(x => (x._2,(x._1,x._3))))//  // group rating by items
val ratingByUser = sc.broadcast(ratings.map(x => (x._1,(x._2,x._3))))//.partitionBy(new HashPartitioner(10)).persist  // group rating by user

var i =0
for( i <- 1 to 10){

	val ratItemVec = myitemMatrix.join(ratingByItem.value)//  // generate rating, item and item vector by item
	val regfactor = 1.0
	val regMatrix = DenseMatrix.zeros[Double](k,k)
	regMatrix(0,::) := DenseVector(regfactor,0,0,0,0).t 
	regMatrix(1,::) := DenseVector(0,regfactor,0,0,0).t 
	regMatrix(2,::) := DenseVector(0,0,regfactor,0,0).t 
	regMatrix(3,::) := DenseVector(0,0,0,regfactor,0).t 
	regMatrix(4,::) := DenseVector(0,0,0,0,regfactor).t 
	val userbyItemMat = ratItemVec.map(x => (x._2._2._1,x._2._1*x._2._1.t )).reduceByKey(_+_).map(x=> (x._1,breeze.linalg.pinv(x._2+ regMatrix))) //cal sum(yiyit+regMatrix)

	val sumruiyi = ratItemVec.map(x => (x._2._2._1,x._2._1 * x._2._2._2.toDouble )).reduceByKey(_+_)  // cal sum(rui * yi) where yi is item vectors

	val joinres = userbyItemMat.join(sumruiyi) // calc
	myuserMatrix = joinres.map(x=> (x._1,x._2._1 * x._2._2)).partitionBy(new HashPartitioner(10)) // cal sum yi yit * sum (rui *yi) this gives update of user vectors

	val ratUserVec = myuserMatrix.join(ratingByUser.value)//generate rating, item and item vector by user key
	val itembyuserMat = ratUserVec.map(x => (x._2._2._1,x._2._1*x._2._1.t )).reduceByKey(_+_).map(x=> (x._1,breeze.linalg.pinv(x._2 + regMatrix ))) //cal sum(xi xiT+regMatrix)
	val sumruixi = ratUserVec.map(x => (x._2._2._1,x._2._1 * x._2._2._2.toDouble )).reduceByKey(_+_)  // calc sum(rui * xi)
	val joinres2 = itembyuserMat.join(sumruixi)
	myitemMatrix = joinres2.map(x=> (x._1,x._2._1 * x._2._2)).partitionBy(new HashPartitioner(10)) // calc sum(xi * xit) * sum(rui*xi)  // give item matrix
} 

val tempU = myuserMatrix.filter(x => x._1 == "1759").collect
val tempI = myitemMatrix.filter(x => x._1 == "231").collect

val rate = tempU(0)._2.t * tempI(0)._2

println(rate)

val theFinalUserM = myuserMatrix.collect()
val theFinalItemM = myitemMatrix.collect()
val results = List((0,0,0.0))
for((x,uVector) <- theFinalUserM){
	for((j,itemV) <- theFinalItemM){
		results :: List((x,j, uVector.t * itemV))
		println((x,j, uVector.t * itemV))
}
}
sc.parallelize(results).saveAsTextFile("hdfs://cshadoop1.utdallas.edu/gga110020/rating/Predratingsnew2.dat")



