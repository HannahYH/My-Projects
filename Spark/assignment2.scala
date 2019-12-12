// Use the named values (val) below whenever your need to
// read/write inputs and outputs in your program. 
val inputFilePath = "sample_input.txt"
val outputDirPath = "output"

// create an RDD from the file
val file = sc.textFile(inputFilePath, 1)

// extract base URL and Payload as key-value pairs
val lines = file.filter(_.length > 0).map(line => (line.split(",")(0), line.split(",")(3)))

// transform the payload from KB and MB to B, the datatype of payload is Long
val pairs = lines.map(line => if (line._2.contains("K")) (line._1, line._2.split("K")(0).toLong*1024) else if (line._2.contains("M")) (line._1, line._2.split("M")(0).toLong*1024*1024) else (line._1, line._2.split("B")(0).toLong))

// group payloads for each base URL
val grouped = pairs.groupByKey()

// define two functions to compute the mean and variance of a list[Long]
def mean(a: List[Long]) = a.sum / a.length 
def variance(a: List[Long]) = {
	val avg = mean(a)
	mean(a.map(x => (x - avg) * (x - avg)))
}

// compute the min, max, mean and variance payload for each base URL
val result = grouped.map(x => x._1 + "," + x._2.toList.min.toString + "B," + x._2.toList.max.toString + "B," + mean(x._2.toList).toString + "B," + variance(x._2.toList).toString + "B")

// write the result into file
result.saveAsTextFile(outputDirPath)
