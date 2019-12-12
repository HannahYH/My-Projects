import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._ 
import org.apache.spark.SparkConf
import java.io.File
import sys.process._
import scala.io.Source
import scala.util.parsing.json._
import scala.xml._

object CaseIndex {
	// get all files from a directory
	def getListOfFiles(dir: String):List[File] = {
		val d = new File(dir)
		if (d.exists && d.isDirectory) {
	    		return d.listFiles.filter(_.isFile).toList
		} else {
		  	List[File]()
		}
	}

	def main(args: Array[String]) {
		// create index in ElasticSearch
		var curlResult_index = List("curl", "-X", "PUT", 
			"http://localhost:9200/legal_idx?pretty"
			).!!

		// create mapping within the index 
		val curlResult_mapping = List("curl", "-X", "PUT", 
			"http://localhost:9200/legal_idx/cases/_mapping?pretty",
			"-H",
			"Content-type: application/json",
			"-d",
			"{\"cases\": {\"properties\": {\"name\": {\"type\": \"text\"},\"url\": {\"type\": \"text\"},\"person\": {\"type\": \"text\"},\"location\": {\"type\": \"text\"},\"organisation\": {\"type\": \"text\"},\"catchphrases\": {\"type\": \"text\"},\"description\": {\"type\": \"text\"}}}}"
			).!!

		// get all files
		val files = getListOfFiles(args(0)) // files' directory
		var fileId = 0 // file's ID
		for(file <- files){
			// read XML file
			val xmlFile = XML.loadFile(file)

			// extract content from tags(name, AustLII, catchphrases, sentences)
			val name = (xmlFile \\ "name").text
			val url = (xmlFile \\ "AustLII").text
			val catchphrases = (xmlFile \\ "catchphrases").text.trim().split("\n").map(s => s.trim()).mkString(" ")
			val description_ = (xmlFile \\ "sentences").text.trim().split("\n").map(s => s.trim()).mkString(" ")
			val description = description_.replace("\"", "");

			// enrich the legal report cases with entity type in Sentences
			val curlResult_NLP = List("wget", "--post-data", description, 
					      "http://localhost:9000/?properties={%22annotators%22%3A%22ner%22%2C%22outputFormat%22%3A%22json%22}",
					      "-O",
					      "-"
					      ).!!

			// extract all entity types for each sentence
			var person = ""
			var location = ""
			var organisation = ""
			val curlResultJson = JSON.parseFull(curlResult_NLP).get.asInstanceOf[Map[String, List[Map[String, Any]]]]
			for(item <- curlResultJson("sentences")){
				val tokens = item("tokens").asInstanceOf[List[Map[String, String]]]
				for(ele <- tokens){
					val ner = ele("ner")
					val text = ele("word")
					if(ner == "PERSON"){
						person += text + " "
					}
					if(ner == "LOCATION"){
						location += text + " "
					}
					if(ner == "ORGANIZATION"){
						organisation += text + " "
					}
				}
		 	}
			person = person.trim()
			location = location.trim()
			organisation = organisation.trim()
	
			// combine all info into a JsonString
			val jsonString = s"""{
				"name":"$name",
				"url":"$url",
				"person":"$person",
				"location":"$location",
				"organisation":"$organisation",
				"catchphrases":"$catchphrases",
				"description":"$description"
				}"""

			// create document
			var curlResult_doc = List("curl", "-X", "PUT", 
						  "http://localhost:9200/legal_idx/cases/doc_" + fileId.toString() + "?pretty",
						  "-H", "Content-type: application/json", "-d", jsonString
						   ).!!
			// augment fileId
			fileId += 1
	       	}
	}

}


