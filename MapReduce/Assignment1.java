package assignment1;

/**
 * 
 * This class solves the problem posed for Assignment1
 *
 */
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Assignment1 {
	// Mapper
	public static class TextMapper
		extends Mapper<Object, Text, Text, Text>{

	    private Text word = new Text(); // output key (ngram)
	    private Text arr = new Text(); // output value (1 filename)
	
	    public void map(Object key, Text value, Context context
	                    ) throws IOException, InterruptedException {
	    	// get the parameter "ngram" we have set before 
	    	Configuration conf = context.getConfiguration();
	    	String ngram = conf.get("ngram");
	    	int N = Integer.parseInt(ngram);
	    	// find the filename of current HDFS from context
	    	String filename = ((FileSplit)context.getInputSplit()).getPath().getName().toString();
	    	// set the output value
	    	arr.set("1  " + filename);
	    	// split the value into separated words
	    	String[] words = value.toString().split("\\s+");
	    	// find all ngrams
	    	for (int i = 0; i <= words.length - N; i++) {
	    		String w = "";
		    	for (int j = i; j < i + N; j++) {
		    		w += words[j] + " ";
		    	}
		    	w = w.trim();
		    	// set each ngram as the key
		        word.set(w);
		        // write the output key-value pair into HDFS
		        context.write(word, arr);
	    	}
	    }
	}

	// Reducer
	public static class TextSumReducer
    	extends Reducer<Text,Text,Text,Text> {
	  
	    private Text result = new Text(); // the output value (count filename)
	
	    public void reduce(Text key, Iterable<Text> values, Context context
	                       ) throws IOException, InterruptedException {
	    	// get the parameter "mincount" we have set before 
	    	Configuration conf = context.getConfiguration();
	    	int mincount = Integer.parseInt(conf.get("mincount"));
	    	
	    	int sum = 0;
	    	List<String> namelist = new ArrayList<String>();
	    	String[] items;
	    	// aggregates the list of values for each key
	    	for (Text val : values) {
	    		// split the value
	    		items = val.toString().split("\\s+");
	    		// the first part is the count
	    		sum += Integer.parseInt(items[0]);
	    		// the second part is the filename
	    		// if the filename does not exist, add it
	    		if (!namelist.contains(items[1])) {
	    			namelist.add(items[1]);
	    		}
	    	}
	    	Collections.sort(namelist);
	    	// if the count of a key is equal or greater than mincount, then write it into HDFS
	    	if (sum >= mincount) {
	    		String valuelist = "";
	    		for (String s: namelist) {
	    			valuelist += s + " ";
	    		}
	    		valuelist = valuelist.trim();
	    		result.set(String.valueOf(sum) + "  " + valuelist);
	    		context.write(key, result);
	    	}
	    }
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		// set the two configuration parameters
		conf.set("ngram", args[0]);
		conf.set("mincount", args[1]);
		// remove the output folder if it exists
		File output =  new File(args[3]);
		if (output.exists()) {
			for(File file: output.listFiles()) {
				if (!file.isDirectory()) {
			        file.delete();
			    }
			}
			output.delete();	
		}
		Job job = Job.getInstance(conf, "word count");
		job.setJarByClass(Assignment1.class);
		job.setMapperClass(TextMapper.class);
		job.setReducerClass(TextSumReducer.class);
		// the types of output key and output value are both Text
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(args[2]));
		FileOutputFormat.setOutputPath(job, new Path(args[3]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
