/**
 * Created by hlb on 06/04/2016.
 */

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.TaskCounter;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.DataInput;
import java.io.DataOutput;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

import java.math.BigDecimal;
import java.util.*;



public class KNNMapReduce
{

    // A data structure for storing model and distance variables
    public static class CompositeWritable implements Writable
    {
        String model = "";
        Double distance = 0.0;

        // Default class constructor
        public CompositeWritable()
        {

        }

        // Class constructor
        public CompositeWritable(String mod, Double dist) {
            this.model = mod;
            this.distance = dist;
        }


        @Override
        public void readFields(DataInput in) throws IOException {
            model = WritableUtils.readString(in);
            distance = in.readDouble();
        }

        @Override
        public void write(DataOutput out) throws IOException {
            WritableUtils.writeString(out, model);
            out.writeDouble(distance);
        }

    } // End CompositeWritable class

    // Mapper class
    public static class WCMapper extends Mapper<Object, Text, NullWritable, Text>
    {
        // Test Data array variables
        private String testAge = "";
        private String testIncome = "";
        private String testMaritalStatus = "";
        private String testGender = "";
        private String testChildren = "";

        // Car Data variables
        private String readCarDataFromFile = "";
        private String[] carDataRecord = {};
        private String age = "";
        private String income = "";
        private String maritalStatus = "";
        private String gender = "";
        private String children = "";
        private String carModel;

        // Age range scaling variables
        private final double ageBaseMin = 18.0;
        private final double ageBaseMax = 77.0;
        private final double ageLimitMin = 0.0;
        private final double ageLimitMax = 1.0;

        // Income range scaling variables
        private final double incomeBaseMin = 5000.0;
        private final double incomeBaseMax = 67789.0;
        private final double incomeLimitMin = 0.0;
        private final double incomeLimitMax = 1.0;

        // Children range scaling variables
        private final double kidsBaseMin = 0.0;
        private final double kidsBaseMax = 5.0;
        private final double kidsLimitMin = 0.0;
        private final double kidsLimitMax = 1.0;


        private BigDecimal bigDecimal;                      // Variable used for calculating a BigDecimal number

        // Variables used in map()
        private Double knnDistance;                         // k Nearest Neighbour
        private NullWritable nullValue;                     // Null variable used as key in map and reduce
        private Text modValue;                              // Text variable used to hold model and distance values
        private ArrayList<String> mapperKeyValueArray;      // ArrayList used to contain model and distance values

        // Method to scale attributes
        public double scale(final double valueIn, final double baseMin, final double baseMax, final double limitMin, final double limitMax)
        {
            return ((limitMax - limitMin) * (valueIn - baseMin) / (baseMax - baseMin)) + limitMin;
        } // End scale()

        // Method to calculate the k Nearest Neighbour distance
        public double knnMatching(String tAge, String tInc, String tMS, String tGend, String tChild, String cAge, String cInc, String cMS, String cGen, String cChild)
        {
            // Method variables
            Double ageKnnTest, ageKnnCar, incomeKnnTest, incomeKnnCar, msKnnTest, msKnnCar, genKnnTest, genKnnCar, childKnnTest, childKnnCar;

            // Assign test variables
            testAge = tAge;
            testIncome = tInc;
            testMaritalStatus = tMS;
            testGender = tGend;
            testChildren = tChild;

            // Assign carVariables
            age = cAge;
            income = cInc;
            maritalStatus = cMS;
            gender = cGen;
            children = cChild;

            // scale attributes

            // age
            ageKnnTest = convertBigDecimal(scale(Double.parseDouble(testAge), ageBaseMin, ageBaseMax, ageLimitMin, ageLimitMax));
            ageKnnCar = convertBigDecimal(scale(Double.parseDouble(age), ageBaseMin, ageBaseMax, ageLimitMin, ageLimitMax));

            // income
            incomeKnnTest = convertBigDecimal(scale(Double.parseDouble(testIncome), incomeBaseMin, incomeBaseMax, incomeLimitMin, incomeLimitMax));
            incomeKnnCar = convertBigDecimal(scale(Double.parseDouble(income), incomeBaseMin, incomeBaseMax, incomeLimitMin, incomeLimitMax));

            // children
            childKnnTest = convertBigDecimal(scale(Double.parseDouble(testChildren), kidsBaseMin, kidsBaseMax, kidsLimitMin, kidsLimitMax));
            childKnnCar = convertBigDecimal(scale(Double.parseDouble(children), kidsBaseMin, kidsBaseMax, kidsLimitMin, kidsLimitMax));

            // initialise non-scaling attributes
            msKnnTest = 1.0;
            msKnnCar = 1.0;
            genKnnTest = 1.0;
            genKnnCar = 1.0;

            // Calculate the distance between attributes
            if (testMaritalStatus.equals(maritalStatus))
            {
                msKnnTest = 0.0;
                msKnnCar = 0.0;
            }

            if (testGender.equals(gender))
            {
                genKnnTest = 0.0;
                genKnnCar = 0.0;
            }

            Double distance = convertBigDecimal(Math.sqrt(Math.pow(ageKnnCar - ageKnnTest, 2) + Math.pow(incomeKnnCar - incomeKnnTest, 2) + Math.pow(msKnnCar - msKnnTest, 2) + Math.pow(genKnnCar - genKnnTest, 2) + Math.pow(childKnnCar - childKnnTest, 2)));

            return distance;
        } // End knnMatching()

        // Method to scale decimal values
        public Double convertBigDecimal(Double valueIn)
        {
            Double newValue = valueIn;
            bigDecimal = new BigDecimal(valueIn);
            bigDecimal = bigDecimal.setScale(2, BigDecimal.ROUND_UP);
            newValue = bigDecimal.doubleValue();
            return newValue;
        } // End convertBigDecimal()

        // Method for sorting an ArrayList
        public void sortArray(ArrayList<String> sortArrayList)
        {
            Collections.sort(sortArrayList, new Comparator<String>()
            {
                @Override
                public int compare(String o1, String o2)
                {
                    String[] firstString = o1.split(":");
                    String[] secondString = o2.split(":");
                    Double firstNum = Double.valueOf(firstString[1]);
                    Double secNum = Double.valueOf(secondString[1]);

                    return firstNum.compareTo(secNum);
                }
            });
        } // End sortArray()

        // setup() method is executed once before map() method; used to initialise variables
        @Override
        protected void setup(Context context) throws IOException, InterruptedException
        {
            // Initialise variables
            knnDistance = 0.0;
            nullValue = NullWritable.get();
            modValue = new Text();
            mapperKeyValueArray = new ArrayList<>();

        } // End setup()

        // cleanup() method is executed once after map(); call getClosestModel(); write to reducer()
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException
        {
            // Call getClosestModel() and set result to text object (modValue)
            // mapperKeyValueArray() is populated with data in map()
            modValue.set(getClosestModel(mapperKeyValueArray));

            // Write to context
            context.write(nullValue, modValue);

        } // End cleanup()

        // A method to determine the closest model to test data (closest distance)
        public static String getClosestModel(ArrayList<String> arrayList)
        {
            HashMap<String, CompositeWritable> hashMap = new HashMap<>();
            ArrayList<String> maxModel = new ArrayList<>();
            ArrayList<CompositeWritable> minValueArray = new ArrayList<>();
            Double minNum = 0.0;
            String minModel = "";
            Double countDistance = 0.0;
            Double distanceVal  = 0.0;

            for (String arr: arrayList)
            {
                String [] keyValArr = arr.split(":");
                String getKey = keyValArr[0];
                Double getDistance = Double.valueOf(keyValArr[1]);
                if (hashMap.containsKey(getKey))
                {
                    distanceVal = hashMap.get(getKey).distance;
                    distanceVal = distanceVal + getDistance;
                    hashMap.put(getKey, new CompositeWritable(getKey,distanceVal));
                    countDistance = distanceVal;
                    if (minValueArray.size()!=0)
                    {
                        if (countDistance < minValueArray.get(0).distance)
                        {
                            minValueArray.clear();
                            minValueArray.add(new CompositeWritable(getKey,countDistance));
                        }
                    }
                }
                else
                {
                    hashMap.put(getKey, new CompositeWritable(getKey,getDistance));

                    if(minValueArray.size()!=0)
                    {
                        minNum = minValueArray.get(0).distance;
                        minModel = minValueArray.get(0).model;
                        if (minNum > hashMap.get(getKey).distance)
                        {
                            minValueArray.clear();
                            minValueArray.add(new CompositeWritable(getKey,getDistance));
                        }
                    }
                    else  if (minValueArray.size()==0)
                    {
                        minValueArray.clear();
                        minValueArray.add(new CompositeWritable(getKey,getDistance));
                    }
                }
            }

            String getModel = minValueArray.get(0).model+" : "+minValueArray.get(0).distance;
            return getModel;

        } // End getClosestModel()

        // Map function: key is bytes so far, value is line from file
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            try
            {
                // Get K value
                Configuration conf = context.getConfiguration();
                final Integer kVariable = Integer.parseInt(conf.get("kValue"));

                // Get test data

                //String param = conf.get("test1");
                //String param = conf.get("test2");
                //String param = conf.get("test3");
                //String param = conf.get("test4");
                String param = conf.get("test5");
                //String param = conf.get("test6");

                String[] testDataArray = param.split(",");

                // Assign values from testRecord to test data variables
                testAge = testDataArray[0];
                testIncome = testDataArray[1];
                testMaritalStatus = testDataArray[2];
                testGender = testDataArray[3];
                testChildren = testDataArray[4];

                // Read car data
                StringTokenizer tokenizer = new StringTokenizer(value.toString(), "\n\r\f");

                while (tokenizer.hasMoreTokens())
                {
                    // Read line from scanner
                    readCarDataFromFile = null;
                    readCarDataFromFile = tokenizer.nextToken();

                    // Split data from string and insert into String array
                    String[] carDataArray = readCarDataFromFile.split(",");
                    carDataRecord = readCarDataFromFile.split(",");

                    // Assign values from carDataRecord to car data variables
                    age = carDataRecord[0];
                    income = carDataRecord[1];
                    maritalStatus = carDataRecord[2];
                    gender = carDataRecord[3];
                    children = carDataRecord[4];
                    carModel = carDataRecord[5];

                    // Match test and car variables
                    knnDistance = knnMatching(testAge, testIncome, testMaritalStatus, testGender, testChildren, age, income, maritalStatus, gender, children);

                    // Create a string variable to store car model and distance
                    String modDist = carModel+":"+knnDistance.toString();

                    // Add record to mapperKeyValueArray()
                    // Determine if knnDistance of current data is lower than highest value in array and if it is replace higher distance value with lower
                    if (mapperKeyValueArray.size()>(kVariable-1))
                    {
                        // Sort mapperKeyValueArray()
                        sortArray(mapperKeyValueArray);

                        String [] getDistanceArray = mapperKeyValueArray.get(0).split(":");
                        Double distanceValue = Double.valueOf(getDistanceArray[1]);

                        if (knnDistance<distanceValue)
                        {
                            mapperKeyValueArray.remove(0);
                            mapperKeyValueArray.set(0,modDist);
                        }
                    }
                    else
                    {
                        mapperKeyValueArray.add(modDist);
                    }
                }
            } // end try
            catch (Exception e)
            {
                e.printStackTrace();
            } // end catch()
        } // end map() function
    } // end WCWrapper class


    // Reducer class
    public static class WCReducer extends Reducer<NullWritable, Text, NullWritable, Text>
    {
        // Reduce function: key is word as emitted by map function and values is array of values associated with each key
        public void reduce(NullWritable key, Text values, Context context) throws IOException, InterruptedException
        {

            //Output key and value from cleanup() in WCMapper()
            context.write(key,values);

        } // End reduce()
    } // End WCReducer()

    // KNNMapReduce main() method
    public static void main(String[] args) throws Exception
    {
        // Main program to run
        Configuration conf = new Configuration();
        conf.set("test1", "67,16668,Single,Male,3");
        conf.set("test2", "51,40271,Married,Female,0");
        conf.set("test3", "72,26472,Married,Female,2");
        conf.set("test4", "58,37111,Single,Female,1");
        conf.set("test5", "68,43312,Widowed,Male,1");
        conf.set("test6", "37,33185,Single,Male,4");
        conf.setInt("kValue", 5);
        Job job = Job.getInstance(conf, "KNN_Counter");

        job.setJarByClass(KNNMapReduce.class);

        job.setMapperClass(WCMapper.class);               // Set mapper class to WCMapper defined above
        job.setCombinerClass(WCReducer.class);            // Set combine class to WCReducer defined above
        job.setReducerClass(WCReducer.class);             // Set reduce class to WCReducer defined above

        job.setMapOutputKeyClass(NullWritable.class);     // NullWritable Map output class
        job.setMapOutputValueClass(Text.class);           // Text Map output class

        job.setOutputKeyClass(NullWritable.class);       // Class of output key is NullWritable
        job.setOutputValueClass(Text.class);             // Class of output key is Text

        FileInputFormat.addInputPath(job, new Path(args[0]));        // Input path is first argument when program called
        FileOutputFormat.setOutputPath(job, new Path(args[1]));        // Output path is second argument when program called

        job.waitForCompletion(true);               // waitForCompletion submits the job and waits for it to complete, parameter is verbose.
        Counters counters = job.getCounters();
        System.out.println("Input Records: " + counters.findCounter(TaskCounter.MAP_INPUT_RECORDS).getValue());
    } // End main()
} // End KNNMapReduce()
