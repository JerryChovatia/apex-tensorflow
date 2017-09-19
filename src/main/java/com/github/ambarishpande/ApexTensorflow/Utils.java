package com.github.ambarishpande.ApexTensorflow;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;

/**
 * Created by ambarish on 19/9/17.
 */
public class Utils
{
  private static final Logger LOG = LoggerFactory.getLogger(Utils.class);

  public static int maxIndex(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }

  public static byte[] readModelFromHdfs(String path)
  {
    org.apache.hadoop.fs.Path location = new org.apache.hadoop.fs.Path(path);
    Configuration configuration = new Configuration();

    try {
      FileSystem hdfs = FileSystem.get(new URI(configuration.get("fs.defaultFS")), configuration);

      if (hdfs.exists(location)) {
        FSDataInputStream hdfsInputStream = hdfs.open(location);
        int length = (int) hdfs.getFileStatus(location).getLen();
        byte[] model = new byte[length];
        hdfsInputStream.readFully(model);
        LOG.info("Model file loaded from " + configuration.get("fs.defaultFS") + location + " and Size " + model
          .length );
        hdfsInputStream.close();
        return model;
      }else{
        LOG.error("Model file not found");
        return null;
      }

    } catch (IOException e) {
      e.printStackTrace();
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }
    return  null;


  }

  public static ArrayList<String> readLabelsFromFile(String path)
  {
    ArrayList<String> labels = new ArrayList<>();

    org.apache.hadoop.fs.Path location = new org.apache.hadoop.fs.Path(path);
    Configuration configuration = new Configuration();
    FileSystem hdfs = null;
    BufferedReader br = null;

    try{
      hdfs = FileSystem.newInstance(new URI(configuration.get("fs.defaultFS")), configuration);
      br = new BufferedReader(new InputStreamReader(hdfs.open(location)));

      String line;
      line=br.readLine();
      while (line != null){
        labels.add(line);
        line = br.readLine();
      }
    } catch (URISyntaxException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }finally {
      try {
        br.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }

    LOG.info("Labels file loaded from " + configuration.get("fs.defaultFS") + location);
    return labels;
  }
}
