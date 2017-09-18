package com.github.ambarishpande.ApexTensorflow;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;
import org.tensorflow.Tensor;

import static org.junit.Assert.*;

/**
 * Created by ambarish on 28/8/17.
 */
public class ApexTensorflowScoringOperatorTest
{

  @Test
  public void score(){
    ApexTensorflowScoringOperator tf = new ApexTensorflowScoringOperator();
    tf.setModelDir("/user/ambarish/apex-tf/models/");

    tf.setModelFileName("tensorflow_inception_graph.pb");
    tf.setLabelsFileName("imagenet_comp_graph_label_strings.txt");
    tf.setup(null);
    tf.beginWindow(0);
    String imageFile = "/user/ambarish/apex-tf/images/";
    String[] images = {"dalmation.jpeg","cat.jpeg","mouse.jpeg"};
    for (String image : images
    ) {
      byte[] imageBytes = readAllBytesOrExit(Paths.get(imageFile + image));
      Data d = new Data();
      d.bytesImage = imageBytes;
      d.fileName = image;
      d.imageType="jpeg";
      ImageNormalizer n = new ImageNormalizer();
      Image i = new Image(n.normalize(d),d);
      tf.input.process(i);
    }

    tf.endWindow();
  }

  public static byte[] readAllBytesOrExit(Path path) {
    try {
      return Files.readAllBytes(path);
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(1);
    }
    return null;
  }
}