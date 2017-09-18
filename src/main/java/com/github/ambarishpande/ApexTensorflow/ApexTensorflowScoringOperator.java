package com.github.ambarishpande.ApexTensorflow;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.common.util.BaseOperator;

/**
 * Created by ambarish on 28/8/17.
 */
public class ApexTensorflowScoringOperator extends BaseOperator
{

  private static final Logger LOG = LoggerFactory.getLogger(ApexTensorflowScoringOperator.class);

  private byte[] graphDef;
  private String modelDir;
  private String modelFileName;

  private String labelsFileName;
  private List<String> labels;


  public transient DefaultInputPort<Image> input = new DefaultInputPort<Image>(){

    @Override
    public void process(Image image)
    {
//      Tensor image = normalize(data.bytesImage);

      float[] labelProbabilities = executeInceptionGraph(graphDef, image.getImage());
      int bestLabelIdx = maxIndex(labelProbabilities);
      LOG.info(
        String.format(
          "BEST MATCH: %s  %s (%.2f%% likely)",image.getD().fileName,
           labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
    }

  };

  public void setup(Context.OperatorContext context) {

    graphDef = readModelFromHdfs(modelDir);
    labels = readLabelsFromFile();

  }

//
//  private static Tensor normalize(byte[] imageBytes) {
//    try (Graph g = new Graph()) {
//      GraphBuilder b = new GraphBuilder(g);
//
//      final int H = 224;
//      final int W = 224;
//      final float mean = 117f;
//      final float scale = 1f;
//
//      // Since the graph is being constructed once per execution here, we can use a constant for the
//      // input image. If the graph were to be re-used for multiple input images, a placeholder would
//      // have been more appropriate.
////      final Output input = b.constant("input", imageBytes);
//      final Output input = b.placeholder("input", imageBytes);
//
//      final Output output =
//        b.div(
//          b.sub(
//            b.resizeBilinear(
//              b.expandDims(
//                b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
//                b.constant("make_batch", 0)),
//              b.constant("size", new int[] {H, W})),
//            b.constant("mean", mean)),
//          b.constant("scale", scale));
//
//      try (Session s = new Session(g)) {
//        return s.runner().feed("input",Tensor.create(imageBytes)).fetch(output.op().name()).run().get(0);
//      }
//    }
//  }


  private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef);

      try (Session s = new Session(g);
        Tensor result = s.runner().feed("input", image).fetch("output").run().get(0)) {
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != 1) {
          throw new RuntimeException(
            String.format(
              "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
              Arrays.toString(rshape)));
        }
        int nlabels = (int) rshape[1];
        return result.copyTo(new float[1][nlabels])[0];
      }
    }
  }

  private static int maxIndex(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }

  public byte[] readModelFromHdfs(String path)
  {
//    Code for fetching saved model in case master is killed.
    org.apache.hadoop.fs.Path location = new org.apache.hadoop.fs.Path(path+"/"+modelFileName);
    Configuration configuration = new Configuration();

    try {
      FileSystem hdfs = FileSystem.get(new URI(configuration.get("fs.defaultFS")), configuration);

      if (hdfs.exists(location)) {
        FSDataInputStream hdfsInputStream = hdfs.open(location);
        int length = (int) hdfs.getFileStatus(location).getLen();
        byte[] model = new byte[length];
        hdfsInputStream.readFully(model);
        LOG.error("Model file loaded from " + configuration.get("fs.defaultFS") + location + " and Size " + model
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

  public ArrayList<String> readLabelsFromFile()
  {
    ArrayList<String> labels = new ArrayList<>();

    org.apache.hadoop.fs.Path location = new org.apache.hadoop.fs.Path(modelDir + "/" + labelsFileName);
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


  public void setModelDir(String modelDir)
  {
    this.modelDir = modelDir;
  }

  public void setModelFileName(String modelFileName)
  {
    this.modelFileName = modelFileName;
  }


  public void setLabelsFileName(String labelsFileName)
  {
    this.labelsFileName = labelsFileName;
  }

  public String getModelDir()
  {
    return modelDir;
  }

  public String getModelFileName()
  {
    return modelFileName;
  }

  public String getLabelsFileName()
  {
    return labelsFileName;
  }
  static class GraphBuilder {
    GraphBuilder(Graph g) {
      this.g = g;
    }

    Output div(Output x, Output y) {
      return binaryOp("Div", x, y);
    }

    Output sub(Output x, Output y) {
      return binaryOp("Sub", x, y);
    }

    Output resizeBilinear(Output images, Output size) {
      return binaryOp("ResizeBilinear", images, size);
    }

    Output expandDims(Output input, Output dim) {
      return binaryOp("ExpandDims", input, dim);
    }

    Output cast(Output value, DataType dtype) {
      return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
    }

    Output decodeJpeg(Output contents, long channels) {
      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
        .addInput(contents)
        .setAttr("channels", channels)
        .build()
        .output(0);
    }

    Output constant(String name, Object value) {
      try (Tensor t = Tensor.create(value)) {
        return g.opBuilder("Const", name)
          .setAttr("dtype", t.dataType())
          .setAttr("value", t)
          .build()
          .output(0);
      }
    }

    Output placeholder(String name, Object value) {
      try (Tensor t = Tensor.create(value)) {
        return g.opBuilder("Placeholder", name)
          .setAttr("dtype", t.dataType())
          .build()
          .output(0);
      }
    }



    private Output binaryOp(String type, Output in1, Output in2) {
      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
    }

    private Graph g;
  }
}
