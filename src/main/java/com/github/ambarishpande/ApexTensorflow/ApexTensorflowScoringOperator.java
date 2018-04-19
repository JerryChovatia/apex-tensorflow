package com.github.ambarishpande.ApexTensorflow;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

import com.datatorrent.api.AutoMetric;
import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.common.util.BaseOperator;
import com.datatorrent.common.util.Pair;

/**
 * Created by ambarish on 28/8/17.
 */
public class ApexTensorflowScoringOperator extends BaseOperator
{

  private static final Logger LOG = LoggerFactory.getLogger(ApexTensorflowScoringOperator.class);

  private transient byte[] graphDef;
  private String modelDir;
  private String modelFileName;
  private transient Graph g;
  private String labelsFileName;
  private transient List<String> labels;
  private int[] counts;

  @FieldSerializer.Bind(JavaSerializer.class)
  @AutoMetric
  private Collection<Collection<Pair<String, Object>>> labelCounts = new ArrayList<>();

  public transient DefaultInputPort<Image> input = new DefaultInputPort<Image>()
  {

    @Override
    public void process(Image image)
    {
      float[] labelProbabilities = executeGraph(image.getImage());
      int bestLabelIdx = Utils.maxIndex(labelProbabilities);

      if (outputBestCategory.isConnected()) {
        outputBestCategory.emit(new TfCategory(bestLabelIdx, labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx]));
      }

      counts[bestLabelIdx]++;

      LOG.info(
        String.format(
          "BEST MATCH: %s  %s (%.2f%% likely)", image.getD().fileName,
          labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));

    }

  };

  public transient DefaultOutputPort<Tensor> outputTensor = new DefaultOutputPort();
  public transient DefaultOutputPort<TfCategory> outputBestCategory = new DefaultOutputPort();


  public void setup(Context.OperatorContext context)
  {

    graphDef = Utils.readModelFromHdfs(modelDir + modelFileName);
    labels = Utils.readLabelsFromFile(modelDir + labelsFileName);
    g = new Graph();
    g.importGraphDef(graphDef);
    counts = new int[labels.size()];

  }

  @Override
  public void beginWindow(long windowId)
  {
    super.beginWindow(windowId);
    labelCounts.clear();
    for (int i = 0; i < counts.length; i++) {
      if (counts[i] > 0) {
        Collection<Pair<String, Object>> row = new ArrayList<>();
        row.add(new Pair<String, Object>("Label", labels.get(i)));
        row.add(new Pair<String, Object>("Count", new Double(counts[i])));
        labelCounts.add(row);
      }
    }
  }

  private float[] executeGraph(Tensor image)
  {
    try (Session s = new Session(g);
      Tensor result = s.runner().feed("input", image).fetch("output").run().get(0)) {
      final long[] rshape = result.shape();
      if (result.numDimensions() != 2 || rshape[0] != 1) {
        throw new RuntimeException(
          String.format(
            "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
            Arrays.toString(rshape)));
      }
      int nlabels = (int)rshape[1];

      if(outputTensor.isConnected()){
        outputTensor.emit(result);
      }

      return result.copyTo(new float[1][nlabels])[0];
    }
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

  static class GraphBuilder
  {
    GraphBuilder(Graph g)
    {
      this.g = g;
    }

    Output div(Output x, Output y)
    {
      return binaryOp("Div", x, y);
    }

    Output sub(Output x, Output y)
    {
      return binaryOp("Sub", x, y);
    }

    Output resizeBilinear(Output images, Output size)
    {
      return binaryOp("ResizeBilinear", images, size);
    }

    Output expandDims(Output input, Output dim)
    {
      return binaryOp("ExpandDims", input, dim);
    }

    Output cast(Output value, DataType dtype)
    {
      return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
    }

    Output decodeJpeg(Output contents, long channels)
    {
      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
        .addInput(contents)
        .setAttr("channels", channels)
        .build()
        .output(0);
    }

    Output constant(String name, Object value)
    {
      try (Tensor t = Tensor.create(value)) {
        return g.opBuilder("Const", name)
          .setAttr("dtype", t.dataType())
          .setAttr("value", t)
          .build()
          .output(0);
      }
    }

    Output placeholder(String name)
    {

      return g.opBuilder("Placeholder", name)
        .setAttr("dtype", DataType.STRING)
        .build()
        .output(0);

    }

    private Output binaryOp(String type, Output in1, Output in2)
    {
      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
    }

    private Graph g;
  }

}

